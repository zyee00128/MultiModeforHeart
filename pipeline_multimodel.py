import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from model.model_code_default import (MultimodalLSTransNet, Hetero_MultimodalTeacherNet,
                                      Cutmix_Multimodal, mask_ecg_signal)
from tools.pytorchtools import EarlyStopping
from tools.evaluation import print_result, find_thresholds
from tools.datacollection import MultimodalDataset_loading
from model.prior_utils import PhysioPriorTool, PhysioDetector

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)
def get_rank_list(num_layers, conv_r=8, trans_r=32):
    # Conv层低rank，Transformer层高rank
    if num_layers == 47:
        rank_list = [conv_r] * 9
        num_trans_layers = (num_layers - 9 - 2) // 3
        rank_list += [trans_r] * (num_trans_layers * 3) 
        rank_list += [trans_r] * 2
    elif num_layers == 30 or num_layers == 42:
        rank_list = [conv_r] * 4 
        num_trans_layers = (num_layers - 4 - 2) // 3
        rank_list += [trans_r] * (num_trans_layers * 3) 
        rank_list += [trans_r] * 2
    
    return np.array(rank_list[:num_layers])

def load_pretrained_model(net, path, args):
    """
    加载预训练模型权重。
    并自动过滤掉可能形状不匹配的 classifier 层权重。
    """
    pretrained_dict = torch.load(path, map_location=args.device)
    model_dict = net.state_dict()
    new_pretrained_dict = {}
    
    for k, v in pretrained_dict.items():
        # 键值完全匹配
        if k in model_dict:
            new_pretrained_dict[k] = v
        # 映射到 backbone 命名空间
        elif f"backbone.{k}" in model_dict:
            new_pretrained_dict[f"backbone.{k}"] = v
        # 映射到多模态网络中的 ECG / PCG 分支命名空间
        elif f"ecg_encoder.{k}" in model_dict:
            new_pretrained_dict[f"ecg_encoder.{k}"] = v
        elif f"pcg_encoder.{k}" in model_dict:
            new_pretrained_dict[f"pcg_encoder.{k}"] = v
        elif f"ecg_encoder.backbone.{k}" in model_dict:
            new_pretrained_dict[f"ecg_encoder.backbone.{k}"] = v
        elif f"pcg_encoder.backbone.{k}" in model_dict:
            new_pretrained_dict[f"pcg_encoder.backbone.{k}"] = v

    # 过滤掉顶层分类器的权重
        pretrained_dict = {k: v 
                       for k, v in new_pretrained_dict.items() 
                       if k.find('classifier.1') < 0 and k.find('classifier') < 0}
    
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    return net
    return net
def loading_lora_checkpoint(net, path, args):
    """
    加载经过 LoRA 微调保存的检查点权重。
    主要用于在蒸馏或者测试推理时，加载预先训练好的带有 LoRA 参数的教师模型。
    """
    pretrained_dict = torch.load(path, map_location=args.device)
    model_dict = net.state_dict()
    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    
    return net
def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n and 'bias' not in n and 'classifier' not in n:
            p.requires_grad = False
    return
def multimodal_model_config_initializations(args, input_length=4096, is_student=False, is_hetero=False):
    device = args.device
    path = os.path.join(args.root, 'pretrained_checkpoint')
    num_leads = 12
    num_class = args.num_class
    expert_dim = 3 if args.alignment_mode == 'dual_stream' else 0

    if is_student:
        # 学生模型
        num_layers, ecg_complexity, pcg_complexity = 9, 64, 64
        net = MultimodalLSTransNet(
            nOUT=num_class, ecg_complexity=ecg_complexity, pcg_complexity=pcg_complexity,
            ecg_inchannels=num_leads, pcg_inchannels=1, input_length=input_length,
            num_layers=num_layers, num_encoder_layers=1, rank_list=0,
            expert_dim=expert_dim # 需确保模型类已更新此参数
        ).to(device)
        optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate)
        
    else:
        # 教师模型
        if is_hetero:
            num_layers, ecg_complexity, pcg_complexity = 47, 512, 512
            r = get_rank_list(num_layers, args.conv_r, args.trans_r)
            if 'lora' in args.ranklist:
                net = Hetero_MultimodalTeacherNet(
                nOUT=num_class, 
                ecg_complexity=ecg_complexity, 
                pcg_complexity=pcg_complexity,
                ecg_inchannels=num_leads, 
                pcg_inchannels=1, 
                input_length=input_length,
                num_layers=num_layers, rank_list=r, 
                loc_dim=5,
                expert_dim=expert_dim
            ).to(device)
                mark_only_lora_as_trainable(net)
            else:
                net = Hetero_MultimodalTeacherNet(
                    nOUT=num_class, 
                    ecg_complexity=ecg_complexity, pcg_complexity=pcg_complexity,
                    ecg_inchannels=num_leads, pcg_inchannels=1, 
                    input_length=input_length,
                    num_layers=num_layers, rank_list=0, 
                    loc_dim=5,
                    expert_dim=expert_dim
                ).to(device)
        else:
            num_layers, ecg_complexity, pcg_complexity = 30, 128, 128
            r = get_rank_list(num_layers, args.conv_r, args.trans_r)
            if 'lora' in args.ranklist:
                net = MultimodalLSTransNet(
                nOUT=num_class, 
                ecg_complexity=ecg_complexity, 
                pcg_complexity=pcg_complexity,
                ecg_inchannels=num_leads, 
                pcg_inchannels=1, 
                input_length=input_length,
                num_layers=num_layers, num_encoder_layers=1, 
                rank_list=r, loc_dim=5,
                expert_dim=expert_dim
            ).to(device)
                mark_only_lora_as_trainable(net)
            else:
                net = MultimodalLSTransNet(
                    nOUT=num_class, 
                    ecg_complexity=ecg_complexity, 
                    pcg_complexity=pcg_complexity,
                    ecg_inchannels=num_leads, 
                    pcg_inchannels=1, 
                    input_length=input_length,
                    num_layers=num_layers, num_encoder_layers=1, 
                    rank_list=0, loc_dim=5,
                    expert_dim=expert_dim
                ).to(device)
        
        if 'lora' in args.ranklist:
            params_to_update =[]
            for name, param in net.named_parameters():
                if name.find('lora') > -1 or name.find('bias') > -1 or name.find('classifier') > -1:
                    params_to_update.append(param)
            optimizer = optim.AdamW(params_to_update, lr=args.learning_rate)
        else:
            optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate)
            
    return net, optimizer

def validate(model, valloader, device, threshold=0.5 * np.ones(5), iftest=False, iftrain=False, args=None):
    model.eval()
    losses, probs, lbls = [], [], []
    for step, (inp_windows_t, lbl_t) in enumerate(valloader):
        inp_windows_t, lbl_t = inp_windows_t.float().to(device), lbl_t.int().to(device)
        if inp_windows_t.dim() == 3 and "NN_default" in str(type(model)):
            inp_windows_t = inp_windows_t.unsqueeze(2)
        with torch.no_grad():
            out = model(inp_windows_t)
            loss = F.binary_cross_entropy_with_logits(out, lbl_t.float())
            prob = out.sigmoid().data.cpu().numpy()
            losses.append(loss.item())
            probs.append(prob)
            lbls.append(lbl_t.data.cpu().numpy())
            # logit.append(out.data.cpu().numpy())
    lbls = np.concatenate(lbls)
    probs = np.concatenate(probs)

    if iftest:
        valid_result = print_result(np.mean(losses), lbls.copy(), probs.copy(), 'test', threshold)
    elif iftrain:
        threshold = find_thresholds(lbls.copy(), probs.copy())
        valid_result = print_result(np.mean(losses), lbls.copy(), probs.copy(), 'train', threshold)
    else:
        threshold = find_thresholds(lbls, probs)
        valid_result = print_result(np.mean(losses), lbls, probs, 'valid', threshold)
    neg_ratio = (len(probs) - np.sum(probs, axis=0)) / np.sum(probs, axis=0)
    valid_result.update({'neg_ratio': neg_ratio})
    valid_result.update({'threshold': threshold})
    return valid_result
def train_one_step_multimodal(net, optimizer, data, args, prior_tool, mode='none'):
    # ecg_data shape: [Batch, 12, Length] -> 取第一导联做 R 波检测
    # pcg_data shape: [Batch, 1, Length]
    ecg_data, pcg_data, pcg_loc, labels = data
    fs = 1000
    with torch.no_grad():
        # 检测 R 波 (使用 lead 0)
        r_peaks = PhysioDetector.detect_ecg_r_peaks(ecg_data[:, 0, :], fs=fs)
        # 检测 S1 波 (依赖 R 波)
        s1_peaks = PhysioDetector.detect_pcg_s1_peaks(pcg_data[:, 0, :], r_peaks, fs=fs)
    prior_mask = None
    expert_feat = None
    physio_loss = 0.0

    if mode == 'cascaded':
        # 传入检测到的 R 波进行位移
        _, pcg_data = prior_tool.get_cascaded_aligned_signals(ecg_data, pcg_data, r_peaks)
    elif mode == 'anchor':
        # 计算特征图对应的 Mask
        prior_mask = prior_tool.get_anchor_mask(
            batch_size=ecg_data.shape[0],
            ecg_len=102, pcg_len=102, 
            r_peaks=[(p/40).int() for p in r_peaks]
        )
    elif mode == 'dual_stream':
        expert_feat = prior_tool.get_medical_expert_features(r_peaks, s1_peaks)

    # 前向传播
    outputs, ecg_feat, pcg_feat, attn_weights = net(ecg_data, pcg_data, pcg_loc, 
                                                   prior_mask=prior_mask, 
                                                   expert_feat=expert_feat)
    
    loss_hard = F.binary_cross_entropy_with_logits(outputs, labels)
    
    if mode == 'constraint' and attn_weights is not None:
        # 范式 3.2: 约束驱动 Loss
        target_map = prior_tool.get_physio_constraint_map(ecg_data.shape[0], 102, 102, [p//40 for p in r_peaks])
        # 使用 KL 散度约束注意力分布
        physio_loss = F.kl_div(attn_weights.log(), target_map, reduction='batchmean')
        
    total_loss = loss_hard + args.lambda_physio * physio_loss
    return total_loss, outputs

def multimodal_kd_teacher_model(args, is_hetero=False):
    device = args.device
    setup_seed(args.seed)
    prefix = "Hetero" if is_hetero else "Homo"
    mode = args.alignment_mode

    model_config = f"{args.model_config}_{prefix}{mode}_Multimodal_Teacher"
    checkpoint_name = f"{args.ft_dataset}_{args.tea_ranklist}_{model_config}_seed{args.seed}"
    path = os.path.join(args.root, 'pretrained_checkpoint')
    os.makedirs(path, exist_ok=True)
    batch_size = args.batch_size
    args.learning_rate = 0.002 if batch_size > 64 else 0.001

    dataset_train, dataset_valid, dataset_test = MultimodalDataset_loading(args=args)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    label_iter = iter(loader_train)
    sample_data, _ = dataset_train[0]
    actual_input_length = sample_data.shape[-1] 

    start_time = time.time()
    prior_tool = PhysioPriorTool(sampling_rate=1000, device=args.device)
    net, optimizer = multimodal_model_config_initializations(args, is_hetero=is_hetero)
    iteration = len(loader_train) * args.ft_epoch
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, dataset_name=checkpoint_name, delta=0, args=args)
    my_lr_scheduler = get_linear_schedule_with_warmup(optimizer, int(iteration*0.01), iteration, last_epoch=-1)

    net.train()
    total_loss = 0.0
    best_threshold = 0.5 * np.ones(args.num_class)
    label_iter = iter(loader_train)
    
    pbar = tqdm(range(iteration), desc=f"【{prefix}_{args.alignment_mode} Teacher】 Finetuning")
    for step in pbar:
        try:
            ecg_data, pcg_data, pcg_loc, labels = next(label_iter)
        except StopIteration:
            label_iter = iter(loader_train)
            ecg_data, pcg_data, pcg_loc, labels = next(label_iter)

        ecg_data = ecg_data.float().to(device)
        pcg_data = pcg_data.float().to(device)
        pcg_loc = pcg_loc.float().to(device)
        labels = labels.float().to(device)

        ecg_data, pcg_data, pcg_loc, labels = Cutmix_Multimodal(ecg_data, pcg_data, pcg_loc, labels, device)
        optimizer.zero_grad()
        outputs, _, _, _ = net(ecg_data, pcg_data, pcg_loc)
        
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        my_lr_scheduler.step()

        current_loss = loss.item()
        total_loss += current_loss
        pbar.set_postfix({'loss': f'{current_loss:.4f}', 'avg_loss': f'{total_loss/(step+1):.4f}'})

        if (step + 1) % len(loader_train) == 0:
            net.eval() 
            valid_result = validate(net, loader_valid, threshold=0.5*np.ones(args.num_class), device=device, args=args)
            early_stopping(1 / valid_result['Map_value'], net)
            if early_stopping.counter == 0: 
                best_threshold = valid_result['threshold']
            if early_stopping.early_stop:
                tqdm.write("\nEarly stopping triggered") 
                break
            net.train()

    pbar.close()
    end_time = time.time()
    actual_steps = step + 1
    running_time = (end_time - start_time) / actual_steps
    allocated_memory = torch.cuda.max_memory_allocated(device)  # max_
  
    net.merge_net() # 部署前融合 LoRA 权重
    net.eval()
    test_result = validate(net, loader_test, device, iftest=True, threshold=best_threshold, args=args)
    test_result.update({
        # 'trainable_num': trainable_num,
        'memory': allocated_memory,
        'time': running_time
    })
    tea_res_dir = os.path.join(args.root, 'results')
    os.makedirs(tea_res_dir, exist_ok=True)
    tea_res_save_path = os.path.join(tea_res_dir, checkpoint_name + '_result.json')
    serializable_res = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in test_result.items()}
    with open(tea_res_save_path, 'w') as f:
        json.dump(serializable_res, f, indent=4)

    return net
def multimodal_kd_student_model(args, fold_idx, is_hetero=False):
    prefix = "Hetero" if is_hetero else "Homo"
    teacher_checkpoint_name = f"{args.ft_dataset}_{args.tea_ranklist}_{args.model_config}_{prefix}_Multimodal_Teacher_seed{args.seed}"
    teacher_checkpoint_path = os.path.join(args.root, 'pretrained_checkpoint', f"{teacher_checkpoint_name}_checkpoint.pt")
    args.ranklist = 'FT'
    device = args.device
    args.learning_rate = 0.002
    batch_size = args.batch_size
    mode = args.alignment_mode
    T = getattr(args, 'kd_temperature', 3.0 if is_hetero else 2.0)
    alpha = getattr(args, 'kd_alpha', 0.6 if is_hetero else 0.5)
    prior_mask = None
    expert_feat = None
    setup_seed(args.seed)
    
    dataset_train, dataset_valid, dataset_test = MultimodalDataset_loading(args=args)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    label_iter = iter(loader_train)
    sample_data, _ = dataset_train[0]
    actual_input_length = sample_data.shape[-1] 
    
    # 初始化教师模型
    if os.path.exists(teacher_checkpoint_path):
        print(f"--- Teacher checkpoint found at {teacher_checkpoint_path}. ---")
        teacher_net, _ = multimodal_model_config_initializations(args, input_length=actual_input_length, is_student=False, is_hetero=is_hetero)
        teacher_net = loading_lora_checkpoint(teacher_net, teacher_checkpoint_path, args)
        teacher_net.eval()
        if hasattr(teacher_net, 'merge_net'):
            teacher_net.merge_net()
    else:
        print("--- Teacher checkpoint not found. Starting teacher training... ---")
        teacher_net = multimodal_kd_teacher_model(args, is_hetero=is_hetero)
    teacher_net.eval()
    for param in teacher_net.parameters():
        param.requires_grad = False

    # 初始化学生模型
    student_config = f"Student_KD_{prefix}_{args.model_config}_{args.tea_ranklist}"
    checkpoint_name = f"{args.ft_dataset}_{student_config}_seed{args.seed}"
    
    prior_tool = PhysioPriorTool(sampling_rate=1000, device=args.device)
    net, optimizer = multimodal_model_config_initializations(args, input_length=actual_input_length, is_student=True, is_hetero=False)
    iteration = len(loader_train) * args.ft_epoch
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, dataset_name=checkpoint_name, delta=0, args=args)
    my_lr_scheduler = get_linear_schedule_with_warmup(optimizer, int(iteration*0.01), iteration, last_epoch=-1)

    net.train()
    best_threshold = 0.5 * np.ones(args.num_class)
    label_iter = iter(loader_train)
    start_time = time.time()
    
    pbar = tqdm(range(iteration), desc=f"【{prefix}_{mode} KD Student】 Training")
    for step in pbar:
        try:
            ecg_data, pcg_data, pcg_loc, labels = next(label_iter)
        except StopIteration:
            label_iter = iter(loader_train)
            ecg_data, pcg_data, pcg_loc, labels = next(label_iter)

        ecg_data = ecg_data.float().to(device)
        pcg_data = pcg_data.float().to(device)
        pcg_loc = pcg_loc.float().to(device)
        labels = labels.float().to(device)
    
        ecg_data, pcg_data, pcg_loc, labels = Cutmix_Multimodal(ecg_data, pcg_data, pcg_loc, labels, device)
        
        # 无梯度产生教师指导软标签
        with torch.no_grad():
            t_out, t_ecg_f, t_pcg_f, _ = teacher_net(ecg_data, pcg_data, pcg_loc, 
                                                    prior_mask=prior_mask, 
                                                    expert_feat=expert_feat)
        # if hasattr(args, 'leads_for_student'):
        #     ecg_data = mask_ecg_signal(ecg_data, pcg_data, args.leads_for_student)

        optimizer.zero_grad()
        outputs, s_ecg_f, s_pcg_f, _ = net(ecg_data, pcg_data, pcg_loc, 
                                                prior_mask=prior_mask, 
                                                expert_feat=expert_feat)

        # 联合交叉熵：Hard loss(Labels) + Soft loss(Teacher Labels)
        loss_hard = F.binary_cross_entropy_with_logits(outputs, labels)
        student_logits_T = outputs / T
        teacher_logits_T = t_out / T
        loss_soft = F.binary_cross_entropy_with_logits(student_logits_T, teacher_logits_T.sigmoid()) * (T * T)
        loss_feat = F.mse_loss(s_ecg_f, t_ecg_f) + F.mse_loss(s_pcg_f, t_pcg_f)
        loss = (1.0 - alpha) * loss_hard + alpha * loss_soft
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        my_lr_scheduler.step()
        
        pbar.set_postfix({'L_all': f'{loss.item():.3f}', 'L_hard': f'{loss_hard.item():.3f}', 'L_soft': f'{loss_soft.item():.3f}'})

        if (step + 1) % len(loader_train) == 0:
            net.eval()
            valid_result = validate(net, loader_valid, threshold=0.5*np.ones(args.num_class), device=device, args=args)
            early_stopping(1 / valid_result['Map_value'], net)
            if early_stopping.counter == 0: 
                best_threshold = valid_result['threshold']
            if early_stopping.early_stop:
                tqdm.write("\nEarly stopping triggered") 
                break
            net.train()
        
    pbar.close()
    end_time = time.time()
    actual_steps = step + 1
    running_time = (end_time - start_time) / actual_steps
    allocated_memory = torch.cuda.max_memory_allocated(device)  # max_

    net.eval()
    test_result = validate(net, loader_test, device, iftest=True, threshold=best_threshold, args=args)
    test_result.update({
        # 'trainable_num': trainable_num,
        'memory': allocated_memory,
        'time': running_time
    })
    return test_result