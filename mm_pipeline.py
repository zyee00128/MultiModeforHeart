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
        target_key = None
        if k in model_dict:
            target_key = k
        elif f"backbone.{k}" in model_dict:
            target_key = f"backbone.{k}"
        elif f"ecg_encoder.{k}" in model_dict:
            target_key = f"ecg_encoder.{k}"
        elif f"pcg_encoder.{k}" in model_dict:
            target_key = f"pcg_encoder.{k}"
        elif f"ecg_encoder.backbone.{k}" in model_dict:
            target_key = f"ecg_encoder.backbone.{k}"
        elif f"pcg_encoder.backbone.{k}" in model_dict:
            target_key = f"pcg_encoder.backbone.{k}"
        if target_key is not None:
            expected_shape = model_dict[target_key].shape

            if len(expected_shape) == len(v.shape) and v.dim() in [3, 4]:
                if v.shape[1] == 12 and expected_shape[1] == 1:
                    print(f"[Weight Surgery] Slicing in_channels of '{target_key}' from 12 to 1 (using Lead II).")
                    if v.dim() == 3:
                        v = v[:, 1:2, :]      # 1D 卷积权重: [C_out, 1, K]
                    elif v.dim() == 4:
                        v = v[:, 1:2, :, :]   # 2D 卷积权重: [C_out, 1, H, W]
            new_pretrained_dict[target_key] = v
                

    # 过滤掉顶层分类器的权重
        filtered_pretrained_dict = {
        k: v for k, v in new_pretrained_dict.items() 
        if 'classifier' not in k}

    model_dict.update(filtered_pretrained_dict)
    net.load_state_dict(model_dict)

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
def get_downsample_factor(model):
    """
    动态获取时序下采样因子。
    
    不同模型的骨干网络深度不同：
    - Hetero 教师模型：NN_default (num_encoder_layers=3)，下采样率为 4^3 = 64
    - Homo 教师模型：LSTransECG -> LSTrans_default (num_encoder_layers=1)，下采样率为 4^1 = 4
    - KD 学生模型：LSTransECG -> LSTrans_default (num_encoder_layers=1)，下采样率为 4^1 = 4
    """
    if hasattr(model, 'ecg_encoder'):
        encoder = model.ecg_encoder
        # 情况 A: 教师网络直接包含 num_encoder_layers
        if hasattr(encoder, 'num_encoder_layers'):
            return 4 ** encoder.num_encoder_layers
        # 情况 B: LSTrans 封装的 backbone 包含 num_encoder_layers
        elif hasattr(encoder, 'backbone') and hasattr(encoder.backbone, 'num_encoder_layers'):
            return 4 ** encoder.backbone.num_encoder_layers
            
    return 4  # 默认降采样倍率为 4

def validate(model, valloader, device, threshold=0.5 * np.ones(5), iftest=False, iftrain=False, args=None):
    model.eval()
    losses, probs, lbls = [], [], []
    prior_tool = PhysioPriorTool(sampling_rate=1000, device=device) if args is not None else None

    for step, batch_data in enumerate(valloader):
        if len(batch_data) == 4:  # 多模态解包
            ecg_data, pcg_data, pcg_loc, lbl_t = batch_data
            ecg_data = ecg_data.float().to(device)
            pcg_data = pcg_data.float().to(device)
            pcg_loc = pcg_loc.float().to(device)
            lbl_t = lbl_t.int().to(device)

            prior_mask = None
            expert_feat = None
            mode = getattr(args, 'alignment_mode', 'none')

            if prior_tool is not None and mode != 'none':
                with torch.no_grad():
                    ecg_lead0 = ecg_data[:, 0, :]
                    pcg_lead0 = pcg_data[:, 0, :]
                    r_peaks = PhysioDetector.detect_ecg_r_peaks(ecg_lead0, fs=1000)
                    s1_peaks = PhysioDetector.detect_pcg_s1_peaks(pcg_lead0, r_peaks, fs=1000)
                if mode == 'cascaded':
                    _, pcg_data = prior_tool.get_cascaded_aligned_signals(ecg_data, pcg_data, r_peaks)
                elif mode == 'anchor':
                    downsample = get_downsample_factor(model)
                    ecg_seq_len = ecg_data.shape[-1] // downsample
                    pcg_seq_len = pcg_data.shape[-1] // downsample
                    r_peaks_downsampled = [torch.clamp((p / downsample).int(), 0, ecg_seq_len - 1) for p in r_peaks]
                    prior_mask = prior_tool.get_anchor_mask(
                        batch_size=ecg_data.shape[0],
                        ecg_len=ecg_seq_len,
                        pcg_len=pcg_seq_len,
                        r_peaks=r_peaks_downsampled
                    )
                elif mode == 'dual_stream':
                    expert_feat = prior_tool.get_medical_expert_features(r_peaks, s1_peaks)

            with torch.no_grad():
                out, _, _, _, _ = model(ecg_data, pcg_data, pcg_loc, prior_mask=prior_mask, expert_feat=expert_feat)

        else:  # 单模态解包兼容
            inp_windows_t, lbl_t = batch_data
            inp_windows_t = inp_windows_t.float().to(device)
            lbl_t = lbl_t.int().to(device)
            if inp_windows_t.dim() == 3 and "NN_default" in str(type(model)):
                inp_windows_t = inp_windows_t.unsqueeze(2)
            with torch.no_grad():
                out = model(inp_windows_t)

        loss = F.binary_cross_entropy_with_logits(out, lbl_t.float())
        prob = out.sigmoid().data.cpu().numpy()
        losses.append(loss.item())
        probs.append(prob)
        lbls.append(lbl_t.data.cpu().numpy())

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
    valid_result.update({'neg_ratio': neg_ratio, 'threshold': threshold})
    return valid_result
def multimodal_model_config_initializations(args, input_length=4096, is_student=False, is_hetero=False):
    device = args.device
    path = os.path.join(args.root, 'pretrained_checkpoint')
    num_leads = 1
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

        student_pretrained_path = getattr(args, 'student_pretrained_path')
        if os.path.exists(student_pretrained_path):
            print(f"--- Loading student pretrained weights from {student_pretrained_path} ---")
            net = load_pretrained_model(net, student_pretrained_path, args)
        else:
            print(f"--- Warning: Student pretrained weights not found at {student_pretrained_path}. Training from scratch. ---")
        optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate)
        
    else:
        prefix = "Hetero" if is_hetero else "Homo"
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

        teacher_pretrained_path = getattr(args, 'teacher_pretrained_path')
        # 根据 alignment_mode 动态构造 joint_pretrain 路径，支持 Homo/Hetero
        mode = args.alignment_mode
        model_config = f"{args.model_config}_EPHNOGRAM_JointPretrain_{prefix}_{mode}"
        checkpoint_name = f"{model_config}_seed{args.seed}"
        joint_path = os.path.join(args.root, 'pretrained_checkpoint', f"{checkpoint_name}_checkpoint.pt")
        teacher_pretrained_path = joint_path

        if os.path.exists(teacher_pretrained_path):
            print(f"--- Loading teacher pretrained weights from {teacher_pretrained_path} ---")
            net = load_pretrained_model(net, teacher_pretrained_path, args)
        else:
            print(f"--- Warning: Teacher pretrained weights not found at {teacher_pretrained_path}. Training from scratch. ---")

        if 'lora' in args.ranklist:
            params_to_update =[]
            for name, param in net.named_parameters():
                if name.find('lora') > -1 or name.find('bias') > -1 or name.find('classifier') > -1:
                    params_to_update.append(param)
            optimizer = optim.AdamW(params_to_update, lr=args.learning_rate)
        else:
            optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate)
            
    return net, optimizer

def multimodal_kd_teacher_model(args, is_hetero=False):
    device = args.device
    setup_seed(args.seed)
    prefix = "Hetero" if is_hetero else "Homo"
    mode = args.alignment_mode
    model_config = f"{args.model_config}_{prefix}{mode}_Multimodal_Teacher"
    checkpoint_name = f"{args.ft_dataset}_{model_config}_seed{args.seed}"
    path = os.path.join(args.root, 'pretrained_checkpoint')
    os.makedirs(path, exist_ok=True)
    batch_size = args.batch_size
    args.learning_rate = 0.002 if batch_size > 64 else 0.001
    lambda_physio = getattr(args, 'lambda_physio', 0.1)

    dataset_train, dataset_valid, dataset_test = MultimodalDataset_loading(args=args)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    label_iter = iter(loader_train)
    iteration = len(loader_train) * args.ft_epoch
    sample_data, _ = dataset_train[0]
    actual_input_length = sample_data.shape[-1] 
    total_loss = 0.0
    best_threshold = 0.5 * np.ones(args.num_class)

    prior_tool = PhysioPriorTool(sampling_rate=1000, device=args.device)
    net, optimizer = multimodal_model_config_initializations(args, input_length=actual_input_length, is_student=False, is_hetero=is_hetero)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, dataset_name=checkpoint_name, delta=0, args=args)
    my_lr_scheduler = get_linear_schedule_with_warmup(optimizer, int(iteration*0.01), iteration, last_epoch=-1)

    net.train()
    start_time = time.time()
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

        # === 1. 级联式对齐（需在 Cutmix 之前对原始物理信号进行粗对齐） ===
        if mode == 'cascaded':
            with torch.no_grad():
                # 使用 Lead 0 检测 R 波并位移对齐
                r_peaks_raw = PhysioDetector.detect_ecg_r_peaks(ecg_data[:, 0, :], fs=1000)
            ecg_data, pcg_data = prior_tool.get_cascaded_aligned_signals(ecg_data, pcg_data, r_peaks_raw)
        # 混合多模态信号
        ecg_data, pcg_data, pcg_loc, labels = Cutmix_Multimodal(ecg_data, pcg_data, pcg_loc, labels, device)
        
        # 初始化对齐参数与物理 Loss
        prior_mask = None
        expert_feat = None
        physio_loss = 0.0

        # === 2. 其它对齐模式（在 Cutmix 后提取当前混合信号的生理特征） ===
        if mode in ['constraint', 'anchor', 'dual_stream']:
            with torch.no_grad():
                r_peaks = PhysioDetector.detect_ecg_r_peaks(ecg_data[:, 0, :], fs=1000)
                s1_peaks = PhysioDetector.detect_pcg_s1_peaks(pcg_data[:, 0, :], r_peaks, fs=1000)
            
            ds = get_downsample_factor(net)
            ecg_seq_len = ecg_data.shape[-1] // ds
            pcg_seq_len = pcg_data.shape[-1] // ds

            if mode == 'anchor':
                # 缩放 R 波位置以匹配下采样后的时序
                r_peaks_downsampled = [(p / ds).int() for p in r_peaks]
                prior_mask = prior_tool.get_anchor_mask(
                    batch_size=ecg_data.shape[0],
                    ecg_len=ecg_seq_len, pcg_len=pcg_seq_len,
                    r_peaks=r_peaks_downsampled
                )
            elif mode == 'dual_stream':
                expert_feat = prior_tool.get_medical_expert_features(r_peaks, s1_peaks)

        optimizer.zero_grad()
        outputs, ecg_feat, pcg_feat, _, attn_weights = net(
            ecg_data, pcg_data, pcg_loc,
            prior_mask=prior_mask,
            expert_feat=expert_feat
        )
        
        loss_hard = F.binary_cross_entropy_with_logits(outputs, labels)
        # === 约束驱动学习（生理一致性 Loss 约束注意力热力图） ===
        if mode == 'constraint' and attn_weights is not None:
            r_peaks_downsampled = [(p / ds).int() for p in r_peaks]
            target_map = prior_tool.get_physio_constraint_map(
                batch_size=ecg_data.shape[0],
                ecg_len=ecg_seq_len, pcg_len=pcg_seq_len,
                r_peaks=r_peaks_downsampled
            )
            # 使用 KL 散度约束交叉注意力分布
            physio_loss = F.kl_div((attn_weights + 1e-9).log(), target_map, reduction='batchmean')
        loss = loss_hard + lambda_physio * physio_loss
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
    teacher_checkpoint_name = f"{args.ft_dataset}_{args.model_config}_{prefix}_Multimodal_Teacher_seed{args.seed}"
    teacher_checkpoint_path = os.path.join(args.root, 'pretrained_checkpoint', f"{teacher_checkpoint_name}_checkpoint.pt")
    device = args.device
    args.learning_rate = 0.002
    batch_size = args.batch_size
    mode = args.alignment_mode
    T = getattr(args, 'kd_temperature', 3.0 if is_hetero else 2.0)
    alpha = getattr(args, 'kd_alpha', 0.6 if is_hetero else 0.5)
    lambda_physio = getattr(args, 'lambda_physio', 0.1)
    setup_seed(args.seed)
    
    dataset_train, dataset_valid, dataset_test = MultimodalDataset_loading(args=args)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    label_iter = iter(loader_train)
    iteration = len(loader_train) * args.ft_epoch
    sample_data, _ = dataset_train[0]
    actual_input_length = sample_data.shape[-1] 
    best_threshold = 0.5 * np.ones(args.num_class)

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
    args.ranklist = 'FT'
    student_config = f"Student_KD_{prefix}_{args.model_config}"
    checkpoint_name = f"{args.ft_dataset}_{student_config}_seed{args.seed}"
    
    prior_tool = PhysioPriorTool(sampling_rate=1000, device=args.device)
    net, optimizer = multimodal_model_config_initializations(args, input_length=actual_input_length, is_student=True, is_hetero=False)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, dataset_name=checkpoint_name, delta=0, args=args)
    my_lr_scheduler = get_linear_schedule_with_warmup(optimizer, int(iteration*0.01), iteration, last_epoch=-1)

    net.train()
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

        # === 1. 级联式对齐（Cutmix 前粗对齐） ===
        if mode == 'cascaded':
            with torch.no_grad():
                r_peaks_raw = PhysioDetector.detect_ecg_r_peaks(ecg_data[:, 0, :], fs=1000)
            ecg_data, pcg_data = prior_tool.get_cascaded_aligned_signals(ecg_data, pcg_data, r_peaks_raw)

        # 多模态 Cutmix
        ecg_data, pcg_data, pcg_loc, labels = Cutmix_Multimodal(ecg_data, pcg_data, pcg_loc, labels, device)
        
        ds_tea = get_downsample_factor(teacher_net)
        ds_stu = get_downsample_factor(net)
        prior_mask_tea = None
        prior_mask_stu = None
        expert_feat = None
        physio_loss = 0.0

        # === 2. 其它对齐模式 ===
        if mode in ['anchor', 'dual_stream']:
            with torch.no_grad():
                r_peaks = PhysioDetector.detect_ecg_r_peaks(ecg_data[:, 0, :], fs=1000)
                s1_peaks = PhysioDetector.detect_pcg_s1_peaks(pcg_data[:, 0, :], r_peaks, fs=1000)
            
            if mode == 'anchor':
                ecg_len_tea = ecg_data.shape[-1] // ds_tea
                pcg_len_tea = pcg_data.shape[-1] // ds_tea
                r_peaks_tea = [(p / ds_tea).int() for p in r_peaks]
                prior_mask_tea = prior_tool.get_anchor_mask(
                    batch_size=ecg_data.shape[0],
                    ecg_len=ecg_len_tea, pcg_len=pcg_len_tea,
                    r_peaks=r_peaks_tea
                )
                ecg_len_stu = ecg_data.shape[-1] // ds_stu
                pcg_len_stu = pcg_data.shape[-1] // ds_stu
                r_peaks_stu = [(p / ds_stu).int() for p in r_peaks]
                prior_mask_stu = prior_tool.get_anchor_mask(
                    batch_size=ecg_data.shape[0],
                    ecg_len=ecg_len_stu, pcg_len=pcg_len_stu,
                    r_peaks=r_peaks_stu
                )

            elif mode == 'dual_stream':
                expert_feat = prior_tool.get_medical_expert_features(r_peaks, s1_peaks)
        
        # 无梯度产生教师指导软标签
        with torch.no_grad():
            t_out, t_ecg_f, t_pcg_f, _, _ = teacher_net(
                ecg_data, pcg_data, pcg_loc, 
                prior_mask=prior_mask_tea, 
                expert_feat=expert_feat
            )
        # if hasattr(args, 'leads_for_student'):
        #     ecg_data = mask_ecg_signal(ecg_data, pcg_data, args.leads_for_student)

        optimizer.zero_grad()
        outputs, s_ecg_f, s_pcg_f, _, attn_weights = net(
            ecg_data, pcg_data, pcg_loc, 
            prior_mask=prior_mask_stu, 
            expert_feat=expert_feat
        )

        student_logits_T = outputs / T
        teacher_logits_T = t_out / T
        if s_ecg_f.dim() == 3:
            s_ecg_f_pooled = s_ecg_f.mean(dim=1)
            t_ecg_f_pooled = t_ecg_f.mean(dim=1)
        else:
            s_ecg_f_pooled, t_ecg_f_pooled = s_ecg_f, t_ecg_f

        if s_pcg_f.dim() == 3:
            s_pcg_f_pooled = s_pcg_f.mean(dim=1)
            t_pcg_f_pooled = t_pcg_f.mean(dim=1)
        else:
            s_pcg_f_pooled, t_pcg_f_pooled = s_pcg_f, t_pcg_f
        # === 约束驱动学习（生理一致性 Loss 约束注意力热力图） ===
        if mode == 'constraint' and attn_weights is not None:
            r_peaks_downsampled = [(p / ds_stu).int() for p in r_peaks]
            target_map = prior_tool.get_physio_constraint_map(
                batch_size=ecg_data.shape[0],
                ecg_len=ecg_len_stu, pcg_len=pcg_len_stu,
                r_peaks=r_peaks_downsampled
            )
            # 使用 KL 散度约束交叉注意力分布
            physio_loss = F.kl_div((attn_weights + 1e-9).log(), target_map, reduction='batchmean')

        loss_hard = F.binary_cross_entropy_with_logits(outputs, labels)
        loss_soft = F.binary_cross_entropy_with_logits(student_logits_T, teacher_logits_T.sigmoid()) * (T * T)
        loss_feat = F.mse_loss(s_ecg_f_pooled, t_ecg_f_pooled) + F.mse_loss(s_pcg_f_pooled, t_pcg_f_pooled)
        loss = (1.0 - alpha) * loss_hard + alpha * loss_soft + 0.1 * loss_feat
        
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