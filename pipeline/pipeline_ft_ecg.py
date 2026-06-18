import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
from tqdm import tqdm
import os
import json
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from model.model_code_default import (LSTransECG, NN_default,
                            Cutmix, Cutmix_ECG, Cutmix_ECG_student,
                            mask_ecg_signal)
from ablation_model.model_ablation import *
from tools.pytorchtools import EarlyStopping
from tools.evaluation import print_result, find_thresholds
from tools.datacollection import ECGfinetunedataset_loading

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)
def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n and 'bias' not in n and 'classifier.1.weight' not in n:
            p.requires_grad = False
    return
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
## model validation on single GPU
def validate(model, valloader, device, threshold=0.5 * np.ones(5), iftest=False, iftrain=False):
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
def validate_student(model, valloader, device, threshold=0.5 * np.ones(5), iftest=False, iftrain=False, args=None):
    return validate(model, valloader, device, threshold, iftest, iftrain)

def load_pretrained_model(net, path, args):
    pretrained_dict = torch.load(path, map_location=args.device)
    model_dict = net.state_dict()
    new_pretrained_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict:
            new_pretrained_dict[k] = v
        elif f"backbone.{k}" in model_dict:
            new_pretrained_dict[f"backbone.{k}"] = v

    pretrained_dict = {k: v for k, v in new_pretrained_dict.items() if k.find('classifier.1') < 0}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    return net
def loading_lora_checkpoint(net, path, args):
    pretrained_dict = torch.load(path, map_location=args.device)
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    return net

def get_teacher_model(args, input_length, num_class):
    device = args.device
    path = os.path.join(args.root, 'pretrained_checkpoint')
    
    ablation_odd = 'none'
    ablation_even = 'none'

    if args.mode == 'homo':
        file_name_pretrain = f'CODE15_Pretrain_{args.model_config}_checkpoint.pt'
        if args.model_config == 'large':
            num_layers, complexity = 42, 768
        else:  # 'light'
            num_layers, complexity = 30, 128
        r = get_rank_list(num_layers, args.conv_r, args.trans_r)
        
        if args.ranklist == 'lora':
            net = LSTransECG(nOUT=num_class, out_channels=complexity, in_channels=12, 
                             input_length=input_length, num_layers=num_layers, rank_list=r,
                             ablation_odd=ablation_odd, ablation_even=ablation_even).to(device)
            mark_only_lora_as_trainable(net)
        else:  # 'ft'
            net = LSTransECG(nOUT=num_class, out_channels=complexity, in_channels=12, 
                             input_length=input_length, num_layers=num_layers, rank_list=0,
                             ablation_odd=ablation_odd, ablation_even=ablation_even).to(device)
            
    elif args.mode == 'hetero':
        file_name_pretrain = 'CODE_testmediumbias_full_checkpoint.pkl'
        num_layers, complexity = 47, 512
        r = get_rank_list(num_layers, args.conv_r, args.trans_r)
        
        if args.ranklist == 'lora':
            net = NN_default(nOUT=num_class, complexity=complexity, inputchannel=12, 
                             input_length=input_length,num_layers=num_layers, rank_list=r).to(device)
            mark_only_lora_as_trainable(net)
        else:  # 'ft'
            net = NN_default(nOUT=num_class, complexity=complexity, inputchannel=12,  
                             input_length=input_length,num_layers=num_layers, rank_list=0).to(device)
    else:
        raise ValueError(f"Teacher is not used in student-only mode: {args.mode}")

    net = load_pretrained_model(net, os.path.join(path, file_name_pretrain), args)
    
    if args.ranklist == 'lora':
        params_to_update = [param for name, param in net.named_parameters() 
                            if 'lora' in name or 'bias' in name or 'classifier.1.weight' in name]
        optimizer = optim.AdamW(params_to_update, lr=args.learning_rate)
    else:
        optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate)
        
    return net, optimizer
def get_student_model(args, input_length, num_class):
    device = args.device
    path = os.path.join(args.root, 'pretrained_checkpoint')
    prestu_checkpoint = "CODE15_Pretrain_student"
    prestu_path = os.path.join(path, prestu_checkpoint + '_checkpoint.pt')
    
    num_layers, complexity = 9, 64
    ablation_odd = getattr(args, 'ablation_odd', 'none')
    ablation_even = getattr(args, 'ablation_even', 'none')

    # 学生模型默认在分类微调时不使用 LoRA (rank_list=0) 且全量训练
    net = LSTransECG(nOUT=num_class, out_channels=complexity, in_channels=12, 
                     input_length=input_length, num_layers=num_layers, rank_list=0,
                     ablation_odd=ablation_odd,
                     ablation_even=ablation_even).to(device)
    net = load_pretrained_model(net, prestu_path, args)
    
    optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate)
    return net, optimizer

def train_teacher(args, fold_idx=0):
    device = args.device
    setup_seed(args.seed)
    path = os.path.join(args.root, 'pretrained_checkpoint')
    batch_size = args.batch_size
    args.learning_rate = 0.002 if batch_size > 64 else 0.001
    fold_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
    checkpoint_name = f"{args.ft_dataset}_{args.ranklist}_{args.mode}_teacher_{args.model_config}_seed{args.seed}{fold_suffix}"

    dataset_train, dataset_valid, dataset_test = ECGfinetunedataset_loading(args=args, fold_idx=fold_idx)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    iteration = len(loader_train) * args.ft_epoch
    if args.ft_dataset in ['WFDB_Ga', 'WFDB_ChapmanShaoxing']:
        iteration *= 2
    sample_data, _ = dataset_train[0]
    actual_input_length = sample_data.shape[-1]
    
    net, optimizer = get_teacher_model(args, actual_input_length, args.num_class)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, dataset_name=checkpoint_name, delta=0, args=args)
    my_lr_scheduler = get_linear_schedule_with_warmup(optimizer, int(iteration * 0.01), iteration, last_epoch=-1)

    start_time = time.time()
    net.train()
    total_loss = 0.0
    best_threshold = 0.5 * np.ones(args.num_class)
    label_iter = iter(loader_train)
    pbar = tqdm(range(iteration), desc=f"Teacher Finetuning [{args.mode}]")
    
    for step in pbar:
        try:
            images, labels = next(label_iter)
        except StopIteration:
            label_iter = iter(loader_train)
            images, labels = next(label_iter)
        images = images.float().to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        
        # 异同构教师的输入格式与 Cutmix 处理方式存在区别
        if args.mode == 'hetero':
            if images.dim() == 3:
                images = images.unsqueeze(2)
            images, labels = Cutmix(images, labels, device)
        else:
            images, labels = Cutmix_ECG(images, labels, device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        my_lr_scheduler.step()

        current_loss = loss.item()
        total_loss += current_loss
        pbar.set_postfix({'loss': f'{current_loss:.4f}', 'avg_loss': f'{(total_loss / (step + 1)):.4f}'})

        if (step + 1) % len(loader_train) == 0:
            net.eval()
            valid_result = validate(net, loader_valid, threshold=0.5 * np.ones(args.num_class), device=device)
            early_stopping(1 / valid_result['Map_value'], net)
            if early_stopping.counter == 0:
                best_threshold = valid_result['threshold']
            if early_stopping.early_stop:
                tqdm.write("\nEarly stopping triggered")
                break
            net.train()

    pbar.close()

    end_time = time.time()
    running_time = (end_time - start_time) / (step + 1)
    allocated_memory = torch.cuda.max_memory_allocated(device)
    net = loading_lora_checkpoint(net, os.path.join(path, checkpoint_name + '_checkpoint.pt'), args)
    trainable_num = count_parameters(net)

    net.eval()
    if hasattr(net, 'merge_net'):
        net.merge_net()
    test_result = validate(net, loader_test, device, iftest=True, threshold=best_threshold)
    test_result.update({
        'trainable_num': trainable_num,
        'memory': allocated_memory,
        'time': running_time
    })

    tea_res_dir = os.path.join(args.root, 'results')
    os.makedirs(tea_res_dir, exist_ok=True)
    with open(os.path.join(tea_res_dir, checkpoint_name + '_result.json'), 'w') as f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in test_result.items()}, f, indent=4)
        
    return net
def train_student(args, fold_idx=0):
    device = args.device
    setup_seed(args.seed)
    path = os.path.join(args.root, 'pretrained_checkpoint')
    batch_size = args.batch_size
    args.learning_rate = 0.002
    fold_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
    checkpoint_name = f"{args.ft_dataset}_student_{args.mode}_{args.ranklist}_leads{args.leads_for_student}_seed{args.seed}{fold_suffix}"

    dataset_train, dataset_valid, dataset_test = ECGfinetunedataset_loading(args=args, fold_idx=fold_idx)
    actual_input_length = dataset_train[0][0].shape[-1]

    # 尝试初始化/加载教师模型（KD 模式）
    teacher_net = None
    if args.mode in ['homo', 'hetero']:
        fold_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
        teacher_checkpoint_name = f"{args.ft_dataset}_{args.ranklist}_{args.mode}_teacher_{args.model_config}_seed{args.seed}{fold_suffix}"
        teacher_checkpoint_path = os.path.join(path, teacher_checkpoint_name + '_checkpoint.pt')
        
        if os.path.exists(teacher_checkpoint_path):
            print(f"--- Loading existing Teacher checkpoint from {teacher_checkpoint_path} ---")
            teacher_net, _ = get_teacher_model(args, actual_input_length, args.num_class)
            teacher_net = loading_lora_checkpoint(teacher_net, teacher_checkpoint_path, args)
            teacher_net.eval()
            if hasattr(teacher_net, 'merge_net'):
                teacher_net.merge_net()
        else:
            print(f"--- Teacher checkpoint not found. Training teacher first... ---")
            teacher_net = train_teacher(args, fold_idx=fold_idx)
        teacher_net.eval()
        for param in teacher_net.parameters():
            param.requires_grad = False
    
    original_ranklist = args.ranklist
    args.ranklist = 'ft'
    # 建立学生模型和相关训练参数
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    iteration = len(loader_train) * args.ft_epoch
    if args.ft_dataset in ['WFDB_Ga', 'WFDB_ChapmanShaoxing']:
        iteration *= 2

    net, optimizer = get_student_model(args, actual_input_length, args.num_class)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, dataset_name=checkpoint_name, delta=0, args=args)
    my_lr_scheduler = get_linear_schedule_with_warmup(optimizer, int(iteration * 0.01), iteration, last_epoch=-1)

    start_time = time.time()
    net.train()
    best_threshold = 0.5 * np.ones(args.num_class)
    label_iter = iter(loader_train)
    pbar = tqdm(range(iteration), desc=f"Student Training [{args.mode} / Leads: {args.leads_for_student}]")

    for step in pbar:
        try:
            images, labels = next(label_iter)
        except StopIteration:
            label_iter = iter(loader_train)
            images, labels = next(label_iter)
        images = images.float().to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        # 针对学生做裁剪
        images, labels = Cutmix_ECG_student(images, labels, device, valid_lead_num=args.leads_for_student)
        
        # KD 模式：学生模型计算前由未被 mask 掉的原整组通道生成教师的 Soft 预测
        if teacher_net is not None:
            with torch.no_grad():
                if args.mode == 'hetero':
                    # 异构教师模型期望输入为 4D tensor
                    teacher_input = images.unsqueeze(2) if images.dim() == 3 else images
                    teacher_outputs = teacher_net(teacher_input)
                else:
                    teacher_outputs = teacher_net(images)

        # 对学生模型应用局部导联丢失遮挡（Mask）
        images = mask_ecg_signal(images, args.leads_for_student)
        
        optimizer.zero_grad()
        outputs = net(images)

        loss_hard = F.binary_cross_entropy_with_logits(outputs, labels)
        if teacher_net is not None:
            T = args.kd_temperature
            alpha = args.kd_alpha
            student_logits_T = outputs / T
            teacher_logits_T = teacher_outputs / T
            loss_soft = F.binary_cross_entropy_with_logits(student_logits_T, teacher_logits_T.sigmoid()) * (T * T)
            loss = (1.0 - alpha) * loss_hard + alpha * loss_soft
        else:
            loss = loss_hard

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        my_lr_scheduler.step()

        current_loss = loss.item()
        if teacher_net is not None:
            pbar.set_postfix({'L_all': f'{current_loss:.3f}', 'L_hard': f'{loss_hard.item():.3f}', 'L_soft': f'{loss_soft.item():.3f}'})
        else:
            pbar.set_postfix({'L_all': f'{current_loss:.3f}'})

        if (step + 1) % len(loader_train) == 0:
            net.eval()
            valid_result = validate(net, loader_valid, threshold=0.5 * np.ones(args.num_class), device=device)
            early_stopping(1 / valid_result['Map_value'], net)
            if early_stopping.counter == 0:
                best_threshold = valid_result['threshold']
            if early_stopping.early_stop:
                tqdm.write("\nEarly stopping triggered")
                break
            net.train()

    pbar.close()

    end_time = time.time()
    running_time = (end_time - start_time) / (step + 1)
    allocated_memory = torch.cuda.max_memory_allocated(device)
    net.load_state_dict(torch.load(os.path.join(path, checkpoint_name + '_checkpoint.pt'), map_location=device))
    trainable_num = count_parameters(net)

    net.eval()
    test_result = validate_student(net, loader_test, device, iftest=True, threshold=best_threshold, args=args)
    test_result.update({
        'trainable_num': trainable_num,
        'memory': allocated_memory,
        'time': running_time
    })

    args.ranklist = original_ranklist
    return test_result
