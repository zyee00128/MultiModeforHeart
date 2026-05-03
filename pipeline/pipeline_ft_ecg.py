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
                                      Cutmix_ECG, Cutmix_ECG_student, Cutmix,
                                      mask_ecg_signal)
from pytorchtools import EarlyStopping
from evaluation import print_result, find_thresholds
from datacollection import ECGfinetunedataset_loading

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
    # for n,p in model.named_parameters():
    #     if p.requires_grad:
    #         print(n)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    last_epoch=-1
):
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
        if 'lora_' not in n and 'bias' not in n and n !='classifier.1.weight':
            # print(n)
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
def validate_student(model, valloader, device, threshold=0.5 * np.ones(5), iftest=False, iftrain=False, args=None):
    model.eval()
    losses, probs, lbls, logit = [], [], [], []
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
            logit.append(out.data.cpu().numpy())
    lbls = np.concatenate(lbls)
    probs = np.concatenate(probs)
    logit = np.concatenate(logit)

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

def load_pretrained_model(net, path, args):
    pretrained_dict = torch.load(path, map_location=args.device)
    model_dict = net.state_dict()
    new_pretrained_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict:
            new_pretrained_dict[k] = v
        elif f"backbone.{k}" in model_dict:
            new_pretrained_dict[f"backbone.{k}"] = v

    pretrained_dict = {k: v 
                       for k, v in new_pretrained_dict.items() 
                       if k.find('classifier.1') < 0}
    
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    return net

def loading_lora_checkpoint(net, path, args):
    pretrained_dict = torch.load(path, map_location=args.device)
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    # net.load_state_dict(torch.load(path))
    return net
def model_config_initializations(args, input_length=4096):
    model_config = args.model_config
    device = args.device
    path = args.root + '/pretrained_checkpoint/'
    
    setup_seed(args.seed) 
    num_leads = 12
    num_class = args.num_class

    if args.task in ['kd_finetune', 'only_finetune']:
        file_name_pretrain = 'CODE15_Pretrain_' + model_config + '_checkpoint.pt'
        if model_config == 'large':
            num_layers, complexity = 42, 768
        elif model_config == 'light':
            num_layers, complexity = 30, 128
        else:
            num_layers, complexity = 30, 128
        r = get_rank_list(num_layers, args.conv_r, args.trans_r)

        if args.ranklist == 'lora_ave': ## equals LoRA
            net = LSTransECG(nOUT=num_class, out_channels=complexity, in_channels=num_leads, 
                                input_length=input_length, 
                                num_layers=num_layers, rank_list=r).to(device)
            mark_only_lora_as_trainable(net)
        else: ## equals Full fine-tuning
            net = LSTransECG(nOUT=num_class, out_channels=complexity, in_channels=num_leads, 
                                input_length=input_length, 
                                num_layers=num_layers, rank_list=0).to(device)
    elif args.task == 'Cross_kd':
        file_name_pretrain = 'CODE_testmediumbias_full_checkpoint.pkl'#CODE_testmediumbias_full_checkpoint.pkl
        num_layers, complexity = 47, 512
        r = get_rank_list(num_layers, args.conv_r, args.trans_r)

        if args.ranklist == 'lora_ave': ## equals LoRA
            net = NN_default(nOUT=num_class, complexity=complexity, inputchannel=num_leads, 
                                num_layers=num_layers, rank_list=r).to(device)
            mark_only_lora_as_trainable(net)
        else: ## equals Full fine-tuning
            net = NN_default(nOUT=num_class, complexity=complexity, inputchannel=num_leads,  
                                num_layers=num_layers, rank_list=0).to(device)
            
    net = load_pretrained_model(net, path + file_name_pretrain, args)

    if 'lora' in args.ranklist:
        params_to_update = []
        for name, param in net.named_parameters():
            if name.find('lora') > -1:
                params_to_update.append(param)
            elif name.find('bias') > -1:
                params_to_update.append(param)
            elif name.find('classifier.1.weight') > -1:
                params_to_update.append(param)
        optimizer = optim.AdamW(params_to_update, lr=args.learning_rate)
    else:
        optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate)
    
    return net, optimizer

## Same Config Finetuning
def kd_teacher_model(args):
    device = args.device
    model_config = args.model_config + '_kd_ft_teacher'
    path = args.root + '/pretrained_checkpoint/'
    batch_size = args.batch_size
    if batch_size > 64:
        args.learning_rate = 0.002
    else:
        args.learning_rate = 0.001
    setup_seed(args.seed)
    checkpoint_name = args.ft_dataset + '_' + args.tea_ranklist + '_' + model_config  + '_seed' + str(args.seed)
    
    ## dataset loading
    dataset_train, dataset_valid, dataset_test = ECGfinetunedataset_loading(args=args)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    label_iter = iter(loader_train)
    iteration = len(loader_train) * args.ft_epoch
    if args.ft_dataset == 'WFDB_Ga' or args.ft_dataset == 'WFDB_ChapmanShaoxing':
        iteration = iteration * 2

    start_time = time.time()
    sample_data, _ = dataset_train[0]
    actual_input_length = sample_data.shape[-1]  
    net, optimizer = model_config_initializations(args, input_length=actual_input_length)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, 
                                   dataset_name=checkpoint_name, 
                                   delta=0, args=args)
    my_lr_scheduler = get_linear_schedule_with_warmup(optimizer,int(iteration*0.01), iteration, last_epoch=-1)
    # asl_loss = AsymmetricLoss(gamma_neg=4, gamma_pos=0).to(device)

    net.train()
    total_loss = 0.0
    best_threshold = 0.5 * np.ones(args.num_class)
    pbar = tqdm(range(iteration), desc=f"KD Teacher Finetuning")
    for step in pbar:
        try:
            images, labels = next(label_iter)
        except StopIteration:
            label_iter = iter(loader_train)
            images, labels = next(label_iter)

        images = images.float().to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        images, labels = Cutmix_ECG(images, labels, device)
        
        optimizer.zero_grad()
        outputs = net(images)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        # loss = asl_loss(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        my_lr_scheduler.step()

        current_loss = loss.item()
        total_loss += current_loss
        avg_loss = total_loss / (step + 1)
        pbar.set_postfix({'loss': f'{current_loss:.4f}', 'avg_loss': f'{avg_loss:.4f}'})

        if (step + 1) % len(loader_train) == 0:
            net.eval() 
            valid_result = validate(net, loader_valid, threshold = 0.5 * np.ones(args.num_class), device=device)
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
    net = loading_lora_checkpoint(net, path + checkpoint_name + '_checkpoint.pt', args)
    trainable_num = count_parameters(net)
    
    net.eval()
    net.merge_net()
    test_result = validate(net, loader_test, device, iftest=True, threshold=best_threshold)
    test_result.update({
        'trainable_num': trainable_num,
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
def kd_student_model(args, fold_idx):
    teacher_checkpoint_name = args.ft_dataset + '_' + args.ranklist + '_' + args.model_config + '_kd_ft_teacher' + '_seed' + str(args.seed)
    teacher_checkpoint_path = os.path.join(args.root, 'pretrained_checkpoint', teacher_checkpoint_name + '_checkpoint.pt')
    dataset_train, dataset_valid, dataset_test = ECGfinetunedataset_loading(args=args, fold_idx=fold_idx)
    actual_input_length = dataset_train[0][0].shape[-1]
    if os.path.exists(teacher_checkpoint_path):
        print(f"--- Teacher checkpoint found at {teacher_checkpoint_path}. Loading existing model. ---")
        teacher_net, _ = model_config_initializations(args, input_length=actual_input_length)
        teacher_net = loading_lora_checkpoint(teacher_net, teacher_checkpoint_path, args)
        teacher_net.eval()
        if hasattr(teacher_net, 'merge_net'):
            teacher_net.merge_net()
    else:
        print("--- Teacher checkpoint not found. Starting teacher training... ---")
        teacher_net = kd_teacher_model(args)
    
    teacher_net.eval() # 教师模型固定，只用于推理
    for param in teacher_net.parameters():
        param.requires_grad = False
    args.ranklist = 'FT'
    device = args.device
    path = args.root + '/pretrained_checkpoint/'
    batch_size = args.batch_size
    args.learning_rate = 0.002
    T = args.kd_temperature      
    alpha = args.kd_alpha   
    # beta = args.kd_beta   
    setup_seed(args.seed)
    model_config = '_student_KD_FineTune_' + args.model_config + '_' + args.tea_ranklist
    checkpoint_name = args.ft_dataset + model_config  + '_seed' + str(args.seed)
    prestu_checkpoint = "CODE15_Pretrain_student"
    prestu_path = os.path.join(path, prestu_checkpoint + '_checkpoint.pt')
    ## dataset loading
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    
    label_iter = iter(loader_train)
    iteration = len(loader_train) * args.ft_epoch
    if args.ft_dataset in ['WFDB_Ga', 'WFDB_ChapmanShaoxing']:
        iteration = iteration * 2

    start_time = time.time() 
    num_layers, complexity = 9, 64 
    net = LSTransECG(nOUT=args.num_class, out_channels=complexity, in_channels=12, 
                             input_length=actual_input_length, 
                             num_layers=num_layers, rank_list=0,
                             use_static_conv=args.static_conv).to(device)
    net = load_pretrained_model(net, prestu_path, args)
    # ## 中间层特征蒸馏
    # # 中间层特征提取
    # teacher_feats = {}
    # student_feats = {}
    # def get_teacher_feat(name):
    #     def hook(model, input, output):
    #         teacher_feats[name] = input[0].detach()
    #     return hook
    # def get_student_feat(name):
    #     def hook(model, input, output):
    #         student_feats[name] = input[0]
    #     return hook
    # teacher_net.classifier[0].register_forward_hook(get_teacher_feat('pre_cls'))
    # net.classifier[0].register_forward_hook(get_student_feat('pre_cls'))
    # # 投影层对齐特征维度
    # if 'large' in args.model_config:
    #     teacher_dim = 768
    # else:
    #     teacher_dim = 128
    # student_dim = complexity
    # projector = nn.Linear(student_dim, teacher_dim).to(device)
    # net.add_module('kd_projector', projector)
    optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate)
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, 
                                   dataset_name=checkpoint_name, delta=0, args=args)
    my_lr_scheduler = get_linear_schedule_with_warmup(optimizer, int(iteration*0.01), iteration, last_epoch=-1)
    # asl_loss = AsymmetricLoss(gamma_neg=4, gamma_pos=0).to(device)

    net.train()
    # projector.train()
    best_threshold = 0.5 * np.ones(args.num_class)
    pbar = tqdm(range(iteration), desc=f"KD Student Training")
    for step in pbar:
        try:
            images, labels = next(label_iter)
        except StopIteration:
            label_iter = iter(loader_train)
            images, labels = next(label_iter)

        images = images.float().to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        images, labels = Cutmix_ECG_student(images, labels, device)
        with torch.no_grad():
            teacher_outputs = teacher_net(images)
        if hasattr(args, 'leads_for_student'):
            images = mask_ecg_signal(images, args.leads_for_student)
        optimizer.zero_grad()
        outputs = net(images)

        # 硬标签损失
        loss_hard = F.binary_cross_entropy_with_logits(outputs, labels)
        # loss_hard = asl_loss(outputs, labels)
        # 带温度系数的软标签蒸馏损失
        student_logits_T = outputs / T
        teacher_logits_T = teacher_outputs / T
        loss_soft = F.binary_cross_entropy_with_logits(student_logits_T, teacher_logits_T.sigmoid()) * (T * T)
        # #  中间层特征 MSE 投影对齐损失
        # s_feat = student_feats['pre_cls']
        # t_feat = teacher_feats['pre_cls']
        # if len(s_feat.shape) == 3:
        #     s_feat = s_feat.mean(dim=-1)
        #     t_feat = t_feat.mean(dim=-1)
        # s_feat_projected = projector(s_feat)
        # s_feat = F.normalize(s_feat_projected, p=2, dim=1)
        # t_feat = F.normalize(t_feat, p=2, dim=1)
        # loss_feat = F.mse_loss(s_feat, t_feat)
        loss = (1.0 - alpha) * loss_hard + alpha * loss_soft #+ beta * loss_feat
        # loss = loss_hard + loss_soft
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(list(net.parameters()) + list(projector.parameters()), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(list(net.parameters()), max_norm=1.0)
        optimizer.step()
        my_lr_scheduler.step()
        current_loss = loss.item()
        # pbar.set_postfix({
        #     'L_all': f'{current_loss:.3f}', 
        #     'L_hard': f'{loss_hard.item():.3f}',
        #     'L_soft': f'{loss_soft.item():.3f}',
        #     'L_feat': f'{loss_feat.item():.4f}'
        # })
        pbar.set_postfix({
            'L_all': f'{current_loss:.3f}', 
            'L_hard': f'{loss_hard.item():.3f}',
            'L_soft': f'{loss_soft.item():.3f}'
        })
        # student_feats.clear()
        # teacher_feats.clear()

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
    actual_steps = step + 1
    running_time = (end_time - start_time) / actual_steps
    allocated_memory = torch.cuda.max_memory_allocated(device)  # max_
    net.load_state_dict(torch.load(path + checkpoint_name + '_checkpoint.pt', map_location=device))
    trainable_num = count_parameters(net)

    net.eval()
    test_result = validate_student(net, loader_test, device, iftest=True, threshold=best_threshold, args=args)
    test_result.update({
        'trainable_num': trainable_num,
        'memory': allocated_memory,
        'time': running_time
    })
    
    return test_result

## NN for teacher & LSNet for student
def cross_kd_tea(args):
    device = args.device
    model_config = args.model_config + '_CrossKD_teacher'
    path = args.root + '/pretrained_checkpoint/'
    batch_size = args.batch_size
    if batch_size > 64:
        args.learning_rate = 0.002
    else:
        args.learning_rate = 0.001
    setup_seed(args.seed)
    checkpoint_name = args.ft_dataset + '_' + args.tea_ranklist + '_' + model_config  + '_seed' + str(args.seed)
    ## dataset loading
    dataset_train, dataset_valid, dataset_test = ECGfinetunedataset_loading(args=args)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    label_iter = iter(loader_train)
    iteration = len(loader_train) * args.ft_epoch
    if args.ft_dataset == 'WFDB_Ga' or args.ft_dataset == 'WFDB_ChapmanShaoxing':
        iteration = iteration * 2

    start_time = time.time()
    sample_data, _ = dataset_train[0]
    actual_input_length = sample_data.shape[-1]  
    net, optimizer = model_config_initializations(args, input_length=actual_input_length)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, 
                                   dataset_name=checkpoint_name, 
                                   delta=0, args=args)
    my_lr_scheduler = get_linear_schedule_with_warmup(optimizer,int(iteration*0.01), iteration, last_epoch=-1)
   
    net.train()
    total_loss = 0.0
    best_threshold = 0.5 * np.ones(args.num_class)
    pbar = tqdm(range(iteration), desc=f"KD Teacher Finetuning")
    for step in pbar:
        try:
            images, labels = next(label_iter)
        except StopIteration:
            label_iter = iter(loader_train)
            images, labels = next(label_iter)

        images = images.float().to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        if images.dim() == 3:
            images = images.unsqueeze(2)
        with torch.no_grad():
            images, labels = Cutmix(images, labels, device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        my_lr_scheduler.step()

        current_loss = loss.item()
        total_loss += current_loss
        avg_loss = total_loss / (step + 1)
        pbar.set_postfix({'loss': f'{current_loss:.4f}', 'avg_loss': f'{avg_loss:.4f}'})

        if (step + 1) % len(loader_train) == 0:
            net.eval() 
            valid_result = validate(net, loader_valid, threshold = 0.5 * np.ones(args.num_class), device=device)
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
    net = loading_lora_checkpoint(net, path + checkpoint_name + '_checkpoint.pt', args)
    trainable_num = count_parameters(net)
    
    net.eval()
    net.merge_net()
    test_result = validate(net, loader_test, device, iftest=True, threshold=best_threshold)
    test_result.update({
        'trainable_num': trainable_num,
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
def cross_kd_stu(args, fold_idx):
    teacher_checkpoint_name = args.ft_dataset + '_' + args.ranklist + '_' + args.model_config + '_CrossKD_teacher' + '_seed' + str(args.seed)
    teacher_checkpoint_path = os.path.join(args.root, 'pretrained_checkpoint', teacher_checkpoint_name + '_checkpoint.pt')
    dataset_train, dataset_valid, dataset_test = ECGfinetunedataset_loading(args=args, fold_idx = fold_idx)
    actual_input_length = dataset_train[0][0].shape[-1]
    if os.path.exists(teacher_checkpoint_path):
        print(f"--- Teacher checkpoint found at {teacher_checkpoint_path}. Loading existing model. ---")
        teacher_net, _ = model_config_initializations(args, input_length=actual_input_length)
        teacher_net = loading_lora_checkpoint(teacher_net, teacher_checkpoint_path, args)
        teacher_net.eval()
        if hasattr(teacher_net, 'merge_net'):
            teacher_net.merge_net()
    else:
        print("--- Teacher checkpoint not found. Starting teacher training... ---")
        teacher_net = cross_kd_tea(args)

    teacher_net.eval() # 教师模型固定，只用于推理
    for param in teacher_net.parameters():
        param.requires_grad = False
    args.ranklist = 'FT'
    device = args.device
    path = args.root + '/pretrained_checkpoint/'
    batch_size = args.batch_size
    args.learning_rate = 0.002
    T = args.kd_temperature      
    alpha = args.kd_alpha    
    setup_seed(args.seed)
    model_config = '_student_KD_Cross_' + args.model_config + '_' + args.tea_ranklist  + '_AblationLSconv'
    checkpoint_name = args.ft_dataset + model_config  + '_seed' + str(args.seed)
    prestu_checkpoint = "CODE15_Pretrain_student"
    prestu_path = os.path.join(path, prestu_checkpoint + '_checkpoint.pt')

    ## dataset loading
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    
    label_iter = iter(loader_train)
    iteration = len(loader_train) * args.ft_epoch
    if args.ft_dataset in ['WFDB_Ga', 'WFDB_ChapmanShaoxing']:
        iteration = iteration * 2

    start_time = time.time() 
    num_layers, complexity = 9, 64 
    net = LSTransECG(nOUT=args.num_class, out_channels=complexity, in_channels=12, 
                             input_length=actual_input_length, 
                             num_layers=num_layers, rank_list=0,
                             use_static_conv=args.static_conv).to(device)
    net = load_pretrained_model(net, prestu_path, args)
    optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate)
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, 
                                   dataset_name=checkpoint_name, delta=0, args=args)
    my_lr_scheduler = get_linear_schedule_with_warmup(optimizer, int(iteration*0.01), iteration, last_epoch=-1)

    net.train()
    best_threshold = 0.5 * np.ones(args.num_class)
    pbar = tqdm(range(iteration), desc=f"KD Student Training")
    for step in pbar:
        try:
            images, labels = next(label_iter)
        except StopIteration:
            label_iter = iter(loader_train)
            images, labels = next(label_iter)

        images = images.float().to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        images, labels = Cutmix_ECG_student(images, labels, device)
        with torch.no_grad():
            teacher_input = images.unsqueeze(2) if images.dim() == 3 else images
            teacher_outputs = teacher_net(teacher_input)
        if hasattr(args, 'leads_for_student'):
            images = mask_ecg_signal(images, args.leads_for_student)
        optimizer.zero_grad()
        outputs = net(images)

        # 硬标签损失
        loss_hard = F.binary_cross_entropy_with_logits(outputs, labels)
        # 带温度系数的软标签蒸馏损失
        student_logits_T = outputs / T
        teacher_logits_T = teacher_outputs / T
        loss_soft = F.binary_cross_entropy_with_logits(student_logits_T, teacher_logits_T.sigmoid()) * (T * T)
        loss = (1.0 - alpha) * loss_hard + alpha * loss_soft #+ beta * loss_feat
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(net.parameters()), max_norm=1.0)
        optimizer.step()
        my_lr_scheduler.step()
        current_loss = loss.item()

        pbar.set_postfix({
            'L_all': f'{current_loss:.3f}', 
            'L_hard': f'{loss_hard.item():.3f}',
            'L_soft': f'{loss_soft.item():.3f}'
        })

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
    actual_steps = step + 1
    running_time = (end_time - start_time) / actual_steps
    allocated_memory = torch.cuda.max_memory_allocated(device)  # max_
    net.load_state_dict(torch.load(path + checkpoint_name + '_checkpoint.pt', map_location=device))
    trainable_num = count_parameters(net)

    net.eval()
    test_result = validate_student(net, loader_test, device, iftest=True, threshold=best_threshold, args=args)
    test_result.update({
        'trainable_num': trainable_num,
        'memory': allocated_memory,
        'time': running_time
    })
    
    return test_result
