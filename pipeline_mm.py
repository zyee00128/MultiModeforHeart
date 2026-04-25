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

from model.model_code_default import LSTrans_default
from pytorchtools import EarlyStopping
from evaluation import print_result, find_thresholds
# from datacollection import MultimodalDataset_loading 

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

# ==========================================
# 1. 多模态核心模型定义及对齐层 (Alignment)
class AlignmentLayer(nn.Module):
    def __init__(self, ecg_dim, pcg_dim):
        super(AlignmentLayer, self).__init__()
        # TODO: 在此实现你的多模态对齐机制（例如 Cross-Attention, 对比学习投影头等）
        pass
        
    def forward(self, ecg_feat, pcg_feat):
        # TODO: 填入特征对齐代码，此处先原样返回空出位置
        return ecg_feat, pcg_feat
class MultimodalStudentNet(nn.Module):
    def __init__(self, num_class, ecg_complexity=64, pcg_complexity=64, input_length=4096):
        super(MultimodalStudentNet, self).__init__()
        
        # 1. 独立提取 ECG 和 PCG 特征的 Backbone
        self.ecg_encoder = LSTrans_default(nOUT=num_class, out_channels=ecg_complexity, 
                                           in_channels=12, input_length=input_length, 
                                           num_layers=9, num_encoder_layers=1, rank_list=0)
        
        # 假设 PCG 输入为单通道或其他通道数，此处设 in_channels=1 示例
        self.pcg_encoder = LSTrans_default(nOUT=num_class, out_channels=pcg_complexity, 
                                           in_channels=1, input_length=input_length, 
                                           num_layers=9, num_encoder_layers=1, rank_list=0)
        
        # 2. 对齐层 (已留空，等待实现)
        self.alignment = AlignmentLayer(ecg_complexity, pcg_complexity)
        
        # 3. 对齐后的联合分类器
        fusion_dim = ecg_complexity + pcg_complexity
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, num_class)
        )

    def forward(self, ecg_x, pcg_x):
        # 独立提取特征
        ecg_feat = self.ecg_encoder.feature_extraction(ecg_x)  # [Batch, ecg_complexity]
        pcg_feat = self.pcg_encoder.feature_extraction(pcg_x)  # [Batch, pcg_complexity]
        
        # 送入对齐层进行对齐
        ecg_feat_aligned, pcg_feat_aligned = self.alignment(ecg_feat, pcg_feat)
        
        # 融合与分类
        fused_feat = torch.cat([ecg_feat_aligned, pcg_feat_aligned], dim=1)
        out = self.classifier(fused_feat)
        
        # 如果需要特征蒸馏，可以将中间对齐后的特征一并返回
        return out, ecg_feat_aligned, pcg_feat_aligned

def validate_multimodal(model, valloader, device, threshold=0.5 * np.ones(5), iftest=False, iftrain=False):
    model.eval()
    losses, probs, lbls = [], [],[]
    for step, (ecg_t, pcg_t, lbl_t) in enumerate(valloader):
        ecg_t = ecg_t.float().to(device)
        pcg_t = pcg_t.float().to(device)
        lbl_t = lbl_t.int().to(device)
        
        with torch.no_grad():
            out, _, _ = model(ecg_t, pcg_t)
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
        
    neg_ratio = (len(probs) - np.sum(probs, axis=0)) / max(np.sum(probs, axis=0), 1)
    valid_result.update({'neg_ratio': neg_ratio})
    valid_result.update({'threshold': threshold})
    return valid_result

def multimodal_kd_student_model(args, fold_idx):
    device = args.device
    batch_size = args.batch_size
    args.learning_rate = getattr(args, 'learning_rate', 0.002)
    T = getattr(args, 'kd_temperature', 2.0)
    alpha = getattr(args, 'kd_alpha', 0.5)
    setup_seed(args.seed)
    
    checkpoint_name = f"{args.ft_dataset}_Student_Multimodal_KD_seed{args.seed}"
    path = os.path.join(args.root, 'pretrained_checkpoint')
    os.makedirs(path, exist_ok=True)
    
    # ---------------- 数据加载 ----------------
    # 请根据实际情况替换为你多模态数据集的 Dataloader
    # dataset_train, dataset_valid, dataset_test = MultimodalDataset_loading(args=args, fold_idx=fold_idx)
    # loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    # loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=4)
    # loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)

    actual_input_length = 4096 
    
    # ---------------- 模型初始化 ----------------
    # 1. 加载教师模型 (此处假定已经有一个预训练好的强大单模态或多模态教师模型)
    # 如果教师模型是单模态，也可以在此处分别实例化 ecg_teacher 和 pcg_teacher
    teacher_net = MultimodalStudentNet(num_class=args.num_class, input_length=actual_input_length).to(device)
    # teacher_net.load_state_dict(torch.load("path_to_teacher.pt"))
    teacher_net.eval()
    for param in teacher_net.parameters():
        param.requires_grad = False
    net = MultimodalStudentNet(num_class=args.num_class, ecg_complexity=64, pcg_complexity=64, input_length=actual_input_length).to(device)
    optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate)
    
    # ---------------- 训练配置 ----------------
    iteration = len(loader_train) * args.ft_epoch
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, dataset_name=checkpoint_name, delta=0, args=args)
    my_lr_scheduler = get_linear_schedule_with_warmup(optimizer, int(iteration * 0.01), iteration, last_epoch=-1)

    # ---------------- 训练循环 ----------------
    start_time = time.time()
    net.train()
    best_threshold = 0.5 * np.ones(args.num_class)
    pbar = tqdm(range(iteration), desc="Multimodal KD Student Training")
    
    label_iter = iter(loader_train)
    for step in pbar:
        try:
            ecg_data, pcg_data, labels = next(label_iter)
        except StopIteration:
            label_iter = iter(loader_train)
            ecg_data, pcg_data, labels = next(label_iter)

        ecg_data = ecg_data.float().to(device, non_blocking=True)
        pcg_data = pcg_data.float().to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        
        # 教师模型产生软标签
        with torch.no_grad():
            teacher_outputs, t_ecg_feat, t_pcg_feat = teacher_net(ecg_data, pcg_data)
            
        # 学生模型前向传播
        optimizer.zero_grad()
        outputs, s_ecg_feat, s_pcg_feat = net(ecg_data, pcg_data)

        loss_hard = F.binary_cross_entropy_with_logits(outputs, labels)
        student_logits_T = outputs / T
        teacher_logits_T = teacher_outputs / T
        loss_soft = F.binary_cross_entropy_with_logits(student_logits_T, teacher_logits_T.sigmoid()) * (T * T)

        # loss_feat_ecg = F.mse_loss(s_ecg_feat, t_ecg_feat)
        # loss_feat_pcg = F.mse_loss(s_pcg_feat, t_pcg_feat)
        
        loss = (1.0 - alpha) * loss_hard + alpha * loss_soft

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        my_lr_scheduler.step()

        pbar.set_postfix({
            'L_all': f'{loss.item():.3f}', 
            'L_hard': f'{loss_hard.item():.3f}',
            'L_soft': f'{loss_soft.item():.3f}'
        })

        if (step + 1) % len(loader_train) == 0:
            net.eval()
            valid_result = validate_multimodal(net, loader_valid, device=device, threshold=0.5 * np.ones(args.num_class))

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
    allocated_memory = torch.cuda.max_memory_allocated(device)

    best_model_path = os.path.join(path, checkpoint_name + '_checkpoint.pt')
    net.load_state_dict(torch.load(best_model_path, map_location=device))
    trainable_num = count_parameters(net)

    net.eval()
    test_result = validate_multimodal(net, loader_test, device, iftest=True, threshold=best_threshold)
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
        
    return test_result