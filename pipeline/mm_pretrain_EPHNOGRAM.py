# -*- coding: utf-8 -*-
"""
EPHNOGRAM 多模态对比学习与生理对齐联合预训练 Pipeline
"""
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
                                      Cutmix_Multimodal)
from tools.pytorchtools import EarlyStopping
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

def get_downsample_factor(model):
    if hasattr(model, 'ecg_encoder'):
        encoder = model.ecg_encoder
        if hasattr(encoder, 'num_encoder_layers'):
            return 4 ** encoder.num_encoder_layers
        elif hasattr(encoder, 'backbone') and hasattr(encoder.backbone, 'num_encoder_layers'):
            return 4 ** encoder.backbone.num_encoder_layers
    return 4

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def load_unimodal_pretrained_backbones(net, ecg_path, pcg_path, args):
    """
    权重手术：从 12 导联单模态预训练模型中切片提取 II 导联参数，并注入到 1 导联多模态骨干网络中。
    """
    model_dict = net.state_dict()
    
    # 1. 针对 ECG 权重执行切片手术
    if ecg_path and os.path.exists(ecg_path):
        print(f"--- Loading ECG Pretrained Backbone with Weight Surgery from {ecg_path} ---")
        ecg_dict = torch.load(ecg_path, map_location=args.device)
        new_ecg_dict = {}
        for k, v in ecg_dict.items():
            # 识别第一层通道输入层并将其从 12 降至 1 (仅保留索引 1 的 Lead II)
            if any(x in k for x in ['conv1', 'patch_embed.proj', 'conv_input', 'backbone.conv1']):
                if v.dim() == 3 and v.shape[1] == 12:   # 1D Conv [C_out, 12, K]
                    v = v[:, 1:2, :]
                elif v.dim() == 4 and v.shape[1] == 12: # 2D Conv [C_out, 12, H, W]
                    v = v[:, 1:2, :, :]
            
            # 映射到多模态 ECG 分支命名空间
            if f"ecg_encoder.{k}" in model_dict:
                new_ecg_dict[f"ecg_encoder.{k}"] = v
            elif f"ecg_encoder.backbone.{k}" in model_dict:
                new_ecg_dict[f"ecg_encoder.backbone.{k}"] = v
            elif k in model_dict:
                new_ecg_dict[k] = v
        model_dict.update(new_ecg_dict)
        
    # 2. 载入单通道 PCG 预训练权重
    if pcg_path and os.path.exists(pcg_path):
        print(f"--- Loading PCG Pretrained Backbone from {pcg_path} ---")
        pcg_dict = torch.load(pcg_path, map_location=args.device)
        new_pcg_dict = {}
        for k, v in pcg_dict.items():
            if f"pcg_encoder.{k}" in model_dict:
                new_pcg_dict[f"pcg_encoder.{k}"] = v
            elif f"pcg_encoder.backbone.{k}" in model_dict:
                new_pcg_dict[f"pcg_encoder.backbone.{k}"] = v
            elif k in model_dict:
                new_pcg_dict[k] = v
        model_dict.update(new_pcg_dict)
        
    net.load_state_dict(model_dict, strict=False)
    return net

def compute_contrastive_loss(ecg_feat, pcg_feat, temp=0.07):
    """
    计算 ECG 和 PCG 特征之间的 InfoNCE 双向对比损失，并评估匹配准确率
    """
    # 如果是 3D 序列特征，执行时序平均池化 [B, L, D] -> [B, D]
    if ecg_feat.dim() == 3:
        ecg_feat = ecg_feat.mean(dim=1)
    if pcg_feat.dim() == 3:
        pcg_feat = pcg_feat.mean(dim=1)
        
    # L2 规范化
    ecg_feat = F.normalize(ecg_feat, p=2, dim=-1)
    pcg_feat = F.normalize(pcg_feat, p=2, dim=-1)
    
    # 相似度矩阵
    sim_matrix = torch.matmul(ecg_feat, pcg_feat.T) / temp # [B, B]
    labels = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)
    
    loss_ecg = F.cross_entropy(sim_matrix, labels)
    loss_pcg = F.cross_entropy(sim_matrix.T, labels)
    
    with torch.no_grad():
        preds = torch.argmax(sim_matrix, dim=-1)
        acc = (preds == labels).float().mean()
        
    return (loss_ecg + loss_pcg) / 2, acc

def validate_pretrain(model, valloader, device, args=None, temp=0.07):
    """
    无监督联合预训练验证函数，返回验证集对比损失与对比匹配准确率
    """
    model.eval()
    losses, accs = [], []
    prior_tool = PhysioPriorTool(sampling_rate=1000, device=device) if args is not None else None
    mode = getattr(args, 'alignment_mode', 'none')
    lambda_physio = getattr(args, 'lambda_physio', 0.1)

    for step, batch_data in enumerate(valloader):
        ecg_data, pcg_data, pcg_loc, _ = batch_data
        ecg_data = ecg_data.float().to(device)
        pcg_data = pcg_data.float().to(device)
        pcg_loc = pcg_loc.float().to(device)

        prior_mask, expert_feat = None, None

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
                    batch_size=ecg_data.shape[0], ecg_len=ecg_seq_len, pcg_len=pcg_seq_len, r_peaks=r_peaks_downsampled
                )
            elif mode == 'dual_stream':
                expert_feat = prior_tool.get_medical_expert_features(r_peaks, s1_peaks)

        with torch.no_grad():
            _, ecg_feat, pcg_feat, _, attn_weights = model(
                ecg_data, pcg_data, pcg_loc, prior_mask=prior_mask, expert_feat=expert_feat
            )
            
            loss_contrast, acc = compute_contrastive_loss(ecg_feat, pcg_feat, temp=temp)
            
            physio_loss = torch.tensor(0.0, device=device)
            if mode == 'constraint' and attn_weights is not None:
                ds = get_downsample_factor(model)
                ecg_seq_len = ecg_data.shape[-1] // ds
                pcg_seq_len = pcg_data.shape[-1] // ds
                r_peaks_downsampled = [(p / ds).int() for p in r_peaks]
                target_map = prior_tool.get_physio_constraint_map(
                    batch_size=ecg_data.shape[0], ecg_len=ecg_seq_len, pcg_len=pcg_seq_len, r_peaks=r_peaks_downsampled
                )
                physio_loss = F.kl_div((attn_weights + 1e-9).log(), target_map, reduction='batchmean')
                
            total_val_loss = loss_contrast + lambda_physio * physio_loss
            losses.append(total_val_loss.item())
            accs.append(acc.item())

    return np.mean(losses), np.mean(accs)

def pretrain_multimodal_model_config(args, input_length=4096, is_hetero=True):
    """
    初始化预训练时的单通道联合多模态网络结构，支持 Hetero 教师与 Homo 骨干。
    """
    device = args.device
    num_leads = 1 # 100% 真实单通道心电输入
    num_class = args.num_class
    expert_dim = 3 if args.alignment_mode == 'dual_stream' else 0

    if is_hetero:
        num_layers, ecg_complexity, pcg_complexity = 47, 512, 512
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
        net = MultimodalLSTransNet(
            nOUT=num_class, 
            ecg_complexity=ecg_complexity, pcg_complexity=pcg_complexity,
            ecg_inchannels=num_leads, pcg_inchannels=1, 
            input_length=input_length,
            num_layers=num_layers, num_encoder_layers=1, 
            rank_list=0, loc_dim=5,
            expert_dim=expert_dim
        ).to(device)
    
    optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    return net, optimizer
def run_multimodal_joint_pretraining(args, ecg_pretrained_path, pcg_pretrained_path, is_hetero=True):
    """
    EPHNOGRAM 联合多模态预训练流程入口
    """
    device = args.device
    setup_seed(args.seed)
    
    prefix = "Hetero" if is_hetero else "Homo"
    mode = args.alignment_mode
    model_config = f"{args.model_config}_EPHNOGRAM_JointPretrain_{prefix}_{mode}"
    checkpoint_name = f"{model_config}_seed{args.seed}"
    
    # 强制将预训练目标数据集配置为 ephnogram
    args.ft_dataset = 'ephnogram'
    batch_size = args.batch_size
    lambda_physio = getattr(args, 'lambda_physio', 0.1)
    temp = getattr(args, 'contrastive_temp', 0.07)

    # 载入预处理好的 EPHNOGRAM HDF5 数据
    dataset_train, dataset_valid, _ = MultimodalDataset_loading(args=args)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=0)
    
    label_iter = iter(loader_train)
    iteration = len(loader_train) * args.ft_epoch
    sample_data, _ = dataset_train[0]
    actual_input_length = sample_data.shape[-1] 
    total_loss = 0.0

    # 构造多模态网络并实施第一层权重手术
    net, optimizer = pretrain_multimodal_model_config(args, input_length=actual_input_length, is_hetero=is_hetero)
    net = load_unimodal_pretrained_backbones(net, ecg_pretrained_path, pcg_pretrained_path, args)
    
    # EarlyStopping 此时监控验证集的对比对齐 Loss
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, dataset_name=checkpoint_name, delta=0, args=args)
    my_lr_scheduler = get_linear_schedule_with_warmup(optimizer, int(iteration*0.01), iteration, last_epoch=-1)
    prior_tool = PhysioPriorTool(sampling_rate=1000, device=device)

    net.train()
    start_time = time.time()
    pbar = tqdm(range(iteration), desc=f"【EPHNOGRAM {prefix} Joint Pretraining】")
    
    for step in pbar:
        try:
            ecg_data, pcg_data, pcg_loc, _ = next(label_iter)
        except StopIteration:
            label_iter = iter(loader_train)
            ecg_data, pcg_data, pcg_loc, _ = next(label_iter)

        ecg_data = ecg_data.float().to(device)
        pcg_data = pcg_data.float().to(device)
        pcg_loc = pcg_loc.float().to(device)

        # 1. 级联粗对齐 (在对比特征提取前)
        if mode == 'cascaded':
            with torch.no_grad():
                r_peaks_raw = PhysioDetector.detect_ecg_r_peaks(ecg_data[:, 0, :], fs=1000)
            ecg_data, pcg_data = prior_tool.get_cascaded_aligned_signals(ecg_data, pcg_data, r_peaks_raw)
        
        # 预训练对比学习不建议执行破坏跨模态单体一致性的物理 Cutmix

        prior_mask, expert_feat = None, None
        physio_loss = 0.0

        # 2. 锚点对齐与双流特征注入
        if mode in ['constraint', 'anchor', 'dual_stream']:
            with torch.no_grad():
                r_peaks = PhysioDetector.detect_ecg_r_peaks(ecg_data[:, 0, :], fs=1000)
                s1_peaks = PhysioDetector.detect_pcg_s1_peaks(pcg_data[:, 0, :], r_peaks, fs=1000)
            
            ds = get_downsample_factor(net)
            ecg_seq_len = ecg_data.shape[-1] // ds
            pcg_seq_len = pcg_data.shape[-1] // ds

            if mode == 'anchor':
                r_peaks_downsampled = [(p / ds).int() for p in r_peaks]
                prior_mask = prior_tool.get_anchor_mask(
                    batch_size=ecg_data.shape[0], ecg_len=ecg_seq_len, pcg_len=pcg_seq_len, r_peaks=r_peaks_downsampled
                )
            elif mode == 'dual_stream':
                expert_feat = prior_tool.get_medical_expert_features(r_peaks, s1_peaks)

        optimizer.zero_grad()
        _, ecg_feat, pcg_feat, _, attn_weights = net(
            ecg_data, pcg_data, pcg_loc, prior_mask=prior_mask, expert_feat=expert_feat
        )
        
        # 3. 对比对齐损耗
        loss_contrast, train_acc = compute_contrastive_loss(ecg_feat, pcg_feat, temp=temp)
        
        # 4. 生理注意力机制约束损耗
        if mode == 'constraint' and attn_weights is not None:
            r_peaks_downsampled = [(p / ds).int() for p in r_peaks]
            target_map = prior_tool.get_physio_constraint_map(
                batch_size=ecg_data.shape[0], ecg_len=ecg_seq_len, pcg_len=pcg_seq_len, r_peaks=r_peaks_downsampled
            )
            physio_loss = F.kl_div((attn_weights + 1e-9).log(), target_map, reduction='batchmean')
            
        loss = loss_contrast + lambda_physio * physio_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        my_lr_scheduler.step()

        current_loss = loss.item()
        total_loss += current_loss
        pbar.set_postfix({'L': f'{current_loss:.3f}', 'L_ctr': f'{loss_contrast.item():.3f}', 'Acc': f'{train_acc:.2f}'})

        # 5. 周期验证与早停机制
        if (step + 1) % len(loader_train) == 0:
            val_loss, val_acc = validate_pretrain(net, loader_valid, device, args=args, temp=temp)
            tqdm.write(f"\n[Validation] Epoch Step {step+1}: Val Loss = {val_loss:.4f} | Val Contrastive Acc = {val_acc:.4f}")
            
            # 以验证集联合 Loss 监控早停
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                tqdm.write("\nEarly stopping triggered") 
                break
            net.train()

    pbar.close()
    print(f"--- 预训练完成。联合权重保存在: {early_stopping.save_path} ---")
    return net