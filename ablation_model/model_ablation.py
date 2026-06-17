# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:57:38 2023

@author: COCHE User
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.Lora_layer_default import *
from model.lsnet_se import LSNet
import copy

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2
def Cutmix(x, y, device, alpha=0.75):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(device)
    x = x.permute(0,2,1,3)
    y_a = y
    y_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    #print(bbx1, bby1, bbx2, bby2)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_mix = lam * y_a + (1 - lam) * y_b
    return  x.permute(0,2,1,3), y_mix
def Cutmix_student(x, y, device, alpha=0.75,valid_lead_num=12):
    if valid_lead_num == 1:
        x_mix = torch.zeros_like(x,device=device) + x.data
        activate_lead = [0]
        x = x[:, activate_lead, :, :]
    elif valid_lead_num == 3:
        x_mix = torch.zeros_like(x, device=device) + x.data
        activate_lead = [1,6,10]
        x = x[:, activate_lead, :, :]
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(device)
    x = x.permute(0,2,1,3)
    y_a = y
    y_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    #print(bbx1, bby1, bbx2, bby2)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_mix = lam * y_a + (1 - lam) * y_b
    if valid_lead_num < 12:
        x_mix[:, activate_lead, :, :] = x.permute(0,2,1,3)
    else:
        x_mix = x.permute(0,2,1,3)
    return  x_mix, y_mix
def rand_interval(L, lam):
    """
    针对 1D 信号生成随机时间段索引
    L: 信号长度
    lam: Beta 分布采样的 lambda 值，决定裁剪长度
    """
    cut_rat = np.sqrt(1. - lam) 
    cut_len = int(L * cut_rat)  
    cx = np.random.randint(L)
    bbx1 = np.clip(cx - cut_len // 2, 0, L)
    bbx2 = np.clip(cx + cut_len // 2, 0, L)

    return bbx1, bbx2
def Cutmix_ECG(x, y, device, alpha=0.75):
    """
    适配 1D ECG 信号的 Cutmix
    x shape: [Batch, Channel, Length] -> 3D Tensor
    """
    if alpha <= 0:
        return x, y

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(device)

    target_a = y
    target_b = y[rand_index]

    L = x.size()[2]
    bbx1, bbx2 = rand_interval(L, lam)
    x[:, :, bbx1:bbx2] = x[rand_index, :, bbx1:bbx2]

    actual_lam = 1. - ((bbx2 - bbx1) / L)
    y_mixed = target_a * actual_lam + target_b * (1. - actual_lam)

    return x, y_mixed
def Cutmix_ECG_student(x, y, device, alpha=0.75, valid_lead_num=12):
    """
    适配 1D ECG 信号的 Cutmix，针对学生模型进行裁剪
    x shape: [Batch, Channel, Length] -> 3D Tensor
    valid_lead_num: 学生模型使用的有效导联数量（1 或 3）
    """
    if alpha <= 0:
        return x, y

    if valid_lead_num == 1:
        x_mix = torch.zeros_like(x, device=device) + x.data
        activate_lead = [0]
        x = x[:, activate_lead, :]
    elif valid_lead_num == 3:
        x_mix = torch.zeros_like(x, device=device) + x.data
        activate_lead = [1,6,10]
        x = x[:, activate_lead, :]

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(device)
    target_a = y
    target_b = y[rand_index]
    L = x.size()[2]
    bbx1, bbx2 = rand_interval(L, lam)
    x[:, :, bbx1:bbx2] = x[rand_index, :, bbx1:bbx2]
    actual_lam = 1. - ((bbx2 - bbx1) / L)

    y_mixed = target_a * actual_lam + target_b * (1. - actual_lam)
    if valid_lead_num < 12:
        x_mix[:, activate_lead, :] = x
    else:
        x_mix = x

    return x_mix, y_mixed
def mask_ecg_signal(signal, valid_lead_num):
    if valid_lead_num == 1:
        mask_lead = np.arange(1,12)
        signal[:, mask_lead, :, :] = 0
        return signal
    elif valid_lead_num == 3:
        mask_lead = [0, 2, 3, 4, 5, 7, 8, 9, 11] #[1,6,10],II, V1, V5
        signal[:, mask_lead, :, :] = 0
        return signal
    else:
        return signal

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=200):
        super(PositionalEncoding, self).__init__()

        positional_encoding = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        x = x + self.positional_encoding[:x.size(1), :]
        return x
class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads,rank_list, dropout_coef, seq_length=96,information='fisher'):
        super(TransformerLayer, self).__init__()
        config = {'n_head':num_heads, 'r':rank_list[0], 'lora_attn_alpha':1,'lora_dropout':0.0} 
        self.self_attention = Attention(hidden_dim, seq_length, config, scale=True,information=information, dropout_coef = dropout_coef)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
#        self.dropout = nn.Dropout(dropout)
        self.fc1 = Linear(hidden_dim, 4 * hidden_dim,r=rank_list[1],information=information, dropout_coef = dropout_coef)
        self.fc2 = Linear(4 * hidden_dim, hidden_dim,r=rank_list[2],information=information, dropout_coef = dropout_coef)
    
    def feed_forward(self,x):
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def compute_grad_layer(self):
        grad_list=[self.self_attention.c_attn.estimate_grad(),self.fc1.estimate_grad(),self.fc2.estimate_grad()]
        return grad_list
    
    def freeze_A_grad_layer(self):
        self.self_attention.c_attn.lora_A.requires_grad = False
        self.fc1.lora_A.requires_grad = False
        self.fc2.lora_A.requires_grad = False
        return 
    
    def forward(self, x):
        x = x + self.self_attention(self.norm1(x))[0]
        x = x + self.feed_forward(self.norm2(x))
        return x

    def reset_rank_state(self):
        self.self_attention.c_attn.enable_deactivation = not self.self_attention.c_attn.enable_deactivation
        self.fc1.enable_deactivation = not self.fc1.enable_deactivation
        self.fc2.enable_deactivation = not self.fc2.enable_deactivation
        return

    def merge_layer(self):
        self.self_attention.c_attn.merge()
        self.fc1.merge()
        self.fc2.merge()
        return
class MyResidualBlock(nn.Module):
    def __init__(self,input_complexity,output_complexity,stride,downsample,rank_list,dropout_coef,information='fisher'):
        super(MyResidualBlock,self).__init__()
        self.downsample = downsample
        self.stride = stride if self.downsample else 1
        K = 9
        P = (K-1)//2
        self.conv1 = Conv2d(in_channels=input_complexity,
                               out_channels=output_complexity,
                               kernel_size=(1,K),
                               stride=(1,self.stride),
                               padding=(0,P),
                               bias=True,r=rank_list[0],information=information,dropout_coef=dropout_coef) #False
        self.bn1 = nn.BatchNorm2d(output_complexity)

        self.conv2 = Conv2d(in_channels=output_complexity,
                               out_channels=output_complexity,
                               kernel_size=(1,K),
                               padding=(0,P),
                               bias=True,r=rank_list[1],information=information,dropout_coef=dropout_coef) #False
        self.bn2 = nn.BatchNorm2d(output_complexity)

        if self.downsample:
            self.idfunc_0 = nn.AvgPool2d(kernel_size=(1,self.stride),stride=(1,self.stride))
            self.conv3 = Conv2d(in_channels=input_complexity,
                                      out_channels=output_complexity,
                                      kernel_size=(1,1),
                                      bias=True,r=rank_list[2],information=information,dropout_coef=dropout_coef) #False

    def forward(self, x):
        identity = x
        x = F.leaky_relu(self.bn1(self.conv1(x)))       
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        if self.downsample:
            identity = self.idfunc_0(identity)
            identity = self.conv3(identity)
        x = x+identity
        return x
    
    def compute_grad_layer(self):
        grad_list=[self.conv1.estimate_grad(),self.conv2.estimate_grad(),self.conv3.estimate_grad()]
        return grad_list
    
    def freeze_A_grad_layer(self):
        self.conv1.lora_A.requires_grad = False
        self.conv2.lora_A.requires_grad = False
        self.conv3.lora_A.requires_grad = False
        return 

    def reset_rank_state(self):
        self.conv1.enable_deactivation = not self.conv1.enable_deactivation
        self.conv2.enable_deactivation = not self.conv2.enable_deactivation
        self.conv3.enable_deactivation = not self.conv3.enable_deactivation
        return

    def merge_layer(self):
        self.conv1.merge()
        self.conv2.merge()
        self.conv3.merge()
        return

class StaticAggregationLSConv(nn.Module):
    """
    实验 A-2: Static Aggregation (LKP + Static-DWConv)
    绕过 LKP 动态权重预测，仅使用静态深度可分离卷积进行聚合，验证动态权重的自适应价值。
    """
    def __init__(self, original_lsconv):
        super().__init__()
        # 推导输入通道数
        if hasattr(original_lsconv, 'bn') and hasattr(original_lsconv.bn, 'num_features'):
            channels = original_lsconv.bn.num_features
        elif hasattr(original_lsconv, 'lkp') and hasattr(original_lsconv.lkp, 'cv1'):
            cv1 = original_lsconv.lkp.cv1
            channels = cv1.out_channels if hasattr(cv1, 'out_channels') else cv1.c.out_channels
        else:
            channels = 64
            
        self.static_dw_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        is_4d = x.dim() == 4
        if is_4d:
            x = x.squeeze(2)
        out = self.bn(self.static_dw_conv(x))
        if is_4d:
            out = out.unsqueeze(2)
        return out
class StaticPerceptionLSConv(nn.Module):
    """
    实验 A-3: Static Perception (Static-LargeConv + SKA)
    使用单层静态大核深度卷积提取上下文，替换 LKP 原本的多层 Bottleneck 投影设计，验证 LKP 结构的特征提纯增益。
    """
    def __init__(self, original_lsconv):
        super().__init__()
        if hasattr(original_lsconv, 'bn') and hasattr(original_lsconv.bn, 'num_features'):
            channels = original_lsconv.bn.num_features
        else:
            channels = 64
            
        if hasattr(original_lsconv, 'lkp') and hasattr(original_lsconv.lkp, 'cv4'):
            cv4 = original_lsconv.lkp.cv4
            out_channels = cv4.out_channels if hasattr(cv4, 'out_channels') else cv4.c.out_channels
        else:
            out_channels = 12
            
        # 静态大核 1D 深度卷积
        self.cv_large = nn.Conv1d(channels, channels, kernel_size=15, padding=7, groups=channels, bias=False)
        self.bn_large = nn.BatchNorm1d(channels)
        self.act = nn.LeakyReLU(0.2)
        self.cv_map = nn.Conv1d(channels, out_channels, kernel_size=1, bias=True)
        
        # 继承原模型的聚合机制 (SKA) 和归一化层
        self.ska = original_lsconv.ska
        self.bn = original_lsconv.bn
        
    def forward(self, x):
        is_4d = x.dim() == 4
        if is_4d:
            x = x.squeeze(2)
            
        feat = self.act(self.bn_large(self.cv_large(x)))
        weights = self.cv_map(feat)
        
        out = self.ska(x, weights)
        out = self.bn(out)
        
        if is_4d:
            out = out.unsqueeze(2)
        return out
class HomogeneousStackingLSConv(nn.Module):
    """
    实验 A-4: Homogeneous-Stacking (Stacked StdConv(3x3))
    用标准堆叠的普通 1D 卷积替代复杂的 LSConv 设计，验证多重交替机制和 LKP+SKA 组件的综合增益。
    """
    def __init__(self, original_lsconv):
        super().__init__()
        if hasattr(original_lsconv, 'bn') and hasattr(original_lsconv.bn, 'num_features'):
            channels = original_lsconv.bn.num_features
        else:
            channels = 64
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        is_4d = x.dim() == 4
        if is_4d:
            x = x.squeeze(2)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if is_4d:
            out = out.unsqueeze(2)
        return out
class StaticDWConvBlock(nn.Module):
    """
    实验 B-2: w/o RepVGG (LSConv + DWConv(3x3) + SE)
    使用单路标准 1D 深度卷积替换多分支重参数化 RepVGGDW，验证结构形态先验对局部形态波形特征提取的价值。
    """
    def __init__(self, original_repvgg):
        super().__init__()
        if hasattr(original_repvgg, 'channels'):
            channels = original_repvgg.channels
        elif hasattr(original_repvgg, 'conv3'):
            cv3 = original_repvgg.conv3
            channels = cv3.out_channels if hasattr(cv3, 'out_channels') else cv3.c.out_channels
        else:
            channels = 64
            
        self.conv_dw = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        is_4d = x.dim() == 4
        if is_4d:
            x = x.squeeze(2)
        out = self.bn(self.conv_dw(x)) + x
        if is_4d:
            out = out.unsqueeze(2)
        return out

def apply_ablation_to_lsnet(encoder, ablation_odd='none', ablation_even='none'):
    """
    递归遍历 LSNet 内部层，根据消融方案动态重构相应 Mixer 与 SE 组件。
    """
    last_lsconv = None
    
    # 第一次遍历：查找并暂存各阶段的奇数层 LSConv Mixer（用于 B-1 实验）
    for name, module in encoder.named_modules():
        if module.__class__.__name__ == 'LSConv':
            last_lsconv = module

    def replace_modules(parent):
        nonlocal last_lsconv
        for child_name, child in parent.named_children():
            class_name = child.__class__.__name__
            
            # 1. 匹配奇数层 Block (含有 LSConv mixer)
            if class_name == 'Block' and hasattr(child, 'mixer') and child.mixer.__class__.__name__ == 'LSConv':
                last_lsconv = child.mixer
                if ablation_odd == 'a2':
                    child.mixer = StaticAggregationLSConv(child.mixer)
                elif ablation_odd == 'a3':
                    child.mixer = StaticPerceptionLSConv(child.mixer)
                elif ablation_odd == 'a4':
                    child.mixer = HomogeneousStackingLSConv(child.mixer)
            
            # 2. 匹配偶数层 Block (含有 RepVGGDW mixer)
            elif class_name == 'Block' and hasattr(child, 'mixer') and (child.mixer.__class__.__name__ == 'RepVGGDW' or 'RepVGG' in child.mixer.__class__.__name__):
                if ablation_even == 'b1':
                    # B-1: 同构堆叠 (复制同一阶段的 LSConv 并替换，同时将 SE 去除)
                    if last_lsconv is not None:
                        child.mixer = copy.deepcopy(last_lsconv)
                    child.se = nn.Identity()
                elif ablation_even == 'b2':
                    # B-2: 移除 RepVGG 先验分支，替换为基础静态深度卷积
                    child.mixer = StaticDWConvBlock(child.mixer)
                elif ablation_even == 'b3':
                    # B-3: 移除 SE 导联自适应门控机制
                    child.se = nn.Identity()
            else:
                replace_modules(child)

    replace_modules(encoder)
    return encoder

class NN_default(nn.Module): ## backbone model definition
    def __init__(self,nOUT,complexity,inputchannel,input_length=1152,
                 num_layers=35,rank_list=32,information='fisher',
                 num_encoder_layers=3,dropout_coef=0.2,pos_max_len=200):
        super(NN_default,self).__init__()
        
        stride = 4
        #assert num_layers!=None
        self.input_length = input_length
        self.num_layers = num_layers
        self.num_encoder_layers=num_encoder_layers
        self.num_classifier_layers=2
        assert (num_layers-3*self.num_encoder_layers-self.num_classifier_layers) % 3 == 0
        if isinstance(rank_list, int):
            self.rank_list = (np.zeros(num_layers)+rank_list).astype(int)
        else:
            self.rank_list = rank_list
            
        self.encoder_layers = nn.ModuleList(
                [MyResidualBlock(inputchannel, complexity, stride, downsample=True, rank_list=self.rank_list[0:3],information=information,dropout_coef=dropout_coef)])
        self.encoder_layers += nn.ModuleList(
                [MyResidualBlock(complexity, complexity, stride, downsample=True, rank_list=self.rank_list[3*(i+1):3*(i+2)],information=information,dropout_coef=dropout_coef) for i in range(self.num_encoder_layers-1)])
        
        self.classifier = nn.ModuleList(
            [Linear(complexity,complexity, r=self.rank_list[num_layers-2-i],merge_weights=False,information=information,dropout_coef=dropout_coef) for i in range(self.num_classifier_layers-1)]
        )
        self.classifier += nn.ModuleList([Linear(complexity,nOUT, r=0,merge_weights=False,information=information,dropout_coef=dropout_coef)])
        
        seq_len = self.input_length // (4 ** self.num_encoder_layers)
        seq_len = max(seq_len, 96)
        self.num_transformer_layers=self.num_layers-(3*self.num_encoder_layers+self.num_classifier_layers)
        complexity=complexity
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(complexity, num_heads=16,seq_length=seq_len,rank_list=self.rank_list[3*(i+self.num_encoder_layers):3*(i+1+self.num_encoder_layers)],information=information,dropout_coef=dropout_coef) for i in range(self.num_transformer_layers // 3)]
        )
        
        self.position_encoding = PositionalEncoding(complexity,max_len=pos_max_len)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.Pretrained = False
        
    def compute_grad(self):
        grad_list=[]
        for layer in self.encoder_layers:
            grad_list.extend(layer.compute_grad_layer())
            
        for layer in self.transformer_layers:
            grad_list.extend(layer.compute_grad_layer())
            
        for layer in self.classifier:
            if layer.r > 0:
                grad_list.extend([layer.estimate_grad()])
        
        return grad_list

    def network_rank_state_reset(self):
        for layer in self.encoder_layers:
            layer.reset_rank_state()

        for layer in self.transformer_layers:
            layer.reset_rank_state()

    def get_network_rank_state_reset(self):
        return self.encoder_layers[0].conv1.enable_deactivation

    def merge_net(self):
        for layer in self.encoder_layers:
            layer.merge_layer()

        for layer in self.transformer_layers:
            layer.merge_layer()

        for layer in self.classifier:
            if layer.r > 0:
                layer.merge()
        return

    def freeze_A_grad(self):
        for layer in self.encoder_layers:
            layer.freeze_A_grad_layer()
            
        for layer in self.transformer_layers:
            layer.freeze_A_grad_layer()
            
        for layer in self.classifier:
            layer.lora_A.requires_grad = False
        return 
    
    def feature_extraction(self, x, semi_flag = False):
        if x.dim() == 3:
            x = x.unsqueeze(2)

        for layer in self.encoder_layers:
            x = layer(x)
        if semi_flag:
            #print(x.shape)
            x = x[0:len(x)//2,:,:,:]
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.position_encoding(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.permute(0, 2, 1)
        
        return x

    def forward(self, x, semi_flag = False):
        x = self.feature_extraction(x, semi_flag)
        out = self.pool(x).squeeze(2)
        for layer in self.classifier:
            out = layer(out)
        return out
class LSTrans_default(nn.Module):
    def __init__(self, nOUT, out_channels, in_channels, input_length,
                 num_layers, num_encoder_layers=1,
                 rank_list=32, information='fisher', 
                 dropout_coef=0.2, pos_max_len=200,
                 ablation_odd='none', ablation_even='none'):
        super(LSTrans_default,self).__init__()

        self.input_length = input_length
        self.num_layers = num_layers
        self.num_encoder_layers = num_encoder_layers
        self.num_classifier_layers = 2
        lsnet_rank_usage = 4 * self.num_encoder_layers
        assert (num_layers - lsnet_rank_usage - self.num_classifier_layers) % 3 == 0

        # rank list processing
        if isinstance(rank_list, int):
            self.rank_list = (np.zeros(num_layers) + rank_list).astype(int)
        else:
            self.rank_list = rank_list
        # print(f"Imput Length: {self.input_length}")
        # lsconv encoder layers
        self.encoder = LSNet(
            in_channels=in_channels, 
            input_length=self.input_length,
            embed_dim=[64, 128, 256, out_channels],  
            rank_list=self.rank_list[0:4],
            information=information
        )

        if ablation_odd != 'none' or ablation_even != 'none':
            self.encoder = apply_ablation_to_lsnet(self.encoder, ablation_odd, ablation_even)

        # transformer layers
        self.num_transformer_layers = self.num_layers - (lsnet_rank_usage + self.num_classifier_layers)
        seq_len = self.input_length // (4 ** self.num_encoder_layers)
        seq_len = max(seq_len, 96)
        self.position_encoding = PositionalEncoding(out_channels, max_len=pos_max_len)
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(out_channels, num_heads=16, seq_length=seq_len, 
                             rank_list=self.rank_list[(3*i+lsnet_rank_usage):(3*(i+1)+lsnet_rank_usage)], 
                             information=information, dropout_coef=dropout_coef) 
            for i in range(self.num_transformer_layers // 3)
        ])

        # # classifier layers
        # self.classifier = nn.ModuleList([
        #     Linear(out_channels, out_channels, r=self.rank_list[num_layers-2-i], merge_weights=False, information=information, dropout_coef=dropout_coef) 
        #     for i in range(self.num_classifier_layers - 1)
        # ])
        # self.classifier += nn.ModuleList([
        #     Linear(out_channels, nOUT, r=0, merge_weights=False, information=information, dropout_coef=dropout_coef)
        # ])

        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
      

    def compute_grad(self):
        grad_list = []
        if hasattr(self.encoder, 'compute_grad_layer'):
            grad_list.extend(self.encoder.compute_grad_layer())
        for layer in self.transformer_layers:
            grad_list.extend(layer.compute_grad_layer())
        return grad_list

    def network_rank_state_reset(self):
        if hasattr(self.encoder, 'reset_rank_state'):
            self.encoder.reset_rank_state()
        for layer in self.transformer_layers:
            layer.reset_rank_state()

    def merge_net(self):
        if hasattr(self.encoder, 'merge_layer'):
            self.encoder.merge_layer()
        for layer in self.transformer_layers:
            layer.merge_layer()

    def freeze_A_grad(self):
        if hasattr(self.encoder, 'freeze_A_grad_layer'):
            self.encoder.freeze_A_grad_layer()
        for layer in self.transformer_layers:
            layer.freeze_A_grad_layer()

    def feature_extraction(self, x):
        # x shape: [Batch, Channel, Length]
        x = self.encoder(x) # [Batch, out_channels, Seq_len]
            
        x = x.permute(0, 2, 1) # [Batch, Seq_len, out_channels]
        x = self.position_encoding(x) 
        for layer in self.transformer_layers:
            x = layer(x) 
        
        # x shape: [Batch, Seq_len, out_channels]
        x = x.permute(0, 2, 1) # [Batch, out_channels, Seq_len]
        return x

    # def forward(self, x):
    #     x = self.feature_extraction(x)
    #     out = self.pool(x).squeeze(2)
    #     for layer in self.classifier:
    #         out = layer(out)
    #     return out

class LSTransECG(nn.Module):
    def __init__(self, nOUT, out_channels, in_channels, input_length,
                 num_layers, num_encoder_layers=1,
                 rank_list=32, information='fisher',
                 dropout_coef=0.2,
                 pos_max_len=200,
                 backbone=LSTrans_default,
                 ablation_odd='none',
                 ablation_even='none'):
        super(LSTransECG,self).__init__()
        self.num_classifier_layers = 2
        # rank list processing
        if isinstance(rank_list, int):
            self.rank_list = (np.zeros(num_layers) + rank_list).astype(int)
        else:
            self.rank_list = rank_list
   
        self.backbone = backbone(nOUT, out_channels, in_channels,
                                 input_length,
                                 num_layers, num_encoder_layers,
                                 rank_list, information, 
                                 dropout_coef, pos_max_len,
                                 ablation_odd=ablation_odd,
                                 ablation_even=ablation_even)
        # classifier layers
        self.classifier = nn.ModuleList([
            Linear(out_channels, out_channels, r=self.rank_list[num_layers-2-i], 
                   merge_weights=False, information=information, dropout_coef=dropout_coef) 
            for i in range(self.num_classifier_layers - 1)
        ])
        self.classifier += nn.ModuleList([
            Linear(out_channels, nOUT, r=0, merge_weights=False, 
                   information=information, dropout_coef=dropout_coef)
        ])

        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
    def compute_grad(self):
        grad_list = []
        if hasattr(self.backbone, 'compute_grad'):
            grad_list.extend(self.backbone.compute_grad())
        for layer in self.classifier:
            if layer.r > 0:
                grad_list.extend([layer.estimate_grad()])
        return grad_list

    def merge_net(self):
        self.backbone.merge_net()
        for layer in self.classifier:
            if layer.r > 0:
                layer.merge()

    def freeze_A_grad(self):
        self.backbone.freeze_A_grad()
        for layer in self.classifier:
            if hasattr(layer, 'lora_A'):
                layer.lora_A.requires_grad = False

    def network_rank_state_reset(self):
        self.backbone.network_rank_state_reset()

    def feature_extraction(self, x):
        return self.backbone.feature_extraction(x) 
    def forward(self, x):
        x = self.feature_extraction(x)
        out = self.pool(x).squeeze(2)
        for layer in self.classifier:
            out = layer(out)
        return out
