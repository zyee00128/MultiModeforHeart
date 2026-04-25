import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import trunc_normal_
from timm.layers import SqueezeExcite, DropPath

from .Lora_layer_default import ConvLoRA_split
from .ska_ecg import SKA

class SqueezeExcite1d(nn.Module):
    def __init__(self, in_chs, rd_ratio=0.25):
        super().__init__()
        rd_chs = int(in_chs * rd_ratio)
        self.fc1 = nn.Conv1d(in_chs, rd_chs, 1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(rd_chs, in_chs, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        # x shape: [Batch, Channel, Length]
        x_se = x.mean(2, keepdim=True) 
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)

class StaticConv1D(nn.Module):
    """
    LSConv1D 的静态卷积替代版本，用于消融实验。
    它使用标准的深度可分离卷积。
    """
    def __init__(self, dim, r=16, information='fisher'):
        super().__init__()
        self.conv = Conv1d_BN(dim, dim, kernel_size=3, padding=1, groups=dim, r=r, information=information)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        out = self.conv(x)
        return self.bn(out) + x

    def compute_grad_layer(self):
        if hasattr(self.conv.c, 'estimate_grad'):
             return [self.conv.c.estimate_grad()]
        return []

    def reset_rank_state(self):
        if hasattr(self.conv.c, 'enable_deactivation'):
            self.conv.c.enable_deactivation = not self.conv.c.enable_deactivation

    def merge_layer(self):
        if hasattr(self.conv.c, 'merge'):
            self.conv.c.merge()

    def freeze_A_grad_layer(self):
        if hasattr(self.conv.c, 'lora_A'):
            self.conv.c.lora_A.requires_grad = False

class LoRAConv1d(ConvLoRA_split):
    def __init__(self, *args, **kwargs):
        super(LoRAConv1d, self).__init__(nn.Conv1d, *args, **kwargs)

class Conv1d_BN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bn_weight_init=1, r=16, information='fisher'):
        super().__init__()
        self.add_module('c', LoRAConv1d(in_channels, out_channels, kernel_size, r=r, stride=stride, padding=padding, dilation=dilation, groups=groups, information=information, bias=False))
        self.add_module('bn', nn.BatchNorm1d(out_channels))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

class Residual(nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

class RepVGGDW1D(nn.Module):
    def __init__(self, ed, r=16, information='fisher'):
        super().__init__()
        self.conv = Conv1d_BN(ed, ed, 3, 1, 1, groups=ed, r=r, information=information)
        self.conv1 = Conv1d_BN(ed, ed, 1, 1, 0, groups=ed, r=r, information=information)
        self.dim = ed

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x
class Attention1D(nn.Module):
    def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4, resolution=0, r=16, information='fisher'):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        self.resolution = resolution

        h = self.dh + nh_kd * 2
        self.qkv = Conv1d_BN(dim, h, 1, r=r, information=information)
        self.proj = nn.Sequential(nn.ReLU(), 
                                  Conv1d_BN(self.dh, dim, 1, bn_weight_init=0, r=r, information=information))
        self.dw = Conv1d_BN(nh_kd, nh_kd, 3, 1, 1, groups=nh_kd, )

        if resolution > 0:
            self.attention_biases = nn.Parameter(torch.zeros(num_heads, 2 * resolution - 1))
            
            pos = torch.arange(resolution)
            relative_indices = pos[:, None] - pos[None, :] 
            relative_indices += resolution - 1
            self.register_buffer('attention_bias_idxs', relative_indices.long())

    def forward(self, x):
        # x shape: (B, C, L)
        B, C, L = x.shape
        qkv = self.qkv(x)
        # q: (B, nh_kd, L), k: (B, nh_kd, L), v: (B, self.dh, L)
        q, k, v = torch.split(qkv, [self.nh_kd, self.nh_kd, self.dh], dim=1)
        k = self.dw(k)
        # q, k: (B, num_heads, key_dim, L)
        # v: (B, num_heads, d, L)
        q = q.reshape(B, self.num_heads, self.key_dim, L)
        k = k.reshape(B, self.num_heads, self.key_dim, L)
        v = v.reshape(B, self.num_heads, self.d, L)
        # q: (B, nH, kd, L) -> (B, nH, L, kd)
        # k: (B, nH, kd, L)
        # attn: (B, nH, L, L) 
        attn = (q.transpose(-2, -1) @ k) * self.scale
        if hasattr(self, 'attention_biases'):
            bias = self.attention_biases[:, self.attention_bias_idxs]
            attn = attn + bias.unsqueeze(0) 
        attn = attn.softmax(dim=-1) # (B, nH, L, L)
        # v: (B, nH, d, L) -> (B, nH, L, d)
        # out: (B, nH, L, d) -> (B, self.dh, L)
        x = (attn @ v.transpose(-2, -1)).transpose(-2, -1).reshape(B, self.dh, L)
        x = self.proj(x)
        
        return x
    
    def compute_grad_layer(self):
        return [self.qkv.c.estimate_grad(), self.proj[1].c.estimate_grad()]
    
    def reset_rank_state(self):
        self.qkv.c.enable_deactivation = not self.qkv.c.enable_deactivation
        self.proj[1].c.enable_deactivation = not self.proj[1].c.enable_deactivation
    
    def merge_layer(self):
        self.qkv.c.merge()
        self.proj[1].c.merge()

    def freeze_A_grad_layer(self):
        self.qkv.c.lora_A.requires_grad = False
        self.proj[1].c.lora_A.requires_grad = False
class FFN1D(nn.Module):
    def __init__(self, ed, h, r=16, information='fisher'):
        super().__init__()
        self.pw1 = Conv1d_BN(ed, h, r=r, information=information)
        self.act = nn.ReLU()
        self.pw2 = Conv1d_BN(h, ed, bn_weight_init=0, r=r, information=information)

    def forward(self, x):
        return self.pw2(self.act(self.pw1(x)))
    
    def compute_grad_layer(self):
        return [self.pw1.c.estimate_grad(), self.pw2.c.estimate_grad()]

    def reset_rank_state(self):
        self.pw1.c.enable_deactivation = not self.pw1.c.enable_deactivation
        self.pw2.c.enable_deactivation = not self.pw2.c.enable_deactivation

    def merge_layer(self):
        self.pw1.c.merge()
        self.pw2.c.merge()

    def freeze_A_grad_layer(self):
        self.pw1.c.lora_A.requires_grad = False
        self.pw2.c.lora_A.requires_grad = False
class LKP(nn.Module):
    def __init__(self, dim, lks, sks, groups, r=16, information='fisher'):
        super().__init__()
        self.cv1 = Conv1d_BN(dim, dim // 2, r=r, information=information)
        self.cv2 = Conv1d_BN(dim // 2, dim // 2, lks, padding=(lks - 1) // 2, groups=dim // 2, r=0, information=information)
        self.cv3 = Conv1d_BN(dim // 2, dim // 2, r=r, information=information)
        self.cv4 = nn.Conv1d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)
        self.act = nn.ReLU()
        self.sks = sks
        self.groups = groups
        self.dim = dim
        
    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, c_out, l = w.size()
        w = w.view(b, self.dim // self.groups, self.sks ** 2, l)
        return w
    
    def compute_grad_layer(self):
        return [self.cv1.c.estimate_grad(), self.cv3.c.estimate_grad()]

    def reset_rank_state(self):
        self.cv1.c.enable_deactivation = not self.cv1.c.enable_deactivation
        self.cv3.c.enable_deactivation = not self.cv3.c.enable_deactivation

    def merge_layer(self):
        self.cv1.c.merge()
        self.cv3.c.merge()

    def freeze_A_grad_layer(self):
        self.cv1.c.lora_A.requires_grad = False
        self.cv3.c.lora_A.requires_grad = False
      
class LSConv1D(nn.Module):
    def __init__(self, dim, r=16, information='fisher'):
        super(LSConv1D,self).__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=8, r=r, information=information)
        self.ska = SKA()
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        # x shape: (B, C, L)
        #  w: (B, Groups, SKS^2, L)
        w = self.lkp(x)
        x_2d = x.unsqueeze(2) # (B, C, 1, L)
        out = self.ska(x, w)
        out = out.squeeze(2) # (B, C, L)
        return self.bn(out) + x
    
    def compute_grad_layer(self):
        grad_layers = []
        # grad_layers.extend(self.lkp.compute_grad_layer(), self.ska.compute_grad_layer())
        grad_layers.extend(self.lkp.compute_grad_layer())
        return grad_layers
    
    def reset_rank_state(self): 
        self.lkp.reset_rank_state()
        # self.ska.reset_rank_state()
    
    def merge_layer(self):
        self.lkp.merge_layer()
        # self.ska.merge_layer()

    def freeze_A_grad_layer(self):
        self.lkp.freeze_A_grad_layer()
        # self.ska.freeze_A_grad_layer()

class Block1D(nn.Module):    
    def __init__(self, ed, kd, nh=8, ar=4, resolution=0, stage=-1, depth=-1, 
                r=16, information='fisher', use_static_conv = False):
        super().__init__()
        if depth % 2 == 0:
            self.mixer = RepVGGDW1D(ed, r=r, information=information)
            self.se = SqueezeExcite1d(ed, 0.25)
        else:
            self.se = nn.Identity()
            if stage == 3:
                self.mixer = Residual(Attention1D(ed, kd, nh, ar, resolution=resolution, r=r, information=information))
            else:
                if use_static_conv:
                    self.mixer = StaticConv1D(ed, r=r, information=information)
                else:
                    self.mixer = LSConv1D(ed, r=r, information=information)

        self.ffn = Residual(FFN1D(ed, int(ed * 2), r=r, information=information))

    def forward(self, x):
        return self.ffn(self.se(self.mixer(x)))
    
    def compute_grad_layer(self):
        grad_layers = []
        if hasattr(self.mixer, 'compute_grad_layer'):
            grad_layers.extend(self.mixer.compute_grad_layer())
        if hasattr(self.ffn, 'compute_grad_layer'):
            grad_layers.extend(self.ffn.compute_grad_layer())
        return grad_layers
    
    def reset_rank_state(self):
        if hasattr(self.mixer, 'reset_rank_state'):
            self.mixer.reset_rank_state()
        if hasattr(self.ffn, 'reset_rank_state'):
            self.ffn.reset_rank_state()

    def merge_layer(self):
        if hasattr(self.mixer, 'merge_layer'):
            self.mixer.merge_layer()
        if hasattr(self.ffn, 'merge_layer'):
            self.ffn.merge_layer()

    def freeze_A_grad_layer(self):
        if hasattr(self.mixer, 'freeze_A_grad_layer'):
            self.mixer.freeze_A_grad_layer()
        if hasattr(self.ffn, 'freeze_A_grad_layer'):
            self.ffn.freeze_A_grad_layer()
    
class LSNet(nn.Module):
    def __init__(self, 
                 in_channels,  
                 input_length,             
                 embed_dim=[64, 128, 256, 384],
                 key_dim=[16, 16, 16, 16],
                 depth=[1, 2, 8, 10],
                 num_heads=[3, 3, 3, 4],
                 rank_list=[16, 32, 64, 128],
                 information='fisher',
                 use_static_conv = False):
        super().__init__()
        self.rank_list = rank_list
        # print(f"Rank list: {self.rank_list}")
        self.resolution = input_length // 8
        # print(f"Imput Length: {input_length}")
        # print(f"Imput Length: {self.resolution}")
        self.patch_embed = nn.Sequential(
            Conv1d_BN(in_channels, embed_dim[0] // 4, 3, 2, 1, r=0, information=information), nn.ReLU(),
            Conv1d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1, r=0, information=information), nn.ReLU(),
            Conv1d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1, r=0, information=information)
        )

        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        
        self.blocks1 = nn.Sequential()
        self.blocks2 = nn.Sequential()
        self.blocks3 = nn.Sequential()
        self.blocks4 = nn.Sequential()
        blocks = [self.blocks1, self.blocks2, self.blocks3, self.blocks4]
        
        for i, (ed, kd, dpth, nh, ar) in enumerate(zip(embed_dim, key_dim, depth, num_heads, attn_ratio)):
            if i > 0:
                self.resolution = self.resolution // 2
            for d in range(dpth):
                blocks[i].append(
                    Block1D(ed, kd, nh, ar, resolution=self.resolution, stage=i, depth=d, 
                            r=self.rank_list[i], information=information, use_static_conv=use_static_conv)
                )
            if i != len(depth) - 1:
                blk = blocks[i+1]
                blk.append(Conv1d_BN(embed_dim[i], embed_dim[i], 3, stride=2, padding=1, groups=embed_dim[i], r=0))
                blk.append(Conv1d_BN(embed_dim[i], embed_dim[i+1], 1, stride=1, padding=0, r=0))
        
        self.BN = nn.BatchNorm1d(embed_dim[-1])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)

        return self.BN(x)
    
    def compute_grad_layer(self):
        grads = []
        for block_group in [self.blocks1, self.blocks2, self.blocks3, self.blocks4]:
            for layer in block_group:
                if hasattr(layer, 'compute_grad_layer'):
                    grads.extend(layer.compute_grad_layer())
        return grads

    def reset_rank_state(self):
        for block_group in [self.blocks1, self.blocks2, self.blocks3, self.blocks4]:
            for layer in block_group:
                if hasattr(layer, 'reset_rank_state'):
                    layer.reset_rank_state()

    def merge_layer(self):
        for block_group in [self.blocks1, self.blocks2, self.blocks3, self.blocks4]:
            for layer in block_group:
                if hasattr(layer, 'merge_layer'):
                    layer.merge_layer()

    def freeze_A_grad_layer(self):
        for block_group in [self.blocks1, self.blocks2, self.blocks3, self.blocks4]:
            for layer in block_group:
                if hasattr(layer, 'freeze_A_grad_layer'):
                    layer.freeze_A_grad_layer()