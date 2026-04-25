import torch
import torch.nn as nn
import torch.nn.functional as F

class SKA(nn.Module):
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入张量 [B, C, L]
            w: 权重张量 [B, WC, KS, L] (WC = C // groups, KS 是卷积核大小)
        返回:
            形状为 [B, C, 1, L] 的张量 (为了匹配原代码中后续的 .squeeze(2) 操作)
        """
        B, C, L = x.shape
        _, WC, KS, _ = w.shape

        pad = (KS - 1) // 2

        x_padded = F.pad(x, (pad, pad)) # [B, C, L + 2*pad]

        x_windows = x_padded.unfold(2, KS, 1)
        if C > WC:
            groups = C // WC
            # w: [B, WC, KS, L] -> [B, C, KS, L]
            w = w.repeat_interleave(groups, dim=1)
        w = w.permute(0, 1, 3, 2)      
        out = (x_windows * w).sum(dim=-1)
        
        return out