import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import find_peaks, butter, lfilter, hilbert

class PhysioDetector:
    """
    针对 Tensor 批处理设计的生理信号峰值检测器
    """
    @staticmethod
    def _butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    @staticmethod
    def _filter_signal(data, lowcut, highcut, fs):
        b, a = PhysioDetector._butter_bandpass(lowcut, highcut, fs)
        return lfilter(b, a, data)

    @classmethod
    def detect_ecg_r_peaks(cls, ecg_tensor, fs=1000):
        """
        简易 Pan-Tompkins 逻辑：带通滤波 + 差分 + 平方 + 找峰
        ecg_tensor: [Batch, Length]
        """
        batch_peaks = []
        ecg_np = ecg_tensor.cpu().numpy()
        
        for sig in ecg_np:
            # 1. 滤波 (5-15Hz 是 R 波能量集中区)
            filtered = cls._filter_signal(sig, 5, 15, fs)
            # 2. 差分与平方
            diff = np.diff(filtered)
            squared = diff ** 2
            # 3. 找峰 (设置最小间距为 0.6s，即心率不超过 200bpm)
            peaks, _ = find_peaks(squared, distance=int(0.6 * fs), height=np.mean(squared))
            batch_peaks.append(torch.tensor(peaks))
            
        return batch_peaks

    @classmethod
    def detect_pcg_s1_peaks(cls, pcg_tensor, r_peaks, fs=1000):
        """
        基于 R 波位置寻找 S1：在 R 波后 20ms-150ms 窗口内寻找包络最大值
        pcg_tensor: [Batch, Length]
        """
        batch_peaks = []
        pcg_np = pcg_tensor.cpu().numpy()
        
        for i, sig in enumerate(pcg_np):
            # 1. 提取包络 (Hilbert 变换)
            analytic_signal = hilbert(sig)
            amplitude_envelope = np.abs(analytic_signal)
            
            s1_indices = []
            cur_r_peaks = r_peaks[i].numpy()
            
            for r_idx in cur_r_peaks:
                # 定义 S1 搜索窗口 (R波后 20ms 到 150ms)
                search_start = int(r_idx + 0.02 * fs)
                search_end = int(r_idx + 0.15 * fs)
                
                if search_start < len(amplitude_envelope):
                    window = amplitude_envelope[search_start : min(search_end, len(amplitude_envelope))]
                    if len(window) > 0:
                        s1_idx = np.argmax(window) + search_start
                        s1_indices.append(s1_idx)
            
            batch_peaks.append(torch.tensor(s1_indices))
            
        return batch_peaks
    
class PhysioPriorTool:
    """
    生理先验知识提取工具箱
    用于支撑：级联对齐、约束驱动、锚点引导、双流融合四种实验范式
    """
    def __init__(self, sampling_rate=1000, device='cpu'):
        self.fs = sampling_rate
        self.device = device

    # --- 1. 级联式对齐 (Cascaded Alignment) ---
    def get_cascaded_aligned_signals(self, ecg, pcg, r_peaks):
        """
        根据第一个R波位置，对PCG进行物理位移对齐
        r_peaks: List[List[int]] 或 Tensor, 形状为 [Batch, N_peaks]
        """
        batch_size, channels, length = pcg.shape
        aligned_pcg = torch.zeros_like(pcg)
        
        for i in range(batch_size):
            # 取第一个检测到的有效R波
            first_r = r_peaks[i][0] if len(r_peaks[i]) > 0 else 0
            
            # 生理先验：S1通常在R波后约 50ms (0.05s)
            expected_s1 = first_r + int(0.05 * self.fs)
            
            # 计算位移量：将信号移动到以第一个参考点为起始的位置
            shift = -int(expected_s1)
            aligned_pcg[i] = torch.roll(pcg[i], shifts=shift, dims=-1)
            
        return ecg, aligned_pcg

    # --- 2. 约束驱动目标图 (Constraint Target) ---
    def get_physio_constraint_map(self, batch_size, ecg_len, pcg_len, r_peaks):
        """
        生成理想的注意力分布图 (用于计算 KL 散度等约束 Loss)
        返回: [Batch, ecg_len, pcg_len]
        """
        target_map = torch.zeros(batch_size, ecg_len, pcg_len, device=self.device)
        
        # 生理窗口：S1-S2 通常出现在 R 波后的 20ms 到 500ms 内
        win_start = int(0.02 * self.fs)
        win_end = int(0.50 * self.fs)

        for i in range(batch_size):
            for r_idx in r_peaks[i]:
                if r_idx < ecg_len:
                    start = min(pcg_len, r_idx + win_start)
                    end = min(pcg_len, r_idx + win_end)
                    target_map[i, r_idx, start:end] = 1.0
        
        # 归一化，使其每行符合概率分布，方便 Loss 计算
        row_sums = target_map.sum(dim=-1, keepdim=True) + 1e-9
        return target_map / row_sums

    # --- 3. 锚点引导掩码 (Anchor Mask) ---
    def get_anchor_mask(self, batch_size, ecg_len, pcg_len, r_peaks, window_ms=100):
        """
        生成 Attention Mask (用于 MultiheadAttention)
        返回: [Batch, ecg_len, pcg_len] 遮蔽处为 -inf
        """
        # MultiheadAttention mask: 0 是不遮蔽，-inf 是遮蔽
        mask = torch.full((batch_size, ecg_len, pcg_len), float('-inf'), device=self.device)
        win_size = int((window_ms / 1000) * self.fs)

        for i in range(batch_size):
            for r_idx in r_peaks[i]:
                if r_idx < ecg_len:
                    # 允许 Attention 在 R 波及其周围 window_ms 范围内寻找 PCG 特征
                    start = max(0, r_idx - win_size)
                    end = min(pcg_len, r_idx + win_size)
                    mask[i, r_idx, start:end] = 0.0
                    
        return mask

    # --- 4. 专家特征提取 (Expert Features) ---
    def get_medical_expert_features(self, r_peaks, s1_peaks):
        """
        提取标量专家特征向量
        r_peaks/s1_peaks: List[Tensor]
        返回: [Batch, 3] (心率, 电机械延迟, RR间期标准差)
        """
        batch_size = len(r_peaks)
        expert_feats = torch.zeros(batch_size, 3, device=self.device)

        for i in range(batch_size):
            cur_r = r_peaks[i].float()
            cur_s1 = s1_peaks[i].float()

            # 1. 平均心率 (基于RR间期)
            if len(cur_r) > 1:
                rr_intervals = torch.diff(cur_r) / self.fs # 秒
                hr = 60.0 / torch.mean(rr_intervals)
                hr_std = torch.std(rr_intervals)
            else:
                hr, hr_std = 75.0, 0.0 # 默认值

            # 2. 电机械延迟 (EMD: R 到 S1 的平均时间)
            if len(cur_r) > 0 and len(cur_s1) > 0:
                # 简化计算：取第一个 R 和 第一个 S1 的差
                emd = (cur_s1[0] - cur_r[0]) / self.fs
            else:
                emd = 0.05

            expert_feats[i] = torch.tensor([hr, emd, hr_std])

        return expert_feats
