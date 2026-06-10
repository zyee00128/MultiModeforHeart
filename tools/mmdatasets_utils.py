# -*- coding: utf-8 -*-
import os
import numpy as np
import h5py
import librosa
import scipy.io as sio
from scipy.signal import resample_poly
import pandas as pd
from tqdm import tqdm

try:
    import wfdb
except ImportError:
    wfdb = None

class MultimodalProcessor:
    def __init__(self, ecg_target_hz=400, pcg_target_hz=4000, ecg_max_len=4096, pcg_max_len=40000):
        self.ecg_hz = ecg_target_hz
        self.pcg_hz = pcg_target_hz
        self.ecg_max_len = ecg_max_len
        self.pcg_max_len = pcg_max_len
    
    def _resample_signal(self, signal, orig_hz, target_hz):
        """
        高精度重采样信号
        """
        if orig_hz == target_hz:
            return signal
        # 使用 Scipy 的多相重采样（比直接线性插值更保真）
        gcd = np.gcd(int(orig_hz), int(target_hz))
        up = int(target_hz // gcd)
        down = int(orig_hz // gcd)
        return resample_poly(signal, up, down)

    def _pad_or_crop_ecg(self, ecg_data):
        """对齐 ECG 至 [1, ecg_max_len]"""
        if ecg_data.ndim > 1:
            ecg_data = np.squeeze(ecg_data)
        
        current_len = len(ecg_data)
        if current_len < self.ecg_max_len:
            pad_width = self.ecg_max_len - current_len
            ecg_data = np.pad(ecg_data, (0, pad_width), mode='constant')
        else:
            ecg_data = ecg_data[:self.ecg_max_len]
        return np.expand_dims(ecg_data, axis=0).astype('float32')

    def _pad_or_crop_pcg(self, pcg_data):
        """对齐 PCG 数据长度至 pcg_max_len"""
        if pcg_data.ndim > 1:
            pcg_data = np.squeeze(pcg_data)
            
        # Z-score 规范化并限幅滤除运动伪影
        pcg_data = (pcg_data - np.mean(pcg_data)) / (np.std(pcg_data) + 1e-8)
        pcg_data = np.clip(pcg_data, -3.0, 3.0)

        current_len = len(pcg_data)
        if current_len < self.pcg_max_len:
            pad_width = self.pcg_max_len - current_len
            pcg_data = np.pad(pcg_data, (0, pad_width), mode='constant')
        elif current_len > self.pcg_max_len:
            start = np.random.randint(0, current_len - self.pcg_max_len)
            pcg_data = pcg_data[start : start + self.pcg_max_len]
                
        return np.expand_dims(pcg_data, axis=0).astype('float32') # [1, pcg_max_len]

    def process_cardiology2016(self, data_dir, output_hdf5):
        """
        解析 Cardiology2016 的 training-a 真实单通道数据
        正常标签映射为 0.0，异常标签映射为 1.0
        """
        print("--- 开始预处理 Cardiology2016 (真实单通道模式) ---")
        if wfdb is None:
            raise ImportError("未检测到 wfdb 库，解析 WFDB 格式文件请先安装: pip install wfdb")
            
        csv_path = os.path.join(data_dir, 'REFERENCE.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"未找到 REFERENCE.csv: {csv_path}")
            
        df = pd.read_csv(csv_path, header=None, names=['record_id', 'label'])
        
        ecg_list, pcg_list, label_list, loc_list, ids = [], [], [], [], []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Cardiology2016"):
            rec_id = row['record_id']
            # CinC 2016 标签规范: -1 代表 Normal, 1 代表 Abnormal
            label_val = 1.0 if row['label'] == 1 else 0.0
            
            hea_file = os.path.join(data_dir, f"{rec_id}.hea")
            wav_file = os.path.join(data_dir, f"{rec_id}.wav")
            
            if not (os.path.exists(hea_file) and os.path.exists(wav_file)):
                continue
                
            try:
                # 1. 提取单路真实心电并重采样
                record = wfdb.rdrecord(os.path.join(data_dir, rec_id))
                sig_names = [s.upper() for s in record.sig_name]
                if 'ECG' in sig_names:
                    ecg_idx = sig_names.index('ECG')
                else:
                    ecg_idx = 1  # 默认在 training-a 中，通道 1 为 ECG
                ecg_orig = record.p_signal[:, ecg_idx]
                # ecg_orig = record.p_signal.T[0] # 取主通道波形
                ecg_resampled = self._resample_signal(ecg_orig, record.fs, self.ecg_hz)
                # ecg_final = self._pad_or_crop_ecg(ecg_resampled)
                
                # 2. 提取心音并重采样
                pcg_orig, pcg_fs = librosa.load(wav_file, sr=None, mono=True)
                pcg_resampled = self._resample_signal(pcg_orig, pcg_fs, self.pcg_hz)
                # pcg_final = self._pad_or_crop_pcg(pcg_resampled)

                # 计算两路信号的物理时间（秒），并取较小值作为可裁切的上限
                ecg_duration = len(ecg_resampled) / self.ecg_hz
                pcg_duration = len(pcg_resampled) / self.pcg_hz
                min_duration = min(ecg_duration, pcg_duration)
                target_duration = 10.0  # 期望对齐的秒数
                if min_duration <= target_duration:
                    # 如果原信号不足 10s，直接进行 Padding 作为一个片段保存
                    ecg_final = self._pad_or_crop_ecg(ecg_resampled)
                    pcg_final = self._pad_or_crop_pcg(pcg_resampled)
                    
                    ecg_list.append(ecg_final)
                    pcg_list.append(pcg_final)
                    label_list.append(np.array([label_val], dtype='float32'))
                    loc_list.append(np.zeros(5, dtype='float32'))
                    ids.append(f"{rec_id}_chunk0".encode('utf-8'))
                else:
                    # 如果信号大于 10s，使用滑动窗口切分（例如：步长 5s 重叠切分）
                    step_sec = 5.0
                    ecg_chunk_pts = int(target_duration * self.ecg_hz)
                    pcg_chunk_pts = int(target_duration * self.pcg_hz)
                    ecg_step_pts = int(step_sec * self.ecg_hz)
                    pcg_step_pts = int(step_sec * self.pcg_hz)
                    
                    idx = 0
                    chunk_cnt = 0
                    while (idx * ecg_step_pts + ecg_chunk_pts) <= len(ecg_resampled):
                        ecg_start = idx * ecg_step_pts
                        pcg_start = idx * pcg_step_pts
                        
                        ecg_chunk = ecg_resampled[ecg_start : ecg_start + ecg_chunk_pts]
                        pcg_chunk = pcg_resampled[pcg_start : pcg_start + pcg_chunk_pts]
                        
                        ecg_final = self._pad_or_crop_ecg(ecg_chunk)
                        pcg_final = self._pad_or_crop_pcg(pcg_chunk)
                        
                        ecg_list.append(ecg_final)
                        pcg_list.append(pcg_final)
                        label_list.append(np.array([label_val], dtype='float32'))
                        loc_list.append(np.zeros(5, dtype='float32'))
                        
                        chunk_name = f"{rec_id}_chunk{chunk_cnt}"
                        ids.append(chunk_name.encode('utf-8'))
                        
                        idx += 1
                        chunk_cnt += 1
                
            except Exception as e:
                print(f"处理文件 {rec_id} 失败: {e}")
                continue

        # 写入 HDF5
        os.makedirs(os.path.dirname(output_hdf5), exist_ok=True)
        with h5py.File(output_hdf5, 'w') as hf:
            hf.create_dataset('ecg_set', data=np.array(ecg_list))
            hf.create_dataset('pcg_set', data=np.array(pcg_list))
            hf.create_dataset('label_set', data=np.array(label_list))
            hf.create_dataset('loc_set', data=np.array(loc_list))
            hf.create_dataset('record_ids', data=np.array(ids))
        print(f"成功导出 {len(ids)} 条真实的单通道 Cardiology2016 记录至 {output_hdf5}\n")

    def process_ephnogram(self, data_dir, output_hdf5, segment_len_sec=10.0, overlap_sec=5.0):
        """
        解析 EPHNOGRAM 的长 MATLAB 数据并执行真实滑动窗口切片
        切片产生的 ECG 与 PCG 均保持 1 通道原始物理特征
        """
        print("--- 开始预处理 EPHNOGRAM ---")
        mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
        ecg_list, pcg_list, label_list, loc_list, ids = [], [], [], [], []

        # 物理切片点数换算
        ecg_chunk_pts = int(segment_len_sec * self.ecg_hz)
        pcg_chunk_pts = int(segment_len_sec * self.pcg_hz)
        ecg_step_pts = int((segment_len_sec - overlap_sec) * self.ecg_hz)
        pcg_step_pts = int((segment_len_sec - overlap_sec) * self.pcg_hz)

        for file_name in tqdm(mat_files, desc="Processing EPHNOGRAM"):
            file_path = os.path.join(data_dir, file_name)
            base_id = file_name.replace('.mat', '')
            
            try:
                data = sio.loadmat(file_path)
                fs_orig = int(data['fs'][0][0])

                # 提取单通道心电与心音
                raw_ecg = data['ECG']
                # ecg_single = raw_ecg[0] if raw_ecg.ndim > 1 else raw_ecg
                if raw_ecg.ndim > 1:
                # 判断哪一个维度是通道数
                    if raw_ecg.shape[0] > raw_ecg.shape[1]:
                        # 代表格式为 [N, 3]，时序在第 0 维，通道在第 1 维
                        ecg_single = raw_ecg[:, 0]
                    else:
                        # 代表格式为 [3, N]，通道在第 0 维，时序在第 1 维
                        ecg_single = raw_ecg[0, :]
                else:
                    ecg_single = raw_ecg
                raw_pcg = data['PCG']
                pcg_single = raw_pcg[0] if raw_pcg.ndim > 1 else raw_pcg
                
                # 统一重采样到物理对齐目标
                ecg_resampled = self._resample_signal(ecg_single, fs_orig, self.ecg_hz)
                pcg_resampled = self._resample_signal(pcg_single, fs_orig, self.pcg_hz)
                
                total_len_sec = len(ecg_resampled) / self.ecg_hz
                if total_len_sec < segment_len_sec:
                    continue
                
                idx = 0
                chunk_cnt = 0
                while (idx * ecg_step_pts + ecg_chunk_pts) <= len(ecg_resampled):
                    ecg_start = idx * ecg_step_pts
                    pcg_start = idx * pcg_step_pts
                    
                    ecg_chunk = ecg_resampled[ecg_start : ecg_start + ecg_chunk_pts]
                    pcg_chunk = pcg_resampled[pcg_start : pcg_start + pcg_chunk_pts]
                    
                    ecg_final = self._pad_or_crop_ecg(ecg_chunk)
                    pcg_final = self._pad_or_crop_pcg(pcg_chunk)
                    
                    # 健康组标签为 0.0
                    label_val = 0.0
                    loc_onehot = np.zeros(5, dtype='float32')
                    
                    ecg_list.append(ecg_final)
                    pcg_list.append(pcg_final)
                    label_list.append(np.array([label_val], dtype='float32'))
                    loc_list.append(loc_onehot)
                    
                    chunk_name = f"{base_id}_chunk{chunk_cnt}"
                    ids.append(chunk_name.encode('utf-8'))
                    
                    idx += 1
                    chunk_cnt += 1
                    
            except Exception as e:
                print(f"处理文件 {file_name} 失败: {e}")
                continue

        # 写入 HDF5
        os.makedirs(os.path.dirname(output_hdf5), exist_ok=True)
        with h5py.File(output_hdf5, 'w') as hf:
            hf.create_dataset('ecg_set', data=np.array(ecg_list))
            hf.create_dataset('pcg_set', data=np.array(pcg_list))
            hf.create_dataset('label_set', data=np.array(label_list))
            hf.create_dataset('loc_set', data=np.array(loc_list))
            hf.create_dataset('record_ids', data=np.array(ids))
        print(f"成功导出 {len(ids)} 条真实的单通道 EPHNOGRAM 段至 {output_hdf5}\n")
       