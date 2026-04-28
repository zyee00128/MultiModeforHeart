# -*- coding: utf-8 -*-
import os
import numpy as np
import h5py
import librosa
import pandas as pd
from tqdm import tqdm
from data_processing.helper_code import load_header, get_labels, get_frequency, recording_normalize
from data_processing.preprocess import PreprocessConfig, preprocess_signal

class MultimodalProcessor:
    def __init__(self, ecg_max_len=4096, pcg_max_len=40000, sample_rate_pcg=4000):
        self.ecg_max_len = ecg_max_len
        self.pcg_max_len = pcg_max_len
        self.sr_pcg = sample_rate_pcg
        # 加载 ECG 预处理配置
        if os.path.exists("preprocess.json"):
            self.ecg_cfg = PreprocessConfig("preprocess.json")
        else:
            print("Warning: preprocess.json not found, using default ECG config.")
            self.ecg_cfg = PreprocessConfig()

    def process_ecg(self, mat_path, hea_path):
        """处理单个 ECG 信号，逻辑与 datacollection_processing 一致"""
        header = load_header(hea_path)
        fs = get_frequency(header)
        # 1. 物理单位转换 (ADC gains & baselines)
        record = recording_normalize(mat_path, hea_path)
        # 2. 滤波与缩放 (Filter & Scaler)
        record = preprocess_signal(record, self.ecg_cfg, fs, self.ecg_max_len)
        
        # 3. 长度裁剪或填充
        if record.shape[1] < self.ecg_max_len:
            padding = np.zeros((record.shape[0], self.ecg_max_len - record.shape[1]))
            record = np.column_stack((record, padding))
        else:
            record = record[:, :self.ecg_max_len]
        
        return record.astype('float32')

    def process_pcg(self, wav_path):
        """处理单个 PCG 信号，逻辑与 CirCorPCGDataset 一致"""
        try:
            # 1. Librosa 加载 (重采样到 4000Hz)
            waveform, _ = librosa.load(wav_path, sr=self.sr_pcg, mono=True)
            
            # 2. Z-score 标准化
            mu = np.mean(waveform)
            sigma = np.std(waveform)
            waveform = (waveform - mu) / (sigma + 1e-8)

            # 3. 幅度截断 (Clipping)
            waveform = np.clip(waveform, -3, 3)

            # 4. 长度裁剪或填充 (默认取中心或填充)
            if len(waveform) > self.pcg_max_len:
                start = (len(waveform) - self.pcg_max_len) // 2
                waveform = waveform[start : start + self.pcg_max_len]
            else:
                pad_width = self.pcg_max_len - len(waveform)
                waveform = np.pad(waveform, (0, pad_width), mode='constant')
            
            return waveform.reshape(1, -1).astype('float32')
        except Exception as e:
            print(f"Error processing PCG {wav_path}: {e}")
            return np.zeros((1, self.pcg_max_len), dtype='float32')

    def label_converter(self, labels, final_label_list):
        """多标签转 One-hot"""
        num_class = len(final_label_list)
        one_hot = np.zeros(num_class, dtype='float32')
        for lb in labels:
            if lb in final_label_list:
                one_hot[final_label_list.index(lb)] = 1
        return one_hot

    def organize_dataset(self, data_dir, output_path, label_mapping_csv, final_label_list):
        """
        同步处理 ECG/PCG 并保存为 HDF5
        假设目录结构中：ID.mat, ID.hea, ID.wav 三者同步
        """
        ecg_list, pcg_list, loc_list, label_list, name_list = [], [], [], [], []
        
        # 获取目录下所有 header 文件
        hea_files = [f for f in os.listdir(data_dir) if f.endswith('.hea')]
        
        for hea_name in tqdm(hea_files, desc="Syncing Multimodal Data"):
            base_id = hea_name.replace('.hea', '')
            mat_path = os.path.join(data_dir, base_id + '.mat')
            hea_path = os.path.join(data_dir, base_id + '.hea')
            wav_path = os.path.join(data_dir, base_id + '.wav')
            
            if not (os.path.exists(mat_path) and os.path.exists(wav_path)):
                continue # 跳过非同步数据
            
            # 1. 处理 ECG
            ecg_data = self.process_ecg(mat_path, hea_path)
            
            # 2. 处理 PCG
            pcg_data = self.process_pcg(wav_path)
            
            # 3. 处理标签
            multi_labels = get_labels(load_header(hea_path))
            one_hot_label = self.label_converter(multi_labels, final_label_list)
            
            if np.sum(one_hot_label) == 0:
                continue
                
            # 4. 获取听诊位置 (如果有) - 此处假设位置信息从文件名或 header 提取，默认占位
            loc_onehot = np.zeros(5) # 默认 5 个位置：AV, PV, TV, MV, PhC
            
            ecg_list.append(ecg_data)
            pcg_list.append(pcg_data)
            label_list.append(one_hot_label)
            loc_list.append(loc_onehot)
            name_list.append(base_id)

        # 写入 HDF5
        with h5py.File(output_path, 'w') as hf:
            hf.create_dataset('ecg_set', data=np.array(ecg_list))
            hf.create_dataset('pcg_set', data=np.array(pcg_list))
            hf.create_dataset('label_set', data=np.array(label_list))
            hf.create_dataset('loc_set', data=np.array(loc_list))
            hf.create_dataset('record_ids', data=np.array(name_list, dtype='S'))
            
        print(f"Successfully saved {len(name_list)} synchronized records to {output_path}")