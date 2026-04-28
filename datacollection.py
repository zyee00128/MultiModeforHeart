# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 11:37:53 2022

@author: COCHE User
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import Dataset, random_split, Subset
from sklearn.model_selection import KFold, train_test_split
import random
import h5py
import librosa
from tqdm import tqdm
from mmdatasets_utils import MultimodalProcessor

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# prepare dataset for backbone model fine-tuning
class FINETUNEDataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_path, preload_devices=None, max_length=4096, return_device=None):
        """
        初始化微调数据集加载器
        """
        self.max_length = max_length
        hf = h5py.File(hdf5_path, 'r')
        self.records = np.array(hf.get('record_set'))
        self.labels = np.array(hf.get('label_set'))
        hf.close()
        
        self.preload_devices = [preload_devices] if isinstance(preload_devices, str) else preload_devices
        self.return_device = return_device if return_device else (self.preload_devices[0] if self.preload_devices else 'cpu')
        
        self.is_preloaded = False
        if self.preload_devices is not None:
            print(f"正在将微调数据集预加载至 {self.preload_devices}...")
            self.preloaded_data = []
            self.preloaded_labels = []

            safe_margin = 1 * 1024 * 1024 * 1024  # 1GB 安全阈值
            device_idx = 0
            current_device = self.preload_devices[device_idx]

            for i in tqdm(range(len(self.records)), desc="Preloading"):
                # 每 100 条检查一次显存状态
                if i % 100 == 0 and current_device.startswith('cuda'):
                    free_mem, _ = torch.cuda.mem_get_info(current_device)
                    if free_mem < safe_margin and device_idx < len(self.preload_devices) - 1:
                        device_idx += 1
                        current_device = self.preload_devices[device_idx]
                        tqdm.write(f"\n[Info] {self.preload_devices[device_idx-1]} 显存达到预警，切换至: {current_device}")

                processed_trace = self._pad_or_crop(self.records[i])
                
                # 使用 try-except 捕获突发 OOM
                try:
                    trace_t = torch.tensor(processed_trace, dtype=torch.float32, device=current_device)
                    label_t = torch.tensor(self.labels[i], dtype=torch.float32, device=current_device)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and device_idx < len(self.preload_devices) - 1:
                        device_idx += 1
                        current_device = self.preload_devices[device_idx]
                        tqdm.write(f"\n[Warning] 捕捉到 OOM，强制切换至: {current_device}")
                        # 在新设备上重试
                        trace_t = torch.tensor(processed_trace, dtype=torch.float32, device=current_device)
                        label_t = torch.tensor(self.labels[i], dtype=torch.float32, device=current_device)
                    else:
                        raise e
                
                self.preloaded_data.append(trace_t)
                self.preloaded_labels.append(label_t)
                
            self.is_preloaded = True

    def _pad_or_crop(self, trace):
        """
        提取的通用代码：处理 HDF5 信号的读取、转置与 Padding/截断
        """
        if trace.ndim == 3:
            trace = np.squeeze(trace)
        current_len = trace.shape[1]
        if current_len < self.max_length:
            padding = np.zeros((trace.shape[0], self.max_length - current_len), dtype=np.float32)
            trace = np.concatenate((trace, padding), axis=1)
        elif current_len > self.max_length:
            trace = trace[:, :self.max_length]
        return trace
           
    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        if self.is_preloaded:
            return self.preloaded_data[idx].to(self.return_device), self.preloaded_labels[idx].to(self.return_device)

        trace = self._pad_or_crop(self.records[idx])
        return torch.tensor(trace, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32) 
def ECGfinetunedataset_loading(args, fold_idx=0):
    data_path = os.path.join(args.root, 'Preprocessed_dataset', 
                             f'class_sepe{args.num_class}_dataset_{args.ft_dataset}_32.hdf5')

    full_dataset = FINETUNEDataset(
        hdf5_path=data_path, 
        max_length=4096,
        preload_devices=getattr(args, 'preload_devices', None), 
        return_device=args.device
    )

    total_len = len(full_dataset)

    indices = np.arange(total_len)
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    splits = list(kf.split(indices))
    train_val_idx, test_idx = splits[fold_idx]
    val_size = int(len(train_val_idx) * 0.2)
    np.random.seed(args.seed)
    np.random.shuffle(train_val_idx)
    train_idx = train_val_idx[val_size:]
    valid_idx = train_val_idx[:val_size]

    dataset_train = Subset(full_dataset, train_idx)
    dataset_valid = Subset(full_dataset, valid_idx)
    dataset_test = Subset(full_dataset, test_idx)
    
    return dataset_train, dataset_valid, dataset_test
# prepare dataset for ECG backbone model pretraining
class CODEDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, hdf5_folder, max_length=4096, preload_devices=None, return_device=None):
        """
        初始化 CODE 数据集加载器
        """
        self.df = pd.read_csv(csv_file)
        self.hdf5_folder = hdf5_folder
        self.max_length = max_length
        self.classes = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']
        self.labels = self.df[self.classes].replace({True: 1.0, False: 0.0}).values.astype(np.float32)
        
        # 兼容单设备字符串传入
        if isinstance(preload_devices, str):
            self.preload_devices = [preload_devices]
        else:
            self.preload_devices = preload_devices
        # 决定数据产出设备，默认返回到列表中的第一个GPU
        self.return_device = return_device if return_device else (self.preload_devices[0] if self.preload_devices else None)

        self.file_to_index_map = {}
        unique_files = self.df['trace_file'].dropna().unique()
        
        for file_name in unique_files:
            file_path = os.path.join(self.hdf5_folder, file_name)
            if os.path.exists(file_path):
                with h5py.File(file_path, 'r') as f:
                    exam_ids_in_file = np.array(f['exam_id'])
                    self.file_to_index_map[file_name] = {eid: i for i, eid in enumerate(exam_ids_in_file)}
            
            else:
                print(f"警告: 找不到文件 {file_path}")

        self.valid_indices = [
            i for i, row in self.df.iterrows() 
            if row['trace_file'] in self.file_to_index_map and row['exam_id'] in self.file_to_index_map[row['trace_file']]
        ]
        print(f"索引建立完成！共找到 {len(self.valid_indices)} 条有效 ECG 记录。")

        self.is_preloaded = False
        if self.preload_devices is not None:
            print(f"正在将全部数据集分布式预加载至 {self.preload_devices}...")
            self.preloaded_data = []
            self.preloaded_labels = []
            
            # 提高 I/O 效率，先对需要读取的数据按 trace_file 分组
            file_groups = {}
            for real_idx in self.valid_indices:
                row = self.df.iloc[real_idx]
                file_name = row['trace_file']
                exam_id = row['exam_id']
                if file_name not in file_groups:
                    file_groups[file_name] = []
                file_groups[file_name].append((real_idx, exam_id))

            device_idx = 0
            current_device = self.preload_devices[device_idx]
            # 设定安全显存余量为 1GB
            safe_margin = 1 * 1024 * 1024 * 1024 
            sample_count = 0

            for file_name, items in tqdm(file_groups.items(), desc="Preloading files"):
                file_path = os.path.join(self.hdf5_folder, file_name)
                with h5py.File(file_path, 'r') as f:
                    tracings = f['tracings']
                    for real_idx, exam_id in items:
                        idx_in_file = self.file_to_index_map[file_name][exam_id]
                        trace = self._process_trace(tracings, idx_in_file)

                        sample_count += 1
                        # 每存 100 条数据检查一次，避免每次检查带来的性能损耗
                        if sample_count % 100 == 0 and current_device.startswith('cuda'):
                            free_mem, _ = torch.cuda.mem_get_info(current_device)
                            # 切换设备
                            if free_mem < safe_margin and device_idx < len(self.preload_devices) - 1:
                                device_idx += 1
                                current_device = self.preload_devices[device_idx]
                                
                                tqdm.write(f"\n[Info] {self.preload_devices[device_idx-1]} 显存已达到安全边界，自动切换至: {current_device}")
                        
                        # ==== 双保险：使用 try-except 捕获真实的突发 OOM ====
                        try:
                            trace_t = torch.tensor(trace, dtype=torch.float32, device=current_device)
                            label_t = torch.tensor(self.labels[real_idx], dtype=torch.float32, device=current_device)
                        except RuntimeError as e:
                            # 如果报错包含 out of memory 且还有备用卡
                            if "out of memory" in str(e).lower() and device_idx < len(self.preload_devices) - 1:
                                device_idx += 1
                                current_device = self.preload_devices[device_idx]
                                tqdm.write(f"\n[Warning] 捕捉到 OOM，强制切换至: {current_device}")
                                # 在新设备上重试
                                trace_t = torch.tensor(trace, dtype=torch.float32, device=current_device)
                                label_t = torch.tensor(self.labels[real_idx], dtype=torch.float32, device=current_device)
                            else:
                                raise e # 所有显卡都满了，或者发生非 OOM 错误，则抛出异常

                        self.preloaded_data.append(trace_t)
                        self.preloaded_labels.append(label_t)
            
            self.is_preloaded = True
            print("预加载完成！")

    def _process_trace(self, h5_tracings, idx_in_file):
        """
        提取的通用代码：处理 HDF5 信号的读取、转置与 Padding/截断
        """
        trace = h5_tracings[idx_in_file].T
        if trace.shape[1] < self.max_length:
            trace = np.column_stack((trace, np.zeros((12, self.max_length - trace.shape[1]))))
        elif trace.shape[1] > self.max_length:
            trace = trace[:, 0:self.max_length]
        return trace

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        if self.is_preloaded:
            trace_t = self.preloaded_data[idx]
            label_t = self.preloaded_labels[idx]
            return trace_t.to(self.return_device), label_t.to(self.return_device)
        
        real_idx = self.valid_indices[idx]
        row = self.df.iloc[real_idx]
        file_name = row['trace_file']
        exam_id = row['exam_id']
        file_path = os.path.join(self.hdf5_folder, file_name)
        idx_in_file = self.file_to_index_map[file_name][exam_id]

        with h5py.File(file_path, 'r') as f:
            trace = self._process_trace(f['tracings'], idx_in_file)

        trace_t = torch.tensor(trace, dtype=torch.float32)
        label_t = torch.tensor(self.labels[real_idx], dtype=torch.float32)
        
        return trace_t, label_t        
def ECGcodedataset_loading(args):
    csv_path = os.path.join(args.pretrain_dataset, 'exams.csv')
    hdf5_dir = args.pretrain_dataset

    full_dataset = CODEDataset(csv_file=csv_path, hdf5_folder=hdf5_dir, max_length=4096, preload_devices=args.preload_devices, return_device=args.device)

    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(args.seed if hasattr(args, 'seed') else 42)
    dataset_train, dataset_valid = random_split(full_dataset, [train_size, valid_size], generator=generator)
    
    return dataset_train, dataset_valid
# prepare dataset for PCG backbone model pretraining
class CirCorPCGDataset(Dataset):
    """
        针对 CirCor DigiScope 心音数据集的自定义 Dataset 类
        Args:
            file_paths: .wav 文件路径列表
            labels: 诊断标签 (0: Normal, 1: Abnormal)
            locations: 听诊位置编码 (One-hot)
            target_length: 目标采样点长度 (默认 4000Hz * 5s = 20000)
            is_train: 是否为训练模式 (决定裁剪策略)

    """
    def __init__(self, file_paths, labels, locations, target_length=40000, is_train=True):
        self.file_paths = file_paths
        self.labels = labels
        self.locations = locations
        self.target_length = target_length
        self.is_train = is_train
        self.sample_rate = 4000

    def __len__(self):
        return len(self.file_paths)

    def _preprocess_signal(self, waveform):
        # Z-score 标准化
        mu = np.mean(waveform)
        sigma = np.std(waveform)
        waveform = (waveform - mu) / (sigma + 1e-8)

        # 幅度截断 (Clipping) 剔除异常极值
        waveform = np.clip(waveform, -3, 3)

        # 长度对齐 (Padding/Cropping)
        if len(waveform) > self.target_length:
            if self.is_train:
                # 训练集：随机裁剪
                start = np.random.randint(0, len(waveform) - self.target_length)
                waveform = waveform[start : start + self.target_length]
            else:
                # 验证/测试集：中心裁剪
                start = (len(waveform) - self.target_length) // 2
                waveform = waveform[start : start + self.target_length]
        else:
            # 零填充 (Zero Padding)
            pad_width = self.target_length - len(waveform)
            waveform = np.pad(waveform, (0, pad_width), mode='constant')
        
        return waveform

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            waveform, _ = librosa.load(path, sr=self.sample_rate, mono=True)
            waveform = self._preprocess_signal(waveform)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            waveform = np.zeros(self.target_length)

        waveform_t = torch.from_numpy(waveform).float().unsqueeze(0) # [1, Length]
        label_idx = int(self.labels[idx])
        label_t = torch.zeros(2, dtype=torch.float32)
        label_t[label_idx] = 1.0
        loc_t = torch.tensor(self.locations[idx], dtype=torch.float32)

        # 返回信号、标签和位置编码
        return (waveform_t, loc_t), label_t
def PCGCirCorDigiScopedataset_loading(args):
    """
    加载并划分 CirCor DigiScope 数据集
    """
    data_dir = args.pcg_data_path 
    csv_path = os.path.join(data_dir, 'training_data.csv')
    df = pd.read_csv(csv_path)

    # 位置映射表 (Location One-hot)
    loc_map = {'AV': 0, 'PV': 1, 'TV': 2, 'MV': 3, 'PhC': 4}
    num_locs = len(loc_map)
    file_paths = []
    labels = []
    locations = []
    patient_ids = [] # 用于病人级划分
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        pid = str(row['Patient ID'])
        # 诊断结果：Abnormal -> 1, Normal -> 0
        outcome = 1 if row['Outcome'] == 'Abnormal' else 0
        # 遍历该病人的所有听诊点
        recorded_locs = row['Recording locations:'].split('+')

        for loc in recorded_locs:
            loc = loc.strip()
            if loc not in loc_map: continue
            wav_file = f"{pid}_{loc}.wav"
            full_path = os.path.join(data_dir, wav_file)
            
            if os.path.exists(full_path):
                file_paths.append(full_path)
                labels.append(outcome)
                patient_ids.append(pid)
                
                # 生成位置的 One-hot
                loc_onehot = np.zeros(num_locs)
                loc_onehot[loc_map[loc]] = 1
                locations.append(loc_onehot)

    # 确保同一个人的所有录音都在同一个 set
    unique_pids = list(set(patient_ids))
    train_pids, test_pids = train_test_split(unique_pids, test_size=0.2, random_state=args.seed)
    train_pids, val_pids = train_test_split(train_pids, test_size=0.1, random_state=args.seed)

    # 根据 PID 索引分配样本
    def get_indices_by_pids(target_pids, patient_ids=patient_ids):
        return [i for i, p in enumerate(patient_ids) if p in target_pids]
    train_idx = get_indices_by_pids(train_pids)
    val_idx = get_indices_by_pids(val_pids)
    test_idx = get_indices_by_pids(test_pids)

    # 实例化 Dataset
    full_data = {
        'file_paths': file_paths,
        'labels': labels,
        'locations': locations
    }
    
    dataset_train = CirCorPCGDataset(
        [file_paths[i] for i in train_idx],
        [labels[i] for i in train_idx],
        [locations[i] for i in train_idx],
        target_length=args.pcg_len, is_train=True
    )
    
    dataset_valid = CirCorPCGDataset(
        [file_paths[i] for i in val_idx],
        [labels[i] for i in val_idx],
        [locations[i] for i in val_idx],
        target_length=args.pcg_len, is_train=False
    )
    
    dataset_test = CirCorPCGDataset(
        [file_paths[i] for i in test_idx],
        [labels[i] for i in test_idx],
        [locations[i] for i in test_idx],
        target_length=args.pcg_len, is_train=False
    )

    print(f"数据集划分完成：训练集 {len(dataset_train)}，验证集 {len(dataset_valid)}，测试集 {len(dataset_test)}")
    return dataset_train, dataset_valid, dataset_test
# prepare dataset for multimodal model pretraining
class MultimodalDataset(Dataset):
    def __init__(self, hdf5_path, ecg_max_length=4096, pcg_max_length=40000, is_train=True, preload_devices=None, return_device=None):
        self.ecg_max_length = ecg_max_length
        self.pcg_max_length = pcg_max_length
        self.is_train = is_train
        
        # 从预处理的 HDF5 文件中同时加载多模态数据
        hf = h5py.File(hdf5_path, 'r')
        self.ecg_records = np.array(hf.get('ecg_set'))
        self.pcg_records = np.array(hf.get('pcg_set'))
        self.locations = np.array(hf.get('loc_set'))
        self.labels = np.array(hf.get('label_set'))
        hf.close()
        
        self.preload_devices = [preload_devices] if isinstance(preload_devices, str) else preload_devices
        self.return_device = return_device if return_device else (self.preload_devices[0] if self.preload_devices else 'cpu')
        
        self.is_preloaded = False
        if self.preload_devices is not None:
            print(f"正在将多模态数据集预加载至 {self.preload_devices}...")
            self.preloaded_ecg =[]
            self.preloaded_pcg = []
            self.preloaded_loc = []
            self.preloaded_labels =[]

            safe_margin = 1 * 1024 * 1024 * 1024  # 1GB 显存安全阈值
            device_idx = 0
            current_device = self.preload_devices[device_idx]

            for i in tqdm(range(len(self.labels)), desc="Preloading Multimodal Data"):
                if i % 100 == 0 and current_device.startswith('cuda'):
                    free_mem, _ = torch.cuda.mem_get_info(current_device)
                    if free_mem < safe_margin and device_idx < len(self.preload_devices) - 1:
                        device_idx += 1
                        current_device = self.preload_devices[device_idx]
                        tqdm.write(f"\n[Info] {self.preload_devices[device_idx-1]} 显存达到预警，切换至: {current_device}")

                ecg_processed = self._process_ecg(self.ecg_records[i])
                pcg_processed = self._process_pcg(self.pcg_records[i])
                
                try:
                    ecg_t = torch.tensor(ecg_processed, dtype=torch.float32, device=current_device)
                    pcg_t = torch.tensor(pcg_processed, dtype=torch.float32, device=current_device)
                    loc_t = torch.tensor(self.locations[i], dtype=torch.float32, device=current_device)
                    label_t = torch.tensor(self.labels[i], dtype=torch.float32, device=current_device)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and device_idx < len(self.preload_devices) - 1:
                        device_idx += 1
                        current_device = self.preload_devices[device_idx]
                        tqdm.write(f"\n[Warning] 捕捉到 OOM，强制切换至: {current_device}")
                        ecg_t = torch.tensor(ecg_processed, dtype=torch.float32, device=current_device)
                        pcg_t = torch.tensor(pcg_processed, dtype=torch.float32, device=current_device)
                        loc_t = torch.tensor(self.locations[i], dtype=torch.float32, device=current_device)
                        label_t = torch.tensor(self.labels[i], dtype=torch.float32, device=current_device)
                    else:
                        raise e
                
                self.preloaded_ecg.append(ecg_t)
                self.preloaded_pcg.append(pcg_t)
                self.preloaded_loc.append(loc_t)
                self.preloaded_labels.append(label_t)
                
            self.is_preloaded = True

    def _process_ecg(self, trace):
        if trace.ndim == 3:
            trace = np.squeeze(trace)
        current_len = trace.shape[-1]
        
        # Padding & Cropping
        if current_len < self.ecg_max_length:
            padding = np.zeros((trace.shape[0], self.ecg_max_length - current_len), dtype=np.float32)
            trace = np.concatenate((trace, padding), axis=-1)
        elif current_len > self.ecg_max_length:
            trace = trace[:, :self.ecg_max_length]
        return trace
    def _process_pcg(self, waveform):
        if waveform.ndim > 1:
            waveform = np.squeeze(waveform)
            
        # Padding & Cropping
        current_len = len(waveform)
        if current_len > self.pcg_max_length:
            if self.is_train:
                # 随机裁剪
                start = np.random.randint(0, current_len - self.pcg_max_length)
                waveform = waveform[start : start + self.pcg_max_length]
            else:
                # 居中裁剪
                start = (current_len - self.pcg_max_length) // 2
                waveform = waveform[start : start + self.pcg_max_length]
        elif current_len < self.pcg_max_length:
            pad_width = self.pcg_max_length - current_len
            waveform = np.pad(waveform, (0, pad_width), mode='constant')
        
        if waveform.ndim == 1:
            waveform = np.expand_dims(waveform, axis=0)
        return waveform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.is_preloaded:
            return (self.preloaded_ecg[idx].to(self.return_device),
                    self.preloaded_pcg[idx].to(self.return_device),
                    self.preloaded_loc[idx].to(self.return_device),
                    self.preloaded_labels[idx].to(self.return_device))

        ecg_trace = self._process_ecg(self.ecg_records[idx])
        pcg_trace = self._process_pcg(self.pcg_records[idx])
        
        return (torch.tensor(ecg_trace, dtype=torch.float32),
                torch.tensor(pcg_trace, dtype=torch.float32),
                torch.tensor(self.locations[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.float32))
def MultimodalDataset_loading(args, fold_idx=0):
    """
    提供外部调用的多模态数据加载及K-Fold划分接口
    """
    data_path = os.path.join(args.root, 'Preprocessed_dataset', 
                             f'class_sepe{args.num_class}_multimodal_dataset_{args.ft_dataset}.hdf5')

    full_dataset = MultimodalDataset(
        hdf5_path=data_path, 
        ecg_max_length=4096,
        pcg_max_length=getattr(args, 'pcg_len', 40000),
        is_train=True,
        preload_devices=getattr(args, 'preload_devices', None), 
        return_device=args.device
    )

    total_len = len(full_dataset)
    indices = np.arange(total_len)
    
    # 交叉验证划分
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    splits = list(kf.split(indices))
    train_val_idx, test_idx = splits[fold_idx]
    
    # 从训练集中划出20%作验证集
    val_size = int(len(train_val_idx) * 0.2)
    np.random.seed(args.seed)
    np.random.shuffle(train_val_idx)
    
    train_idx = train_val_idx[val_size:]
    valid_idx = train_val_idx[:val_size]

    dataset_train = Subset(full_dataset, train_idx)
    dataset_valid = Subset(full_dataset, valid_idx)
    dataset_test = Subset(full_dataset, test_idx)
    
    return dataset_train, dataset_valid, dataset_test
def run_multimodal_processing():
    root = ''
    processor = MultimodalProcessor(ecg_max_len=4096, pcg_max_len=40000)
    
    # 定义需要的类别列表 (从 label_mapping.csv 中筛选出的高频类)
    final_label_list = ['164889003', '426783006', '270492004', '164891005'] # 示例 SNOMED 代码
    
    data_dir = os.path.join(root, '')
    output_hdf5 = os.path.join(root, 'Preprocessed_dataset/multimodal_dataset_v1.hdf5')
    
    processor.organize_dataset(
        data_dir=data_dir,
        output_path=output_hdf5,
        label_mapping_csv='',
        final_label_list=final_label_list
    )

if __name__ == '__main__':
    run_multimodal_processing()