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
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import librosa

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

# prepare dataset for LSNet
class ECGImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, hdf5_path, transform=None, preload_devices=None, return_device=None):
        """
        Args:
            img_dir: 生成的图片存放的总目录
            hdf5_path: 对应的 HDF5 文件路径
            transform: 图像预处理变换
            preload_devices: 预加载的目标设备列表 ['cuda:0', 'cuda:1']
            return_device: 训练时返回数据的设备
        """
        self.img_dir = img_dir
        self.transform = transform
        
        with h5py.File(hdf5_path, 'r') as hf:
            all_labels = np.array(hf.get('label_set')).astype(np.float32)
            all_ids = [x.decode('utf-8') for x in hf.get('record_ids')] 
        # id_to_idx = {rec_id: i for i, rec_id in enumerate(all_ids)}
        id_to_idx = {(id.decode('utf-8') if isinstance(id, bytes) else str(id)): i 
        for i, id in enumerate(all_ids)}

        self.valid_img_paths = []
        self.target_labels = []
        

        print(f"正在匹配图片与 HDF5 标签 (Dataset: {os.path.basename(hdf5_path)})...")
        for root, _, files in os.walk(img_dir):
            for f in files:
                if f.endswith('.png'):
                    # 1. 获取完整文件名，如 '04188_lr-0'
                    full_img_id = os.path.splitext(f)[0].strip()
                    
                    # 2. 去掉后缀，如 '04188_lr-0' -> '04188'
                    base_id = full_img_id.split('_')[0]
                    
                    # 3. 尝试多种可能的 ID 格式去 HDF5 里碰运气
                    match_id = None
                    
                    # 格式A: 去除前导零的纯数字 (针对 PTB-XL: '04188' -> '4188')
                    if base_id.isdigit() and str(int(base_id)) in id_to_idx:
                        match_id = str(int(base_id))
                    
                    # 格式B: 保持原样 (针对 Georgia/Chapman: 'E00001' 或 '04188')
                    elif base_id in id_to_idx:
                        match_id = base_id
                        
                    # 格式C: 连带后缀的完整名 (某些特定数据集可能需要)
                    elif full_img_id in id_to_idx:
                        match_id = full_img_id

                    # 4. 如果找到了匹配的 ID，就加入训练列表
                    if match_id is not None:
                        self.valid_img_paths.append(os.path.join(root, f))
                        self.target_labels.append(all_labels[id_to_idx[match_id]])


        self.preload_devices = [preload_devices] if isinstance(preload_devices, str) else preload_devices
        self.return_device = return_device if return_device else (self.preload_devices[0] if self.preload_devices else 'cpu')
        self.is_preloaded = False
        if self.preload_devices is not None:
            print(f"正在将 {len(self.valid_img_paths)} 张 ECG 轨迹图预加载至 {self.preload_devices}...")
            self.preloaded_imgs = []
            self.preloaded_labels = []
            safe_margin = 1.5 * 1024 * 1024 * 1024
            device_idx = 0
            current_device = self.preload_devices[device_idx]
            for i in tqdm(range(len(self.valid_img_paths)), desc="Image Preloading"):
                if i % 50 == 0 and current_device.startswith('cuda'):
                    free_mem, _ = torch.cuda.mem_get_info(current_device)
                    if free_mem < safe_margin and device_idx < len(self.preload_devices) - 1:
                        device_idx += 1
                        current_device = self.preload_devices[device_idx]
                        tqdm.write(f"\n[Info] {self.preload_devices[device_idx-1]} 显存不足，切换至: {current_device}")
                
                img_path = self.valid_img_paths[i]
                img_tensor = self._load_and_transform(img_path)
                label_tensor = torch.tensor(self.target_labels[i], dtype=torch.float32)
                
                try:
                    self.preloaded_imgs.append(img_tensor.to(current_device))
                    self.preloaded_labels.append(label_tensor.to(current_device))
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and device_idx < len(self.preload_devices) - 1:
                        device_idx += 1
                        current_device = self.preload_devices[device_idx]
                        tqdm.write(f"\n[Warning] 捕捉到 OOM，切换至: {current_device}")
                        self.preloaded_imgs.append(img_tensor.to(current_device))
                        self.preloaded_labels.append(label_tensor.to(current_device))
                    else:
                        raise e
            
            self.is_preloaded = True
            print(f"成功预加载 {len(self.preloaded_imgs)} 张图片。")

    def _load_and_transform(self, img_path):
        """辅助函数：读取图片并转为 Tensor"""
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"\n[严重警告] 发现损坏的图片文件: {img_path}")
            print(f"报错信息: {e}")
            raise e # 打印出路径后再抛出异常，方便你定位去删除它
        
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.valid_img_paths)

    def __getitem__(self, idx):
        if self.is_preloaded:
            return (self.preloaded_imgs[idx].to(self.return_device), 
                    self.preloaded_labels[idx].to(self.return_device))

        img_path = self.valid_img_paths[idx]
        image = self._load_and_transform(img_path)
        label = torch.tensor(self.target_labels[idx], dtype=torch.float32)
        return image, label
def ECGimagedataset_loading(args):
    """
    完善后的 2D 轨迹图加载函数
    """
    img_dir = os.path.join(args.root, 'Preprocessed_dataset/ECG_Imgs', args.ft_dataset)
    label_hdf5_path = os.path.join(args.root, 'Preprocessed_dataset', 
                                   f'class{args.num_class}_dataset_{args.ft_dataset}_32.hdf5')
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"找不到图片目录: {img_dir}，请先生成轨迹图。")
    
    # 标准 LSNet/ImageNet 预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = ECGImageDataset(
        img_dir=img_dir, 
        hdf5_path=label_hdf5_path, 
        transform=transform,
        preload_devices=getattr(args, 'preload_devices', None),
        return_device=args.device
    )

    total_len = len(full_dataset)
    test_size = int(total_len * (1 - args.ft_data_ratio))
    train_val_size = total_len - test_size
    valid_size = int(train_val_size * 0.2)
    train_size = train_val_size - valid_size

    generator = torch.Generator().manual_seed(args.seed)
    dataset_train, dataset_valid, dataset_test = random_split(
        full_dataset, [train_size, valid_size, test_size], generator=generator
    )
    
    return dataset_train, dataset_valid, dataset_test
# prepare dataset for mm
class ECGMultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, hdf5_path, transform=None, max_length=4096, preload_devices=None, return_device=None):
        """
        多模态数据集加载器：同时读取 2D 图像和 1D 信号，并通过 ID 绝对对齐。
        """
        self.img_dir = img_dir
        self.transform = transform
        self.max_length = max_length
        
        # 1. 打开 HDF5 文件，加载 1D 信号、标签和 ID
        with h5py.File(hdf5_path, 'r') as hf:
            all_records_1d = np.array(hf.get('record_set'))
            all_labels = np.array(hf.get('label_set')).astype(np.float32)
            all_ids = [x.decode('utf-8') for x in hf.get('record_ids')] 
            
        id_to_idx = {(id.decode('utf-8') if isinstance(id, bytes) else str(id)): i 
                     for i, id in enumerate(all_ids)}

        self.valid_img_paths = []
        self.target_1d_records = []
        self.target_labels = []

        print(f"正在严格对齐多模态数据 (Images + 1D HDF5)...")
        for root, _, files in os.walk(img_dir):
            for f in files:
                if f.endswith('.png'):
                    full_img_id = os.path.splitext(f)[0].strip()
                    base_id = full_img_id.split('_')[0]
                    
                    match_id = None
                    if base_id.isdigit() and str(int(base_id)) in id_to_idx:
                        match_id = str(int(base_id))
                    elif base_id in id_to_idx:
                        match_id = base_id
                    elif full_img_id in id_to_idx:
                        match_id = full_img_id

                    if match_id is not None:
                        idx = id_to_idx[match_id]
                        self.valid_img_paths.append(os.path.join(root, f))
                        # 直接截断或补齐 1D 信号并存入列表
                        self.target_1d_records.append(self._pad_or_crop(all_records_1d[idx]))
                        self.target_labels.append(all_labels[idx])

        print(f"匹配完成！共成功对齐 {len(self.valid_img_paths)} 个多模态样本。")

        self.preload_devices = [preload_devices] if isinstance(preload_devices, str) else preload_devices
        self.return_device = return_device if return_device else (self.preload_devices[0] if self.preload_devices else 'cpu')
        self.is_preloaded = False
        
        # 2. 预加载逻辑 (同时加载 1D 和 2D 进显存)
        if self.preload_devices is not None:
            print(f"正在将 {len(self.valid_img_paths)} 个多模态样本预加载至 {self.preload_devices}...")
            self.preloaded_1d = []
            self.preloaded_2d = []
            self.preloaded_labels = []
            
            safe_margin = 1.5 * 1024 * 1024 * 1024
            device_idx = 0
            current_device = self.preload_devices[device_idx]
            
            for i in tqdm(range(len(self.valid_img_paths)), desc="Multimodal Preloading"):
                if i % 50 == 0 and current_device.startswith('cuda'):
                    free_mem, _ = torch.cuda.mem_get_info(current_device)
                    if free_mem < safe_margin and device_idx < len(self.preload_devices) - 1:
                        device_idx += 1
                        current_device = self.preload_devices[device_idx]
                        tqdm.write(f"\n[Info] {self.preload_devices[device_idx-1]} 显存不足，切换至: {current_device}")
                
                img_path = self.valid_img_paths[i]
                img_tensor = self._load_and_transform(img_path)
                trace_tensor = torch.tensor(self.target_1d_records[i], dtype=torch.float32)
                label_tensor = torch.tensor(self.target_labels[i], dtype=torch.float32)
                
                try:
                    self.preloaded_1d.append(trace_tensor.to(current_device))
                    self.preloaded_2d.append(img_tensor.to(current_device))
                    self.preloaded_labels.append(label_tensor.to(current_device))
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and device_idx < len(self.preload_devices) - 1:
                        device_idx += 1
                        current_device = self.preload_devices[device_idx]
                        tqdm.write(f"\n[Warning] 捕捉到 OOM，切换至: {current_device}")
                        self.preloaded_1d.append(trace_tensor.to(current_device))
                        self.preloaded_2d.append(img_tensor.to(current_device))
                        self.preloaded_labels.append(label_tensor.to(current_device))
                    else:
                        raise e
            
            self.is_preloaded = True
            print(f"成功分布式预加载多模态数据。")

    def _pad_or_crop(self, trace):
        """处理 1D HDF5 信号的 Padding/截断"""
        if trace.ndim == 3:
            trace = np.squeeze(trace)
        current_len = trace.shape[1]
        if current_len < self.max_length:
            padding = np.zeros((trace.shape[0], self.max_length - current_len), dtype=np.float32)
            trace = np.concatenate((trace, padding), axis=1)
        elif current_len > self.max_length:
            trace = trace[:, :self.max_length]
        return trace

    def _load_and_transform(self, img_path):
        """读取图片并转为 Tensor"""
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"\n[严重警告] 发现损坏的图片文件: {img_path}")
            raise e
        
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.valid_img_paths)

    def __getitem__(self, idx):
        # 如果已经预加载到显存，直接返回 3 个 Tensor
        if self.is_preloaded:
            return (self.preloaded_1d[idx].to(self.return_device), 
                    self.preloaded_2d[idx].to(self.return_device),
                    self.preloaded_labels[idx].to(self.return_device))

        # 否则实时读取
        trace = torch.tensor(self.target_1d_records[idx], dtype=torch.float32)
        image = self._load_and_transform(self.valid_img_paths[idx])
        label = torch.tensor(self.target_labels[idx], dtype=torch.float32)
        
        return trace, image, label
def ECG_multimodal_dataset_loading(args):
    """
    供 pipeline_mm 调用的数据加载和划分函数
    """
    img_dir = os.path.join(args.root, 'Preprocessed_dataset/ECG_Imgs', args.ft_dataset)
    label_hdf5_path = os.path.join(args.root, 'Preprocessed_dataset', 
                                   f'class{args.num_class}_dataset_{args.ft_dataset}_32.hdf5')
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"找不到图片目录: {img_dir}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = ECGMultimodalDataset(
        img_dir=img_dir, 
        hdf5_path=label_hdf5_path, 
        transform=transform,
        max_length=4096,
        preload_devices=getattr(args, 'preload_devices', None),
        return_device=args.device
    )

    total_len = len(full_dataset)
    test_size = int(total_len * (1 - getattr(args, 'ft_data_ratio', 1.0)))
    train_val_size = total_len - test_size
    valid_size = int(train_val_size * 0.2)
    train_size = train_val_size - valid_size

    generator = torch.Generator().manual_seed(args.seed)
    dataset_train, dataset_valid, dataset_test = random_split(
        full_dataset, [train_size, valid_size, test_size], generator=generator
    )
    
    return dataset_train, dataset_valid, dataset_test

