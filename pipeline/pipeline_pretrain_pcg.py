# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:25:30 2023

@author: COCHE User
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader

from tools.evaluation import print_result, find_thresholds
from tools.pytorchtools import EarlyStopping

from model.model_code_default import LSTransPCG, Cutmix_ECG, Cutmix_PCG
from tools.datacollection import PCGCirCorDigiScopedataset_loading

def cleanup():
    dist.destroy_process_group()
def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def count_parameters(model):
    # for n,p in model.named_parameters():
    #     if p.requires_grad:
    #         print(n)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
## model validation on single GPU
def validate(model, valloader, device, iftest=False, threshold=0.5 * np.ones(5), iftrain=False, args=None):
    model.eval()
    losses, probs, lbls = [], [], []
    for step, (inp_windows_t, lbl_t) in enumerate(valloader):
        waveforms, locs = inp_windows_t
        waveforms = waveforms.to(device)
        locs = locs.to(device)
        lbl_t = lbl_t.to(device)
        with torch.no_grad():
            out = model(waveforms, locs)
            loss = F.binary_cross_entropy_with_logits(out, lbl_t.float())
            prob = out.sigmoid().data.cpu().numpy()
            losses.append(loss.item())
            probs.append(prob)
            lbls.append(lbl_t.data.cpu().numpy())
            # logit.append(out.data.cpu().numpy())
    lbls = np.concatenate(lbls)
    probs = np.concatenate(probs)

    num_classes = probs.shape[1]
    threshold = 0.5 * np.ones(num_classes)
    if iftest:
        valid_result = print_result(np.mean(losses), lbls.copy(), probs.copy(), 'test', threshold)
    elif iftrain:
        threshold = find_thresholds(lbls.copy(), probs.copy())
        valid_result = print_result(np.mean(losses), lbls.copy(), probs.copy(), 'train', threshold)
    else:
        threshold = find_thresholds(lbls, probs)
        valid_result = print_result(np.mean(losses), lbls, probs, 'valid', threshold)
    # neg_ratio = (len(probs) - np.sum(probs, axis=0)) / np.sum(probs, axis=0)
    neg_ratio = (len(probs) - np.sum(probs, axis=0)) / (np.sum(probs, axis=0) + 1e-6)
    valid_result.update({'neg_ratio': neg_ratio})
    valid_result.update({'threshold': threshold})
    return valid_result
## pretraining backbone on single GPU
def Large_model_pretraining(args):
    batch_size = args.batch_size if args.batch_size else 128
    model_config = args.model_config
    model_arch = getattr(args, 'model_arch', 'LSTrans') # 新增：'LSTrans' 或 'NN_default'
    device = args.device

    setup_seed(args.seed)

    # 加载数据
    dataset_train, dataset_valid = PCGCirCorDigiScopedataset_loading(args=args)
    loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, 
                              num_workers=0, pin_memory=False, drop_last=True)
    loader_valid = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=False, 
                              num_workers=0, pin_memory=False, drop_last=False)
    (sample_waveform, sample_loc), _ = dataset_train[0]
    actual_input_length = sample_waveform.shape[-1]
    print(f"Input Length: {actual_input_length} | Model Arch: {model_arch}")

    num_leads = 1
    num_class = args.num_class
    loc_dim = args.loc_dim
    if model_arch == 'LSTrans':
        if model_config == 'large':
            num_layers, complexity, args.learning_rate = 42, 768, 0.0001
        elif model_config == 'light':
            num_layers, complexity, args.learning_rate = 30, 128, 0.001
        elif model_config == 'student':
            num_layers, complexity, args.learning_rate = 9, 64, 0.001
        net = LSTransPCG(nOUT=num_class, out_channels=complexity, 
                         in_channels=num_leads, 
                         input_length=actual_input_length, 
                         num_layers=num_layers, rank_list=0, loc_dim=loc_dim).to(device)
    else:
        num_layers, complexity, args.learning_rate = 47, 512, 0.0001
        model_config = 'medium'
        from model.model_code_default import NN_PCG
        net = NN_PCG(nOUT=num_class, complexity=complexity, 
                     inputchannel=num_leads, 
                     input_length=actual_input_length,
                     num_layers=num_layers, rank_list=0, loc_dim=loc_dim).to(device)

    print(f"Number of parameters: {count_parameters(net)}")

    optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=0.01)
    early_stopping = EarlyStopping(10, verbose=True, 
                                   dataset_name=f"CirCorDigiScopePCG_{model_arch}_{model_config}", 
                                   delta=0, args=args)
    
    Epoch = args.pretrain_epoch
    for epoch in range(Epoch):
        net.train()
        total_loss = 0.0
        pbar = tqdm(enumerate(loader_train), total=len(loader_train), desc=f"Epoch {epoch + 1}/{Epoch}")

        for i, (data, labels) in pbar:
            waveforms, locs = data
            waveforms = waveforms.float().to(device, non_blocking=True)
            locs = locs.float().to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            waveforms, locs, labels = Cutmix_PCG(waveforms, locs, labels, device)
            optimizer.zero_grad()
            outputs = net(waveforms, locs)
            
            loss = F.binary_cross_entropy_with_logits(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss / (i + 1):.4f}'})

        valid_result = validate(net, loader_valid, device)
        early_stopping(1 / valid_result['Map_value'], net)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    return