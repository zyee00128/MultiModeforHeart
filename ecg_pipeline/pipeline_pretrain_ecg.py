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

from model.model_code_default import LSTransECG, Cutmix, Cutmix_ECG
from datacollection import ECGcodedataset_loading
from evaluation import print_result, find_thresholds
from pytorchtools import EarlyStopping

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
        inp_windows_t, lbl_t = inp_windows_t.float().to(device), lbl_t.int().to(device)
        with torch.no_grad():
            out = model(inp_windows_t)
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
    neg_ratio = (len(probs) - np.sum(probs, axis=0)) / np.sum(probs, axis=0)
    valid_result.update({'neg_ratio': neg_ratio})
    valid_result.update({'threshold': threshold})
    return valid_result
## pretraining backbone on single GPU
def Large_model_pretraining(args):
    batch_size = args.batch_size if args.batch_size else 128
    model_config = args.model_config
    device = args.device

    if model_config == 'large':
        num_layers, complexity, args.learning_rate = 42, 768, 0.0001
    elif model_config == 'light':
        num_layers, complexity, args.learning_rate = 30, 128, 0.001
    elif model_config == 'student':
        num_layers, complexity, args.learning_rate = 9, 64, 0.001
    setup_seed(args.seed)

    dataset_train, dataset_valid = ECGcodedataset_loading(args=args)
    loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    loader_valid = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=False)
    
    early_stopping = EarlyStopping(10, verbose=True, 
                                   dataset_name=f"CODE15_Pretrain_{model_config}", 
                                   delta=0, args=args)

    Epoch = args.pretrain_epoch
    num_leads = 12
    num_class = args.num_class
    sample_data, _ = dataset_train[0]
    actual_input_length = sample_data.shape[-1]
    print(f"Imput Length: {actual_input_length}")
    net = LSTransECG(nOUT=num_class, out_channels=complexity, in_channels=num_leads, 
                            input_length=actual_input_length, 
                            num_layers=num_layers, rank_list=0).to(device)
    print(f"Number of parameters: {count_parameters(net)}")
    optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    for epoch in range(Epoch):
        total_loss = 0.0
        net.train()
        pbar = tqdm(enumerate(loader_train), total=len(loader_train), desc=f"Epoch {epoch + 1}/{Epoch}")

        for i, (images, labels) in pbar:
            images = images.float().to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            images, labels = Cutmix_ECG(images, labels, device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = F.binary_cross_entropy_with_logits(outputs, labels)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            current_loss = loss.item()
            total_loss += current_loss
            pbar.set_postfix({'loss': f'{current_loss:.4f}', 'avg_loss': f'{total_loss / (i + 1):.4f}'})
        avg_epoch_loss = total_loss / len(loader_train)
        print(f'Epoch {epoch + 1} finished. Average Training Loss: {avg_epoch_loss:.4f}')

        valid_result = validate(net, loader_valid, device)
        early_stopping(1 / valid_result['Map_value'], net)
        if early_stopping.early_stop:
            print("Early stopping")
            break
            
    return
