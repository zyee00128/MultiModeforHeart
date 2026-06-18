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
from torch.utils.data import Dataset, random_split
from skmultilearn.model_selection import iterative_train_test_split
import random
import h5py
from helper_code import *
from preprocess import *
import csv
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

def file_name(file_dir,file_class):  
  L=[]  
  for root, dirs, files in os.walk(file_dir): 
    for file in files: 
      if os.path.splitext(file)[1] == file_class: 
        L.append(os.path.join(root, file)) 
  return L
def conut_nums(dataset_name, csv_file):
    if dataset_name == 'WFDB_PTBXL':
        column_index = 7
    elif dataset_name == 'WFDB_Ga':
        column_index = 8
    elif dataset_name == 'WFDB_Ningbo':
        column_index = 10
    else:
        column_index = 9
    count = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) > column_index:
                count.append(row[column_index])
    count = count[1:]
    count = [int(item) for item in count]
    return count
def multi_label_converter_sepe(multi_label, final_label_list, final_count):
    final_count = np.array(final_count)
    num_class = len(final_label_list)
    one_hot_label = np.zeros(num_class)
    for i in multi_label:
        if i in final_label_list:
            one_hot_label[final_label_list.index(i)] = 1
    return one_hot_label, final_count
def load_dataset_super_sepe(dataset_name, max_length=6144, Norm_type='channel'):
    preprocess_cfg = PreprocessConfig("preprocess.json")
    csv_file = 'label_mapping.csv'  # the label_mapping.csv file can be downloaded from: https://github.com/physionetchallenges/evaluation-2021/blob/main/dx_mapping_scored.csv
    column_index = 1
    data_list = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) > column_index:
                data_list.append(row[column_index])
    data_list = data_list[1:]
    count = conut_nums(dataset_name, csv_file)
    final_label_list = [label for (label, num) in zip(data_list, count) if num > 200]
    final_count = [num for (label, num) in zip(data_list, count) if num > 200]
    print(len(final_label_list))
    print(final_count)
    path = 'C:/Users/赵毅/OneDrive/桌面/LS-ECGNet' ## modify the path based on your project
    ## all the dataset for fine-tuning can be downloaded from https://physionet.org/content/challenge-2021/1.0.3/
    os.chdir(path + '/raw_data/' + dataset_name)
    file_list_record = sorted(file_name(os.getcwd(), '.mat'))
    file_list_head = sorted(file_name(os.getcwd(), '.hea'))
    record_list = []
    label_list = []
    name_list = []
    for i in zip(file_list_record, file_list_head):
        file_name_mat, file_name_head = i[0], i[1]
        multi_label = get_labels(load_header(file_name_head))
        rec_id = os.path.basename(file_name_mat).split('.')[0]
        one_hot_label,count = multi_label_converter_sepe(multi_label, final_label_list, final_count)
        if np.sum(one_hot_label) == 0:
            continue
        if Norm_type == 'channel':
            record = preprocess_signal(recording_normalize(file_name_mat, file_name_head), preprocess_cfg,
                                       get_frequency(load_header(file_name_head)), max_length)
        else:
            record = recording_normalize(file_name_mat, file_name_head)
        if record.shape[1] < max_length:
            record = np.column_stack((record, np.zeros((12, max_length - record.shape[1]))))
        elif record.shape[1] > max_length:
            record = record[:, 0:max_length]
        record = record.astype('float32')
        name_list.append(rec_id)
        record_list.append(record.reshape((record.shape[0], 1, record.shape[1])))
        label_list.append(one_hot_label)
    print(f'dataset {dataset_name}, label_num {len(count)}')
    return record_list, label_list, name_list
def dataset_organize(root): ## prepare dataset for backbone model fine-tuning
    dataset_list=['WFDB_Ga','WFDB_PTBXL','WFDB_Ningbo','WFDB_ChapmanShaoxing']
    for i in range(len(dataset_list)):
        test_dataset_name = dataset_list[i]
        print(test_dataset_name)
        os.chdir(root)
        record_list, label_list, name_list = load_dataset_super_sepe(dataset_name=test_dataset_name)
        test_record_set=np.stack(record_list,axis=0)
        test_label_set = np.vstack(label_list)
        num_of_class = str(test_label_set.shape[1])
        os.chdir(root+'/Preprocessed_dataset')
        hf = h5py.File('class' + num_of_class + '_dataset_' + test_dataset_name + '_' + '32.hdf5', 'w')
        hf.create_dataset('record_set', data=test_record_set)
        hf.create_dataset('label_set', data=test_label_set)
        hf.create_dataset('record_ids', data=np.array(name_list, dtype='S')) 
        print(test_label_set.shape)
        hf.close()
        del record_list, label_list, name_list

if __name__ == '__main__':
    root = 'C:/Users/赵毅/OneDrive/桌面/LS-ECGNet'
    print("\n" + "="*50)
    print("      Starting Data Organization (Mat -> HDF5)    ")
    print("="*50)
    print(f"Root Directory: {root}")
    dataset_organize(root)
    print("\n[Success] Data organization completed.")

