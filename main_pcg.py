import argparse
import os
import json
import torch
import warnings
import numpy as np
from datacollection import setup_seed
from pcg_pipeline.pipeline_pretrain_pcg import Large_model_pretraining
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description="LSTransPCG Model Pretraining & Fine-tuning")
    
    ## 基础环境与路径设置 (Basic Environment & Path Settings)
    parser.add_argument('--root', type=str, default='/home/xcy/zy/LSnet4ECG', 
                        help='项目根目录 (Project root directory)')
    parser.add_argument('--pcg_data_path', type=str, default='/data/CirCorDigiScope', 
                        help='CirCor DigiScope数据集路径 (Path to CirCor dataset, contains training_data.csv)')    
    parser.add_argument('--device', type=str, default='cuda:0', 
                        help='计算设备 (cuda or cpu)') 
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子 (Random seed)')
    
    ## 模型配置 (Model Configuration)
    parser.add_argument('--model_config', type=str, default='large', choices=['large', 'light', 'student'],
                        help='模型规模配置')
    parser.add_argument('--num_class', type=int, default=2, 
                        help='分类任务类别数 (CirCor Outcome默认为2: Normal/Abnormal)')
    parser.add_argument('--loc_dim', type=int, default=5, 
                        help='听诊位置特征维度 (AV, PV, TV, MV, PhC)')
    parser.add_argument('--pcg_len', type=int, default=40000, 
                        help='输入信号采样点长度 (4000Hz * 10s = 40000)')

    ## 训练超参数 (Training Hyperparameters)
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='批次大小')
    parser.add_argument('--pretrain_epoch', type=int, default=100, 
                        help='预训练总Epoch数')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='学习率')
    parser.add_argument('--patience', type=int, default=15,
                        help='EarlyStopping 耐心值')

    parser.add_argument('--task', type=str, default='pretrain', choices=['pretrain', 'finetune'],
                        help='执行的任务类型: pretrain (预训练) 或 finetune (微调)')
    parser.add_argument('--model_arch', type=str, default='LSTrans', choices=['LSTrans', 'NN_default'],
                        help='模型架构选择')
    args = parser.parse_args()
    return args

def pretrain_main(args):
    """
    PCG 预训练主函数
    """
    expected_save_dir = os.path.join(args.root, 'checkpoints', 'pcg_pretrain')
    os.makedirs(expected_save_dir, exist_ok=True)
    
    expected_filename = f"CirCor_Pretrain_{args.model_config}_checkpoint.pt"
    expected_save_path = os.path.join(expected_save_dir, expected_filename)

    print("\n" + "="*50)
    print("           LSTransPCG Pretraining (CirCor)      ")
    print("="*50)
    print(f"Model Config : {args.model_config}")
    print(f"Model Arch   : {args.model_arch}")
    print(f"Device       : {args.device}")
    print(f"Epochs       : {args.pretrain_epoch}")
    print(f"Batch Size   : {args.batch_size}")
    print(f"Signal Len   : {args.pcg_len}")
    print(f"Data Path    : {args.pcg_data_path}")
    print(f"Save Path    : {expected_save_path}") 
    print("="*50 + "\n")

    setup_seed(args.seed)

    try:
        Large_model_pretraining(args)
        print(f"\n[Success] Pretraining completed. Model saved at {expected_save_path}")
    except Exception as e:
        print(f"\n[Error] Pretraining failed: {e}")
        raise e

if __name__ == '__main__':
    args = get_args()

    if 'cuda' in args.device and not torch.cuda.is_available():
        print("[Warning] CUDA is not available, switching to CPU.")
    else:
        pretrain_main(args)
