import argparse
import os
import json
import torch
import warnings
import numpy as np
import datetime
from tools.datacollection import setup_seed
from pipeline.pipeline_ft_ecg import train_student

warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description="LSNet 1D ECG Training & Ablation Pipeline")
    
    ## 实验核心模式 (Core Experiment Settings)
    parser.add_argument('--mode', type=str, default='homo', choices=['homo', 'hetero', 'only_student'],
                        help='Training mode:\n'
                             'homo: Homogeneous Knowledge Distillation (LSTransECG Teacher -> Student)\n'
                             'hetero: Heterogeneous Knowledge Distillation (NN_default Teacher -> Student)\n'
                             'only_student: Direct student-only fine-tuning (without KD)\n')
    parser.add_argument('--ranklist', type=str, default='lora', choices=['lora', 'ft'],
                        help='Fine-tuning method applied to teacher model: lora (LoRA) or ft (Full fine-tuning)')
    parser.add_argument('--leads_for_student', type=int, default=12, choices=[1, 3, 12],
                        help='Leads configuration for student training (e.g. 1 or 3 for lead-missing scenarios)')
    
    ## 消融实验核心设置 (Ablation Settings)
    parser.add_argument('--ablation_odd', type=str, default='none', choices=['none', 'a1', 'a2', 'a3', 'a4'],
                        help='Odd layers component ablation:\n'
                             'none/a1: Full LSConv (LKP + SKA)\n'
                             'a2: Static Aggregation (LKP + Static-DWConv)\n'
                             'a3: Static Perception (Static-LargeConv + SKA)\n'
                             'a4: Homogeneous-Stacking (Stacked StdConv 3x3)\n')
    parser.add_argument('--ablation_even', type=str, default='none', choices=['none', 'b0', 'b1', 'b2', 'b3'],
                        help='Even layers/Prior components ablation:\n'
                             'none/b0: Full LSNet (LSConv + RepVGGDW + SE)\n'
                             'b1: Homogeneous-LS (LSConv + LSConv stacked)\n'
                             'b2: w/o RepVGG (LSConv + DWConv 3x3 + SE)\n'
                             'b3: w/o SE (LSConv + RepVGGDW)\n')

    ## 路径与环境设定 (Paths & Environment Settings)
    parser.add_argument('--root', type=str, default='/home/xcy/zy/LSTrans', 
                        help='Project root directory (stores checkpoints, results)')
    parser.add_argument('--pretrain_dataset', type=str, default='/data2/zy/LSTrans/data_pretrain', 
                        help='Path to HDF5 pre-training dataset directory')    
    parser.add_argument('--device', type=str, default='cuda:0', 
                        help='Target GPU/CPU computation device') 
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    
    ## 模型参数与实验组级设置 (Model & Class Setting lists)
    parser.add_argument('--model_config', type=str, default='light', choices=['large', 'light'],
                        help='Teacher network model complexity capacity (used for homogeneous setup)')
    parser.add_argument('--ftdata_list', type=str, nargs='+', default=['WFDB_Ga','WFDB_PTBXL','WFDB_ChapmanShaoxing'])
    parser.add_argument('--numclass_list', type=int, nargs='+', default=[18, 19, 16])
    
    ## 数据与单任务细节配置 (Task & Batch Settings)
    parser.add_argument('--ft_dataset', type=str, default='WFDB_Ga')
    parser.add_argument('--num_class', type=int, default=18)
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--ft_epoch', type=int, default=200, help='Max training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    
    ## 秩设定 (LoRA Settings)
    parser.add_argument('--conv_r', type=int, default=8, help='LoRA rank for convolutional layers')
    parser.add_argument('--trans_r', type=int, default=32, help='LoRA rank for transformer modules')
    
    ## 蒸馏参数 (Distillation Hyperparameters)
    parser.add_argument('--kd_temperature', type=float, default=5.0, help='KD Temperature scaling coefficient')
    parser.add_argument('--kd_alpha', type=float, default=0.5, help='Hard-to-Soft label weighting (alpha)')
    
    parser.add_argument('--static_conv', type=bool, default=False, help='Static convolution layer replacement')
    parser.add_argument('--kfold', action='store_true', default=False, help='Perform 5-fold cross-validation instead of single fold')

    args = parser.parse_args()
    return args

def run_single_fold_experiment(args):
    setup_seed(args.seed)
    all_results = {}
    
    # 动态获取消融后缀
    ablation_suffix = ""
    if args.ablation_odd != 'none':
        ablation_suffix += f"_odd_{args.ablation_odd}"
    if args.ablation_even != 'none':
        ablation_suffix += f"_even_{args.ablation_even}"

    print("\n" + "="*50)
    print("      LSNet Single-Fold Experiment Execution      ")
    print("="*50)
    print(f"Mode         : {args.mode}")
    print(f"Teacher FT   : {args.ranklist}")
    print(f"Student Leads: {args.leads_for_student}")
    print(f"Ablation Odd : {args.ablation_odd}")
    print(f"Ablation Even: {args.ablation_even}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"GPU Device   : {args.device}")
    print("="*50 + "\n")

    for i in range(len(args.ftdata_list)):
        args.ft_dataset = args.ftdata_list[i]
        args.num_class = args.numclass_list[i]
        
        print(f">> Dataset: {args.ft_dataset} | num_class: {args.num_class}")
        results = train_student(args, fold_idx=0)
        all_results[args.ft_dataset] = results
        torch.cuda.empty_cache()

    results_save_dir = os.path.join(args.root, 'results')
    os.makedirs(results_save_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    save_filename = f"SingleFold_{args.mode}_{args.ranklist}_leads{args.leads_for_student}{ablation_suffix}_seed{args.seed}_{timestamp}.json"
    save_path = os.path.join(results_save_dir, save_filename)
    
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=4, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
        
    print(f"\n[Success] Experiment results saved to: {save_path}")

def run_kfold_experiment(args):
    setup_seed(args.seed)
    
    # 动态获取消融后缀
    ablation_suffix = ""
    if args.ablation_odd != 'none':
        ablation_suffix += f"_odd_{args.ablation_odd}"
    if args.ablation_even != 'none':
        ablation_suffix += f"_even_{args.ablation_even}"

    print("\n" + "="*50)
    print("        LSNet 5-Fold Experiment Execution         ")
    print("="*50)
    print(f"Mode         : {args.mode}")
    print(f"Teacher FT   : {args.ranklist}")
    print(f"Student Leads: {args.leads_for_student}")
    print(f"Ablation Odd : {args.ablation_odd}")
    print(f"Ablation Even: {args.ablation_even}")
    print(f"GPU Device   : {args.device}")
    print("="*50 + "\n")

    for i in range(len(args.ftdata_list)):
        args.ft_dataset = args.ftdata_list[i]
        args.num_class = args.numclass_list[i]
        print(f">> Dataset: {args.ft_dataset} | Total classes: {args.num_class}")
        
        all_fold_results = []
        for fold in range(5):
            print(f"\n>>> Fold {fold+1}/5 <<<")
            results = train_student(args, fold_idx=fold)
            all_fold_results.append(results)
            torch.cuda.empty_cache()
            
        # 聚合折数并计算均值与标准差
        final_metrics = {}
        keys = all_fold_results[0].keys()
        for key in keys:
            if isinstance(all_fold_results[0][key], (int, float, np.float64, np.float32)):
                values = [res[key] for res in all_fold_results]
                final_metrics[f"{key}_mean"] = float(np.mean(values))
                final_metrics[f"{key}_std"] = float(np.std(values))

        results_save_dir = os.path.join(args.root, 'results')
        os.makedirs(results_save_dir, exist_ok=True)
        save_path = os.path.join(results_save_dir, f"KFold_{args.mode}_{args.ranklist}_leads{args.leads_for_student}{ablation_suffix}_{args.ft_dataset}.json")
        
        with open(save_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)
            
        print(f"\n[Success] 5-Fold results saved to: {save_path}")

if __name__ == '__main__':
    args = get_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[Warning] CUDA is unavailable. Falling back to CPU mode.")
        args.device = 'cpu'

    if args.kfold:
        run_kfold_experiment(args)
    else:
        run_single_fold_experiment(args)