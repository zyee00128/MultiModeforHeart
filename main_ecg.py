import argparse
import os
import json
import torch
import warnings
import numpy as np
from tools.datacollection import setup_seed
from pipeline.pipeline_pretrain_ecg import Large_model_pretraining
from pipeline.pipeline_ft_ecg import kd_student_model,cross_kd_stu
from pipeline.pipeline_pretrain_pcg import Large_model_pretraining
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description="LSNet 1D ECG Model Pretraining")
    ## 基础环境与路径设置 (Basic Environment & Path Settings)
    parser.add_argument('--ranklist', type=str, default='FT', 
                        help='训练模式标识，决定保存逻辑，预训练模型和学生模型只使用FT')
    parser.add_argument('--tea_ranklist', type=str, default='lora_ave', 
                        help='训练模式标识，决定保存逻辑，教师模型可选择FT or lora_ave')
    parser.add_argument('--root', type=str, default='/home/xcy/zy/LSTrans', 
                        help='项目根目录，用于存放checkpoints等 (Project root directory)')
    parser.add_argument('--pretrain_dataset', type=str, default='/data2/zy/LSTrans/data_pretrain', 
                        help='预训练数据集文件夹路径,应包含exams.csv和hdf5文件 (Path to pretraining dataset folder, which should contain exams.csv and HDF5 files)')    
    parser.add_argument('--device', type=str, default='cuda:0', 
                        help='计算设备 (cuda or cpu)') 
    parser.add_argument('--preload_devices', type=str, nargs='+', default=['cuda:0', 'cuda:1'], 
                        help='预加载数据到的设备 (Device to preload data onto, e.g., "cuda:0" or "cpu")')
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子，保证实验可复现 (Random seed)')
    
    ## 模型配置 (Model Configuration)
    parser.add_argument('--model_config', type=str, default='light', choices=['large', 'light', 'student'],
                            help= 'large: 42 layers, 1 encoder, 768 dim\n'
                                'light: 30 layers, 1 encoder, 128 dim\n'
                                'student: 9 layers, 1 encoder, 64 dim\n')
    parser.add_argument('--ftdata_list', type=str, nargs='+', default=['WFDB_Ga','WFDB_PTBXL','WFDB_ChapmanShaoxing'])
    parser.add_argument('--ft_dataset', type=str, default='WFDB_Ga', choices=['WFDB_Ga','WFDB_PTBXL','WFDB_ChapmanShaoxing'],
                        help='微调数据集名称 (Fine-tuning dataset name)')
    parser.add_argument('--numclass_list',type=int, nargs='+', default=[18, 19, 16])
    parser.add_argument('--num_class', type=int, default=6, choices=[6, 18, 19, 16],
                        help='分类任务的类别数量 (Number of output classes)')
    parser.add_argument('--ft_data_ratio', type=float, default=0.10,
                        help='微调时使用的训练和测试数据比例 (Data ratio for fine-tuning)')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='批次大小 (Batch size)')
    parser.add_argument('--ft_epoch', type=int, default=200,
                        help='微调的总Epoch数 (Total fine-tuning epochs)')
    parser.add_argument('--pretrain_epoch', type=int, default=100, 
                        help='预训练的总Epoch数 (Total training epochs)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率 (Learning rate)')
    parser.add_argument('--conv_r', type=int, default=8, help='网络浅层卷积层的lora_rank')
    parser.add_argument('--trans_r', type=int, default=32, help='网络深层Transformer层的lora_rank')
    #KD
    parser.add_argument('--kd_temperature', type=float, default=5.0,
                        help='温度系数 (Temperature scaling)，平滑教师输出概率')
    parser.add_argument('--kd_alpha', type=float, default=0.5,
                        help='软标签损失 (Soft Loss) 的权重')
    # parser.add_argument('--kd_beta', type=float, default=10.0,
    #                     help=' 特征级对齐损失 (MSE Loss) 的权重')
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--leads_for_student', type=int, default=12, choices=[1, 3, 12],
                        help='用于学生模型的导联索引 (Leads indices for student model)')
    
    parser.add_argument('--task', type=str, default='pretrain', 
                        help='执行的任务类型 (Task type to execute)')
    parser.add_argument('--static_conv',  type=bool, default=False)
    args = parser.parse_args()
    return args

def pretrain_main(args):
    expected_save_dir = os.path.join(args.root, 'pretrained_checkpoint')
    expected_filename = f"CODE15_Pretrain_{args.model_config}_checkpoint.pt"
    expected_save_path = os.path.join(expected_save_dir, expected_filename)

    print("\n" + "="*50)
    print("               LSNet Pretraining      ")
    print("="*50)
    print(f"Model Config : {args.model_config}")
    print(f"Device       : {args.device}")
    print(f"Epochs       : {args.pretrain_epoch}")
    print(f"Batch Size   : {args.batch_size}")
    print(f"Data Path    : {args.pretrain_dataset}")
    print(f"Save Mode    : {args.ranklist}")
    print(f"Save Path    : {expected_save_path}") 
    print("="*50 + "\n")
    setup_seed(args.seed)

    #启动预训练
    try:
        Large_model_pretraining(args)
        print("\n[Success] Training pipeline completed successfully.")
    except Exception as e:
        print(f"\n[Error] Training failed with error: {e}")
        raise e

def run_finetune_loop(args):
    setup_seed(args.seed)
    
    for mode in ['lora_ave', 'FT']:
        all_results = {}
        args.tea_ranklist = mode
        if args.task == 'kd_finetune':
            print("\n" + "="*50)
            print("           LSNet KD Finetuning      ")
            print("="*50)
            print(f"Teacher Config: {args.model_config}_{args.tea_ranklist}")
            print(f"Student Leads : {args.leads_for_student}")
            print(f"Device: {args.device}")
            print(f"Learning Rate: {args.learning_rate}")
            print(f"Dataset_list:{args.ftdata_list}")
            print("="*50 + "\n")
            for i in range(len(args.ftdata_list)):
                args.ft_dataset = args.ftdata_list[i]
                args.num_class = args.numclass_list[i]
                args.ranklist = args.tea_ranklist
                print(f"Dataset:{args.ft_dataset}")
                print(f"nOUT:{args.num_class}")
                 
                results = kd_student_model(args)
                all_results[args.ft_dataset] = results
                torch.cuda.empty_cache()

        results_save_dir = os.path.join(args.root, 'results')
        os.makedirs(results_save_dir, exist_ok=True)
        
        suffix = "KD_FT" if args.task == 'kd_finetune' else "OnlyFt"
        import datetime
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        save_filename = f"{suffix}_{args.model_config}_{args.tea_ranklist}_seed{args.seed}_{timestamp}.json"
        save_path = os.path.join(results_save_dir, save_filename)
        
        with open(save_path, 'w') as f:
            json.dump(all_results, f, indent=4, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
            
        print(f"\n[Success] All results saved to: {save_path}")

def run_cross_loop(args):
    setup_seed(args.seed)
    
    for mode in ['lora_ave', 'FT']:
        all_results = {}
        args.tea_ranklist = mode
        print("\n" + "="*50)
        print("           LSNet KD-Cross Finetuning      ")
        print("="*50)
        print(f"Teacher Config: {args.model_config}_{args.tea_ranklist}")
        print(f"Device: {args.device}")
        print(f"Learning Rate: {args.learning_rate}")
        print(f"Dataset_list: {args.ftdata_list}")
        print(f"KD-Temperature: {args.kd_temperature}")
        print(f"KD-Alpha: {args.kd_alpha}")
        print(f"EarlyStopping Patience: {args.patience}")
        print("="*50 + "\n")
        for i in range(len(args.ftdata_list)):
            args.ft_dataset = args.ftdata_list[i]
            args.num_class = args.numclass_list[i]
            args.ranklist = args.tea_ranklist
            print(f"Dataset:{args.ft_dataset}")
            print(f"nOUT:{args.num_class}")
             
            results = cross_kd_stu(args)
            all_results[args.ft_dataset] = results
            torch.cuda.empty_cache()
  
        results_save_dir = os.path.join(args.root, 'results')
        os.makedirs(results_save_dir, exist_ok=True)
        
        suffix = "KD_Cross"
        import datetime
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        save_filename = f"{suffix}_{args.model_config}_{args.tea_ranklist}_seed{args.seed}_{timestamp}.json"
        save_path = os.path.join(results_save_dir, save_filename)
        
        with open(save_path, 'w') as f:
            json.dump(all_results, f, indent=4, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
            
        print(f"\n[Success] All results saved to: {save_path}")

def run_kfold_experiment(args):
    # args.task = 'kd_finetune'
    # args.task = 'Cross_kd'
    print(f"Task:{args.task}")
    all_results = []
    for i in range(len(args.ftdata_list)):
        args.ft_dataset = args.ftdata_list[i]
        args.num_class = args.numclass_list[i]
        print(f"Dataset:{args.ft_dataset}")
        print(f"nOUT:{args.num_class}")

        for fold in range(5):
            print(f"\n>>> Starting Fold {fold+1}/5 <<<")
            if args.task == 'Homo':
                results = kd_student_model(args, fold_idx=fold)
            elif args.task == 'Hetero':
                results = cross_kd_stu(args, fold_idx=fold)
            all_results.append(results)
            torch.cuda.empty_cache()
        
        final_metrics = {}
        keys = all_results[0].keys()
        
        for key in keys:
            if isinstance(all_results[0][key], (int, float, np.float64, np.float32)):
                values = [res[key] for res in all_results]
                final_metrics[f"{key}_mean"] = np.mean(values)
                final_metrics[f"{key}_std"] = np.std(values)
    
        save_path = os.path.join(args.root, 'results', f"KFold_Results_{args.ft_dataset}.json")
        with open(save_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)
    print(f"\n[Success] 5-Fold results saved to: {save_path}")

if __name__ == '__main__':
    args = get_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[Warning] CUDA不可用，自动切换至CPU模式。")
        args.device = 'cpu'

    if args.task == 'pretrain':
        pretrain_main(args)
    elif args.task == 'Homo' or args.task =='Hetero':
        run_kfold_experiment(args)