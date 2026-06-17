import os
import argparse
import torch
import numpy as np
import json

# 导入管道和核心组件
from pipeline.mm_pretrain_EPHNOGRAM import run_multimodal_joint_pretraining
from pipeline.mm_pipeline import (
    multimodal_kd_teacher_model,
    multimodal_kd_student_model,
    setup_seed
)

def parse_args():
    parser = argparse.ArgumentParser(description="ECG-PCG Multimodal Pretraining and Fine-tuning Pipeline")
    
    # ========================== 核心运行模式 ==========================
    parser.add_argument('--mode', type=str, default='finetune_teacher',
                        choices=['pretrain', 'finetune_teacher', 'finetune_student'],
                        help="运行模式: pretrain (联合预训练), finetune_teacher (微调教师模型), finetune_student (蒸馏微调学生)")
    parser.add_argument('--seed', type=int, default=42, help="随机种子")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="计算设备 (例如: 'cuda', 'cuda:0', 'cpu')")
    parser.add_argument('--root', type=str, default='./', 
                        help="项目根目录，用于保存 checkpoint 和 outputs")

    # ========================== 数据集与分类配置 ==========================
    parser.add_argument('--ft_dataset', type=str, default='cardiology2016',
                        choices=['cardiology2016', 'ephnogram'],
                        help="下游微调数据集名称 (预训练时会自动强制为 'ephnogram')")
    parser.add_argument('--num_class', type=int, default=2,
                        help="分类任务类别数 (Cardiology2016 默认为 2 分类 [Normal, Abnormal])")
    parser.add_argument('--pcg_len', type=int, default=40000,
                        help="PCG 时序截断/填充的目标长度 (4000Hz * 10s = 40000)")
    parser.add_argument('--preload_devices', type=str, nargs='*', default=None,
                        help="显存预加载设备列表。若有多张显卡可传入多个设备以分散内存负载，如: cuda:0 cuda:1")

    # ========================== 物理与生理对齐配置 ==========================
    parser.add_argument('--alignment_mode', type=str, default='none',
                        choices=['none', 'cascaded', 'constraint', 'anchor', 'dual_stream'],
                        help="生理对齐模式: none (不对齐), cascaded (级联粗对齐), constraint (损失约束), anchor (锚点掩码), dual_stream (双流注入)")
    parser.add_argument('--lambda_physio', type=type(0.1), default=0.1,
                        help="生理一致性对齐 Loss 的权重 (constraint 模式下的 KL 散度系数)")

    # ========================== 网络结构与微调方法 (LoRA) ==========================
    parser.add_argument('--model_config', type=str, default='LSTrans',
                        help="骨干网络基础标识命名 (用于拼装 checkpoint 文件名)")
    parser.add_argument('--is_hetero', action='store_true',
                        help="教师模型是否采用高参数量的 Hetero 47 层网络 (不加该参数则默认为 Homo 30 层)")
    parser.add_argument('--ranklist', type=str, default='FT',
                        choices=['FT', 'lora'],
                        help="微调模式: FT (Full Tuning 全参数微调), lora (仅更新 LoRA + Classifier 等层)")
    parser.add_argument('--conv_r', type=int, default=8, help="网络中 Conv2d 层的 LoRA 秩 (Rank)")
    parser.add_argument('--trans_r', type=int, default=32, help="网络中 Transformer 层的 LoRA 秩 (Rank)")

    # ========================== 训练超参数 ==========================
    parser.add_argument('--batch_size', type=int, default=64, help="批样本数")
    parser.add_argument('--ft_epoch', type=int, default=50, help="训练总 Epoch 数")
    parser.add_argument('--patience', type=int, default=10, help="早停机制容忍度 (Patience Epochs)")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="基础学习率")

    # ========================== 知识蒸馏 (KD) 特有超参数 ==========================
    parser.add_argument('--kd_temperature', type=float, default=2.0, help="知识蒸馏温度系数 T")
    parser.add_argument('--kd_alpha', type=float, default=0.5, help="软标签 Loss 的占比权重 (1-alpha 为硬标签权重)")

    # ========================== 各种外部权重路径配置 ==========================
    # 1. 联合预训练所需的单模态 Backbone 路径 (用于权重切片手术)
    parser.add_argument('--ecg_unimodal_pretrained', type=str, default='',
                        help="ECG 单模态 12 导联预训练权重路径 (用于预训练第一步提取 Lead II)")
    parser.add_argument('--pcg_unimodal_pretrained', type=str, default='',
                        help="PCG 单模态预训练权重路径")
    
    # 2. 下游微调时，初始化 Teacher 基础 Backbone 的多模态预训练权重路径
    parser.add_argument('--teacher_pretrained_path', type=str, default='',
                        help="多模态教师模型 Backbone 预训练权重路径 (如果不传，将使用默认命名格式在 pretrained_checkpoint 目录下检索)")
    
    # 3. 下游微调时，初始化 Student 基础 Backbone 的预训练权重路径
    parser.add_argument('--student_pretrained_path', type=str, default='',
                        help="多模态学生模型 Backbone 预训练权重路径 (若无预训练权重，可留空，则从头随机初始化)")

    return parser.parse_args()

def main():
    args = parse_args()
    
    # 规范化预加载设备格式为 list
    if args.preload_devices is not None and len(args.preload_devices) == 0:
        args.preload_devices = None
        
    setup_seed(args.seed)
    
    # 打印核心配置信息
    print("=" * 60)
    print(f"运行模式: {args.mode.upper()}")
    print(f"生理对齐模式: {args.alignment_mode}")
    print(f"计算设备: {args.device}")
    print(f"LoRA 配置: {'启用 (Rank=' + str(args.trans_r) + ')' if args.ranklist == 'lora' else '禁用 (全参数微调)'}")
    print("=" * 60)

    # 路径安全保障
    checkpoint_dir = os.path.join(args.root, 'pretrained_checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(args.root, 'results'), exist_ok=True)

    if args.mode == 'pretrain':
        # ==================== 多模态联合预训练 ====================
        print("启动 EPHNOGRAM 联合多模态对比学习 + 生理对齐预训练...")
        
        # 检验单模态 Backbone 是否指定
        if not args.ecg_unimodal_pretrained or not args.pcg_unimodal_pretrained:
            print("[警告] 未提供 ECG/PCG 单模态预训练权重路径。模型将直接使用随机初始化进行多模态联合预训练。")
            
        run_multimodal_joint_pretraining(
            args=args,
            ecg_pretrained_path=args.ecg_unimodal_pretrained,
            pcg_pretrained_path=args.pcg_unimodal_pretrained
        )

    elif args.mode == 'finetune_teacher':
        # ==================== 教师网络微调 ====================
        print(f"启动教师网络微调 (Dataset: {args.ft_dataset}) ...")
        
        # 默认权重检索逻辑
        if not args.teacher_pretrained_path:
            prefix = "Hetero" if args.is_hetero else "Homo"
            default_path = os.path.join(checkpoint_dir, f"{prefix}_teacher_pretrained.pt")
            if os.path.exists(default_path):
                args.teacher_pretrained_path = default_path
            else:
                # 尝试检索联合预训练产出的权重
                joint_pretrain_name = f"{args.model_config}_EPHNOGRAM_JointPretrain_{args.alignment_mode}_seed{args.seed}_checkpoint.pt"
                joint_path = os.path.join(checkpoint_dir, joint_pretrain_name)
                if os.path.exists(joint_path):
                    args.teacher_pretrained_path = joint_path

        multimodal_kd_teacher_model(args=args, is_hetero=args.is_hetero)

    elif args.mode == 'finetune_student':
        # ==================== 知识蒸馏微调学生 ====================
        print(f"启动知识蒸馏微调 (Dataset: {args.ft_dataset}) ...")
        
        # 蒸馏通常使用一个固定的折数 (例如 fold 0)
        fold_idx = 0
        test_res = multimodal_kd_student_model(args=args, fold_idx=fold_idx, is_hetero=args.is_hetero)
        
        # 保存学生网络最终测试指标
        prefix = "Hetero" if args.is_hetero else "Homo"
        res_save_path = os.path.join(args.root, 'results', f"{args.ft_dataset}_Student_KD_{prefix}_seed{args.seed}_test_result.json")
        serializable_res = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in test_res.items()}
        with open(res_save_path, 'w') as f:
            json.dump(serializable_res, f, indent=4)
        print(f"学生网络蒸馏微调结束，最终测试结果已保存至: {res_save_path}")

if __name__ == '__main__':
    main()