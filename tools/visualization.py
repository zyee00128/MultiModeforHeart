# -*- coding: utf-8 -*-
"""
Created on 2026
@author: ECG Interpretability Visualizer (Headless & Save-only Edition)

"""

import numpy as np
import matplotlib
# 强制使用 Agg 后端，确保在无图形界面的服务器环境下也能稳定运行并保存图片
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# 设置全局绘图基础风格
sns.set_theme(style="white")


def plot_kernel_adaptation_heatmap(weights, labels, class_names=None, save_path="kernel_adaptation_heatmap.png"):
    """
    动态卷积核空间分布热力图 (Kernel Adaptation Heatmap)
    证明 LKP（动态卷积）生成的权重是根据输入信号的病理特征进行动态自适应调整的。
    
    Args:
        weights (np.ndarray): 维度为 [样本数, 卷积核尺寸]
        labels (np.ndarray): 维度为 [样本数, 类别数] 或 [样本数]
        class_names (list of str): 类别名称列表
        save_path (str): 图像保存路径，默认为 'kernel_adaptation_heatmap.png'
    """
    if labels.ndim == 1:
        num_classes = len(np.unique(labels))
        one_hot = np.zeros((len(labels), num_classes))
        one_hot[np.arange(len(labels)), labels] = 1
        labels = one_hot
        
    num_classes = labels.shape[1]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
        
    kernel_size = weights.shape[1]
    avg_kernel = np.zeros((num_classes, kernel_size))
    
    for c in range(num_classes):
        class_mask = labels[:, c] == 1
        if np.any(class_mask):
            avg_kernel[c] = np.mean(weights[class_mask], axis=0)
        else:
            avg_kernel[c] = np.mean(weights, axis=0)
            
    plt.figure(figsize=(10, 5), dpi=300)
    sns.heatmap(
        avg_kernel, 
        cmap="coolwarm", 
        center=0.0, 
        annot=True, 
        fmt=".3f",
        xticklabels=[f"Tap {i+1}" for i in range(kernel_size)],
        yticklabels=class_names,
        linewidths=0.5,
        cbar_kws={'label': 'Generated Kernel Weights ($w_i$)'}
    )
    plt.title("Dynamic LKP Kernel Weights Profile across ECG Pathologies", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Kernel Spatial/Temporal Dimension", fontsize=11)
    plt.ylabel("ECG Pathologies", fontsize=11)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    # 仅执行保存，并彻底释放内存
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_feature_enhancement(original_signal, feat_before_ska, feat_after_ska, save_path="feature_enhancement.png"):
    """
    SKA 特征增强细节对比图 (Feature Enhancement Map)
    展示在 ECG 关键波群区（如 QRS 复合波、P 波），经过 SKA 聚合前后的特征信噪比与边缘锐度变化。
    
    Args:
        original_signal (np.ndarray): [时间步], 原始单通道或单导联 ECG 信号
        feat_before_ska (np.ndarray): [时间步], 经过 SKA 聚合前的静态特征
        feat_after_ska (np.ndarray): [时间步], 经过 SKA 聚合强化后的动态特征
        save_path (str): 图像保存路径，默认为 'feature_enhancement.png'
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, dpi=300)
    x = np.arange(len(original_signal))
    
    axes[0].plot(x, original_signal, color="#2c3e50", linewidth=1.5, label="Raw ECG Signal")
    axes[0].set_ylabel("Amplitude", fontsize=11)
    axes[0].legend(loc="upper right")
    axes[0].set_title("SKA Feature Enhancement Analysis", fontsize=14, fontweight='bold')
    axes[0].grid(True, linestyle=":", alpha=0.6)
    
    axes[1].plot(x, feat_before_ska, color="#e74c3c", linewidth=1.5, label="Before SKA (Static Conv)")
    axes[1].fill_between(x, feat_before_ska, alpha=0.15, color="#e74c3c")
    axes[1].set_ylabel("Activation", fontsize=11)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    
    axes[2].plot(x, feat_after_ska, color="#2ecc71", linewidth=1.5, label="After SKA (Dynamic Aggregation)")
    axes[2].fill_between(x, feat_after_ska, alpha=0.15, color="#2ecc71")
    axes[2].set_ylabel("Activation", fontsize=11)
    axes[2].set_xlabel("Time (Samples)", fontsize=11)
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    # 仅执行保存，并彻底释放内存
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_lead_importance(se_gates, labels, lead_names=None, class_names=None, save_path="lead_importance.png"):
    """
    导联重要性热力图 (Lead Importance Heatmap)
    展示偶数层 SE 模块作为自适应导联选择器，在不同疾病样本下的导联门控缩放权重。
    
    Args:
        se_gates (np.ndarray): 维度为 [样本数, 12] (来自 SE Block 的激活动态系数)
        labels (np.ndarray): 维度为 [样本数, 类别数]
        lead_names (list of str, optional): 12导联名称
        class_names (list of str, optional): 类别名称列表
        save_path (str): 图像保存路径，默认为 'lead_importance.png'
    """
    if lead_names is None:
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    if class_names is None:
        class_names = [f"Pathology {i}" for i in range(labels.shape[1])]
        
    num_classes = labels.shape[1]
    avg_weights = np.zeros((num_classes, len(lead_names)))
    
    for c in range(num_classes):
        class_mask = labels[:, c] == 1
        if np.any(class_mask):
            avg_weights[c] = np.mean(se_gates[class_mask], axis=0)
        else:
            avg_weights[c] = np.mean(se_gates, axis=0)
            
    plt.figure(figsize=(11, 5), dpi=300)
    sns.heatmap(
        avg_weights, 
        annot=True, 
        fmt=".3f", 
        cmap="YlGnBu", 
        xticklabels=lead_names, 
        yticklabels=class_names,
        linewidths=0.5,
        cbar_kws={'label': 'SE Gate Activation Weight'}
    )
    plt.title("Lead-level Self-Attention Filtering Matrix across Pathologies", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("ECG 12 Leads", fontsize=11)
    plt.ylabel("ECG Pathologies", fontsize=11)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    # 仅执行保存，并彻底释放内存
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_erf_growth(model_erf, homogeneous_erf, layer_names=None, save_path="erf_growth.png"):
    """
    逐层有效感受野扩展对比 (Layer-wise ERF Growth)
    展示 LSNet 奇偶交替架构相较于静态同构堆叠在感受野建模上的“阶梯式”扩大优势。
    
    Args:
        model_erf (list or np.ndarray): LSNet 各层的有效感受野范围大小
        homogeneous_erf (list or np.ndarray): 静态同构堆叠网络各层的有效感受野范围大小
        layer_names (list of str, optional): 层级名称列表
        save_path (str): 图像保存路径，默认为 'erf_growth.png'
    """
    sns.set_theme(style="whitegrid")
    layers = np.arange(1, len(model_erf) + 1)
    if layer_names is None:
        layer_names = [f"Layer {i}" for i in layers]
        
    plt.figure(figsize=(9, 5), dpi=300)
    
    plt.step(layers, model_erf, where='mid', label='LSNet (Alternating LKP & SKA + SE)', 
             linewidth=2.5, color='#1f77b4', marker='o')
    plt.plot(layers, homogeneous_erf, label='Homogeneous-LS Baseline (Stacked LSConv)', 
             linewidth=2.0, color='#ff7f0e', linestyle='--', marker='s')
    
    plt.title("Layer-wise Effective Receptive Field (ERF) Progression", fontsize=13, fontweight='bold')
    plt.xlabel("Network Layer Depth", fontsize=11)
    plt.ylabel("Effective Receptive Field Size (Samples)", fontsize=11)
    plt.xticks(layers, layer_names, rotation=30)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10, loc="upper left")
    
    plt.tight_layout()
    # 仅执行保存，并彻底释放内存
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_manifold_progression(features_by_stage, labels, class_names=None, save_path="manifold_progression.png"):
    """
    多阶段特征流形演进图 (t-SNE Manifold Progression)
    展示“交替机制”对类内特征一致性（类内距离收敛）和类间边界清晰化的渐进式提纯过程。
    
    Args:
        features_by_stage (dict): 阶段名与特征映射。键如 'Stage 1'，值为二维特征数组 [样本数, 特征维度]
        labels (np.ndarray): 维度为 [样本数, 类别数] 或 [样本数]
        class_names (list of str, optional): 类别名称列表
        save_path (str): 图像保存路径，默认为 'manifold_progression.png'
    """
    stages = list(features_by_stage.keys())
    num_stages = len(stages)
    
    if labels.ndim > 1 and labels.shape[1] > 1:
        vis_labels = np.argmax(labels, axis=1)
    else:
        vis_labels = labels
        
    num_classes = len(np.unique(vis_labels))
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
        
    fig, axes = plt.subplots(1, num_stages, figsize=(5.5 * num_stages, 5), dpi=300)
    if num_stages == 1:
        axes = [axes]
        
    sns.set_theme(style="white")
    
    for i, stage_name in enumerate(stages):
        feat = features_by_stage[stage_name]
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feat) - 1))
        feat_2d = tsne.fit_transform(feat)
        
        ax = axes[i]
        scatter = ax.scatter(
            feat_2d[:, 0], feat_2d[:, 1],
            c=vis_labels, cmap="tab10", alpha=0.75, s=25, edgecolors='w', linewidths=0.2
        )
        ax.set_title(f"Manifold at {stage_name}", fontsize=13, fontweight='bold')
        ax.set_xlabel("t-SNE 1", fontsize=9)
        ax.set_ylabel("t-SNE 2", fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle="--", alpha=0.4)
        
    handles, _ = scatter.legend_elements()
    fig.legend(handles, class_names[:num_classes], loc='lower center', ncol=min(num_classes, 6), 
               bbox_to_anchor=(0.5, -0.06), fontsize=10)
    
    plt.tight_layout()
    # 仅执行保存，并彻底释放内存
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()