# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:39:27 2022

@author: COCHE User
"""

from sklearn.metrics import roc_auc_score, hamming_loss,label_ranking_loss,accuracy_score,precision_recall_curve
from sklearn.metrics import coverage_error
import numpy as np

def challenge_metrics(y_true, y_pred, beta1=2, beta2=2, class_weights=None, single=True):
    f_beta = 0
    g_beta = 0
    if single: # if evaluating single class in case of threshold-optimization
        sample_weights = np.ones(y_true.sum(axis=1).shape)
    else:
        sample_weights = y_true.sum(axis=1)
    sample_weights = np.where(sample_weights == 0, 1.0, sample_weights)

    f_beta_each_class = []
    g_beta_each_class = []
    num_classes = y_true.shape[1]

    for classi in range(num_classes):
        y_truei, y_predi = y_true[:, classi], y_pred[:, classi]
        
        tp_mask = (y_truei == 1) & (y_predi == 1)
        fp_mask = (y_predi == 1) & (y_truei != y_predi)
        tn_mask = (y_truei == 0) & (y_predi == 0)
        fn_mask = (y_predi == 0) & (y_truei != y_predi)
        
        TP = np.sum(1.0 / sample_weights[tp_mask]) if np.any(tp_mask) else 0.0
        FP = np.sum(1.0 / sample_weights[fp_mask]) if np.any(fp_mask) else 0.0
        TN = np.sum(1.0 / sample_weights[tn_mask]) if np.any(tn_mask) else 0.0
        FN = np.sum(1.0 / sample_weights[fn_mask]) if np.any(fn_mask) else 0.0
        
        denom_f = (1 + beta1**2) * TP + FP + (beta1**2) * FN
        f_beta_i = ((1 + beta1**2) * TP) / denom_f if denom_f > 0 else 0.0
        denom_g = TP + FP + beta2 * FN
        g_beta_i = TP / denom_g if denom_g > 0 else 0.0
        
        f_beta += f_beta_i
        g_beta += g_beta_i
        f_beta_each_class.append(f_beta_i)
        g_beta_each_class.append(g_beta_i)
        
    return f_beta / num_classes, g_beta / num_classes, f_beta_each_class, g_beta_each_class

def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i

def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean(), 100*ap

def evaluation(label,predict,thres=0.5):
    ## logit-based
    auc,auc_each_class=roc_auc_score(label,predict,average='macro'),roc_auc_score(label,predict,average=None)
    rankingloss=label_ranking_loss(label,predict)
    Coverage=coverage_error(label,predict)
    Map_value,map_each_class=mAP(label,predict)

    ## one-hot-based
    if np.isscalar(thres):
        thres = np.ones(predict.shape[1]) * thres
    predict_bin = np.zeros_like(predict)
    for i in range(predict.shape[1]):
        predict_bin[:, i] = (predict[:, i] > thres[i]).astype(np.float32)

    hammingloss = hamming_loss(label, predict_bin)
    acc = accuracy_score(label, predict_bin)
    F1score_b, Gscore_b, f_beta_each_class, g_beta_each_class = challenge_metrics(label, predict_bin)   
    F1score, _, f_each_class, _ = challenge_metrics(label, predict_bin, beta1=1, beta2=1)

    performance_table = {
        'auc': auc, 'ranking': rankingloss, 'hamming': hammingloss, 'acc': acc, 
        'F1score_b': F1score_b, 'Gscore_b': Gscore_b, 'Map_value': Map_value, 'Coverage': Coverage,
        'auc_class': auc_each_class, 'map_class': map_each_class,
        'F1score_b_class': f_beta_each_class, 'Gscore_b_class': g_beta_each_class,
        'F1score': F1score, 'F1score_class': f_each_class
    }
    return performance_table

def print_result(loss, label, predict, datatype, thres = None):
    """
    Args:
        loss: float, 阶段损失值 (BCE loss 等)
        label: np.ndarray, 真实标签 (Ground Truth) [样本数, 类别数]
        predict: np.ndarray, 模型预测概率 (Sigmoid logits) [样本数, 类别数]
        datatype: str, 数据集类型 (如 'train', 'valid', 'test')
        thres: np.ndarray, 决策二值化阈值。如果未传入，默认使用 0.5 作为全局阈值
    """
    if thres is None:
        thres = 0.5 * np.ones(label.shape[1])
    performance_table = evaluation(label, predict, thres=thres)
    performance_table.update({'yloss': loss})
    performance_table.update({'threshold': thres})
    
    # # 开始美化打印
    # print("=" * 70)
    # print(f"  [EVALUATION REPORT] - {datatype.upper()} PHASE ")
    # print("=" * 70)
    
    # # ---------------------------------------------------------
    # # 类别 1: 基础损失与辅助配置指标
    # # ---------------------------------------------------------
    # print("\n>>> [1. Loss & Configurations / 基础损失与辅助配置]")
    # print(f"  - Epoch Loss (损失值):             {loss:.6f}")
    # print(f"  - Decision Thresholds (决策阈值):  {np.array2string(thres, precision=4, separator=', ')}")
    
    # # ---------------------------------------------------------
    # # 类别 2: 连续概率与排序评估指标 (直接由 Logit 计算，不依赖硬二值化阈值)
    # # ---------------------------------------------------------
    # print("\n>>> [2. Probability & Ranking Metrics / 连续概率与排序评估 (不依赖阈值)]")
    # # Macro AUC: 宏观受试者工作特征曲线下面积。反映模型对各类别区分能力的平均水平。
    # print(f"  - Macro ROC-AUC:        {performance_table['auc']:.4f}")
    
    # # Mean AP (mAP): 平均精准度。评估多标签排序推荐能力，对正样本召回位置敏感。
    # print(f"  - Mean Average Precision:   {performance_table['Map_value']:.2f}%")
    
    # # Label Ranking Loss: 标签排序损失。度量排序错误的标签对比例，值越低代表预测真实的标签越靠前。
    # print(f"  - Label Ranking Loss:   {performance_table['ranking']:.4f}")
    
    # # Coverage Error: 覆盖误差。平均需要预测前多少个高置信度标签才能完整覆盖所有真实标签。
    # print(f"  - Coverage Error:        {performance_table['Coverage']:.4f}")
    # # ---------------------------------------------------------
    # # 类别 3: 硬决策分类指标 (由概率通过指定阈值二值化后计算)
    # # ---------------------------------------------------------
    # print("\n>>> [3. Binary Classification Metrics / 硬决策二分类指标 (依赖阈值二值化)]")
    # # Subset Accuracy: 子集准确率（样本完全匹配率）。多标签场景下，必须所有标签全对才算正确，指标最严苛。
    # print(f"  - Subset Accuracy:    {performance_table['acc']:.4f}")
    
    # # Hamming Loss: 汉明损失。错分标签比例（包括漏诊和误诊），对标签稀疏任务较敏感。越低越好。
    # print(f"  - Hamming Loss: {performance_table['hamming']:.4f}")
    
    # # Macro F1-score: 传统的宏观 F1 分数 (beta=1)，精准率与召回率的等权重调和平均。
    # print(f"  - Macro F1-score:  {performance_table['F1score']:.4f}")
    
    # # Macro F_beta-score (默认 beta=2): 加权 F 分数，更侧重于召回率 (Recall)，符合心电图疾病筛查防漏诊的要求。
    # print(f"  - Macro F_beta-score: {performance_table['F1score_b']:.4f}")
    
    # # G_beta-score: Challenge赛事指标。综合权衡了漏诊(FN)和误诊(FP)非平衡代价的乘积项评估。
    # print(f"  - Macro G_beta-score:   {performance_table['Gscore_b']:.4f}")
    
    # # ---------------------------------------------------------
    # # 类别 4: 逐类别细分指标 (辅助分析各类心电异常的特征捕捉情况)
    # # ---------------------------------------------------------
    # print("\n>>> [4. Class-wise Detailed Metrics / 逐类别细分评估]")
    # num_classes = label.shape[1]
    # for c in range(num_classes):
    #     print(f"  Class {c:02d}:")
    #     print(f"    * Threshold (分类阈值):    {thres[c]:.4f}")
    #     print(f"    * ROC-AUC (曲线下面积):   {performance_table['auc_class'][c]:.4f}")
    #     print(f"    * Average Precision (AP): {performance_table['map_class'][c]:.2f}%")
    #     print(f"    * F1-score (F1 标准值):    {performance_table['F1score_class'][c]:.4f}")
    #     print(f"    * F_beta-score (F_beta):  {performance_table['F1score_b_class'][c]:.4f}")
    #     print(f"    * G_beta-score (G_beta):  {performance_table['Gscore_b_class'][c]:.4f}")
        
    # print("=" * 70 + "\n")
    
    return performance_table

def find_thresholds(label, predict, beta=2):
    N = label.shape[1]
    f1prcT = np.zeros((N,))
    for j in range(N):
        prc, rec, thr = precision_recall_curve(y_true=label[:, j], y_score=predict[:, j])
        if len(thr) == 0:
            f1prcT[j] = 0.5
            continue
        fscore = (1 + beta**2) * prc * rec / ((beta**2) * prc + rec)
        idx = np.nanargmax(fscore)
        # 修复 Scikit-Learn中 thr 长度比 prc 长度少 1 的越界 Bug
        if idx >= len(thr):
            idx = len(thr) - 1
        f1prcT[j] = thr[idx]
    return f1prcT

    
    
    
    
    
    
    
    