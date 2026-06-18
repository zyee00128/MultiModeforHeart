ROOT_DIR="/home/xcy/zy/LSTrans"
DATASET_DIR="/data/xcy_group/zy/Data/G12EC_PTBXL_Ningbo_Chapman"
PRETRAIN_DIR="/data/xcy_group/zy/Data/Code15_preprocessing"
DEVICE="cuda:5"
KFOLD=10

# 基础训练命令模板
BASE_CMD="python main_ecg.py \
  --root ${ROOT_DIR} \
  --dataset_dir ${DATASET_DIR} \
  --pretrain_dataset ${PRETRAIN_DIR} \
  --device ${DEVICE} \
  --preload_devices cuda:3 cuda:2 cuda:1 cuda:0 \
  --patience 50"

# ==============================================================================
# 3. 第二阶段：模式与微调方式交叉组合实验 (Core Combination Matrix)
#    在确定最佳参数后，执行四种核心组合实验：
#    (1) Homo + LoRA (2) Homo + Full FT (3) Hetero + LoRA (4) Hetero + Full FT
# ==============================================================================

BEST_TEMP=5.0
BEST_ALPHA=0.5
echo "=============================================================================="
echo -e "\n>>> [Phase 2/3] 开始进行 4 种核心组合完整实验 (使用最佳参数: T=${BEST_TEMP}, alpha=${BEST_ALPHA})..."

MODES=("homo" "hetero")
FINETUNES=("lora" "ft")

for mode in "${MODES[@]}"; do
  for ft in "${FINETUNES[@]}"; do
    echo "------------------------------------------------------------"
    echo " 运行核心组合任务: mode=${mode} | ranklist=${ft}"
    echo "------------------------------------------------------------"
    
    ${BASE_CMD} \
      --mode "${mode}" \
      --ranklist "${ft}" \
      --kd_temperature "${BEST_TEMP}" \
      --kd_alpha "${BEST_ALPHA}" \
      --kfold ${KFOLD}
      
    if [ $? -ne 0 ]; then
      echo "[Error] 核心组合任务失败: mode=${mode}, ranklist=${ft}"
      exit 1
    fi
  done
done

echo ">>> 四种核心组合完整实验运行完毕。"
echo "=============================================================================="

# ==============================================================================
# 4. 第三阶段：消融实验 (Ablation Study)
#    基于 Hetero+ LoRA 基线与确定的最佳蒸馏参数，对奇数层与偶数层组件进行消融
# ==============================================================================
echo -e "\n>>> [Phase 3/3] 开始运行消融实验 (基于 Homo + LoRA + 最佳蒸馏参数)..."

# --- 4.1 奇数层组件消融 (Odd Layers Components Ablation) ---
# a2: 静态聚合 (LKP + Static-DWConv)
# a3: 静态感知 (Static-LargeConv + SKA)
# a4: 同构堆叠 (Stacked StdConv 3x3)
ODD_ABLATIONS=("a2" "a3" "a4")

for odd in "${ODD_ABLATIONS[@]}"; do
  echo "------------------------------------------------------------"
  echo " 运行奇数层消融任务: ablation_odd=${odd} (Even=none)"
  echo "------------------------------------------------------------"
  
  ${BASE_CMD} \
    --mode hetero \
    --ranklist lora \
    --ablation_odd "${odd}" \
    --ablation_even none \
    --kd_temperature "${BEST_TEMP}" \
    --kd_alpha "${BEST_ALPHA}" \
    --kfold ${KFOLD}
    
  if [ $? -ne 0 ]; then
    echo "[Error] 奇数层消融实验失败: odd=${odd}"
    exit 1
  fi
done

# --- 4.2 偶数层与先验组件消融 (Even Layers / Prior Components Ablation) ---
# b1: 静态同构堆叠 (LSConv + LSConv stacked)
# b2: 无 RepVGG 旁支 (LSConv + DWConv 3x3 + SE)
# b3: 无 SE 门控 (LSConv + RepVGGDW)
EVEN_ABLATIONS=("b1" "b2" "b3")

for even in "${EVEN_ABLATIONS[@]}"; do
  echo "------------------------------------------------------------"
  echo " 运行偶数层消融任务: ablation_even=${even} (Odd=none)"
  echo "------------------------------------------------------------"
  
  ${BASE_CMD} \
    --mode hetero \
    --ranklist lora \
    --ablation_odd none \
    --ablation_even "${even}" \
    --kd_temperature "${BEST_TEMP}" \
    --kd_alpha "${BEST_ALPHA}" \
    --kfold ${KFOLD}
    
  if [ $? -ne 0 ]; then
    echo "[Error] 偶数层消融实验失败: even=${even}"
    exit 1
  fi
done

echo -e "\n=============================================================================="
echo "                   所有自动化实验任务已全部执行完毕！                           "
echo "=============================================================================="