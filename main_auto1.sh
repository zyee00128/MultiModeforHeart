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

echo "=============================================================================="
echo -e "\n>>> [Phase 1/3] 开始进行知识蒸馏超参数网格搜索..."

# 定义待搜索的模式、温度和 alpha 的参数网格
MODES=("hetero" "homo")
TEMPS=(2.0 5.0 8.0 10.0)
ALPHAS=(0.3 0.5 0.7 0.9)

for mode in "${MODES[@]}"; do
  echo "============================================================"
  echo "        开始进行 [${mode}] 模式下的超参数搜索"
  echo "============================================================"
  
  for temp in "${TEMPS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
      echo "------------------------------------------------------------"
      echo " 运行网格搜索任务: mode=${mode} | kd_temperature=${temp} | kd_alpha=${alpha}"
      echo "------------------------------------------------------------"
      
      ${BASE_CMD} \
        --mode "${mode}" \
        --ranklist lora \
        --kd_temperature "${temp}" \
        --kd_alpha "${alpha}" \
        --kfold ${KFOLD}
        
      if [ $? -ne 0 ]; then
        echo "[Error] 网格搜索任务失败: mode=${mode}, temp=${temp}, alpha=${alpha}"
        exit 1
      fi
    done
  done
done

echo ">>> 网格搜索结束。请检查 '${ROOT_DIR}/results' 目录下的 JSON 结果，确定最佳的 T 和 alpha。"
echo "=============================================================================="
echo -e "\n=============================================================================="
echo "                   所有自动化实验任务已全部执行完毕！                           "
echo "=============================================================================="