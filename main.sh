/home/xcy/yes/envs/lsm/bin/python /home/xcy/zy/LSTrans/main_ecg.py \
    --mode hetero \
    --ranklist lora \
    --leads_for_student 12 \
    --ablation_odd none \
    --ablation_even none \
    --device cuda:5 \
    --preload_devices cuda:3 cuda:2 cuda:1 cuda:0 \
    --seed 42 \
    --model_config light \
    --batch_size 32 \
    --learning_rate 0.001 \
    --patience 50 \
    --kd_temperature 5.0 \
    --kd_alpha 0.5 \
    --kfold 10

/home/xcy/yes/envs/lsm/bin/python /home/xcy/zy/LSTrans/main_pcg.py \
    --model_config large \
    --device cuda:4 \
    --batch_size 32 \
    --task pretrain \
    --patience 50 \
    --model_arch LSTrans


