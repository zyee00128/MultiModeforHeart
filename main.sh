/home/xcy/yes/envs/lsm/bin/python /home/xcy/zy/LSTrans/main_ecg.py \
    --tea_ranklist lora_ave \
    --model_config light \
    --device cuda:5 \
    --preload_devices cuda:5 cuda:3 cuda:2 cuda:1 cuda:0\
    --batch_size 64 \
    --conv_r 4 \
    --trans_r 16 \
    --patience 30 \
    --kd_temperature 9.0 \
    --kd_alpha 0.9 \
    --task kfold_exp

/home/xcy/yes/envs/lsm/bin/python /home/xcy/zy/LSTrans/main_pcg.py \
    --model_config large \
    --device cuda:4 \
    --batch_size 32 \
    --task pretrain \
    --patience 50 \
    --model_arch LSTrans