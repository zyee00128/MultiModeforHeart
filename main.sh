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

python main.py \
    --mode pretrain \
    --alignment_mode constraint \
    --lambda_physio 0.1 \
    --ecg_unimodal_pretrained ./pretrained_checkpoint/unimodal_ecg_12lead.pt \
    --pcg_unimodal_pretrained ./pretrained_checkpoint/unimodal_pcg_1channel.pt \
    --batch_size 64 \
    --ft_epoch 30

python main.py \
    --mode finetune_teacher \
    --ft_dataset cardiology2016 \
    --alignment_mode constraint \
    --ranklist lora \
    --conv_r 8 \
    --trans_r 32 \
    --batch_size 64 \
    --ft_epoch 50

python main.py \
    --mode finetune_student \
    --ft_dataset cardiology2016 \
    --alignment_mode constraint \
    --kd_temperature 3.0 \
    --kd_alpha 0.6 \
    --batch_size 64 \
    --ft_epoch 50