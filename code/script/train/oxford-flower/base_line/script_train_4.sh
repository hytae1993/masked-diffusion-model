#!/bin/bash        

# ==============================
task="train"
dir_work="/nas/users/hyuntae/code/doctor/masked-diffusion-model"
dir_dataset="/nas2/dataset/hyuntae"
data_name="oxford-flower"
data_set="train"
data_size=64
date=""
time=""
title="test"
# ==============================
model=default
batch_size=16
num_epochs=100
optim="adam"
lr=1e-4
lr_scheduler="cosine"
lr_warmup_steps=500
gradient_accumulation_steps=1
sample_num=100
sample_epoch_ratio=0.2
resume_from_checkpoint=False
num_workers=32
use_ema=True
ema_inv_gamma=1.0
ema_power=0.75
ema_max_decay=0.9999



mixed_precision="fp16"
ddpm_num_steps=1000
ddpm_beta_schedule="linear"
checkpointing_steps=500
save_images_epochs=1
save_images_batch=100
save_loss=1
ddpm_num_inference_steps=1000

device="2"



# ==============================
list_iter=(0)
echo ${list_iter[@]}
# list_iter=`seq 10 10 1000`
# list_tier=0
# echo ${list_iter}
# ==============================
hostname=$HOSTNAME
time_stamp=`date +"%Y-%m-%d-%T"`
code=${dir_work}"/code/main_train_baseline.py"
# ==============================
<<comment
comment
# ==============================
for iter in ${list_iter[@]}
do 
    echo "${hostname}.${task}.${data_name}.${time_stamp}.log"
    accelerate launch --config_file '/nas/users/hyuntae/code/doctor/masked-diffusion-model/code/script/train/config/gpu3_config.yaml' ${code} \
        --task=${task} \
        --dir_work=${dir_work} \
        --dir_dataset=${dir_dataset} \
        --data_name=${data_name} \
        --data_set=${data_set} \
        --data_size=${data_size} \
        --date=${date} \
        --time=${time} \
        --title=${title} \
        --model=${model} \
        --batch_size=${batch_size} \
        --num_epochs=${num_epochs} \
        --optim=${optim} \
        --lr=${lr} \
        --lr_scheduler=${lr_scheduler} \
        --lr_warmup_steps=${lr_warmup_steps} \
        --gradient_accumulation_steps=${gradient_accumulation_steps} \
        --sample_num=${sample_num} \
        --sample_epoch_ratio=${sample_epoch_ratio} \
        --resume_from_checkpoint=${resume_from_checkpoint} \
        --num_workers=${num_workers} \
        --use_ema=${use_ema} \
        --ema_inv_gamma=${ema_inv_gamma} \
        --ema_power=${ema_power} \
        --ema_max_decay=${ema_max_decay} \
        --mixed_precision=${mixed_precision} \
        --ddpm_num_steps=${ddpm_num_steps} \
        --ddpm_beta_schedule=${ddpm_beta_schedule} \
        --checkpointing_steps=${checkpointing_steps} \
        --save_images_epochs=${save_images_epochs} \
        --save_images_batch=${save_images_batch} \
        --save_loss=${save_loss} \
        --ddpm_num_inference_steps=${ddpm_num_inference_steps} \
        --device=${device} \
        # > ${hostname}.${task}.${data_name}.${time_stamp}.log
done