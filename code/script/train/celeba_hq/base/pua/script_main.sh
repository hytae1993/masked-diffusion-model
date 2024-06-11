#!/bin/bash        

# ==============================
use_wandb=True
use_mlflow=False
task="train"
content="code_progress"
dir_work="/nas/users/hyuntae/code/doctor/masked-diffusion-model"
dir_dataset="/ssd1/dataset/"
data_name="celeba_hq"
data_set="train"
data_size=64
data_subset=True
data_subset_num=1024
date=""
time=""
method="mean_shift"
title="all-area-constant-shift_1024-image_input-mean-original-demean-to-0_lr-3e-5_celeba_hq"
# ==============================
model=default
batch_size=32
in_channel=3
out_channel=3
num_attention=5
num_epochs=500001
optim="adamw"
lr=3e-5
lr_scheduler="cosine"
lr_warmup_steps=0
lr_cycle=100.5
gradient_accumulation_steps=1
mixed_precision="no"
# ==============================
use_ema=True
ema_inv_gamma=1.0
ema_power=0.75
ema_max_decay=0.9999
loss_weight_use=False
loss_weight_power_base=2.0
loss_space="x_0"
ddpm_num_steps=16
ddpm_schedule="log"
ddpm_schedule_base=10.5
scheduler_num_scale_timesteps=1
sampling="momentum"
sampling_mask_dependency="independent"
mean_option="degraded_area"
mean_area="image-wise"
mean_value_accumulate=False
shift_type="constant"
noise_mean=0
# ==============================
sample_num=100
sample_epoch_ratio=0.2
resume_from_checkpoint=False
num_workers=1
checkpointing_steps=10000
save_images_epochs=5000


# ==============================
list_iter=(0)
echo ${list_iter[@]}
# list_iter=`seq 10 10 1000`
# list_tier=0
# echo ${list_iter}
# ==============================
hostname=$HOSTNAME
time_stamp=`date +"%Y-%m-%d-%T"`
code=${dir_work}"/code/main_train_masked.py"
# ==============================
<<comment
comment
# ==============================
for iter in ${list_iter[@]}
do 
    echo "${hostname}.${task}.${data_name}.${time_stamp}.log"
    python -u -m accelerate.commands.launch --config_file '/nas/users/hyuntae/code/doctor/masked-diffusion-model/code/script/train/config/gpuMulti_config.yaml' ${code} \
        --use_wandb=${use_wandb} \
        --use_mlflow=${use_mlflow} \
        --task=${task} \
        --content=${content} \
        --dir_work=${dir_work} \
        --dir_dataset=${dir_dataset} \
        --data_name=${data_name} \
        --data_set=${data_set} \
        --data_size=${data_size} \
        --data_subset=${data_subset} \
        --data_subset_num=${data_subset_num} \
        --date=${date} \
        --time=${time} \
        --method=${method} \
        --title=${title} \
        --model=${model} \
        --in_channel=${in_channel} \
        --out_channel=${out_channel} \
        --num_attention=${num_attention} \
        --batch_size=${batch_size} \
        --num_epochs=${num_epochs} \
        --optim=${optim} \
        --lr=${lr} \
        --lr_scheduler=${lr_scheduler} \
        --lr_warmup_steps=${lr_warmup_steps} \
        --lr_cycle=${lr_cycle} \
        --gradient_accumulation_steps=${gradient_accumulation_steps} \
        --sample_num=${sample_num} \
        --sample_epoch_ratio=${sample_epoch_ratio} \
        --resume_from_checkpoint=${resume_from_checkpoint} \
        --num_workers=${num_workers} \
        --use_ema=${use_ema} \
        --ema_inv_gamma=${ema_inv_gamma} \
        --ema_power=${ema_power} \
        --ema_max_decay=${ema_max_decay} \
        --loss_weight_use=${loss_weight_use} \
        --loss_weight_power_base=${loss_weight_power_base} \
        --mixed_precision=${mixed_precision} \
        --ddpm_num_steps=${ddpm_num_steps} \
        --updated_ddpm_num_steps=${ddpm_num_steps} \
        --ddpm_schedule=${ddpm_schedule} \
        --ddpm_schedule_base=${ddpm_schedule_base} \
        --checkpointing_steps=${checkpointing_steps} \
        --save_images_epochs=${save_images_epochs} \
        --scheduler_num_scale_timesteps=${scheduler_num_scale_timesteps} \
        --mean_value_accumulate=${mean_value_accumulate} \
        --mean_option=${mean_option} \
        --mean_area=${mean_area} \
        --shift_type=${shift_type} \
        --noise_mean=${noise_mean} \
        --sampling_mask_dependency=${sampling_mask_dependency} \
        --sampling=${sampling} \
        --loss_space=${loss_space} \
        # > ${hostname}.${task}.${data_name}.${time_stamp}.log
done