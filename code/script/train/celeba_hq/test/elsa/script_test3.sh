#!/bin/bash        

# ==============================
use_wandb=False
use_mlflow=False
task="train"
content="code_progress"
dir_work="/nas/users/hyuntae/code/doctor/masked-diffusion-model"
dir_dataset="/ssd1/dataset/"
data_name="celeba_hq"
data_set="train"
data_size=64
data_subset=True
data_subset_num=2048
date=""
time=""
method="test"
test_method="mean_shift"
title="mean-shift-train_base-momentum-sampling_100-steps_2048_data"
# ==============================
model=default
batch_size=32
in_channel=3
out_channel=3
num_attention=1
num_epochs=40001
optim="adam"
lr=3e-4
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
loss_weight_power_base=1000.0
loss_space="x_0"
ddpm_num_steps=100
ddpm_schedule="log"
ddpm_schedule_base=0.5
scheduler_num_scale_timesteps=1
# ==============================
sampling="momentum"
momentum_adaptive="base_momentum"
adaptive_decay_rate=0.999
adaptive_momentum_rate=0.999
sampling_mask_dependency="dependent"
mean_option="degraded_area"
mean_area="image-wise"
mean_value_accumulate=False
sample_num=32
sample_epoch_ratio=0.2
resume_from_checkpoint=False
num_workers=8
checkpointing_steps=1000
save_images_epochs=10000
# ========== for test ==========
test_model_path="/nas/users/hyuntae/code/doctor/masked-diffusion-model/result/code_progress/celeba_hq/mean_shift/2024_04_17_12_33_19/2048_momentum_100-step_log-base_loss-lr_3e-4-degrade_image_wise-celeba_hq/checkpoint/checkpoint-epoch-9999"

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
    accelerate launch --config_file '/nas/users/hyuntae/code/doctor/masked-diffusion-model/code/script/train/config/gpu3_config.yaml' ${code} \
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
        --test_method=${test_method} \
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
        --momentum_adaptive=${momentum_adaptive} \
        --adaptive_decay_rate=${adaptive_decay_rate} \
        --adaptive_momentum_rate=${adaptive_momentum_rate} \
        --mean_value_accumulate=${mean_value_accumulate} \
        --mean_option=${mean_option} \
        --mean_area=${mean_area} \
        --sampling_mask_dependency=${sampling_mask_dependency} \
        --sampling=${sampling} \
        --loss_space=${loss_space} \
        --test_model_path=${test_model_path} \
        # > ${hostname}.${task}.${data_name}.${time_stamp}.log
done