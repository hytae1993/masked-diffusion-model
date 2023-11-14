#!/bin/bash        

# ==============================
task="train"
dir_work="/nas/users/hyuntae/code/doctor/segmentation-energy-based-model"
dir_dataset="/nas2/dataset/hyuntae"
data_name="DUTS"
data_set="train"
data_subset_use=True
data_subset_label=5
data_size=352
date=""
time=""
title="without_mse_of_gt_without_noise_one_lang_lr"
# ==============================
method="from_image"
architecture="ResNet"
channel_reduced_gen=32
latent_dim=8
channel_reduced_des=64
out_channel=1
batch_size=4
epoch_length=30
optim="adam"
lr_scheduler="Decay"
lr_generator_max=5e-5
lr_discriminator_max=1e-3
lr_decay_rate=0.9
lr_decay_epoch=20
dim_feature=8
weight_reg=0.001
langevin_length=3
langevin_lr=0.001
langevin_noise_lr=0
energy_form='identity' 
save_every=1
epoch_resume=1
num_workers=4
cuda_device=0

# ==============================
list_iter=(0)
echo ${list_iter[@]}
# list_iter=`seq 10 10 1000`
# list_tier=0
# echo ${list_iter}
# ==============================
hostname=$HOSTNAME
time_stamp=`date +"%Y-%m-%d-%T"`
code=${dir_work}"/code/main_train_saliency.py"
# ==============================
<<comment
comment
# ==============================
for iter in ${list_iter[@]}
do 
    echo "${hostname}.${task}.${data_name}.${time_stamp}.log"
    python ${code} \
        --task=${task} \
        --dir_work=${dir_work} \
        --dir_dataset=${dir_dataset} \
        --data_name=${data_name} \
        --data_set=${data_set} \
        --data_subset_use=${data_subset_use} \
        --data_subset_label=${data_subset_label} \
        --data_size=${data_size} \
        --date=${date} \
        --time=${time} \
        --title=${title} \
        --method=${method} \
        --architecture=${architecture} \
        --channel_reduced_gen=${channel_reduced_gen} \
        --latent_dim=${latent_dim} \
        --channel_reduced_des=${channel_reduced_des} \
        --dim_feature=${dim_feature} \
        --out_channel=${out_channel} \
        --batch_size=${batch_size} \
        --epoch_length=${epoch_length} \
        --optim=${optim} \
        --lr_scheduler=${lr_scheduler} \
        --lr_generator_max=${lr_generator_max} \
        --lr_discriminator_max=${lr_discriminator_max} \
        --lr_decay_rate=${lr_decay_rate} \
        --lr_decay_epoch=${lr_decay_epoch} \
        --weight_reg=${weight_reg} \
        --langevin_length=${langevin_length} \
        --langevin_lr=${langevin_lr} \
        --langevin_noise_lr=${langevin_noise_lr} \
        --energy_form=${energy_form} \
        --save_every=${save_every} \
        --epoch_resume=${epoch_resume} \
        --num_workers=${num_workers} \
        --cuda_device=${cuda_device} \
        # > ${hostname}.${task}.${data_name}.${time_stamp}.log
done