#!/bin/bash        

# ==============================
task="train"
dir_work="/nas/users/hyuntae/code/doctor/segmentation-energy-based-model"
dir_dataset="/nas2/dataset/dataset"
data_name="mnist"
data_set="train"
data_subset_use=True
data_subset_label=5
data_size=32
date="0"
time="0"
# ==============================
dim_latent=128
dim_feature=8
batch_size=625
epoch_length=500
optim="adamw"
lr_scheduler="CyclicLR"
lr_generator_min=0.001
lr_generator_max=0.001
lr_discriminator_min=0.001
lr_discriminator_max=0.001
weight_reg=0.001
langevin_length=10
langevin_lr=1
langevin_noise_lr=0
save_every=10
epoch_resume=0
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
code=${dir_work}"/code/main_train.py"
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
        --dim_latent=${dim_latent} \
        --dim_feature=${dim_feature} \
        --batch_size=${batch_size} \
        --epoch_length=${epoch_length} \
        --optim=${optim} \
        --lr_scheduler=${lr_scheduler} \
        --lr_generator_min=${lr_generator_min} \
        --lr_generator_max=${lr_generator_max} \
        --lr_discriminator_min=${lr_discriminator_min} \
        --lr_discriminator_max=${lr_discriminator_max} \
        --weight_reg=${weight_reg} \
        --langevin_length=${langevin_length} \
        --langevin_lr=${langevin_lr} \
        --langevin_noise_lr=${langevin_noise_lr} \
        --save_every=${save_every} \
        --epoch_resume=${epoch_resume} \
        --num_workers=${num_workers} \
        --cuda_device=${cuda_device} \
        # > ${hostname}.${task}.${data_name}.${time_stamp}.log
done