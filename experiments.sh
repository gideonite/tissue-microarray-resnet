#!/bin/bash

# CUDA_VISIBLE_DEVICES=0; python /mnt/code/main_pure.py --experiment_name norelu_imgs=3 --num_images 1 --num_epochs 20 --batch_size 32 
# CUDA_VISIBLE_DEVICES=0; python /mnt/code/main_pure.py --experiment_name norelu_imgs=5 --num_images 5 --num_epochs 20 --batch_size 32 
# CUDA_VISIBLE_DEVICES=0; python /mnt/code/main_pure.py --experiment_name norelu_imgs=10 --num_images 10 --num_epochs 20 --batch_size 32 
# CUDA_VISIBLE_DEVICES=0; python /mnt/code/main_pure.py --experiment_name norelu_imgs=50 --num_images 50 --num_epochs 50 --batch_size 32 
# CUDA_VISIBLE_DEVICES=0; python /mnt/code/main_pure.py --experiment_name norelu_imgs=100 --num_images 100 --num_epochs 50 --batch_size 32
# CUDA_VISIBLE_DEVICES=0; python /mnt/code/main_pure.py --experiment_name norelu_imgs=all --num_images -1 --num_epochs 50 --batch_size 32

# CUDA_VISIBLE_DEVICES=0; python /mnt/code/main_pure.py --experiment_name norelu_patchsize=16 --num_images -1 --patch_size 16 --stride 8 --num_epochs 50 --batch_size 32 
# CUDA_VISIBLE_DEVICES=0; python /mnt/code/main_pure.py --experiment_name norelu_patchsize=32 --num_images -1 --patch_size 32 --stride 16 --num_epochs 50 --batch_size 32 
CUDA_VISIBLE_DEVICES=3; python /mnt/code/main_pure.py --experiment_name norelu_patchsize=64_dup --num_images -1 --patch_size 64 --stride 32 --num_epochs 50 --batch_size 16 
CUDA_VISIBLE_DEVICES=3; python /mnt/code/main_pure.py --experiment_name norelu_patchsize=128_dup --num_images -1 --patch_size 128 --stride 64 --num_epochs 50 --batch_size 16
