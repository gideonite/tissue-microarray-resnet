#!/bin/bash
python /mnt/code/main_pure.py --experiment_name imgs=1 --num_images 1 --num_epochs 20 &&
python /mnt/code/main_pure.py --experiment_name imgs=5 --num_images 5 --num_epochs 20 &&
python /mnt/code/main_pure.py --experiment_name imgs=10 --num_images 10 --num_epochs 20 &&
python /mnt/code/main_pure.py --experiment_name imgs=50 --num_images 50 --num_epochs 50 &&
python /mnt/code/main_pure.py --experiment_name imgs=100 --num_images 100 --num_epochs 50 &&
python /mnt/code/main_pure.py --experiment_name imgs=all --num_images -1 --num_epochs 50 &&
python /mnt/code/main_pure.py --experiment_name patchsize=16 --num_images -1 --patch_size 16 --num_epochs 50 &&
python /mnt/code/main_pure.py --experiment_name patchsize=32 --num_images -1 --patch_size 32 --num_epochs 50 &&
python /mnt/code/main_pure.py --experiment_name patchsize=64 --num_images -1 --patch_size 64 --num_epochs 50 &&
python /mnt/code/main_pure.py --experiment_name patchsize=128 --num_images -1 --patch_size 128 --num_epochs 50 --batch_size 32
