#!/bin/bash
#SBATCH --job-name=filter_heads
#SBATCH --time=2-00:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=gnodee1,gnodeg3,gnodec2
#SBATCH --array=12,16,39
#SBATCH --constraint=gmem24G
##SBATCH --mail-user=okupyn@robots.ox.ac.uk
##SBATCH --mail-type=START,END,FAIL,ARRAY_TASKS
pwd; hostname; date
nvidia-smi
source ~/.bashrc
date +"%R activating conda env"
conda activate head_det
date +"%R starting script"
cd /users/okupyn/head_detector/data_generator || exit
export PYTHONPATH=..
python filter_pipeline.py /work/okupyn/VGGHeadNew/large /users/okupyn/RT-DETR/rtdetr_pytorch/model.onnx /work/okupyn/VGGHeadsViz

date +"%R slurm job done"
