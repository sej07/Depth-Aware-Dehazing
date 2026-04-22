#!/bin/bash
#SBATCH --job-name=aodnet_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/barshikar.s/depth-aware-dehazing/experiments/aodnet_baseline/slurm_%j.out
#SBATCH --error=/home/barshikar.s/depth-aware-dehazing/experiments/aodnet_baseline/slurm_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

module load anaconda3/2024.06

source activate dehazing

mkdir -p /home/barshikar.s/depth-aware-dehazing/experiments/aodnet_baseline

cd /home/barshikar.s/depth-aware-dehazing

python scripts/train_aodnet.py \
    --data_dir /home/barshikar.s/depth-aware-dehazing/data/reside/SOTS \
    --split outdoor \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.0001 \
    --experiment_dir /home/barshikar.s/depth-aware-dehazing/experiments/aodnet_baseline

echo "End time: $(date)"