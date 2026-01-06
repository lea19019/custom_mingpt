#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J "gpt_training"
#SBATCH --output=logs/%x_%j.out
#SBATCH --qos=cs
#SBATCH --gres=gpu:1

mkdir -p logs

echo "========================================================="
echo "SLURM JOB ID: $SLURM_JOB_ID"
echo "JOB NAME: $SLURM_JOB_NAME"
echo "Running on: $SLURM_JOB_NODELIST"
echo "Flag: $FLAG"
echo "========================================================="

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /home/vacl2/advanced-deep-learning/transformer
source .venv/bin/activate

echo "--- Diagnostics ---"
nvidia-smi
python -c "import torch; print('Free GPU memory:', torch.cuda.mem_get_info()[0]/1e9, 'GB')"
echo "-------------------"

echo "Running with flag: $FLAG"
uv run python lab4.py $FLAG

echo "========================================================="
echo "Training finished"
echo "========================================================="