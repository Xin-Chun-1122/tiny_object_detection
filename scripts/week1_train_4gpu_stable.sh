#!/bin/bash
#SBATCH --job-name=week1_4gpu_stable
#SBATCH --account=ACD114147
#SBATCH --partition=gp1d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/u2249585/uav/logs/week1_stable_%j.out
#SBATCH --error=/home/u2249585/uav/logs/week1_stable_%j.err

echo "=========================================="
echo "Week 1 Baseline Training (Stable Env)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 4x V100-32GB"
echo "Start Time: $(date)"
echo "=========================================="

# 啟動穩定環境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmdet_stable

# 顯示環境資訊
echo ""
echo "Environment Info:"
python -c "
import torch
import mmcv
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'mmcv: {mmcv.__version__}')
print(f'GPU Count: {torch.cuda.device_count()}')
"

# 進入工作目錄
cd /home/u2249585/uav/code/mmdet-aitod

# 設定環境變數
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/home/u2249585/uav/code/mmdet-aitod:$PYTHONPATH

# 使用 PyTorch 1.x 的分散式訓練腳本
echo ""
echo "Starting 4-GPU distributed training..."
./tools/dist_train.sh \
    configs/aitod/week1_faster_rcnn_r50_baseline_aitodv2.py \
    4 \
    --work-dir /home/u2249585/uav/output/week1_baseline_stable

echo ""
echo "=========================================="
echo "Training Completed!"
echo "End Time: $(date)"
echo "=========================================="
