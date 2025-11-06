#!/bin/bash
#SBATCH --job-name=week1_test
#SBATCH --account=ACD114147
#SBATCH --partition=gtest
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/u2249585/uav/logs/week1_test_%j.out
#SBATCH --error=/home/u2249585/uav/logs/week1_test_%j.err

echo "=========================================="
echo "Week 1 Baseline Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "=========================================="

# 啟動環境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmdet_stable

# 進入工作目錄
cd /home/u2249585/uav/code/mmdet-aitod

# 清理 Python 快取檔案並重新安裝
echo "Cleaning cache and reinstalling mmdet..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1
pip install --force-reinstall --no-deps -e . > /dev/null 2>&1

# 評測模型
echo ""
echo "Evaluating on test set..."
python tools/test.py \
    configs/aitod/week1_faster_rcnn_r50_baseline_aitodv2.py \
    /home/u2249585/uav/output/week1_baseline_stable/epoch_12.pth \
    --out /home/u2249585/uav/output/week1_baseline_stable/test_results.pkl \
    --eval bbox

echo ""
echo "=========================================="
echo "Evaluation Completed!"
echo "End Time: $(date)"
echo "=========================================="
