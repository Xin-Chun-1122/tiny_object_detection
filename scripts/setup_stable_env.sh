#!/bin/bash
# 創建穩定的訓練環境：PyTorch 1.13 + CUDA 11.7 + mmcv 1.7.1

# 初始化 conda
source ~/miniconda3/etc/profile.d/conda.sh

conda create -n mmdet_stable python=3.8 -y
conda activate mmdet_stable

# 安裝穩定版本的 PyTorch（mmcv 官方測試過的版本）
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 安裝 mmcv-full（預編譯版本，支援 PyTorch 1.13）
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html

# 安裝其他依賴
pip install opencv-python pillow matplotlib scipy
pip install yapf==0.40.1

# 安裝 mmdet
cd /home/u2249585/uav/code/mmdet-aitod
pip install -e .

echo "✅ 穩定環境安裝完成！"
echo "PyTorch 1.13.1 + CUDA 11.7 + mmcv-full 1.7.1"
echo ""
echo "啟動方式: conda activate mmdet_stable"
