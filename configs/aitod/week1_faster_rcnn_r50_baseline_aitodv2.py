"""
Week 1: Faster R-CNN Baseline on AI-TOD-v2

目標: 13-14% mAP
這是你的第一個 baseline，用來學習：
1. 訓練流程
2. 評測指標
3. 各類別檢測難度
4. 建立實驗記錄習慣
"""

_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_aitod.py',
    '../_base_/datasets/aitodv2_detection.py',  # 使用 AI-TOD-v2
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=8)),  # AI-TOD-v2 有 8 個類別
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                gpu_assign_thr=1024)),
        rpn_proposal=dict(
            nms_pre=3000,  # AI-TOD-v2 需要更多 proposals
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                gpu_assign_thr=1024))),
    test_cfg=dict(
        rpn=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=3000)  # max det = 3000 for tiny objects
    ))

fp16 = dict(loss_scale=512.)

# optimizer - 4 GPU 版本
# 基礎 lr = 0.01 for 2 samples/gpu × 2 GPUs = 4
# 4 GPU = 2 samples/gpu × 4 GPUs = 8, 所以 lr = 0.01 * 8/4 = 0.02
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

# learning policy
checkpoint_config = dict(interval=4)  # 每 4 epochs 保存一次

# 工作目錄會由命令行參數指定
work_dir = None
