from ultralytics import YOLO
import yaml
import os
import torch

# 清理CUDA缓存
torch.cuda.empty_cache()

# 设置环境变量以优化内存管理
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

# 初始化模型
model = YOLO('ultralytics/cfg/models/11/yolo11-SDC-improved-iAFF.yaml')
model.load("/disk16t/www/YOLOv11-SDC-main-original/yolo11n.pt")
# 开始训练
model.train(
    data='ultralytics/cfg/datasets/newdataset.yaml',
    epochs=320,  # 直接设置训练轮数
    imgsz=640,   # 直接设置图像尺寸
    batch=32,    # 批次大小
    workers=8,   # 工作线程数
    optimizer='Adam',  # 优化器
    lr0=0.001,   # 初始学习率
    lrf=0.01,    # 最终学习率因子
    momentum=0.937,  # 动量
    weight_decay=0.0005,  # 权重衰减
    hsv_h=0.015,  # HSV色调增强
    hsv_s=0.7,    # HSV饱和度增强
    hsv_v=0.4,    # HSV亮度增强
    degrees=0.0,  # 旋转角度
    translate=0.1,  # 平移
    scale=0.5,     # 缩放
    shear=0.0,     # 剪切
    perspective=0.0,  # 透视变换
    flipud=0.0,    # 上下翻转
    fliplr=0.5,    # 左右翻转
    mosaic=1.0,    # 马赛克增强
    mixup=0.0,     # 混合增强
    project='runs/train',  # 项目目录
    name='exp_improved_iAFF_new_dataset',    # 实验名称
 # 允许覆盖已有实验
  # 使用的GPU设备
    device = '1',
    amp=True,   # 关闭自动混合精度
    cache=False  # 关闭缓存
)
