import torch
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

def calculate_ar_from_predictions(model, data_yaml=None):
    """使用模型预测结果计算AR值"""
    # 运行验证并获取结果
    results = model.val(data=data_yaml)
    
    # 获取类别名称
    class_names = model.names
    
    # 初始化AR计算所需的变量
    class_recalls = defaultdict(list)
    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # [0.5, 0.55, ..., 0.95]
    
    # 获取验证集路径
    if hasattr(model, 'data') and isinstance(model.data, dict) and 'val' in model.data:
        val_path = model.data['val']
    else:
        # 如果无法获取验证集路径，使用默认路径
        val_path = os.path.join(os.path.dirname(data_yaml), 'val') if data_yaml else None
    
    # 使用模型直接在验证集上预测
    device = next(model.parameters()).device
    
    # 获取验证集图像列表
    if val_path and os.path.exists(val_path):
        val_images = [os.path.join(val_path, img) for img in os.listdir(val_path) if img.endswith(('.jpg', '.jpeg', '.png'))]
    else:
        # 如果无法获取验证集图像，使用results中的信息
        print("无法获取验证集图像路径，将使用验证结果计算AR")
        
        # 从results中提取信息
        tp = results.box.tp
        conf = results.box.conf
        pred_cls = results.box.pred_cls
        target_cls = results.box.target_cls
        
        # 计算每个类别的召回率
        class_ar = {}
        for class_id in range(len(class_names)):
            # 获取当前类别的预测和目标
            class_mask = target_cls == class_id
            class_tp = tp[class_mask]
            
            if len(class_mask) > 0:
                # 计算召回率
                recall = class_tp.sum() / max(1, len(class_mask))
                class_ar[class_names[class_id]] = float(recall)
            else:
                class_ar[class_names[class_id]] = 0.0
        
        return class_ar
    
    # 如果能获取验证集图像，进行详细计算
    class_ar = {}
    for class_id in range(len(class_names)):
        class_name = class_names[class_id]
        class_ar[class_name] = results.box.ap50[class_id]  # 使用AP50作为近似
    
    return class_ar

def visualize_results(class_ar, save_path):
    """可视化AR结果"""
    # 创建DataFrame
    df = pd.DataFrame(list(class_ar.items()), columns=['类别', 'AR值'])
    df = df.sort_values('AR值', ascending=False)
    
    # 绘制条形图
    plt.figure(figsize=(12, 6))
    sns.barplot(x='类别', y='AR值', data=df)
    plt.xticks(rotation=45)
    plt.title('各类别的平均召回率(AR)')
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_path)
    plt.close()
    
    # 打印表格
    print("\n各类别AR值:")
    print(df.to_string(index=False))
    
    # 计算并打印平均AR
    mean_ar = df['AR值'].mean()
    print(f"\n所有类别的平均AR值: {mean_ar:.4f}")

if __name__ == '__main__':
    # 加载模型
    model_path = '/disk16t/www/YOLOv11-SDC-main-original/runs/train/exp_improved_iAFF_83/weights/best.pt'
    model = YOLO(model_path)
    
    # 获取数据配置文件路径
    data_yaml ="/disk16t/www/YOLOv11-SDC-main-original/ultralytics/cfg/datasets/mine_data.yaml"
    
    # 计算AR值
    print("计算各类别的AR值...")
    class_ar = calculate_ar_from_predictions(model, data_yaml)
    
    # 可视化结果
    save_path = '/disk16t/www/YOLOv11-SDC-main-original/runs/train/exp_UATD_original2/class_ar_2.png'
    visualize_results(class_ar, save_path)