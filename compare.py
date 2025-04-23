import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from ultralytics.utils.plotting import plt_settings
from ultralytics import YOLO

def print_csv_columns(file=""):
    """
    打印CSV文件中所有列的名称和索引
    
    参数:
        file (str, optional): CSV文件路径
    """
    csv_path = Path(file)
    
    if not csv_path.exists():
        print(f"未找到CSV文件: {csv_path}")
        return
    
    # 读取CSV文件
    data = pd.read_csv(csv_path)
    print("\n=== CSV文件列名及索引 ===")
    for i, col in enumerate(data.columns):
        print(f"索引 {i}: {col}")
    print("========================\n")


@plt_settings()
def custom_plot_results(file="", on_plot=None, show_columns=False):
    """
    自定义绘制训练结果图表，布局与原始plot_results相同，但修改了部分索引。
    
    参数:
        file (str, optional): 包含训练结果的CSV文件路径。
        on_plot (callable, optional): 绘图完成后的回调函数。默认为None。
        show_columns (bool, optional): 是否显示CSV文件中的所有列名。默认为False。
    """
    csv_path = Path(file)
    save_dir = csv_path.parent
    
    # 如果需要，显示列名
    if show_columns:
        print_csv_columns(file)
    
    # 创建2x5的子图布局，与原始目标检测任务相同
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    
    # 自定义索引 - 这里可以根据需要修改
    # 原始索引为 [2, 3, 4, 5, 6, 9, 10, 11, 7, 8]
    index = [2, 3, 4, 5, 6, 7, 10, 11, 12, 13]  # 可以根据print_csv_columns的输出修改
    
    if not csv_path.exists():
        print(f"未找到CSV文件: {csv_path}")
        return
    
    try:
        data = pd.read_csv(csv_path)
        s = [x.strip() for x in data.columns]
        x = data.values[:, 0]  # epochs
        for i, j in enumerate(index):
            if j < len(s):  # 确保索引在列范围内
                y = data.values[:, j].astype("float")
                ax[i].plot(x, y, marker=".", label=csv_path.stem, linewidth=2, markersize=8)  # 实际结果
                ax[i].plot(x, gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2)  # 平滑曲线
                ax[i].set_title(s[j], fontsize=12)
            else:
                print(f"警告: 索引 {j} 超出列范围 {len(s)}")
    except Exception as e:
        print(f"警告: 绘图错误: {e}")
    
    ax[1].legend()
    fname = save_dir / "custom_results.png"
    fig.savefig(fname, dpi=200)
    plt.close()
    print(f"已保存自定义结果图表到 {fname}")
    
    if on_plot:
        on_plot(fname)


@plt_settings()
def compare_metrics_pr_curves(metrics_list, names=None, save_dir=None, title="PR Curve Comparison", on_plot=None):
    """
    Compare PR curves of multiple models in one plot
    
    Args:
        metrics_list (list): List of DetMetrics objects
        names (list, optional): Names for each model. If None, will use "model1", "model2", etc.
        save_dir (str, optional): Directory to save the plot. Default is current directory
        title (str, optional): Plot title
        on_plot (callable, optional): Callback function after plotting
    """
    if save_dir is None:
        save_dir = Path(".")
    else:
        save_dir = Path(save_dir)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    
    # Different colors for different models
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Plot PR curves for each model
    for i, metrics in enumerate(metrics_list):
        model_name = names[i] if names and i < len(names) else f"model{i+1}"
        color = colors[i % len(colors)]
        
        # Get PR curve data
        px = metrics.box.px
        py_mean = metrics.box.prec_values.mean(0)
        ap_mean = metrics.box.ap50.mean()
        
        # Plot PR curve
        ax.plot(px, py_mean, linewidth=2, color=color, label=f"{model_name} mAP@0.5: {ap_mean:.3f}")
    
    # Set plot properties
    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower left', fontsize=12)
    
    # Save plot
    fname = save_dir / "pr_curves_comparison.png"
    fig.savefig(fname, dpi=300)
    plt.close()
    print(f"PR curve comparison saved to {fname}")
    
    if on_plot:
        on_plot(fname)


def get_model_metrics(model_path, data_path='coco.yaml'):
    """
    加载模型并获取其评估指标
    
    参数:
        model_path (str): 模型文件路径
        data_path (str): 数据集配置文件路径
        
    返回:
        DetMetrics对象
    """
    model = YOLO(model_path)
    results = model.val(data=data_path)
    # 修改这里：results本身就是DetMetrics对象，不需要再访问.metrics属性
    return results


def compare_models_pr_curves(model_paths, names=None, save_dir=None, data_path='coco.yaml', title="PR Curve Comparison"):
    """
    Compare PR curves of multiple models
    
    Args:
        model_paths (list): List of model file paths
        names (list, optional): Names for each model
        save_dir (str, optional): Directory to save the plot
        data_path (str): Dataset configuration file path
        title (str): Plot title
    """
    metrics_list = []
    
    for model_path in model_paths:
        print(f"Evaluating model: {model_path}")
        metrics = get_model_metrics(model_path, data_path)
        metrics_list.append(metrics)
    
    compare_metrics_pr_curves(metrics_list, names, save_dir, title)


if __name__ == "__main__":
    # 使用示例
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='/Users/baijingyuan/Desktop/results.csv', 
                        help='results.csv文件路径，默认为桌面上的results.csv')
    parser.add_argument('--show-columns', action='store_true', help='显示CSV文件中的所有列名')
    parser.add_argument('--compare-pr', action='store_true', help='比较多个算法的PR曲线')
    parser.add_argument('--files', nargs='+', help='用于PR曲线比较的多个CSV文件路径')
    parser.add_argument('--names', nargs='+', help='每个文件对应的算法名称')
    parser.add_argument('--models', nargs='+', help='用于PR曲线比较的多个模型文件路径')
    parser.add_argument('--data', type=str, default='coco.yaml', help='数据集配置文件路径')
    args = parser.parse_args()
    
    if args.show_columns:
        print_csv_columns(file=args.file)
    elif args.compare_pr:
        if args.models:
            # 比较多个模型的PR曲线
            # 修改保存目录为绝对路径
            save_dir = Path('/disk16t/www/YOLOv11-SDC-main-original/runs/compare')
            # 确保目录存在
            save_dir.mkdir(parents=True, exist_ok=True)
            
            compare_models_pr_curves(
                model_paths=args.models,
                names=args.names,
                save_dir=save_dir,
                data_path=args.data,
                title="PR Curve Comparison"  # 使用英文标题
            )
        else:
            print("Please use --models parameter to provide at least one model file path")
    else:
        custom_plot_results(file=args.file, show_columns=False)