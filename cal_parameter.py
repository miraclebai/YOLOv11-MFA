#!/usr/bin/env python3
import argparse
import torch

def count_parameters(model):
    """计算 nn.Module 模型中所有参数的总数"""
    return sum(p.numel() for p in model.parameters())

def count_parameters_state_dict(state_dict):
    """计算 state_dict 中所有张量参数的总数"""
    return sum(t.numel() for t in state_dict.values())

def main():
    parser = argparse.ArgumentParser(description="计算模型参数量")
    parser.add_argument('--weights', type=str, required=True,
                        help='模型权重路径，例如：/disk16t/www/YOLOv11-SDC-main-original/runs/train/exp_improved_iAFF_8/weights/best.pt')
    args = parser.parse_args()

    # 加载权重文件（加载到 CPU 上以避免 GPU 内存占用）
    checkpoint = torch.load(args.weights, map_location='cpu')

    # 判断加载内容的类型
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint and hasattr(checkpoint['model'], 'parameters'):
            # 如果 checkpoint 中包含 'model' 键，并且该对象拥有 parameters() 方法，则认为是 nn.Module 对象
            model = checkpoint['model']
            total_params = count_parameters(model)
            print("模型参数总数：", total_params)
        elif 'state_dict' in checkpoint:
            # 如果包含 state_dict 键，则按 state_dict 计算
            total_params = count_parameters_state_dict(checkpoint['state_dict'])
            print("模型参数总数 (state_dict)：", total_params)
        else:
            # 否则假设整个 checkpoint 为 state_dict 格式
            total_params = count_parameters_state_dict(checkpoint)
            print("模型参数总数 (state_dict)：", total_params)
    else:
        # 如果加载的 checkpoint 不是 dict，则假设为 nn.Module 对象
        model = checkpoint
        total_params = count_parameters(model)
        print("模型参数总数：", total_params)

if __name__ == "__main__":
    main()
