#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import torch
import importlib
from pathlib import Path

# 添加项目根目录到系统路径，并确保它是第一位
ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))  # 确保项目目录优先于系统目录

# 检查导入路径
print(f"项目根目录: {ROOT}")
print(f"Python 路径: {sys.path[0]}")

# 直接从项目目录导入
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils.torch_utils import get_flops, get_num_params

class ShapeAnalyzer:
    def __init__(self, model):
        self.model = model
        self.layer_shapes = {}
        self.hooks = []
        self.register_hooks()
        
    def register_hooks(self):
        """为模型的每一层注册前向传播钩子，记录输入输出形状"""
        for name, module in self.model.named_modules():
            if name == '':  # 跳过模型本身
                continue
                
            # 注册钩子
            hook = module.register_forward_hook(
                lambda m, inp, out, name=name: self._hook_fn(m, inp, out, name)
            )
            self.hooks.append(hook)
    
    def _hook_fn(self, module, inp, out, name):
        """钩子函数，记录每一层的输入输出形状"""
        # 处理输入形状
        if isinstance(inp, tuple) and len(inp) > 0:
            input_shape = [tuple(i.shape) if isinstance(i, torch.Tensor) else type(i) for i in inp]
            if len(input_shape) == 1:
                input_shape = input_shape[0]
        else:
            input_shape = None
            
        # 处理输出形状
        if isinstance(out, tuple):
            output_shape = [tuple(o.shape) if isinstance(o, torch.Tensor) else type(o) for o in out]
            if len(output_shape) == 1:
                output_shape = output_shape[0]
        elif isinstance(out, torch.Tensor):
            output_shape = tuple(out.shape)
        elif isinstance(out, dict):
            output_shape = {k: tuple(v.shape) if isinstance(v, torch.Tensor) else type(v) for k, v in out.items()}
        else:
            output_shape = type(out)
            
        self.layer_shapes[name] = {
            'input_shape': input_shape,
            'output_shape': output_shape,
            'module_type': module.__class__.__name__
        }
    
    def analyze(self, input_size=(1, 3, 640, 640)):
        """分析模型各层形状"""
        # 创建随机输入
        x = torch.randn(input_size).to(next(self.model.parameters()).device)
        
        # 推理模式
        self.model.eval()
        
        # 前向传播
        with torch.no_grad():
            try:
                out = self.model(x)
            except Exception as e:
                print(f"前向传播出错: {e}")
                # 尝试使用更简单的方式进行前向传播
                try:
                    out = self.model.forward_once(x)
                    print("使用 forward_once 方法成功")
                except Exception as e2:
                    print(f"forward_once 也失败: {e2}")
                    return None
            
        # 移除钩子
        for hook in self.hooks:
            hook.remove()
            
        # 获取模型输入输出形状
        model_input_shape = input_size
        if isinstance(out, tuple):
            model_output_shape = [tuple(o.shape) if isinstance(o, torch.Tensor) else type(o) for o in out]
        elif isinstance(out, torch.Tensor):
            model_output_shape = tuple(out.shape)
        elif isinstance(out, dict):
            model_output_shape = {k: tuple(v.shape) if isinstance(v, torch.Tensor) else type(v) for k, v in out.items()}
        else:
            model_output_shape = type(out)
            
        return {
            'model_input_shape': model_input_shape,
            'model_output_shape': model_output_shape,
            'layer_shapes': self.layer_shapes
        }
        
    def print_shapes(self, results, detailed=True):
        """打印形状分析结果"""
        if results is None:
            print("分析失败，无法打印结果")
            return
            
        print("\n" + "="*80)
        print(f"模型输入形状: {results['model_input_shape']}")
        print(f"模型输出形状: {results['model_output_shape']}")
        print("="*80)
        
        if detailed:
            print("\n各层形状详情:")
            print("-"*80)
            print(f"{'层名称':<50} {'类型':<20} {'输入形状':<30} {'输出形状'}")
            print("-"*80)
            
            for name, info in results['layer_shapes'].items():
                print(f"{name:<50} {info['module_type']:<20} {str(info['input_shape']):<30} {str(info['output_shape'])}")


def main():
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='分析YOLOv11-SDC模型各层的输入输出形状')
    parser.add_argument('--model', type=str, default='yolo11-SDC.yaml', help='模型配置文件或权重文件路径')
    parser.add_argument('--img-size', type=int, default=640, help='输入图像大小')
    parser.add_argument('--batch-size', type=int, default=1, help='批次大小')
    parser.add_argument('--device', type=str, default='', help='运行设备，如 cuda:0 或 cpu')
    parser.add_argument('--detailed', action='store_true', help='是否显示详细的层信息')
    parser.add_argument('--save', action='store_true', help='是否保存分析结果到文件')
    args = parser.parse_args()
    
    # 确保自定义模块已经被导入
    print("正在确保自定义模块可用...")
    try:
        # 直接从项目目录导入模块
        import ultralytics.nn.modules.block as block_module
        print(f"模块导入路径: {block_module.__file__}")
        
        # 检查是否包含所需的自定义模块
        if hasattr(block_module, 'C3k2_DRB'):
            print("成功找到 C3k2_DRB 模块")
        else:
            print("警告: 在导入的模块中未找到 C3k2_DRB")
            
        # 导入 YOLO 类
        from ultralytics import YOLO
        
    except Exception as e:
        print(f"导入自定义模块时出错: {e}")
        return
    
    # 加载模型
    print(f"正在加载模型: {args.model}")
    try:
        if args.model.endswith('.yaml'):
            # 从YAML配置文件加载
            model = YOLO(args.model).model
        elif args.model.endswith('.pt'):
            # 从权重文件加载
            model = attempt_load_one_weight(args.model, device=args.device)
        else:
            # 尝试使用YOLO类加载
            model = YOLO(args.model).model
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 设置设备
    device = args.device if args.device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 分析模型
    print(f"正在分析模型形状，输入大小: ({args.batch_size}, 3, {args.img_size}, {args.img_size})")
    analyzer = ShapeAnalyzer(model)
    results = analyzer.analyze(input_size=(args.batch_size, 3, args.img_size, args.img_size))
    
    if results is not None:
        # 打印模型信息
        try:
            num_params = get_num_params(model)[0] / 1e6
            flops = get_flops(model, (args.batch_size, 3, args.img_size, args.img_size))[0] / 1e9
            print(f"\n模型参数量: {num_params:.2f}M")
            print(f"模型计算量: {flops:.2f}G FLOPS")
        except Exception as e:
            print(f"计算模型参数量和计算量时出错: {e}")
    
    # 打印形状信息
    analyzer.print_shapes(results, detailed=args.detailed)
    
    # 保存结果到文件
    if args.save and results is not None:
        import json
        from datetime import datetime
        
        # 将结果转换为可序列化的格式
        serializable_results = {
            'model_input_shape': str(results['model_input_shape']),
            'model_output_shape': str(results['model_output_shape']),
            'layer_shapes': {
                name: {
                    'module_type': info['module_type'],
                    'input_shape': str(info['input_shape']),
                    'output_shape': str(info['output_shape'])
                }
                for name, info in results['layer_shapes'].items()
            }
        }
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_shapes_{timestamp}.json"
        
        # 保存到文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n分析结果已保存到: {filename}")


if __name__ == "__main__":
    main()
