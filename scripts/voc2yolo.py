import xml.etree.ElementTree as ET
import os
from pathlib import Path

def convert_box(size, box):
    """将VOC格式的边界框转换为YOLO格式"""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    
    # VOC格式的坐标
    xmin, ymin, xmax, ymax = box
    
    # 确保坐标在有效范围内并处理可能的异常情况
    xmin = max(0.0, min(size[0], float(xmin)))
    ymin = max(0.0, min(size[1], float(ymin)))
    xmax = max(xmin + 1.0, min(size[0], float(xmax)))
    ymax = max(ymin + 1.0, min(size[1], float(ymax)))
    
    # 转换为YOLO格式
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    
    # 归一化并确保值在0-1范围内，使用round避免精度问题
    x = round(max(0.0001, min(0.9999, x_center * dw)), 6)
    y = round(max(0.0001, min(0.9999, y_center * dh)), 6)
    w = round(max(0.0001, min(0.9999, w * dw)), 6)
    h = round(max(0.0001, min(0.9999, h * dh)), 6)
    
    # 最后的安全检查
    if x <= 0 or y <= 0 or w <= 0 or h <= 0 or x >= 1 or y >= 1 or w >= 1 or h >= 1:
        raise ValueError(f"Invalid box values: x={x}, y={y}, w={w}, h={h}")
    
    return (x, y, w, h)

def convert_voc_to_yolo():
    """将VOC格式数据集转换为YOLO格式"""
    # 设置路径
    dataset_path = Path('/disk16t/www/UATD_Test_1')
    xml_path = dataset_path / 'annotations'
    labels_path = dataset_path / 'labels'
    
    # 创建labels目录
    labels_path.mkdir(exist_ok=True)
    
    # 类别映射（根据实际类别修改）
    classes = ['ball',
  		'circle cage',
 		'cube',
 		'cylinder',
 		'human body',
  		'metal bucket',
  		'plane',
  		'rov',
 		'square cage',
  		'tyre'
	]  # 添加你的数据集中的所有类别
    
    # 处理每个xml文件
    for xml_file in xml_path.glob('*.xml'):
        # 解析XML文件
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 获取图像尺寸
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        
        # 创建对应的txt文件
        txt_file = labels_path / f"{xml_file.stem}.txt"
        
        with txt_file.open('w') as f:
            # 处理每个目标
            for obj in root.findall('object'):
                # 获取类别
                class_name = obj.find('name').text
                
                # 如果类别不在预定义列表中，跳过
                if class_name not in classes:
                    continue
                    
                class_id = classes.index(class_name)
                
                # 获取边界框坐标
                bbox = obj.find('bndbox')
                box = [
                    float(bbox.find('xmin').text),
                    float(bbox.find('ymin').text),
                    float(bbox.find('xmax').text),
                    float(bbox.find('ymax').text)
                ]
                
                # 转换为YOLO格式
                bb = convert_box((width, height), box)
                
                # 写入txt文件
                f.write(f"{class_id} {' '.join(map(str, bb))}\n")

if __name__ == '__main__':
    convert_voc_to_yolo()
    print("转换完成！标注文件已保存到labels目录")
    