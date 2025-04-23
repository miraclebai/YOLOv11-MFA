import xml.etree.ElementTree as ET
import os
from pathlib import Path
from collections import Counter

def get_classes_from_annotations():
    # 设置标注文件路径
    dataset_path = Path('/disk16t/www/UATD_Training')
    xml_path = dataset_path / 'annotations'
    
    # 用于存储所有类别
    classes = set()
    # 用于统计每个类别的数量
    class_counter = Counter()
    
    # 遍历所有XML文件
    for xml_file in xml_path.glob('*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 获取每个目标的类别
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            classes.add(class_name)
            class_counter[class_name] += 1
    
    # 将类别转换为排序后的列表
    classes_list = sorted(list(classes))
    
    # 打印结果
    print("\n数据集中的所有类别:")
    for i, class_name in enumerate(classes_list):
        print(f"{i}: {class_name} (数量: {class_counter[class_name]})")
    
    print("\nYAML格式的类别配置:")
    print("names:")
    for i, class_name in enumerate(classes_list):
        print(f"  {i}: {class_name}")
    
    return classes_list

if __name__ == '__main__':
    get_classes_from_annotations()