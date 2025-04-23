import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import glob
import cv2

# 改成自己的标签，同时该标签顺序对应了转换后YOLO数据集的标签顺序
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
	] 

def convert(size, box):
    dw = 1.0 / (size[0] + 1)
    dh = 1.0 / (size[1] + 1)
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_name, img_path, xml_path, yolo_path):
    # 正确获取文件名（不含路径）
    base_name = os.path.basename(image_name)
    # 获取不含扩展名的文件名
    name_without_ext = os.path.splitext(base_name)[0]
    
    # 正确拼接XML文件路径
    xml_file = os.path.join(xml_path, f"{name_without_ext}.xml")
    # 正确拼接输出文件路径
    txt_file = os.path.join(yolo_path, f"{name_without_ext}.txt")
    
    # 打开并读取XML文件
    with open(xml_file, 'r') as f:
        xml_text = f.read()
    root = ET.fromstring(xml_text)
    
    # 读取图片获取尺寸
    img = cv2.imread(os.path.join(img_path, base_name))
    w = img.shape[1]
    h = img.shape[0]
    
    # 写入转换后的标注
    with open(txt_file, 'w') as out_file:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                print(cls)
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

if __name__ == '__main__':
    yolo_path = '/disk16t/www/UATD_Training/labels_2/'
    img_path = '/disk16t/www/UATD_Training/images'
    xml_path = '/disk16t/www/UATD_Training/annotations'
    
    # 创建输出目录（如果不存在）
    os.makedirs(yolo_path, exist_ok=True)
    
    # 遍历图片文件
    for image_path in glob.glob(os.path.join(img_path, '*')):
        print(f"Processing: {image_path}")
        convert_annotation(image_path, img_path, xml_path, yolo_path)
