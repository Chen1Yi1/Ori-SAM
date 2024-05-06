from pycocotools.coco import COCO
from pycocotools.mask import decode
import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import glob
from concurrent.futures import ThreadPoolExecutor
import shutil

path = "../dataset/SA_1B_GT/"
directory = '../dataset/SA_1B'
json_files = glob.glob(f"{directory}/*.json")

def sort_by_num(path, dataset='SA_1B'):
    return int(path.split("_")[-1].split(".")[0])

sorted_targets = sorted(json_files, key=sort_by_num)

color_list = [
    [1.0, 0.0, 0.0, 0.9],  # 红色 (Red)
    [0.0, 1.0, 0.0, 0.9],  # 绿色 (Green)
    [0.0, 0.0, 1.0, 0.9],  # 蓝色 (Blue)
    [1.0, 1.0, 0.0, 0.9],  # 黄色 (Yellow)
    [1.0, 0.0, 1.0, 0.9],  # 洋红 (Magenta)
    [0.0, 1.0, 1.0, 0.9],  # 青色 (Cyan)
    [0.5, 0.0, 0.0, 0.9],  # 深红 (Maroon)
    [0.0, 0.5, 0.0, 0.9],  # 深绿 (Dark Green)
    [0.0, 0.0, 0.5, 0.9],  # 深蓝 (Navy)
    [0.5, 0.5, 0.0, 0.9],  # 橄榄 (Olive)
    [0.0, 0.5, 0.5, 0.9],  # 深青 (Teal)
    [0.5, 0.0, 0.5, 0.9],  # 紫色 (Purple)
    [0.3, 0.3, 0.3, 0.9],  # 灰色 (Gray)
    [1.0, 0.5, 0.0, 0.9],  # 橙色 (Orange)
    [0.5, 1.0, 0.0, 0.9],  # 鲜绿 (Lime)
    [0.0, 0.5, 1.0, 0.9],  # 天蓝 (Sky Blue)
    [1.0, 0.0, 0.5, 0.9],  # 玫瑰红 (Rose)
    [0.5, 0.0, 1.0, 0.9],  # 靛蓝 (Indigo)
    [1.0, 0.5, 0.5, 0.9],  # 褐红 (Crimson)
    [0.5, 1.0, 0.5, 0.9],  # 橙黄 (Gold)
    [0.5, 0.5, 1.0, 0.9],  # 蓝紫 (Blue Violet)
    [0.8, 0.2, 0.2, 0.9],  # 深粉 (Deep Pink)
    [0.2, 0.8, 0.2, 0.9],  # 鲜绿 (Bright Green)
    [0.2, 0.2, 0.8, 0.9],  # 矢车菊蓝 (Cornflower Blue)
    [0.8, 0.8, 0.2, 0.9],  # 明黄 (Lemon Yellow)
    [0.2, 0.8, 0.8, 0.9],  # 湖蓝 (Lake Blue)
]

def get_color(index):
    return color_list[index % len(color_list)]

def process_json_file(json_file):
    print("Processing:", json_file)
    image_path = json_file.replace('json', 'jpg')
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load '{image_path}' as an image, skipping...")
        return

    with open(json_file, 'r') as file:
        data = json.load(file)

    save_folder = os.path.join(path, os.path.basename(json_file).replace('.json', ''))
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder, exist_ok=True)

    final_image = image.copy()

    for i, annotation in enumerate(data['annotations']):
        rle = {
            'counts': annotation['segmentation']['counts'],
            'size': annotation['segmentation']['size']
        }
        mask = decode(rle)
        
        filename = f"{i}_mask.png"
        cv2.imwrite(os.path.join(save_folder, filename), mask * 255)

        color = np.array(get_color(i)) * 255  * 0.35
        colored_mask = cv2.merge([
            (mask * color[0]).astype(np.uint8),
            (mask * color[1]).astype(np.uint8),
            (mask * color[2]).astype(np.uint8)
        ])

        final_image = cv2.addWeighted(final_image, 1, colored_mask, 1, 0)

    output_name = f"{save_folder}/{os.path.basename(json_file).replace('.json', '_gt.png')}"
    cv2.imwrite(output_name, final_image)
    print("Saved:", output_name)

num_files = 10000
skip = 1000
max_threads = 20

with ThreadPoolExecutor(max_workers=max_threads) as executor:
    executor.map(process_json_file, sorted_targets[skip:num_files])