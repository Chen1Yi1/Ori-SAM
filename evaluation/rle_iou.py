
import glob
import os
from typing import Any, Dict, List
from pycocotools.coco import COCO
from pycocotools.mask import iou, decode, encode
import json
import matplotlib.pyplot as plt
import cv2  # type: ignore
import numpy as np
import shutil
import argparse
from pathlib import Path
import re
import threading
import queue

def sort_by_num(path,dataset='SA_1B'):
    if dataset == "./LVIS_output/":
        num = int(path.split("/")[-1].split(".")[0])
    else:
        num = int(path.split("_")[-1].split(".")[0])
    return num

def sort_by_img(path):
    if dir == "./LVIS_output/":
        num = int(path.split("/")[-1].split(".")[0])
    else:
        num = int(path.split(".")[0])
    return num

def sort_folder_by_img(dir,method):
    targets = [
        f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))
    ]
    targets = [os.path.join(dir, f) for f in targets]
    sorted_targets = sorted(targets, key=method)
    return sorted_targets

def sort_by_dir(path):
    if dir == "./LVIS_output/":
        num = int(path.split("/")[-1].split(".")[0])
    else:
        num = int(path.split("_")[-1])
    return num

def sort_gt(path):

    num = int(path.split("_")[0])
    return num

def find_files(directory, suffix):
    files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(suffix + '.json'):
                files.append(os.path.join(dirpath, filename))
    return files

def process_json_file(json_file):
    # print("Processing:", json_file)

    gt_file = f"{dir_gt}/{json_file}.json"
    predict_files = f"{predict_folder}/{json_file}/{json_file}{predict}.json"

    with open(gt_file, 'r') as file:
        gt_data = json.load(file)
    with open(predict_files, 'r') as file:
        predict_data = json.load(file)
    

    score = 0
    matched = 0
    for i, annotation in enumerate(gt_data['annotations']):
        gt_rle = {
            'counts': annotation['segmentation']['counts'],
            'size': annotation['segmentation']['size']
        }
        # gt_mask = decode(gt_rle)
        best_iou = 0  # 初始化最佳 IoU 值
        best_predict_file = None  # 初始化最佳预测文件
        
        for predict_file in predict_data:
            # print("gt_file is ",gt_file)
            # print("predict_file is ",predict_files)
            
            pr_rle = {
                'counts': predict_file['segmentation']['counts'],
                'size': predict_file['segmentation']['size']
            }
            # pr_mask = decode(pr_rle)
            # pr_mask = encode(pr_mask)
            
            
            iou_score = iou([pr_rle], [gt_rle], [0])[0][0]

            # print(iou_score)
            if (iou_score > best_iou) and (iou_score > 0.9):
                best_iou = iou_score
                best_predict_file = predict_file
                # best_predict_file_area = predict_image.sum()
        
            # Skipping the rest of comparision to save time, value can be adjusted
            if best_iou > 0.99:
                # if best_iou != 1:
                #     print(f"Best match for {gt_file} is {best_predict_file} with IoU: {best_iou}")
                break
        
        if best_predict_file:
            # print(f"Best match for {gt_file} is {best_predict_file} with IoU: {best_iou}")
            # remain_gt_files.remove(gt_file)
            # sorted_predict_files.remove(best_predict_file)
            # print(best_iou)
            matched +=1
            score += best_iou#*(best_predict_file_area/sum)
        # if not gt_files:
        #     break 
    avg_score = score / len(gt_data['annotations']) if gt_data['annotations'] else 0
    logging.info(f"File: {json_file} - Avg IoU: {avg_score:.4f}, Total Annotations: {len(gt_data['annotations'])}, Predictions Made: {len(predict_data)}, Matches Found: {matched}, True positives: {matched/len(gt_data['annotations']):.4f}, False positives: {(len(predict_data)-matched)/len(predict_data):.4f}")
    return avg_score, len(gt_data['annotations']), len(predict_data), matched

# dataset directory
dir_gt = '../dataset/SA_1B'

# get all of the groundtruth json files in the directory
json_files = glob.glob(f"{dir_gt}/*.json")

gt_sorted_targets = sorted(json_files, key=sort_by_num)

predict_sorted_targets = []
for directory in predict_files:
    
    json_files = find_files(directory, predict)
    
    for file in json_files:
        predict_sorted_targets.append(file)
        
predict_folder = '../SAM/SA_1B_Output'
predict = '_h'

predict_files = sort_folder_by_img(predict_folder,sort_by_dir)


import logging


logging.basicConfig(filename=f'iou{predict}.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


num_files = 10000
skip = 0
score_sum = 0
gt_sum = 0
prediction_sum = 0
matched_sum = 0

# with ThreadPoolExecutor(max_workers=max_threads) as executor:
#     executor.map(process_json_file, sorted_targets[skip:num_files])

for i in range(skip,num_files):
    base = f'sa_{i+1}'
    score, gt, predictions, matched = process_json_file(base)
    score_sum += avg_score
    gt_sum += gt
    prediction_sum += predictions
    matched_sum += matched
    
logging.info(f"Summary for {predict}: Avg IoU: {score_sum/(num_files-skip):.4f}, Total Annotations: {gt_sum}, Predictions Made: {prediction_sum}, Matches Found: {matched_sum}, True positives: {matched_sum/gt_sum:.4f}, False positives: {(prediction_sum-matched_sum)/prediction_sum:.4f}")