import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
from PIL import __version__
import argparse
from pathlib import Path
import re
import threading
import queue
import logging

dir_gt = '../dataset/SA_1B_GT'
# predict_folder = '../SAM/SA_1B_Output'
predict_folder = '../MobileSAM/outputs'
predict = '_h'
Pmethod = '_yolo'
groundtruth = ''
profile_num = 1000


# Configure logging
logging.basicConfig(filename=f'log{Pmethod}{predict}.txt', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')


def calculate_iou(image_gt, image_pr):
    intersection = np.logical_and(image_gt, image_pr)
    union = np.logical_or(image_gt, image_pr)
    return np.sum(intersection) / np.sum(union)

def sort_gt(path):

    num = int(path.split("_")[0])
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

def load_and_threshold_image(path, threshold=128):
    image = Image.open(path).convert('L')
    return np.array(image) > threshold

def process_target(target, predict_folder,predict, scores_queue, num_queue, lock):
    # This is a simplified version of your processing logic
    # Add your detailed processing logic here
    groundtruth_folder = target
   
    gt_files = {file for file in os.listdir(groundtruth_folder) if not (file.endswith('.csv') or 'gt' in file)}
    gt_files = sorted(gt_files, key=sort_gt)
    sum = 0
    # for file in os.listdir(groundtruth_folder):
    #     if file.endswith('.csv'):
    #         file_path = os.path.join(groundtruth_folder, file)
            
    #         df = pd.read_csv(file_path)
            
    #         if 'area' in df.columns:
    #             sum += df['area'].sum()
            # else:
            #     print(f"File: {file} does not contain 'area' column")
    num = len(gt_files)
    # predict_folder = os.path.join(target, f'out{predict}')

    score = 0
    base = target.split('/')[-1]
    # if args.gt:
    #     predict_path = os.path.join(predict_folder,base)
    #     predict_files = os.listdir(predict_path)
    # else:
    predict_path = os.path.join(predict_folder,base, f'out{predict}')

    predict_files = os.listdir(os.path.join(predict_folder,base, f'out{predict}'))
    if 'metadata.csv' in predict_files:
        predict_files.remove('metadata.csv')

    sorted_predict_files = sorted(predict_files, key=sort_by_img)
    # print(f"sorted_predict_files is {sorted_predict_files}")
    # print(f"gt_files is {gt_files}")
    remain_gt_files = gt_files.copy()
    # print(sorted_predict_files)
    matched = 0
    for gt_file in list(gt_files):
        gt_image = load_and_threshold_image(os.path.join(groundtruth_folder, gt_file))
        
        best_iou = 0
        best_predict_file = None

        for predict_file in sorted_predict_files:
            # print("gt_file is ",gt_file)
            # print("predict_file is ",predict_file)
            predict_image = load_and_threshold_image(os.path.join(predict_path, predict_file))
            iou_score = calculate_iou(predict_image, gt_image)
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
            matched += 1
            # print(f"Best match for {gt_file} is {best_predict_file} with IoU: {best_iou}")
            remain_gt_files.remove(gt_file)
            sorted_predict_files.remove(best_predict_file)
            score += best_iou#*(best_predict_file_area/sum)
        if not gt_files:
            break 
    # print(f"scores : {score}")
    # logging.info(f"{target}: {score/num}, {num} masks")
 
            
    # logging.info(f"There are {len(sorted_predict_files)} predicted files left, which are {sorted_predict_files}")
    print(f"There are {len(remain_gt_files)} gt files left, which are {remain_gt_files}")
    print(f"{target}: {score/num}, {matched} masks")
    logging.info(f"There are {len(remain_gt_files)} gt files left, which are {remain_gt_files}")
    logging.info(f"{target}: {score/num}, {matched} masks")
    scores_queue.put(score)
    num_queue.put(matched)
    

def SA_area_weighted_iou_parallel(dir, predict_folder,predict, profile_num, num_threads=20):
    sorted_targets = sort_folder_by_img(dir, sort_by_dir)
    lock = threading.Lock()  # For thread-safe prints or shared data modifications
    scores_queue = queue.Queue()  # For storing the scores from each thread
    num_queue = queue.Queue()
    
    # Define chunks for each thread to process
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # Split sorted_targets into approximately equal chunks for each thread
    target_chunks = list(chunks(sorted_targets[:profile_num], profile_num // num_threads + 1))

    threads = []
    for chunk in target_chunks:
        for target in chunk:
            thread = threading.Thread(target=process_target, args=(target, predict_folder,predict, scores_queue,num_queue,lock))
            threads.append(thread)
            thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    total_score = 0
    total_num = 0
    while not scores_queue.empty():
        total_score += scores_queue.get()
    while not num_queue.empty():
        total_num += num_queue.get()
    
    
    print(f"Total IoU score: {total_score/profile_num} with {total_num} masks")
    print("All threads have finished.")
    logging.info(f"Total IoU score: {total_score/profile_num} with {total_num} masks")

SA_area_weighted_iou_parallel(dir_gt,predict_folder, predict, profile_num)
