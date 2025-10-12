from ultralytics import YOLO
import os
import json
import pandas as pd
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path

def infer_one_image(model, image_path):
    '''
    Perform inference on a single image using the provided YOLO model.
    input: model (YOLO): Pre-trained YOLO model.
           image_path (str): Path to the input image.
    output: results (list): List of detected objects with their details.
    '''
    results = model(image_path)
    return results

def evaluate_model_on_dataset(model, annots, base_path, occ_trunc = True):
    '''
    Evaluate the YOLO model on a dataset and compare predictions with ground truth annotations.
    input: model (YOLO): Pre-trained YOLO model.
           annots (list): List of annotations loaded from the JSON file.
           base_path (str): Base path to the images.
    output: eval_df (DataFrame): DataFrame containing evaluation results.
    '''
    eval_data = []
    for annot in annots:
        image_id = annot['name']
        image_path = os.path.join(base_path, image_id)
        results = infer_one_image(model, image_path)
        
        # Extract ground truth boxes
        gt_boxes = []
        for obj in annot.get('labels', []):
            if not occ_trunc:
                if obj['attributes']['occluded'] or obj['attributes']['truncated']:
                        continue
            if 'box2d' in obj:
                box2d = obj['box2d']
                x1, y1, x2, y2 = box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']
                category = obj['category']
                gt_boxes.append({'category': category, 'bbox': [x1, y1, x2, y2]})
        
        # Extract predicted boxes
        pred_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                pred_boxes.append({'category': cls, 'bbox': [x1, y1, x2, y2], 'confidence': conf})
        
        eval_data.append({
            'image_id': image_id,
            'ground_truth': gt_boxes,
            'predictions': pred_boxes
        })
    
    eval_df = pd.DataFrame(eval_data)
    return eval_df

def calculate_iou(boxA, boxB):
    '''
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    input: boxA (list): Bounding box A in the format [x1, y1, x2, y2].
           boxB (list): Bounding box B in the format [x1, y1, x2, y2].
    output: iou (float): Intersection over Union value.
    '''
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def calculate_map(eval_df, iou_threshold=0.5):
    '''
    Calculate mean Average Precision (mAP) for the evaluation DataFrame.
    input: eval_df (DataFrame): DataFrame containing evaluation results.
           iou_threshold (float): IoU threshold to consider a detection as true positive.
    output: mAP (float): Mean Average Precision value.
    '''
    for index, row in eval_df.iterrows():
        gt_boxes = row['ground_truth']
        pred_boxes = row['predictions']
        tp, fp, fn = 0, 0, len(gt_boxes)
        
        matched_gt = set()
        for pred in pred_boxes:
            pred_box = pred['bbox']
            best_iou = 0
            best_gt_idx = -1
            for idx, gt in enumerate(gt_boxes):
                if idx in matched_gt:
                    continue
                iou = calculate_iou(pred_box, gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        fn -= tp
        eval_df.at[index, 'TP'] = tp
        eval_df.at[index, 'FP'] = fp
        eval_df.at[index, 'FN'] = fn
    precision = eval_df['TP'].sum() / (eval_df['TP'].sum() + eval_df['FP'].sum() + 1e-6)
    recall = eval_df['TP'].sum() / (eval_df['TP'].sum() + eval_df['FN'].sum() + 1e-6)
    mAP = (precision + recall) / 2
    return mAP

if __name__ == "__main__":
    # Load the YOLO model
    model = YOLO('yolov8n.pt')

    # Load annotations
    with open('path_to_annotations.json') as f:
        annots = json.load(f)

    # Define base path to images
    base_path = 'path_to_images_directory'

    # Evaluate model on dataset
    eval_df = evaluate_model_on_dataset(model, annots, base_path, occ_trunc=False)

    # Calculate mAP
    mAP = calculate_map(eval_df)
    print(f'mean Average Precision (mAP): {mAP}')