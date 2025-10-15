from ultralytics import YOLO
import os
import json
import pandas as pd
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path

def infer_one_image(model, image_path):
    '''
    Perform inference on a single image using the provided YOLO model.
    input: model (YOLO)- Pre-trained YOLO model.
           image_path (str)- Path to the input image.
    output: results (list)- List of detected objects with their details.
    '''
    results = model(image_path, verbose = False)
    return results

def evaluate_model_on_dataset(model, annots, base_path, occ_trunc = True, max_count = None):
    '''
    Evaluate the YOLO model on a dataset and compare predictions with ground truth annotations.
    input: model (YOLO)- Pre-trained YOLO model.
           annots (list)- List of annotations loaded from the JSON file.
           base_path (str)- Base path to the images.
    output: eval_df (DataFrame)- DataFrame containing evaluation results.
    '''
    eval_data = []
    for annot in annots:
        if max_count is not None and max_count == 0:
            break
        max_count -= 1
        image_id = annot['name']
        image_path = os.path.join(base_path, image_id)
        results = infer_one_image(model, image_path)
        
        # Extract ground truth boxes
        gt_boxes = []
        for obj in annot.get('labels', []):
            if 'box2d' in obj:
                # if not occ_trunc:
                #     if obj['attributes']['occluded'] or obj['attributes']['truncated']:
                #             continue
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
                pred_boxes.append({'category': result.names[cls], 'bbox': [x1, y1, x2, y2], 'confidence': conf})
        
        eval_data.append({
            'image_id': image_id,
            'ground_truth': gt_boxes,
            'predictions': pred_boxes
        })
    
    eval_df = pd.DataFrame(eval_data)

    #print(eval_df)

    return eval_df

def calculate_iou(boxA, boxB):
    '''
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    input: boxA (list)- Bounding box A in the format [x1, y1, x2, y2].
           boxB (list)- Bounding box B in the format [x1, y1, x2, y2].
    output: iou (float)- Intersection over Union value.
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
    input: eval_df (DataFrame)- DataFrame containing evaluation results.
           iou_threshold (float)- IoU threshold to consider a detection as true positive.
    output: mAP (float)- Mean Average Precision value.
    '''
    # Prepare structures for classwise metrics
    # Collect all class ids present in GT and predictions
    all_classes = set()
    for _, row in eval_df.iterrows():
        for gt in row['ground_truth']:
            all_classes.add(gt['category'])
        for pred in row['predictions']:
            all_classes.add(pred['category'])
    all_classes = sorted(list(all_classes))

    # We'll include a 'none' label for unmatched GT or predictions
    none_label = 'none'
    class_labels = [str(c) for c in all_classes] + [none_label]

    # Initialize count-based confusion matrix: rows=GT class (including 'none'), cols=pred class (including 'none')
    # Values are integer counts of instances
    conf_counts = {gt_c: {pred_c: 0 for pred_c in class_labels} for gt_c in class_labels}

    # Per-class counts for TP/FP/FN
    per_class_counts = {str(c): {'TP': 0, 'FP': 0, 'FN': 0} for c in all_classes}

    for index, row in eval_df.iterrows():
        gt_boxes = row['ground_truth']
        pred_boxes = row['predictions']
        tp, fp, fn = 0, 0, len(gt_boxes)

        matched_gt = set()
        # For each prediction try to match best GT; update counts in confusion matrix
        for pred in pred_boxes:
            pred_box = pred['bbox']
            pred_cls = str(pred.get('category'))
            best_iou = 0
            best_gt_idx = -1
            for idx, gt in enumerate(gt_boxes):
                if idx in matched_gt:
                    continue
                iou = calculate_iou(pred_box, gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp += 1
                matched_gt.add(best_gt_idx)
                gt_cls = str(gt_boxes[best_gt_idx]['category'])
                # increment confusion count at (GT class, predicted class)
                conf_counts[gt_cls][pred_cls] = conf_counts[gt_cls].get(pred_cls, 0) + 1
                # update per-class TP
                if gt_cls in per_class_counts:
                    per_class_counts[gt_cls]['TP'] += 1
                else:
                    per_class_counts[gt_cls] = {'TP': 1, 'FP': 0, 'FN': 0}
            else:
                fp += 1
                # increment confusion count at ('none' GT, predicted class) to capture false positive
                conf_counts[none_label][pred_cls] = conf_counts[none_label].get(pred_cls, 0) + 1
                # update per-class FP for predicted class
                if pred_cls in per_class_counts:
                    per_class_counts[pred_cls]['FP'] += 1
                else:
                    per_class_counts[pred_cls] = {'TP': 0, 'FP': 1, 'FN': 0}

        fn -= tp
        eval_df.at[index, 'TP'] = tp
        eval_df.at[index, 'FP'] = fp
        eval_df.at[index, 'FN'] = fn

        # For any GTs that remained unmatched, increment (GT class, 'none') to indicate missed GT
        for idx, gt in enumerate(gt_boxes):
            if idx not in matched_gt:
                gt_cls = str(gt['category'])
                conf_counts[gt_cls][none_label] = conf_counts[gt_cls].get(none_label, 0) + 1
                if gt_cls in per_class_counts:
                    per_class_counts[gt_cls]['FN'] += 1
                else:
                    per_class_counts[gt_cls] = {'TP': 0, 'FP': 0, 'FN': 1}

    # Compute overall precision/recall/mAP as before
    total_TP = eval_df['TP'].sum()
    total_FP = eval_df['FP'].sum()
    total_FN = eval_df['FN'].sum()
    precision = total_TP / (total_TP + total_FP + 1e-6)
    recall = total_TP / (total_TP + total_FN + 1e-6)
    overall_map = (precision + recall) / 2

    # Build count-based confusion matrix DataFrame
    conf_rows = []
    for gt_c in sorted(conf_counts.keys(), key=lambda x: (int(x) if x.isdigit() else x)):
        row_vals = [conf_counts[gt_c].get(pred_c, 0) for pred_c in class_labels]
        conf_rows.append(row_vals)
    conf_df = pd.DataFrame(conf_rows, index=[str(c) for c in sorted([int(x) if x.isdigit() else x for x in conf_counts.keys()])], columns=class_labels)
    # Drop the 'none' row and column from the final confusion matrix as requested
    if none_label in conf_df.columns:
        conf_df.drop(columns=[none_label], inplace=True, errors='ignore')
    if none_label in conf_df.index:
        conf_df.drop(index=[none_label], inplace=True, errors='ignore')
    # Compute classwise mAP using (precision+recall)/2 per class
    classwise_map = {}
    for cls in per_class_counts.keys():
        counts = per_class_counts[cls]
        tp_c = counts.get('TP', 0)
        fp_c = counts.get('FP', 0)
        fn_c = counts.get('FN', 0)
        prec_c = tp_c / (tp_c + fp_c + 1e-6)
        rec_c = tp_c / (tp_c + fn_c + 1e-6)
        classwise_map[cls] = (prec_c + rec_c) / 2

    return overall_map, conf_df, classwise_map

def plot_confusion_matrix(conf_df, title='Confidence Matrix'):
    '''
    Plot the classwise confidence matrix as a heatmap.
    input: conf_df (DataFrame)- DataFrame containing confidence values.
           title (str)- Title of the plot.
    output: fig (Figure)- Matplotlib Figure object containing the plot.
    '''
    conf_df.fillna(0, inplace=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_df, annot=True, fmt= 'd',cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Ground Truth Class')
    plt.tight_layout()
    plt.show()
    
