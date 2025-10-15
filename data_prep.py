import json
import pandas as pd
import os
import numpy as np
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm

def create_instance_df(annots):
    '''
    Creates a DataFrame containing instance-level annotations from the provided annotations.
    input: annots (list): List of annotations loaded from the JSON file.
    output: instance_df (DataFrame): DataFrame containing instance-level annotations.
    '''
    instance_data = []
    for annot in annots:
        image_id = annot['name']
        for obj in annot.get('labels', []):
            if 'box2d' in obj:
                category = obj['category']
                box2d = obj['box2d']
                x1, y1, x2, y2 = box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']
                occluded = obj['attributes']['occluded']
                truncated = obj['attributes']['truncated']
                instance_data.append({
                    'image_id': image_id,
                    'category': category,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'height': y2 - y1,
                    'width': x2 - x1,
                    'occluded': occluded,
                    'truncated': truncated
                })
    instance_df = pd.DataFrame(instance_data)
    print(f'Instance DataFrame created with {len(instance_df)} entries.')
    return instance_df

#Converting to YOLO format

def df_to_yolo(df, dst_folder):
    '''
    Function to convert the image annotations to YOLO format from json
    inputs: 
    df: (pandas DataFrame)
    dst_folder: str|Path for path of the folder
    outputs
    '''
    #Default image size for normalisation
    img_w, img_h = 1280, 720
    category_id = {'traffic light':0,
                   'traffic sign':1,
                   'car':2,
                   'person':3,
                   'bus':4, 
                   'truck':5,
                   'rider':6, 
                   'bike':7,
                   'motor':8, 
                   'train':9}
    
    image_groups = df.groupby('image_id')
    for name, group in tqdm(image_groups):
        yolo_lines = []
        for _, row in group.iterrows():
            if row['occluded'] or row['truncated']:
                continue
            category = category_id[row['category']]
            x_center = ((row['x1'] + row['x2']) / 2) / img_w
            y_center = ((row['y1'] + row['y2']) / 2) / img_h
            width = (row['x2'] - row['x1']) / img_w
            height = (row['y2'] - row['y1']) / img_h
            yolo_lines.append(f"{category} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        txt_filename = os.path.splitext(name)[0] + '.txt'
        txt_path = os.path.join(dst_folder, txt_filename)
        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        

if __name__ == "__main__":
    # Load annotations
    with open('/home/nikhils/Desktop/object_detection_testing/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json', 'r') as f:
        train_annots = json.load(f)

    with open('/home/nikhils/Desktop/object_detection_testing/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json', 'r') as f:
        val_annots = json.load(f)

    # Create instance DataFrame
    train_instance_df = create_instance_df(train_annots)
    val_instance_df = create_instance_df(val_annots)

    # Copying the images to the yolo foldera
    src_train_img_folder = '/home/nikhils/Desktop/object_detection_testing/bdd100k_images_100k/bdd100k/images/100k/train'
    src_val_img_folder = '/home/nikhils/Desktop/object_detection_testing/bdd100k_images_100k/bdd100k/images/100k/val'
    dst_train_img_folder = '/home/nikhils/Desktop/object_detection_testing/YOLO_Data/train/images'
    dst_val_img_folder = '/home/nikhils/Desktop/object_detection_testing/YOLO_Data/val/images'
    dst_train_label_folder = '/home/nikhils/Desktop/object_detection_testing/YOLO_Data/train/labels'
    dst_val_label_folder = '/home/nikhils/Desktop/object_detection_testing/YOLO_Data/val/labels'

    # train_images = list(Path(src_train_img_folder).glob('*.jpg'))
    # val_images = list(Path(src_val_img_folder).glob('*.jpg'))
    # for img_path in tqdm(train_images):
    #     name = os.path.basename(img_path)
    #     shutil.copy(img_path, os.path.join(dst_train_img_folder, name))

    # for img_path in tqdm(val_images):
    #     name = os.path.basename(img_path)
    #     shutil.copy(img_path, os.path.join(dst_val_img_folder, name))

    # Convert to YOLO format and save labels
    df_to_yolo(train_instance_df, dst_train_label_folder)
    df_to_yolo(val_instance_df, dst_val_label_folder)

