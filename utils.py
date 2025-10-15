'''
General functions for data processing, analysis and visualization.
'''
import numpy as np
import pandas as pd
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

def load_data(file_path):
    '''
    Loads JSON data from the specified file path and returns it as a Python object.
    input: file_path (str)- Path to the JSON file.
    output: annots (list)- List of annotations loaded from the JSON file.
    '''
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, 'r') as f:
        annots = json.load(f)
    print(f'JSON file loaded!')
    return annots

def create_base_df(annots):
    '''
    Creates a base DataFrame from the annotations list.
    input: annots (list)- List of annotations loaded from the JSON file.
    output: base_df (DataFrame)- DataFrame containing image-level information.
    '''
    base_data = []
    for annot in annots:
        img_info = {
            'name': annot['name'],
            'weather': annot['attributes']['weather'],
            'scene': annot['attributes']['scene'],
            'timeofday': annot['attributes']['timeofday'],
            'num_objects': len(annot['labels'])
        }
        base_data.append(img_info)
    base_df = pd.DataFrame(base_data)
    return base_df

def create_instance_df(annots):
    '''
    Creates a DataFrame containing instance-level annotations from the provided annotations.
    input: annots (list)- List of annotations loaded from the JSON file.
    output: instance_df (DataFrame)- DataFrame containing instance-level annotations.
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

def draw_count_plot(df, column, title, xlabel, ylabel='Count', axes=None, hue = None):
    '''
    Draws a count plot for the specified column in the DataFrame.
    input: df (DataFrame)- DataFrame containing the data.
           column (str)- Column name to plot.
           title (str)- Title of the plot.
           xlabel (str)- Label for the x-axis.
           ylabel (str)- Label for the y-axis.
           axes (matplotlib.axes.Axes)- Axes object to plot on. If None, creates a new figure.
           rotation (int)- Rotation angle for x-axis labels.
    output: axes (matplotlib.axes.Axes)- Axes object with the plot.
    '''
    sns.countplot(data=df, x=column, hue=hue, order=df[column].value_counts().index, ax=axes)
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    return axes

def draw_hist_plot(df, column, title, xlabel, ylabel='Frequency', axes=None, bins=30):
    '''
    Draws a histogram for the specified column in the DataFrame.
    input: df (DataFrame)- DataFrame containing the data.
           column (str)- Column name to plot.
           title (str)- Title of the plot.
           xlabel (str)- Label for the x-axis.
           ylabel (str)- Label for the y-axis.
           axes (matplotlib.axes.Axes)- Axes object to plot on. If None, creates a new figure.
           bins (int)- Number of bins for the histogram.
    output: axes (matplotlib.axes.Axes)- Axes object with the plot.
    '''
    sns.histplot(data=df, x=column, bins=bins, ax=axes)
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    return axes

def visualise_bounding_boxes(name, instance_df, base_path):
    '''
    Visualizes bounding boxes on the image.
    inputs: name - image file name
            instance_df - dataframe containing bounding box details
            base_path - path to the directory containing images
    output: image with bounding boxes drawn on it
    '''
    image = cv2.imread(os.path.join(base_path, name))
    for index, row in instance_df[instance_df['image_id'] == name].iterrows():
        left_top = (int(row['x1']), int(row['y1']))
        right_bottom = (int(row['x2']), int(row['y2']))
        cv2.rectangle(img=image, pt1=left_top, pt2=right_bottom, color=(0, 255, 0), thickness=2)
        cv2.putText(image, row['category'], (int(row['x1']), int(row['y1']-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return image

def plot_images(image, instance_df, base_path):
    '''
    Plots images with bounding boxes.
    inputs: image_list - list of image file names
            instance_df - dataframe containing bounding box details
            base_path - path to the directory containing images
            title - title for the plot
    output: None
    '''
    fig, axes = plt.subplots(1, 3, figsize=(30, 15))
    img_1 = visualise_bounding_boxes(image, instance_df, base_path)
    axes[0].imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))
    axes[0].set_title('All Objects', fontsize=16)
    axes[0].axis('off')
    instance_df = instance_df[instance_df['occluded'] == 0]
    img2 = visualise_bounding_boxes(image, instance_df, base_path)
    axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Non-Occluded Objects', fontsize=16)
    axes[1].axis('off')
    instance_df = instance_df[instance_df['truncated'] == 0]
    img3 = visualise_bounding_boxes(image, instance_df, base_path)
    axes[2].imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Non-Truncated Objects', fontsize=16)
    axes[2].axis('off')
    plt.show()
