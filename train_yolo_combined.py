import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def train_model(data_yaml, model_name='yolov11m.pt', epochs=100, batch_size=32, img_size=640, save_dir='runs/train'):
    '''
    Function to train a YOLOv8 model using the ultralytics library.
    
    Parameters:
    - data_yaml (str or Path): Path to the data configuration YAML file.
    - model_name (str): Pre-trained model name or path. Default is 'yolov8n.pt'.
    - epochs (int): Number of training epochs. Default is 50.
    - batch_size (int): Batch size for training. Default is 16.
    - img_size (int): Image size for training. Default is 640.
    - save_dir (str or Path): Directory to save training results. Default is 'runs/train'.
    
    Returns:
    - model: The trained YOLOv8 model.
    '''
    
    # Load the pre-trained YOLOv8 model
    model = YOLO(model_name)
    
    # Train the model with specified parameters
    model.train(data=data_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                save=True,
                project=save_dir,
                single_cls = True)
    
    return model

if __name__ == "__main__":
    data_yaml = '/home/nikhils/Desktop/object_detection_testing/baseline.yaml'  # Path to your data configuration file
    trained_model = train_model(data_yaml, model_name='yolo11m.pt', epochs=100, batch_size=32, img_size=640, save_dir='runs/train')