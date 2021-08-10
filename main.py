#imports
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os

from tensorflow import keras
from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4

HEIGHT, WIDTH = (512, 640)

#bounding box expansion
import json
import math
f = open('C:/Users/aryam/Documents/FLIR ADAS dataset/FLIR_ADAS_1_3/train/thermal_annotations.json')
data=json.load(f)
store={}
category_id=[0]*50
bbox=[[0,0,0,0]]*50
id=0
count=0
for i in data['annotations']:
   

    if(i['image_id']!=id or count==50):
       
        n=int(id)        
        store.setdefault(n, {}).update({'image_id':str(id)})
        store.setdefault(n, {}).update({'category_id':category_id})
        store.setdefault(n, {}).update({'bbox':bbox})
        id=i['image_id']
        category_id=[0]*50
        bbox=[[0,0,0,0]]*50
        count=0
       
    else:
        category_id[count]=i['category_id']
        height = i['bbox'][2]
        width = i['bbox'][3]
        HeightChange=math.floor(0.175*height)
        WidthChange=math.floor(0.175*width)
        Ypos=i['bbox'][0]
        Xpos=i['bbox'][1]
        Ypos=(Ypos-HeightChange)/640
        if Ypos<0:
            Ypos=0
        Xpos=(Xpos-WidthChange)/960
        if Xpos<0:
            Xpos=0
        height=(HeightChange+2*HeightChange)/640
        if height>640:
            height=1
        width=(width+2*WidthChange)/960
        if width>960:
            width=1
        bbox[count]=[Ypos, Xpos, height, width]
        count+=1
        
        
        
        
        
#importing an example image for the YOLOv4 model
img=cv2.imread("C:/Users/aryam/Documents/FLIR ADAS processed RGB images/FLIR ADAS RGB 2.jpg")
imgs = tf.expand_dims(img, axis=0)/255

#defining the model
model = YOLOv4(
    input_shape=(HEIGHT, WIDTH, 3),
    anchors = YOLOV4_ANCHORS, 
    num_classes=80, training=False, 
    yolo_max_boxes=50, 
    yolo_iou_threshold=0.5, 
    yolo_score_threshold=0.5,
    weights="C:/Users/aryam/Documents/yolov4.h5")

#defining the classes and the colours for the bounding boxes of the model
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush']
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
GTRUTH_COLOR = [0, 0, 0]

#getting predictions from the model
boxes, scores, classes, valid_detections = model.predict(imgs)

#Function to display the image and predictions
%config InlineBackend.figure_format = 'retina'

def plot_results(pil_img, boxes, scores, classes, category_id, bbox):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()

    for (xmin, ymin, xmax, ymax), score, cl in zip(boxes.tolist(), scores.tolist(), classes.tolist()):
        if score > 0:
          ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=COLORS[cl % 6], linewidth=3))
          text = f'{CLASSES[cl]}: {score:0.2f}'
          ax.text(xmin, ymin, text, fontsize=15,
                  bbox=dict(facecolor='yellow', alpha=0.5))
    for element in bbox:
        ax.add_patch(plt.Rectangle((element[1], element[0]), element[3], element[2], fill=False, color=GTRUTH_COLOR, linewidth=3))
    plt.axis('off')
    plt.show()
    
#calling the function to display the predictions
import pandas as pd
plot_results(
    imgs[0],
    boxes[0] * [WIDTH, HEIGHT, WIDTH, HEIGHT],
    scores[0],
    classes[0].astype(int),
    store[1]['category_id'],
    store[1]['bbox']
)
