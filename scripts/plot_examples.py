# load csv file and plot image with bounding boxes
import pandas as pd
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ast
import torch
from src.x_reporto.data_loader.custom_dataset import CustomDataset
from src.x_reporto.data_loader.tokenizer import Tokenizer
from src.language_model.GPT2.config import Config
from src.object_detector.data_loader.custom_augmentation import CustomAugmentation
from torchvision.transforms import v2


# load the dataset
dataset_path = "datasets/annotations.csv"
dataset = CustomDataset(dataset_path)

# get the image and bounding boxes
img, bboxes, bbox_labels, bbox_phrases = dataset[0]

# plot the image
fig, ax = plt.subplots(1)
ax.imshow(img)

# plot the bounding boxes
for i in range(len(bboxes)):
    x, y, w, h = bboxes[i]
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y, bbox_phrases[i], fontsize=12, color='white', weight='bold', backgroundcolor='red')

plt.show()
