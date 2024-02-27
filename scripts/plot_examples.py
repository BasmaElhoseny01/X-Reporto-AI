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

# load the dataset
dataset_path = "datasets/valid.csv"
dataset = CustomDataset(dataset_path,transform_type='val')

data_info = pd.read_csv(dataset_path, header=None)
    # remove the first row (column names)
data_info = data_info.iloc[1:]

def plot_image_with_boxes(index):
    # get the image and bounding boxes
    object_detector_sample,selection_classifier_sample,abnormal_classifier_sample,language_model_sample = dataset[index]

    img = object_detector_sample['image']
    bboxes = object_detector_sample['bboxes']

    # convert the tensor to numpy array
    img = img.permute(1, 2, 0).numpy()
    img = img * 255


    # plot the image
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # plot the bounding boxes
    for i in range(len(bboxes)):
        x, y, w, h = bboxes[i]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, "", fontsize=12, color='white', weight='bold', backgroundcolor='red')

    plt.show()

    # save the image
    plt.savefig('images/output' + str(index) + '.png')

    # close the image
    plt.close()

def plot_image_without_augmentation(data_info,index):

    # get the image path
    img_path = data_info.iloc[index, 3]

    # read the image with parent path of current folder + image path
    img_path = os.path.join(os.getcwd(), img_path)
    img = cv2.imread(img_path)
    assert img is not None, f"Image at {img_path} is None"

    # get the bounding boxes
    bboxes = data_info.iloc[index, 4]

    # convert the string representation of bounding boxes into list of list
    bboxes = eval(bboxes)

    
    # plot the image
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # plot the bounding boxes
    for i in range(len(bboxes)):
        x, y, w, h = bboxes[i]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, "", fontsize=12, color='white', weight='bold', backgroundcolor='red')

    plt.show()

    # save the image
    plt.savefig('images/true_output' + str(index) + '.png')

    # close the image
    plt.close()
if __name__ == "__main__":

    for i in range(1,20):
        # generate random index in range of dataset
        index = np.random.randint(0, len(dataset))
        plot_image_with_boxes(index)
        plot_image_without_augmentation(data_info,index)