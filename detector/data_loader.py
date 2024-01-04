import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import random
import pandas as pd 

class F-RCNNDataset(Dataset):
    def __init__(self, dataset_path: str, transform =None):
        self.dataset_path = dataset_path # path to csv file
        self.transform = transform

        # read the csv file
        self.data_info = pd.read_csv(dataset_path, header=None)
        # row contains (subject_id,	study_id, image_id, mimic_image_file_path, bbox_coordinates list of list, bbox_labels list,
        #               bbox_phrases list of str, bbox_phrase_exists list of booleans, bbox_is_abnormal list of booleans)


    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # get the image path
        img_path = self.data_info.iloc[idx, 3]

        # read the image
        img = cv2.imread(img_path)
        
        # get the bounding boxes
        bboxes = self.data_info.iloc[idx, 4]

        # get the labels
        labels = self.data_info.iloc[idx, 5]

        # convert everything into a torch.Tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # define the bounding box
        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels

        if self.transform:
            img = self.transform(img)

        return img, target

# implement transforms as augmentation with gaussian noise, random rotation

class Augmentation(object):
    def __init__(self, noise_std=0.1, rotate_angle=10):
        self.noise_std = noise_std
        self.rotate_angle = rotate_angle

    def __call__(self, img):
        # add gaussian noise
        img = img + torch.randn(img.size()) * self.noise_std

        # random rotation
        angle = random.randint(-self.rotate_angle, self.rotate_angle)
        img = img.rotate(angle)

        return img

