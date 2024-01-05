import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import random
import pandas as pd 
from torchvision.transforms import v2

class F_RCNNDataset(Dataset):
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

        # resize the image with the bounding boxes accordingly to 512x512
        img, bboxes = self.resize(img, bboxes, (512, 512))

        # get the labels
        labels = self.data_info.iloc[idx, 5]

        # convert everything into a torch.Tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int32)



        if self.transform:
            transform_dict = self.transform(img,bboxes,labels)
            img = transform_dict['image']
            bboxes = transform_dict['bboxes']
            labels = transform_dict['labels']

        # define the bounding box
        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        return img, target
    def resize(self, img, bboxes, size):
        # resize the image
        img = cv2.resize(img, size)

        # resize the bounding boxes
        height, width, _ = img.shape
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * (size[0] / width)
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * (size[1] / height)

        return img, bboxes

# implement transforms as augmentation with gaussian noise, random rotation

class Augmentation(object):
    def __init__(self, noise_std=0.1, rotate_angle=10):
        self.noise_std = noise_std
        self.rotate_angle = rotate_angle

        self.transform = v2.Compose([
            v2.ToTensor(),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize with imagenet mean and std
        ])

        self.transforms = v2.Compose([
            # resize the image to 512x512 with same aspect ratio and pad the image with zeros if necessary
            v2.Resize((512, 512)),

            # randomly rotate the image
            v2.RandomRotation(self.rotate_angle),

            # add gaussian noise to the image
            # v2.GaussianNoise(self.noise_std),

            # convert the image to tensor and normalize with imagenet mean and std
            v2.ToTensor(),
            v2.Normalize(0.485, 0.229) # normalize with imagenet mean and std
        ],
        # resize the bounding boxes accordingly
        bbox_params=v2.BboxParams(format='pascal_voc', label_fields=['labels'])
        )

    def __call__(self, img,bboxes,labels):
        
        # apply the transforms to the image and the bounding boxes
        return self.transforms(image=img, bboxes=bboxes,class_labels=labels)

