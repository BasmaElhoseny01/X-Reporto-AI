import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import torch
from PIL import Image
from torchvision.transforms import functional as F
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

        # read the image with parent path of current folder + image path
        img_path = os.path.join(os.getcwd(), img_path)
        img = cv2.imread(img_path,0)
        assert img is not None
        
        # get the bounding boxes
        bboxes = self.data_info.iloc[idx, 4]

        # convert the string representation of bounding boxes into list of list
        bboxes = eval(bboxes)

        # resize the image with the bounding boxes accordingly to 512x512
        # img, bboxes = self.resize(img, bboxes, (512, 512))
        img,bboxes = ResizeAndPad((512,512))(img,bboxes)

        # get the labels
        labels = self.data_info.iloc[idx, 5]

        # convert the string representation of labels into list
        labels = eval(labels)

        if self.transform:
            transform_dict = self.transform(img,bboxes,labels)
            img = transform_dict['image']
            bboxes = transform_dict['bboxes']
            labels = transform_dict['labels']
        else:
            img = torch.as_tensor(img, dtype=torch.float32)
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int32)

        # define the bounding box
        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        return img, target

# implement transforms as augmentation with gaussian noise, random rotation

class Augmentation(object):
    def __init__(self, noise_std=0.1, rotate_angle=10):
        self.noise_std = noise_std
        self.rotate_angle = rotate_angle

        self.transforms = v2.Compose([
            # randomly rotate the image
            v2.RandomRotation(self.rotate_angle),
            v2.ToTensor(),

            # add gaussian noise to the image
            AddGaussianNoise(0,self.noise_std),

            # convert the image to tensor and normalize with imagenet mean and std
            v2.Normalize(0.485, 0.229), # normalize with imagenet mean and std
        ],
        # resize the bounding boxes accordingly
        bbox_params=v2.BboxParams(format='pascal_voc', label_fields=['labels'])
        )

    def __call__(self, img,bboxes,labels):
        # apply the transforms to the image and the bounding boxes
        return self.transforms(image=img, bboxes=bboxes,class_labels=labels)


# Define a custom transform to add Gaussian noise
class AddGaussianNoise(object):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # Add Gaussian noise to the tensor
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


# Define a custom transform to resize and pad the image and update bounding boxes
class ResizeAndPad(object):
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size

    def __call__(self, image, boxes=None):
        # Resize the  image while maintaining aspect ratio
        width, height = image.shape[0],image.shape[1]
        aspect_ratio = width / height
        long_side = max(self.target_size)
        short_side = min(self.target_size)

        new_width = 0
        new_height = 0
        if width > height:
            new_width = long_side
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = long_side
            new_width = int(new_height * aspect_ratio)

        resized_image = cv2.resize(image, (new_width, new_height))
        # Create a new black grayscale image with the desired size
        new_image = np.zeros(self.target_size, dtype=np.float32)
        # add the resized image to top left corner of the new image
        new_image[:new_height, :new_width] = resized_image
        # change the bounding boxes accordingly
        if boxes is not None:
            boxes = np.array(boxes)
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * (new_width / width)
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * (new_height / height)
        return new_image, boxes