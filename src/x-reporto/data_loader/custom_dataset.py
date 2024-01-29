import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import torch
from torchvision.transforms import functional as F
import pandas as pd 
import matplotlib.patches as patches
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from src.object_detector.data_loader.custom_augmentation import CustomAugmentation

class CustomDataset(Dataset):
    def __init__(self, dataset_path: str, transform_type:str ='train'):
        self.dataset_path = dataset_path # path to csv file
        
        self.transform_type = transform_type
        self.transform = CustomAugmentation(transform_type=self.transform_type)
        # read the csv file
        self.data_info = pd.read_csv(dataset_path, header=None)
        # remove the first row (column names)
        self.data_info = self.data_info.iloc[1:]

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # get the image path
        img_path = self.data_info.iloc[idx, 3]

        # read the image with parent path of current folder + image path
        # img_path = os.path.join("datasets/", img_path)
        img_path = os.path.join(os.getcwd(), img_path)
        img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
        assert img is not None, f"Image at {img_path} is None"
        
        # get the bounding boxes
        bboxes = self.data_info.iloc[idx, 4]

        # convert the string representation of bounding boxes into list of list
        bboxes = eval(bboxes)

        # get the bbox_labels
        bbox_labels = self.data_info.iloc[idx, 5]

        # convert the string representation of labels into list
        bbox_labels = np.array(eval(bbox_labels))

        # get the bbox_labels
        bbox_phrases = self.data_info.iloc[idx, 6]
        # get the bbox_labels
        bbox_phrase_exists = self.data_info.iloc[idx, 7]
        # get the bbox_labels
        bbox_is_abnormal = self.data_info.iloc[idx, 8]
        # tranform image
        transformed = self.transform(image=img, bboxes=bboxes, class_labels=bbox_labels)
        transformed_image = transformed["image"]
        transformed_bboxes = transformed["bboxes"]
        transformed_bbox_labels = transformed["class_labels"]
        # convert the bounding boxes to tensor
        transformed_bboxes = torch.as_tensor(transformed_bboxes, dtype=torch.float32)
        transformed_bbox_labels = torch.as_tensor(transformed_bbox_labels, dtype=torch.int64)
        #object_detector_targets
        object_detector_sample = {}
        object_detector_sample["image"]=transformed_image
        object_detector_sample["boxes"] = transformed_bboxes
        object_detector_sample["bbox_labels"] = transformed_bbox_labels

        #classifier_targets
        classifier_sample= dict(object_detector_sample)
        classifier_sample["bbox_phrase_exists"]=bbox_phrase_exists
        classifier_sample["bbox_is_abnormal"]=bbox_is_abnormal

        #language_model_targets
        language_model_sample=dict(classifier_sample)
        language_model_sample["bbox_phrases"]=bbox_phrases

        return object_detector_sample,classifier_sample,language_model_sample
