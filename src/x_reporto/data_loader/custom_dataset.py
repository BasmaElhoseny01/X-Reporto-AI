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
import ast
from src.object_detector.data_loader.custom_augmentation import CustomAugmentation
from src.x_reporto.data_loader.tokenizer import Tokenizer
from src.language_model.GPT2.config import Config
class CustomDataset(Dataset):
    def __init__(self, dataset_path: str, transform_type:str ='train',checkpoint:str="healx/gpt-2-pubmed-medium"):
        self.dataset_path = dataset_path # path to csv file
        
        self.transform_type = transform_type
        self.transform = CustomAugmentation(transform_type=self.transform_type)
        # read the csv file
        self.data_info = pd.read_csv(dataset_path, header=None)
        # remove the first row (column names)
        self.data_info = self.data_info.iloc[1:]
        self.tokenizer = Tokenizer(checkpoint)

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
        object_detector_sample["bboxes"] = transformed_bboxes
        object_detector_sample["bbox_labels"] = transformed_bbox_labels

        #classifier_targets
        # Safely evaluate the string and convert it to a Python list
        bbox_phrase_exists = ast.literal_eval(bbox_phrase_exists)

        # Convert the Python list to a PyTorch tensor
        bbox_phrase_exists = torch.tensor(bbox_phrase_exists, dtype=torch.bool)

        selection_classifier_sample= {}
        selection_classifier_sample["bbox_phrase_exists"]=bbox_phrase_exists

        #classifier_targets
        # Safely evaluate the string and convert it to a Python list
        bbox_is_abnormal = ast.literal_eval(bbox_is_abnormal)

        # Convert the Python list to a PyTorch tensor
        bbox_is_abnormal = torch.tensor(bbox_is_abnormal, dtype=torch.bool)
         
        abnormal_classifier_sample= {}
        abnormal_classifier_sample["bbox_is_abnormal"]=bbox_is_abnormal

        #language_model_targets
        language_model_sample={}
        tokenize_phrase = self.tokenizer(bbox_phrases)  
        language_model_sample["bbox_phrases"]=bbox_phrases
        padded_lists_by_pad_token = [tokenize_phrase_lst + [tokenize_phrase[0]] * (Config.max_seq_len - len(tokenize_phrase_lst)) for tokenize_phrase_lst in tokenize_phrase["input_ids"]]
        padded_lists_by_ignore_token = [tokenize_phrase_lst + [Config.ignore_index] * (Config.max_seq_len - len(tokenize_phrase_lst)) for tokenize_phrase_lst in tokenize_phrase["input_ids"]]
        language_model_sample["input_ids"]=padded_lists_by_pad_token
        language_model_sample["label_ids"]=padded_lists_by_ignore_token
        padded_mask = [mask_phrase_lst + [0] * (Config.max_seq_len - len(mask_phrase_lst)) for mask_phrase_lst in tokenize_phrase["attention_mask"]]
        language_model_sample["attention_mask"]=padded_mask
        return object_detector_sample,selection_classifier_sample,abnormal_classifier_sample,language_model_sample
