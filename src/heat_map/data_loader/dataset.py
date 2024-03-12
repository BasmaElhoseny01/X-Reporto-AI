import os
import sys
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.heat_map.data_loader.custom_augmentation import CustomAugmentation

class HeatMapDataset(Dataset):
    def __init__(self, dataset_path, transform_type:str ='train'):
        self.dataset_path = dataset_path # path to csv file
        self.transform_type = transform_type
        self.transform = CustomAugmentation(transform_type=self.transform_type)
        self.data_info = pd.read_csv(dataset_path, header=None)
        self.data_info = self.data_info.iloc[1:]

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # imag_path_col = self.data_info.columns.get_loc('mimic_image_file_path')

        img_path = self.data_info.iloc[idx,16]
        # read the image with parent path of current folder + image path
        # img_path = os.path.join("datasets/", img_path)
        img_path = os.path.join(os.getcwd(), img_path)

        # Fix Problem of \
        img_path = img_path.replace("\\", "/")
        
        # read the image
        # Fix Problem of \

        img_path = img_path.replace("\\", "/")
        img_path = img_path.replace('\files', "/files")
        img = cv2.imread(img_path)
        assert img is not None, f"Image at {img_path} is None"
        # convert the image from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # get the labels and if column is 1 then it is true if empty then false
        labels = self.data_info.iloc[idx, 2:15]
        # make labels as numpy array of bool values true if value is 1 else false
        if labels.isnull().values.any():
            labels = labels.fillna(0)
        # replace the -1 values with 0
        labels = labels.replace("-1.0", 0)
        labels = labels.replace("0.0", 0)


        labels = labels.astype(bool)
        labels = labels.to_numpy(dtype=bool)
  
        labels=torch.as_tensor(labels, dtype=torch.bool)
        # tranform image
        transformed = self.transform(image=img)
        transformed_image = transformed["image"]
        return transformed_image, labels
        