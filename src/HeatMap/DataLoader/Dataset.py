import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.HeatMap.DataLoader.custom_augmentation import CustomAugmentation

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

        img_path = self.data_info.iloc[idx,17]
        # read the image with parent path of current folder + image path
        # img_path = os.path.join("datasets/", img_path)
        img_path = os.path.join(os.getcwd(), img_path)

        # read the image
        img = cv2.imread(img_path)
        assert img is not None, f"Image at {img_path} is None"

        labels = self.data_info.iloc[idx, 2:16]
        # make labels as numpy array of bool values true if value is 1 else false
        labels = labels.to_numpy(dtype=bool)
        labels=torch.as_tensor(labels, dtype=torch.bool)
        # tranform image
        transformed = self.transform(image=img)
        transformed_image = transformed["image"]
        return transformed_image, labels
        
        
