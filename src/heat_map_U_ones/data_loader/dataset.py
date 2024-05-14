import os
import sys
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from config import CLASSES

from src.heat_map_U_ones.data_loader.custom_augmentation import CustomAugmentation

class HeatMapDataset(Dataset):
    def __init__(self, dataset_path, transform_type:str ='train'):
        # Read CSV
        # self.data_info = pd.read_csv(dataset_path,nrows=100)
        self.data_info = pd.read_csv(dataset_path)

        self.image_pathes=self.data_info.iloc[:,3]

        # Select Columns of Class
        self.data_info = self.data_info.loc[:, self.data_info.columns.isin(CLASSES)]
        
        # if (transform_type != 'train'):
        # Make sure CLASSES is a list of valid column names in self.data_info
        valid_columns = [col for col in CLASSES if col in self.data_info.columns]

        # Replace -1.0 with 1.0 in specified columns
        self.data_info[valid_columns] = self.data_info[valid_columns].replace(-1.0, 0.0) #[OLd NAns]
        # self.data_info[valid_columns] = self.data_info[valid_columns].replace(np.nan, -1.0)
          
        # Get the headers
        self.headers = self.data_info.columns.tolist()

        # Transform
        self.transform = CustomAugmentation(transform_type=transform_type)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_path=self.image_pathes[idx]
        
        # Getting image path  with parent path of current folder + image path
        img_path = os.path.join(os.getcwd(), img_path)
      
        # Fix Problem of \
        img_path = img_path.replace("\\", "/")
        #Read Image  
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        assert img is not None, f"Image at {img_path} is None"
        
        # convert the image from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #(3056, 2544, 3) [0-255]
        
        # get the labels and if column is 1 then it is true if empty then false
        # use Values bec this is a series [Drop Headers] + Covert dtyoe to ve float32 not obj
        labels=self.data_info.values[idx,:].astype('float32')
        labels = torch.FloatTensor(labels) #Tensor([13])
        
        # tranform image
        # 3channel 224x224 Normalized 0-1
        transformed_image = self.transform(image=img)["image"] #([3, 224, 224])
        return transformed_image, labels,img_path
