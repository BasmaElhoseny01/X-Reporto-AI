import os
import sys
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from src.heat_map_U_ones.data_loader.custom_augmentation import CustomAugmentation

class HeatMapDataset(Dataset):
    def __init__(self, dataset_path, transform_type:str ='train'):
        # Read CSV
        self.data_info = pd.read_csv(dataset_path)
        

        # Print the second row
        print(self.data_info.iloc[0, 0:15])
        
        print(self.data_info.iloc[1, 0:15])
        
        
        # Print the third row, columns 0 to 14
        print(self.data_info.iloc[2, 0:15])
#         sys.exit()
        
        # Get the headers
        self.headers = self.data_info.columns.tolist()
        
        # Replace Uncertain Labels
        self.data_info.iloc[:, 2:15] = self.data_info.iloc[:, 2:15].replace(np.nan, 0.0)
        self.data_info.iloc[:, 2:15] = self.data_info.iloc[:, 2:15].replace(-1.0, 1.0)
                
        #Data Types of each column print(self.data_info.dtypes)              
        #print(self.data_info.iloc[76, :])
        #sys.exit()
        
        self.transform = CustomAugmentation(transform_type=transform_type)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_path = self.data_info.iloc[idx,16]
        
        # Getting image path  with parent path of current folder + image path
        img_path = os.path.join(os.getcwd(), img_path)
      
        # Fix Problem of \
        img_path = img_path.replace("\\", "/")
        
        #Read Image  
        img = cv2.imread(img_path)
        assert img is not None, f"Image at {img_path} is None"
        # convert the image from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0 #(3056, 2544, 3) [0-1]
        
        # get the labels and if column is 1 then it is true if empty then false
        # use Values bec this is a series [Drop Headers] + Covert dtyoe to ve float32 not obj
        labels=self.data_info.iloc[idx, 2:15].values.astype('float32')
        labels = torch.FloatTensor(labels) #Tensor([13])
        
        # tranform image
        # 3channel 224x224 Normalized 0-1
        transformed_image = self.transform(image=img)["image"] #([3, 224, 224])  # I think [-3,-3]
        return transformed_image, labels