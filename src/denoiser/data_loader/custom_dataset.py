import numpy as np
import os
import cv2
import torch
import h5py
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from src.denoiser.data_loader.generate_noise import *
import matplotlib.pylab as plb
from src.denoiser.config import*

class CustomDataset(Dataset):
    def __init__(self, csv_file_path: str):
        self.data_info = pd.read_csv(csv_file_path, header=None)
        self.data_info =self.data_info.iloc[1:]
    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        try:
            img_path =self.data_info.iloc[idx, 3]
            img_path = os.path.join(os.getcwd(), img_path)
            image = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
            image=np.array(image).astype("float32")
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            if image is  None:
                assert image is not None, f"Image at {img_path} is None"
            choise= np.random.choice([0,1,2,3,4,5,6])
            choise=0
            if choise == 0:
                image,label= add_block_pixel_noise(image, probability=0.05)
            elif choise == 1:
                image,label= add_convolve_noise(image, sigma=1.5, sigma_noise=25) 
            elif choise == 2:
                image,label= add_keep_patch_noise(image, height_patch_size=image.shape[0]-25,width_patch_size=image.shape[1]-25 )
            elif choise == 3:
                image,label= add_pad_rotate_project_noise(image, max_rotation=2)
            elif choise == 4:
                image,label= add_gaussian_projection_noise(image, sigma=0.1)
            elif choise == 5:
                image,label= add_line_strip_noise(image, strip_width=5, intensity=0.5)
            else:
                image,label= np.copy(image),np.copy(image)
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            label = cv2.resize(label, (IMAGE_SIZE,IMAGE_SIZE))
            image = np.expand_dims(image, axis=0)
            label = np.expand_dims(label, axis=0)
            return image, label
        except Exception as e:
            print(e)
            return None
        
if __name__ == "__main__":
    # create a dataset object
    dataset = CustomDataset("datasets/train.csv")
    # create a dataloader object
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # get the first batch
    for image,label in dataloader:
        print(image.shape, label.shape)
        break
    # display the first image
    plb.imshow(image[0][0])
    plb.show()
    plb.imshow(label[0][0])
    plb.show()

