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
from config import *

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
            # replace \ with / for windows
            img_path = img_path.replace("\\", "/")
            image = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
            image=np.array(image).astype("float32")
            # image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            image=self.longest_max_size(image=image,max_size=IMAGE_SIZE)
            if image is  None:
                assert image is not None, f"Image at {img_path} is None"
            choise= np.random.choice([0,1,2,3,4])
            # choise=3
            if choise == 0:
                image,label= add_block_pixel_noise(image, probability=0.05)
            elif choise == 1:
                image,label= add_convolve_noise(image, sigma=1, sigma_noise=18) 
            elif choise == 2:
                image,label= add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05)
            elif choise == 3:
                image,label= add_gaussian_projection_noise(image, sigma=20)
            else:
                image,label= np.copy(image),np.copy(image)
            # image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            # label = cv2.resize(label, (IMAGE_SIZE,IMAGE_SIZE))

            image = np.expand_dims(image, axis=0)
            label = np.expand_dims(label, axis=0)
            # check for image and label type is float32
            if image.dtype != np.float32:
                image = image.astype(np.float32)
            if label.dtype != np.float32:
                label = label.astype(np.float32)
            image /= 255.0
            label /= 255.0
            
            return image, label
        except Exception as e:
            print(e)
            return None

    def longest_max_size(self,image, max_size, interpolation=cv2.INTER_AREA):
        """
        Resize the image such that the longest side matches max_size, maintaining the aspect ratio.

        Args:
        image (numpy.ndarray): Input image.
        max_size (int): Desired size of the longest side of the image.
        interpolation (int): Interpolation method (default is cv2.INTER_AREA).

        Returns:
        numpy.ndarray: Resized image.
        """
        # Get the dimensions of the image
        height, width = image.shape[:2]

        # Determine the scaling factor
        if height > width:
            scale_factor = max_size / height
        else:
            scale_factor = max_size / width

        # Calculate new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        
        return resized_image
        
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

