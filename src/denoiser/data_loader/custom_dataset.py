import numpy as np
import os
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from src.denoiser.data_loader.generate_noise import *
import matplotlib.pylab as plb
import albumentations as A
# import configuration file
from config import *
from src.denoiser.config import*

class CustomDataset(Dataset):
    def __init__(self, csv_file_path: str):
        """
        Custom dataset for loading images and applying transformations and noise.

        Args:
            csv_file_path (str): Path to the CSV file containing image paths.
        """
        # read the csv file
        self.data_info = pd.read_csv(csv_file_path, header=None)
        # first column contains the image paths
        self.data_info =self.data_info.iloc[1:]
        # Define transformations using Albumentations library
        self.transform =  A.Compose(
                        [
                            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
                        ]
                    )
        self.transform2 =  A.Compose(
                        [
                            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE,border_mode= cv2.BORDER_CONSTANT,value=0),
                        ]
                    )
        
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data_info)

    def __getitem__(self, idx):
        """
        Loads an image and applies transformations and noise.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the noisy image and the original image.
        """
        try:
            # Read image path from the dataframe
            img_path =self.data_info.iloc[idx, 3]
            img_path = os.path.join(os.getcwd(), img_path)
            # replace \ with / for windows
            img_path = img_path.replace("\\", "/")
            # Read image using OpenCV
            image = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
            image=np.array(image).astype("float32")
            # Check if image is None
            if image is  None:
                assert image is not None, f"Image at {img_path} is None"
            # Randomly choose noise type to apply
            choise= np.random.choice([0,1,2,3,4,5])
            image=self.transform(image=image)["image"] # Apply first transformation
            # Apply different noise types based on choice
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

            # Apply second transformation
            image=self.transform2(image=image)["image"]
            label=self.transform2(image=label)["image"]

            # Convert types to float32 and normalize
            if image.dtype != np.float32:
                image = image.astype(np.float32)
            if label.dtype != np.float32:
                label = label.astype(np.float32)
            # Add channel dimension
            image = np.expand_dims(image, axis=0)
            label = np.expand_dims(label, axis=0)
            # Normalize the image and label
            image /= 255.0
            label /= 255.0
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
    # display the first image and label
    print(image.max(), image.min())
    print(label.max(), label.min())
    plb.imshow(image[0][0],cmap="gray")
    plb.show()
    plb.imshow(label[0][0],cmap="gray")
    plb.show()

# python -m src.denoiser.data_loader.custom_dataset