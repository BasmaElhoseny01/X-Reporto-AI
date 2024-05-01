import os
import os.path as osp
import sys
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import cv2
import pandas as pd
save_data_h5_dir = 'datasets/'
from PIL import Image
import h5py
import numpy as np
from tqdm import tqdm
from src.denoiser.data_loader.generate_noise import *

def store_many_hdf5(h5file, image, label,idx):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, H, W, 3) to be stored
        labels       labels array, (N, H, W, 3) to be stored
    """
    dataset = h5file.create_dataset(
        "image"+str(idx), np.shape(image), dtype=np.float32, data=image
    )
    meta_set = h5file.create_dataset(
        "label"+str(idx), np.shape(label), dtype=np.float32, data=label
    )

def read_many_hdf5(h5file,idx):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read
        Returns:
        ----------
        images      images array, (N, H, W, 3)) to be stored
        labels      associated meta data, int label (N, H, W, 3)
    """
    image = np.array(h5file["image"+str(idx)]).astype("float32")
    label= np.array(h5file["label"+str(idx)]).astype("float32")
    return image, label

def generate_noise(label):
    #apply noise
    image = label
    return label, image

def proccess_images(csv_file_path,type_data_to_save):
    data_info = pd.read_csv(csv_file_path, header=None)
    # remove the first row (column names)
    data_info = data_info.iloc[1:]
    h5file = h5py.File(osp.join(save_data_h5_dir, type_data_to_save+'.h5'), 'w')
    for idx in tqdm(range(len(data_info))):
        img_path =data_info.iloc[idx, 3]
        img_path = os.path.join(os.getcwd(), img_path)
        image_org = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
        image_org=np.array(image_org).astype("float32")
        image_org = cv2.resize(image_org, (1024, 1024))
        if image_org is  None:
            print(f"Image at {img_path} is None")
            continue
        # for i in range(3):
        image = np.copy(image_org)
        choise= np.random.choice([0,1,2,3,4,5,6])
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
        image = cv2.resize(image, (1024, 1024))
        label = cv2.resize(label, (1024,1024))
        store_many_hdf5(h5file, image, label,idx)
    h5file.close()



if __name__ == '__main__':
    proccess_images('datasets/train.csv','train_noise_new')
    h5file = h5py.File(osp.join(save_data_h5_dir, 'train_noise_new.h5'), 'r')
    image,label = read_many_hdf5(h5file,4)
    # Image._show(Image.fromarray(labels[0]))
    print(image.shape)
    plb.imshow(image)
    plb.show()


#  python -m src.denoiser.data_loader.data_storing 