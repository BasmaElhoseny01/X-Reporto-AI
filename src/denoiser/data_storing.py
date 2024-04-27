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

def store_many_hdf5(h5file, images, labels):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, H, W, 3) to be stored
        labels       labels array, (N, H, W, 3) to be stored
    """
    dataset = h5file.create_dataset(
        "images", np.shape(images), dtype=np.float32, data=images
    )
    meta_set = h5file.create_dataset(
        "labels", np.shape(labels), dtype=np.float32, data=labels
    )

def read_many_hdf5(h5file):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read
        Returns:
        ----------
        images      images array, (N, H, W, 3)) to be stored
        labels      associated meta data, int label (N, H, W, 3)
    """
    images = np.array(h5file["images"]).astype("float32")
    labels = np.array(h5file["labels"]).astype("float32")

    return images, labels

def generate_noise(label):
    #apply noise
    image = label
    return label, image

def proccess_images(csv_file_path,type_data_to_save):
    data_info = pd.read_csv(csv_file_path, header=None)
    # remove the first row (column names)
    data_info = data_info.iloc[1:]
    h5file = h5py.File(osp.join(save_data_h5_dir, type_data_to_save+'.h5'), 'w')
    images=[]
    labels=[]
    for idx in tqdm(range(len(data_info))):
        img_path =data_info.iloc[idx, 3]
        img_path = os.path.join(os.getcwd(), img_path)
        image = cv2.imread(img_path)
        if image is  None:
            print(f"Image at {img_path} is None")
            continue
        image=np.array(image).astype("float32")
        label ,image= generate_noise(image)
        image = cv2.resize(image, (512, 512))
        label = cv2.resize(label, (512,512))
        images.append(image)
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    store_many_hdf5(h5file, images, labels)
    h5file.close()



if __name__ == '__main__':
    proccess_images('datasets/train.csv','train')
    # h5file = h5py.File(osp.join(dataset_dir, 'train.h5'), 'r')
    # images,labels = read_many_hdf5(h5file)
    # # Image._show(Image.fromarray(labels[0]))
    # print(images[0].shape)
    # labels[0]= labels[0] / 255.0
    # images[0]= images[0] / 255.0
    # plb.imshow(images[0])
    # plb.show()

