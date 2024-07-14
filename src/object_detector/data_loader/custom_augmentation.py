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
from PIL import Image
import random
from torchvision import transforms

# Parameters
IMAGE_INPUT_SIZE=512
MEAN=0.474
STD=0.301
ANGLE=2

# implement transforms as augmentation with gaussian noise, random rotation

class CustomAugmentation(object):
    def __init__(self, transform_type:str ='train'):
        if(transform_type == 'train'or transform_type == 'val'):
            self.transform=TransformLibrary(transform_type)
        else:
            self.transform=CustomTransform(transform_type)

    def __call__(self,image,bboxes,class_labels):
        return self.transform(image=image, bboxes=bboxes, class_labels=class_labels)


class TransformLibrary(object):
    
    def __init__(self, transform_type:str ='train'):
        if (transform_type == 'train'):
            self.transform =A.Compose(
                [
                    # we want the long edge of the image to be resized to IMAGE_INPUT_SIZE, and the short edge of the image to be padded to IMAGE_INPUT_SIZE on both sides,
                    # such that the aspect ratio of the images are kept, while getting images of uniform size (IMAGE_INPUT_SIZE x IMAGE_INPUT_SIZE)
                    # LongestMaxSize: resizes the longer edge to IMAGE_INPUT_SIZE while maintaining the aspect ratio
                    # INTER_AREA works best for shrinking images
                    A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
                    A.GaussNoise(),
                    #  rotate between -2 and 2 degrees
                    A.Affine(mode=cv2.BORDER_CONSTANT, cval=0, rotate=(-ANGLE, ANGLE),keep_ratio=True),

                    # PadIfNeeded: pads both sides of the shorter edge with 0's (black pixels)
                    A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
                    A.Normalize(mean=MEAN, std=STD),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['class_labels'])
                )
        else:
            self.transform = A.Compose(
                        [
                            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
                            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
                            A.Normalize(mean=MEAN, std=MEAN),
                            ToTensorV2(),
                        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['class_labels'])
                    )
    def __call__(self,image,bboxes,class_labels):
        #print("TransformLibrary called in call")
        return self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
# Custom function to add Gaussian noise to an image
def add_gaussian_noise(img, mean=0, std=25):
    return img + torch.randn_like(img) * std + mean

# Custom function to apply transformations to the bounding box
def apply_transform_to_bbox(bbox, transform, image_size):
    # Convert bounding box to relative coordinates (0 to 1)
    bbox_normalized = [bbox[i] / image_size[i % 2] for i in range(4)]

    # Convert to NumPy array
    bbox_np = np.array(bbox_normalized)

    # Apply the same transformation to the bounding box
    bbox_transformed_np = transform(bbox_np)

    # Convert back to absolute coordinates
    bbox_transformed = (bbox_transformed_np * image_size).astype(int).tolist()

    return bbox_transformed

class CustomTransform(object):
    
    def __init__(self, transform_type:str ='custom_train'):
       # Define a composite transform
        if(transform_type == 'custom_train'):
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE),interpolation=cv2.INTER_AREA),                # Resize the image
                transforms.Pad(IMAGE_INPUT_SIZE),   # Pad the image with reflection padding
                transforms.RandomRotation(ANGLE),                 # Rotate the image randomly by up to 10 degrees
                transforms.ToTensor(),                        # Convert to PyTorch tensor
                transforms.Normalize(mean=[MEAN], std=[STD]),  # Normalize the pixel values
                transforms.Lambda(lambda x: add_gaussian_noise(x) if x.ndim == 3 else x)  # Add Gaussian noise (only for 3D tensors)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Pad(IMAGE_INPUT_SIZE),   # Pad the image with reflection padding
                transforms.Resize((IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE),interpolation=cv2.INTER_AREA),                # Resize the image
                transforms.ToTensor(),                        # Convert to PyTorch tensor
                transforms.Normalize(mean=[MEAN], std=[STD]),  # Normalize the pixel values
            ])
    def __call__(self,image,bboxes,class_labels):
        image = self.transform(Image.fromarray(image))
        #print size of image
        transformed_bbox=[apply_transform_to_bbox(bboxes[i], self.transform.transforms[-3], image.size())for i in range(len(bboxes))] 
        return {'image':image,'bboxes':transformed_bbox,'class_labels':class_labels}
