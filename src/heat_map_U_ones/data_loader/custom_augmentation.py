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
from config import HEAT_MAP_IMAGE_SIZE

ANGLE=2

class CustomAugmentation(object):
    """
    Custom augmentation class using Albumentations for image transformations.

    Args:
        transform_type (str): Type of transformation ('train' or 'test').

    Attributes:
        transform (TransformLibrary): Instance of TransformLibrary based on transform_type.

    Methods:
        __call__(image): Applies transformations on the input image.
    """
    def __init__(self, transform_type):    
        """
        Initializes CustomAugmentation with specified transformation type.

        Args:
            transform_type (str): Type of transformation ('train' or 'test').
        """
        self.transform=TransformLibrary(transform_type)
        
    def __call__(self,image):
        """
        Applies transformations on the input image.

        Args:
            image (numpy.ndarray): Input image as numpy array.

        Returns:
            dict: Transformed image as a dictionary with 'image' key.
        """
        return self.transform(image=image)

    
class TransformLibrary(object):
    """
    Transformation library using Albumentations for image preprocessing.

    Args:
        transform_type (str): Type of transformation ('train' or 'test').

    Attributes:
        transform (albumentations.Compose): Albumentations composition based on transform_type.

    Methods:
        __call__(image): Applies transformations on the input image.
    """
    
    def __init__(self, transform_type:str ='train'):
        """
        Initializes TransformLibrary with specified transformation type.

        Args:
            transform_type (str, optional): Type of transformation ('train' or 'test'). Default is 'train'.
        """
        if (transform_type == 'train'):
            self.transform =A.Compose([
            A.LongestMaxSize(max_size=HEAT_MAP_IMAGE_SIZE, interpolation=cv2.INTER_AREA),
            # A.Resize(height=HEAT_MAP_IMAGE_SIZE, width=HEAT_MAP_IMAGE_SIZE, p=1.0),

            # A.RandomCrop(height=256, width=256, p=1.0),
            # A.RandomHorizontalFlip(),
            A.GaussNoise(),
            #  rotate between -2 and 2 degrees
            A.Affine(mode=cv2.BORDER_CONSTANT, cval=0, rotate=(-ANGLE, ANGLE)),
            
            # PadIfNeeded: pads both sides of the shorter edge with 0's (black pixels)
            A.PadIfNeeded(min_height=HEAT_MAP_IMAGE_SIZE, min_width=HEAT_MAP_IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT),

            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(p=1.0),], p=1.0)
                    
        else:
            self.transform = A.Compose([
            # A.Resize(height=HEAT_MAP_IMAGE_SIZE, width=HEAT_MAP_IMAGE_SIZE, p=1.0),
            A.LongestMaxSize(max_size=HEAT_MAP_IMAGE_SIZE, interpolation=cv2.INTER_AREA),
      
            # PadIfNeeded: pads both sides of the shorter edge with 0's (black pixels)
            A.PadIfNeeded(min_height=HEAT_MAP_IMAGE_SIZE, min_width=HEAT_MAP_IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT),

            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(p=1.0),])
            
    def __call__(self,image):
        """
        Applies transformations on the input image.

        Args:
            image (numpy.ndarray): Input image as numpy array.

        Returns:
            dict: Transformed image as a dictionary with 'image' key.
        """
        return self.transform(image=image)