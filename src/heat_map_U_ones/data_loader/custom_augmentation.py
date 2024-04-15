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
    def __init__(self, transform_type):
            self.transform=TransformLibrary(transform_type)
        
    def __call__(self,image):
        return self.transform(image=image)

    
class TransformLibrary(object):
    
    def __init__(self, transform_type:str ='train'):
        if (transform_type == 'train'):
            self.transform =A.Compose([
            # A.RandomCrop(height=256, width=256, p=1.0),
            # A.RandomHorizontalFlip(),
            A.GaussNoise(),
            #  rotate between -2 and 2 degrees
            A.Affine(mode=cv2.BORDER_CONSTANT, cval=0, rotate=(-ANGLE, ANGLE)),

            A.Resize(height=HEAT_MAP_IMAGE_SIZE, width=HEAT_MAP_IMAGE_SIZE, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(p=1.0),], p=1.0)
                    
        else:
            self.transform = A.Compose([
            A.Resize(height=HEAT_MAP_IMAGE_SIZE, width=HEAT_MAP_IMAGE_SIZE, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(p=1.0),])
            
    def __call__(self,image):
        return self.transform(image=image)