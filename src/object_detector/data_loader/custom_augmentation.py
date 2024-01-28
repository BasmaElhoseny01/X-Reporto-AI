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
ANGLE=150

# implement transforms as augmentation with gaussian noise, random rotation

class CustomAugmentation(object):
    def __init__(self, transform_type:str ='train'):
        if(transform_type == 'train'or transform_type == 'val'):
            self.transform=TransformLibrary(transform_type)
        else:
            self.transform=CustomTransform(transform_type)

    def __call__(self,image,bboxes,class_labels):
        print("CustomAugmentation called")
        return self.transform(image=image, bboxes=bboxes, class_labels=class_labels)

# implement transforms as augmentation with gaussian noise, random rotation

class TransformLibrary(object):
    
    def __init__(self, transform_type:str ='train'):
        if (transform_type == 'train'):
            print("TransformLibrary called in init train")
            self.transform =A.Compose(
                [
                    # we want the long edge of the image to be resized to IMAGE_INPUT_SIZE, and the short edge of the image to be padded to IMAGE_INPUT_SIZE on both sides,
                    # such that the aspect ratio of the images are kept, while getting images of uniform size (IMAGE_INPUT_SIZE x IMAGE_INPUT_SIZE)
                    # LongestMaxSize: resizes the longer edge to IMAGE_INPUT_SIZE while maintaining the aspect ratio
                    # INTER_AREA works best for shrinking images
                    A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
                    A.GaussNoise(),
                    #  rotate between -2 and 2 degrees
                    # A.Affine(rotate=(-ANGLE, ANGLE)),
                    A.Affine(mode=cv2.BORDER_CONSTANT, cval=0, rotate=(-ANGLE, ANGLE),keep_ratio=True),

                    # PadIfNeeded: pads both sides of the shorter edge with 0's (black pixels)
                    A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
                    A.Normalize(mean=MEAN, std=STD),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['class_labels'])
                )
        elif (transform_type == 'val'):
            self.transform = A.Compose(
                        [
                            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
                            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
                            A.Normalize(mean=MEAN, std=MEAN),
                            ToTensorV2(),
                        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['class_labels'])
                    )
    def __call__(self,image,bboxes,class_labels):
        print("TransformLibrary called in call")
        return self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
# Custom function to add Gaussian noise to an image
def add_gaussian_noise(img, mean=0, std=25):
    return img + torch.randn_like(img) * std + mean

# Custom function to apply transformations to the bounding box
def apply_transform_to_bbox(bbox, transform, image_size):
    # Convert bounding box to relative coordinates (0 to 1)
    bbox_normalized = [bbox[i] / image_size[i % 2] for i in range(4)]

    # Convert to PyTorch tensor
    bbox_tensor = torch.tensor(bbox_normalized)

    # Apply the same transformation to the bounding box
    bbox_transformed = torch.tensor(transform(bbox_tensor).numpy())

    # Convert back to absolute coordinates
    bbox_transformed *= image_size

    return bbox_transformed.int().tolist()

class CustomTransform(object):
    
    def __init__(self, transform_type:str ='train'):
       # Define a composite transform
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE),interpolation=cv2.INTER_AREA),                # Resize the image
            transforms.Pad(IMAGE_INPUT_SIZE),   # Pad the image with reflection padding
            transforms.RandomRotation(ANGLE),                 # Rotate the image randomly by up to 10 degrees
            transforms.ToTensor(),                        # Convert to PyTorch tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the pixel values
            transforms.Lambda(lambda x: add_gaussian_noise(x))  # Add Gaussian noise
        ])
    def __call__(self,image,bboxes,class_labels):
        print("CustomTransform called in call")
        image = self.transform(image)
        transformed_bbox = apply_transform_to_bbox(bboxes, self.transform, image.size)

        return {'image':image,'bboxes':transformed_bbox,'class_labels':class_labels}

# class CustomTransform(object):
    
#     def __init__(self, transform_type:str ='train'):
#         print("CustomTransform called in init")
#         self .resize_padding=ResizeAndPad()
#         self.rotation_transforms = v2.Compose([
#             v2.RandomRotation(degrees=(-ANGLE, ANGLE)),
#             v2.ToTensor(),
#             ]
#         )
#         self.normalize_transforms = v2.Compose([
#             # randomly rotate the image
#             GaussianNoise(0,1),
#             Normalize(MEAN,STD), # normalize with imagenet mean and std
#             v2.ToTensor(),
#             ]
#         )
#     def __call__(self,image,bboxes,class_labels):
#         print("CustomTransform called in call")
#         # img = self.normalize_transforms(image) 
#         x=self.rotation_transforms(Image.fromarray(image),bboxes,class_labels)
#         print(x)
#         return {'image':x[0],'bboxes':x[1],'class_labels':x[2]}
# # Define a custom transform to add Gaussian noise
# class GaussianNoise(object):
#     def __init__(self, mean=0, std=0.0001):
#         self.mean = mean
#         self.std = std

#     def __call__(self, img):
#         # check if image type is uint8
#         if img.dtype == np.uint8:
#             img = np.array(img)
#             noise = np.random.normal(self.mean, self.std, img.shape)
#             img = img + noise
#             img = np.clip(img, 0,255)
#             img = img.astype(np.uint8)
#             return img
#         else:
#             noise = np.random.normal(self.mean, self.std, img.shape)
#             img = img + noise
#             return img
        
# class Normalize(object):
#     def __init__(self, mean=0, std=0.0001):
#         self.mean = mean
#         self.std = std

#     def __call__(self, img):
#         # check if image type is uint8
#         if img.dtype == np.uint8:
#             img = np.array(img)
#             # img = img/255
#             # # subtract mean
#             # img = img - self.mean
#             # # divide by std
#             # img = img/self.std
#             return img
#         else:
#             img = np.array(img)
#             # subtract mean
#             img = img - self.mean
#             # divide by std
#             img = img/self.std
#             return img

# # Define a custom transform to resize and pad the image and update bounding boxes
# class ResizeAndPad(object):
#     def __init__(self, target_size=(512, 512)):
#         self.target_size = target_size

#     def __call__(self, image, boxes=None):
#         # Resize the  image while maintaining aspect ratio
#         width, height = image.shape[0],image.shape[1]

#         # calculate aspect ratio
#         aspect_ratio = width / height

#         # get the long and short side of the image
#         long_side = max(self.target_size)
#         short_side = min(self.target_size)

#         new_width = 0
#         new_height = 0

#         # resize the image according to the long side
#         # if width > height then resize according to width
#         if width > height:
#             # width = 1024 , height = 512, long_side = 512, aspect ratio = 2 => new_width = 512, new_height = 256
#             new_width = long_side
#             new_height = int(new_width / aspect_ratio)

#         else:
#             # width = 512 , height = 1024, long_side = 512, aspect ratio = 0.5 => new_width = 256, new_height = 512
#             new_height = long_side
#             new_width = int(new_height * aspect_ratio)
        
#         # new width is row, new height is column
#         # resize the image
#         resized_image = cv2.resize(image, (new_height, new_width))

#         # Create a new black grayscale image with the desired size
#         new_image = np.zeros(self.target_size, dtype=np.float32)

#         # add the resized image to the new image and center it
#         # if width > height then the resized image will be added to the new image with the same width but different height
#         if new_width > new_height:
#             # calculate the starting and ending column
#             start = int((new_width - new_height) / 2)
#             end = start + new_height
#             new_image[:, start:end] = resized_image
#             # change the bounding boxes accordingly and add shift to the x coordinate
#             if boxes is not None:
#                 boxes = np.array(boxes)
#                 boxes[:, [0, 2]] = boxes[:, [0, 2]] * (new_height / height) + start
#                 boxes[:, [1, 3]] = boxes[:, [1, 3]] * (new_width / width)
#         else:
#             # if width < height then the resized image will be added to the new image with the same height but different width
#             # calculate the starting and ending row
#             start = int((new_height - new_width) / 2)
#             end = start + new_width
#             new_image[start:end, :] = resized_image
#             # change the bounding boxes accordingly and add shift to the y coordinate
#             if boxes is not None:
#                 boxes = np.array(boxes)
#                 boxes[:, [0, 2]] = boxes[:, [0, 2]] * (new_height / height)
#                 boxes[:, [1, 3]] = boxes[:, [1, 3]] * (new_width / width) + start

#         # convert the image to uint8
#         new_image = new_image.astype(np.uint8)
#         return new_image, boxes