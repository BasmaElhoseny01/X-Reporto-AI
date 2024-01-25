import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import torch
from PIL import Image
from torchvision.transforms import functional as F
import random
import pandas as pd 
from torchvision.transforms import v2
from torchvision import tv_tensors  # we'll describe this a bit later, bare with us
import matplotlib.patches as patches


import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
# from .utils import plot_example_with_boxes

class F_RCNNDataset(Dataset):
    def __init__(self, dataset_path: str, transform =None):
        self.dataset_path = dataset_path # path to csv file
        self.transform = transform

        # read the csv file
        self.data_info = pd.read_csv(dataset_path, header=None)
        # remove the first row (column names)
        self.data_info = self.data_info.iloc[1:]


        # row contains (subject_id,	study_id, image_id, mimic_image_file_path, bbox_coordinates list of list, bbox_labels list,
        #               bbox_phrases list of str, bbox_phrase_exists list of booleans, bbox_is_abnormal list of booleans)
        self.train_transforms = A.Compose(
        [
            # we want the long edge of the image to be resized to IMAGE_INPUT_SIZE, and the short edge of the image to be padded to IMAGE_INPUT_SIZE on both sides,
            # such that the aspect ratio of the images are kept, while getting images of uniform size (IMAGE_INPUT_SIZE x IMAGE_INPUT_SIZE)
            # LongestMaxSize: resizes the longer edge to IMAGE_INPUT_SIZE while maintaining the aspect ratio
            # INTER_AREA works best for shrinking images
            A.LongestMaxSize(max_size=512, interpolation=cv2.INTER_AREA),
            # A.ColorJitter(hue=0.0),
            # A.GaussNoise(),
            # randomly (by default prob=0.5) translate and rotate image
            # mode and cval specify that black pixels are used to fill in newly created pixels
            # translate between -2% and 2% of the image height/width, rotate between -2 and 2 degrees
            # A.Affine(mode=cv2.BORDER_CONSTANT, cval=0, translate_percent=(-0.02, 0.02), rotate=(-2, 2)),
            # PadIfNeeded: pads both sides of the shorter edge with 0's (black pixels)
            A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=0.499, std=0.291),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['class_labels'])
    )

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # get the image path
        img_path = self.data_info.iloc[idx, 3]

        # read the image with parent path of current folder + image path
        img_path = os.path.join(os.getcwd(), img_path)
        img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
        assert img is not None, f"Image at {img_path} is None"
        
        # get the bounding boxes
        bboxes = self.data_info.iloc[idx, 4]

        # convert the string representation of bounding boxes into list of list
        bboxes = eval(bboxes)

        # get the labels
        labels = self.data_info.iloc[idx, 5]

        # convert the string representation of labels into list
        labels = np.array(eval(labels))

        # resize the image with the bounding boxes accordingly to 512x512
        # img,bboxes = ResizeAndPad((512,512))(img,bboxes)



        # if self.transform:
        #     bboxes = tv_tensors.BoundingBoxes(
        #         bboxes,
        #         format="XYXY", canvas_size=(512, 512))
        #     img, bboxes = self.transform(img, bboxes)
        # else:
        #     img = Image.fromarray(img)
        #     img = v2.ToTensor()(img)
        #     bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        # labels = torch.as_tensor(labels, dtype=torch.int64)
        transformed = self.train_transforms(image=img, bboxes=bboxes, class_labels=labels)

        transformed_image = transformed["image"]
        transformed_bboxes = transformed["bboxes"]
        transformed_bbox_labels = transformed["class_labels"]

        # convert the bounding boxes to tensor
        transformed_bboxes = torch.as_tensor(transformed_bboxes, dtype=torch.float32)
        transformed_bbox_labels = torch.as_tensor(transformed_bbox_labels, dtype=torch.int64)
        target = {}
        target["boxes"] = transformed_bboxes
        target["labels"] = transformed_bbox_labels
        return transformed_image,target
        # define the bounding box
        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels

        return img, target

# implement transforms as augmentation with gaussian noise, random rotation

class Augmentation(object):
    def __init__(self, noise_std=0.1, rotate_angle=2):
        self.noise_std = noise_std
        self.rotate_angle = rotate_angle

        self.rotation_transforms = v2.Compose([
            # randomly rotate the image
            # v2.RandomRotation(self.rotate_angle),
            
            v2.RandomRotation(degrees=(-self.rotate_angle, self.rotate_angle)),
            v2.ToTensor(),
            ]
        )
        self.normalize_transforms = v2.Compose([
            # randomly rotate the image
            AddGaussianNoise(0,self.noise_std),
            v2.Normalize([0.499,0.499], [0.291,0.291]), # normalize with imagenet mean and std
            ]
        )

    def __call__(self, img,bboxes):
        # apply the transforms to the image and the bounding boxes
        img = self.normalize_transforms(img)
        img = Image.fromarray(img)
        return self.rotation_transforms(img,bboxes)


# Define a custom transform to add Gaussian noise
class AddGaussianNoise(object):
    def __init__(self, mean=0, std=0.0001):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        # Add Gaussian noise to the PIL image
        img = np.array(img)
        img = img.astype(np.float32)/255
        noise = np.random.normal(self.mean, self.std, img.shape)
        # print("noise",noise)
        
        # print("img",img[:,100:200])
        img = img + noise
        
        img = np.clip(img, 0,1)
        # print("img",img[:,100:200])
        return img


# Define a custom transform to resize and pad the image and update bounding boxes
class ResizeAndPad(object):
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size

    def __call__(self, image, boxes=None):
        # Resize the  image while maintaining aspect ratio
        width, height = image.shape[0],image.shape[1]

        # calculate aspect ratio
        aspect_ratio = width / height

        # get the long and short side of the image
        long_side = max(self.target_size)
        short_side = min(self.target_size)

        new_width = 0
        new_height = 0

        # resize the image according to the long side
        # if width > height then resize according to width
        if width > height:
            # width = 1024 , height = 512, long_side = 512, aspect ratio = 2 => new_width = 512, new_height = 256
            new_width = long_side
            new_height = int(new_width / aspect_ratio)

        else:
            # width = 512 , height = 1024, long_side = 512, aspect ratio = 0.5 => new_width = 256, new_height = 512
            new_height = long_side
            new_width = int(new_height * aspect_ratio)
        
        # new width is row, new height is column
        # resize the image
        resized_image = cv2.resize(image, (new_height, new_width))

        # Create a new black grayscale image with the desired size
        new_image = np.zeros(self.target_size, dtype=np.float32)

        # add the resized image to the new image and center it
        # if width > height then the resized image will be added to the new image with the same width but different height
        if new_width > new_height:
            # calculate the starting and ending column
            start = int((new_width - new_height) / 2)
            end = start + new_height
            new_image[:, start:end] = resized_image
            # change the bounding boxes accordingly and add shift to the x coordinate
            if boxes is not None:
                boxes = np.array(boxes)
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * (new_height / height) + start
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * (new_width / width)
        else:
            # if width < height then the resized image will be added to the new image with the same height but different width
            # calculate the starting and ending row
            start = int((new_height - new_width) / 2)
            end = start + new_width
            new_image[start:end, :] = resized_image
            # change the bounding boxes accordingly and add shift to the y coordinate
            if boxes is not None:
                boxes = np.array(boxes)
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * (new_height / height)
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * (new_width / width) + start

        return new_image, boxes


def plot_example_with_boxes(img,boxes,name = "test.jpg"):
    """
    img: numpy array of shape (H,W)
    boxes: list of lists of shape (4,)
    """
    img = img.copy()
    # add the bounding boxes to the image using matplotlib
    for box in boxes:
        plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, color='red')
    # show the image in grayscale using matplotlib
    plt.imshow(img, cmap='gray')
    plt.savefig(name)
    plt.show()


def plot_image(img, boxes):
    '''
    Function that draws the BBoxes on the image.

    inputs:
        img: input-image as numpy.array (shape: [H, W, C])
        boxes: list of bounding boxes (Format [N, 4] => N times [xmin, ymin, xmax, ymax])
    '''
    cmap = plt.get_cmap("tab20b")
    height, width = img.shape[:2]
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(16, 8))
    # Display the image
    ax.imshow(img, cmap="gray"  )
    for i, box in enumerate(boxes):
        width = box[2] - box[0]
        height = box[3] - box[1]
        rect = patches.Rectangle(
            (box[0], box[1]),
            width,
            height,
            linewidth=1,  # Increase linewidth
            edgecolor="red",  # Set the box border color
            facecolor="none",  # Set facecolor to none
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
    
    plt.show()

if __name__ == '__main__':
    # load the csv file
    data = pd.read_csv('datasets/train-200.csv', header=None)
    img_path = data.iloc[1, 3]

    # read the image with parent path of current folder + image path
    img_path = os.path.join(os.getcwd(), img_path)
    img = cv2.imread(img_path,0)
    assert img is not None, f"Image at {img_path} is None"
    # get the bounding boxes
    bboxes = data.iloc[1, 4]

    # convert the string representation of bounding boxes into list of list
    bboxes = eval(bboxes)

    # plot the image with the bounding boxes
    # plot_example_with_boxes(img, bboxes,name = "before.jpg")
    
    # create the dataset
    dataset = F_RCNNDataset(dataset_path= 'datasets/train-200.csv', transform = Augmentation())

    # get the image and the target

    for i in range(1):
        img, target = dataset[i]
        # img = img.numpy()[0]
        img = img.numpy().transpose(1, 2, 0)
        # convert img to uint8
        # img = img*255
        # img = img.astype(np.uint8)
        bboxes = target['boxes']
        # convert image to uint8
        # img = img*255
        # img = img.astype(np.uint8)
        # convert imafe to numpy array
        img = np.array(img)
        # plot the image with the bounding boxes
        print("bboxes",bboxes)
        print("img",img)
        # plot_example_with_boxes(img, bboxes,name = "after.jpg")
        plot_image(img, bboxes)

    # # apply the augmentation
    # transform = Augmentation()
    # img = Image.fromarray(img)
    # bboxes = tv_tensors.BoundingBoxes(
    #     bboxes,
    #     format="XYXY", canvas_size=(512, 512))

    # img, bboxes = transform(img, bboxes)

    # img = np.array(img)[0]
    # bboxes = np.array(bboxes)
    # print("bboxes",bboxes)
    # print("img",img)
    # # plot the image with the bounding boxes
    # plot_example_with_boxes(img,bboxes,name = "after_augmentation.jpg")
