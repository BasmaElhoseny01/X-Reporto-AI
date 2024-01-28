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
from custom_augmentation import CustomAugmentation

class CustomDataset(Dataset):
    def __init__(self, dataset_path: str, transform_type:str ='train'):
        self.dataset_path = dataset_path # path to csv file
        
        self.transform_type = transform_type
        self.transform = CustomAugmentation(transform_type=self.transform_type)
        # read the csv file
        self.data_info = pd.read_csv(dataset_path, header=None)
        # remove the first row (column names)
        self.data_info = self.data_info.iloc[1:]

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # get the image path
        img_path = self.data_info.iloc[idx, 3]

        # read the image with parent path of current folder + image path
        # img_path = os.path.join("datasets/", img_path)
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

        transformed = self.transform(image=img, bboxes=bboxes, class_labels=labels)

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

# implement transforms as augmentation with gaussian noise, random rotation
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
    # for i, box in enumerate(boxes):
    width = boxes[8][2] - boxes[8][0]
    height = boxes[8][3] - boxes[8][1]
    rect = patches.Rectangle(
        (boxes[8][0], boxes[8][1]),
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
    data = pd.read_csv('E:/Graduation Project/Graduation-Project-Repo/datasets/train-200.csv', header=None)
    img_path = data.iloc[1, 3]

    # read the image with parent path of current folder + image path
    img_path = os.path.join(os.getcwd(), img_path)
    img = cv2.imread(img_path,0)
    assert img is not None, f"Image at {img_path} is None"
    # get the bounding boxes
    bboxes = data.iloc[1, 4]

    # convert the string representation of bounding boxes into list of list
    bboxes = eval(bboxes)
    plot_image(img, bboxes)

    # plot the image with the bounding boxes
    # plot_example_with_boxes(img, bboxes,name = "before.jpg")
    
    # create the dataset
    dataset = CustomDataset(dataset_path= 'E:/Graduation Project/Graduation-Project-Repo/datasets/train-200.csv', transform_type = 'train')

    # get the image and the target

    for i in range(1):
        img, target = dataset[i]
        # img = img.numpy()[0]
        img = img.numpy().transpose(1, 2, 0)
        # convert img to uint8
        # img = img*85
        # img = img.astype(np.uint8)
        bboxes = target['boxes']
        # convert image to uint8
        # img = img*85
        # img = img.astype(np.uint8)
        # convert imafe to numpy array
        img = np.array(img)
        # plot the image with the bounding boxes
        # print("bboxes",bboxes)
        # print("img",img)
        # plot_example_with_boxes(img, bboxes,name = "after.jpg")
        plot_image(img, bboxes)
