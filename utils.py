import numpy as np
import os
import cv2
import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional as F
import random
import pandas as pd
from torchvision.transforms import v2


def plot_example_with_boxes(img,boxes,name = "test.jpg"):
    """
    img: numpy array of shape (H,W)
    boxes: list of lists of shape (4,)
    """
    img = img.copy()
    for box in boxes:
        x1,y1,x2,y2 = box
        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
    plt.imshow(img)
    plt.show()

    # save the image
    cv2.imwrite(name, img)


