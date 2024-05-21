import imageio
import numpy as np
import torch
import numpy as np
from PIL import Image
import os
from skimage.metrics import structural_similarity as compare_ssim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import *
import cv2
# transform =  A.Compose(
#                         [
#                             A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
#                         ])
def save2image(d_img, filename):
    img = d_img
    _min, _max = img.min(), img.max()
    if np.abs(_max - _min) < 1e-4:
        f_img = np.zeros(img.shape)
    else:
        f_img = (img - _min)*255 / (_max - _min)
    # check if f_img is a tensor
    if isinstance(f_img, torch.Tensor):
        f_img = f_img.detach().cpu().numpy()
    else:
        f_img = f_img
    # add padding
    # f_img = transform(image=f_img)["image"]
    # normalize the image
    f_img = (f_img - np.min(f_img)) / (np.max(f_img) - np.min(f_img))
    f_img = f_img * 255
    f_img = f_img.astype(np.uint8)
    imageio.imwrite(filename, f_img)

def eval_metrics(actual, pred):
    # move actual to cpu
    actual = actual.detach().cpu().numpy()
    actual = (actual-np.min(actual) )/ (np.max(actual) - np.min(actual))
    pred = (pred-np.min(pred) )/ (np.max(pred) - np.min(pred))

    ssim = compare_ssim(actual, pred, data_range=1, full=True)[0]
    mse = np.mean((actual - pred) ** 2) 
    if(mse == 0): 
        psnr = 100
    else:
        max_pixel = 1
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return ssim, psnr