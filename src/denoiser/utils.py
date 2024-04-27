import imageio
import numpy as np
import torch
import numpy as np
from PIL import Image
import os
from skimage.metrics import structural_similarity as compare_ssim

def save2image(d_img, filename):
    img = d_img
    _min, _max = img.min(), img.max()
    if np.abs(_max - _min) < 1e-4:
        f_img = np.zeros(img.shape)
    else:
        f_img = (img - _min)*255 / (_max - _min)
    
    f_img = f_img.astype('uint8')
    imageio.imwrite(filename, f_img)

def eval_metrics(actual, pred):
    ssim = compare_ssim(actual, pred)
    mse = np.mean((actual - pred) ** 2) 
    if(mse == 0): 
        psnr = 100
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return ssim, psnr