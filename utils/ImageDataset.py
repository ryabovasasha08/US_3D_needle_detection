import torch 
import os
import torch
from utils.type_reader import get_image_array
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize

from utils.type_reader import get_image_array
from skimage.transform import resize

# since resolution is 3px=mm and we have +-1mm error, default mask is 3px radius
# 135 is initial image size so also the default ResizeTo
def getDatapointResized(filename, frame_num, labels, mask_diam=3, resizeTo = 135): 
    # create frame image based on frame number and filename
    input_image = get_image_array(filename)[:, :, :, frame_num]

    # create mask for the frame image
    mask_image = np.zeros((input_image.shape))
    mask_image[
        np.around(labels[0]-mask_diam/2).astype(int):np.around(labels[0]+mask_diam/2).astype(int), 
        np.around(labels[1]-mask_diam/2).astype(int):np.around(labels[1]+mask_diam/2).astype(int), 
        np.around(labels[2]-mask_diam/2).astype(int):np.around(labels[2]+mask_diam/2).astype(int)
    ] = 1
    
    us_tip_coords = np.around(labels).astype(int)
    
    #resize everything to 128*128*128 and normalize
    ratio = 128/135
    us_tip_coords_resized = np.around(us_tip_coords*ratio).astype(int)
    
    input_image = resize(input_image, (128, 128, 128))[np.newaxis, :, :, :]
    mean = np.mean(input_image)
    std = np.std(input_image)
    input_image = (input_image - mean) / std
    
    mask_image = resize(mask_image, (128, 128, 128))[np.newaxis, :, :, :]

    return input_image, mask_image, us_tip_coords_resized

class ImageDataset(Dataset):
    """US Images with a tip needle dataset."""

    def __init__(self, X, y, mask_diam=3, resizeTo=128, transform=None):
        """
        Arguments:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.X = X
        self.y = y
        self.resizeTo = resizeTo
        self.mask_diam = mask_diam
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_img, mask_img, us_tip_coords_resized = \
            getDatapointResized(self.X[idx, 0], int(float(self.X[idx, 1])), self.y[idx], mask_diam=self.mask_diam, resizeTo=self.resizeTo)

        sample = {'image': input_img, 'mask': mask_img, 'label': us_tip_coords_resized}

        if self.transform:
            sample = self.transform(sample)

        return sample