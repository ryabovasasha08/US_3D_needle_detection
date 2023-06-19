import torch 
import os
import torch
from utils.type_reader import get_image_array
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class ImageDataset(Dataset):
    """US Images with a tip needle dataset."""

    def __init__(self, X, y, transform=None):
        """
        Arguments:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # create frame image based on frame number and filename
        input_image = get_image_array(self.X[idx, 0])[:, :, :, int(float(self.X[idx, 1]))]

        # create mask for the frame image
        us_tip_coords = np.around(self.y[idx]).astype(int)
        mask_radius = 3 # since resolution is 3px=mm and we have +-1mm error, mask should be 3px radius
        mask_image = np.zeros((input_image.shape))
        mask_image[
            us_tip_coords[0]-mask_radius:us_tip_coords[0]+mask_radius, 
            us_tip_coords[1]-mask_radius:us_tip_coords[1]+mask_radius, 
            us_tip_coords[2]-mask_radius:us_tip_coords[2]+mask_radius
        ] = np.full((mask_radius*2, mask_radius*2,mask_radius*2), 1)
        
        sample = {'image': input_image, 'mask': mask_image, 'label': us_tip_coords}

        if self.transform:
            sample = self.transform(sample)

        return sample