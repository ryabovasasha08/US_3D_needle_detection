import torch 
import os
from utils.type_reader import get_image_array
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
import h5py
from torch.utils.data._utils.collate import default_collate
from utils.type_reader import get_image_array

class FrameDiffDataset(Dataset):
    """US Images with a tip needle dataset."""

    # passed X must contain two pairs of frames of the same frame sequence with difference of frame_diff
    # passed y must contain needle tip coords for the second of the needle frames
    def __init__(self, X, y, resizeTo=128, transform=None):
        """
        Arguments:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.X = X
        self.y = y
        self.resizeTo = resizeTo
        self.transform = transform

    def __len__(self):
        return len(self.y)
    
    def getDatapointResized(self, input_image, mask, labels):     
        us_tip_coords = labels.astype(float)
        
        #resize everything to 128*128*128 and normalize
        ratio = self.resizeTo/input_image.shape[0]
        us_tip_coords_resized = np.around(us_tip_coords*ratio).astype(int)
        us_tip_coord_flattened = (self.resizeTo * self.resizeTo * us_tip_coords_resized[2]) + (self.resizeTo * us_tip_coords_resized[1]) + us_tip_coords_resized[0]
        
        input_image = ndimage.zoom(input_image, (ratio, ratio, ratio))[np.newaxis, :, :, :]
        mean = np.mean(input_image)
        std = np.std(input_image)
        input_image = (input_image - mean) / std
        
        mask = ndimage.zoom(mask, (ratio, ratio, ratio))[np.newaxis, :, :, :]
        mean = np.mean(mask)
        std = np.std(mask)
        mask = (mask - mean) / std

        return input_image, mask, us_tip_coords_resized, us_tip_coord_flattened

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        f = self.X[idx][0][:-4].split("/")[-1]
        print(f)
        h5_img_file = h5py.File('../train/trainh5/'+f+'.hdf5', 'r')
        h5_mask_file = h5py.File('../train/needle_masks_h5/'+f+'.hdf5', 'r')
        input_img_1 = h5_img_file['default'][int(float(self.X[idx, 1]))]
        mask_1 = h5_mask_file['default'][int(float(self.X[idx, 1]))]
        input_img_2 = h5_img_file['default'][int(float(self.X[idx, 2]))]
        mask_2 = h5_mask_file['default'][int(float(self.X[idx, 2]))]
        h5_img_file.close()
        h5_mask_file.close()
    
        input_img = input_img_2-input_img_1
        mask = mask_2-mask_1
        
        input_img_resized, mask_resized, label_resized, label_1D_resized = self.getDatapointResized(input_img, mask, self.y[idx])
        
        sample = {'image': input_img_resized, 'mask': mask_resized, 'label': label_resized, 'label_1D': label_1D_resized}

        # TODO: use transforms: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        if self.transform:
            sample = self.transform(sample)
            
        # Check if data is correct, i.e. label is within range and mask is of correct size
        if (sample['label'] > 1).all() and (sample['label'] < self.resizeTo).all(): 
            return sample
        else:
            # return None to skip this sample
            return None





class ImageDataset(Dataset):
    """US Images with a tip needle dataset."""
    
    # since resolution is 3px=mm and we have +-1mm error, default mask is 3px radius
    # 135 is initial image size so also the default ResizeTo
    def getDatapointResized(filename, frame_num, labels, mask_diam=6, resizeTo = 135): 
        # create frame image based on frame number and filename
        f = filename[:-4].split("/")[-1]
        h5_file = h5py.File('../train/trainh5/'+f+'.hdf5', 'r')
        input_image = h5_file['default'][frame_num]
        h5_file.close()

        '''
        # create mask for the frame image
        mask_image = np.zeros((input_image.shape))
        mask_image[
            np.around(labels[0]-mask_diam/2).astype(int):np.around(labels[0]+mask_diam/2).astype(int), 
            np.around(labels[1]-mask_diam/2).astype(int):np.around(labels[1]+mask_diam/2).astype(int), 
            np.around(labels[2]-mask_diam/2).astype(int):np.around(labels[2]+mask_diam/2).astype(int)
        ] = 1
        '''
        
        us_tip_coords = np.around(labels).astype(int)
        
        #resize everything to 128*128*128 and normalize
        ratio = resizeTo/input_image.shape[0]
        us_tip_coords_resized = np.around(us_tip_coords*ratio).astype(int)
        us_tip_coord_flattened = (resizeTo * resizeTo * us_tip_coords_resized[2]) + (resizeTo * us_tip_coords_resized[1]) + us_tip_coords_resized[0]
        
        input_image = ndimage.zoom(input_image, (ratio, ratio, ratio))[np.newaxis, :, :, :]
        mean = np.mean(input_image)
        std = np.std(input_image)
        input_image = (input_image - mean) / std
        
        mask_image = np.zeros((input_image.shape))
        
        mask_image[
            0,
            np.around(us_tip_coords_resized[0]-mask_diam/2).astype(int):np.around(us_tip_coords_resized[0]+mask_diam/2).astype(int), 
            np.around(us_tip_coords_resized[1]-mask_diam/2).astype(int):np.around(us_tip_coords_resized[1]+mask_diam/2).astype(int), 
            np.around(us_tip_coords_resized[2]-mask_diam/2).astype(int):np.around(us_tip_coords_resized[2]+mask_diam/2).astype(int)
        ] = 1

        return input_image, mask_image, us_tip_coords_resized, us_tip_coord_flattened


    def __init__(self, X, y, mask_diam=6, resizeTo=128, transform=None):
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

        input_img, mask_img, us_tip_coords_resized, us_tip_coords_flattened_resized = \
            self.getDatapointResized(self.X[idx, 0], int(float(self.X[idx, 1])), self.y[idx], mask_diam=self.mask_diam, resizeTo=self.resizeTo)

        sample = {'image': input_img, 'mask': mask_img, 'label': us_tip_coords_resized, 'label_1D':us_tip_coords_flattened_resized}

        # TODO: use transforms: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        if self.transform:
            sample = self.transform(sample)
            
        # Check if data is correct, i.e. label is within range and mask is of correct size
        if (sample['label'] > 1).all() and (sample['label'] < self.resizeTo).all() and (np.count_nonzero(sample['mask']) == self.mask_diam**3): 
            return sample
        else:
            # return None to skip this sample
            return None
        
# custom collate function to filter out None samples
def my_collate(batch):

  batch = [b for b in batch if b is not None]

  return default_collate(batch)