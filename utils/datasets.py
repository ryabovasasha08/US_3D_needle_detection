import torch 
import os
from utils.type_reader import get_image_array
import pandas as pd
from skimage import io, transform
import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
import h5py
from torch.utils.data._utils.collate import default_collate
from utils.type_reader import get_image_array
from utils.dataset_utils import *

class FrameDiffDataset(Dataset):
    """US Images with a tip needle dataset."""

    # passed X must contain two pairs of frames of the same frame sequence with difference of frame_diff
    # passed y must contain needle tip coords for the second of the needle frames
    def __init__(self, X, y):
        """
        Arguments:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        f = self.X[idx][0][:-4].split("/")[-1]
        h5_file = h5py.File('../train/resized_train/'+f+'.hdf5', 'r')
        
        
        img_1 = h5_file["img"+"_"+self.X[idx][1]+"_"+str(10)][()]
        img_2 = h5_file["img"+"_"+self.X[idx][2]+"_"+str(10)][()]
        mask_1 = h5_file["mask"+"_"+self.X[idx][1]+"_"+str(10)][()]
        mask_2 = h5_file["mask"+"_"+self.X[idx][2]+"_"+str(10)][()]
        tip_coords = h5_file["labels"+"_"+self.X[idx][2]+"_"+str(10)][()]
        tip_coords_original = h5_file["labels_original"][int(float(self.X[idx][2])), :]
        
        h5_file.close
    
        img = img_2-img_1
        mask = mask_2-mask_1
        
        img, mask, coords = transformWithLabel(img, mask, tip_coords)
        
        sample = {'image': img, 'mask': mask, 'label': coords, 'label_original': tip_coords_original}
            
        # Check if data is correct, i.e. label is within range and mask is of correct size
        if (sample['label'] > 1).all(): 
            return sample
        else:
            # return None to skip this sample
            return None


class CustomMaskDataset(Dataset):
    """US Images with a tip needle dataset."""
    
    def __init__(self, X, maskType='full', maskRadius=6):
        """
        Arguments:
            X: a 2D array, each row consists of [filename, frameNum]
            maskType: 'full' (default)
                      'tip' (cube 3^3 around needle tip)
        """
        self.X = X
        self.maskType = maskType
        self.maskRadius = maskRadius

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # create frame image based on frame number and filename
        f = self.X[idx, 0]
        transformNum = random.randint(0, 10)
        frameNumStr = self.X[idx, 1]
        h5_file = h5py.File('../train/resized_train/'+f+'.hdf5', 'r')
        img = h5_file["img"+"_"+frameNumStr+"_"+str(transformNum)][()]
        tip_coords = h5_file["labels"+"_"+frameNumStr+"_"+str(transformNum)][()]
        tip_coords_original = h5_file["labels_original"][int(float(frameNumStr)), :]
        
        if self.maskType=='full':
            mask = h5_file["mask"+"_"+frameNumStr+"_"+str(transformNum)][()]
        else:
            mask = np.zeros((img.shape))
            mask[np.around(tip_coords[0]-self.maskRadius).astype(int):np.around(tip_coords[0]+self.maskRadius).astype(int), 
                 np.around(tip_coords[1]-self.maskRadius).astype(int):np.around(tip_coords[1]+self.maskRadius).astype(int), 
                 np.around(tip_coords[2]-self.maskRadius).astype(int):np.around(tip_coords[2]+self.maskRadius).astype(int)
                ] = 1
            
        h5_file.close()

        img, mask, coords = transformWithLabel(img, mask, tip_coords)
        
        sample = {'image': img, 'mask': mask, 'label': coords, 'label_original': tip_coords_original}
            
        # Check if data is correct, i.e. label is within range and mask is of correct size
        if (sample['label'] > 1).all(): 
            return sample
        else:
            # return None to skip this sample
            return None



# custom collate function to filter out None samples
def my_collate(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch)





#################################
###  Older versions of datasets:
#################################

class ImageDataset(Dataset):
    """US Images with a tip needle dataset."""

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
            getImageDatasetDatapointResized(self.X[idx, 0], int(float(self.X[idx, 1])), self.y[idx], mask_diam=self.mask_diam, resizeTo=self.resizeTo)

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


class FullMaskDataset(Dataset):
    """US Images with a tip needle dataset."""

    def __init__(self, X, y, cropTo=128):
        """
        Arguments:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.X = X
        self.y = y
        self.cropTo = cropTo

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # create frame image based on frame number and filename
        f = self.X[idx, 0][:-4].split("/")[-1]
        h5_file = h5py.File('../train/trainh5_chunked/'+f+'.hdf5', 'r')
        img = h5_file['default'][int(float(self.X[idx, 1]))]
        h5_file.close()
        h5_mask_file = h5py.File('../train/needle_masks_h5/'+f+'.hdf5', 'r')
        mask = h5_mask_file['default'][int(float(self.X[idx, 1]))]
        h5_mask_file.close()

        img, mask, coords = transformAndCropWithLabel(img, mask, self.y[idx], self.cropTo)
    
        
        sample = {'image': img, 'mask': mask, 'label': coords}
            
        # Check if data is correct, i.e. label is within range and mask is of correct size
        if (sample['label'] > 1).all() and (sample['label'] < self.cropTo).all(): 
            return sample
        else:
            # return None to skip this sample
            return None