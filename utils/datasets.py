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

    # passed X must contain set of pairs of frames of the same frame sequence with difference of frame_diff
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        h5_file = h5py.File(self.X[idx][0], 'r')
        
        #remove '.0' at the end of each frame number
        idx1 = self.X[idx][1][:-2]
        idx2 = self.X[idx][2][:-2]
        
        img_1 = h5_file["img"+"_"+idx1+"_"+str(10)][()]
        img_2 = h5_file["img"+"_"+idx2+"_"+str(10)][()]
        mask_1 = h5_file["mask"+"_"+idx1+"_"+str(10)][()]
        mask_2 = h5_file["mask"+"_"+idx2+"_"+str(10)][()]
        tip_coords = (h5_file["labels"+"_"+idx2+"_"+str(10)][()]+h5_file["labels"+"_"+idx1+"_"+str(10)][()])/2
        tip_coords_original = (h5_file["labels_original_new"][int(idx2), :]+h5_file["labels_original_new"][int(idx1), :])/2
        
        h5_file.close
    
        img = img_2-img_1
        mask = mask_2-mask_1
        
        img, mask, coords, tip_coords_original = transformWithLabel(img, mask, tip_coords, tip_coords_original)
        
        sample = {'image': img, 'mask': mask, 'label': coords, 'label_original': tip_coords_original}
            
        # Check if data is correct, i.e. label is within range and mask is of correct size
        if (sample['label'] > 1).all(): 
            return sample
        else:
            # return None to skip this sample
            return None


class CustomMaskDataset(Dataset):
        
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
        transformNum = random.randint(0, 10)
        frameNumInt = int(float(self.X[idx, 1]))
        frameNumStr = str(frameNumInt)
        h5_file = h5py.File(self.X[idx, 0], 'r')
        img = h5_file["img"+"_"+frameNumStr+"_"+str(transformNum)][()]
        tip_coords = h5_file["labels"+"_"+frameNumStr+"_"+str(transformNum)][()]
        tip_coords_original = h5_file["labels_original_new"][frameNumInt, :]
        
        if self.maskType=='full':
            mask = h5_file["mask"+"_"+frameNumStr+"_"+str(transformNum)][()]
        else:
            mask = np.zeros((img.shape))
            mask[np.around(tip_coords[0]-self.maskRadius).astype(int):np.around(tip_coords[0]+self.maskRadius).astype(int), 
                 np.around(tip_coords[1]-self.maskRadius).astype(int):np.around(tip_coords[1]+self.maskRadius).astype(int), 
                 np.around(tip_coords[2]-self.maskRadius).astype(int):np.around(tip_coords[2]+self.maskRadius).astype(int)
                ] = 1
            
        h5_file.close()

        img, mask, coords, tip_coords_original = transformWithLabel(img, mask, tip_coords, tip_coords_original)
        
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