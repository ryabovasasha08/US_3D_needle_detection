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


"""
This file contains different functions for augmenting 3D image together with its mask and tip label.
All augmentations are applied randomly. 
All augmentations are collected in transformWithLabel function.
"""

def flipxTransformWithLabel(img, mask, tip_coords, coords_original, IMG_INITIAL_SIZE = 235):
    rand_int = random.randint(0, 1)
    
    if rand_int == 0:
        pass # No rotation
    else:
        img = np.flip(img, axis=0)
        mask = np.flip(mask, axis=0)
        shift_to_origin_vector = np.array([(img.shape[0]-1)/2, 0, (img.shape[2]-1)/2])
        shift_to_origin_unresized_vector = np.array([(IMG_INITIAL_SIZE-1)/2, 0, (IMG_INITIAL_SIZE-1)/2])

        tip_coords = tip_coords - shift_to_origin_vector
        coords_original = coords_original - shift_to_origin_unresized_vector
        
        tip_coords = np.array([-tip_coords[0], tip_coords[1], tip_coords[2]])
        coords_original = np.array([-coords_original[0], coords_original[1], coords_original[2]])

        tip_coords = tip_coords + shift_to_origin_vector
        coords_original = coords_original + shift_to_origin_unresized_vector

    return img, mask, tip_coords, coords_original

def flipzTransformWithLabel(img, mask, tip_coords, coords_original, IMG_INITIAL_SIZE = 235):
    rand_int = random.randint(0, 1)
    
    if rand_int == 0:
        pass # No rotation
    else:
        img = np.flip(img, axis=2)
        mask = np.flip(mask, axis=2)
        shift_to_origin_vector = np.array([(img.shape[0]-1)/2, 0, (img.shape[2]-1)/2])
        shift_to_origin_unresized_vector = np.array([(IMG_INITIAL_SIZE-1)/2, 0, (IMG_INITIAL_SIZE-1)/2])

        tip_coords = tip_coords - shift_to_origin_vector
        coords_original = coords_original - shift_to_origin_unresized_vector
        
        tip_coords = np.array([tip_coords[0], tip_coords[1], -tip_coords[2]])
        coords_original = np.array([coords_original[0], coords_original[1], -coords_original[2]])
        
        tip_coords = tip_coords + shift_to_origin_vector
        coords_original = coords_original + shift_to_origin_unresized_vector
        
    return img, mask, tip_coords, coords_original

        
def rotateTransformWithLabel(img, mask, tip_coords, coords_original, IMG_INITIAL_SIZE = 235):
    rand_int = random.randint(0, 3)
    shift_to_origin_vector = np.array([(img.shape[0]-1)/2, 0, (img.shape[2]-1)/2])
    shift_to_origin_unresized_vector = np.array([(IMG_INITIAL_SIZE-1)/2, 0, (IMG_INITIAL_SIZE-1)/2])
    tip_coords = tip_coords - shift_to_origin_vector
    coords_original = coords_original - shift_to_origin_unresized_vector
  
    # Perform rotation based on random integer
    if rand_int == 0:
        pass # No rotation
    elif rand_int == 1: 
        img = np.rot90(img, k=1, axes=(0, 2)) # Rotate 90 degrees
        mask = np.rot90(mask, k=1, axes=(0, 2)) # Rotate 90 degrees
        tip_coords = np.array([-tip_coords[2], tip_coords[1], tip_coords[0]])
        coords_original = np.array([-coords_original[2], coords_original[1], coords_original[0]])
    elif rand_int == 2:
        img = np.rot90(img, k=2, axes=(0, 2)) # Rotate 180 degrees
        mask = np.rot90(mask, k=2, axes=(0, 2)) # Rotate 180 degrees
        tip_coords = np.array([-tip_coords[0], tip_coords[1], -tip_coords[2]])
        coords_original = np.array([-coords_original[0], coords_original[1], -coords_original[2]])
    elif rand_int == 3: 
        img = np.rot90(img, k=3, axes=(0, 2)) # Rotate 270 degrees
        mask = np.rot90(mask, k=3, axes=(0, 2))
        tip_coords = np.array([tip_coords[2], tip_coords[1], -tip_coords[0]])
        coords_original = np.array([coords_original[2], coords_original[1], -coords_original[0]])

    tip_coords = tip_coords + shift_to_origin_vector
    coords_original = coords_original + shift_to_origin_unresized_vector
    
    return img, mask, tip_coords, coords_original
        
def normalize(image):
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / std

        
def transformWithLabel(img, mask, tip_coords, coords_original):
    # add crop to cropTo size,  add random rotation and horizontal flip
    img, mask, tip_coords, coords_original = rotateTransformWithLabel(img, mask, tip_coords, coords_original)
    img, mask, tip_coords, coords_original = flipxTransformWithLabel(img, mask, tip_coords, coords_original)
    img, mask, tip_coords, coords_original = flipzTransformWithLabel(img, mask, tip_coords, coords_original)
    return normalize(img).copy()[np.newaxis, :, :, :], mask.copy()[np.newaxis, :, :, :], tip_coords.copy(), coords_original.copy()