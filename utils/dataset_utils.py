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

def getFrameDiffDatasetDatapointResized(self, input_image, mask, labels, resizeTo):     
    us_tip_coords = labels.astype(float)
    
    #resize everything to 128*128*128 and normalize
    ratio = resizeTo/input_image.shape[0]
    us_tip_coords_resized = np.around(us_tip_coords*ratio).astype(int)
    us_tip_coord_flattened = (resizeTo * resizeTo * us_tip_coords_resized[2]) + (resizeTo * us_tip_coords_resized[1]) + us_tip_coords_resized[0]
    
    input_image = ndimage.zoom(input_image, (ratio, ratio, ratio))[np.newaxis, :, :, :]
    mean = np.mean(input_image)
    std = np.std(input_image)
    input_image = (input_image - mean) / std
    
    mask = ndimage.zoom(mask, (ratio, ratio, ratio))[np.newaxis, :, :, :]
    mean = np.mean(mask)
    std = np.std(mask)
    mask = (mask - mean) / std

    return input_image, mask, us_tip_coords_resized, us_tip_coord_flattened


# since resolution is 3px=mm and we have +-1mm error, default mask is 3px radius
# 135 is initial image size so also the default ResizeTo
def getImageDatasetDatapointResized(filename, frame_num, labels, mask_diam=6, resizeTo = 135): 
    # create frame image based on frame number and filename
    f = filename[:-4].split("/")[-1]
    h5_file = h5py.File('../train/trainh5_chunked/'+f+'.hdf5', 'r')
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
        
def cropOrResizeTransformWithLabel(image, mask, tip_coords, cropTo):
    k = random.randint(0, image.shape[1]-cropTo)
    
    if all(coord > k + 5 and coord < k + cropTo - 5 for coord in tip_coords): # if needle tip is within cropped area, then crop
        image = image[k:k + cropTo, k:k + cropTo, k:k + cropTo]
        mask = mask[k:k + cropTo, k:k + cropTo, k:k + cropTo]
        tip_coords = np.array(tip_coords)-k
            
    else: # else resize
        #resize everything to 128*128*128
        ratio = cropTo/image.shape[0]
        tip_coords = np.around(tip_coords*ratio).astype(int)
        image = ndimage.zoom(image, (ratio, ratio, ratio))
        mask = ndimage.zoom(mask, (ratio, ratio, ratio))
     
    return image, mask, tip_coords


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
    

def transformAndCropWithLabel(img, mask, tip_coords, cropTo):
    # add crop to cropTo size,  add random rotation and horizontal flip
    img, mask, tip_coords = rotateTransformWithLabel(img, mask, tip_coords)
    img, mask, tip_coords = flipxTransformWithLabel(img, mask, tip_coords)
    img, mask, tip_coords = flipzTransformWithLabel(img, mask, tip_coords)
    #crop transform should be the last, because it crops image to the even dimensions and prev transforms work on odd dimensions with center pixel
    img, mask, tip_coords = cropOrResizeTransformWithLabel(img, mask, tip_coords, cropTo)
    return normalize(img).copy()[np.newaxis, :, :, :], mask.copy()[np.newaxis, :, :, :], tip_coords.copy()
