import matplotlib.pyplot as plt
from utils.type_reader import get_image_array
import numpy as np
import torch


def show_sample(sample):
    plt.figure()
    
    plt.subplot(1, 3, 1)
    plt.title("OYZ")
    x = sample['label'][0]
    plt.imshow(sample['image'][0, x, :, :], cmap='seismic',  interpolation='none')
    plt.imshow(sample['mask'][0, x, :, :], cmap='jet',  interpolation='none', alpha = 0.7)
    
    plt.subplot(1, 3, 2)
    plt.title("OXZ")
    y = sample['label'][1]
    plt.imshow(sample['image'][0, :, y, :], cmap='seismic',  interpolation='none')
    plt.imshow(sample['mask'][0, :, y, :], cmap='jet',  interpolation='none', alpha = 0.7)
    
    plt.subplot(1, 3, 3)
    plt.title("OXY")
    z = sample['label'][2]
    plt.imshow(sample['image'][0, :, :, z], cmap='seismic',  interpolation='none')
    plt.imshow(sample['mask'][0, :, :, z], cmap='jet',  interpolation='none', alpha = 0.7)
    
    plt.axis('off')
    plt.show()
    

def compare_input_target(inp_mask, target_mask):
    target_mask_np = target_mask.detach().cpu().numpy()
    inp_mask_np = inp_mask.detach().cpu().numpy()
    print("Number of non-zero pixels of input: "+str(np.count_nonzero(inp_mask_np)) + " vs "+ str(np.count_nonzero(target_mask_np)))

    pattern = (target_mask_np != 0) & (inp_mask_np != 0)
    num_same_pixels = np.count_nonzero(pattern)
    print("Number of non-zero pixels on correct positions: "+str(num_same_pixels))