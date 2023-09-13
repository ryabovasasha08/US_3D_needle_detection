import matplotlib.pyplot as plt
from utils.type_reader import get_image_array
import numpy as np
import torch
from matplotlib import patches

def show_sample(sample):
    fig = plt.figure(figsize=(12, 12))
        
    x = sample['label'][0].int()
    y = sample['label'][1].int()
    z = sample['label'][2].int()

    ax1 = fig.add_subplot(1, 3, 1)
    plt.title("OYZ")
    plt.imshow(sample['image'][0, x, :, :], interpolation='none')
    plt.imshow(sample['mask'][0, x, :, :], interpolation='none', alpha = 0.7)
    ax1.add_patch(patches.Circle((y, z), radius=0.5, color='red'))


    ax2 = fig.add_subplot(1, 3, 2)
    plt.title("OXZ")
    plt.imshow(sample['image'][0, :, y, :], interpolation='none')
    plt.imshow(sample['mask'][0, :, y, :], interpolation='none', alpha = 0.7)
    ax2.add_patch(patches.Circle((z, x), radius=0.5, color='red'))


    ax3 = fig.add_subplot(1, 3, 3)
    plt.title("OXY")
    plt.imshow(sample['image'][0, :, :, z].T, interpolation='none')
    plt.imshow(sample['mask'][0, :, :, z].T, interpolation='none', alpha = 0.7)
    ax3.add_patch(patches.Circle((x, y), radius=0.5, color='red'))

    plt.axis('off')
    plt.show()