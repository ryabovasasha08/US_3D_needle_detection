import matplotlib.pyplot as plt
from utils.type_reader import get_image_array

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
    