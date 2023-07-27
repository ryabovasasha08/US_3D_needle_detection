import torch
import numpy as np
from utils.mask_utils import get_center_of_nonzero_4d_slice
from utils.mask_utils import binarize_with_otsu

def get_pixel_accuracy_percent(inputs, targets):
        
    # Flatten tensors 
    mask_flat = targets.view(-1)  
    pred_flat = binarize_with_otsu(inputs).view(-1)

    # Calculate number of same predictions
    num_same = torch.sum(mask_flat == pred_flat)

    # Total number of pixels 
    total_pixels = mask_flat.numel()  

    # Pixel Accuracy
    acc = num_same / total_pixels

    return acc.item()*100

def get_central_pixel_distance(inputs, labels):
    batch_size = len(inputs)
    distance = 0
        
    for i in range(0, batch_size):
        input_center = get_center_of_nonzero_4d_slice(inputs[i])
        distance += np.linalg.norm(np.array(input_center) - labels[i].cpu().numpy())
    
    return distance/batch_size