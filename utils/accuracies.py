import torch
import numpy as np
from utils.mask_utils import get_center_of_nonzero_4d_slice, get_ends_of_nonzero_4d_slice

def get_pixel_accuracy_percent(inputs, targets):
    """To get pixelwise accuracy. Invariant to dimensionality, as long as arguments have same shape
    Args:
        inputs: predicted mask by network. Type - Torch tensor
        targets: GT mask. Type - Torch tensor
    """
    
    inputs = torch.round(torch.sigmoid(inputs))
    targets = torch.round(targets)
        
    # Flatten tensors 
    mask_flat = targets.view(-1)  
    pred_flat = inputs.view(-1)

    # Calculate number of same predictions
    num_same = torch.sum(mask_flat == pred_flat)

    # Total number of pixels 
    total_pixels = mask_flat.numel()  

    # Pixel Accuracy
    acc = num_same / total_pixels

    return acc.item()*100

def get_precision(inputs, targets):
    """
    To get precision. 
    Precision is a ratio of correctly defined mask pixels wrt all defined mask pixels: TP/(TP+FP)
    Function is invariant to dimensionality, as long as arguments have same shape
    Args:
        inputs: predicted mask by network. Type - Torch tensor
        targets: GT mask. Type - Torch tensor
    """
    inputs = torch.round(torch.sigmoid(inputs))
    targets = torch.round(targets)
    
    # Flatten tensors 
    mask_flat = targets.view(-1).bool() 
    pred_flat = inputs.view(-1).bool()

    num_tp = torch.sum(mask_flat & pred_flat)
    num_p = torch.sum(pred_flat) 

    # Precision
    precision = num_tp / num_p

    return precision.item()*100

# recall is a ratio of correctly defined mask pixels wrt all GT mask pixels: TP/(TP+FN)
def get_recall(inputs, targets):
    """
    To get recall. 
    Recall is a ratio of correctly defined mask pixels wrt all GT mask pixels: TP/(TP+FN)
    Function is invariant to dimensionality, as long as arguments have same shape
    Args:
        inputs: predicted mask by network. Type - Torch tensor
        targets: GT mask. Type - Torch tensor
    """
    inputs = torch.round(torch.sigmoid(inputs))
    targets = torch.round(targets)

    # Flatten tensors 
    mask_flat = targets.view(-1).bool() 
    pred_flat = inputs.view(-1).bool()

    num_tp = torch.sum(mask_flat & pred_flat)
    num_gt_mask = torch.sum(mask_flat) 

    # Recall
    recall = num_tp / num_gt_mask

    return recall.item()*100
    

def get_central_pixel_distance(inputs, labels):
    """
    To calculate the 3D Euclidean distance between center of the predicted mask and GT tip coordinate. 
    Is used for distance estimation Mask 3 and Mask 1
    Args:
        inputs: batch of predicted masks by network. Type - 4D Torch tensor
        labels: batch of correponding GT labels (after augmentation).
    """
    
    inputs = torch.round(torch.sigmoid(inputs))
    batch_size = len(inputs)
    distance = 0
    
    batches_to_ignore = 0
        
    for i in range(0, batch_size):
        input_center = get_center_of_nonzero_4d_slice(inputs[i])
        if sum(input_center) == 0:
            batches_to_ignore +=1
        else:
            distance += np.linalg.norm(np.array(input_center) - labels[i].cpu().numpy())
    
    return distance/(batch_size-batches_to_ignore)

def get_full_mask_tip_pixel_distance(inputs, labels):
    
    """
    To predict needle tip position for Mask 2 and ccalculate the 3D Euclidean distance between this prediction and GT tip coordinate. 
    First calculates ends of the needle on each image from the batch, then defines "correct" end and calculates distance to GT label.
    Args:
        inputs: batch of predicted masks by network. Type - 4D Torch tensor
        labels: batch of correponding GT labels (after augmentation).
    """
    
    inputs = torch.round(torch.sigmoid(inputs))
    batch_size = len(inputs)
    distance = 0
    
    batches_to_ignore = 0
        
    for i in range(0, batch_size):
        input_ends = get_ends_of_nonzero_4d_slice(inputs[i])
        if sum(input_ends[0]) == 0 and sum(input_ends[1]) == 0:
            batches_to_ignore +=1
        else:    
            label_numpy =labels[i].cpu().numpy()
            distance += min(np.linalg.norm(np.array(input_ends[0]) - label_numpy), np.linalg.norm(np.array(input_ends[1]) - label_numpy))
    
    return distance/(batch_size-batches_to_ignore)