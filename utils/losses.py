import torch.nn as nn
import torch
import torch.nn.functional as F    
    
# PyTorch
# modified to add more weight to the input in denominator to punish big masks harder
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=0.1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = (torch.sigmoid(inputs) > 0.5).float()
   
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
        
        return 1 - IoU


'''
Here are some best practices for modifying the Jaccard index (Intersection over Union) to punish large mask predictions:

Use log scale on mask sizes:
Take log of mask sizes before computing IoU. This makes the metric more sensitive to differences between small masks compared to large masks.

Add a mask size regularization term:
IoU_modified = IoU - lambda * mask_size

The regularization constant lambda controls how much to penalize large masks. Tune lambda based on your dataset.

Power scale the intersection:
Instead of simple intersection, take intersection^p, where p < 1. Smaller p more strongly penalizes large intersections (large masks).

Only scale ground truth size:
IoU_modified = Intersection / (gt_mask_size + Pred_mask_size - Intersection)
'''
class IoULossModified(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULossModified, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        inputs = (inputs>0.5).int()
       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        iou = (intersection + smooth) / (union + smooth)
        
        # TODO: add weighting
        
        return 1 - iou
    
    
# PyTorch
# This implements the complete IoU loss as described in the paper "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression" by Zheng et al. (AAAI 2020).
# It computes the standard IoU between predicted and target boxes, then penalizes predictions that enclose target boxes by a larger area.
# We will use binarized sigmoid predictions
class CompleteIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CompleteIoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = torch.sigmoid(inputs)
        inputs = (inputs>0.5).int()
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
        
        enclosed_mask = (inputs + targets == 2).float()
        enclose_area = enclosed_mask.sum()
        ciou = IoU - 1/enclose_area * (enclose_area - union)
                
        return 1 - ciou