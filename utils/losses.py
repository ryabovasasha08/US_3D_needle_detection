import torch.nn as nn
import torch
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target, smooth=1e-5):
        # Flatten the input and target tensors
        input = input.view(input.size(0), -1)
        target = target.view(target.size(0), -1)

        # Compute intersection and union
        intersection = (input * target).sum(dim=1)
        union = input.sum(dim=1) + target.sum(dim=1)

        # Calculate Dice score
        dice_score = (2 * intersection + smooth) / (union + smooth)

        # Compute the average Dice score across the batch
        dice_loss = 1 - dice_score.mean()

        return dice_loss
    
    
# PyTorch
# modified to add more weight to the input in denominator to punish big masks harder
class IoULossModified(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULossModified, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs_weight = 0.5
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs_weight * inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
        
        # Punish large predicted masks   
        size_penalty = torch.clamp(inputs.sum() / targets.sum(), max=1)  
                
        return 1 - IoU * size_penalty