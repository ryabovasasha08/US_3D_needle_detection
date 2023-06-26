import torch
import torch.nn as nn

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