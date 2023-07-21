from tqdm.auto import tqdm
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *

class CNN_classification_model (nn.Module):
    
    # for side_size = 128 and shape (batch_size, 1, 128, 128, 128)
    def __init__(self, side_size = 128): # try with 64
        super(CNN_classification_model, self).__init__()
        
        layer_1_out_ = int(side_size/2)
        layer_2_out = side_size
        layer_3_out = side_size*2
        layer_linear_1_in = int(side_size*2 * (side_size/8 -2)*(side_size/8 -2)*(side_size/8 -2))
        layer_linear_1_out = 128
        layer_linear_2_out = 3

        print(int(side_size/2))
        
        self.model= nn.Sequential(
        
        #Conv layer 1    
        nn.Conv3d(1, layer_1_out, kernel_size=(3, 3, 3), padding=0), # output of shape: (batch_size, 64, 126, 126, 126) (batch_size, side_size/2, side_size - 2, side_size - 2, side_size - 2)
        nn.Softplus(beta=7),
        nn.MaxPool3d((2, 2, 2)),   # output of shape: (batch_size, 64, 63, 63, 63)
        
        #Conv layer 2  
        nn.Conv3d(layer_1_out, layer_2_out, kernel_size=(3, 3, 3), padding=0), # output of shape: (batch_size, 128, 61, 61, 61) (batch_size, side_size, side_size/2 -3, side_size/2 -3, side_size/2 -3)
        nn.Softplus(beta=7),
        nn.MaxPool3d((2, 2, 2)),  # output of shape: (batch_size, 128, 30, 30, 30) 
        
        #Conv layer 3  
        nn.Conv3d(layer_2_out, layer_3_out, kernel_size=(3, 3, 3), padding=0), # output of shape: (batch_size, 256, 28, 28, 28) (batch_size, side_size*2, side_size/4 -4, side_size/4 -4, side_size/4 -4)
        nn.Softplus(beta=7),
        nn.MaxPool3d((2, 2, 2)),  # output of shape: (batch_size, 256, 14, 14, 14)    (batch_size, side_size*2, side_size/8 -2,  side_size/8 -2,  side_size/8 -2)
            
        #Flatten
        nn.Flatten(),  # output of shape: (batch_size, 702464) (batch_size, side_size*2 * (side_size/8 -2,  side_size/8 -2,  side_size/8 -2)
        #Linear 1
        nn.Linear(layer_linear_1_in, layer_linear_1_out), # output of shape: (batch_size, 128) (batch_size, 128)
        #SoftPlus
        nn.Softplus(beta=7),
        #BatchNorm1d
        nn.BatchNorm1d(128),
        #Dropout
        nn.Dropout(p=0.15),
        #Linear 2
        nn.Linear(layer_linear_1_out, layer_linear_2_out) # output of shape: (batch_size, 3)
        )
    

    def forward(self, x):
        # Set 1
        out = self.model(x)
        return out