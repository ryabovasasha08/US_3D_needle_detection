import torch
from skimage.filters import threshold_otsu


def get_center_of_nonzero_4d_slice(tensor_4d):
    # Get nonzero indices
    nz_indices = torch.nonzero(tensor_4d)

    # Find min and max indices 
    na, xmin, ymin, zmin = nz_indices.min(dim=0)[0].tolist()  
    na, xmax, ymax, zmax = nz_indices.max(dim=0)[0].tolist()

    # Calculate center point
    x_center = (xmin + xmax) // 2
    y_center = (ymin + ymax) // 2 
    z_center = (zmin + zmax) // 2

    center = (x_center, y_center, z_center)

    return center



def binarize_with_otsu(inputs):
    thresh = threshold_otsu(inputs.detach().cpu().numpy())
    binary = inputs > thresh
    return binary.int()