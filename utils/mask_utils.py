import torch
from skimage.filters import threshold_otsu
from tqdm import tqdm
import numpy as np
import torch 
import numpy as np
from skimage.measure import label


# First define the largest connected component of the 4D slice, 
# then calculate the mean of its pixels' coordinates
def get_center_of_nonzero_4d_slice(tensor_4d):
    mask = torch.squeeze(tensor_4d, dim=0).detach().cpu().numpy()
    # Label connected components
    labels = label(mask)
    if len(np.bincount(labels.flat)[1:])>0:
        largest_cc = np.bincount(labels.flat)[1:].argmax() + 1
    else:
        # return the origin
        print("np.bincount(labels.flat)[1:] is empty")
        return (0, 0, 0)

    # Create density map 
    mask_largest_cc = np.zeros_like(mask)
    mask_largest_cc[labels==largest_cc] = 1 

    # Get indices of pixels with value 1
    idx = np.where(mask_largest_cc == 1) 

    # Get coordinates   
    x = idx[0]
    y = idx[1]
    z = idx[2]

    # Calculate mean 
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)
    
    return x_mean, y_mean, z_mean

# For the accuracy 
def get_ends_of_nonzero_4d_slice(tensor_4d):
    mask = torch.squeeze(tensor_4d, dim=0).detach().cpu().numpy()
    # Label connected components
    labels = label(mask)
    if len(np.bincount(labels.flat)[1:])>0:
        largest_cc = np.bincount(labels.flat)[1:].argmax() + 1
    else:
        # return the origin
        print("np.bincount(labels.flat)[1:] is empty")
        return [(0, 0, 0), (0, 0, 0)]

    # Create density map 
    mask_largest_cc = np.zeros_like(mask)
    mask_largest_cc[labels==largest_cc] = 1 

    # Get indices of pixels with value 1
    idx = np.where(mask_largest_cc == 1) 

    # Get coordinates   
    x = idx[0]
    y = idx[1]
    z = idx[2]
    

    # calculate along which coordinates the range of values is the biggest (= which side is the longest). 
    # It is never the y axis (height), so either x or z
    
    range1 = max(x) - min(x)  
    range3 = max(z) - min(z)
    
    y_mean = np.mean(y)
    
    if range1>range3: # so needle is located in the volume along x axis
        z_mean = np.mean(z)
        x_min = min(x)
        x_max = max(x)
        return [(x_min, y_mean, z_mean), (x_max, y_mean, z_mean)]
    else: # so needle is located in the volume along z axis
        x_mean = np.mean(x)
        z_min = min(z)
        z_max = max(z)
        return [(x_mean, y_mean, z_min), (x_mean, y_mean, z_max)]


def contains_in_needle(point, vertex, radius, tip_length):
    
    # first check if x coord is greater than x_vertex. If it is - point is for sure not part of the needle
    if point[0]>np.ceil(vertex[0]):
        return False
    
    # If x coord is within x_vertex-length..x_vertex, than it may be part of the needle tip cone
    if point[0] > np.floor(vertex[0]-tip_length):
    
        # then calculate the radius of the slice perpendicular to the cone at point[0]
        radius_slice = (vertex[0]-point[0])*radius/tip_length

        # Now calculate 2d distance between point and cone axis in the plane YZ
        point_dist = np.linalg.norm(point[1:3]-vertex[1:3])

        # If point_dist is bigger than radius_slice, then point is out of the needle tip cone
        return point_dist <= radius_slice+np.sqrt(2)/3
    
    # otherwise x coord is less than x_vertex-length so point may be part of the main needle cylinder
    
    # calculate 2d distance between point and needle center axis in the plane YZ
    point_dist = np.linalg.norm(point[1:3]-vertex[1:3])
    
    return point_dist <= radius+np.sqrt(2)/3


 # needle_diam - in mm, far from needle tip
 # tip_length - #in mm, distance from the needle tip until the moment when needle reached needle_diam
def get_needle_mask_of_frame_sequence(sequence_4d, labels, needle_diam=2.5, tip_length = 4, resolution=3):
     
     frameSequenceMask = np.zeros(sequence_4d.shape)
     frame_3d = sequence_4d[:, :, :, 0]
     coords_3d = np.indices(frame_3d.shape).reshape(3, -1).T
     
     for frame_num in range(0, sequence_4d.shape[3]):
          for i in tqdm(range(0,len(coords_3d))):
               coord = coords_3d[i]
               if contains_in_needle(coord, labels[frame_num], needle_diam*resolution/2, tip_length*resolution):
                    frameSequenceMask[coord[0], coord[1], coord[2], frame_num] = 1
                    
     frameSequenceMask[sequence_4d==0]=0
                    
     return frameSequenceMask
 
def get_needle_mask_of_frame(sequence_3d, label, needle_diam=2.5, tip_length = 4, resolution=3):
     
    frameMask = np.zeros(sequence_3d.shape)
    coords_3d = np.indices(sequence_3d.shape).reshape(3, -1).T
     
    for i in tqdm(range(0,len(coords_3d))):
        coord = coords_3d[i]
        if contains_in_needle(coord, label, needle_diam*resolution/2, tip_length*resolution):
            frameMask[coord[0], coord[1], coord[2]] = 1
                    
    frameMask[sequence_3d==0]=0
                   
    return frameMask