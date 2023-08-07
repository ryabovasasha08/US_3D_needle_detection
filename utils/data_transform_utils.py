from tqdm import tqdm
import h5py
from utils.type_reader import get_image_array, mha_read_header
from utils.mask_utils import get_needle_mask_of_frame_sequence
from utils.labels_utils import get_labels
import numpy as np

#---------------------------
# ONLY NEEDS TO BE EXECUTED ONCE!
#---------------------------
def store_all_data_as_h5(filenames_array):
    # read all files and store them under corresponding names 
    for filename in tqdm(filenames_array):
        f = filename[:-4].split("/")[-1]
        new_file = h5py.File('../train/trainh5/'+f+'.hdf5', 'w')
        imageData = np.transpose(get_image_array(filename),(3, 0, 1, 2))
        new_file.create_dataset("default", data=imageData, chunks=(1, imageData.shape[1], imageData.shape[2], imageData.shape[3]), dtype='f4')
        new_file.close()
        

# 227 - number of pixels in the needle tip cone
# 6776 - number of pixels in the needle cylinder
# 176 - theoretical volume of the cone
# 5345 - theoretical volume of the needle cylinder

def create_and_store_needle_masks_as_h5(filenames_array):
    for filename in tqdm(filenames_array):
        f = filename[:-4].split("/")[-1]
        new_file = h5py.File('../train/needle_masks_h5/'+f+'.hdf5', 'w')
        sequence_4d = get_image_array(filename)
        info = mha_read_header(filename)
        labels = get_labels(f, info)
        needle_mask_sequence = get_needle_mask_of_frame_sequence(sequence_4d, labels)
        needle_mask_sequence_frame_first = np.transpose(needle_mask_sequence,(3, 0, 1, 2))
        img_shape = needle_mask_sequence_frame_first.shape
        new_file.create_dataset("default", data=needle_mask_sequence_frame_first, chunks=(1, img_shape[1], img_shape[2], img_shape[3]))
        new_file.close()
        
        
#%% to check the function above use:
'''
from matplotlib import pyplot as plt, patches
from utils.type_reader import get_image_array

# test needle mask creating and saving
import h5py
from utils.type_reader import get_image_array, mha_read_header
from utils.mask_utils import get_needle_mask_of_frame_sequence
from utils.labels_utils import get_labels
import numpy as np

filename = all_files_mhd[0]
f = filename[:-4].split("/")[-1]
sequence_4d = get_image_array(filename)
info = mha_read_header(filename)
labels = get_labels(f, info)
needle_mask_sequence = get_needle_mask_of_frame_sequence(sequence_4d, labels)

#Check that frames are being displayed correctly
frame_num = 10
print(labels[frame_num])
fig = plt.figure()
ax = fig.add_subplot()
plt.imshow(sequence_4d[:, :, int(labels[frame_num][2]), frame_num].T)
plt.imshow(needle_mask_sequence[:, :, int(labels[frame_num][2]), frame_num].T, alpha = 0.4)
ax.add_patch(patches.Circle((int(labels[frame_num][0]), int(labels[frame_num][1])), radius=1, color='red'))
plt.show()
'''