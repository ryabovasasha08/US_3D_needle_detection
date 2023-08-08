from tqdm import tqdm
import h5py
from utils.type_reader import get_image_array, mha_read_header
from utils.mask_utils import get_needle_mask_of_frame_sequence
from utils.labels_utils import get_labels
from scipy import ndimage
import numpy as np
import os
import random
from utils.type_reader import mha_read_header
import numpy as np
from utils.mask_utils import get_needle_mask_of_frame
from utils.labels_utils import get_labels
from tqdm import tqdm
from utils.labels_utils import get_all_files_mhd


from tqdm import tqdm
import h5py
from utils.type_reader import get_image_array, mha_read_header
from utils.mask_utils import get_needle_mask_of_frame_sequence
from scipy import ndimage
import numpy as np
import os
import random
from utils.type_reader import mha_read_header
import numpy as np
from utils.labels_utils import get_labels
from tqdm import tqdm
from utils.labels_utils import get_all_files_mhd

# import all files from necessary directory
all_files_mhd = get_all_files_mhd("../train/train_depth_0_70/")
len(all_files_mhd)

# merge all images in one huge array

all_frames_filenames_array = np.empty((1))
frame_nums = np.empty((1))
labels = np.empty((1, 3))
i = 0
for f in all_files_mhd:
    if (i%20 == 0):
        print("File "+str(i)+"...")
    info = mha_read_header(f)
    labels = np.concatenate((labels, get_labels(f, info)), axis = 0)
    all_frames_filenames_array = np.concatenate((all_frames_filenames_array, [f]*info['Dimensions'][3]), axis=0)
    frame_nums = np.concatenate((frame_nums, np.arange(0, info['Dimensions'][3])), axis=0)
    i+=1
    
all_frames_filenames_array = all_frames_filenames_array[ 1:]
labels = labels[1:, :]
frame_nums = frame_nums[1:]

X = np.vstack((all_frames_filenames_array, frame_nums)).transpose()
y = labels

CROP_TO = 128
startFileIdx = 0

for i in range(len(all_files_mhd)):
    f = all_files_mhd[i][:-4].split("/")[-1]
    input_image = np.transpose(get_image_array(all_files_mhd[i]),(3, 0, 1, 2))
    frameTotal = input_image.shape[0]
    
    print(startFileIdx, frameTotal)
        
    if os.path.exists('../train/resized_train/'+f+'.hdf5'):
        print('skip:'+str(i)+" "+f)
        new_file = h5py.File('../train/resized_train/'+f+'.hdf5', 'a')
        if "labels_original" not in list(new_file.keys()):
            new_file.create_dataset("labels_original", data=y[(startFileIdx):(startFileIdx+frameTotal), :])
        new_file.close()
        startFileIdx += frameTotal
        continue
    
    if np.any(y[(startFileIdx):(startFileIdx+frameTotal), :]<0):
        print(f+" has wrong labels. Skipping..")
        startFileIdx += frameTotal
        continue
    
    new_file = h5py.File('../train/resized_train/'+f+'.hdf5', 'w')
    new_file.create_dataset("labels_original", data=y[(startFileIdx):(startFileIdx+frameTotal), :])
        
    for frameNum in tqdm(range(frameTotal)):

        img = input_image[frameNum]
        tip_coords = y[frameNum+startFileIdx, :]
        mask = get_needle_mask_of_frame(img, tip_coords)

        #print(f, frameTotal, frameNum, frameNum+startFileIdx)

        ratio = CROP_TO/img.shape[0]
        tip_coords_resized = np.around(tip_coords*ratio).astype(int)
        image_resized = ndimage.zoom(img, (ratio, ratio, ratio))
        mask_resized = ndimage.zoom(mask, (ratio, ratio, ratio))

        mean = np.mean(image_resized)
        std = np.std(image_resized)
        image_resized = (image_resized - mean) / std

        transformNum = 10

        new_file.create_dataset("img"+"_"+str(frameNum)+"_"+str(transformNum), data=image_resized, dtype='f4')
        new_file.create_dataset("mask"+"_"+str(frameNum)+"_"+str(transformNum), data=mask_resized, dtype='f4')
        new_file.create_dataset("labels"+"_"+str(frameNum)+"_"+str(transformNum), data=tip_coords_resized, dtype='f4')
        
        #print("before transforms")

        for transformNum in range(10):
            k = 1000
            while not all(coord > k + 5 and coord < k + CROP_TO - 5 for coord in tip_coords): # if needle tip is within cropped area, then crop
                k = random.randint(0, img.shape[1]-CROP_TO)
                
            img_cropped = img[k:k + CROP_TO, k:k + CROP_TO, k:k + CROP_TO]
            mask_cropped = mask[k:k + CROP_TO, k:k + CROP_TO, k:k + CROP_TO]
            tip_coords_cropped = np.array(tip_coords)-k
            
            new_file.create_dataset("img"+"_"+str(frameNum)+"_"+str(transformNum), data=img_cropped, dtype='f4')
            new_file.create_dataset("mask"+"_"+str(frameNum)+"_"+str(transformNum), data=mask_cropped, dtype='f4')
            new_file.create_dataset("labels"+"_"+str(frameNum)+"_"+str(transformNum), data=tip_coords_cropped, dtype='f4')
            
    new_file.close()
    
    #os.remove('../train/trainh5_chunked/'+f+'.hdf5')
    #os.remove('../train/needle_masks_h5_chunked/'+f+'.hdf5')

    startFileIdx += frameTotal