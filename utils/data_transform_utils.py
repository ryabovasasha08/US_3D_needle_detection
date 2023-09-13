from tqdm import tqdm
from utils.type_reader import get_image_array, mha_read_header
from utils.mask_utils import get_needle_mask_of_frame_sequence
from utils.labels_utils import get_labels
import numpy as np
import h5py
from utils.type_reader import get_image_array, mha_read_header
from utils.mask_utils import get_needle_mask_of_frame_sequence
from scipy import ndimage
import os
import random
from utils.mask_utils import get_needle_mask_of_frame
from utils.labels_utils import get_all_files_mhd

def store_all_data_as_h5(filenames_array):
    """
    Helper function to change format of all files from mhd+raw to hdf5
    
    ---------------------------
    ONLY NEEDS TO BE EXECUTED ONCE!
    ---------------------------
    """
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
    """
    Helper function to create hdf5 files of needle frames. 
    One image sequence corresponds to one hdf5 file, containing needle masks for each frame of the sequence
    
    ---------------------------
    ONLY NEEDS TO BE EXECUTED ONCE!
    ---------------------------
    """
    
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
        
        
def first_augmentation_step(all_files_mhd):
    """
    Helper function to create hdf5 files with pre-augmented data. 
    Each image sequence results in one hdf5 file, where:
        1) under names "img_frameNum_transformNum" are stored augmented images.
        2) under names "mask_frameNum_transformNum" are stored corresponding cropped/resized Masks2
        3) under names "label_frameNum_transformNum" are stored corresponding labels
    transformNum=0..9 corresponds to cropping, transformNum=10 correponds to resize
    
    ---------------------------
    ONLY NEEDS TO BE EXECUTED ONCE!
    ---------------------------
    """

    #Merge all files+frames in one array
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

        startFileIdx += frameTotal