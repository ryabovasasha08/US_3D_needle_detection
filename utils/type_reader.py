import os
import numpy as np
from utils.type_reader import get_image_array
import h5py
from tqdm import tqdm

'''
Function for reading the header of an Insight Meta-Image (.mha,.mhd) file

'''

def mha_read_header(filename):
    
    info = {}
    
    if not filename:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        filename = filedialog.askopenfilename(filetypes=[('MHA Files', '*.mha')], title='Read mha-file')
    
    fid = open(filename, 'rb')
    
    info['Filename'] = filename
    info['Format'] = 'MHA'
    info['CompressedData'] = 'false'
    readelementdatafile = False
    
    while not readelementdatafile:
        str = fid.readline().decode('utf-8')
        s = str.find('=')
        if s != -1:
            type = str[:s-1].strip()
            data = str[s+1:].strip()
        else:
            type = ''
            data = str.strip()
        
        if type.lower() == 'ndims':
            info['NumberOfDimensions'] = list(map(int, data.split()))
        elif type.lower() == 'dimsize':
            info['Dimensions'] = list(map(int, data.split()))
        elif type.lower() == 'elementspacing':
            info['PixelDimensions'] = list(map(float, data.split()))
        elif type.lower() == 'elementsize':
            info['ElementSize'] = list(map(float, data.split()))
            if 'PixelDimensions' not in info:
                info['PixelDimensions'] = info['ElementSize']
        elif type.lower() == 'elementbyteordermsb':
            info['ByteOrder'] = data.lower()
        elif type.lower() == 'anatomicalorientation':
            info['AnatomicalOrientation'] = data
        elif type.lower() == 'centerofrotation':
            info['CenterOfRotation'] = list(map(float, data.split()))
        elif type.lower() == 'offset':
            info['Offset'] = list(map(float, data.split()))
        elif type.lower() == 'binarydata':
            info['BinaryData'] = data.lower()
        elif type.lower() == 'compresseddatasize':
            info['CompressedDataSize'] = list(map(int, data.split()))
        elif type.lower() == 'objecttype':
            info['ObjectType'] = data.lower()
        elif type.lower() == 'transformmatrix':
            info['TransformMatrix'] = list(map(float, data.split()))
        elif type.lower() == 'compresseddata':
            info['CompressedData'] = data.lower()
        elif type.lower() == 'binarydatabyteordermsb':
            info['ByteOrder'] = data.lower()
        elif type.lower() == 'elementdatafile':
            info['DataFile'] = data
            readelementdatafile = True
        elif type.lower() == 'elementtype':
            info['DataType'] = data[5:].lower()
        elif type.lower() == 'headersize':
            val = list(map(int, data.split()))
            if val[0] > 0:
                info['HeaderSize'] = val[0]
        else:
            info[type] = data
    
    data_type = info['DataType']
    if data_type == 'char':
        info['BitDepth'] = 8
    elif data_type == 'uchar':
        info['BitDepth'] = 8
    elif data_type == 'short':
        info['BitDepth'] = 16
    elif data_type == 'ushort':
        info['BitDepth'] = 16
    elif data_type == 'int':
        info['BitDepth'] = 32
    elif data_type == 'uint':
        info['BitDepth'] = 32
    elif data_type == 'float':
        info['BitDepth'] = 32
    elif data_type == 'double':
        info['BitDepth'] = 64
    else:
        info['BitDepth'] = 0
    
    if 'HeaderSize' not in info:
        info['HeaderSize'] = fid.tell()
    
    fid.close()
    
    return info

'''
Function for retrieving a 3d image based on the .mhd file

'''

def get_image_array(full_fileName_mhd):
    
    full_fileName_raw = full_fileName_mhd[:-3]+"raw"
    
    '''
    
    if os.path.isfile(full_fileName_mhd) and os.path.isfile(full_fileName_raw):
        print('US-files exist.')
    else:
        warningMessage = f'Warning: one or both US-files do not exist. Please check the path-name:\n{full_fileName_mhd}\n{full_fileName_raw}'
        input('Press Enter to continue...')
        raise ValueError(warningMessage) 
        
    '''
    
    # Get information from mhd-file:
    info_US = mha_read_header(full_fileName_mhd)

    # US-image-size -> calculate number of values in 1 subse
    size_1_slice = info_US['Dimensions'][0]*info_US['Dimensions'][1]*info_US['Dimensions'][2]

    # open raw-file
    fileID_US = open(full_fileName_raw, 'rb')
    bn = 0
    batchSize = 100
    img_total = []

    while True:
        image_US = []
        raw = []
        bn += 1
        data = fileID_US.read(size_1_slice * batchSize * 4) # Assuming float32 data type (4 bytes)
        
        if not data:
            break

        raw = np.frombuffer(data, dtype=np.float32)
        
        if len(raw) % size_1_slice == 0:
            image_US = raw.reshape((info_US['Dimensions'][0], info_US['Dimensions'][1], info_US['Dimensions'][2], len(raw) // size_1_slice), order="F")
        else:
            slices_complete = len(raw) // size_1_slice
            raw = raw[:size_1_slice * slices_complete]
            image_US = raw.reshape((info_US['Dimensions'][0], info_US['Dimensions'][1], slices_complete), order="F")
        
        if not img_total:
            img_total = image_US
        else:
            img_total = np.concatenate((img_total,image_US), axis=2)

    fileID_US.close()

    return img_total

#---------------------------
# ONLY NEEDS TO BE EXECUTED ONCE!
#---------------------------
def store_all_data_as_h5(filenames_array):
    # read all files and store them under corresponding names 
    for filename in tqdm(X[:, 0]):
        f = filename[:-4].split("/")[-1]
        new_file = h5py.File('../train/trainh5/'+f+'.hdf5', 'w')
        new_file.create_dataset("default", data=get_image_array(filename))
        new_file.close()