import numpy as np
from os import listdir
import os
from os.path import isfile, join

# import all files from necessary directory
def get_all_files_mhd(dir = "/data/Riabova/train3"):
    all_files = [join(dir, f) for f in listdir(dir)]
    all_files_mhd = [string for string in all_files if string.endswith("mhd")]
    return all_files_mhd

def get_labels(f, info):
    labels = np.zeros((info['Dimensions'][3], 3))
        
    for frame in range(0, info['Dimensions'][3]):
        if frame > 9:
            info_frame_timestamp_field = "Seq_Frame00" + str(frame) + "_Timestamp"
        else:
            info_frame_timestamp_field = "Seq_Frame000" + str(frame) + "_Timestamp"
            
        frame_timestamp = float(info[info_frame_timestamp_field])
        
        # filename in format 'x_y_z_alpha_beta_gamma_velocity_axis_timestamp
        filename = f.split("/")[-1].split(".")[0] # 0_22_5_0_0_0_3_1_1686308127391259
        frame_timestamp_micros = frame_timestamp*1000000 #in microseconds, correct
        
         # extract params from filename
        filename_params = filename.split('_')
        start_timestamp_micros = float(filename_params[-1]) # in micros
        axis = int(filename_params[-2])
        velocity = int(filename_params[-3])  # in mm/s

        # get tip coords in hexapode system
        dt = frame_timestamp_micros - start_timestamp_micros  # in micros
        distance = velocity * dt / 1000000  # in mm
        hex_coords = [0, 0, 0]
        for i in range(3):
            hex_coords[i] = float(filename_params[i])
            if axis == i+1:
                hex_coords[i] += distance
                        
        # get tip coords in US system
        labels[frame, :] = transformToUS(hex_coords)
        
    return labels


''' 
Example usage:
    coords_1 = [x1, x2, x3]  # Replace x1, x2, x3 with the actual values
    method = "hex2us"  # Or "us2hex"
    isShifted = True  # Or False
    coords_2 = transformTo(coords_1, method, isShifted) 

'''
def transformTo(coords_1, method, isShifted):
    '''
    -------------------------------------------------------
    ---- FOR OLD_DATA AND OLD_CALIBRATION_METHOD-----------
    --------------------------------------------------------
    # Transformation matrices
    T_needle_tip = np.array([[-0.1601, 3.1823, 0.1383, 13.8778],
                            [0.0178, 0.0806, -3.3695, 112.5630],
                            [-2.4504, -0.1146, -0.3139, 110.3655],
                            [-0.0000, -0.0000, -0.0000, 1.0000]])

    T_needle_tip_shifted = np.array([[-0.1601, 3.1823, 0.1383, 20.8778],
                                     [0.0178, 0.0806, -3.3695, 109.5630],
                                     [-2.4504, -0.1146, -0.3139, 110.3655],
                                     [-0.0000, -0.0000, -0.0000, 1.0000]])

    if isShifted:
        T = T_needle_tip_shifted
    else:
        T = T_needle_tip'''
        
    T = np.array([[-0.0000,    3.2500,    0.0064,  97.3511],
                  [-0.0000,    0.0000,   -3.3511,   99.4415],
                  [-3.3500,   -0.0000,   -0.0000,  159.1625],
                  [0.0000,   -0.0000,   -0.0000,    1.0000]])


    coords_1 = np.append(coords_1, 1)  # Add 1 to coords_1

    if method == "hex2us":
        coords_2 = np.dot(T, coords_1.T)
    else:
        coords_2 = np.linalg.solve(T, coords_1.T)  # Equivalent to inv(T) * coords_1.T

    coords_2 = coords_2.T
    coords_2 = coords_2[:-1]

    return coords_2


def transformToUS(coords_1):
    return transformTo(coords_1, "hex2us", True)