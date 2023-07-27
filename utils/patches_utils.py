import h5py

# create and return 3 2d-patches around some pixel
# patch size has to be 1)odd to make the pixel in question to be exactly in the center, 2) divisible by number of layers of network
def get_2d_patches(filename, frame_num, pixel_coords, patch_size=21):
    f = filename[:-4].split("/")[-1]
    h5_file = h5py.File('../train/trainh5/'+f+'.hdf5', 'r')
    input_image = h5_file['default'][frame_num]
    h5_file.close()
    patch_yx = input_image[int(pixel_coords[0]-(patch_size-1)/2):int(pixel_coords[0]+(patch_size-1)/2), int(pixel_coords[1]-(patch_size-1)/2):int(pixel_coords[1]+(patch_size-1)/2), pixel_coords[2]].T
    patch_yz = input_image[pixel_coords[0], int(pixel_coords[1]-(patch_size-1)/2):int(pixel_coords[1]+(patch_size-1)/2), int(pixel_coords[2]-(patch_size-1)/2):int(pixel_coords[2]+(patch_size-1)/2)]
    patch_xz = input_image[int(pixel_coords[0]-(patch_size-1)/2):int(pixel_coords[0]+(patch_size-1)/2), pixel_coords[1], int(pixel_coords[2]-(patch_size-1)/2):int(pixel_coords[2]+(patch_size-1)/2)]
    return [patch_yx, patch_xz, patch_yz]


# define the class of the pixel (0 - not needle pixel, 1-needle pixel)
# pixel coords is an array of 3 coords
def get_pixel_class(filename, frame_num, pixel_coords):
    f = filename[:-4].split("/")[-1]
    needle_mask_h5_file = h5py.File('../train/needle_masks_h5/'+f+'.hdf5', 'r')
    needle_mask = needle_mask_h5_file['default'][frame_num]
    needle_mask_h5_file.close()
    return needle_mask[tuple(pixel_coords)]