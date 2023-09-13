# Masterarbeit

### This project contains a deep learning approach for needle traking in 3D US images. Three alternative ways to train the segmentation network are fully implemented and tested. The work was based on the PyTorch framework. Full dataset is stored at the mtec server. HDF5 file from this archive is a sample file to show the structure of the stored data.

##### The training was conducted in files main_tip_mask.py (for Mask 1), main_full_mask.py (for Mask 2) and main_frame_diff_mask.py (for Mask 3). "models->unet.py" contains implementation of U-Net, based on the code from elektronn3. Package "utils" contains all helper function for processing files with different extensions (HDF5, MHD, RAW); for datasets creation and augmentation; for creating and comparing masks; for metrics calculation. drafts.ipynb is a playground where different appproaches were tested (in particular, intermediate outputs of the model, some figures for the thesis etc.). Directory "outputs" contains final weights for all three networks, some intermediate predictions and intermediate metrics in train_test_data.hdf5.