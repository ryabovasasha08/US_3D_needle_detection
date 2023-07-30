# %% [markdown]
# ## Prepare the datasets

# %% [markdown]
# <font size="2"> To install new library (since everything is running on kernel python3.10): 
# /$ python3.10 -m pip install *pandas*
# </font>

# %%
# import all files from necessary directory
from utils.labels_utils import get_all_files_mhd
all_files_mhd = get_all_files_mhd("/data/Riabova/train/train_depth_0_70/")
len(all_files_mhd)

# %%
from utils.type_reader import mha_read_header
import numpy as np
from utils.labels_utils import get_labels
from tqdm import tqdm

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

# %%
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image
from utils.type_reader import get_image_array

#Check that frames are being displayed correctly
input_image = get_image_array(all_files_mhd[0])
plt.imshow(input_image[:, :, 69, 3])

# %%
X = np.vstack((all_frames_filenames_array, frame_nums)).transpose()
y = labels
print(X.shape)
print(X[0, :])
print(y[0, :])

# %%
VALID_PERCENT = 0.2
TEST_PERCENT = 0.2
BATCH_TRAIN = 10
BATCH_VALID = 10
BATCH_TEST = 10
# note: original size - 235; resizing to 200 + batch size 5 caused cuda out of memory
RESIZE_TO = 128

# %%
from sklearn.model_selection import train_test_split

X_train, X_vt, y_train, y_vt = train_test_split(X[:, :], y[:, :], test_size=VALID_PERCENT+TEST_PERCENT, random_state=42, shuffle = True)
X_valid, X_test, y_valid, y_test = train_test_split(X_vt[:, :], y_vt[:, :], test_size=TEST_PERCENT/(VALID_PERCENT+TEST_PERCENT), random_state=42,shuffle=True)
print(f"Total training images: {X_train.shape[0]}")
print(f"Total validation images: {X_valid.shape[0]}")
print(f"Total test images: {X_test.shape[0]}")

# %%
X_train.shape

# %%
from utils.datasets import ImageDataset, my_collate
import torch

train_dataset = ImageDataset(X_train, y_train, resizeTo=RESIZE_TO)
valid_dataset = ImageDataset(X_valid, y_valid, resizeTo=RESIZE_TO)
test_dataset = ImageDataset(X_test, y_test, resizeTo=RESIZE_TO)
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_TRAIN,shuffle=True, collate_fn=my_collate)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset,batch_size=BATCH_VALID,shuffle=True, collate_fn=my_collate)
test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_TEST,shuffle=True, collate_fn=my_collate)

# %% [markdown]
# ## u-net
# ##### mask was created after image is resized to make the mask area a bit bigger

# %%
# init number of epochs to train for, and the
# batch size of train and validation sets
EPOCHS = 10
UNET_DEPTH = 4 # size of the image should divide by this number
UNET_START_FILTERS = 3

#For LR scheduler
INIT_LR = 0.001
WEIGHT_DECAY = 1e-8
MOMENTUM = 0.999

PATH_DIR = 'outputs_ciou'

# %%
import torch
# device will be 'cuda' if a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
device

# %%
import torch.nn as nn
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from models.unet import UNet
from utils.losses import CompleteIoULoss
from utils.save_model_utils import SaveBestModel

model = UNet(out_channels = 1, n_blocks=UNET_DEPTH, start_filts = UNET_START_FILTERS)
optimizer = optim.Adam(model.parameters()) #optim.RMSprop(model.parameters(),lr=INIT_LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM, foreach=True)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
criterion = CompleteIoULoss()
save_best_model = SaveBestModel(PATH_DIR)


# %%
model

# %%
import numpy as np
import torch
from utils.save_model_utils import save_model, save_plots, save_sample_mask
from utils.accuracies import get_central_pixel_distance, get_pixel_accuracy_percent

import numpy as np
import torch
from utils.save_model_utils import save_model, save_plots, save_sample_mask
from utils.accuracies import get_central_pixel_distance, get_pixel_accuracy_percent

class TrainerUNET:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False,
                 path_dir=''
                 ):

        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook
        self.path_dir = path_dir

        self.training_loss = []
        self.validation_loss = []
        self.center_pixel_distances = []
        self.pixelwise_accuracy = []
        self.learning_rate = []

    def run_trainer(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:

            """Epoch counter"""
            self.epoch += 1  # epoch counter
            
            print(f"[INFO]: Epoch {self.epoch} of {self.epochs}")

            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()
                
            print(f"Training loss: {self.training_loss[-1]:.3f}")
            print(f"Validation loss: {self.validation_loss[-1]:.3f}")
            # save the best model till now if we have the least loss in the current epoch
            save_best_model(self.validation_loss[-1], self.epoch, self.model, self.optimizer, self.criterion)
            save_model(self.epochs, self.model, self.optimizer, self.criterion, self.path_dir+'/epoch_'+str(self.epoch)+'_model.pth')
            save_plots(self.epochs, self.training_loss, self.validation_loss, self.center_pixel_distances, self.pixelwise_accuracy, self.path_dir)

            print('-'*50)
            
            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(self.validation_loss[i])  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.batch()  # learning rate scheduler step
        
        # save the trained model weights for a final time
        save_model(self.epochs, self.model, self.optimizer, self.criterion, self.path_dir+'/final_model.pth')
        # save the loss and accuracy plots
        save_plots(self.epochs, self.training_loss, self.validation_loss, self.center_pixel_distances, self.pixelwise_accuracy, self.path_dir)
        print('TRAINING COMPLETE')
        return self.training_loss, self.validation_loss, self.learning_rate, self.pixelwise_accuracy, self.center_pixel_distances

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        pixelwise_accuracy_within_batch = []
        center_pixel_distance_within_batch = []
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)

        for i, sample_batched in batch_iter:
            input, target, labels = sample_batched['image'].to(self.device), sample_batched['mask'].to(self.device), sample_batched['label'].to(self.device)  # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters
            out = self.model(input)  # one forward pass
            # out = out[:, np.newaxis, :, :, :]
            if i%30 == 0:
                save_sample_mask(self.epoch, i, out[0], sample_batched['mask'][0], self.path_dir)
            loss = self.criterion(out, target)  # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)
            pixelwise_accuracy_within_batch.append(get_pixel_accuracy_percent(out, target))
            center_pixel_distance_within_batch.append(get_central_pixel_distance(out, labels))
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar

        self.training_loss.append(np.mean(train_losses))
        self.pixelwise_accuracy.append(np.mean(pixelwise_accuracy_within_batch))
        self.center_pixel_distances.append(np.mean(center_pixel_distance_within_batch))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)

        for i, sample_batched in batch_iter:
            input, target = sample_batched['image'].to(self.device), sample_batched['mask'].to(self.device)   # send to device (GPU or CPU)

            
            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                valid_losses.append(loss_value)

                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')

        self.validation_loss.append(np.mean(valid_losses))

        batch_iter.close()

# %%
# trainer

trainer = TrainerUNET(model=model,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=train_dataloader,
                  validation_DataLoader=valid_dataloader,
                  lr_scheduler=None,
                  epochs=50,
                  epoch=0,
                  path_dir=PATH_DIR,
                  notebook=False)

# %%
# start training
training_losses, validation_losses, lr_rates, pixelwise_accuracy, center_pixel_distances = trainer.run_trainer()
print(list(training_losses))
print(list(validation_losses))
print(list(lr_rates))
print(list(pixelwise_accuracy))
print(list(center_pixel_distances))
