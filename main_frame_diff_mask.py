# %% [markdown]
# ## Prepare the datasets

# %% [markdown]
# <font size="2"> To install new library (since everything is running on kernel python3.10): 
# /$ python3.10 -m pip install *pandas*
# </font>

# %%
# import all files from necessary directory
from os import listdir, path
dir = "/data/Riabova/train/resized_train/"
all_files = [path.join(dir, f) for f in listdir(dir)]

# %%
import numpy as np
import h5py
# merge all images in one huge array

FRAME_DIFF = 2

all_frames_filenames_array = np.empty((1))
frame_nums = np.empty((1))
frame_nums_next = np.empty((1))
i = 0
for f in all_files:
    if (i%20 == 0):
        print("File "+str(i)+"...")
    h5_file = h5py.File(f, 'r')
    framesTotal = sum(item.startswith('img_') for item in list(h5_file.keys()))//11
    h5_file.close()
    all_frames_filenames_array = np.concatenate((all_frames_filenames_array, [f]*(framesTotal-FRAME_DIFF)), axis=0)
    frame_nums = np.concatenate((frame_nums, np.arange(0, framesTotal-FRAME_DIFF)), axis=0)
    frame_nums_next = np.concatenate((frame_nums_next, np.arange(FRAME_DIFF, framesTotal)), axis=0)
    i+=1
    
all_frames_filenames_array = all_frames_filenames_array[1:]
frame_nums = frame_nums[1:]
frame_nums_next = frame_nums_next[1:]

X = np.vstack((all_frames_filenames_array, frame_nums, frame_nums_next)).transpose()
print(X.shape)
print(X[10, :])

# %%
VALID_PERCENT = 0.2
BATCH_TRAIN = 10
BATCH_VALID = 10
BATCH_TEST = 10
# note: original size - 235; resizing to 200 + batch size 5 caused cuda out of memory

# %%
from sklearn.model_selection import train_test_split

# If splitting train+validation+test as 6:2:2,you get ???, so separating 300 images before to fix the test set
X_train, X_valid = train_test_split(X[300:, :], test_size=VALID_PERCENT, random_state=42, shuffle = True)
X_test = X[:300, :]

print(f"Total training images: {X_train.shape[0]}")
print(f"Total validation images: {X_valid.shape[0]}")
print(f"Total test images: {X_test.shape[0]}")

# %%
X_train.shape

# %%
from utils.datasets import FrameDiffDataset, my_collate
import torch

train_dataset = FrameDiffDataset(X_train)
valid_dataset = FrameDiffDataset(X_valid)
test_dataset = FrameDiffDataset(X_test)
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_TRAIN,shuffle=True, collate_fn=my_collate)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset,batch_size=BATCH_VALID,shuffle=True, collate_fn=my_collate)
test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_TEST,shuffle=True, collate_fn=my_collate)

# %% [markdown]
# ## u-net

# %%
# init number of epochs to train for, and the
# batch size of train and validation sets
EPOCHS = 200
UNET_DEPTH = 4 # size of the image should divide by this number
UNET_START_FILTERS = 3

#For LR scheduler
INIT_LR = 0.01
WEIGHT_DECAY = 1e-8
MOMENTUM = 0.999

# define threshold to filter weak predictions
THRESHOLD = 0.5

PATH_DIR = 'outputs/outputs_new_diff_mask_2'

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
from utils.losses import IoULoss
from utils.save_model_utils import SaveBestModel

# state_epoch_20 = torch.load('outputs_side_128_epochs_0_50/epoch_20_model.pth')
model = UNet(out_channels = 1, n_blocks=UNET_DEPTH, start_filts = UNET_START_FILTERS)
optimizer = optim.Adam(model.parameters(), lr = INIT_LR)# optim.RMSprop(model.parameters(),lr=INIT_LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM, foreach=True)
scheduler = None # optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1)  # goal: minimize loss
criterion = IoULoss()
save_best_model = SaveBestModel(PATH_DIR)

#model.to(device).load_state_dict(state_epoch_20['model_state_dict'])
#optimizer.load_state_dict(state_epoddch_20['optimizer_state_dict'])


# %%
model

# %%
import numpy as np
import torch
from utils.save_model_utils import save_model, save_plots, save_sample_mask
from utils.accuracies import get_central_pixel_distance, get_pixel_accuracy_percent, get_precision, get_recall

class TrainerUNET:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 test_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False,
                 path_dir = ''
                 ):

        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.test_DataLoader = test_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook
        self.path_dir = path_dir
        
        self.training_loss = []
        self.validation_loss = []
        self.tip_pixel_distances = []
        self.pixelwise_accuracy = []
        self.precisions = []
        self.recalls = []
        self.learning_rate = []
        
    def run_trainer(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epoch, self.epochs, desc='Progress')
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
            if self.epoch % 10 == 0:    
                save_model(self.epochs, self.model, self.optimizer, self.criterion, self.training_loss[-1], self.validation_loss[-1], self.pixelwise_accuracy[-1], self.precisions[-1], self.recalls[-1], self.tip_pixel_distances[-1], self.learning_rate[-1], self.path_dir+'/epoch_'+str(self.epoch)+'_model.pth')
            save_plots(self.epoch, self.training_loss, self.validation_loss, self.tip_pixel_distances, self.pixelwise_accuracy, self.precisions, self.recalls, self.learning_rate, self.path_dir)

            print('-'*50)
            
            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(self.validation_loss[i])  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.batch()  # learning rate scheduler step
        
        # save the trained model weights for a final time
        save_model(self.epochs, self.model, self.optimizer, self.criterion, self.training_loss[-1], self.validation_loss[-1], self.pixelwise_accuracy[-1], self.precisions[-1], self.recalls[-1], self.tip_pixel_distances[-1], self.learning_rate[-1],  self.path_dir+'/final_model.pth')
        # save the loss and accuracy plots
        save_plots(self.epochs, self.training_loss, self.validation_loss, self.tip_pixel_distances, self.pixelwise_accuracy, self.precisions, self.recalls, self.learning_rate, self.path_dir)
        print('TRAINING COMPLETE')
        
        new_file = h5py.File(self.path_dir+'/train_test_data.hdf5', 'w')
        new_file.create_dataset("training_loss", data=self.training_loss)
        new_file.create_dataset("valid_loss", data=self.validation_loss)
        new_file.create_dataset("tr_tip_pixel_distance", data=self.tip_pixel_distances)
        new_file.create_dataset("tr_pixelwise_accuracy", data=self.pixelwise_accuracy)
        new_file.create_dataset("tr_precisions", data=self.precisions)
        new_file.create_dataset("tr_recalls", data=self.recalls)
        new_file.create_dataset("tr_lr", data=self.learning_rate)

        new_file.close()
        
        if self.test_DataLoader is not None:
            self._test()
            print('TESTING COMPLETE')

        
        return self.training_loss, self.validation_loss, self.learning_rate, self.pixelwise_accuracy, self.tip_pixel_distances

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        pixelwise_accuracy_within_batch = []
        precision_within_batch = []
        recall_within_batch = []
        tip_pixel_distance_within_batch = []
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)

        for i, sample_batched in batch_iter:
            input, target, labels = sample_batched['image'].to(self.device), sample_batched['mask'].to(self.device), sample_batched['label'].to(self.device)  # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters
            #out = binarize_with_softmax(self.model(input), dimToSqueeze=1)  # one forward pass
            out = self.model(input)
            
            target.requires_grad_(True)
            #if i%30 == 0:
            #    save_sample_mask(self.epoch, i, input[0], out[0], sample_batched['mask'][0], path = self.path_dir)        
        
            loss = self.criterion(out, target)  # calculate loss
            loss_value = loss.item()
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters
                        
            train_losses.append(loss_value)
            pixelwise_accuracy_within_batch.append(get_pixel_accuracy_percent(out, target))
            precision_within_batch.append(get_precision(out, target))
            recall_within_batch.append(get_recall(out, target))
            tip_pixel_distance_within_batch.append(get_central_pixel_distance(out, labels))

            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar
        
        if self.epoch % 10 == 0: 
            save_sample_mask(self.epoch, i, input[0], out[0], sample_batched['mask'][0], path = self.path_dir)

        self.training_loss.append(np.mean(train_losses))
        self.pixelwise_accuracy.append(np.mean(pixelwise_accuracy_within_batch))
        self.precisions.append(np.mean(precision_within_batch))
        self.recalls.append(np.mean(recall_within_batch))
        self.tip_pixel_distances.append(np.mean(tip_pixel_distance_within_batch))
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
                # out = binarize_with_softmax(self.model(input), dimToSqueeze=1) 
                out = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                valid_losses.append(loss_value)

                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')

        self.validation_loss.append(np.mean(valid_losses))

        batch_iter.close()
        
    def _test(self):
        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        test_losses = []  # accumulate the losses here
        pixelwise_accuracies = []
        precisions = []
        recalls = []
        tip_pixel_distances = []
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Test', total=len(self.validation_DataLoader),
                          leave=False)

        for i, sample_batched in batch_iter:
            input, target, labels = sample_batched['image'].to(self.device), sample_batched['mask'].to(self.device), sample_batched['label'].to(self.device)  # send to device (GPU or CPU)

            with torch.no_grad():
                # out = binarize_with_softmax(self.model(input), dimToSqueeze=1) 
                out = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                test_losses.append(loss_value)
                pixelwise_accuracies.append(get_pixel_accuracy_percent(out, target))
                precisions.append(get_precision(out, target))
                recalls.append(get_recall(out, target))
                tip_pixel_distances.append(get_central_pixel_distance(out, labels))
            
                batch_iter.set_description(f'Test: (loss {loss_value:.4f})')
                
        new_file = h5py.File(self.path_dir+'/train_test_data.hdf5', 'a')
        new_file.create_dataset("test_loss", data=np.mean(test_losses))
        new_file.create_dataset("test_tip_pixel_distance", data=np.mean(tip_pixel_distances))
        new_file.create_dataset("test_pixelwise_accuracy", data=np.mean(pixelwise_accuracies))
        new_file.create_dataset("test_precision", data=np.mean(precisions))
        new_file.create_dataset("test_recall", data=np.mean(recalls))
        new_file.close()

        batch_iter.close()

# %%
# trainer

trainer = TrainerUNET(model=model,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=train_dataloader,
                  validation_DataLoader=valid_dataloader,
                  test_DataLoader=test_dataloader,
                  lr_scheduler=scheduler,
                  epochs=EPOCHS,
                  epoch=0,
                  notebook=False,
                  path_dir = PATH_DIR)

# %%
# start training
training_losses, validation_losses, lr_rates, pixelwise_accuracy, tip_pixel_distances = trainer.run_trainer()