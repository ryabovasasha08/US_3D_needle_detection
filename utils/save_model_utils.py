import torch
import matplotlib.pyplot as plt
import numpy as np
from utils.mask_utils import get_center_of_nonzero_4d_slice

"""
This file contains utils for saving intermediate model weights + some sample results and metrics after each X epoch.
"""
# https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/

# CHeck if it indeed saves the best model or not
class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, path, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        self.path = path
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, self.path+'/best_model.pth')


def save_model(epochs, model, optimizer, criterion, train_loss, valid_loss, accuracy, precision, recall, center_distance, learning_rate, path):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'train_loss':train_loss,
                'valid_loss':valid_loss,
                'accuracy':accuracy,
                'precision':precision,
                'recall':recall,
                'center_distance':center_distance,
                'lr':learning_rate
                }, path)


# if save = False, this function just shows sample mask instead of saving it
# if save = True, this function just saves sample mask without displaying it
def save_sample_mask(epoch, batch, image, inp_mask, target_mask, path = "", save = True):
    target_mask_np = target_mask.detach().cpu().numpy()
    inp_mask_np = inp_mask.detach().cpu().numpy()
    image_np = image.detach().cpu().numpy()
    
    fig, axes = plt.subplots(3, 3, figsize=(12,12))

    # Get nonzero indices 
    x, y, z = get_center_of_nonzero_4d_slice(target_mask)
    x = int(x)
    y = int(y)
    z = int(z)

    axes[0, 2].imshow(inp_mask_np[0, x, :, :], cmap = 'gray', interpolation='none')
    axes[0, 2].set_title("OYZ - pred")
    
    axes[0, 1].imshow(target_mask_np[0, x, :, :], cmap = 'gray', interpolation='none')
    axes[0, 1].set_title("OYZ - gt")
    
    axes[0, 0].imshow(image_np[0, x, :, :], cmap = 'gray', interpolation='none')
    axes[0, 0].set_title("OYZ - image")
    
    axes[1, 2].imshow(inp_mask_np[0, :, y, :], cmap = 'gray', interpolation='none')
    axes[1, 2].set_title("OXZ - pred")
    
    axes[1, 1].imshow(target_mask_np[0, :, y, :], cmap = 'gray', interpolation='none')
    axes[1, 1].set_title("OXZ - gt")
    
    axes[1, 0].imshow(image_np[0, :, y, :], cmap = 'gray', interpolation='none')
    axes[1, 0].set_title("OXZ - image")
    
    axes[2, 2].imshow(inp_mask_np[0, :, :, z], cmap = 'gray', interpolation='none')
    axes[2, 2].set_title("OXY - pred")
    
    axes[2, 1].imshow(target_mask_np[0, :, :, z], cmap = 'gray', interpolation='none')
    axes[2, 1].set_title("OXY - gt")
    
    axes[2, 0].imshow(image_np[0, :, :, z], cmap = 'gray', interpolation='none')
    axes[2, 0].set_title("OXY - image")
    
    fig.tight_layout()
    
    plt.axis('off')
    
    if (save):
        plt.savefig(path+'/epoch_'+str(epoch)+'_batch_'+str(batch)+'.jpg')
    else:
        plt.show()
        
    plt.close()
    
    
# Function to save the loss plots to disk.   
def save_plots(epochs, train_loss, valid_loss, center_pixel_distances, pixelwise_accuracy, precision, recall, learning_rate, path):
    
    # accuracy plots
    plt.figure()
    plt.plot(np.arange(0, epochs), pixelwise_accuracy, color='green', linestyle='-')
    plt.title("Pixelwise accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy, %')
    plt.savefig(path+'/pixelwise_accuracy.jpg')
    plt.close()
    
    # precision plots
    plt.figure()
    plt.plot(np.arange(0, epochs), precision, color='green', linestyle='-')
    plt.title("Precision")
    plt.xlabel('Epochs')
    plt.ylabel('Precision, %')
    plt.savefig(path+'/precision.jpg')
    plt.close()
    
    # recall plots
    plt.figure()
    plt.plot(np.arange(0, epochs), recall, color='green', linestyle='-')
    plt.title("Recall over epochs")
    plt.xlabel('Epochs')
    plt.ylabel('Recall, %')
    plt.savefig(path+'/recall.jpg')
    plt.close()
    
    plt.figure()
    plt.plot(np.arange(0, epochs), center_pixel_distances, color='green', linestyle='-')
    plt.title("Distances between between actual and predicted tip position")
    plt.xlabel('Epochs')
    plt.ylabel('Distance, px')
    plt.savefig(path+'/center_pixel_distance.jpg')
    plt.close()
    
    plt.figure()
    plt.plot(np.arange(0, epochs), learning_rate, color='green', linestyle='-')
    plt.title("Learning rates")
    plt.xlabel('Epochs')
    plt.ylabel('learning rate')
    plt.savefig(path+'/learning_rate.jpg')
    plt.close()
    
    # loss plots
    plt.figure()
    plt.plot(np.arange(0, epochs), train_loss, color='green', linestyle='-', label='train loss')
    plt.plot(np.arange(0, epochs), valid_loss, color='red', linestyle='-', label='validation loss')
    plt.title("Losses over epochs")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path+'/loss.jpg')
    plt.close()