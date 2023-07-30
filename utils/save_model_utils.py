import torch
import matplotlib.pyplot as plt
import numpy as np
from utils.mask_utils import get_center_of_nonzero_4d_slice

# https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/

# CHeck if it indeed saves the best model or not
class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
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
                }, 'outputs/best_model.pth')



def save_model(epochs, model, optimizer, criterion, train_loss, valid_loss, accuracy, center_distance, path):
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
                'center_distance':center_distance
                }, path)


# if save = False, this function just shows sample mask instead of saving it
# if save = True, this function just saves sample mask without displaying it
def save_sample_mask(epoch, batch, inp_mask, target_mask, save = True):
    target_mask_np = target_mask.detach().cpu().numpy()
    inp_mask_np = inp_mask.detach().cpu().numpy()
    
    plt.figure()
    # Get nonzero indices 
    x, y, z = get_center_of_nonzero_4d_slice(target_mask)

    plt.subplot(1, 3, 1)
    plt.title("OYZ")
    plt.imshow(target_mask_np[0, x, :, :], cmap='gray',  interpolation='none')
    plt.imshow(inp_mask_np[0, x, :, :], cmap='jet',  interpolation='none', alpha = 0.7)
    plt.subplot(1, 3, 2)
    plt.title("OXZ")
    plt.imshow(target_mask_np[0, :, y, :], cmap='gray',  interpolation='none')
    plt.imshow(inp_mask_np[0, :, y, :], cmap='jet',  interpolation='none', alpha = 0.7)
    plt.subplot(1, 3, 3)
    plt.title("OXY")
    plt.imshow(target_mask_np[0, :, :, z], cmap='gray',  interpolation='none')
    plt.imshow(inp_mask_np[0, :, :, z], cmap='jet',  interpolation='none', alpha = 0.7)
    
    plt.axis('off')
    
    if (save):
        plt.savefig('outputs/epoch_'+str(epoch)+'_batch_'+str(batch)+'.png')
    else:
        plt.show()
        
    plt.close()
    
    
# Function to save the loss plots to disk.   
def save_plots(epochs, train_loss, valid_loss, center_pixel_distances, pixelwise_accuracy):
    
    # accuracy plots
    plt.figure()
    plt.plot(np.arange(0, epochs), pixelwise_accuracy, color='green', linestyle='-')
    plt.title("Pixelwise accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy, %')
    plt.savefig('outputs/pixelwise_accuracy.jpg')
    plt.close()
    
    
    plt.figure()
    plt.plot(np.arange(0, epochs), center_pixel_distances, color='green', linestyle='-')
    plt.title("Distances between between actual and predicted tip position")
    plt.xlabel('Epochs')
    plt.ylabel('Distance, px')
    plt.savefig('outputs/center_pixel_distance.jpg')
    plt.close()
    
    # loss plots
    plt.figure()
    plt.plot(np.arange(0, epochs), train_loss, color='green', linestyle='-', label='train loss')
    plt.plot(np.arange(0, epochs), valid_loss, color='red', linestyle='-', label='validation loss')
    plt.title("Losses over epochs")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/loss.jpg')
    plt.close()