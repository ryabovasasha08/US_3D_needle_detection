import torch
import matplotlib.pyplot as plt

# https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/

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



def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'outputs/final_model.pth')


def save_sample_mask(epoch, batch, inp_mask, target_mask):
    target_mask_np = target_mask.detach().cpu().numpy()
    inp_mask_np = inp_mask.detach().cpu().numpy()
    
    plt.figure()
    # Get nonzero indices 
    nz_indices = torch.nonzero(target_mask) 
    mid = len(nz_indices) // 2
    extr, x, y, z = nz_indices[mid]

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
    plt.savefig('outputs/epoch_'+str(epoch)+'_batch_'+str(batch)+'.png')
    
    
    
def save_plots(
    # train_acc, valid_acc, 
    train_loss, valid_loss):
    """
    Function to save the loss plots to disk.
    """
    
    '''
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('outputs/accuracy.png')
    '''
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='green', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validation loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/loss.png')