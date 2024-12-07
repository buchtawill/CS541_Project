import sys
import time
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from autoencoder_conv import *
from spectrogram_dataset import SpectrogramDataset

# import torchinfo

NUM_EPOCHS = 100
BATCH_SIZE = 4
LEARN_RATE = 5e-4

def model_dataloader_inference(model, dataloader, device, criterion, optimzer):
    """
    Run the forward pass of model on all samples in dataloader with criterion loss. If optimizer is set to None,
    this function will NOT perform gradient updates or optimizations.
    Args:
        model(nn.Module): The neural network model
        dataloader(torch.utils.data.DataLoader): PyTorch dataloader 
        criterion(): Loss criterion (e.g. MSE loss)
        optimizer(torch.optim): Optimizer for NN
    """
    running_loss = 0.0
    for batch in dataloader:
        
        optimizer.zero_grad()
        
        batch = batch.to(device)
        
        inference = model(batch)
        loss = criterion(inference, batch)
        
        # print(f"INFO [model_dataloader_inference()] batch shape:     {batch.shape}")
        # print(f"INFO [model_dataloader_inference()] inference shape: {inference.shape}")
        
        if(optimzer is not None):
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        # print(f"INFO LOSS ITEM: {loss.item() / len(batch)}")
    loss = running_loss / float(len(dataloader.dataset))
    return loss


def train_normal(model, 
                 train_dataloader:torch.utils.data.DataLoader, 
                 test_dataloader:torch.utils.data.DataLoader,
                 optimizer:torch.optim, 
                 tb_writer:torch.utils.tensorboard.SummaryWriter, 
                 scheduler:torch.optim.lr_scheduler.StepLR, 
                 criterion, 
                 device):
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0

        # Set optimizer to None to run in 
        model.train()
        train_loss = model_dataloader_inference(model=model, dataloader=train_dataloader, device=device, criterion=criterion, optimzer=optimizer)
        model.eval()
        test_loss  = model_dataloader_inference(model=model, dataloader=test_dataloader, device=device, criterion=criterion, optimzer=None)
        
        tb_writer.add_scalar("Loss/train", train_loss, epoch + 1)
        tb_writer.add_scalar("Loss/test",  test_loss,  epoch + 1)
        
        print(f'Epoch {epoch:>{6}} | Train loss: {train_loss:.8f} | Test Loss: {test_loss:.8f}', flush=True)
        
        # Step the scheduler to adjust the learning rate
        scheduler.step()

        # Log the current learning rate to TensorBoard
        current_lr = scheduler.get_last_lr()[0]
        tb_writer.add_scalar("Learning Rate", current_lr, epoch + 1)
            

def sec_to_human(seconds):
    """Return a number of seconds to hours, minutes, and seconds"""
    seconds = seconds % (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hours, minutes, seconds)

if __name__ == '__main__':
    tstart = time.time()
    print(f"INFO [train_2.py] Starting script at {tstart}")
    
    #Set up device, model, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'INFO [train_2.py] Using device: {device} [torch version: {torch.__version__}]')
    print(f'INFO [train_2.py] Python version: {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')
    # model = AutoencoderLargeKernels().to(device)
    model = Autoencoder_ConvLinear().to(device)
    # model.load_state_dict(torch.load('./saved_weights/100E_5em4_b64.pth', weights_only=True))
    
    # torchinfo.summary(model, input_size=(4, 1, 128, 1290))
    # exit()
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
    
    # Get dataset
    seed = 50  # Set the seed for reproducibility
    torch.manual_seed(seed)
    print("INFO [train_2.py] Loading Tensor dataset")
    full_dataset = SpectrogramDataset('./data/normal_128m_512h_x_medium.pt')
    
    # Create train and test datasets. Set small train set for faster training

    train_dataset, valid_dataset, test_dataset = \
            torch.utils.data.random_split(full_dataset, [0.85, 0.10, 0.05], generator=torch.Generator())
    num_train_samples = len(train_dataset)
    print(f'INFO [train_2.py] Total num data samples:    {len(full_dataset)}')
    print(f'INFO [train_2.py] Num of training samples:   {num_train_samples}')
    print(f'INFO [train_2.py] Num of validation samples: {len(valid_dataset)}')
    print(f'INFO [train_2.py] Num of test samples:       {len(test_dataset)}')
    
    # Get Dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader  = torch.utils.data.DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=True)
    print(f'INFO [train_2.py] Num training batches:      {len(train_dataloader)}', flush = True)
    scheduler = StepLR(optimizer=optimizer, step_size=10, gamma=0.5)
    tb_writer = SummaryWriter()
    
    train_normal(model=model, 
                 train_dataloader=train_dataloader, 
                 test_dataloader=test_dataloader,
                 optimizer=optimizer, 
                 tb_writer=tb_writer, 
                 scheduler=scheduler, 
                 criterion=criterion, 
                 device=device)
                
    tb_writer.flush()
    torch.save(model.state_dict(), './conv_linear_weights.pth')
    
    tEnd = time.time()
    print(f"INFO [train_2.py] Ending script. Took {tEnd-tstart:.2f} seconds.")
    print(f"INFO [train_2.py] HH:MM:SS --> {sec_to_human(tEnd-tstart)}")