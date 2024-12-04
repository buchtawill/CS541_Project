import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from spectrogram_dataset import SpectrogramDataset
import torch.optim as optim
from tqdm import tqdm
from autoencoder_conv import Autoencoder_FullyConv
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
NUM_EPOCHS = 500

TRAINED_MODEL_NAME = "trained_autoencoder_conv_3.pth"
TENSORBOARD_LOG_DIR = 'runs/autoencoder_fully_conv_3'

def train_autoencoder(model, train_loader, optimizer, criterion, device, writer, epoch):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc='Training', leave=False)):
        batch = batch.unsqueeze(1).float().to(device)
        
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        writer.add_scalar('Loss/batch', loss.item(), epoch * len(train_loader) + batch_idx)
        
        if batch_idx % 100 == 0:
            writer.add_images('Original', batch[:4], epoch * len(train_loader) + batch_idx)
            writer.add_images('Reconstructed', outputs[:4], epoch * len(train_loader) + batch_idx)
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def main():
    print("device: ", DEVICE)

    # Initialize dataset and dataloader
    try:
        dataset = SpectrogramDataset('spec_tens_512hop_128mel_x.pt') 
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Initialize model, optimizer, and loss function
    model = Autoencoder_FullyConv().to(DEVICE)
    # model = Autoencoder_ConvLinear().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Initialize TensorBoard writer
    writer = SummaryWriter(TENSORBOARD_LOG_DIR)
    
    # Log model graph
    dummy_input = torch.randn(1, 1, 128, 1290).to(DEVICE)  # Adjust dimensions as needed
    writer.add_graph(model, dummy_input)

    # Training loop with progress bar
    print("Training...")
    for epoch in tqdm(range(NUM_EPOCHS), desc='Epochs'):
        avg_loss = train_autoencoder(model, train_loader, optimizer, criterion, DEVICE, writer, epoch)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.6f}")
        
        writer.add_scalar('Loss/epoch', avg_loss, epoch)
        
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

    # Save the trained model
    torch.save(model.state_dict(), TRAINED_MODEL_NAME)
    writer.close()

if __name__ == "__main__":
    main()
