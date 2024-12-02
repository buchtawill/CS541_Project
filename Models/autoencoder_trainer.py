import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from spectrogram_dataset import SpectrogramDataset
import torch.optim as optim
from tqdm import tqdm
from autoencoder_conv import Autoencoder_FullyConv, Autoencoder_ConvLinear

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 1

TRAINED_MODEL_NAME = "trained_autoencoder_conv.pth"

def train_autoencoder(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    # Add progress bar for batches
    for batch in tqdm(train_loader, desc='Training', leave=False):
        # Add channel dimension and move to device
        batch = batch.unsqueeze(1).float().to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def main():
    print("device: ", DEVICE)

    # Initialize dataset and dataloader
    try:
        dataset = SpectrogramDataset('spectrogram_tensors.pt') 
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Initialize model, optimizer, and loss function
    model = Autoencoder_FullyConv().to(DEVICE)
    # model = Autoencoder_ConvLinear().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Training loop with progress bar
    print("Training...")
    for epoch in tqdm(range(NUM_EPOCHS), desc='Epochs'):
        avg_loss = train_autoencoder(model, train_loader, optimizer, criterion, DEVICE)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.6f}")

    # Save the trained model
    torch.save(model.state_dict(), 'trained_autoencoder_test_3.pth')

if __name__ == "__main__":
    main()
