import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from spectrogram_dataset import SpectrogramDataset
import torch.optim as optim
from tqdm import tqdm
from autoencoder_conv import Autoencoder_ConvLinear, Autoencoder_FullyConv
from autoencoder_masked_attn import AutoencoderTransformer
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 500
EVALUATE_FREQ = 10

TRAINED_MODEL_NAME = "autoencoder_transformer_basesize.pth"
TENSORBOARD_LOG_DIR = f'runs/{TRAINED_MODEL_NAME}'

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
        
        # writer.add_scalar('Loss/batch', loss.item(), epoch * len(train_loader) + batch_idx)
        
        # if batch_idx % 100 == 0:
        #     writer.add_images('Original', batch[:4], epoch * len(train_loader) + batch_idx)
        #     writer.add_images('Reconstructed', outputs[:4], epoch * len(train_loader) + batch_idx)
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def evaluate_autoencoder(model, test_loader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing', leave=False):
            batch = batch.unsqueeze(1).float().to(device)
            outputs = model(batch)
            loss = criterion(outputs, batch)
            total_loss += loss.item()
            
    avg_loss = total_loss / len(test_loader)
    return avg_loss

def main():
    print("device: ", DEVICE)
    current_epoch = 0
    model = None
    writer = None

    try:
        # Initialize datasets
        try:
            train_dataset = SpectrogramDataset('spec_tens_512hop_128mel_x.pt', split='train')
            # test_dataset = SpectrogramDataset('spec_tens_512hop_128mel_x.pt', split='test')
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
            
            print(f"Train dataset size: {len(train_dataset)} samples")
            # print(f"Test dataset size: {len(test_dataset)} samples")
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            return

        # Initialize model, optimizer, and loss function
        model = AutoencoderTransformer().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()

        # Initialize TensorBoard writer
        writer = SummaryWriter(TENSORBOARD_LOG_DIR)

        # Training loop
        print("Training...")
        for epoch in tqdm(range(NUM_EPOCHS), desc='Epochs'):
            current_epoch = epoch + 1  
            train_loss = train_autoencoder(model, train_loader, optimizer, criterion, DEVICE)
            print(f"Epoch [{current_epoch}/{NUM_EPOCHS}], Train Loss: {train_loss:.6f}")
            writer.add_scalar('Loss/train', train_loss, epoch)
            
            # if (epoch + 1) % EVALUATE_FREQ == 0:
            #     test_loss = evaluate_autoencoder(model, test_loader, criterion, DEVICE)
            #     print(f"Epoch [{current_epoch}/{NUM_EPOCHS}], Test Loss: {test_loss:.6f}")
            #     writer.add_scalar('Loss/test', test_loss, epoch)

        # Save the trained model
        torch.save(model.state_dict(), TRAINED_MODEL_NAME)
        writer.close()

    except KeyboardInterrupt:
        print(f'\nTraining interrupted by user at epoch {current_epoch}.')
        if model is not None:
            early_save_name = TRAINED_MODEL_NAME.replace('.pth', f'_terminated_epoch_{current_epoch}.pth')
            torch.save(model.state_dict(), early_save_name)
            print(f'Model saved as {early_save_name}')
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()