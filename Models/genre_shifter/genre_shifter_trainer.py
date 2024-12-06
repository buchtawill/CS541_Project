import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from spectrogram_dataset import SpectrogramDataset
from tqdm import tqdm
from genre_shifter_fully_connected import Genre_Shifter_Fully_Connected
from autoencoder_conv import Autoencoder_FullyConv, Autoencoder_ConvLinear

from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
NUM_EPOCHS = 500
ALPHA = 0.7 # Alpha for custom loss function

TRAINED_MODEL_NAME = "genreshifter_test_1"
TENSORBOARD_LOG_DIR = "runs/genreshifter_test_1"

AUTOENCODER_NN_PATH = "ae_samplepath/location.pth"
GENRE_CLASSIFIER_NN_PATH = "gc_samplepath/location.pth"

# TODO add documentation, but used as a criteron, with outputs and batch as the main inputs, the rest can be
# curried into a lambda function when used. alpha=0 means seeks to trick classifier, alpha=1 seeks to make no changes (identity matrix optimal then).
def genre_shifter_loss(outputs, batch, autoencoder_model_path, classifier_model_path, alpha):
    # Load models needed for genre shifter
    # TODO would prefer in future if models had architecture saved too so they didn't need to be "applied" since they may have different shapes
   
    # Get autoencoder model
    autoencoder_NN_path = autoencoder_model_path
    autoencoder_NN_state_dict = torch.load(autoencoder_NN_path, weights_only=True)
    Autoencoder_NN = Autoencoder_ConvLinear()
    Autoencoder_NN.load_state_dict(autoencoder_NN_state_dict)
    Autoencoder_NN.eval() # Set to evaluate stuff for loss of genre shifter

    # Get genre classifier model
    genre_classifier_NN_path = classifier_model_path
    genre_classifier_NN_state_dict = torch.load(genre_classifier_NN_path, weights_only=True)
    Genre_Classifier_NN = Genre_Classifier() # TODO need model to load
    Genre_Classifier_NN.load_state_dict(genre_classifier_NN_state_dict)
    Genre_Classifier_NN.eval() # Set to evaluate stuff for loss of genre shifter

    # Note that outputs are the outputs of the genre shifter only, need to load other outputs from fixed models
    with torch.no_grad():
        autoencoder_classifier_outputs = Autoencoder_NN(outputs)
        # TODO outputs need to be "chopped" into 3 second clips for Genre_Classifier (should probably break this out into it's own section)
        genre_classifier_outputs = Genre_Classifier_NN(outputs) 

    similarity_loss = nn.MSELoss(autoencoder_classifier_outputs, batch) 
    classification_loss = nn.MSELoss(genre_classifier_outputs, batch) 

    # Balance keeping the song similar and tricking the classifier into a new genre with alpha
    overall_loss = (alpha * similarity_loss) + (1 - alpha) * (classification_loss)
    return overall_loss

def train_genre_shifter(model, train_loader, optimizer, criterion, device, writer, epoch):
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
    model = Genre_Shifter_FullyConnected().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = lambda outputs, batch: genre_shifter_loss(outputs, batch, 
                                                          AUTOENCODER_NN_PATH, GENRE_CLASSIFIER_NN_PATH,
                                                          ALPHA)

    # Initialize TensorBoard writer
    writer = SummaryWriter(TENSORBOARD_LOG_DIR)

    # Training loop with progress bar
    print("Training...")
    for epoch in tqdm(range(NUM_EPOCHS), desc='Epochs'):
        avg_loss = train_genre_shifter(model, train_loader, optimizer, criterion, DEVICE, writer, epoch)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.6f}")
        
        writer.add_scalar('Loss/epoch', avg_loss, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

    # Save the trained model
    torch.save(model.state_dict(), TRAINED_MODEL_NAME)
    writer.close()

if __name__ == "__main__":
    main()
