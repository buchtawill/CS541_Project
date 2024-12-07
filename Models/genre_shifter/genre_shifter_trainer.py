import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from spectrogram_dataset import SpectrogramDataset
from tqdm import tqdm
from genre_shifter_fully_connected import Genre_Shifter_Fully_Connected
from autoencoder_conv import Autoencoder_FullyConv, Autoencoder_ConvLinear
from Models.classifier import GenreClassifier
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

def genre_shifter_loss(outputs, batch, autoencoder_model_path, classifier_model_path, alpha):
    """
        Calculate the loss for the genre shifter model using a combination of autoencoder reconstruction
        loss and genre classification loss.

        The function balances two objectives:
        1. Maintaining similarity to the original audio (via autoencoder)
        2. Changing the perceived genre (via genre classifier)

        Args:
            outputs (torch.Tensor): Output spectrograms from the genre shifter model
                                  Shape: (batch_size, channels, freq_bins, time_frames)
            batch (torch.Tensor): Original input spectrograms and their genre labels
                                 Shape matches outputs for spectrogram data
            autoencoder_model_path (str): Path to the pretrained autoencoder model weights
            classifier_model_path (str): Path to the pretrained genre classifier model weights
            alpha (float): Weight factor between similarity and classification loss
                          Range [0,1] where:
                          - alpha = 0: Focus entirely on changing genre
                          - alpha = 1: Focus entirely on maintaining similarity

        Returns:
            torch.Tensor: Combined loss value balancing similarity preservation and genre modification

        Notes:
            - Processes spectrograms in 3-second segments (129 frames) for genre classification
            - Uses pretrained models for both autoencoder and genre classifier
            - Genre classifier expects 7 genre classes
            - All tensor operations are performed with gradient calculation disabled
        """
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
    Genre_Classifier_NN = GenreClassifier(input_shape=(1, 128, 129), num_genres=7)
    Genre_Classifier_NN.load_state_dict(genre_classifier_NN_state_dict)
    Genre_Classifier_NN.eval() # Set to evaluate stuff for loss of genre shifter

    # Calculate loss
    with torch.no_grad():
        autoencoder_classifier_outputs = Autoencoder_NN(outputs)

        # Extract dimensions from outputs
        batch_size, channels, freq_bins, time_frames = outputs.shape
        segment_length = 129  # 3 seconds of frames as required by GenreClassifier

        num_segments = time_frames // segment_length

        # Split into 3-second segments using unfold operation
        segments = outputs.unfold(3, segment_length, segment_length)
        # Adjust dimensions to match expected input format
        segments = segments.transpose(1, 2)
        # Reshape into format expected by classifier: (batch * segments, channels, freq_bins, segment_length)
        segments = segments.reshape(-1, 1, freq_bins, segment_length)

        genre_classifier_outputs = Genre_Classifier_NN(segments)


        genre_classifier_outputs = genre_classifier_outputs.reshape(batch_size, num_segments, -1)
        genre_classifier_outputs = torch.mean(genre_classifier_outputs, dim=1)

    similarity_loss = nn.MSELoss()(autoencoder_classifier_outputs, batch)

    # Convert logits to probabilities
    genre_probs = nn.functional.softmax(genre_classifier_outputs, dim=1)
    # Extract probability of the target genre for each sample
    target_probs = torch.gather(genre_probs, 1, batch.unsqueeze(1))
    # Classification loss is mean probability of original genre (want to minimize)
    classification_loss = torch.mean(target_probs)

    overall_loss = (alpha * similarity_loss) + (1 - alpha) * classification_loss
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
    model = Genre_Shifter_Fully_Connected().to(DEVICE)
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
