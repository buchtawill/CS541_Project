import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Models.autoencoder_conv import Autoencoder_FullyConv, Autoencoder_ConvLinear
from Models.classifier import GenreClassifier
from Models.spectrogram_dataset import SpectrogramDataset
from Models.genre_shifter.genre_shifter_models import Genre_Shifter_Fully_Connected
from Models.genre_shifter.genre_shifter_dataset import GenreShifterDataset

from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 8
ALPHA = 0.5 # Alpha for custom loss function

TRAINED_MODEL_NAME = "./genre_shifter_models/paper/alpha_0.5_v2"
TENSORBOARD_LOG_DIR = "runs/genre_shifter_models/paper/alpha_0.5_v2"
 
AUTOENCODER_NN_PATH = r"Models\data\models\best\conv_linear_1em4_100_12_8_h100.pth"
GENRE_CLASSIFIER_NN_PATH = r"Models\data\models\best\genre_classifier_v3_small.pth"

# Trained on SMALL DATASET
DATASET_X = r"Models\data\test_dataset\normal_128m_512h_x_small.pt"
DATASET_Y = r"Models\data\test_dataset\normal_128m_512h_y_small.pt"

# TODO change to 7 for 7 genres
GENRE_OUTPUT = torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float32).reshape(1, 6) # Output same for every input of training set

def split_clips_genre_classifier_output(genre_input, Genre_Classifier_NN):
    # Calculate loss TODO fix genre classifier portion of this
    with torch.no_grad():
        # Extract dimensions from outputs
        genre_batch_size, channels, freq_bins, time_frames = genre_input.shape # NOTE: outputs is genre_outputs
        segment_length = 129  # 3 seconds of frames as required by GenreClassifier

        num_segments = time_frames // segment_length

        # Split into 3-second segments using unfold operation
        segments = genre_input.unfold(3, segment_length, segment_length)
        # Adjust dimensions to match expected input format
        segments = segments.transpose(1, 2)
        # Reshape into format expected by classifier: (batch * segments, channels, freq_bins, segment_length)
        segments = segments.reshape(-1, 1, freq_bins, segment_length)

        genre_classifier_outputs = Genre_Classifier_NN(segments)
        genre_classifier_outputs = genre_classifier_outputs.reshape(genre_batch_size, num_segments, -1)
        genre_classifier_outputs = torch.mean(genre_classifier_outputs, dim=1)
    
    return genre_classifier_outputs

def get_loss(mel_input, genre_input, genre_output, model, Autoencoder_NN, Genre_Classifier_NN):
    # Get encoded output for model
    #print("TEST AE OUT ", Autoencoder_NN.encoder(mel_input).shape)
    shifted_encoder_output = Autoencoder_NN.encoder(mel_input)
    # TODO add one hot encoding to this
    
    combined_tensor = torch.cat((shifted_encoder_output, genre_input, genre_output), dim=1)
    genre_shifter_output = model(combined_tensor) # Run encoder output into genre shifter
    shifted_decoder_output = Autoencoder_NN.decoder(genre_shifter_output)
    shifted_decoder_output =  shifted_decoder_output[:, :, :, :1290] 
    #print("SHIFTED DECODER OUTPUT", shifted_decoder_output.shape)

    classification_means = split_clips_genre_classifier_output(shifted_decoder_output, Genre_Classifier_NN)
    
    mse_loss = nn.MSELoss()
    similarity_loss = mse_loss(mel_input, shifted_decoder_output) # From keeping song similar

    ce_loss = nn.CrossEntropyLoss()
    classification_loss = ce_loss(classification_means, genre_output) 

    loss = (ALPHA * similarity_loss) + (1 - ALPHA) * classification_loss
    return loss

def train_genre_shifter(model, Autoencoder_NN, Genre_Classifier_NN, train_loader, optimizer, device, writer, epoch):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc='Training', leave=False)):
        mel_input = batch[0].float().to(device)
        genre_input = batch[1].float().to(device) # TODO remove [:-1] for 7
        genre_input = genre_input[:, :-1] #  Remove this for 7 genres
 
        genre_output = GENRE_OUTPUT.repeat(genre_input.shape[0], 1).to(DEVICE) # Copy it over this batch size
        optimizer.zero_grad()

        loss = get_loss(mel_input, genre_input, genre_output, model, Autoencoder_NN, Genre_Classifier_NN)
        loss.backward()
        optimizer.step()
        
        writer.add_scalar('Train Loss/batch', loss, batch_idx, epoch)
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def test_genre_shifter(model, Autoencoder_NN, Genre_Classifier_NN, test_loader, device):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(test_loader, desc='Testing', leave=False)):
        mel_input = batch[0].float().to(device)
        genre_input = batch[1].float().to(device) 
        genre_input = genre_input[:, :-1] # TODO Remove this for 7 genres

        genre_output = GENRE_OUTPUT.repeat(genre_input.shape[0], 1).to(DEVICE) # Copy it over this batch size
        loss = get_loss(mel_input, genre_input, genre_output, model, Autoencoder_NN, Genre_Classifier_NN)
        
        total_loss += loss.item()
            
    avg_loss = total_loss / len(test_loader)
    return avg_loss

def main():
    print("device: ", DEVICE)

    # Initialize dataset and dataloader
    try:
        genre_data = GenreShifterDataset(DATASET_X, DATASET_Y) # TODO add training, testing, validation splits
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    total_size = len(genre_data)
    train_size = int(0.95 * total_size)
    test_size = total_size - train_size

    train_dataset, test_dataset = random_split(
        genre_data,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(50)
    )  
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, optimizer, and loss function
    model = Genre_Shifter_Fully_Connected().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Get autoencoder model
    autoencoder_NN_path = AUTOENCODER_NN_PATH
    autoencoder_NN_state_dict = torch.load(autoencoder_NN_path, weights_only=True)
    Autoencoder_NN = Autoencoder_ConvLinear()
    Autoencoder_NN.load_state_dict(autoencoder_NN_state_dict)
    Autoencoder_NN.to(DEVICE)
    Autoencoder_NN.eval() # Set to evaluate stuff for loss of genre shifter

    # Get genre classifier model
    genre_classifier_NN_path = GENRE_CLASSIFIER_NN_PATH
    genre_classifier_NN_state_dict = torch.load(genre_classifier_NN_path, weights_only=True)
    Genre_Classifier_NN = GenreClassifier(input_shape=(1, 128, 129), num_genres=6) # TODO set to 7 for 7 genres
    Genre_Classifier_NN.load_state_dict(genre_classifier_NN_state_dict)
    Genre_Classifier_NN.to(DEVICE)
    Genre_Classifier_NN.eval() # Set to evaluate stuff for loss of genre shifter

    # Initialize TensorBoard writer
    writer = SummaryWriter(TENSORBOARD_LOG_DIR)

    # Training loop with progress bar
    print("Training...")
    for epoch in tqdm(range(NUM_EPOCHS), desc='Epochs'):
        avg_loss = train_genre_shifter(model, Autoencoder_NN, Genre_Classifier_NN, train_loader, optimizer, DEVICE, writer, epoch)
        
        writer.add_scalar('Train Loss/epoch', avg_loss, epoch)
        test_avg_loss = test_genre_shifter(model, Autoencoder_NN, Genre_Classifier_NN, test_loader, DEVICE)
        writer.add_scalar('Test Loss/epoch', test_avg_loss, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_loss:.6f}, Test Loss: {test_avg_loss:.6f}")

    # Save the trained model
    torch.save(model.state_dict(), TRAINED_MODEL_NAME)
    writer.close()

if __name__ == "__main__":
    main()
