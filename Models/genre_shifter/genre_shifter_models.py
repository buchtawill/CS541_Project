import torch
import torch.nn as nn
import Models.spectrogram_dataset as spec_data

# Genre Shifter Hyperparameters (reasonably simple)
NUM_GENRES = 7 # Will be one hot encoded as input
LATENT_SIZE  = 1024  # Channels for each encoder layer, keep same size for simplicity
NUM_LAYERS = 5

# Fully convolutional autoencoder (latent space shape: [batch_size, 256, 16, 162])
class Genre_Shifter_Fully_Connected(nn.Module):
    def __init__(self):
        super(Genre_Shifter_Fully_Connected, self).__init__()
        
        # Fully Connected Layers (from 2048 latent representation of song)
        fcc_layers = []
        input_layer_size = LATENT_SIZE + 2*NUM_GENRES # Calculate layer size with one hot (x2 because of input genre and output genre encoding)

        # First layer to get down to size of output without genre info
        fcc_layers.extend([
                nn.Linear(input_layer_size, LATENT_SIZE),
                nn.ReLU()
        ])
        for i in range(NUM_LAYERS-1):
            fcc_layers.extend([
                nn.Linear(LATENT_SIZE, LATENT_SIZE),
                nn.ReLU() # TODO 0.2 was a somewhat large leaky ReLU valus
            ])
        
        self.layers = nn.Sequential(*fcc_layers)
        
    def forward(self, x):
        x = self.layers(x) 
        return x
