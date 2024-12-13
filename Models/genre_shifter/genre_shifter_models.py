import torch
import torch.nn as nn
import Models.spectrogram_dataset as spec_data

# Genre Shifter Hyperparameters (reasonably simple)
NUM_GENRES = 6 # Will be one hot encoded as input
LATENT_SIZE  = 4096  # Channels for each encoder layer, keep same size for simplicity
NUM_LAYERS = 1

# Fully convolutional autoencoder (latent space shape: [batch_size, 256, 16, 162])
class Genre_Shifter_Fully_Connected(nn.Module):
    def __init__(self):
        super(Genre_Shifter_Fully_Connected, self).__init__()
        
        # Fully Connected Layers (from 2048 latent representation of song)
        fcc_layers = []
        input_layer_size = LATENT_SIZE + 2*NUM_GENRES # Calculate layer size with one hot (x2 because of input genre and output genre encoding)

        # First layer to get down to size of output without genre info
        first_lin_layer = nn.Linear(input_layer_size, LATENT_SIZE)
        # Set to identitiy initially since model makes small changes
        nn.init.eye_(first_lin_layer.weight)
        nn.init.zeros_(first_lin_layer.bias)
        fcc_layers.extend([
                first_lin_layer,
                nn.ReLU()
        ])
        for i in range(NUM_LAYERS-1):
            lin_layer = nn.Linear(LATENT_SIZE, LATENT_SIZE)
            nn.init.eye_(lin_layer.weight)
            nn.init.zeros_(lin_layer.bias)
            fcc_layers.extend([
                lin_layer,
                nn.ReLU() # TODO 0.2 was a somewhat large leaky ReLU valus
            ])
        
        self.layers = nn.Sequential(*fcc_layers)
        
    def forward(self, x):
        x = self.layers(x) 
        return x
