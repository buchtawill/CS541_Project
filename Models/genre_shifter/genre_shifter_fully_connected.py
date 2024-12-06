import torch.nn as nn

# Genre Shifter Hyperparameters (reasonably simple)
NUM_GENRES = 2 # Will be one hot encoded as input
LATENT_SIZE  = 2048  # Channels for each encoder layer, keep same size for simplicity
NUM_LAYERS = 5

# Fully convolutional autoencoder (latent space shape: [batch_size, 256, 16, 162])
class Genre_Shifter_Fully_Connected(nn.Module):
    def __init__(self):
        super(Genre_Shifter_Fully_Connected, self).__init__()
        
        # Fully Connected Layers (from 2048 latent representation of song)
        fcc_layers = []
        layers_size = 2048 + 2*NUM_GENRES # Calculate layer size with one hot (x2 because of input genre and output genre encoding)
        for i in range(NUM_LAYERS):
            fcc_layers.extend([
                nn.Linear(layers_size, layers_size),
                nn.LeakyReLU(0.2)
            ])
        
        self.layers = nn.Sequential(*fcc_layers)
        
    def forward(self, x):
        x = self.layers(x) 
        return x


    

