import torch.nn as nn

# Autoencoder Hyperparameters
ENCODER_DEPTH = 3          # Number of convolutional layers in the encoder
DECODER_DEPTH = 3          # Number of convolutional layers in the decoder
KERNEL_SIZE = 3            # Size of the convolutional kernels
LATENT_VECTOR_SIZE = 512  # Size of the encoded feature vector (if using linear)
ENCODER_CHANNELS = [1, 16, 32, 64]  # Channels for each encoder layer
DECODER_CHANNELS = [64, 32, 16, 1]  # Channels for each decoder layer
STRIDE = 2                 # Stride for convolutional layers
PADDING = 1                # Padding for convolutional layers

# Fully convolutional autoencoder (latent space shape: [batch_size, 64, 16, 162])
class Autoencoder_FullyConv(nn.Module):
    def __init__(self):
        super(Autoencoder_FullyConv, self).__init__()
        
        # Encoder
        encoder_layers = []
        for i in range(ENCODER_DEPTH):
            in_channels = ENCODER_CHANNELS[i]
            out_channels = ENCODER_CHANNELS[i + 1]
            encoder_layers.extend([
                nn.Conv2d(in_channels, out_channels, 
                    kernel_size=KERNEL_SIZE, 
                    stride=STRIDE, 
                    padding=PADDING),
                nn.ReLU()
            ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        
        for i in range(DECODER_DEPTH):
            in_channels = DECODER_CHANNELS[i]
            out_channels = DECODER_CHANNELS[i + 1]
            decoder_layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels,
                    kernel_size=KERNEL_SIZE,
                    stride=STRIDE,
                    padding=PADDING,
                    output_padding=1),
                nn.ReLU() if i < DECODER_DEPTH - 1 else nn.Sigmoid()
            ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        x = self.encoder(x) 
        x = self.decoder(x)
        return x[:, :, :, :1290] # Crop the output to match the input size
    
# Convolutional autoencoder with linear layers (latent space shape: [batch_size, 512(LATENT_VECTOR_SIZE)])
class Autoencoder_ConvLinear(nn.Module):
    def __init__(self):
        super(Autoencoder_ConvLinear, self).__init__()
        
        # Encoder
        encoder_layers = []
        for i in range(ENCODER_DEPTH):
            in_channels = ENCODER_CHANNELS[i]
            out_channels = ENCODER_CHANNELS[i + 1]
            encoder_layers.extend([
                nn.Conv2d(in_channels, out_channels, 
                    kernel_size=KERNEL_SIZE, 
                    stride=STRIDE, 
                    padding=PADDING),
                nn.ReLU()
            ])
        
        encoder_layers.extend([
            nn.Flatten(),
            nn.Linear(ENCODER_CHANNELS[-1] * 16 * 162, LATENT_VECTOR_SIZE)
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = [
            nn.Linear(LATENT_VECTOR_SIZE, DECODER_CHANNELS[0] * 16 * 162),
            nn.ReLU(),
            nn.Unflatten(1, (DECODER_CHANNELS[0], 16, 162))
        ]
        
        for i in range(DECODER_DEPTH):
            in_channels = DECODER_CHANNELS[i]
            out_channels = DECODER_CHANNELS[i + 1]
            decoder_layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels,
                    kernel_size=KERNEL_SIZE,
                    stride=STRIDE,
                    padding=PADDING,
                    output_padding=1),
                nn.ReLU() if i < DECODER_DEPTH - 1 else nn.Sigmoid()
            ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x[:, :, :, :1290] # Crop the output to match the input size