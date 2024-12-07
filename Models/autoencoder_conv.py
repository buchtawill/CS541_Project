import torch.nn as nn
import torch.nn.functional as F
import torch

# Autoencoder Hyperparameters
ENCODER_DEPTH = 5          # Number of convolutional layers in the encoder
DECODER_DEPTH = 5          # Number of convolutional layers in the decoder
KERNEL_SIZE = 3            # Size of the convolutional kernels
# LATENT_VECTOR_SIZE = 512   # Size of the encoded feature vector (if using linear)
ENCODER_CHANNELS = [1, 16, 32, 64, 128, 256]  # Channels for each encoder layer
DECODER_CHANNELS = [256, 128, 64, 32, 16, 1]  # Channels for each decoder layer
STRIDE = 2                 # Stride for convolutional layers
PADDING = 1                # Padding for convolutional layers

# Fully convolutional autoencoder (latent space shape: [batch_size, 256, 16, 162])
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
                nn.LeakyReLU(0.2)
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
                nn.LeakyReLU(0.2) if i < DECODER_DEPTH - 1 else nn.Sigmoid()
            ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        x = self.encoder(x) 
        x = self.decoder(x)
        return x[:, :, :, :1290] # Crop the output to match the input size
    
class AutoencoderLargeKernels(nn.Module):
    """
    6 layers encoder, 6 layers decoder
    Input size is 128x1290
    """
    def __init__(self):
        super(AutoencoderLargeKernels, self).__init__()
        
        encoder_ch      = [1, 16, 32, 64, 128, 256, 512]
        encoder_kernels = [9, 7, 5, 3, 3, 3]
        
        decoder_ch = [512, 256, 128, 64, 32, 16, 1]
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, stride=(2, 4), padding=4),  # Output: 64 x 64 x 645
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=7, stride=(2, 3), padding=3),  # Output: 128 x 32 x 323
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=5, stride=(2, 3), padding=2),  # Output: 256 x 16 x 162
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # Output: 512 x 8 x 81
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # Output: 1024 x 4 x 41
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=(2, 4), padding=(1, 0), output_padding=(1, 0)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=(2, 3), padding=(1, 0), output_padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=(2, 3), padding=(2, 0), output_padding=(1, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=7, stride=2, padding=3, output_padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=9, stride=2, padding=(4, 3), output_padding=(1, 1)),
        )
        
        bottleneck_size = 4096
        self.linear_down = nn.Linear(512 * 4 * 9, bottleneck_size)
        self.linear_up = nn.Linear(bottleneck_size, 512 * 4 * 9)
        
    def forward(self, x):
        # Add channel dimension to x
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = F.leaky_relu(self.linear_down(x), 0.1)
        
        # Encoded shape: [batch_size, 512, 4, 9] (verified)
        # print(f"Encoded shape: {x.shape}")
        x = self.linear_up(x)
        
        x = x.view(-1, 512, 4, 9)
        
        x = self.decoder(x)
        
        return x
        
# Convolutional autoencoder with linear layers (latent space shape: [batch_size, 512(LATENT_VECTOR_SIZE)])
# class Autoencoder_ConvLinear(nn.Module):
#     def __init__(self):
#         super(Autoencoder_ConvLinear, self).__init__()
        
#         # Encoder
#         encoder_layers = []
#         for i in range(ENCODER_DEPTH):
#             in_channels = ENCODER_CHANNELS[i]
#             out_channels = ENCODER_CHANNELS[i + 1]
#             encoder_layers.extend([
#                 nn.Conv2d(in_channels, out_channels, 
#                     kernel_size=KERNEL_SIZE, 
#                     stride=STRIDE, 
#                     padding=PADDING),
#                 nn.ReLU()
#             ])
        
#         encoder_layers.extend([
#             nn.Flatten(),
#             nn.Linear(ENCODER_CHANNELS[-1] * 16 * 162, LATENT_VECTOR_SIZE)
#         ])
        
#         self.encoder = nn.Sequential(*encoder_layers)
        
#         # Decoder
#         decoder_layers = [
#             nn.Linear(LATENT_VECTOR_SIZE, DECODER_CHANNELS[0] * 16 * 162),
#             nn.ReLU(),
#             nn.Unflatten(1, (DECODER_CHANNELS[0], 16, 162))
#         ]
        
#         for i in range(DECODER_DEPTH):
#             in_channels = DECODER_CHANNELS[i]
#             out_channels = DECODER_CHANNELS[i + 1]
#             decoder_layers.extend([
#                 nn.ConvTranspose2d(in_channels, out_channels,
#                     kernel_size=KERNEL_SIZE,
#                     stride=STRIDE,
#                     padding=PADDING,
#                     output_padding=1),
#                 nn.ReLU() if i < DECODER_DEPTH - 1 else nn.Sigmoid()
#             ])
        
#         self.decoder = nn.Sequential(*decoder_layers) 
        
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x[:, :, :, :1290] # Crop the output to match the input size