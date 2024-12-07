import torch
import torch.nn as nn
import torch.nn.functional as F

# Modified Hyperparameters
PATCH_SIZE = 16  # Increased to reduce number of patches
NUM_HEADS = 8
ENCODER_DEPTH = 4  # Reduced from 6
DECODER_DEPTH = 4  # Reduced from 6
LATENT_DIM = 256  # Reduced from 512
HIDDEN_DIM = 512  # Reduced from 768
MLP_DIM = 2048  # Reduced from 3072
DROPOUT = 0.1
LEARNING_RATE = 1e-4

class PatchEmbed(nn.Module):
    def __init__(self, img_size=(128, 1290), patch_size=PATCH_SIZE, in_channels=1, embed_dim=HIDDEN_DIM):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, E, H', W')
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(HIDDEN_DIM)
        self.attn = nn.MultiheadAttention(HIDDEN_DIM, NUM_HEADS, dropout=DROPOUT)
        self.norm2 = nn.LayerNorm(HIDDEN_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN_DIM, MLP_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(MLP_DIM, HIDDEN_DIM),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        x = x + self._attention_block(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def _attention_block(self, x):
        x = x.transpose(0, 1)  # (N, B, E)
        x, _ = self.attn(x, x, x)
        x = x.transpose(0, 1)  # (B, N, E)
        return x

class AutoencoderTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.patch_embed = PatchEmbed()
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, HIDDEN_DIM))
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder() for _ in range(ENCODER_DEPTH)
        ])
        
        # Latent projection
        self.latent_proj = nn.Linear(HIDDEN_DIM * self.patch_embed.n_patches, LATENT_DIM)
        
        # Decoder (reverse projection + transformer layers)
        self.latent_decode = nn.Linear(LATENT_DIM, HIDDEN_DIM * self.patch_embed.n_patches)
        self.decoder_layers = nn.ModuleList([
            TransformerEncoder() for _ in range(DECODER_DEPTH)
        ])
        
        # Final reconstruction
        self.final_layer = nn.Sequential(
            nn.Linear(HIDDEN_DIM, PATCH_SIZE * PATCH_SIZE),
            nn.Sigmoid()  # Assuming spectrogram values are normalized between 0 and 1
        )
        
        # Initialize position embeddings
        nn.init.normal_(self.pos_embed, std=0.02)

    def encode(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        # Transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
            
        # Project to latent space
        x = x.flatten(1)
        latent = self.latent_proj(x)
        return latent

    def decode(self, latent):
        # Project back to sequence
        x = self.latent_decode(latent)
        x = x.view(-1, self.patch_embed.n_patches, HIDDEN_DIM)
        
        # Transformer decoder layers
        for layer in self.decoder_layers:
            x = layer(x)
        
        # Final reconstruction
        x = self.final_layer(x)
        
        # Reshape back to image dimensions
        H = 128 // PATCH_SIZE
        W = 1290 // PATCH_SIZE
        x = x.view(-1, H, W, PATCH_SIZE, PATCH_SIZE)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(-1, 1, 128, 1290)
        return x

    def forward(self, x):
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent
