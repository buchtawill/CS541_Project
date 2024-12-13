
import torch
import torch.nn as nn
from torchviz import make_dot
import librosa

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Models.autoencoder_conv import Autoencoder_FullyConv, Autoencoder_ConvLinear
from Models.classifier import GenreClassifier
from Models.spectrogram_dataset import SpectrogramDataset
from Models.genre_shifter.genre_shifter_models import Genre_Shifter_Fully_Connected
from Models.genre_shifter.genre_shifter_dataset import GenreShifterDataset

# Instantiate the model
for ModelArchitecture, input_size, save_path in [
                (Autoencoder_ConvLinear().encoder, (1, 1, 128, 1290), r"Models\architecture_visualization\outputs\ae_convLin_encoder"),
                (Autoencoder_ConvLinear().decoder, (1, 1024), r"Models\architecture_visualization\outputs\ae_convLin_decoder"),
                (Genre_Shifter_Fully_Connected(), (1, 1038), r"Models\architecture_visualization\outputs\genre_shifter"),
                (GenreClassifier(), (1, 1, 128, 129), r"Models\architecture_visualization\outputs\genre_classifier")
                ]:

    print("Architecture: ", str(ModelArchitecture))
    model = ModelArchitecture

    # Create a dummy input tensor with the correct shape
    dummy_input = torch.randn(*input_size)  # Batch size 1,

    # Pass the dummy input through the model
    output = model(dummy_input)

    # Combine inputs and outputs for visualization
    # Use a tuple to pass both the input and output
    dot = make_dot((output, dummy_input), params=dict(model.named_parameters()))

    # Render and view the graph
    dot.render(save_path, format="png")  # Saves the graph as a PNG file
    dot.view()  # Opens the graph in the default image viewer
