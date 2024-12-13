import torch
import librosa, librosa.display
import sounddevice as sd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Models.data.gen_spectrograms import *
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from mutagen import File
from genre_shifter_models import Genre_Shifter_Fully_Connected
from Models.autoencoder_conv import Autoencoder_ConvLinear
from Models.classifier import GenreClassifier

# Qualitatively allows one to listen to autoencoder

# Some examples below
# Rap "Models\data\fma_small\146\146019.mp3"
# Models\data\fma_small\127\127180.mp3"
# Trip-Hop Models\data\fma_small\073\073124.mp3
# Dubsetp Models\data\fma_small\151\151404.mp3
# Experimental Pop Models\data\fma_small\153\153956.mp3

# TODO not using GPU for simplicity at the moment to eval an example or two
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = 'cpu'
print("Device: ", DEVICE)

AUTOENCODER_NN_PATH = r"Models\data\models\best\conv_linear_1em4_100_12_8_h100.pth"
GENRE_CLASSIFIER_NN_PATH = r"Models\data\models\best\genre_classifier_v3_small.pth"
GENRE_SHIFTER_NN_PATH = r"genre_shifter_models\best\small_dataset_layer_1_alpha_0.995-lr-1e-6"

# Neural Networks
# Get autoencoder model
autoencoder_NN_state_dict = torch.load(AUTOENCODER_NN_PATH, weights_only=True)
Autoencoder_NN = Autoencoder_ConvLinear()
Autoencoder_NN.load_state_dict(autoencoder_NN_state_dict)
Autoencoder_NN.to(DEVICE)
Autoencoder_NN.eval() # Set to evaluate stuff for loss of genre shifter

# Get genre classifier model
genre_classifier_NN_state_dict = torch.load(GENRE_CLASSIFIER_NN_PATH, weights_only=True)
Genre_Classifier_NN = GenreClassifier(input_shape=(1, 128, 129), num_genres=6)
Genre_Classifier_NN.load_state_dict(genre_classifier_NN_state_dict)
Genre_Classifier_NN.to(DEVICE)
Genre_Classifier_NN.eval() # Set to evaluate stuff for loss of genre shifter

# Get genre classifier model
genre_classifier_NN_state_dict = torch.load(GENRE_SHIFTER_NN_PATH, weights_only=True)
Genre_Shifter_NN = Genre_Shifter_Fully_Connected()
Genre_Shifter_NN.load_state_dict(genre_classifier_NN_state_dict)
Genre_Shifter_NN.to(DEVICE)
Genre_Shifter_NN.eval() # Set to evaluate stuff for loss of genre shifter

# r"Models\data\fma_small\043\043020.mp3"
# input_song = r"Models\data\fma_small\112\112066.mp3"
# input_song = r"Models\test_song_001083.mp3" # Tom's path
input_song = r"Models\data\Mariah Carey - All I Want For Christmas Is You.mp3"
genre_input = torch.tensor([1, 0, 0, 0, 0, 0]).reshape(1, 6).to(DEVICE)
genre_output = torch.tensor([0, 1, 0, 0, 0, 0]).reshape(1, 6).to(DEVICE)
sr = 22050

signal, sr = librosa.load(input_song, sr=sr)

# Generate Mel spectrogram
mel_db = get_mel_db(path=input_song, n_mels=128)
mel_db = mel_db[:, :1290]

# Normalize the mel_db values from -80 to 0 into the range 0 to 1
mel_db_normalized = (mel_db + 80) / 80

reconstructed_mel = reconstruct_audio_mel(mel_db, sr=sr) * 32

# Run genre shifter
with torch.no_grad():
    encoded_mel = Autoencoder_NN.encoder(torch.tensor(
        mel_db_normalized.reshape(1, 1, mel_db_normalized.shape[0], mel_db_normalized.shape[1]),
        dtype=torch.float32
    ))
    genre_shifter_input = torch.cat((encoded_mel, genre_input, genre_output), dim=1)
    shifted_latent_vector = Genre_Shifter_NN(genre_shifter_input)
    shifted_decoded_vector = Autoencoder_NN.decoder(shifted_latent_vector)
    shifted_decoded_vector = shifted_decoded_vector[:, :, :, :1290] # Chop to proper size (needed when manually applying encode and decode)

# Compute MSE loss between original and autoencoded spectrograms
print('autoencoded mel spec. checking loss...')
mse_loss = torch.nn.MSELoss()
mel_tensor = torch.tensor(
    mel_db_normalized.reshape(1, 1, mel_db_normalized.shape[0], mel_db_normalized.shape[1]),
    dtype=torch.float32
)
loss = mse_loss(shifted_decoded_vector, mel_tensor)
print(f"MSE Loss: {loss.item():.6f}")

# Convert autoencoded Mel spectrogram to NumPy and denormalize
shifted_decoded_vector = shifted_decoded_vector.cpu().detach().numpy()
mel_db_shifted = (shifted_decoded_vector * 80) - 80 # TODO already normalized now?
reconstructed_nn_mel = reconstruct_audio_mel(mel_db_shifted.reshape(mel_db_shifted.shape[2], mel_db_shifted.shape[3]), sr=sr)
#reconstructed_nn_mel = reconstructed_nn_mel ** (0.5)

print(reconstruct_audio_mel)

import soundfile as sf
sf.write('All I Want For Christmas Alpha 0.99 1e-6.wav', reconstructed_nn_mel, sr, 'PCM_24')

def play_audio(audio, sr, description):
    print(f"\nPlaying {description} (Ctrl+C to skip)")
    try:
        sd.play(audio, sr)
        while sd.get_stream().active: # sd.wait() blocks Ctrl+C, so repeatedly sleep instead
            sd.sleep(100)
    except KeyboardInterrupt:
        sd.stop()
        print(f"Skipped {description}")

def plot_spectrograms(specs, titles, sr): # Plot multiple spectrograms side by side
    fig, axes = plt.subplots(1, len(specs), figsize=(15, 5))
    
    for i, (spec, title) in enumerate(zip(specs, titles)):
        img = librosa.display.specshow(spec, 
            y_axis='mel', 
            x_axis='time',
            sr=sr,
            ax=axes[i])
        axes[i].set_title(title)
    
    plt.colorbar(img, ax=axes, format='%+2.0f dB')
    plt.show()

play_audio(signal, sr, "Original")
play_audio(reconstructed_mel, sr, "Reconstructed")
play_audio(reconstructed_nn_mel, sr, "Autoencoded and Reconstructed")

spectrograms = [mel_db, 
                mel_db_shifted.reshape(mel_db.shape[0], mel_db.shape[1])]
titles = ['Original', 'Autoencoded']
print('Plotting spectrograms...')
plot_spectrograms(spectrograms, titles, sr)
