import torch
import librosa, librosa.display
import sounddevice as sd
from data.gen_spectrograms import *
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from mutagen import File
from autoencoder_conv import Autoencoder_FullyConv
import soundfile as sf

# Qualitatively allows one to listen to autoencoder

# Some examples below
# Rap "Models\data\fma_small\146\146019.mp3"
# Models\data\fma_small\127\127180.mp3"
# Trip-Hop Models\data\fma_small\073\073124.mp3
# Dubsetp Models\data\fma_small\151\151404.mp3
# Experimental Pop Models\data\fma_small\153\153956.mp3

# Neural Network
autoencoder_NN_path = r"trained_autoencoder_conv_halfsize_linear_1024.pth"
autoencoder_NN_state_dict = torch.load(autoencoder_NN_path, weights_only=True)
Autoencoder_NN = Autoencoder_FullyConv()
Autoencoder_NN.load_state_dict(autoencoder_NN_state_dict)
Autoencoder_NN.eval()

# r"Models\data\fma_small\043\043020.mp3"
# input_song = r"Models\data\fma_small\112\112066.mp3"
input_song = r"Models\yummy.mp3" # Tom's path
# input_song = r"Path\to\test_song.mp3"
sr = 22050

signal, sr = librosa.load(input_song, sr=sr)

# Generate Mel spectrogram
mel_db = get_mel_db(path=input_song, n_mels=128)
mel_db = mel_db[:, :1290]

# Normalize the mel_db values from -80 to 0 into the range 0 to 1
mel_db_normalized = (mel_db + 80) / 80

reconstructed_mel = reconstruct_audio_mel(mel_db, sr=sr)
print("MELDB RECON", mel_db.shape)
with torch.no_grad():
    autoencoded_mel = Autoencoder_NN(torch.tensor(
        mel_db_normalized.reshape(1, 1, mel_db_normalized.shape[0], mel_db_normalized.shape[1]),
        dtype=torch.float32
    ))
# Compute MSE loss between original and autoencoded spectrograms
print('autoencoded mel spec. checking loss...')
mse_loss = torch.nn.MSELoss()
mel_tensor = torch.tensor(
    mel_db_normalized.reshape(1, 1, mel_db_normalized.shape[0], mel_db_normalized.shape[1]),
    dtype=torch.float32
)
loss = mse_loss(autoencoded_mel, mel_tensor)
print(f"MSE Loss: {loss.item():.6f}")

# Convert autoencoded Mel spectrogram to NumPy and denormalize
autoencoded_mel_arr = autoencoded_mel.cpu().detach().numpy()
mel_db_autoencoded = (autoencoded_mel_arr * 80) - 80
reconstructed_nn_mel = reconstruct_audio_mel(mel_db_autoencoded.reshape(mel_db.shape[0], mel_db.shape[1]), sr=sr)

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

save_reconstructed = True
save_autoencoded = True
model_tag = "linear_1024"
if save_reconstructed:
    print("Saving reconstructed mel")
    sf.write(input_song.replace('.mp3', f'_{model_tag}_mel_reconstructed.wav'), reconstructed_mel, sr)
if save_autoencoded:
    print("Saving autoencoded mel")
    sf.write(input_song.replace('.mp3', f'_{model_tag}_autoencoded.wav'), reconstructed_nn_mel, sr)

do_play_audio = True
if do_play_audio:
    # play_audio(signal, sr, "Original")
    play_audio(reconstructed_mel, sr, "Reconstructed")
    play_audio(reconstructed_nn_mel, sr, "Autoencoded and Reconstructed")

do_plot_spectrograms = True
if do_plot_spectrograms:
    spectrograms = [mel_db, 
                    mel_db_autoencoded.reshape(mel_db.shape[0], mel_db.shape[1])]
    titles = ['Original', 'Autoencoded']
    print('Plotting spectrograms...')
    plot_spectrograms(spectrograms, titles, sr)
