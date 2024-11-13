# Modified from https://importchris.medium.com/how-to-create-understand-mel-spectrograms-ff7634991056

import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import sounddevice as sd
from mutagen import File

def reconstruct_audio(mel_spec, n_fft, sr, hop_length):
    #mel_spec = librosa.db_to_amplitude(mel_spec_db)
    mel_to_stft = librosa.feature.inverse.mel_to_stft(mel_spec, n_fft=n_fft, sr=sr)
    audio = librosa.griffinlim(mel_to_stft, n_fft=n_fft, hop_length=hop_length, n_iter=32)
    return audio


# Rap "Models\data\fma_small\146\146019.mp3"
# "Models\data\fma_small\127\127180.mp3"
# Trip-Hop Models\data\fma_small\073\073124.mp3
# Dubsetp Models\data\fma_small\151\151404.mp3
# Experimental Pop Models\data\fma_small\153\153956.mp3
path = r"Models\data\fma_small\112\112066.mp3"
print("Path: " + path)
# Load the file metadata
audio_file = File(path)

# Retrieve artist and genre metadata
title = audio_file.get('TIT2')
artist = audio_file.get('TPE1') 
genre = audio_file.get('TCON')   
print("Song: " + str(title))
print("Artist: " + str(artist))
print("Genre: " + str(genre))

signal, sr = librosa.load(path)

print("Duration: " + str(librosa.get_duration(y=signal, sr=sr)))

# Generate the spectrogram
n_fft = 2048
hop_length = 512

spectrogram = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)

mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128, fmax=8000)
mel_spectrogram_db = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)

plot = True
if (plot):
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mel_spectrogram, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency (mel)')
    plt.tight_layout()
    plt.show()
    
# Convert the spectrogram back to audio
reconstructed_audio = reconstruct_audio(spectrogram, n_fft, sr, hop_length)
#reconstructed_audio = librosa.istft(spectrogram, n_fft=n_fft, hop_length=hop_length)
#reconstructed_audio = reconstructed_audio[:, :1290]

# Play the reconstructed audio
print("Spectrogram Shape: " + str(mel_spectrogram.shape))
print("Reconstructed Audio Length " + str(librosa.get_duration(y=reconstructed_audio, sr=sr)))
print("Playing...")
sd.play(reconstructed_audio, sr)
sd.wait()
print("Done.")
