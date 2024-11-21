# Modified from https://importchris.medium.com/how-to-create-understand-mel-spectrograms-ff7634991056

import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import sounddevice as sd
from mutagen import File
import os
import torch
from tqdm import tqdm

"""
# Here are some fuctions that seem to corrupt the audio, but I'm keeping around in case we try to get mel spectrogram to work.
def reconstruct_audio_mel_broken(mel_spec_db, n_fft, sr, hop_length):
    amplitude_ref = np.max(np.abs(spectrogram))
    mel_spec = librosa.db_to_amplitude(mel_spec_db, ref=amplitude_ref)
    mel_to_stft = librosa.feature.inverse.mel_to_stft(mel_spec, n_fft=n_fft, sr=sr)
    audio = librosa.griffinlim(mel_to_stft, n_fft=n_fft, hop_length=hop_length, n_iter=32)
    return audio

def reconstruct_audio_1(spectrogram_db, n_fft, hop_length):
    amplitude_ref = np.max(np.abs(spectrogram_db))
    spectrogram_db_constructed = librosa.db_to_amplitude(spectrogram_db, ref=amplitude_ref) * np.exp(1j * np.angle(spectrogram))
    reconstructed_audio = librosa.istft(spectrogram_db_constructed, n_fft=n_fft, hop_length=hop_length)
    return reconstructed_audio"""

def reconstruct_audio(spectrogram_db, n_fft, hop_length, num_iters=30):
    # Convert decibel spectrogram to linear amplitude
    amplitude_ref = np.max(np.abs(spectrogram_db))
    spectrogram_amplitude = librosa.db_to_amplitude(spectrogram_db, ref=amplitude_ref)
    
    # Use Griffin-Lim to approximate the phase and reconstruct the audio
    reconstructed_audio = librosa.griffinlim(
        spectrogram_amplitude,
        n_iter=num_iters,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    return reconstructed_audio

def reconstruct_audio_mel(mel_db, n_fft=2048, hop_length=512, ref=0, n_iter=64):
    if (ref==0): # No manually set reference
        ref = 10 ** (np.max(mel_db) / 10.0)
    mel = librosa.db_to_power(mel_db, ref=ref)
    stft = librosa.feature.inverse.mel_to_stft(mel_db, sr=sr, n_fft=n_fft, power=2.0)
    reconstructed_audio = librosa.griffinlim(stft, hop_length=hop_length, n_iter=n_iter)
    return reconstructed_audio

def get_spectrogram(path, n_fft=2048, hop_length=512, sr=22050):
    signal, sr = librosa.load(path, sr=sr)

    spectrogram = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    spectrogram = spectrogram[:, :1290] # Cut to EXACT value around 30s (clips vary slightly)

    return spectrogram

def get_spectrogram_db(path, n_fft=2048, hop_length=512, sr=22050):
    signal, sr = librosa.load(path, sr=sr)

    spectrogram = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    spectrogram = spectrogram[:, :1290] # Cut to EXACT value around 30s (clips vary slightly)
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)

    return spectrogram_db

def get_mel_db(path, n_fft=2048, hop_length=512, n_mels=128, sr=22050):
    signal, sr = librosa.load(path, sr=sr)
    mel = librosa.feature.melspectrogram(y=signal, n_fft=n_fft, hop_length=hop_length, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max(mel))
    return mel_db

if __name__=="__main__":

    path = "C:\\Users\\bucht\\OneDrive\\Desktop\\CS541_SongShifter\\Models\\data\\000\\000002.mp3"
    dir = "C:\\Users\\bucht\\OneDrive\\Desktop\\CS541_SongShifter\\Models\\data\\000"
    names = os.listdir(dir)

    n_fft, hop_length, sr = 2048, 512, 22050

    # Convert the spectrogram back to audio
    tensor_list = []
    for name in tqdm(names):
        path = dir + "\\" + name
        mel_db = get_mel_db(path)
        mel_db = mel_db[0:, 0:1290]
        # print("Spectrogram Shape: " + str(get_spectrogram_db(path).shape))
        spec_tensor = torch.from_numpy(mel_db)

        tensor_list.append(spec_tensor)
    
    spectrogram_tensors = torch.stack(tensor_list)
    print(f"Tensor shape: {spectrogram_tensors.shape}")
    torch.save(spectrogram_tensors, "spectrogram_tensors_test.pt")