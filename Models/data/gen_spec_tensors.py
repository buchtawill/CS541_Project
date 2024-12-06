# Modified from https://importchris.medium.com/how-to-create-understand-mel-spectrograms-ff7634991056

import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
# import IPython.display as ipd
import os
import csv
import torch
from tqdm import tqdm

GENRE_MAP_NAME_ID = {
    'Rock'         : 12,
    'Electronic'   : 15,
    'Experimental' : 38,
    'Hip-Hop'      : 211,
    'Folk'         : 17,
    'Instrumental' : 1235,
    'Pop'          : 10,
    'Classical'    : 5
}

GENRE_MAP_ID_NAME = {
    12      :   'Rock',
    15      :   'Electronic',
    38      :   'Experimental',
    211     :   'Hip-Hop',
    17      :   'Folk',
    1235    :   'Instrumental',
    10      :   'Pop',
    5       :   'Classical'
}

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

def get_mel_db(path, n_fft=2048, hop_length=256, n_mels=256, sr=22050):
    signal, sr = librosa.load(path, sr=sr)
    mel = librosa.feature.melspectrogram(y=signal, n_fft=n_fft, hop_length=hop_length, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max(mel))
    return mel_db

if __name__=="__main__":

    fma_path = 'fma_small'
    song_info = 'mp3_titles_and_genres_medium.csv'

    # Convert the spectrogram back to audio
    tensor_list = []
    genre_list = []
    error_list = []
    counts = {}
    # dir: data/fma_medium/000, ...
    
    with open(song_info, 'r') as file:
        reader = csv.DictReader(file)
        total_songs_processed = 0
        for row in reader:
            mp3_path = fma_path + '/' + row['path']
            genre_id = int(row['genre_id'])
            
            # if(total_songs_processed == 10000):
            #     break
            # Only process file if the genre id is in GENRE_MAP_ID_NAME
            if(genre_id not in GENRE_MAP_ID_NAME):
                print(f"INFO: Not processing song {mp3_path}, id not in genre map")
            else:
                print(f"  INFO: Processing song {mp3_path}. Genre: {genre_id}: {GENRE_MAP_ID_NAME[genre_id]} ", end='',)
                
                if(os.path.isfile(mp3_path)):
                    # Catch librosa errors
                    try:
                        mel_db = get_mel_db(mp3_path, hop_length=256, n_mels=128)
                        if(mel_db.shape[1] >= 1580):
                            print(f"Shape: {mel_db.shape}", flush=True)
                            mel_db = mel_db[0:, 0:1580]
                            # print("Spectrogram Shape: " + str(get_spectrogram_db(mp3_path).shape))
                            spec_tensor = torch.from_numpy(mel_db)
                            
                            tensor_list.append(spec_tensor)
                            genre_list.append(genre_id)
                            total_songs_processed += 1
                            if(GENRE_MAP_ID_NAME[genre_id] in counts):
                                counts[GENRE_MAP_ID_NAME[genre_id]] += 1
                            else:
                                counts[GENRE_MAP_ID_NAME[genre_id]] = 1
                        else:
                            print(f"Not adding, shape: {mel_db.shape}")
                    except:
                        print()
                        print(f"  ERROR: Error opening {mp3_path}")
                        error_list.append(mp3_path)
    
    spectrogram_tensors = torch.stack(tensor_list)
    print(f"INFO: Spectrogram tensor shape: {spectrogram_tensors.shape}")
    genre_tensors = torch.tensor(genre_list)
    print(f"INFO: Number of errors: {len(error_list)}")
    torch.save(spectrogram_tensors, "spec_tens_256hop_128mel_x.pt")
    torch.save(genre_tensors, "spec_tens_256hop_128mel_y.pt")
    print("Genre breakdown:")
    for key in counts:
        print(f"{key}: {counts[key]}")