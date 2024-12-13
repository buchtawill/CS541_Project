# Modified from https://importchris.medium.com/how-to-create-understand-mel-spectrograms-ff7634991056

import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import sounddevice as sd
from mutagen import File

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

    n_fft, hop_length, sr = 2048, 512, 22050

    #mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128, fmax=8000)
    #mel_spectrogram_db = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)

    plot = True
    if (plot):
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(get_spectrogram_db(path), sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', fmax=8000, cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram Specshow')
        plt.xlabel('Time')
        plt.ylabel('Frequency (mel)')
        plt.tight_layout()

        plt.figure(figsize=(10, 6))
        mel_db = get_mel_db(path)
        times = librosa.frames_to_time(np.arange(mel_db.shape[1]), sr=sr, hop_length=hop_length) # get x axis times
        plt.imshow(mel_db, aspect='auto', origin='lower', extent=[times[0], times[-1], 0, mel_db.shape[0]], cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram Imshow')
        plt.xlabel('Time')
        plt.ylabel('Frequency (mel)')
        plt.tight_layout()

        plt.show()

    # Convert the spectrogram back to audio\
    mel_db = get_mel_db(path)
    reconstructed_audio = reconstruct_audio_mel(mel_db, n_fft=n_fft, hop_length=hop_length)

    # Play the reconstructed audio
    print("Spectrogram Shape: " + str(get_spectrogram_db(path).shape))
    print("Reconstructed Audio Length " + str(librosa.get_duration(y=reconstructed_audio, sr=sr)))

    print("Playing...")
    sd.play(reconstructed_audio, sr)
    sd.wait()
    print("Done.")

    print("Playing Original Audio")
    signal, sr = librosa.load(path, sr=sr) # Signal for playing original audio
    sd.play(signal, sr)
    sd.wait()
    print("Done.")
