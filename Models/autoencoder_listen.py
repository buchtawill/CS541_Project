import torch
import sounddevice as sd
from data.gen_spectrograms import get_mel_db, reconstruct_audio_mel

# Qualitatively allows one to listen to autoencoder

# Neural Network
#autoencoder_nn = torch.load(r"trained_autoencoder_test_3.pth")

input_song = r"Models\data\fma_small\043\043020.mp3"
mel = get_mel_db(path=input_song)

sr = 22050
reconstructed_mel = reconstruct_audio_mel(mel, sr=sr)
reconstructed_nn_mel = reconstruct_audio_mel(mel, sr=sr)

# Original
print("Playing Original")
sd.play(input_song, sr)
sd.wait()

# Reconstructed
print("Playing Reconstructed")
sd.play(reconstructed_mel, sr)
sd.wait()

# Autoencoded and Reconstructed
print("Playing Autoencoded and Reconstructed")
sd.play(reconstructed_nn_mel, sr)
sd.wait()
