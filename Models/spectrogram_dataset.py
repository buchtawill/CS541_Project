import torch
from torch.utils.data import Dataset

class NormalizedSpectrogramDataset(Dataset):
    def __init__(self, tensors: str):
        try:
            
            # Load the tensor and unsqueeze to add a channel dimension
            self.data = torch.load(tensors, weights_only=True)
            # self.data = (self.data + 80) / 80  # Normalize to [0, 1]
        except (FileNotFoundError, RuntimeError) as e:
            print(f"ERROR [spectrogram_dataset.py::__init__()] Error loading tensor file: {e}")
            raise Exception(e)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

class SpectrogramDataset(Dataset):
    def __init__(self, tensors: str):
        try:
            self.data = torch.load(tensors, weights_only=True)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"ERROR [spectrogram_dataset.py::__init__()] Error loading tensor file: {e}")
            raise Exception(e)
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]