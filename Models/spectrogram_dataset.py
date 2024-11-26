import torch
from torch.utils.data import Dataset

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