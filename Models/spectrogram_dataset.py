import torch
from torch.utils.data import Dataset

class SpectrogramDataset(Dataset):
    def __init__(self, tensors: str):
        try:
            
            # Load the tensor and unsqueeze to add a channel dimension
            self.data = torch.load(tensors, weights_only=True).unsqueeze(1)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"ERROR [spectrogram_dataset.py::__init__()] Error loading tensor file: {e}")
            raise Exception(e)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]