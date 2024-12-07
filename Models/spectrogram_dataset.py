import torch
from torch.utils.data import Dataset
import numpy as np

class SpectrogramDataset(Dataset):
    def __init__(self, tensors: str, split='train', train_ratio=0.8):
        try:
            self.data = torch.load(tensors, weights_only=True)
            self.data = (self.data + 80) / 80 # Normalize to [0, 1]
            
            # Set seed to 0 for reproducibility
            np.random.seed(0)
            
            # Generate indices and split them
            indices = np.arange(len(self.data))
            np.random.shuffle(indices)
            split_idx = int(len(indices) * train_ratio)
            
            # Select indices based on split
            if split == 'train':
                self.indices = indices[:split_idx]
            elif split == 'test':
                self.indices = indices[split_idx:]
            else:
                raise ValueError("Split must be either 'train' or 'test'")
                
        except (FileNotFoundError, RuntimeError) as e:
            print(f"ERROR [spectrogram_dataset.py::__init__()] Error loading tensor file: {e}")
            raise Exception(e)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.data[self.indices[idx]]