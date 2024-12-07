import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class GenreShifterDataset():

    def __init__(self, data_file, label_file):
        # TODO document
        self.full_spectrograms = torch.load(data_file)
        self.full_labels = torch.load(label_file)

        # Genre mapping
        self.genre_map = {
            12: 0,  #rock
            15: 1,  #electronic
            38: 2,  #experimental
            17: 3,  #folk
            1235: 4,  #instrumental
            10: 5,  #pop
            5: 6  #classical
        }

        new_labels = []
        for label in self.full_labels:
            label_item = label.item()
            if label_item not in self.genre_map:
                raise ValueError(f"Found unexpected label {label_item} not in genre mapping")
            new_labels.append(self.genre_map[label_item])

        self.full_labels = torch.tensor(new_labels)
        self.full_labels = F.one_hot(self.full_labels, num_classes=len(self.genre_map))

        # Print label distribution
        unique_labels, counts = torch.unique(self.full_labels, return_counts=True)
        genre_names = ['Rock', 'Electronic', 'Experimental',
                       'Folk', 'Instrumental', 'Pop', 'Classical']
        
        for label, count in zip(unique_labels, counts):
            print(f"{genre_names[label.item()]}: {count.item()}")

    def __len__(self):
        return len(self.full_spectrograms)
    
    def __getitem__(self, idx):
        return self.full_spectrograms[idx], self.full_labels[idx]

if __name__=="__main__":
    g = GenreShifterDataset(data_file=r"Models\data\test_dataset\normal_128m_512h_x.pt", label_file=r"Models\data\test_dataset\normal_128m_512h_y.pt")
    print(g)