import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

class MelSpectrogramDataset3Sec(Dataset):
    def __init__(self, data_file, label_file, segment_length=129, transform=None):
        """
        Dataset for handling mel spectrograms split into 3-second segments.

        Args:
            data_file (str): Path to the spectrogram data file
            label_file (str): Path to the label file
            segment_length (int): Length of each segment in frames (129 frames ≈ 3 seconds)
            transform: Optional transform to be applied to the segments
        """
        self.full_spectrograms = torch.load(data_file)
        self.full_labels = torch.load(label_file)
        self.segment_length = segment_length
        self.transform = transform

        # Genre mapping
        self.genre_map = {
            12: 0,  #rock
            15: 1,  #electronic
            38: 2,  #experimental
            17: 3,  #folk
            1235: 4,  #instrumental
            10: 5,  #pop
            #5: 6  #classical
        }

        new_labels = []
        for label in self.full_labels:
            label_item = label.item()
            if label_item not in self.genre_map:
                raise ValueError(f"Found unexpected label {label_item} not in genre mapping")
            new_labels.append(self.genre_map[label_item])

        self.full_labels = torch.tensor(new_labels)

        # Calculate number of segments per spectrogram
        self.spec_length = self.full_spectrograms[0].shape[1]  # Time dimension
        self.num_segments = self.spec_length // segment_length

        # Total number of segments across all spectrograms
        self.total_segments = len(self.full_spectrograms) * self.num_segments

        print(f"\nDataset initialized:")
        print(f"Original spectrogram shape: {self.full_spectrograms[0].shape}")
        print(f"Segment length: {segment_length} frames (≈ 3 seconds)")
        print(f"Number of segments per spectrogram: {self.num_segments}")
        print(f"Total number of segments: {self.total_segments}")

        # Print label distribution
        unique_labels, counts = torch.unique(self.full_labels, return_counts=True)
        print("\nLabel distribution:")
        genre_names = ['Rock', 'Electronic', 'Experimental',
                       'Folk', 'Instrumental', 'Pop', 'Classical']
        for label, count in zip(unique_labels, counts):
            print(f"{genre_names[label.item()]}: {count.item() * self.num_segments} segments")

    def __len__(self):
        return self.total_segments

    def __getitem__(self, idx):
        # Calculate which spectrogram and segment index we need
        spectrogram_idx = idx // self.num_segments
        segment_idx = idx % self.num_segments

        # Get the start and end indices for this segment
        start_idx = segment_idx * self.segment_length
        end_idx = start_idx + self.segment_length

        # Extract the segment
        spectrogram = self.full_spectrograms[spectrogram_idx]
        segment = spectrogram[:, start_idx:end_idx]

        # Add channel dimension
        segment = segment.unsqueeze(0)

        # Get the corresponding label
        label = self.full_labels[spectrogram_idx]

        if self.transform:
            segment = self.transform(segment)

        return segment, label


class GenreClassifierOld(nn.Module):
    def __init__(self, input_shape=(1, 128, 129), num_genres=7):
        super(GenreClassifierOld, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Calculate the size of the features output
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            features_output = self.features(x)
            self._to_linear = features_output.numel() // features_output.size(0)

        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_genres),
        )

        print(f"Initialized classifier with input size: {self._to_linear}")
        print(f"Number of output classes: {num_genres}")

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class GenreClassifier(nn.Module):
    def __init__(self, input_shape=(1, 128, 129), num_genres=6):
        super(GenreClassifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.1)
        )
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            self._to_linear = x.numel() // x.size(0)
        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_genres),
            nn.Softmax(dim=1)
        )
        print(f"Initialized classifier with input size: {self._to_linear}")
        print(f"Number of output classes: {num_genres}")
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    """
    Train the model with configurable epochs and batch size.
    Returns lists of training and validation metrics for plotting.
    """
    print("Starting training...")
    print(f"Training device: {device}")
    print(f"Number of epochs: {epochs}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Lists to store metrics
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            try:
                if inputs.isnan().any():
                    raise ValueError(f"NaN values in inputs at batch {batch_idx}")

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}')

            except RuntimeError as e:
                print(f"\nError in batch {batch_idx}:")
                print(f"Input shape: {inputs.shape}")
                print(f"Label shape: {labels.shape}")
                print(f"Unique labels in batch: {labels.unique().tolist()}")
                raise e

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f'Epoch {epoch + 1}/{epochs} - Training Loss: {epoch_loss:.4f}')

        if val_loader:
            val_loss, val_acc = validate_model(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

    return train_losses, val_losses, val_accuracies


def validate_model(model, val_loader, criterion, device):
    """
    Validate the model and return validation metrics.
    """
    model.eval()
    val_loss = 0.0
    genre_names = ['Rock', 'Electronic', 'Experimental', 'Folk', 'Instrumental', 'Pop', 'Classical']
    genre_correct = {genre: 0 for genre in genre_names}
    genre_total = {genre: 0 for genre in genre_names}

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            for label, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                genre = genre_names[label]
                genre_total[genre] += 1
                if label == pred:
                    genre_correct[genre] += 1

    avg_val_loss = val_loss / len(val_loader)
    print(f'\nValidation Loss: {avg_val_loss:.4f}')
    print('\nPer-genre accuracy:')

    accuracies = []
    for genre in genre_correct.keys():
        if genre_total[genre] > 0:
            accuracy = 100 * genre_correct[genre] / genre_total[genre]
            accuracies.append(accuracy)
            print(f'{genre}: {accuracy:.2f}% ({genre_correct[genre]}/{genre_total[genre]})')

    total_correct = sum(genre_correct.values())
    total_samples = sum(genre_total.values())
    overall_accuracy = 100 * total_correct / total_samples
    print(f'\nOverall accuracy: {overall_accuracy:.2f}%')

    return avg_val_loss, overall_accuracy


def load_and_test_model(model_path, data_file, label_file, batch_size=64, device='cuda'):
    """
    Load a pre-trained model and evaluate its training results.

    Args:
        model_path (str): Path to the saved model weights
        data_file (str): Path to the spectrogram data file
        label_file (str): Path to the label file
        batch_size (int): Batch size for evaluation
        device (str): Device to use for evaluation ('cuda' or 'cpu')

    Returns:
        tuple: Training metrics (losses, accuracies) if available
    """
    # Load dataset
    dataset = MelSpectrogramDataset3Sec(data_file, label_file, segment_length=129)
    num_classes = len(torch.unique(dataset.full_labels))

    # Initialize model
    model = GenreClassifier(input_shape=(1, 128, 129), num_genres=num_classes)

    # Load and evaluate the model
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict):
            # Load model state
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            # Set model to evaluation mode
            model = model.to(device)
            model.eval()

            # Extract training metrics if available
            training_metrics = {
                'train_losses': checkpoint.get('train_losses', []),
                'val_losses': checkpoint.get('val_losses', []),
                'val_accuracies': checkpoint.get('val_accuracies', [])
            }

            # Print training summary
            print("\nTraining Summary:")
            if training_metrics['train_losses']:
                print(f"Final training loss: {training_metrics['train_losses'][-1]:.4f}")
            if training_metrics['val_losses']:
                print(f"Final validation loss: {training_metrics['val_losses'][-1]:.4f}")
            if training_metrics['val_accuracies']:
                print(f"Final validation accuracy: {training_metrics['val_accuracies'][-1]:.2f}%")

            return model, training_metrics

        else:
            # If checkpoint is just the state dict
            model.load_state_dict(checkpoint)
            model = model.to(device)
            model.eval()
            print("\nNote: No training metrics available in checkpoint")
            return model, None

    except Exception as e:
        print(f"\nError loading model: {e}")
        raise

    return model, None



def main():
    # Configuration variables
    DATA_FILE = "/Volumes/T7/spec_tens_512hop_128mel_x.pt"
    LABEL_FILE = "/Volumes/T7/spec_tens_512hop_128mel_y.pt"
    SAVE_PATH = "/genremodel.pt"
    BATCH_SIZE = 64
    EPOCHS = 400
    LEARNING_RATE = 0.0001
    USE_CPU = True
    MODEL_PATH = "genre_classifier_softmax_v1.pth"

    TRAIN_MODEL = False  # Set to False to only test pre-trained model

    device = torch.device('cuda' if torch.cuda.is_available() and not USE_CPU else 'cpu')
    print(f"Using device: {device}")

    if TRAIN_MODEL:
        dataset = MelSpectrogramDataset3Sec(DATA_FILE, LABEL_FILE, segment_length=129)
        num_classes = len(torch.unique(dataset.full_labels))
        print(f"Number of unique classes detected: {num_classes}")

        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = GenreClassifier(input_shape=(1, 128, 129), num_genres=num_classes)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        train_losses, val_losses, val_accuracies = train_model(
            model, train_loader, val_loader, criterion, optimizer, device, EPOCHS
        )

        torch.save({
            'model_state_dict': model.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }, SAVE_PATH)

        print("\nTraining completed and model saved!")

    else:
        try:
            print("\nLoading and evaluating pre-trained model...")
            model, training_metrics = load_and_test_model(
                model_path=MODEL_PATH,
                data_file=DATA_FILE,
                label_file=LABEL_FILE,
                batch_size=BATCH_SIZE,
                device=device
            )

            # Create test dataset for evaluation
            dataset = MelSpectrogramDataset3Sec(DATA_FILE, LABEL_FILE, segment_length=129)
            total_size = len(dataset)
            train_size = int(0.7 * total_size)
            val_size = int(0.15 * total_size)
            test_size = total_size - train_size - val_size

            _, val_dataset, _ = random_split(
                dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )

            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            criterion = nn.CrossEntropyLoss()

            # Evaluate current model performance
            print("\nEvaluating current model performance...")
            val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)

            print(f"\nCurrent Model Performance:")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.2f}%")

            if training_metrics and any(training_metrics.values()):
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                if training_metrics['train_losses']:
                    plt.plot(training_metrics['train_losses'], label='Training Loss', color='blue')
                if training_metrics['val_losses']:
                    plt.plot(training_metrics['val_losses'], label='Validation Loss', color='red')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()

                plt.subplot(1, 2, 2)
                if training_metrics['val_accuracies']:
                    plt.plot(training_metrics['val_accuracies'], label='Validation Accuracy', color='green')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()

                plt.tight_layout()
                plt.show()

            print("\nModel evaluation completed!")

        except Exception as e:
            print(f"Evaluation failed with error: {e}")

if __name__ == "__main__":
    main()