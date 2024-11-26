import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

def load_and_preprocess_mel(mel_path):
    """
    Load and preprocess mel spectrogram from PNG file for genre prediction.

    Parameters:
        mel_path (str): Path to the mel spectrogram PNG file

    Returns:
        np.ndarray: Preprocessed mel spectrogram ready for model input (None, 130, 13, 1)
    """
    img = Image.open(mel_path)

    if img.mode != 'L':
        img = img.convert('L')

    mel_spec = np.array(img).astype(np.float32)
    mel_spec = mel_spec / 255.0

    img_resized = Image.fromarray((mel_spec * 255).astype(np.uint8))
    img_resized = img_resized.resize((130, 13), Image.Resampling.LANCZOS)
    mel_spec = np.array(img_resized).astype(np.float32) / 255.0

    mel_spec = np.expand_dims(mel_spec, axis=-1)

    mel_spec = np.expand_dims(mel_spec, axis=0)

    mel_spec = np.swapaxes(mel_spec, 1, 2)  # This should give us (1, 130, 13, 1)

    print("Shape after preprocessing:", mel_spec.shape)
    return mel_spec

def predict_genre(model, mel_features):
    """
    Predict genre for mel spectrogram features.

    Parameters:
        model: Loaded Keras model
        mel_features (np.ndarray): Preprocessed mel spectrogram

    Returns:
        list: Top 3 (genre name, confidence score) pairs
    """
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']

    predictions = model.predict(mel_features, verbose=0)

    sorted_idx = np.argsort(predictions[0])[::-1]
    top_predictions = [(genres[i], predictions[0][i]) for i in sorted_idx[:3]]

    return top_predictions

def main(model_path, mel_path):
    """
    Main function to load model and predict genre using CNN3 model.

    Parameters:
        model_path (str): Path to the saved model weights
        mel_path (str): Path to the mel spectrogram PNG file
    """
    try:
        model = load_model(model_path)
        print("Model's input shape:", model.input_shape)

        mel_features = load_and_preprocess_mel(mel_path)

        print(f"Final mel spectrogram shape: {mel_features.shape}")

        predictions = predict_genre(model, mel_features)

        print("\nTop 3 Genre Predictions:")
        for genre, confidence in predictions:
            print(f"{genre}: {confidence:.2%}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    MODEL_PATH = "./model_cnn3.h5"
    MEL_PATH = "./mfcc.png"

    main(MODEL_PATH, MEL_PATH)