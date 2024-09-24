import numpy as np
import tensorflow as tf
import librosa

model  = tf.keras.models.load_model("./ai_voice_recognition_model.h5")

# Function to extract MFCC features from audio files
def extract_mfcc(file_path, max_pad_len=100):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        pad_width = max_pad_len - mfcc.shape[1]
        if pad_width > 0:
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to predict if a voice is human or AI-generated
def predict_voice(file_path, model):
    mfcc = extract_mfcc(file_path)
    if mfcc is not None:
        mfcc = np.expand_dims(mfcc, axis=-1)  # Add channel dimension
        mfcc = np.expand_dims(mfcc, axis=0)   # Add batch dimension
        prediction = model.predict(mfcc)
        return "AI-Generated" if prediction >= 0.5 else "Human"
    else:
        return "Error processing the file."
    

response = predict_voice(r'C:\Users\riya2\Downloads\AUDIO\REAL\taylor-original.wav',model)
print(response)