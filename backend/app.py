import os
import numpy as np
from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import librosa
from flask_cors import CORS
import fitz  # PyMuPDF
import docx
import pyttsx3


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}) 


# Load the trained model once at startup
#model = tf.keras.models.load_model("ai_voice_recognition_model.h5")
model_path = os.path.join(os.path.dirname(__file__), "ai_voice_recognition_model.h5")
model = tf.keras.models.load_model(model_path)


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
        print(prediction)
        return "AI-Generated" if prediction >= 0.5 else "Human"
    else:
        return "Error processing the file."

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        # Ensure the /tmp directory exists
        file_dir = '/tmp'
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        
        file_path = os.path.join(file_dir, file.filename)
        file.save(file_path)
        
        try:
            # Predict the voice type
            result = predict_voice(file_path, model)
        finally:
            # Clean up the saved file
            if os.path.exists(file_path):
                os.remove(file_path)
        print(result)
        return jsonify({"result": result})
    
#tts
# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ''
    for page in doc:
        text += page.get_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    text = ''
    for para in doc.paragraphs:
        text += para.text
    return text

# Function to convert text to speech
def text_to_speech(text, audio_path):
    engine = pyttsx3.init()
    engine.save_to_file(text, audio_path)
    engine.runAndWait()

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # Ensure the /tmp directory exists
        file_dir = '/tmp'
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        file_path = os.path.join(file_dir, file.filename)
        file.save(file_path)

        try:
            if file.filename.endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            elif file.filename.endswith('.docx'):
                text = extract_text_from_docx(file_path)
            else:
                return jsonify({'error': 'Unsupported file type'}), 400

            audio_path = os.path.join(file_dir, 'output.mp3')
            text_to_speech(text, audio_path)

            # Use send_file without opening the file explicitly
            response = send_file(
                audio_path,
                as_attachment=True,
                download_name='output.mp3',
                mimetype='audio/mp3'
            )
        finally:
            # Clean up the saved files
            if os.path.exists(file_path):
                os.remove(file_path)
            # Delay the removal of the audio file until after the response is sent
            @response.call_on_close
            def cleanup():
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            return response

if __name__ == '__main__':
    app.run(debug=True)
