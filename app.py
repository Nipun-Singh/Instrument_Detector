import flask
from flask import Flask, request, render_template
import os
import librosa
import numpy as np
import pickle
import uuid

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Optional: Load scaler if used
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def Feature_extractor(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None, res_type='kaiser_fast')

        if len(audio) == 0:
            raise ValueError("Audio file has no data!")

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs = mfccs[0:-1]

        if mfccs.shape[1] == 0:
            raise ValueError("MFCC extraction failed, got empty feature matrix.")

        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled.reshape(1, -1)

    except Exception as e:
        raise ValueError(f"Feature extraction error: {e}")


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        if 'audio_file' not in request.files:
            prediction = "No file part"
        file = request.files['audio_file']
        if file.filename == '':
            prediction = "No selected file"
        elif file and allowed_file(file.filename):
            filename = f"{uuid.uuid4()}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            file.stream.close()
            
        try:
            features = Feature_extractor(filepath)
            # If scaler used:
            # features = scaler.transform(features)
            prediction = model.predict(features)
            prediction = prediction[0].split('_')[1]
        except Exception as e:
            prediction = f"Error: {str(e)}"
        finally:
            os.remove(filepath)  # clean up

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
