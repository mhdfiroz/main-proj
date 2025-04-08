from flask import Flask, render_template, request, redirect, send_file
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import pandas as pd
import os
import joblib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

RECORDINGS_FOLDER = 'recordings'
os.makedirs(RECORDINGS_FOLDER, exist_ok=True)

# Load model
model = joblib.load('parkinson_model.pkl')

# Feature extraction from audio
def extract_features(filename):
    y, sr = librosa.load(filename)
    features = {}
    
    features['MDVP:Fo(Hz)'] = np.mean(librosa.yin(y, fmin=50, fmax=500, sr=sr))
    features['MDVP:Fhi(Hz)'] = np.max(librosa.yin(y, fmin=50, fmax=500, sr=sr))
    features['MDVP:Flo(Hz)'] = np.min(librosa.yin(y, fmin=50, fmax=500, sr=sr))
    features['MDVP:Jitter(%)'] = np.std(librosa.effects.harmonic(y)) * 100
    features['MDVP:Jitter(Abs)'] = np.mean(np.abs(np.diff(y)))
    features['MDVP:RAP'] = np.mean(librosa.feature.rms(y=y))
    features['MDVP:PPQ'] = np.std(librosa.feature.rms(y=y))
    features['Jitter:DDP'] = np.var(librosa.feature.rms(y=y))
    features['MDVP:Shimmer'] = np.std(y)
    features['MDVP:Shimmer(dB)'] = 10 * np.log10(np.var(y))
    features['Shimmer:APQ3'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['Shimmer:APQ5'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features['MDVP:APQ'] = np.std(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['Shimmer:DDA'] = np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features['NHR'] = np.mean(librosa.feature.zero_crossing_rate(y))
    features['HNR'] = np.max(librosa.effects.harmonic(y))
    features['RPDE'] = np.mean(librosa.feature.spectral_flatness(y=y))
    features['DFA'] = np.std(librosa.feature.spectral_flatness(y=y))
    features['spread1'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features['spread2'] = np.var(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features['D2'] = np.mean(librosa.feature.mfcc(y=y, sr=sr)[0])
    features['PPE'] = np.std(librosa.feature.mfcc(y=y, sr=sr)[1])
    
    return pd.DataFrame([features])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record', methods=['POST'])
def record():
    duration = 5
    fs = 22050
    output_path = os.path.join(RECORDINGS_FOLDER, 'output.wav')

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(output_path, fs, recording)

    features_df = extract_features(output_path)
    features_df.to_csv('voice_features.csv', index=False)

    pred = model.predict(features_df)[0]
    result = 'Parkinson Detected' if pred == 1 else 'Healthy'

    return render_template('result.html', result=result, table=features_df.to_html(index=False))

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'csv_file' not in request.files:
        return "No file uploaded.", 400

    file = request.files['csv_file']
    if file.filename == '':
        return "Empty file name.", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    df = pd.read_csv(filepath)
    pred = model.predict(df)[0]
    result = 'Parkinson Detected' if pred == 1 else 'Healthy'

    return render_template('result.html', result=result, table=df.to_html(index=False))

@app.route('/download_csv')
def download_csv():
    return send_file('voice_features.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
