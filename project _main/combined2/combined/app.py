import os
import numpy as np
import tensorflow as tf
import pandas as pd
import joblib
from flask import Flask, request, redirect, render_template
from werkzeug.utils import secure_filename
import cv2

# Initialize Flask app
app = Flask(__name__)

# Define the folder to store uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained CNN model (for image-based predictions)
image_model = tf.keras.models.load_model('parkinsons_cnn_model.h5')

# Load the Random Forest model (for voice-based predictions)
voice_model = joblib.load('parkinson_model.pkl')

# Define allowed file extensions for both image and CSV
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_CSV_EXTENSIONS = {'csv'}

# Check if the file is of allowed type (image or CSV)
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Preprocess the image for prediction
def preprocess_image(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (128, 128))  # Resizing to match model input
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize to range [0, 1]
    return img

# Preprocess the voice CSV file for prediction
def preprocess_voice_features(file_path):
    df = pd.read_csv(file_path)
    # Assuming the model expects specific features
    # Modify this if the model expects particular columns/features
    features = df.values
    return features

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling the file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the selected prediction method
    method = request.form.get('method')

    # Check if a file was uploaded
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    # If no file was uploaded or an incorrect file type was selected
    if file.filename == '':
        return redirect(request.url)
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # If the user selected the image-based prediction method
    if method == 'image' and allowed_file(filename, ALLOWED_IMAGE_EXTENSIONS):
        # Preprocess the image and make a prediction
        img = preprocess_image(file_path)
        prediction = image_model.predict(img)
        result = "Parkinson's Detected" if prediction > 0.5 else "Healthy"

    # If the user selected the voice-based prediction method
    elif method == 'voice' and allowed_file(filename, ALLOWED_CSV_EXTENSIONS):
        # Preprocess the voice features and make a prediction
        voice_features = preprocess_voice_features(file_path)
        prediction = voice_model.predict(voice_features)
        result = "Parkinson's Detected" if prediction[0] == 1 else "Healthy"

    else:
        return "Invalid file type. Please upload a valid image or CSV file."

    # Return the prediction result
    return f'Prediction: {result}'

# Run the Flask app
if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
