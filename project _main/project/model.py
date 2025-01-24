import pickle
import numpy as np
import joblib

# Load the trained model
with open('parkinson_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the pre-fitted scaler
scaler = joblib.load('scaler.pkl')

def predict_parkinson(features):
    # Preprocess the input features (scale them)
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    # Make a prediction
    prediction = model.predict(features_scaled)
    return prediction[0]
