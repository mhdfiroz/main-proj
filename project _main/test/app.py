from flask import Flask, request, render_template
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained Random Forest model
with open('parkinson_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define a function to make predictions
def make_prediction(features):
    # Scale the input features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features.reshape(-1, 1))
    
    # Make a prediction using the Random Forest model
    prediction = model.predict(features_scaled)
    
    return prediction[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the input features from the form
        features = []
        for i in range(10):
            feature_name = f'feature{i+1}'
            feature_value = float(request.form[feature_name])
            features.append(feature_value)
        
        # Make a prediction using the input features
        prediction = make_prediction(np.array(features))
        
        # Render the template with the prediction result
        return render_template('index.html', prediction=prediction)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
