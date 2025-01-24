from flask import Flask, render_template, request
from model import predict_parkinson

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Extract features from the form
        features = [float(request.form.get(f'feature{i}')) for i in range(1, 23)]  # Assuming 22 features
        
        # Make a prediction
        prediction = predict_parkinson(features)
        prediction = 'Parkinson’s Positive' if prediction == 1 else 'Parkinson’s Negative'

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

    
