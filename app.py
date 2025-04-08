import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import torchvision.transforms.functional as TF

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform for input image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load ResNet18 model
def load_resnet18_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("resnet18_spiral_parkinsons.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model_dict = {
    "ResNet18": load_resnet18_model()
}

class_names = ['Healthy', 'Parkinson']

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    selected_model = "ResNet18"

    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)

        file = request.files["image"]
        selected_model = request.form.get("model")

        if file and selected_model in model_dict:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Load image and predict
            image = Image.open(filepath).convert("L")
            image = transform(image).unsqueeze(0).to(device)
            model = model_dict[selected_model]

            with torch.no_grad():
                outputs = model(image)
                probs = torch.softmax(outputs, dim=1)
                conf, predicted = torch.max(probs, 1)
                prediction = class_names[predicted.item()]
                confidence = conf.item() * 100

            return render_template("index.html", prediction=prediction,
                                   confidence=confidence, model=selected_model)

    return render_template("index.html", prediction=prediction,
                           confidence=confidence, model=selected_model)
    
if __name__ == "__main__":
    app.run(debug=True)
