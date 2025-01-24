import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset
data = pd.read_csv('parkinsons.data')

# Separate the features and target variable
X = data.drop(columns=['name', 'status'])  # 'name' is the identifier, 'status' is the target
y = data['status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Initialize the Random Forest model
model = RandomForestClassifier(random_state=42)

# Train the model on the training data
model.fit(X_train_scaled, y_train)


# Make predictions on the test data
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Save the model
with open('parkinson_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the scaler as well if needed
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
