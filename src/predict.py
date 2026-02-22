import os
import joblib
import numpy as np

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "attrition_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Load model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ⚠️ IMPORTANT:
# The number of values must match number of features used in training.
# For now, we give random values just to test.

sample_employee = np.random.rand(1, model.n_features_in_)

# Scale input
sample_employee_scaled = scaler.transform(sample_employee)

# Predict
prediction = model.predict(sample_employee_scaled)

if prediction[0] == 1:
    print("Prediction: Employee Will Leave")
else:
    print("Prediction: Employee Will Stay")