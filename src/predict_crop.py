# ==========================================
# File 10: predict_crop.py
# Purpose: Load the saved model and perform prediction on sample input data.
# ==========================================

import os
import joblib
import pandas as pd

def load_saved_model():
    """
    Helper function to load the saved crop model.
    Checks if it exists; if not, warns the user.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "..", "models", "crop_model.pkl")
    
    if not os.path.exists(model_path):
        print(f"Model file not found at: {model_path}")
        print("Please train and save the model first by running 'python src/save_model.py'.")
        # For ease of use, we will try to train and save it automatically
        print("Attempting to run save_model.py automatically to generate the model...")
        from save_model import save_trained_model
        save_trained_model()
        
    # Load model
    model = joblib.load(model_path)
    return model

def make_prediction(n, p, k, temperature, humidity, ph, rainfall):
    """
    Function to accept features, structure them as a DataFrame, and predict.
    """
    # 1. Load the model
    model = load_saved_model()
    
    # 2. Convert sample input to a DataFrame with same column names as training
    # Columns must match: 'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'
    input_data = pd.DataFrame([[n, p, k, temperature, humidity, ph, rainfall]], 
                              columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    
    # 3. Predict the crop class
    predicted_crop = model.predict(input_data)[0]
    
    # 4. (Optional but premium) Get the prediction confidence (probability)
    probabilities = model.predict_proba(input_data)
    # Get the index of the predicted crop in the model's classes list
    crop_index = list(model.classes_).index(predicted_crop)
    confidence = probabilities[0][crop_index]
    
    return predicted_crop, confidence

if __name__ == "__main__":
    print("--- Making Sample Crop Prediction ---")
    
    # Example input as specified in prompt:
    # N=90, P=42, K=43, temperature=20, humidity=82, ph=6.5, rainfall=202
    sample_inputs = {
        'N': 90,
        'P': 42,
        'K': 43,
        'temperature': 20.0,
        'humidity': 82.0,
        'ph': 6.5,
        'rainfall': 202.0
    }
    
    print("\nInput parameters:")
    for key, val in sample_inputs.items():
        print(f"  - {key}: {val}")
        
    # Perform prediction
    crop, conf = make_prediction(
        sample_inputs['N'],
        sample_inputs['P'],
        sample_inputs['K'],
        sample_inputs['temperature'],
        sample_inputs['humidity'],
        sample_inputs['ph'],
        sample_inputs['rainfall']
    )
    
    print(f"\nRecommended Crop: {crop.capitalize()}")
    print(f"Prediction Confidence: {conf * 100:.2f}%")
