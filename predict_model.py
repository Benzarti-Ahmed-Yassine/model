import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Paths relative to script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "disease_prediction_multi_label.pkl")
MLB_PATH = os.path.join(BASE_DIR, "multi_label_binarizer.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "categorical_encoders.pkl")

# Load model and encoders
try:
    model = joblib.load(MODEL_PATH)
    mlb = joblib.load(MLB_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
except Exception as e:
    raise Exception(f"Erreur lors du chargement des fichiers modèle/encodeurs : {str(e)}")

def predict_from_sensor_data(sensor_data):
    """
    Predict diseases from sensor data and generate a clear probability graph.
    
    Args:
        sensor_data (dict): Raw sensor data from Firestore/React (e.g., {'ecg': 80, 'temperature': 36.6, 'spo2': 98}).
    
    Returns:
        dict: Predictions, probabilities, and graph URL.
    """
    # Define expected features and defaults
    feature_mapping = {
        'ecg': 'ECG (bpm)',
        'temperature': 'Température corporelle (°C)',
        'spo2': 'SpO2 (%)',
        'pression_systolique': 'Pression artérielle (systolique)',
        'pression_diastolique': 'Pression artérielle (diastolique)',
        'glucose': 'Glucose (mg/dL)',
        'rythme_respiratoire': 'Rythme respiratoire (rpm)',
        'temperature_ambiante': 'Température ambiante (°C)',
        'activite': 'Activité',
        'humidite_peau': 'Humidité de la peau',
        'stress': 'Niveau de stress'
    }
    defaults = {
        'ECG (bpm)': 70.0, 'Température corporelle (°C)': 36.6, 'SpO2 (%)': 98.0,
        'Pression artérielle (systolique)': 120.0, 'Pression artérielle (diastolique)': 80.0,
        'Glucose (mg/dL)': 90.0, 'Rythme respiratoire (rpm)': 16.0, 'Température ambiante (°C)': 20.0,
        'Activité': 'Moyenne', 'Humidité de la peau': 'Normale', 'Niveau de stress': 'Modéré'
    }

    # Map and validate input data
    mapped_data = {}
    for input_key, model_key in feature_mapping.items():
        value = sensor_data.get(input_key, defaults[model_key])
        try:
            mapped_data[model_key] = float(value) if isinstance(defaults[model_key], float) else str(value)
        except (ValueError, TypeError):
            mapped_data[model_key] = defaults[model_key]

    # Encode categorical features
    for col in ['Activité', 'Humidité de la peau', 'Niveau de stress']:
        le = label_encoders[col]
        try:
            mapped_data[col] = le.transform([mapped_data[col]])[0] if mapped_data[col] in le.classes_ else le.transform([defaults[col]])[0]
        except Exception:
            mapped_data[col] = le.transform([defaults[col]])[0]

    # Create input DataFrame
    input_data = pd.DataFrame([mapped_data])

    # Get predictions and probabilities
    try:
        proba = np.array([clf.predict_proba(input_data)[:, 1] for clf in model.estimators_]).T[0]  # Probabilities for each class
        predictions = model.predict(input_data)[0]  # Binary predictions
    except Exception as e:
        raise Exception(f"Erreur lors de la prédiction : {str(e)}")

    # Adjust threshold for custom predictions (raised to 0.3 for better specificity)
    threshold = 0.3
    custom_predictions = [mlb.classes_[i] for i, p in enumerate(proba) if p >= threshold]
    predicted_diseases = [mlb.classes_[i] for i, p in enumerate(predictions) if p == 1]

    # Prepare probabilities dictionary
    probabilities = dict(zip(mlb.classes_, proba.tolist()))

    # Generate a clear graph (top 5 probabilities)
    top_n = 5
    top_indices = np.argsort(proba)[-top_n:][::-1]  
    top_classes = [mlb.classes_[i] for i in top_indices]
    top_proba = proba[top_indices]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(top_classes, top_proba, color='teal', edgecolor='black')
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Seuil ({threshold})')
    plt.xlabel('Maladies potentielles', fontsize=12)
    plt.ylabel('Probabilité', fontsize=12)
    plt.title('Top 5 probabilités de maladies', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add probability values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    graph_url = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
    plt.close()

    return {
        'sensor_data': mapped_data,
        'predicted_diseases': predicted_diseases,
        'custom_diseases': custom_predictions,
        'probabilities': probabilities,
        'graph_url': graph_url
    }
