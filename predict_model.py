import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Chemins des fichiers dans le même répertoire que le script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "disease_prediction_multi_label.pkl")
MLB_PATH = os.path.join(BASE_DIR, "multi_label_binarizer.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "categorical_encoders.pkl")

# Charger le modèle et les encodeurs avec gestion d'erreur
try:
    model = joblib.load(MODEL_PATH)  # MultiOutputClassifier
    mlb = joblib.load(MLB_PATH)      # MultiLabelBinarizer
    label_encoders = joblib.load(ENCODERS_PATH)  # Dictionnaire des LabelEncoders
except Exception as e:
    raise Exception(f"Erreur lors du chargement des fichiers modèle/encodeurs : {str(e)}")

def predict_from_sensor_data(sensor_data):
    """
    Prédire les maladies à partir des données de capteurs et générer un graphique des probabilités.
    
    Args:
        sensor_data (dict): Données brutes des capteurs depuis Firestore.
    
    Returns:
        dict: Données d'entrée, prédictions, probabilités et URL du graphique.
    """
    # Mapper les données avec validation
    mapped_data = {}
    defaults = {
        'ECG (bpm)': 0.0, 'Température corporelle (°C)': 0.0, 'SpO2 (%)': 0.0,
        'Pression artérielle (systolique)': 0.0, 'Pression artérielle (diastolique)': 0.0,
        'Glucose (mg/dL)': 0.0, 'Rythme respiratoire (rpm)': 0.0, 'Température ambiante (°C)': 20.0,
        'Activité': 'Moyenne', 'Humidité de la peau': 'Normale', 'Niveau de stress': 'Modéré'
    }
    for key, default in defaults.items():
        value = sensor_data.get(key.split(' (')[0].lower().replace(' ', ''), default)  # Ex. 'ecg' -> 'ECG (bpm)'
        try:
            mapped_data[key] = float(value) if isinstance(default, float) else str(value)
        except (ValueError, TypeError):
            mapped_data[key] = default

    # Encoder les caractéristiques catégoriques
    for col in ['Activité', 'Humidité de la peau', 'Niveau de stress']:
        le = label_encoders[col]
        try:
            mapped_data[col] = le.transform([mapped_data[col]])[0] if mapped_data[col] in le.classes_ else le.transform([defaults[col]])[0]
        except Exception:
            mapped_data[col] = le.transform([defaults[col]])[0]  # Valeur par défaut en cas d'erreur

    # Créer un DataFrame d'entrée
    input_data = pd.DataFrame([mapped_data])

    # Obtenir les probabilités et les prédictions
    try:
        proba = np.array([clf.predict_proba(input_data)[:, 1] for clf in model.estimators_]).T[0]
        predictions = model.predict(input_data)[0]
    except Exception as e:
        raise Exception(f"Erreur lors de la prédiction : {str(e)}")

    # Appliquer un seuil personnalisé (0.05)
    threshold = 0.05
    custom_predictions = [mlb.classes_[i] for i, p in enumerate(proba) if p >= threshold]

    # Filtrer les probabilités pour la visualisation
    significant_indices = [i for i, p in enumerate(proba) if p > 0.05]
    if not significant_indices:
        significant_indices = np.argsort(proba)[-10:]
    filtered_classes = [mlb.classes_[i] for i in significant_indices]
    filtered_proba = proba[significant_indices]

    # Générer le graphique
    plt.figure(figsize=(12, 6))
    plt.bar(filtered_classes, filtered_proba, color='skyblue')
    plt.axhline(y=0.5, color='red', linestyle='--', label='Seuil par défaut (0.5)')
    plt.axhline(y=threshold, color='green', linestyle='--', label=f'Seuil personnalisé ({threshold})')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Maladies')
    plt.ylabel('Probabilité prédite')
    plt.title('Probabilités significatives des maladies')
    plt.legend()
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    graph_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return {
        'sensor_data': mapped_data,
        'predicted_diseases': list(mlb.classes_[predictions == 1]),
        'custom_diseases': custom_predictions,
        'probabilities': dict(zip(mlb.classes_, proba.tolist())),
        'graph_url': f"data:image/png;base64,{graph_url}"
    }
