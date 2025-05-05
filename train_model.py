import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# 🔹 Load dataset with optimized dtypes
df = pd.read_csv("dataset_medical_complet.csv", dtype={
    'ECG (bpm)': 'float32',
    'Température corporelle (°C)': 'float32',
    'SpO2 (%)': 'float32',
    'Pression artérielle (systolique)': 'float32',
    'Pression artérielle (diastolique)': 'float32',
    'Glucose (mg/dL)': 'float32',
    'Rythme respiratoire (rpm)': 'float32',
    'Température ambiante (°C)': 'float32',
    'Activité': 'category',
    'Humidité de la peau': 'category',
    'Niveau de stress': 'category',
    'ECG irrégulier': 'bool',
    'Problèmes cutanés': 'bool',
    'Maladie prédite': 'string'
})

# 🔹 Clean and preprocess data
df = df.dropna()  # Remove missing values

# Encode categorical features
categorical_cols = ['Activité', 'Humidité de la peau', 'Niveau de stress']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare features (X)
X = df[['ECG (bpm)', 'Température corporelle (°C)', 'SpO2 (%)', 'Pression artérielle (systolique)',
        'Pression artérielle (diastolique)', 'Glucose (mg/dL)', 'Rythme respiratoire (rpm)',
        'Température ambiante (°C)', 'Activité', 'Humidité de la peau', 'Niveau de stress']]

# 🔹 Handle multi-label target
# Split the 'Maladie prédite' column into a list of diseases
df['Maladie prédite'] = df['Maladie prédite'].apply(lambda x: x.split(', '))
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['Maladie prédite'])
disease_classes = mlb.classes_
print(f"Nombre de maladies uniques : {len(disease_classes)}")
print(f"Maladies : {disease_classes}")

# 🔹 Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🚀 Train a multi-label Random Forest model
base_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model = MultiOutputClassifier(base_rf, n_jobs=-1)

# Training with progress bar
with tqdm(total=100, desc="Entraînement du modèle multi-label", dynamic_ncols=True) as pbar:
    model.fit(X_train, y_train)
    pbar.update(100)

# 📊 Evaluate the model
y_pred = model.predict(X_test)

# Accuracy per label and overall subset accuracy
subset_accuracy = accuracy_score(y_test, y_pred)  # Exact match ratio
print(f"✅ Subset Accuracy (exact match) : {subset_accuracy:.4f}")

# Detailed report per disease
print("Rapport de classification par maladie :")
print(classification_report(y_test, y_pred, target_names=disease_classes, zero_division=0))

# 💾 Save the model and encoders
joblib.dump(model, "disease_prediction_multi_label.pkl")
joblib.dump(mlb, "multi_label_binarizer.pkl")
joblib.dump(label_encoders, "categorical_encoders.pkl")

print("🎉 Modèle multi-label entraîné et sauvegardé dans 'ml_model/disease_prediction_multi_label.pkl'")