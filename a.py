import pandas as pd
import numpy as np

# Nombre d'échantillons (200 000 patients)
N_SAMPLES = 10_000  

# Fonctions pour générer des valeurs réalistes
def generate_ecg():
    return np.random.normal(75, 15)  # Battements par minute

def generate_temp_body():
    return np.random.normal(36.5, 0.7)  # °C

def generate_spo2():
    return np.random.normal(98, 2)  # Saturation oxygène %

def generate_bp():
    return (np.random.randint(90, 180), np.random.randint(60, 120))  # Pression artérielle

def generate_resp_rate():
    return np.random.normal(16, 3)  # Respirations par minute

def generate_glucose():
    return np.random.randint(70, 250)  # Glucose sanguin (mg/dL)

def generate_activity():
    return np.random.choice(["Faible", "Moyenne", "Élevée"], p=[0.3, 0.4, 0.3])

def generate_humidity_skin():
    return np.random.choice(["Basse", "Normale", "Élevée"], p=[0.2, 0.6, 0.2])

def generate_temp_ambient():
    return np.random.normal(22, 5)  # °C

def generate_stress_level():
    return np.random.choice(["Faible", "Modéré", "Élevé"], p=[0.4, 0.4, 0.2])

# Fonction pour déterminer la maladie
def determine_disease(ecg, temp, spo2, bp, glucose, resp, activity, humidity, temp_amb, stress):
    diseases = []

    # ECG
    if ecg > 100:
        diseases.append("Tachycardie")
    elif ecg < 50:
        diseases.append("Bradycardie")
    if 120 <= ecg <= 150:
        diseases.append("Arythmie cardiaque")
    if ecg > 160:
        diseases.append("Fibrillation auriculaire")

    # Température corporelle
    if temp > 38:
        diseases.append("Fièvre/Infection")
    if temp > 40:
        diseases.append("Coup de chaleur")
    if temp < 35:
        diseases.append("Hypothermie")

    # SpO2
    if spo2 < 92:
        diseases.append("Hypoxémie")
    if spo2 < 85:
        diseases.append("Insuffisance respiratoire")

    # Pression artérielle
    if bp[0] > 140 or bp[1] > 90:
        diseases.append("Hypertension")
    if bp[0] < 90 or bp[1] < 60:
        diseases.append("Hypotension")
    if bp[0] > 180 or bp[1] > 110:
        diseases.append("Hypertension sévère")

    # Glucose sanguin
    if glucose > 180:
        diseases.append("Diabète")
    elif glucose < 70:
        diseases.append("Hypoglycémie")
    elif 140 <= glucose <= 180:
        diseases.append("Prédiabète")

    # Rythme respiratoire
    if resp > 25:
        diseases.append("Hyperventilation")
    if resp < 10:
        diseases.append("Apnée du sommeil")
    if resp > 30:
        diseases.append("Détresse respiratoire")

    # Activité
    if activity == "Faible":
        diseases.append("Fatigue chronique")

    # Humidité de la peau
    if humidity == "Basse":
        diseases.append("Déshydratation")
    if humidity == "Élevée":
        diseases.append("Hyperhidrose")

    # Température ambiante
    if temp_amb > 35:
        diseases.append("Stress thermique")
    if temp_amb < 10:
        diseases.append("Coup de froid")

    # Stress
    if stress == "Élevé":
        diseases.append("Trouble anxieux généralisé")

    return ", ".join(diseases) if diseases else "Aucune"

# Génération des données
data = {
    "ID": np.arange(1, N_SAMPLES + 1),
    "ECG (bpm)": [generate_ecg() for _ in range(N_SAMPLES)],
    "Température corporelle (°C)": [generate_temp_body() for _ in range(N_SAMPLES)],
    "SpO2 (%)": [generate_spo2() for _ in range(N_SAMPLES)],
    "Pression artérielle (systolique)": [generate_bp()[0] for _ in range(N_SAMPLES)],
    "Pression artérielle (diastolique)": [generate_bp()[1] for _ in range(N_SAMPLES)],
    "Glucose (mg/dL)": [generate_glucose() for _ in range(N_SAMPLES)],
    "Rythme respiratoire (rpm)": [generate_resp_rate() for _ in range(N_SAMPLES)],
    "Activité": [generate_activity() for _ in range(N_SAMPLES)],
    "Humidité de la peau": [generate_humidity_skin() for _ in range(N_SAMPLES)],
    "Température ambiante (°C)": [generate_temp_ambient() for _ in range(N_SAMPLES)],
    "Niveau de stress": [generate_stress_level() for _ in range(N_SAMPLES)]
}

df = pd.DataFrame(data)

# Ajout de la colonne "Maladie prédite"
df["Maladie prédite"] = df.apply(lambda row: determine_disease(
    row["ECG (bpm)"],
    row["Température corporelle (°C)"],
    row["SpO2 (%)"],
    (row["Pression artérielle (systolique)"], row["Pression artérielle (diastolique)"]),
    row["Glucose (mg/dL)"],
    row["Rythme respiratoire (rpm)"],
    row["Activité"],
    row["Humidité de la peau"],
    row["Température ambiante (°C)"],
    row["Niveau de stress"]
), axis=1)

# Sauvegarde du dataset en CSV
df.to_csv("dataset_medical.csv", index=False)

print("✅ Dataset mis à jour et enregistré avec le glucose ajouté ! 📂")
