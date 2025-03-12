from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from predict_model import predict_from_sensor_data

app = Flask(__name__)
CORS(app)  # Enable CORS for Firebase Hosting

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'API de prédiction des maladies à partir des données des capteurs',
        'endpoint': '/predict (POST) - Analyse des données de capteurs'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sensor_data = request.get_json()
        if not sensor_data or not isinstance(sensor_data, dict):
            return jsonify({'error': 'Données invalides ou absentes'}), 400

        result = predict_from_sensor_data(sensor_data)
        response = {
            'predicted_disease': result['predicted_diseases'][0] if result['predicted_diseases'] else "Aucune maladie détectée",
            'possible_conditions': result['custom_diseases'],
            'confidence': max(result['probabilities'].values(), default=0.0) * 100,
            'ai_insight': f"Prédiction basée sur les données mesurées avec une confiance maximale de {max(result['probabilities'].values(), default=0.0) * 100:.2f}%.",
            'graph_url': result['graph_url']
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': f"Erreur lors de la prédiction : {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
