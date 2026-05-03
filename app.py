from flask import Flask, request, jsonify
import joblib
import numpy as np
import math

app = Flask(__name__)

# Singleton Pattern for Model Loading
class ModelService:
    _instance = None
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            print("Loading Shot Quality Model...")
            cls._model = joblib.load('shot_quality_model.joblib')
        return cls._model

def calculate_distance(x, y):
    return np.sqrt((100 - x)**2 + (50 - y)**2)

def calculate_angle(x, y):
    return math.atan2(abs(50 - y), (100 - x))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        x = float(data.get('x'))
        y = float(data.get('y'))
        shot_type = data.get('shot_type', 'Other')

        # Feature Engineering
        dist = calculate_distance(x, y)
        angle = calculate_angle(x, y)
        
        mapping = {'RightFoot': 2, 'LeftFoot': 0, 'Head': 1, 'OtherBodyPart': 1}
        strong_foot = mapping.get(shot_type, 2)

        features = np.array([[dist, angle, strong_foot]])
        
        model = ModelService.get_model()
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        return jsonify({
            'shot_quality': float(probability),
            'goal_prediction': int(prediction),
            'features': {
                'distance': float(dist),
                'angle': float(angle)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': ModelService._model is not None})

if __name__ == '__main__':
    # Initialize model on startup
    ModelService.get_model()
    app.run(host='0.0.0.0', port=5000)
