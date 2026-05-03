from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import math
import pickle
import os

app = Flask(__name__)
CORS(app)

# Singleton Pattern for Model Loading
class ModelService:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            print("Loading Production Shot Quality Model (139MB)...")
            # Path for HF Space/Docker environment
            model_path = os.path.join(os.path.dirname(__file__), 'shot_quality_model.pickle')
            if not os.path.exists(model_path):
                 # Fallback for local testing if path differs
                 model_path = 'shot_quality_model.pickle'
            
            with open(model_path, 'rb') as f:
                cls._model = pickle.load(f)
        return cls._model

def calculate_features(x, y, shot_type):
    # 1. Shot Distance (Euclidean to 100, 50)
    dist = np.sqrt((100 - x) ** 2 + (50 - y) ** 2)
    
    # 2. Shot Angle (Law of Cosines with posts at 100,46 and 100,54)
    x2, y2 = 100, 46 # Left Post
    x3, y3 = 100, 54 # Right Post
    
    a = np.sqrt((x - x2)**2 + (y - y2)**2)
    b = np.sqrt((x - x3)**2 + (y - y3)**2)
    c = 8 # Distance between posts
    
    denominator = 2 * a * b
    if denominator == 0:
        angle_deg = 0.0
    else:
        cos_angle = (a**2 + b**2 - c**2) / denominator
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = (angle_rad * 180) / np.pi
    
    # 3. Strong Foot Mapping (0: Weak, 1: Headers/Other, 2: Strong)
    mapping = {
        'RightFoot': 2,
        'LeftFoot': 0,
        'Head': 1,
        'OtherBodyPart': 1
    }
    strong_foot = mapping.get(shot_type, 2)
    
    return dist, angle_deg, strong_foot

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        x = float(data.get('x'))
        y = float(data.get('y'))
        # Handle both shot_type and shotType
        shot_type = data.get('shot_type') or data.get('shotType') or 'RightFoot'
        
        dist, angle, strong_foot = calculate_features(x, y, shot_type)
        features = np.array([[dist, angle, strong_foot]])
        
        model = ModelService.get_model()
        probability = model.predict_proba(features)[0][1]

        return jsonify({
            'shot_quality': float(probability),
            'goal_prediction': 1 if probability > 0.5 else 0,
            'features': {
                'distance': float(dist),
                'angle': float(angle),
                'strong_foot': int(strong_foot)
            }
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': ModelService._model is not None})

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'name': 'Shot Quality xG API',
        'status': 'online',
        'endpoints': ['/predict', '/health']
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    # Pre-load model
    ModelService.get_model()
    app.run(host='0.0.0.0', port=port)
