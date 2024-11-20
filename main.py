from flask import Flask, jsonify, request
from flask_cors import CORS
from firestoredb import store_data
from datetime import datetime
import tensorflow as tf
import numpy as np
import uuid
import os
import base64
import json

app = Flask(__name__)
CORS(app)

# Fungsi callback untuk memverifikasi apakah model berhasil dimuat
def model_loaded_callback(success, message):
    if success:
        print("Model berhasil dimuat!")
    else:
        print(f"Model gagal dimuat: {message}")

def decode_base64_json(data):
    # Mendekode data Base64 menjadi bytes
    decoded_bytes = base64.b64decode(data)
    
    # Mengonversi bytes menjadi string (UTF-8) dan kemudian parsing JSON
    decoded_str = decoded_bytes.decode('utf-8')
    return json.loads(decoded_str)

# Coba untuk memuat model dan beri callback
try:
    model = tf.keras.models.load_model('model/model.h5')
    model_loaded_callback(True, "Model berhasil dimuat.")
except Exception as e:
    model_loaded_callback(False, str(e))
    model = None

@app.route("/", methods=["POST"])
def home():
    if model is None:
        return jsonify({"error": "Model tidak tersedia"}), 500
    try:
        payload = request.get_json()
        pubsubMessage = decode_base64_json(payload['message']['data'])
        newId = uuid.uuid4()
        new_data = np.array([
            [
            pubsubMessage['gender'],
            pubsubMessage['age'],
            pubsubMessage['sleepDuration'], 
            pubsubMessage['qualitySleep'], 
            pubsubMessage['physicalActivity'], 
            pubsubMessage['stressLevel'], 
            pubsubMessage['BMI'], 
            pubsubMessage['heartRate'], 
            pubsubMessage['dailySteps'], 
            pubsubMessage['systolic'], 
            pubsubMessage['diastolic']
            ]
        ])
        classResult = ['Normal','Sleep Apnea','Insomnia']
        createdAt = datetime.now().isoformat()
        prediction = model.predict(new_data)
        print(f"Data:")
        print("Predicted Probabilities:", prediction[0])  
        predicted_class = np.argmax(prediction)  
        print("Predicted Class:", predicted_class)
        result = classResult[predicted_class]
        data = {
                "userId":pubsubMessage["userId"],
                "inferenceId":str(newId),
                "result":result,
                "createdAt":createdAt,
            }
        store_data(pubsubMessage["userId"],str(newId),data)
        return jsonify({
            "status":"Success",
            "statusCode":201,
            "message":"Successfully to do inference",
            "data":data
        })
    except Exception as e:
        return jsonify({
            "status":"Fail to do Inference",
            "statusCode":400,
            "message":f"Error: {e}",
        })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0',port=port)
    app.run(debug=True)
