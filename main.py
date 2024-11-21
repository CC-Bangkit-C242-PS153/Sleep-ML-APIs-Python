from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from firestoredb import store_data
from datetime import datetime
import tensorflow as tf
import numpy as np
import os
import base64
import json

# Membuat aplikasi FastAPI
app = FastAPI()

# Menambahkan middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Sesuaikan dengan domain yang dibutuhkan
    allow_credentials=True,
    allow_methods=["*"],  # Izinkan semua metode HTTP
    allow_headers=["*"],  # Izinkan semua header
)

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

@app.post("/")
async def home(request: Request):
    if model is None:
        raise HTTPException(status_code=500, detail="Model tidak tersedia")
    
    try:
        payload = await request.json()
        print(payload)
        pubsubMessage = decode_base64_json(payload['message']['data'])
        print(pubsubMessage)

        new_data = np.array([
            [
                int(pubsubMessage['data']['gender']),
                int(pubsubMessage['data']['age']),
                float(pubsubMessage['data']['sleepDuration']),
                int(pubsubMessage['data']['qualitySleep']),
                int(pubsubMessage['data']['physicalActivity']),
                int(pubsubMessage['data']['stressLevel']),
                int(pubsubMessage['data']['BMI']),
                int(pubsubMessage['data']['heartRate']),
                int(pubsubMessage['data']['dailySteps']),
                int(pubsubMessage['data']['systolic']),
                int(pubsubMessage['data']['diastolic'])
            ]
        ])

        classResult = ['Normal', 'Sleep Apnea', 'Insomnia']
        createdAt = datetime.now().isoformat()

        prediction = model.predict(new_data)
        print(f"Data:")
        print("Predicted Probabilities:", prediction[0])
        predicted_class = np.argmax(prediction)
        print("Predicted Class:", predicted_class)
        
        result = classResult[predicted_class]
        data = {
            "userId": pubsubMessage["userId"],
            "inferenceId": pubsubMessage["inferenceId"],
            "result": result,
            "createdAt": createdAt,
        }

        store_data(pubsubMessage["userId"], pubsubMessage["inferenceId"], data)
        
        return JSONResponse(
            status_code=201,
            content={
                "status": "Success",
                "statusCode": 201,
                "message": "Successfully to do inference",
                "data": data,
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "status": "Fail to do Inference",
                "statusCode": 400,
                "message": f"Error: {e}",
            }
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get('PORT', 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
