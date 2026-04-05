from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
import pickle

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Allow additional HTTP methods if needed
    allow_headers=["*"],
)

# Load your TensorFlow model
def load_model():
    return tf.keras.models.load_model("./models/model.h5", compile=False)

MODEL = load_model()

CLASS_NAMES = [
    {"type":"Corn(maize)", "name":"Cercospora leaf spot (Gray leaf spot)"},
    {"type":"Corn(maize)", "name":"Common rust"},
    {"type":"Corn(maize)", "name":"Healthy"},
    {"type":"Peach", "name":"Bacterial spot"},
    {"type":"Peach", "name":"Healthy"},
]

crop_recommendation_model_path = './models/RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

def preprocess_image(image):
    input_shape = MODEL.input_shape[1:3]
    input_data = np.expand_dims(image, axis=0)
    input_data = tf.image.resize(input_data, input_shape)
    return input_data

def predict_tflite(image):
    return MODEL.predict(image)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    input_data = preprocess_image(image)
    
    predictions = predict_tflite(input_data)

    # Convert predictions array to list
    predictions_list = predictions.tolist()
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    
    return {
        'predictions': predicted_class,
        'confidence': confidence,
    }


@app.post('/crop-predict')
async def crop_prediction(
    nitrogen: int = Form(...),
    phosphorous: int = Form(...),
    pottasium: int = Form(...),
    ph: float = Form(...),
    rainfall: float = Form(...),
    city: str = Form(...),
):
    title = 'Harvestify - Crop Recommendation'

    if weather_fetch(city) is not None:
        temperature, humidity = weather_fetch(city)
        data = np.array([[nitrogen, phosphorous, pottasium, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]

        return {'prediction': final_prediction, 'title': title}

    else:
        raise HTTPException(status_code=400, detail='Failed to fetch weather information')


def weather_fetch(city_name: str):
    api_key = 'eda6c14bf3d0d6f911ed16f4faa5cb2a'
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = f"{base_url}appid={api_key}&q={city_name}"
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)

# uvicorn main:app  --reload --host 0.0.0.0 --port 8000

#pip install --upgrade typing-extensions



# pip install fastapi
# pip install uvicorn
# pip install pillow
# pip install requests
# pip install scikit-learn

# pip uninstall tensorflow
# pip install tensorflow==2.12.0 --upgrade