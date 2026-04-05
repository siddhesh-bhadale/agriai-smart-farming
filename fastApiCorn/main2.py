import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = FastAPI()

class_labels = [
    'Cercospora leaf spot (Gray leaf spot)',
    'Common rust',
    'Northern Leaf Blight',
    'healthy'
]

class_preventive_measures = [
    'Hybrids with partial resistance to GLS are available. Ask your seed supplier for these hybrids.A two-year crop rotation away from corn is effective if reduced tillage must be maintained for conservation purposes or a one-year rotation with clean plowing is recommended in fields that have had a problem with the disease.',
    'Use resistant/tolerant sweet corn products. Many sweet corn products have resistance gene that provides nearly complete control. Applying strobilurin-and sterol-inhibiting fungicides as preventive measure.',
    'Management of Northern Leaf Blight can be achieved primarily by using hybrids with resistance, but because resistance may not be complete or may fail, it is advantageous to utilize an integrated approach with different cropping practices and fungicides',
    'Your plant is healthy :)'
]

image_size = [244, 244, 3]  # Update the image size to match the model's expected input

def get_model():
    global model
    model = load_model('./models/maize_disease_detection_new_model.h5')
    print(" * Model loaded!")

def allowed_file(filename: str):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check(path: str):
    img = load_img(path, target_size=(image_size[0], image_size[1]))  # Ensure target size matches model input
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x.astype('float32') / 255
    z = model.predict(x)
    index = np.argmax(z)
    accuracy = int(np.array(z).max() * 100)
    return index, accuracy

get_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file and allowed_file(file.filename):
        # Ensure the directory exists
        os.makedirs('static/images', exist_ok=True)
        
        filename = file.filename
        file_extension = filename.split('.')[-1]
        file_path = os.path.join('static/images', f"testing-image.{file_extension}")
        
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        result = check(file_path)
        disease_name = class_labels[result[0]]
        accuracy = result[1]
        preventive_measure = class_preventive_measures[result[0]]

        response = {
            "disease_name": disease_name,
            "accuracy": accuracy,
            "preventive_measures": preventive_measure
        }
        
        return JSONResponse(content=response)
    else:
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a png, jpg, or jpeg file.")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
