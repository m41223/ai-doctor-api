from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

model = tf.keras.models.load_model("skin_disease_model.h5")

@app.get("/")
def home():
    return {"message":"API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image = image.resize((224,224))
    img = np.array(image)/255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return {"prediction": prediction.tolist()}
