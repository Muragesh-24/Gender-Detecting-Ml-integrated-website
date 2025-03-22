import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (You can restrict it later)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load your model
model = tf.keras.models.load_model("gender_classification_model.keras")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # Preprocess image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)
    result = "Female" if prediction[0][0] > 0.5 else "Male"

    return {"prediction": result}
