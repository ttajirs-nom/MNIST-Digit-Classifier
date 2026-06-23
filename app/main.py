from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
import uvicorn


app = FastAPI(title="MNIST Digit Classifier API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = load_model('../models/mnist_ann_model.h5')


class DigitInput(BaseModel):
    image: list


@app.get("/")
def home():
    return {"message": "MNIST ANN Classifier API is running!"}


@app.post("/predict")
def predict_digit(data: DigitInput):

    if len(data.image) != 784:
        return {
            "error": "Image must contain exactly 784 pixels."
        }

    img = np.array(data.image).reshape(1, 28, 28)
    img = img.astype('float32') / 255.0

    prediction = model.predict(img)

    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {
        "predicted_digit": predicted_digit,
        "confidence": round(confidence * 100, 2)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)