import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("models/mnist_ann_model.h5")

st.title("MNIST Digit Classifier (ANN)")
st.write("Upload a handwritten digit image (0–9)")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width=250)

    image = image.convert("L")
    image = image.resize((28, 28))

    img_array = np.array(image)

    # Invert colors for MNIST
    img_array = 255 - img_array

    img_array = img_array / 255.0

    # Flatten to 784 features
    img_array = img_array.reshape(1, 784)

    if st.button("Predict Digit"):

        prediction = model.predict(img_array)

        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.subheader("Prediction Result")
        st.write(f"Predicted Digit: {digit}")
        st.write(f"Confidence: {confidence:.2f}%")