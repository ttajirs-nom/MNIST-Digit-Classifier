import streamlit as st
import requests
from PIL import Image
import numpy as np

st.title("MNIST Digit Classifier (ANN)")

st.write("Upload a handwritten digit image (0–9)")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = image.convert("L")
    image = image.resize((28, 28))

    img_array = np.array(image)

    # invert colors (VERY IMPORTANT for MNIST)
    img_array = 255 - img_array

    img_array = img_array / 255.0

    # Flatten
    img_array = img_array.flatten().tolist()

    if st.button("Predict Digit"):

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"image": img_array}
        )

        result = response.json()

        st.subheader("Prediction Result")

        st.write(result)