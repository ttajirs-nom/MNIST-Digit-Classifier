import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# --------------------------
# Load Model
# --------------------------

try:
    model = load_model("models/fresh_streamlit.h5", compile=False)
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# --------------------------
# UI
# --------------------------

st.title("MNIST Digit Classifier (ANN)")
st.write("Upload a handwritten digit image (0–9)")

if model_loaded:
    st.success("✅ Model loaded successfully")
else:
    st.error("❌ Model failed to load")
    st.code(model_error)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

# --------------------------
# Prediction
# --------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=250
    )

    image = image.convert("L")
    image = image.resize((28, 28))

    img_array = np.array(image)

    # MNIST needs white digit on black background
    img_array = 255 - img_array

    img_array = img_array / 255.0

    img_array = img_array.reshape(1, 784)

    if st.button("Predict Digit"):

        if not model_loaded:
            st.error("Prediction unavailable because model did not load.")

        else:

            prediction = model.predict(
                img_array,
                verbose=0
            )

            digit = np.argmax(prediction)

            confidence = np.max(prediction) * 100


            st.subheader("Prediction Result")

            st.write(
                f"Predicted Digit: {digit}"
            )

            st.write(
                f"Confidence: {confidence:.2f}%"
            )