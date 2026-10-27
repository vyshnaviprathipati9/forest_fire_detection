
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
st.set_page_config(
    page_title="Forest Fire Detection",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🌲 Forest Fire Detection")
st.write("Upload an image to check the CNN model's prediction.")

MODEL_PATH = Path("models/forest_fire_cnn.keras")

if not MODEL_PATH.exists():
    st.error("Model not found. Run python train_model.py first.")
    st.stop()


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()
uploaded_file = st.file_uploader(
    "Choose a forest image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Prepare image and predict
    resized_image = image.resize((128, 128))
    image_array = np.array(resized_image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    prediction = float(
        model.predict(image_array, verbose=0)[0][0]
    )

    # Display image on the left and prediction on the right
    col1, col2 = st.columns([1, 1], gap="small")

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, width=500)

    with col2:
        st.subheader("Prediction")

        if prediction < 0.5:
            confidence = (1 - prediction) * 100
            st.error("🔥 Fire detected")
        else:
            confidence = prediction * 100
            st.success("🌳 No fire detected")

        st.metric("Model Confidence", f"{confidence:.2f}%")
    