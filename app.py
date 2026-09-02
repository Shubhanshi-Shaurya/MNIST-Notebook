import os
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="MNIST Digit Classifier")

st.title("MNIST Handwritten Digit Classifier")
st.write("Draw a digit (0-9) on the canvas or upload an image")

@st.cache_resource
def load_resources():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "classifier.keras")
    classes_path = os.path.join(base_dir, "mnist_classes.txt")

    model = tf.keras.models.load_model(model_path)

    with open(classes_path, "r") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]

    return model, class_names

try:
    model, class_names = load_resources()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


def preprocess_and_predict(image_data, is_canvas=False):
    gray_image = image_data.convert("L")

    if not is_canvas:
        gray_image = ImageOps.invert(gray_image)

    resized_img = gray_image.resize((28, 28))

    img_array = np.array(resized_img, dtype=np.float32) / 255.0

    img_tensor = np.expand_dims(img_array, axis=(0, -1))

    predictions = model.predict(img_tensor)
    pred_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][pred_idx]) * 100

    return class_names[pred_idx], confidence, predictions[0]


tab1, tab2 = st.tabs(["Draw Digit", "Upload Image"])

with tab1:
    st.write("Draw a digit in the box below:")
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Predict Canvas Digit"):
        if canvas_result.image_data is not None:
            canvas_img = Image.fromarray(canvas_result.image_data.astype("uint8"))
            
            if np.max(canvas_result.image_data[:, :, :3]) > 0:
                with st.spinner("Classifying digit..."):
                    label, conf, all_preds = preprocess_and_predict(canvas_img, is_canvas=True)

                st.success(f"**Predicted Digit:** {label}")
                st.info(f"**Confidence:** {conf:.2f}%")
                st.bar_chart(all_preds)
            else:
                st.warning("Please draw a digit before predicting.")

with tab2:
    uploaded_file = st.file_uploader("Upload a digit image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=200)

        if st.button("Predict Uploaded Image"):
            with st.spinner("Classifying..."):
                label, conf, all_preds = preprocess_and_predict(image, is_canvas=False)

            st.success(f"**Predicted Digit:** {label}")
            st.info(f"**Confidence:** {conf:.2f}%")
            st.bar_chart(all_preds)