# MNIST Handwritten Digit Classifier 

An interactive, full-stack machine learning web application built with **Streamlit**, **TensorFlow / Keras**, and **Streamlit-Drawable-Canvas**. This project allows users to either interactively draw a digit on a live canvas or upload an image file (PNG/JPG) to get real-time digit classifications alongside full confidence probability distributions.

---

## Features

* **Interactive Drawing Canvas:** Integrated `streamlit-drawable-canvas` enabling users to draw digits (0–9) directly in their web browser.
* **Image Upload Support:** Upload handwritten digit images with automatic RGB-to-grayscale conversion, background inversion, and dimension standardization.
* **Real-Time CNN Inference:** Pretrained Convolutional Neural Network processing normalized $28 \times 28$ grayscale tensors.
* **Probability Visualizations:** Interactive probability distribution bar charts displaying the confidence score across all 10 digit classes.
* **Resource Optimization:** Utilizes Streamlit's `@st.cache_resource` for instant, cached model loading in memory.

---

## Project Architecture & Directory Structure

```text
mnist_streamlit_app/
├── app.py                   # Streamlit web application & inference pipeline
├── mnist_cnn_model.keras    # Exported trained Keras CNN model weights
├── mnist_classes.txt        # Class mapping file (digits 0 through 9)
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation

```

---

## Tech Stack
1. Framework: Streamlit
2. Deep Learning Engine: TensorFlow / Keras (CNN)
3. Image Processing: Pillow (PIL), NumPy
4. Interactive Canvas: streamlit-drawable-canvas

---

## Getting Started
1. Clone the Repository

```text
git clone [https://github.com/Shubhanshi-Shaurya/MNIST-Notebook](https://github.com/Shubhanshi-Shaurya/MNIST-Notebook)
cd mnist-digit-classifier

```

2. Set Up a Virtual Environment 
```text
# Windows
python -m venv .venv
.venv\Scripts\activate

```

3. Install Dependencies
```text 
pip install -r requirements.txt
```

4. Run the Streamlit Application
```text
streamlit run app.py
```

The web dashboard will automatically launch in your default browser at http://localhost:8501.

