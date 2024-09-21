import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("C:\\Users\\pshri\\Downloads\\fpretrained_gender_classification_model.h5")

# Define the labels (male/female)
labels = {0: 'Female', 1: 'Male'}

# Streamlit app title and instructions
st.title('Gender Classification App')
st.write('Upload an image, and the model will predict the gender (Male or Female).')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image to match model input size
    img = img.resize((128, 128))  # Resize to model's expected input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    # Make prediction
    prediction = model.predict(img_array)
    prediction_class = 1 if prediction > 0.5 else 0  # Threshold for binary classification

    # Display result
    st.write(f'Prediction: {labels[prediction_class]}')
