import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os

file_id = "1wAangUK3NGUJ_OR0_D6CLN_sgDmaw00m"
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "trained_plant_disease_model.keras"

# Download the model if it doesn't exist
if not os.path.exists(model_path):
    st.warning("Downloading model from drive...")
    gdown.download(url, model_path, quiet=False)
    if os.path.exists(model_path):
        st.success("Model downloaded successfully!")
    else:
        st.error("Failed to download the model.")

# Verify the model file exists
if not os.path.exists(model_path):
    st.error(f"Model file not found at: {os.path.abspath(model_path)}")
else:
    st.success(f"Model file found at: {os.path.abspath(model_path)}")

def model_prediction(test_image):
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Streamlit app
st.sidebar.title("Plant Disease System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Disease Recognition'])

from PIL import Image
img = Image.open('diseases.png')
st.image(img)

if app_mode == 'Home':
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

elif app_mode == 'Disease Recognition':
    st.header('Plant Disease Detection System For Sustainable Agriculture')

    test_image = st.file_uploader('Choose an image:', type=['jpg', 'png', 'jpeg'])
    if test_image is not None:
        if st.button('Show Image'):
            st.image(test_image, width=4, use_container_width=True)

        if st.button('Predict'):
            st.snow()
            st.write('Our Prediction')
            # Save the uploaded file temporarily
            with open("temp_image.jpg", "wb") as f:
                f.write(test_image.getbuffer())
            
            # Pass the saved file path to the model_prediction function
            result_index = model_prediction("temp_image.jpg")
            if result_index is not None:
                class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
                st.success(f'Model is predicting it\'s a {class_name[result_index]}')