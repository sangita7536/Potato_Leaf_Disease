import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os

file_id = "1wAangUK3NGUJ_OR0_D6CLN_sgDmaw00m"
url = "https://drive.google.com/file/d/1wAangUK3NGUJ_OR0_D6CLN_sgDmaw00m/view?usp=sharing"
model_path = "trained_plant_disease_model.keras"

if not os.path.exists(model_path):
    st.warning("Downloding model from drive...")
    gdown.download(url,model_path,quite=False)

model_path = "trained_plant_disease_model.keras"
def model_prediction(test_image):
    model= tf.keras.models.load_model(model_path)
    image= tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr]) #convert single imageyo batch
    predictions= model.predict(input_arr)
    return np.argmax(predictions) #return index of max element


st.sidebar.title("Plant Disease system for Sustainable Agriculture")
app_mode = st.sidebar.selectbox('select page',['Home','Disease Recognition'])

from PIL import Image
img= Image.open('diseases.png')
st.image(img)

if(app_mode=='HOME'):
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture", unsafe_allow_html=True)

elif(app_mode=='Disease Recognition'):
    st.header('Plant Disease Detection System For Sustainable Agriculture')

test_image= st.file_uploader('Choose an image:')
if(st.button('Show Image')):
    st.image(test_image,width=4,use_container_width=True)

if (st.button('Predict')):
    st.snow()
    st.write('our prediction')
    result_index = model_prediction(test_image)
    class_name=['Potato___Early_blight','Potato___Late_blight','Potato___healthy']
    st.success('Model is predicting its a {}'.format(class_name[result_index]))

