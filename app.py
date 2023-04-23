import os
import sys
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np

unique_labels = [0,1]

# preprocess image
IMG_SIZE = 224
#batch size
BATCH_SIZE = 32


import base64

def preprocess(image,img_size= IMG_SIZE):
    # decodes images
    image = tf.io.decode_jpeg(image,channels=3)
    image = tf.image.convert_image_dtype(image,tf.float32)
    image = tf.image.resize(image, [img_size,img_size])
    # Return preprocessed image
    return image


#load model
hub_layer = hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5")
model = tf.keras.models.load_model('C:/Users/user/brain_tumor/model/20230421-211629-Brain_tumor_model.h5', custom_objects={'KerasLayer': hub_layer})

# creating streamlit app

st.title('BRAIN TUMOR DETECTION APP')

uploader_file = st.file_uploader('Upload MRI scan', type=['png','jpg','jpeg','jfif'], accept_multiple_files=False)

if uploader_file is not None:
    # Read the image from the in-memory file
    # Use the read method to get the contents of the file as bytes
    
    try:
        img = uploader_file.read()
    except:
        st.error('failed to reading in image')
        st.stop()
    #process files
    data = tf.data.Dataset.from_tensor_slices(tf.expand_dims(img, axis=0))
    data_batch = data.map(preprocess).batch(BATCH_SIZE)
    st.image(img,caption='MRI SCAN',width=500)
    
    button = st.button('Predict')
    if button :
        predict = model.predict(data_batch)
        pred_label = unique_labels[np.argmax(predict)]
        if pred_label == 0 :
            st.success(f'NO TUMOR  ===== {round(np.amax(predict)*100)}%') 
        else:
            st.success(f'TUMOR ==== {round(np.amax(predict)*100)}%')
    

