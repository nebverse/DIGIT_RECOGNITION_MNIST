from tensorflow import keras
from keras import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st
import streamlit_drawable_canvas
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import cv2

# Specify canvas parameters in application
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = "#ffffff"
bg_color =  "#000000"
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line")
)
realtime_update = st.sidebar.checkbox("Update in realtime", True)

model_new = keras.models.load_model('Digit_Recognizer.hdf5')

# Create a canvas component
st.write("DRAW HERE!!")
canvas_result = st_canvas(
    fill_color="#000000",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color= bg_color,
    update_streamlit=realtime_update,
    height=250,width=250,
    drawing_mode=drawing_mode,
    key="canvas",
)
if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (250,250), interpolation=cv2.INTER_NEAREST)
    st.write('Model Input')
    st.image(rescaled)

if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    val = model_new.predict(test_x.reshape(1, 28, 28))
    st.write(f'result: {np.argmax(val[0])}')
    st.bar_chart(val[0])
