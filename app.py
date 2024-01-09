# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_yU8inxI_D5a8Rq2Rkfv9enY9nD_6H5K
"""

!pip install streamlit

import joblib
import streamlit as st
import pandas as pd

import streamlit as st
from PIL import Image
import os
import pandas as pd
import numpy as np
import joblib

model = joblib.load('RegressionModel.pkl')

st.title('Predict Number of Coins in an Image')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        img_array = np.array(image.convert('L').resize((28, 28)))

        img_flatten = img_array.flatten().reshape(1, -1)

        predicted_coins = model.predict(img_flatten)

        st.write(f"Predicted Number of Coins: {predicted_coins[0]}")


