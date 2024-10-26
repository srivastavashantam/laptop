import streamlit as st
import pickle
import numpy as np

# Load the model and data
with open('pipe.pkl', 'rb') as pipe_file, open('df.pkl', 'rb') as df_file:
    pipe = pickle.load(pipe_file)
    df = pickle.load(df_file)

st.title("Laptop Price Predictor")

# Brand
company = st.selectbox('Brand', df['Company'].unique())

# Type of laptop
laptop_type = st.selectbox('Type', df['TypeName'].unique())

# RAM
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# Screen size
screen_size = st.number_input('Screen Size')

# Resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# CPU
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

# HDD
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU', df['Gpu brand'].unique())

# OS
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    # Calculate PPI
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = (((X_res ** 2) + (Y_res ** 2)) ** 0.5) / screen_size

    # Encode categorical variables
    touchscreen_encoded = 1 if touchscreen == 'Yes' else 0
    ips_encoded = 1 if ips == 'Yes' else 0

    # Create the query
    query = np.array([company, laptop_type, ram, weight, touchscreen_encoded, ips_encoded, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, -1)

    # Make prediction and display result
    predicted_price = np.exp(pipe.predict(query))
    st.title("The predicted price of this configuration is Rs." + str(round(predicted_price[0], 2)))
