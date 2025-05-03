import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Load scaler dan label encoder
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load model tflite
interpreter = tf.lite.Interpreter(model_path="fish_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Streamlit UI
st.title("🎣 Prediksi Berat Ikan")
st.markdown("Isi detail ikan berikut untuk memprediksi beratnya (dalam gram) :")

# Input dari pengguna
species_list = le.classes_.tolist()
species = st.selectbox("Species", species_list)

weight = st.number_input("Weight (cm)", min_value=0.0, value=25.0)
length1 = st.number_input("Length1 (cm)", min_value=0.0, value=25.0)
length2 = st.number_input("Length2 (cm)", min_value=0.0, value=27.0)
length3 = st.number_input("Length3 (cm)", min_value=0.0, value=29.0)
height = st.number_input("Height (cm)", min_value=0.0, value=11.5)
width = st.number_input("Width (cm)", min_value=0.0, value=4.0)

# Prediksi
if st.button("Prediksi Berat"):
    species_encoded = le.transform([species])[0]
    data = np.array([[species_encoded, length1, length2, length3, height, width]])
    data_scaled = scaler.transform(data)

    interpreter.set_tensor(input_details[0]['index'], data_scaled.astype(np.float32))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    st.success(f"🎯 Perkiraan berat ikan : **{prediction:.2f} gram**")