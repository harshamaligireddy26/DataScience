# streamlit_app.py
import streamlit as st
import pickle
import numpy as np

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter passenger details to predict if they would survive.")

# Input features
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ['Male', 'Female'])
age = st.slider("Age", 1, 80, 25)

# Map inputs
sex_val = 0 if sex == 'Male' else 1

# Prediction
if st.button("Predict"):
    features = np.array([[pclass, sex_val, age]])
    prediction = model.predict(features)[0]
    outcome = "🟢 Survived" if prediction == 1 else "🔴 Did Not Survive"
    st.success(f"The prediction is: **{outcome}**")

