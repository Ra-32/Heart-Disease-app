import streamlit as st
import pandas as pd
import pickle

# Load trained pipeline using pickle
with open('model_pipeline.pkl', 'rb') as f:
    pipe = pickle.load(f)

# Title
st.title("Heart Disease Prediction App By RK ❤️")

# Sidebar input form
st.sidebar.header("Enter Patient Details")

def user_input():
    Age = st.sidebar.slider('Age', 18, 100, 50)
    Sex = st.sidebar.selectbox('Sex', ['M', 'F'])
    ChestPainType = st.sidebar.selectbox('Chest Pain Type', ['TA', 'ATA', 'NAP', 'ASY'])
    RestingBP = st.sidebar.number_input('Resting Blood Pressure', 80, 200, 120)
    Cholesterol = st.sidebar.number_input('Cholesterol', 100, 600, 200)
    FastingBS = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
    RestingECG = st.sidebar.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
    MaxHR = st.sidebar.slider('Max Heart Rate', 60, 202, 150)
    ExerciseAngina = st.sidebar.selectbox('Exercise-induced Angina', ['Y', 'N'])
    Oldpeak = st.sidebar.slider('Oldpeak', 0.0, 6.0, 1.0)
    ST_Slope = st.sidebar.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

    # Build dataframe from input
    data = {
        'Age': [Age],
        'Sex': [Sex],
        'ChestPainType': [ChestPainType],
        'RestingBP': [RestingBP],
        'Cholesterol': [Cholesterol],
        'FastingBS': [FastingBS],
        'RestingECG': [RestingECG],
        'MaxHR': [MaxHR],
        'ExerciseAngina': [ExerciseAngina],
        'Oldpeak': [Oldpeak],
        'ST_Slope': [ST_Slope]
    }

    return pd.DataFrame(data)

# Get user input
input_df = user_input()

# Show the input data
st.subheader('Patient Input')
st.write(input_df)

# Predict button
if st.button('Predict'):
    prediction = pipe.predict(input_df)[0]
    proba = pipe.predict_proba(input_df)[0]

    st.subheader('Prediction')
    st.write("🚨 Heart Disease Detected" if prediction == 1 else "✅ No Heart Disease Detected")

    st.subheader("Prediction Probability")
    st.write(f"Probability of Heart Disease: {proba[1]*100:.2f}%")


