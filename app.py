import streamlit as st
import numpy as np
import pickle

# Load the model and scaler
rfc = pickle.load(open('rfc_classifier.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define prediction function
def predict(model, scaler, male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose):
    # Encode categorical variables
    male_encoded = 1 if male.lower() == "male" else 0
    currentSmoker_encoded = 1 if currentSmoker.lower() == "yes" else 0
    BPMeds_encoded = 1 if BPMeds.lower() == "yes" else 0
    prevalentStroke_encoded = 1 if prevalentStroke.lower() == "yes" else 0
    prevalentHyp_encoded = 1 if prevalentHyp.lower() == "yes" else 0
    diabetes_encoded = 1 if diabetes.lower() == "yes" else 0

    # Prepare features array
    features = np.array([[male_encoded, age, currentSmoker_encoded, cigsPerDay, BPMeds_encoded,
                          prevalentStroke_encoded, prevalentHyp_encoded, diabetes_encoded,
                          totChol, sysBP, diaBP, BMI, heartRate, glucose]])

    # Scale the features
    scaled_features = scaler.transform(features)

    # Predict using the model
    result = model.predict(scaled_features)

    return result[0]

# Streamlit App
st.title('Heart Disease Prediction')

# Image display
st.image('img.jpg', caption='Heart Disease Analyzer', use_container_width=True)

# Input form
male = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 20, 100, 56)
currentSmoker = st.selectbox('Do you currently smoke?', ['Yes', 'No'])
cigsPerDay = st.number_input('Cigarettes per Day', min_value=0.0, max_value=100.0, value=0.0)
BPMeds = st.selectbox('On Blood Pressure Medication?', ['Yes', 'No'])
prevalentStroke = st.selectbox('History of Stroke?', ['Yes', 'No'])
prevalentHyp = st.selectbox('Hypertension?', ['Yes', 'No'])
diabetes = st.selectbox('Diabetes?', ['Yes', 'No'])
totChol = st.number_input('Total Cholesterol', min_value=100.0, max_value=600.0, value=200.0)
sysBP = st.number_input('Systolic Blood Pressure', min_value=90.0, max_value=250.0, value=120.0)
diaBP = st.number_input('Diastolic Blood Pressure', min_value=60.0, max_value=140.0, value=80.0)
BMI = st.number_input('BMI', min_value=10.0, max_value=50.0, value=26.0)
heartRate = st.number_input('Heart Rate', min_value=40, max_value=200, value=70)
glucose = st.number_input('Glucose Level', min_value=50.0, max_value=300.0, value=100.0)

# Prediction
if st.button('Predict'):
    result = predict(rfc, scaler, male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp,
                     diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose)

    if result == 1:
        st.error("⚠️ The patient is likely to have heart disease. Please consult a cardiologist.")
    else:
        st.success("✅ No heart disease detected. Stay healthy!")
