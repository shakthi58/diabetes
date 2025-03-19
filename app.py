import streamlit as st
import joblib
import numpy as np

model_path = "random_forest_diabetes_model.pkl"
model = joblib.load(model_path)

st.title("Diabetes Prediction App")
st.write("Enter patient details to predict diabetes risk.")

glucose = st.number_input("Glucose Level", min_value=0,
                          max_value=300, value=100)
blood_pressure = st.number_input(
    "Blood Pressure", min_value=0, max_value=200, value=80)
skin_thickness = st.number_input(
    "Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=30)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

if st.button("Predict"):
    input_data = np.array(
        [[glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error(
            "The model predicts that the patient is at risk of diabetes. It is advised that you get medical attention.")
    else:
        st.success(
            "The model predicts that the patient is not at risk of diabetes.")
