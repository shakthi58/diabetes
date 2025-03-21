import streamlit as st
import joblib
import numpy as np

# Load the model
model_path = "random_forest_diabetes_model.pkl"
model = joblib.load(model_path)

# App title and description
st.title("Diabetes Prediction App")
st.write("Enter patient details to predict diabetes risk.")

# Input fields
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

# Session state to store prediction result
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

if st.button("Predict"):
    # Prepare input data
    input_data = np.array(
        [[glucose, blood_pressure, skin_thickness,
            insulin, bmi, diabetes_pedigree, age]]
    )
    st.session_state.prediction = model.predict(input_data)[0]

# Check prediction result
if st.session_state.prediction == 1:
    st.error("The model predicts that the patient is at risk of diabetes. It is advised that you get medical attention.")

    # Follow-up questions
    st.write("Please answer a few more questions to get tailored advice:")

    meals = st.radio("How many meals do you eat per day?",
                     ("1-2", "3", "4 or more"), key="meals")
    food_type = st.radio(
        "What type of food do you eat more?", ("High in carbs", "Balanced diet", "High in protein/fats"), key="food_type"
    )
    activity_level = st.radio(
        "Are you regularly active?", ("Not active", "Moderately active", "Highly active"), key="activity_level"
    )

    if st.button("Get Advice"):
        # Provide advice based on answers
        st.write("**Our Recommendations:**")
        if meals == "1-2":
            st.write(
                "- Try to maintain 3 balanced meals a day to regulate blood sugar levels.")
        elif meals == "4 or more":
            st.write(
                "- Eating too frequently may lead to unnecessary spikes in blood sugar. Aim for 3 main meals and healthy snacks if needed.")

        if food_type == "High in carbs":
            st.write(
                "- Reduce carbohydrate intake and focus on incorporating more proteins and vegetables into your meals.")
        elif food_type == "High in protein/fats":
            st.write(
                "- Ensure you're consuming healthy fats and enough fiber to support overall health.")

        if activity_level == "Not active":
            st.write(
                "- Start with light physical activities like walking for 20-30 minutes daily.")
        elif activity_level == "Moderately active":
            st.write(
                "- Maintain your activity level and consider adding strength training to your routine.")
        elif activity_level == "Highly active":
            st.write("- Great job! Keep up your active lifestyle.")

        st.write("- Stay hydrated and monitor your blood sugar levels regularly.")

elif st.session_state.prediction == 0:
    st.success("The model predicts that the patient is not at risk of diabetes.")
