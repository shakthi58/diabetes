import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the model
model_path = "random_forest_diabetes_model.pkl"
model = joblib.load(model_path)

# Feature names
feature_names = ["Glucose", "Blood Pressure", "Skin Thickness",
                 "Insulin", "BMI", "Diabetes Pedigree Function", "Age"]

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

    # Precautionary measures for non-diabetic users
    st.write("**Precautionary Measures to Maintain Good Health:**")
    st.write("- Maintain a balanced diet rich in whole grains, vegetables, lean proteins, and healthy fats.")
    st.write("- Engage in regular physical activity, such as walking, jogging, or any form of exercise you enjoy, for at least 30 minutes a day.")
    st.write("- Stay hydrated by drinking plenty of water throughout the day.")
    st.write("- Monitor your weight and aim to maintain a healthy BMI.")
    st.write("- Schedule regular health check-ups, including blood sugar tests, to stay aware of any potential health issues early on.")
    st.write("- Avoid excessive consumption of sugary drinks and processed foods.")
    st.write("- Manage stress through relaxation techniques like meditation, yoga, or deep breathing exercises.")


if st.button("Predict"):
    # Prepare input data
    input_data = np.array(
        [[glucose, blood_pressure, skin_thickness,
            insulin, bmi, diabetes_pedigree, age]]
    )
    st.session_state.prediction = model.predict(input_data)[0]

    # Display feature importance graph
    st.write("### Feature Importance in Diabetes Prediction")
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Plotting the feature importance
    fig, ax = plt.subplots()
    ax.barh(importance_df["Feature"],
            importance_df["Importance"], color="skyblue")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance in Diabetes Prediction")
    plt.gca().invert_yaxis()
    st.pyplot(fig)

    # Explanation of feature importance
    st.write("#### Understanding the Features:")
    st.write("- **Glucose**: High glucose levels are a primary indicator of diabetes, as the condition directly affects the body's ability to regulate blood sugar.")
    st.write("- **BMI (Body Mass Index)**: BMI is a measure of body fat based on height and weight. Higher BMI values are strongly associated with a higher risk of developing diabetes.")
    st.write("- **Age**: As age increases, the risk of diabetes also rises due to reduced insulin sensitivity and lifestyle factors.")
    st.write("- **Diabetes Pedigree Function**: This feature represents the genetic likelihood of diabetes based on family history. A higher value indicates a greater risk.")
    st.write("- **Insulin**: Insulin levels provide insight into how the body is managing blood sugar, with abnormal levels being a concern.")
    st.write("- **Blood Pressure**: High blood pressure is often linked to diabetes and other metabolic disorders.")
    st.write("- **Skin Thickness**: This feature measures subcutaneous fat, which can be a marker for insulin resistance.")
