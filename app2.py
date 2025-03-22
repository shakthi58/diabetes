import base64
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the trained Random Forest model
model_path = "random_forest_diabetes_model.pkl"
model = joblib.load(model_path)

# Streamlit UI
st.title("Diabetes Prediction App")
st.write("Enter patient details (excluding pregnancies) to predict diabetes risk.")

# Initialize session state for menu navigation
if "menu" not in st.session_state:
    st.session_state.menu = "Preliminary Questions"

menu = st.session_state.menu  # Use session state for navigation


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_background("background.jpeg")

if menu == "Preliminary Questions":
    st.subheader("User Information")
    user_name = st.text_input("What is your name?", key="user_name")
    user_city = st.selectbox("Select your city:", [
        "Bangalore", "Chennai", "Mangalore", "Hyderabad", "Kochi", "Trivandrum"], key="user_city")

    st.subheader("Preliminary Questions")
    q1 = st.radio("1. Do you have increased thirst or frequent urination, especially at night?", [
                  "No", "Occasionally", "Yes"], key="q1")
    q2 = st.radio("2. Have you experienced sudden or unexplained weight loss recently?", [
                  "No", "Not sure", "Yes"], key="q2")
    q3 = st.radio("3. Do you feel fatigued or tired more often than usual?", [
                  "No", "Sometimes", "Yes"], key="q3")
    q4 = st.radio("4. Does anyone in your immediate family have diabetes?", [
                  "No", "Not sure", "Yes"], key="q4")
    q5 = st.radio("5. What does your diet and physical activity routine look like?", [
                  "Healthy diet and active", "Moderate diet and activity", "Unhealthy diet and sedentary"], key="q5")

    if st.button("Submit Answers"):
        st.success(
            f"Thank you {user_name} for answering the preliminary questions! You will now be redirected to the next step.")

        # Automatically switch to Patient Details
        st.session_state.menu = "Patient Details"
        st.rerun()  # Refresh UI to show the next section

elif menu == "Patient Details":
    st.subheader("Patient Details")
    glucose = st.number_input(
        "Glucose Level", min_value=0, max_value=300, value=100)
    blood_pressure = st.number_input(
        "Blood Pressure", min_value=0, max_value=200, value=80)
    skin_thickness = st.number_input(
        "Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=30)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input("Age", min_value=0, max_value=120, value=30)

    # Prediction button
    if st.button("Predict"):
        input_data = np.array(
            [[glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        if prediction[0] == 1:
            st.error(
                f"The model predicts that the patient is at risk of diabetes. (Probability: {probability:.2%})")
        else:
            st.success(
                f"The model predicts that the patient is not at risk of diabetes. (Probability: {probability:.2%})")

        risk_percentage = probability * 100

        # Health Badge
        if risk_percentage < 10:
            st.success("🏆 *Health Badge: Low Risk Champion!*")
        elif risk_percentage < 50:
            st.warning("🎖 *Health Badge: Moderate Risk Warrior!*")
        else:
            st.error("⚠ *Health Badge: High Risk Fighter!*")

        # Checklist
        st.write("### Your Health Checklist:")
        st.write("- 🥗 Eat a balanced diet with fewer carbs.")
        st.write("- 🏃‍♂ Exercise for at least 30 minutes daily.")
        st.write("- 💧 Stay hydrated and avoid sugary drinks.")

        # Health and diet recommendations based on glucose levels
        st.subheader("Health Recommendations:")
        if glucose <= 140:
            st.success("Normal glucose levels detected.")
            st.write("### Diet Recommendations for Normal Glucose Levels:")
            st.write(
                "- Maintain a balanced diet with vegetables, whole grains, and lean proteins.")
            st.write("- Stay physically active to keep glucose levels stable.")
        elif 140 < glucose <= 180:
            st.warning("Pre-Diabetes glucose levels detected.")
            st.write("### Diet Recommendations for Pre-Diabetes:")
            st.write(
                "- Reduce carbohydrate intake and focus on high-fiber foods like oats and lentils.")
            st.write("- Incorporate more healthy fats, such as nuts and olive oil.")
            st.write("- Avoid sugary drinks and processed snacks.")
        else:
            st.error("Diabetic glucose levels detected. Please consult a doctor.")
            st.write("### Diet Recommendations for Diabetic Glucose Levels:")
            st.write(
                "- Prioritize low glycemic index foods such as green leafy vegetables and legumes.")
            st.write("- Avoid refined carbs and sugary beverages entirely.")
            st.write("- Monitor portion sizes and meal timings strictly.")

        # Health and diet recommendations based on BMI
        if bmi < 18.5:
            st.warning("Underweight BMI detected.")
            st.write("### Diet Recommendations for Underweight BMI:")
            st.write(
                "- Include calorie-dense, nutritious foods like nuts, seeds, and avocados.")
            st.write("- Eat more frequently and include protein-rich foods.")
            st.image("protein.jpeg")
        elif 18.5 <= bmi < 24.9:
            st.success("Normal BMI detected.")
            st.write("### Diet Recommendations for Normal BMI:")
            st.write(
                "- Maintain a balanced diet with a mix of carbs, proteins, and healthy fats.")
            st.write("- Continue regular physical activity.")
            st.image("protein.jpeg")
        elif 25 <= bmi < 30:
            st.warning("Overweight BMI detected.")
            st.write("### Diet Recommendations for Overweight BMI:")
            st.write(
                "- Focus on low-calorie, nutrient-dense foods like vegetables and lean proteins.")
            st.write("- Limit sugar and saturated fat intake.")
            st.image("protein.jpeg")
        else:
            st.error("Obese BMI detected. Immediate action is recommended.")
            st.write("### Diet Recommendations for Obese BMI:")
            st.write("- Consult a dietitian to create a structured meal plan.")
            st.write(
                "- Emphasize low-calorie foods and avoid processed foods entirely.")

        # Health and diet recommendations based on blood pressure
        if blood_pressure < 90:
            st.warning("Low blood pressure detected.")
            st.write("### Diet Recommendations for Low Blood Pressure:")
            st.write("- Include more salt in your diet but within healthy limits.")
            st.write("- Stay hydrated and eat small, frequent meals.")
        elif 90 <= blood_pressure <= 120:
            st.success("Normal blood pressure detected.")
            st.write("### Diet Recommendations for Normal Blood Pressure:")
            st.write(
                "- Maintain a diet rich in fruits, vegetables, and whole grains.")
            st.write("- Stay active and avoid excessive sodium intake.")
            st.image("vegetables.webp")
        elif 120 < blood_pressure <= 140:
            st.warning("Elevated blood pressure detected.")
            st.write("### Diet Recommendations for Elevated Blood Pressure:")
            st.write("- Reduce salt intake and avoid high-sodium foods.")
            st.write(
                "- Include potassium-rich foods like bananas and sweet potatoes.")
            st.image("carbohydrates.jpg")
        else:
            st.error(
                "High blood pressure detected. Immediate action is recommended.")
            st.write("### Diet Recommendations for High Blood Pressure:")
            st.write(
                "- Follow a DASH diet (Dietary Approaches to Stop Hypertension).")
            st.write("- Avoid salty snacks, processed meats, and sugary drinks.")

        # Health and diet recommendations based on skin thickness
        if skin_thickness < 10:
            st.warning(
                "Low skin thickness detected. Potential nutritional concerns.")
            st.write("### Diet Recommendations for Low Skin Thickness:")
            st.write(
                "- Increase protein intake through lean meats, eggs, and legumes.")
            st.write(
                "- Ensure adequate intake of vitamins A and C through carrots, oranges, and spinach.")
            st.image("vegetables.webp")
        elif skin_thickness > 50:
            st.warning("High skin thickness detected. Monitor closely.")
            st.write("### Diet Recommendations for High Skin Thickness:")
            st.write(
                "- Focus on maintaining a healthy diet and consult a healthcare provider if concerned.")

        # Health and diet recommendations based on insulin levels
        if insulin < 16:
            st.warning("Low insulin levels detected.")
            st.write("### Diet Recommendations for Low Insulin Levels:")
            st.write("- Include complex carbohydrates like brown rice and quinoa.")
            st.write("- Incorporate more healthy fats and proteins into meals.")
            st.image("healthy fats.jpg")
        elif insulin > 166:
            st.warning("High insulin levels detected.")
            st.write("### Diet Recommendations for High Insulin Levels:")
            st.write("- Avoid refined carbohydrates and sugary foods.")
            st.write("- Focus on high-fiber vegetables and lean proteins.")
            st.image("vegetables.webp")

        # Health and diet recommendations based on diabetes pedigree function
        if diabetes_pedigree < 0.5:
            st.success("Low Diabetes Pedigree Function detected.")
            st.write("### General Health Tips:")
            st.write("- Maintain a balanced diet and regular exercise routine.")
            st.image("vegetables.webp")
        elif diabetes_pedigree >= 0.5:
            st.warning(
                "High Diabetes Pedigree Function detected. Increased risk of diabetes.")
            st.write(
                "### Diet Recommendations for High Diabetes Pedigree Function:")
            st.write("- Prioritize whole foods, avoiding processed options.")
            st.write("- Monitor carbohydrate intake carefully.")
            st.image("carbohydrates.jpg")

            st.subheader("Feature Importance in Prediction")
            importances = model.feature_importances_
            features = ["Glucose", "Blood Pressure", "Skin Thickness",
                        "Insulin", "BMI", "Diabetes Pedigree", "Age"]

            fig, ax = plt.subplots()
            ax.barh(features, importances, color="skyblue")
            ax.set_xlabel("Importance Score")
            ax.set_title("Feature Importance")

            st.pyplot(fig)
