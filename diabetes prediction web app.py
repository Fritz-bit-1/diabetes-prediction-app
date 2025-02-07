# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 22:46:52 2025

@author: yadav
"""

import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open(r'D:\deploying ML\diabetes_model.sav', 'rb'))

# Function for prediction
def diabetes_prediction(input_data):
    input_data_as = np.asarray(input_data, dtype=float)  # Convert input to float
    input_data_reshape = input_data_as.reshape(1, -1)

    # Make prediction
    prediction = loaded_model.predict(input_data_reshape)

    # Output result
    if prediction[0] == 0:
        return "🟢 The person does NOT have Diabetes."
    else:
        return "🔴 The person HAS Diabetes."

# Main function
def main():
    # Set page config
    st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")

    # Custom CSS for styling
    st.markdown(
        """
        <style>
        .stButton>button {
            background-color: #008CBA;
            color: white;
            font-size: 18px;
            padding: 10px;
            border-radius: 10px;
        }
        .stButton>button:hover {
            background-color: #005f73;
        }
        .stTextInput>label {
            font-size: 16px;
            font-weight: bold;
        }
        .stTitle {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title with emoji
    st.markdown("<h1 style='text-align: center; color: #2E3B55;'>🔍 Diabetes Prediction Web App</h1>", unsafe_allow_html=True)
    st.write("---")

    # Sidebar with instructions
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("This app uses **Machine Learning (SVM)** to predict whether a person has diabetes or not.")
        st.write("**Enter the details and click 'Predict' to check the result.**")
        st.write("---")
        st.write("🔹 **Model**: SVM (Support Vector Machine)")
        st.write("🔹 **Dataset**: Pima Indian Diabetes Dataset")
        st.write("🔹 **Developed with ❤️ using Streamlit**")

    # Creating layout using columns
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
        Glucose = st.text_input("Glucose Level")
        BloodPressure = st.text_input("Blood Pressure")
        SkinThickness = st.text_input("Skin Thickness")
        
    with col2:
        Insulin = st.text_input("Insulin Level")
        BMI = st.text_input("BMI Value")
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
        Age = st.text_input("Age")

    # Prediction button
    diagnosis = ''
    if st.button("🔮 Predict"):
        try:
            diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
            st.success(diagnosis)
        except:
            st.error("⚠️ Please enter valid numeric values for all fields.")

    # Footer
    st.write("---")
    st.markdown("<p style='text-align: center; font-size: 14px;'>🔬 Powered by Machine Learning | Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)

# Run the app
if __name__ == '__main__':
    main()
