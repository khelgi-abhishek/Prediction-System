# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 19:03:41 2025

@author: user
"""

import numpy as np
import pickle
import streamlit as st

# Load the trained model
model_path = r'C:/Users/user/Downloads/deploying ml model/trained_model.sav'
try:
    loaded_model = pickle.load(open(model_path, 'rb'))
except FileNotFoundError:
    st.error(f"Error: Model file not found at {model_path}")
    exit()
except Exception as e:
    st.error(f"Error loading model: {e}")
    exit()

# Load the scaler (if saved separately)
scaler_path = r'C:/Users/user/Downloads/deploying ml model/scaler.pkl'
try:
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    st.warning(f"Scaler file not found at {scaler_path}. Using raw input.")
    scaler = None
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    exit()
    
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Standardize data if scaler is available
    if scaler:
        standardized_data = scaler.transform(input_data_reshaped)
    else:
        standardized_data = input_data_reshaped  # Use raw input if scaler is missing

    # Make prediction using the trained model
    try:
        prediction = loaded_model.predict(standardized_data)
    except Exception as e:
        return f"Error during prediction: {e}"

    # Return prediction result
    return 'The person IS diabetic' if prediction[0] == 1 else 'The person is NOT diabetic'
    
def main():
    st.title('Diabetes Prediction Web App')

    # User inputs
    Pregnancies = st.text_input('Number of Pregnancies', '0')
    Glucose = st.text_input('Glucose Level', '0')
    BloodPressure = st.text_input('Blood Pressure Value', '0')
    SkinThickness = st.text_input('Skin Thickness Value', '0')
    Insulin = st.text_input('Insulin Level', '0')
    BMI = st.text_input('BMI Value', '0')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function', '0')
    Age = st.text_input('Age of the Person', '0')

    diagnosis = ''
    
    if st.button('Diabetes Test Result'):
        try:
            # Convert inputs to float before passing to function
            input_values = [float(Pregnancies), float(Glucose), float(BloodPressure),
                            float(SkinThickness), float(Insulin), float(BMI),
                            float(DiabetesPedigreeFunction), float(Age)]
            
            diagnosis = diabetes_prediction(input_values)
        except ValueError:
            st.error("Please enter valid numerical values.")

    st.success(diagnosis)
    
if __name__ == "__main__":
    main()
