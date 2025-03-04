# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""





# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# Load the trained model
model_path = r'C:/Users/user/Downloads/deploying ml model/trained_model.sav'
try:
    loaded_model = pickle.load(open(model_path, 'rb'))
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load the scaler (if saved separately)
scaler_path = r'C:/Users/user/Downloads/deploying ml model/scaler.pkl'
try:
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    print(f"Warning: Scaler file not found at {scaler_path}. Using raw input.")
    scaler = None
except Exception as e:
    print(f"Error loading scaler: {e}")
    exit()

# Define input data
input_data = (4, 110, 92, 0, 0, 37.6, 0.191, 30)
input_data_as_numpy_array = np.asarray(input_data)
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
    print(f"Error during prediction: {e}")
    exit()

# Print prediction result
print("Prediction:", prediction)
if prediction[0] == 0:
    print('The person is NOT diabetic')
else:
    print('The person IS diabetic')
