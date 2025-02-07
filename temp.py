# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle  
import pandas as pd
#loading the save model
loaded_model = pickle.load(open('D:\deploying ML\diabetes_model.sav', 'rb'))
# Sample input data (make sure it matches the number of features)
input_data = (58,1,0,100,234,0,1,156,0,0.1,2,1,3)

# Convert input_data to a Pandas DataFrame to match feature names




# Make prediction
prediction = model.predict(input_data_df)
print(prediction)

# Output result
if prediction[0] == 0:
    print("The person does NOT have heart disease.")
else:
    print("The person HAS heart diseas e.")