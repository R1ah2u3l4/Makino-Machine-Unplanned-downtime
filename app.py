# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:20:30 2023

@author: hp
"""

import pandas as pd
import numpy as np
import joblib, pickle
import streamlit as st

import os
os.getcwd()

impute = joblib.load('medianimpute')
winsor = joblib.load('winsor')
minmax = joblib.load('minmax')
model1= pickle.load(open('xgb.pkl', 'rb'))

# Load data
data = pd.read_csv(r"C:/Users/hp/Desktop/Datasets/data_makino.csv")
data.Downtime = np.where(data.Downtime =='Machine_Failure',1,0)


# Train the model
data1 = data.drop(["Date","Machine_ID","Assembly_Line_No"], axis =1)
data2 = data1.iloc[:,:12]
clean = pd.DataFrame(impute.transform(data2), columns = data2.select_dtypes(exclude = ['object']).columns)
clean1 = pd.DataFrame(winsor.transform(clean),columns=clean.columns)
clean2 = pd.DataFrame(minmax.transform(clean1))
prediction = pd.DataFrame(model1.predict(clean2), columns = ['Downtime'])



# Add user input section
machineid = st.number_input('Enter machine id')
assemblylineno = st.number_input('Enter assembly line no')
Hydralic_pressure = st.number_input('Enter hydralic pressure')
Coolant_pressure = st.number_input('Enter coolant pressure')
Air_System_Pressure = st.number_input('Enter air system pressure')
Coolant_Temperature = st.number_input('Enter coolant temperature')
Hydraulic_Oil_Temperature = st.number_input('Enter hydraulic oil temperature')
Spindle_Bearing_Temperature = st.number_input('Enter spindle bearing temperature')
Spindle_Vibration = st.number_input('Enter splindle vibration')
Tool_Vibration = st.number_input('Enter tool vibraton')
Spindle_Speed = st.number_input('Enter splindle speed')
Voltage = st.number_input('Enter voltage')
Torque = st.number_input('Enter torque')
Cutting = st.number_input('Enter cutting')


# Display prediction
prediction =" "
if st.button('predict'):
    prediction = model1.predict([[ Hydralic_pressure, Coolant_pressure,Air_System_Pressure,Coolant_Temperature,Hydraulic_Oil_Temperature,Spindle_Bearing_Temperature,Spindle_Vibration,Tool_Vibration,Spindle_Speed,Voltage,Torque,Cutting]])
    if prediction[0] == 0:
        st.write('No Machine Failure.')
    else:
        st.write('Machine Failure.')