import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import shap
import pickle
from streamlit_gsheets import GSheetsConnection

import xgboost as xgb

def get_data() -> pd.DataFrame:
    return conn.read(
        worksheet="prediction_suggestion_tol",
        ttl=0, 
        usecols=[i for i in range(0,10)]
        ).dropna()

def insert_data(new_data):
    with status:
        st.write('Inserting to Database.')

    old_data = get_data()
    new_data['Predict Time'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    # data = old_data.append(new_data, ignore_index=True)
    data = pd.concat([old_data, new_data], ignore_index=True)

    conn.update(
        worksheet="prediction_suggestion",
        data=data,
    )

def predictImpedance(test_data):

    impedance = modelImpedance.predict(test_data)[0]

    return impedance

def predictTolerance(test_data):

    tolerance = modelTolerance.predict(test_data)[0]

    return tolerance

# Predict Main Page
st.set_page_config(
    page_title="Prediction",
    page_icon="📊",
)

conn = st.connection("gsheets", type=GSheetsConnection)

dataset_impedance = conn.read(
        worksheet="dataset_impedance",
        ttl=0, 
        usecols=[i for i in range(0, 13)]
    ).dropna()

dataset_tolerance = conn.read(
        worksheet="dataset_tolerance",
        ttl=0, 
        usecols=[i for i in range(0, 13)]
    ).dropna()

with open('models/xgb.pkl', 'rb') as file:
    modelImpedance = pickle.load(file) 

with open('models/xgb_tolerance.pkl', 'rb') as file:
    modelTolerance = pickle.load(file)

# Load Dataset Impedance
dataset = pd.DataFrame(dataset_impedance)
dataset_impedance = dataset[modelImpedance.feature_names_in_]

# Load Dataset Tolerance
dataset = pd.DataFrame(dataset_tolerance)
dataset_tolerance = dataset[modelTolerance.feature_names_in_]

st.title('Coil Impedance & Tolerance Prediction')
with st.form(key='input_form'):
    left, right = st.columns(2)
    params = {
        'Inner Diameter Length LV' : int(left.number_input('Inner Diameter Length LV')),
        'Inner Diameter Width LV': int(left.number_input('Inner Diameter Width LV')),
        'Inner Diameter Height LV': int(left.number_input('Inner Diameter Height LV')),

        'Outer Diameter Length LV': int(left.number_input('Outer Diameter Length  LV')),
        'Outer Diameter Width LV': int(left.number_input('Outer Diameter Width LV')),
        'Outer Diameter Height LV': int(left.number_input('Outer Diameter Height LV')),

        'Inner Diameter Length HV': int(right.number_input('Inner Diameter Length HV')),
        'Inner Diameter Width HV': int(right.number_input('Inner Diameter Width HV')),
        'Inner Diameter Height HV': int(right.number_input('Inner Diameter Height HV')),

        'Outer Diameter Length HV': int(right.number_input('Outer Diameter Length HV')),
        'Outer Diameter Width HV': int(right.number_input('Outer Diameter Width HV')),
        'Outer Diameter Height HV': int(right.number_input('Outer Diameter Height HV')),
    }

    submitted = st.form_submit_button(label='Predict Impedance')

suggestion = []

def modify_testing(df, column, adjusted, ori_imp, ori_tol):
    original_value = df.at[0, column]
    df.at[0, column] = original_value + adjusted

    # ! PREDICT IMPEDANCE
    impedance = predictImpedance(df)


    # ! PREDICT TOLERANCE
    tolerance = predictTolerance(df)
    
    if (ori_imp > 0):
        if ((ori_imp - impedance) > 0):
            suggestion.append({
                'Adjusted Column' : column,
                'Adjusted Value' : adjusted,
                'Adjusted' : f"{column}{adjusted}",
                'Difference Impedance' : ori_imp - impedance,
                'Original Impedance' : ori_imp,
                'Adjusted Impedance' : impedance,
            })
    else:
        if ((impedance - ori_imp) > 0):
            suggestion.append({
                'Adjusted Column' : column,
                'Adjusted Value' : adjusted,
                'Adjusted' : f"{column}{adjusted}",
                'Difference Impedance' : impedance - ori_imp,
                'Original Impedance' : ori_imp,
                'Adjusted Impedance' : impedance,
            })

    if (ori_tol > 0):
        if ((ori_tol - tolerance) > 0):
            suggestion.append({
                'Adjusted Column' : column,
                'Adjusted Value' : adjusted,
                'Adjusted' : f"{column}{adjusted}",
                'Difference Tolerance' : ori_tol - tolerance,
                'Original Tolerance' : ori_tol,
                'Adjusted Tolerance' : tolerance,
            })
    else:
        if ((tolerance - ori_tol) > 0):
            suggestion.append({
                'Adjusted Column' : column,
                'Adjusted Value' : adjusted,
                'Adjusted' : f"{column}{adjusted}",
                'Difference Tolerance' : tolerance - ori_tol,
                'Original Tolerance' : ori_tol,
                'Adjusted Tolerance' : tolerance,
            })

    df.at[0, column] = original_value

    return

if submitted:
    status = st.status('Processing...', expanded=True)

    # ! INPUT DATA
    input_data = pd.DataFrame(params, index=[0])
    test_data_imp = input_data[modelImpedance.feature_names_in_]
    test_data_tol = input_data[modelTolerance.feature_names_in_]
    st.subheader('Input Data')
    st.dataframe(test_data_imp)

    with status:
            st.write(f'Calculating Initial Data')

    # ! PREDICT IMPEDANCE
    impedance = predictImpedance(test_data_imp)
    st.write(f'The predicted impedance is {impedance:.2f} ohms')

    # ! PREDICT TOLERANCE
    tolerance = predictTolerance(test_data_tol)
    st.write(f'The predicted tolerance is {tolerance:.2f}')

    st.divider()

    # List of columns to modify
    columns = test_data_imp.columns.tolist()

    for i in range (-10, 11):
        with status:
            st.write(f'Calculating Range: {i:.2f}')

        # Modify each column
        for column in columns:
            modify_testing(test_data_imp, column, i, impedance, tolerance)

    save= pd.DataFrame(suggestion)
    
    st.subheader("Suggestion Impedance")
    st.dataframe(save.style.highlight_max(axis=0))

    insert_data(save)

    with status:
        status.update(label="Process Complete!", state="complete", expanded=False)
        time.sleep(1)



