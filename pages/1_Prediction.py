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
    return conn.read(ttl=0, usecols=[i for i in range(0, 14)]).dropna()

def insert_data(new_data, impedance, tolerance):
    with status:
        st.write('Inserting to Database.')
        time.sleep(1)

    old_data = get_data()
    new_data['Predict Time'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    new_data['Impedance'] = impedance
    new_data['Tolerance'] = tolerance

    # data = old_data.append(new_data, ignore_index=True)
    data = pd.concat([old_data, new_data], ignore_index=True)

    df = conn.update(
        worksheet=0,
        data=data,
    )
    # st.cache_data.clear()
    # st.experimental_rerun()

def predictImpedance(test_data):
    with status:
        st.write('Calculating Impedance')
        time.sleep(1)

    impedance = modelImpedance.predict(test_data)[0]

    return impedance

def predictTolerance(test_data):
    with status:
        st.write('Calculating Tolerance')
        time.sleep(1)

    tolerance = modelTolerance.predict(test_data)[0]

    return tolerance

def visualize(model, test_data, dataset):

    @st.cache_resource
    def init_shap_js():
        shap.initjs()
    
    with status:
        st.write('Calculating SHAP Values.')
        time.sleep(1)

    # ! SHAP PREPARATION
    init_shap_js()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_data)
    shap_values_dataset = explainer.shap_values(dataset)

    shap_values_df = pd.DataFrame(shap_values, columns=[str(col+' Shap') for col in test_data.columns])
    shap_values_df.index = test_data.index
    st.write(shap_values_df)
    explanation = shap.Explanation(shap_values, data=test_data, feature_names=test_data.columns)

    # ! SHAP PLOT
    shap_plot(shap_values_dataset, dataset, shap_values, test_data, explainer=explainer, explanation=explanation, show=False)
    
    return shap_values, explanation

def shap_plot(shap_val_dataset, dataset_data, shap_val, feature_data, explainer=None, explanation=None, show=True):
    with status:
        st.write('Visualizing Result.')
        time.sleep(1)
    
    summary, bar, waterfall = st.tabs(['Summary', 'Bar', 'Waterfall'])

    # Summary from Dataset
    with summary:
        st.header('Summary Plot')
        plt.figure(figsize=(10, 8))
        
        shap.summary_plot(shap_val_dataset, features=dataset_data, feature_names=dataset_data.columns, show=show)
        
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()  # Membersihkan plot setelah ditampilkan
        plt.close()  # Menutup plot agar tidak ada sisa plot)
        
    # with decision:
    #     st.header('Decision Plot')
    #     shap.decision_plot(explainer.expected_value, shap_val, features=feature_data)
    #     st.pyplot(plt.gcf())
    
    # with force:
    #     st.header('Force Plot')
    #     shap.force_plot(explainer.expected_value, shap_val[0], feature_data.iloc[0], matplotlib=True)
    #     st.pyplot(plt.gcf())

    with bar:
        st.header('Bar Plot')
        print(explanation)
        plt.figure(figsize=(10, 8))

        shap.bar_plot(shap_val[0], feature_data.values[0], feature_data.columns)

        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()  # Membersihkan plot setelah ditampilkan
        plt.close()  # Menutup plot agar tidak ada sisa plot
    
    with waterfall:
        st.header('Waterfall Plot')
        plt.figure(figsize=(10, 8))

        shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_val[0], feature_names=feature_data.columns)
        
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()  # Membersihkan plot setelah ditampilkan
        plt.close()  # Menutup plot agar tidak ada sisa plot

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

st.title('Coil Impedance Prediction')
with st.form(key='input_form'):
    left, right = st.columns(2)
    params = {
        'PID LV': int(left.number_input('PID LV', format='%.0f')),
        'LID LV': int(left.number_input('LID LV', format='%.0f')),
        'TID LV': int(left.number_input('TID LV', format='%.0f')),

        'POD LV': int(left.number_input('POD LV', format='%.0f')),
        'LOD LV': int(left.number_input('LOD LV', format='%.0f')),
        'TOD LV': int(left.number_input('TOD LV', format='%.0f')),

        'PID HV': int(right.number_input('PID HV', format='%.0f')),
        'LID HV': int(right.number_input('LID HV', format='%.0f')),
        'TID HV': int(right.number_input('TID HV', format='%.0f')),

        'POD HV': int(right.number_input('POD HV', format='%.0f')),
        'LOD HV': int(right.number_input('LOD HV', format='%.0f')),
        'TOD HV': int(right.number_input('TOD HV', format='%.0f')),
    }

    submitted = st.form_submit_button(label='Predict Impedance')

if submitted:
    status = st.status('Processing...', expanded=True)

    # Load Dataset Impedance
    dataset = pd.DataFrame(dataset_impedance)
    dataset_impedance = dataset[modelImpedance.feature_names_in_]

    # Load Dataset Tolerance
    dataset = pd.DataFrame(dataset_tolerance)
    dataset_tolerance = dataset[modelTolerance.feature_names_in_]

    # ! INPUT DATA
    input_data = pd.DataFrame(params, index=[0])
    test_data_imp = input_data[modelImpedance.feature_names_in_]
    test_data_tol = input_data[modelTolerance.feature_names_in_]
    st.subheader('Input Data')
    st.dataframe(test_data_imp)

    # ! PREDICT IMPEDANCE
    impedance = predictImpedance(test_data_imp)
    st.subheader('Result')
    st.write(f'The predicted impedance is {impedance:.2f} ohms')

    tolerance = predictTolerance(test_data_tol)
    st.write(f'The predicted tolerance is {tolerance:.2f}')
    
    # ! INSERT DATA TO DB
    insert_data(input_data, impedance, tolerance)

    # ! SHAP VISUALIZATION
    st.subheader('SHAP Visualization Impedance')
    visualize(modelImpedance, test_data_imp, dataset_impedance)
    st.subheader('SHAP Visualization Tolerance')
    visualize(modelTolerance, test_data_tol, dataset_tolerance)

    with status:
        status.update(label="Process Complete!", state="complete", expanded=False)
        time.sleep(1)