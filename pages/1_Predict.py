import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import shap
import pickle
# import sqlite3
from streamlit_gsheets import GSheetsConnection

import xgboost as xgb

def get_data() -> pd.DataFrame:
    return conn.read(ttl=0, usecols=[i for i in range(0, 14)]).dropna()

def insert_data(new_data, impedance):
    with status:
        st.write('Inserting to Database.')
        time.sleep(1)
    # with st.connection('history_db', type='sql').session as s:
    #     s.execute(f'''
    #         INSERT INTO history (
    #             pid_lv, lid_lv, tid_lv, pod_lv, lod_lv, tod_lv,
    #             pid_hv, lid_hv, tid_hv, pod_hv, lod_hv, tod_hv, impedance
    #         )
    #         VALUES (
    #             {params['PID LV'][0]}, {params['LID LV'][0]}, {params['TID LV'][0]}, 
    #             {params['POD LV'][0]}, {params['LOD LV'][0]}, {params['TOD LV'][0]},
    #             {params['PID HV'][0]}, {params['LID HV'][0]}, {params['TID HV'][0]},
    #             {params['POD HV'][0]}, {params['LOD HV'][0]}, {params['TOD HV'][0]}, {impedance}
    #         )
    #     ''')
    #     s.commit()
    # query = f'''INSERT INTO History (
    #                 "PID LV", "LID LV", "TID LV", "POD LV", "LOD LV", "TOD LV",
    #                 "PID HV", "LID HV", "TID HV", "POD HV", "LOD HV", "TOD HV", "Impedance"
    #             )
    #             VALUES (
    #                 {data['PID LV'][0]}, {data['LID LV'][0]}, {data['TID LV'][0]}, 
    #                 {data['POD LV'][0]}, {data['LOD LV'][0]}, {data['TOD LV'][0]},
    #                 {data['PID HV'][0]}, {data['LID HV'][0]}, {data['TID HV'][0]},
    #                 {data['POD HV'][0]}, {data['LOD HV'][0]}, {data['TOD HV'][0]}, {data['Impedance'][0]}
    #             ) '''

    old_data = get_data()
    new_data['Predict Time'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    new_data['Impedance'] = impedance

    # data = old_data.append(new_data, ignore_index=True)
    data = pd.concat([old_data, new_data], ignore_index=True)

    df = conn.update(
        worksheet=0,
        data=data,
    )
    # st.cache_data.clear()
    # st.experimental_rerun()

def predict(test_data):
    with status:
        st.write('Calculating Impedance')
        time.sleep(1)

    # model = pd.read_pickle('coil_impedance_model.pkl')
    impedance = model.predict(test_data)[0]

    return impedance

def visualize(model, test_data):

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

    shap_values_df = pd.DataFrame(shap_values, columns=[str(col+' Shap') for col in test_data.columns])
    shap_values_df.index = test_data.index
    st.write(shap_values_df)
    explanation = shap.Explanation(shap_values, data=test_data, feature_names=test_data.columns)

    # ! SHAP PLOT
    shap_plot(shap_values, test_data, explainer=explainer, explanation=explanation, show=False)
    
    return shap_values, explanation

def shap_plot(shap_val, feature_data, explainer=None, explanation=None, show=True):
    with status:
        st.write('Visualizing Result.')
        time.sleep(1)
    
    summary, decision, force, bar, waterfall = st.tabs(['Summary', 'Decision', 'Force', 'Bar', 'Waterfall'])

    with summary:
        st.header('Summary Plot')
        shap.summary_plot(shap_val, features=feature_data, feature_names=feature_data.columns, show=show)
        st.pyplot(plt.gcf())
    
    with decision:
        st.header('Decision Plot')
        shap.decision_plot(explainer.expected_value, shap_val, features=feature_data)
        st.pyplot(plt.gcf())
    
    # with dependence:
    #     st.header('Dependence Plot - Not Compatible')
    #     select1, select2 = st.columns(2)
    #     feature1 = select1.selectbox('Feature 1', options=feature_data.columns, key='dependence_feature1')
    #     feature2 = select2.selectbox('Feature 2', options=feature_data.columns, key='dependence_feature2')

    #     shap.dependence_plot(feature1, interaction_index=feature2, shap_values=shap_val, features=feature_data)
    #     st.pyplot(plt.gcf())
    
    with force:
        st.header('Force Plot')
        shap.force_plot(explainer.expected_value, shap_val[0], feature_data.iloc[0], matplotlib=True)
        st.pyplot(plt.gcf())

    with bar:
        st.header('Bar Plot')
        print(explanation)
        plt.figure(figsize=(10, 8))

        shap.bar_plot(shap_val[0], feature_data.values[0], feature_data.columns)
        # shap.plots.bar(explanation, max_display=3, show=False)

        st.pyplot(plt.gcf())
    
    # with embed:
    #     st.header('Embedding Plot')
    #     feature1 = st.selectbox('Select Feature', options=feature_data.columns)
    #     shap.embedding_plot(feature1, shap_val, feature_data.columns)
    #     st.pyplot(plt.gcf())
    
    with waterfall:
        st.header('Waterfall Plot')
        plt.figure(figsize=(10, 8))

        shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_val[0])
        # shap.plots.waterfall(explanation[0])
        
        st.pyplot(plt.gcf())

st.set_page_config(
    page_title="Predict",
    page_icon="📊",
)

# conn = sqlite3.connect('database.db')
# cursor = conn.cursor()
conn = st.connection("gsheets", type=GSheetsConnection)
        
with open('models/xgb.pkl', 'rb') as file:
    model = pickle.load(file)
# model = xgb.Booster(model_file='models/xgb_fs.xgb')

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

    # ! INPUT DATA
    input_data = pd.DataFrame(params, index=[0])
    test_data = input_data[model.feature_names_in_]
    st.subheader('Input Data')
    st.dataframe(test_data)

    # ! PREDICT IMPEDANCE
    impedance = predict(test_data)
    st.subheader('Result')
    st.write(f'The predicted impedance is {impedance:.2f} ohms')
    
    # ! INSERT DATA TO DB
    insert_data(input_data, impedance)

    # ! SHAP VISUALIZATION
    st.subheader('SHAP Visualization')
    visualize(model, test_data)

    with status:
        status.update(label="Process Complete!", state="complete", expanded=False)
        time.sleep(1)