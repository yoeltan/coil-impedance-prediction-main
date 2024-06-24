import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import shap
import pickle
from streamlit_gsheets import GSheetsConnection

def get_data_adjusted() -> pd.DataFrame:
    return conn.read(
        worksheet="prediction_adjusted",
        ttl=0, 
        usecols=[i for i in range(0, 17)]
        ).dropna()

def get_data_history() -> pd.DataFrame:
    return conn.read(
        worksheet="History",
        ttl=0, 
        # kolom A = 1 pada excel
        usecols=[i for i in range(0, 14)]
        ).dropna()

def insert_data_adjusted(new_data):
    with status:
        st.write('Inserting to Database.')

    old_data = get_data_adjusted()
    new_data['Predict Time'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    # data = old_data.append(new_data, ignore_index=True)
    data = pd.concat([old_data, new_data], ignore_index=True)

    conn.update(
        worksheet="prediction_adjusted",
        data=data,
    )

def insert_data_history(new_data):
    with status:
        st.write('Inserting to Database.')

    old_data = get_data_history()
    new_data['Predict Time'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    # data = old_data.append(new_data, ignore_index=True)
    data = pd.concat([old_data, new_data], ignore_index=True)

    conn.update(
        worksheet="History",
        data=data,
    )   

def predictImpedance(test_data):
    impedance = modelImpedance.predict(test_data)[0]

    return impedance

def visualize(model, test_data, plot_on, dataset=None):

    @st.cache_resource
    def init_shap_js():
        shap.initjs()

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
    if(plot_on == 1):
        with st.popover("Open Shap Plot"):
            shap_plot(plot_on, shap_values_dataset, dataset, shap_values, test_data, explainer=explainer, explanation=explanation, show=False)
    else:
        shap_plot(plot_on, shap_values_dataset, dataset, shap_values, test_data, explainer=explainer, explanation=explanation, show=False)

    return shap_values_df

def shap_plot(plot_on, shap_val_dataset, dataset_data, shap_val, feature_data, explainer=None, explanation=None, show=True):
    
    if(plot_on == 1):
        bar, waterfall = st.tabs(['Bar', 'Waterfall'])
    else:
        summary, bar, waterfall = st.tabs(['Summary', 'Bar', 'Waterfall'])

    # Summary from Dataset
    if(plot_on == 0):
        with summary:
            st.header('Summary Plot')
            plt.figure(figsize=(10, 8))
            
            shap.summary_plot(shap_val_dataset, features=dataset_data, feature_names=dataset_data.columns, show=show)
            
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.clf()  # Membersihkan plot setelah ditampilkan
            plt.close()  # Menutup plot agar tidak ada sisa plot)

    with bar:
        st.header('Bar Plot')
        print(explanation)
        plt.figure(figsize=(10, 8))

        shap.bar_plot(shap_val[0], feature_data.values[0], feature_data.columns, max_display=len(feature_data.columns))

        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()  # Membersihkan plot setelah ditampilkan
        plt.close()  # Menutup plot agar tidak ada sisa plot
    
    with waterfall:
        st.header('Waterfall Plot')
        plt.figure(figsize=(10, 8))

        shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_val[0], feature_names=feature_data.columns, max_display=len(feature_data.columns))
        
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

with open('models/xgb.pkl', 'rb') as file:
    modelImpedance = pickle.load(file) 

summary = pd.DataFrame()

st.title('Coil Impedance Prediction')
if 'error_msg' not in st.session_state:
    st.session_state['error_msg'] = "Please fill required inputs"

# Validation function
def validate_inputs():
    # Diameter Length
    if st.session_state['Inner Diameter Length LV'] >= st.session_state['Outer Diameter Length LV']:
        st.session_state['error_msg'] = "'Outer Diameter Length LV' must be greater than 'Inner Diameter Length LV'"
    elif st.session_state['Outer Diameter Length LV'] >= st.session_state['Inner Diameter Length HV']:
        st.session_state['error_msg'] = "'Inner Diameter Length HV' must be greater than 'Outer Diameter Length LV'"    
    elif st.session_state['Inner Diameter Length HV'] >= st.session_state['Outer Diameter Length HV']:
        st.session_state['error_msg'] = "'Outer Diameter Length HV' must be greater than 'Inner Diameter Length HV'"
    # Diameter Width
    elif st.session_state['Inner Diameter Width LV'] >= st.session_state['Outer Diameter Width LV']:
        st.session_state['error_msg'] = "'Outer Diameter Width LV' must be greater than 'Inner Diameter Width LV'"
    elif st.session_state['Outer Diameter Width LV'] >= st.session_state['Inner Diameter Width HV']:
        st.session_state['error_msg'] = "'Inner Diameter Width HV' must be greater than 'Outer Diameter Width LV'"
    elif st.session_state['Inner Diameter Width HV'] >= st.session_state['Outer Diameter Width HV']:
        st.session_state['error_msg'] = "'Outer Diameter Width HV' must be grater than 'Inner Diameter Width HV'"
    # Diameter Height
    elif (st.session_state['Inner Diameter Height LV'] < 135 or
        st.session_state['Outer Diameter Height LV'] < 135 or
        st.session_state['Inner Diameter Height HV'] < 135 or
        st.session_state['Outer Diameter Height HV'] < 135):
        st.session_state['error_msg'] = "'Inner, Outer Diameter Height LV, HV' must be grater than 134 mm"
    else:
        st.session_state['error_msg'] = ""

# Display error message if validation fails
if st.session_state['error_msg']:
    st.error(st.session_state['error_msg'])

left, right = st.columns(2)
with st.form(key='input_form'):
    params = {
        'Inner Diameter Length LV' : int(left.number_input('Inner Diameter Length LV', min_value=0, on_change=validate_inputs, key='Inner Diameter Length LV')),
        'Inner Diameter Width LV': int(left.number_input('Inner Diameter Width LV', min_value=0, on_change=validate_inputs, key='Inner Diameter Width LV')),
        'Inner Diameter Height LV': int(left.number_input('Inner Diameter Height LV', min_value=0, on_change=validate_inputs, key='Inner Diameter Height LV')),

        'Outer Diameter Length LV': int(left.number_input('Outer Diameter Length  LV', min_value=0, on_change=validate_inputs, key='Outer Diameter Length LV')),
        'Outer Diameter Width LV': int(left.number_input('Outer Diameter Width LV', min_value=0, on_change=validate_inputs, key='Outer Diameter Width LV')),
        'Outer Diameter Height LV': int(left.number_input('Outer Diameter Height LV', min_value=0, on_change=validate_inputs, key='Outer Diameter Height LV')),

        'Inner Diameter Length HV': int(right.number_input('Inner Diameter Length HV', min_value=0, on_change=validate_inputs, key='Inner Diameter Length HV')),
        'Inner Diameter Width HV': int(right.number_input('Inner Diameter Width HV', min_value=0, on_change=validate_inputs, key='Inner Diameter Width HV')),
        'Inner Diameter Height HV': int(right.number_input('Inner Diameter Height HV', min_value=0, on_change=validate_inputs, key='Inner Diameter Height HV')),

        'Outer Diameter Length HV': int(right.number_input('Outer Diameter Length HV', min_value=0, on_change=validate_inputs, key='Outer Diameter Length HV')),
        'Outer Diameter Width HV': int(right.number_input('Outer Diameter Width HV', min_value=0, on_change=validate_inputs, key='Outer Diameter Width HV')),
        'Outer Diameter Height HV': int(right.number_input('Outer Diameter Height HV', min_value=0, on_change=validate_inputs, key='Outer Diameter Height HV')),

        'ADJUSTED': int(right.number_input('ADJUSTED', min_value=-10, on_change=validate_inputs)),
    }

    # Initial placeholder for the submit button
    if st.session_state['error_msg']:
        submitted = st.form_submit_button(label='Predict Impedance', disabled=True)
    else:
        submitted = st.form_submit_button(label='Predict Impedance')

def modify_and_restore(df, column, adjusted, ori_imp):
    
    st.subheader(f"{column} {adjusted}")

    original_value = df.at[0, column]
    df.at[0, column] = original_value + adjusted
    st.dataframe(df)

    # ! PREDICT IMPEDANCE
    impedance = predictImpedance(df)
    st.write(f'The predicted impedance is {impedance:.2f}%')

    plot_on = 1
    shap_values = visualize(modelImpedance, df, plot_on, dataset_impedance)
    save = pd.DataFrame(shap_values, index=[0])
    save['ADJUSTED'] = [column + str(adjusted)]
    save['Difference'] = [ori_imp - impedance]
    save['Original Impedance'] = [ori_imp]
    save['Impedance'] = impedance

    global summary
    summary = pd.concat([summary, save], ignore_index=True)

    # ! INSERT DATA TO DB
    insert_data_adjusted(save)

    df.at[0, column] = original_value

    st.divider()

if submitted and not st.session_state['error_msg']:
    dataset_impedance = conn.read(
        worksheet="dataset_impedance",
        ttl=0, 
        usecols=[i for i in range(0, 12)]
    ).dropna()

    # Load Dataset Impedance
    dataset = pd.DataFrame(dataset_impedance)
    dataset_impedance = dataset[modelImpedance.feature_names_in_]

    status = st.status('Processing...', expanded=True)
    
    # Load Dataset Impedance
    dataset = pd.DataFrame(dataset_impedance)
    dataset_impedance = dataset[modelImpedance.feature_names_in_]

    # ! INPUT DATA
    input_data = pd.DataFrame(params, index=[0])
    test_data_imp = input_data[modelImpedance.feature_names_in_]
    st.subheader('Input Data')
    st.dataframe(test_data_imp)

    # ! PREDICT IMPEDANCE
    with status:
        st.write('Calculating Initial Impedance')
    impedance = predictImpedance(test_data_imp)
    st.subheader('Result')
    st.write(f'The predicted impedance is {impedance:.2f}%')

    plot_on = 0
    # ! SHAP VISUALIZATION
    with status:
        st.write('Calculating Initial SHAP Values.')
    st.subheader('SHAP Visualization Impedance')
    shap_values = visualize(modelImpedance, test_data_imp, plot_on, dataset_impedance)

    save = pd.DataFrame(test_data_imp, index=[0])
    save['Impedance'] = impedance
    insert_data_history(save)

    # List of columns to modify
    columns = test_data_imp.columns.tolist()

    st.write("---")
    # Modify each column
    # column string
    for column in columns:
        with status:
            st.write(f'Calculating {column}: {input_data.iloc[0, 12]:.2f}')
        modify_and_restore(test_data_imp, column, input_data.iloc[0, 12], impedance)

    st.subheader("Summary Table")
    st.dataframe(summary)

    # Select columns 1 to 8 (B to I in this case, assuming 0-based index)
    subset_summary = summary.iloc[:, 1:12]

    # Apply the highlight_max style to the subset DataFrame for each row
    styled0_subset_summary = subset_summary.style.highlight_min(axis=1)
    st.dataframe(styled0_subset_summary)

    styled1_subset_summary = subset_summary.style.highlight_max(axis=1)
    st.dataframe(styled1_subset_summary)

    with status:
        status.update(label="Process Complete!", state="complete", expanded=False)
        time.sleep(1)



