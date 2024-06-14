import streamlit as st
import pandas as pd
import time
import pickle
from streamlit_gsheets import GSheetsConnection

def get_data_smaller() -> pd.DataFrame:
    return conn.read(
        worksheet="prediction_suggestion_smaller",
        ttl=0, 
        usecols=[i for i in range(0,7)]
        ).dropna()

def get_data_larger() -> pd.DataFrame:
    return conn.read(
        worksheet="prediction_suggestion_larger",
        ttl=0, 
        usecols=[i for i in range(0,7)]
        ).dropna()

def insert_data_smaller(new_data):
    with status:
        st.write('Inserting to Database.')

    old_data = get_data_smaller()
    new_data['Predict Time'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    # data = old_data.append(new_data, ignore_index=True)
    data = pd.concat([old_data, new_data], ignore_index=True)

    conn.update(
        worksheet="prediction_suggestion_smaller",
        data=data,
    )

def insert_data_larger(new_data):
    with status:
        st.write('Inserting to Database.')

    old_data = get_data_larger()
    new_data['Predict Time'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    # data = old_data.append(new_data, ignore_index=True)
    data = pd.concat([old_data, new_data], ignore_index=True)

    conn.update(
        worksheet="prediction_suggestion_larger",
        data=data,
    )

def predictImpedance(test_data):

    impedance = modelImpedance.predict(test_data)[0]

    return impedance

# Predict Main Page
st.set_page_config(
    page_title="Prediction",
    page_icon="📊",
)

conn = st.connection("gsheets", type=GSheetsConnection)



with open('models/xgb.pkl', 'rb') as file:
    modelImpedance = pickle.load(file)



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
    }

    # Initial placeholder for the submit button
    if st.session_state['error_msg']:
        submitted = st.form_submit_button(label='Predict Impedance', disabled=True)
    else:
        submitted = st.form_submit_button(label='Predict Impedance')

suggestion_imp_larger = []
suggestion_imp_smaller = []

def modify_testing(df, column, adjusted, ori_imp):
    original_value = df.at[0, column]
    df.at[0, column] = original_value + adjusted

    # ! PREDICT IMPEDANCE
    impedance = predictImpedance(df)
    
    # smaller semakin mendekati 0
    if (ori_imp > impedance):
        suggestion_imp_smaller.append({
            'Adjusted Column' : column,
            'Adjusted Value' : adjusted,
            'Adjusted' : f"{column}{adjusted}",
            'Difference Impedance' : ori_imp - impedance,
            'Original Impedance' : ori_imp,
            'Adjusted Impedance' : impedance,
        })
    # larger semakin menjauhi 0
    elif (ori_imp < impedance):
        suggestion_imp_larger.append({
            'Adjusted Column' : column,
            'Adjusted Value' : adjusted,
            'Adjusted' : f"{column}{adjusted}",
            'Difference Impedance' : ori_imp - impedance,
            'Original Impedance' : ori_imp,
            'Adjusted Impedance' : impedance,
        })

    df.at[0, column] = original_value

    return

if submitted:
    dataset_impedance = conn.read(
        worksheet="dataset_impedance",
        ttl=0, 
        usecols=[i for i in range(0, 12)]
    ).dropna()

    # Load Dataset Impedance
    dataset = pd.DataFrame(dataset_impedance)
    dataset_impedance = dataset[modelImpedance.feature_names_in_]

    status = st.status('Processing...', expanded=True)

    # ! INPUT DATA
    input_data = pd.DataFrame(params, index=[0])
    test_data_imp = input_data[modelImpedance.feature_names_in_]
    st.subheader('Input Data')
    st.dataframe(test_data_imp)

    with status:
            st.write(f'Calculating Initial Data')

    # ! PREDICT IMPEDANCE
    impedance = predictImpedance(test_data_imp)
    st.write(f'The predicted impedance is {impedance:.2f} ohms')

    st.divider()

    # List of columns to modify
    columns = test_data_imp.columns.tolist()

    for i in range (-10, 11):
        with status:
            st.write(f'Calculating Range: {i:.2f}')

        # Modify each column
        for column in columns:
            modify_testing(test_data_imp, column, i, impedance)

    suggestion_imp_larger = pd.DataFrame(suggestion_imp_larger)
    suggestion_imp_smaller = pd.DataFrame(suggestion_imp_smaller)
    
    st.subheader("Suggestion - Impedance Smaller")
    st.dataframe(suggestion_imp_smaller.style.background_gradient(subset="Difference Impedance", cmap='viridis').highlight_min(subset="Adjusted Impedance"))

    st.subheader("Suggestion - Impedance Larger")
    st.dataframe(suggestion_imp_larger.style.background_gradient(subset="Difference Impedance", cmap='viridis').highlight_max(subset="Adjusted Impedance"))

    insert_data_smaller(suggestion_imp_smaller)
    insert_data_larger(suggestion_imp_larger)

    with status:
        status.update(label="Process Complete!", state="complete", expanded=False)
        time.sleep(1)



