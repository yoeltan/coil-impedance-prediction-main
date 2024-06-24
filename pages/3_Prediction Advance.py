import streamlit as st
import pandas as pd
import time
import pickle
from streamlit_gsheets import GSheetsConnection

def get_data() -> pd.DataFrame:
    return conn.read(
        worksheet="prediction_advance",
        ttl=0, 
        usecols=[i for i in range(0,17)]
        ).dropna()

def insert_data(new_data):
    with status:
        st.write('Inserting to Database.')
        time.sleep(1)

    old_data = get_data()
    new_data['Predict Time'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    # data = old_data.append(new_data, ignore_index=True)
    data = pd.concat([old_data, new_data], ignore_index=True)

    conn.update(
        worksheet="prediction_advance",
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

        'RANGE 1': int(right.number_input('Smaller', format='%.0f')),
        'RANGE 2': int(right.number_input('Larger', format='%.0f')),
    }

    st.subheader("Column to Adjust")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    A0 = col1.checkbox("IDL LV")
    A1 = col2.checkbox("IDW LV")
    A2 = col3.checkbox("IDH LV")
    A3 = col4.checkbox("ODL LV")
    A4 = col5.checkbox("ODW LV")
    A5 = col6.checkbox("ODH LV")
    A6 = col1.checkbox("IDL HV")
    A7 = col2.checkbox("IDW HV")
    A8 = col3.checkbox("IDH HV")
    A9 = col4.checkbox("ODL HV")
    A10 = col5.checkbox("ODW HV")
    A11 = col6.checkbox("ODH HV")

    # Initial placeholder for the submit button
    if st.session_state['error_msg']:
        submitted = st.form_submit_button(label='Predict Impedance', disabled=True)
    else:
        submitted = st.form_submit_button(label='Predict Impedance')

suggestion_imp_larger = pd.DataFrame()
suggestion_imp_smaller = pd.DataFrame()

def value_duplicates(data_modified, data_ori):
    # Daftar untuk menyimpan kolom yang termodifikasi
    columns_edited = []

    # Loop melalui kolom
    for col in data_ori.columns:
        # Bandingkan nilai
        if not data_ori[col].equals(data_modified[col]):
            # Identifikasi indeks baris yang termodifikasi
            for i in range(len(data_ori)):
                if data_ori[col][i] != data_modified[col][i]:
                    if (data_ori[col][i] > data_modified[col][i]):
                        # Tambahkan kolom yang termodifikasi ke dalam daftar
                        columns_edited.append(col + " -" + str(data_ori[col][i] - data_modified[col][i]))
                    elif(data_ori[col][i] < data_modified[col][i]):
                        # Tambahkan kolom yang termodifikasi ke dalam daftar
                        columns_edited.append(col + " " + str(data_modified[col][i] - data_ori[col][i]))

    return columns_edited

def calculating(df, ori_imp, data_ori):
    
    impedance = predictImpedance(df)

    global suggestion_imp_larger
    global suggestion_imp_smaller

    # smaller semakin mendekati 0
    if (ori_imp > impedance):
        save1 = pd.DataFrame(df, index=[0])
        save1['Columns Adjusted'] = [value_duplicates(save1, data_ori)]
        save1['Original Impedance'] = ori_imp
        save1['Predicted Impedance'] = impedance
        save1['Difference'] = [ori_imp - impedance]
        # save['Columns Adjusted'] = [columns_adjusted]

        suggestion_imp_smaller = pd.concat([suggestion_imp_smaller, save1], ignore_index=True)

    elif (ori_imp < impedance):
        save2 = pd.DataFrame(df, index=[0])
        save2['Columns Adjusted'] = [value_duplicates(save2, data_ori)]
        save2['Original Impedance'] = ori_imp
        save2['Predicted Impedance'] = impedance
        save2['Difference'] = [impedance - ori_imp]
        # save['Columns Adjusted'] = [columns_adjusted]

        suggestion_imp_larger = pd.concat([suggestion_imp_larger, save2], ignore_index=True)
        
    return df

def reset_value(df, column, original_value):
    df.at[0, column] = original_value

def modify_and_test_v1(df, columns, value_range, impedance, x, data_ori):
    num_columns = len(columns)

    for i in value_range:
        if(columns[x] == columns[0]):
            with status:
                st.write(f'Calculating Layer {x}: {columns[x]} {i:.2f}')
                   
        original_value1 = df.at[0, columns[x]]
        df.at[0, columns[x]] = df.at[0, columns[x]] + i
        calculating(df, impedance, data_ori)

        x += 1

        if x < num_columns:
            modify_and_test_v1(df, columns, value_range, impedance, x, data_ori)
         
        x -= 1

        reset_value(df, columns[x], original_value1)
        calculating(df, impedance, data_ori)

def modify_and_test_v2(df, columns, value_range, impedance, x, data_ori):
    num_columns = len(columns)

    for i in value_range:
        if(columns[x] == columns[num_columns-1]):
            with status:
                st.write(f'Calculating Layer {x}: {columns[x]} {i:.2f}')
        
        original_value1 = df.at[0, columns[x]]
        df.at[0, columns[x]] = df.at[0, columns[x]] + i
        calculating(df, impedance, data_ori)

        x -= 1

        if x > -1:
            modify_and_test_v2(df, columns, value_range, impedance, x, data_ori)
         
        x += 1

        reset_value(df, columns[x], original_value1)
        calculating(df, impedance, data_ori)

def modify_and_test_v3(df, columns, value_range, impedance, x, data_ori):

    for i in value_range:
        for column in columns:

            original_value1 = df.at[0, column]
            df.at[0, column] = df.at[0, column] + i
            calculating(df, impedance, data_ori)
            reset_value(df, column, original_value1)

def columns_checker(columns_name):
    columns = []
    for i in range(12):
        if checkboxes[i]:
            columns.append(columns_name[i])

    return columns

if submitted:
    status = st.status('Processing...', expanded=True)

    # ! INPUT DATA
    input_data = pd.DataFrame(params, index=[0])
    test_data_imp = input_data[modelImpedance.feature_names_in_]
    st.subheader('Input Data')
    st.dataframe(test_data_imp)

    data_ori = input_data[modelImpedance.feature_names_in_]

    # ! PREDICT IMPEDANCE
    impedance = predictImpedance(test_data_imp)
    st.write(f'The predicted impedance is {impedance:.2f}%')

    # Define column names and ranges
    # List of columns to modify
    columns = test_data_imp.columns.tolist()

    # List of checkbox states
    checkboxes = [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]

    columns = columns_checker(columns)
    ranges = range(input_data.iloc[0, 12], input_data.iloc[0, 13] + 1)

    x = 0
    # Call the function
    modify_and_test_v1(test_data_imp, columns, ranges, impedance, x, data_ori)
    
    modify_and_test_v3(test_data_imp, columns, ranges, impedance, x, data_ori)
    
    num_columns = len(columns)
    # Call the function
    modify_and_test_v2(test_data_imp, columns, ranges, impedance, num_columns - 1, data_ori)

    suggestion_imp_smaller=suggestion_imp_smaller.drop_duplicates(subset=['Inner Diameter Length LV', 'Inner Diameter Width LV', 'Inner Diameter Height LV',
                                      'Outer Diameter Length LV', 'Outer Diameter Width LV', 'Outer Diameter Height LV',
                                      'Inner Diameter Length HV', 'Inner Diameter Width HV', 'Inner Diameter Height HV',
                                      'Outer Diameter Length HV', 'Outer Diameter Width HV', 'Outer Diameter Height HV'], keep='last')

    suggestion_imp_larger=suggestion_imp_larger.drop_duplicates(subset=['Inner Diameter Length LV', 'Inner Diameter Width LV', 'Inner Diameter Height LV',
                                      'Outer Diameter Length LV', 'Outer Diameter Width LV', 'Outer Diameter Height LV',
                                      'Inner Diameter Length HV', 'Inner Diameter Width HV', 'Inner Diameter Height HV',
                                      'Outer Diameter Length HV', 'Outer Diameter Width HV', 'Outer Diameter Height HV'], keep='last')

    st.subheader("Suggestion - Impedance Smaller")
    st.dataframe(suggestion_imp_smaller)

    st.subheader("Suggestion - Impedance Larger")
    st.dataframe(suggestion_imp_larger)

    insert_data(suggestion_imp_smaller)

    with status:
        status.update(label="Process Complete!", state="complete", expanded=False)
        time.sleep(1)



