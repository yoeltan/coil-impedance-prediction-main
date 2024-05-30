import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection

st.set_page_config(
    page_title="Dataset",
    page_icon="📚",
    layout="centered",
)

# ! STREAMLIT GSPREAD
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

st.title('Dataset Impedance')
st.write(dataset_impedance)

st.title('Dataset Tolerance')
st.write(dataset_tolerance)