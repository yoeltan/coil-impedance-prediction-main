import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection

st.set_page_config(
    page_title="Association Rules",
    page_icon="📚",
    layout="centered",
)

st.title('Association Rules')

# ! STREAMLIT GSPREAD
conn = st.connection("gsheets", type=GSheetsConnection)
dataset = conn.read(
    worksheet="association_rules_confidence",
    ttl=0, 
    usecols=[i for i in range(0, 5)]
).dropna()

st.write(dataset)