import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import pickle
from streamlit_gsheets import GSheetsConnection

def visualize(model, dataset=None):

    @st.cache_resource
    def init_shap_js():
        shap.initjs()

    # ! SHAP PREPARATION
    init_shap_js()
    explainer = shap.TreeExplainer(model)
    shap_values_dataset = explainer.shap_values(dataset)
    explanation = shap.Explanation(shap_values_dataset, data=dataset, feature_names=dataset.columns)

    # ! SHAP PLOT
    shap_plot(shap_values_dataset, dataset, explainer=explainer, explanation=explanation, show=False)

    return

def shap_plot(shap_val_dataset, dataset_data, explainer=None, explanation=None, show=True):
    
    summary, heatmap, dependence, bar,  = st.tabs(['Summary', 'Heatmap', 'Dependence', 'Bar'])

    # Summary from Dataset
    with summary:
        st.header('Summary Plot')
        plt.figure(figsize=(10, 8))

        
        shap.summary_plot(shap_val_dataset, features=dataset_data, feature_names=dataset_data.columns, show=show)
        
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()  # Membersihkan plot setelah ditampilkan
        plt.close()  # Menutup plot agar tidak ada sisa plot)

    with heatmap:
        st.header('Heatmap Plot')
        plt.figure(figsize=(10, 8))

        shap.heatmap_plot(explanation, max_display=12)
        
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()  # Membersihkan plot setelah ditampilkan
        plt.close()  # Menutup plot agar tidak ada sisa plot
    
    with dependence:
        st.header('Dependence Plot')
        plt.figure(figsize=(10, 8))

        shap.dependence_plot("Inner Diameter Length LV", shap_val_dataset, dataset_data, interaction_index=1)
        
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()  # Membersihkan plot setelah ditampilkan
        plt.close()  # Menutup plot agar tidak ada sisa plot

    with bar:
        st.header('Bar Plot')
        print(explanation)
        plt.figure(figsize=(10, 8))

        shap.bar_plot(shap_val_dataset[0], dataset_data.values[0], dataset_data.columns, max_display=len(dataset_data.columns))

        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()  # Membersihkan plot setelah ditampilkan
        plt.close()  # Menutup plot agar tidak ada sisa plot

st.set_page_config(
    page_title="Association Rules",
    page_icon="📚",
    layout="centered",
)

st.title('Association Rules x SHAP')
st.divider()
st.subheader('Association Rules')

# ! STREAMLIT GSPREAD
conn = st.connection("gsheets", type=GSheetsConnection)
dataset = conn.read(
    worksheet="ARM_V1",
    ttl=0, 
    usecols=[i for i in range(0, 5)]
).dropna()

dataset_impedance = conn.read(
        worksheet="dataset_impedance",
        ttl=0, 
        usecols=[i for i in range(0, 12)]
    ).dropna()

with open('models/xgb.pkl', 'rb') as file:
    modelImpedance = pickle.load(file) 

# Load Dataset Impedance
dataset_impedance = pd.DataFrame(dataset_impedance)
dataset_impedance = dataset_impedance[modelImpedance.feature_names_in_]

st.write(dataset)

st.subheader('SHAP Visualization')
visualize(modelImpedance, dataset_impedance)