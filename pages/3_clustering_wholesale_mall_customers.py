import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import plotly.express as px

# =====================================================
# LAYOUT SETTINGS
# =====================================================
PLOT_FONT_SIZE = 18
AXIS_TITLE_SIZE = 22
TITLE_SIZE = 26
LEGEND_SIZE = 20

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Come ragiona un algoritmo",
    layout="wide"
)

st.title("🧠 Come ragiona un algoritmo di Clustering")
st.markdown("""
Immaginiamo di analizzare le **abitudini di acquisto di alcuni clienti di un supermercato**.

Un viaggio visivo:
dal modo **umano** di guardare i dati  
al modo **algoritmico** di comprenderli.
""")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
    return pd.read_csv(url)

df = load_data()

features = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen"
]

feature_names_it = {
    "Fresh": "Prodotti Freschi",
    "Milk": "Latticini",
    "Grocery": "Prodotti da Scaffale",
    "Frozen": "Surgelati",
    "Detergents_Paper": "Casa & Pulizia",
    "Delicassen": "Gastronomia"
}

X = df[features]

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙️ Parametri")

k = st.sidebar.slider("Numero di gruppi di clienti", 2, 6, 3)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Naming semantico PCA")

pca1_name = st.sidebar.text_input(
    "PCA 1",
    "Spesa standard e ricorrente"
)

pca2_name = st.sidebar.text_input(
    "PCA 2",
    "Orientamento a fresco e qualità"
)

pca3_name = st.sidebar.text_input(
    "PCA 3",
    "Comportamenti di nicc
