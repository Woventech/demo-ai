import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import plotly.express as px

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Clustering & PCA – Live Demo",
    layout="wide"
)

st.title("🧠 Clustering e Riduzione Dimensionale")
st.markdown("""
Questa demo mostra **come un algoritmo ragiona nello spazio multidimensionale**
e come possiamo **renderlo visibile**.
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

X = df[features]

# =====================================================
# SIDEBAR – PARAMETRI
# =====================================================
st.sidebar.header("⚙️ Parametri")

k = st.sidebar.slider(
    "Numero di cluster (KMeans)",
    min_value=2,
    max_value=6,
    value=3
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Naming semantico PCA")

pca1_name = st.sidebar.text_input(
    "Nome PCA 1",
    "Orientamento alla Distribuzione di Massa"
)

pca2_name = st.sidebar.text_input(
    "Nome PCA 2",
    "Orientamento a Fresco e Specialità"
)

pca3_name = st.sidebar.text_input(
    "Nome PCA 3",
    "Comportamento di Nicchia"
)

# =====================================================
# ACTION BUTTON
# =====================================================
st.markdown("## ▶️ Avvio Analisi")

start = st.button("🚀 Avvia calcolo")

if not start:
    st.info("""
    Premi **Avvia calcolo** per:
    - preprocessare i dati
    - calcolare i cluster
    - ridurre le dimensioni con PCA
    - visualizzare lo spazio dell'algoritmo
    """)
    st.stop()

# =====================================================
# PREPROCESSING
# =====================================================
with st.spinner("🔄 Preprocessamento dei dati..."):
    X_log = np.log1p(X)
    X_scaled = StandardScaler().fit_transform(X_log)

# =====================================================
# CLUSTERING
# =====================================================
with st.spinner("🧩 Calcolo dei cluster..."):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster"] = clusters.astype(str)

# =====================================================
# PCA
# =====================================================
with st.spinner("📉 Riduzione dimensionale (PCA)..."):
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    df["PCA1"] = X_pca[:, 0]
    df["PCA2"] = X_pca[:, 1]
    df["PCA3"] = X_pca[:, 2]

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PCA1", "PCA2", "PCA3"],
        index=features
    )

# =====================================================
# PCA 2D – TABELLA + GRAFICO
# =====================================================
st.header("📊 PCA 2D – Interpretazione delle Componenti")

st.subheader("🔎 Composizione delle Componenti (PCA 1 & 2)")
st.dataframe(
    loadings[["PCA1", "PCA2"]]
    .rename(columns={
        "PCA1": pca1_name,
        "PCA2": pca2_name
    })
    .style.format("{:.3f}")
)

fig_pca_2d = px.scatter(
    df,
    x="PCA1",
    y="PCA2",
    color="Cluster",
    labels={
        "PCA1": pca1_name,
        "PCA2": pca2_name
    },
    title="Spazio latente bidimensionale (PCA)"
)

st.plotly_chart(fig_pca_2d, use_container_width=True)

st.info("""
Gli assi **non sono variabili reali**  
ma **combinazioni intelligenti** create dall’algoritmo.
""")

# =====================================================
# PCA 3D – TABELLA + GRAFICO
# =====================================================
st.header("📐 PCA 3D – Multidimensionalità")

st.subheader("🔎 Composizione delle Componenti (PCA 1, 2 e 3)")
st.dataframe(
    loadings
    .rename(columns={
        "PCA1": pca1_name,
        "PCA2": pca2_name,
        "PCA3": pca3_name
    })
    .style.format("{:.3f}")
)

fig_pca_3d = px.scatter_3d(
    df,
    x="PCA1",
    y="PCA2",
    z="PCA3",
    color="Cluster",
    labels={
        "PCA1": pca1_name,
        "PCA2": pca2_name,
        "PCA3": pca3_name
    },
    title="Spazio latente tridimensionale (PCA)"
)

st.plotly_chart(fig_pca_3d, use_container_width=True)

st.warning("""
Questo è il **limite della visualizzazione umana**.  
L’algoritmo però lavora senza problemi in **molte più dimensioni**.
""")

# =====================================================
# FOOTER – MESSAGGIO CHIAVE
# =====================================================
st.markdown("---")
st.markdown("""
### 🧠 Messaggio chiave

> L’intelligenza artificiale non semplifica i dati.  
> Li **riorganizza** per trovare strutture che noi non vediamo.
""")
