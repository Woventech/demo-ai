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
LEGEND_SIZE = 16

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Come ragiona un algoritmo",
    layout="wide"
)

st.title("🧠 Come ragiona un algoritmo di Clustering")
st.markdown("""
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

X = df[features]

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙️ Parametri")

k = st.sidebar.slider("Numero di cluster", 2, 6, 3)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Naming semantico PCA")

pca1_name = st.sidebar.text_input(
    "PCA 1",
    "Orientamento alla Distribuzione di Massa"
)

pca2_name = st.sidebar.text_input(
    "PCA 2",
    "Orientamento a Fresco e Specialità"
)

pca3_name = st.sidebar.text_input(
    "PCA 3",
    "Comportamento di Nicchia"
)

# =====================================================
# ACTION
# =====================================================
st.markdown("## ▶️ Avvio Analisi")

start = st.button("🚀 Avvia calcolo")

if not start:
    st.info("""
    Premi **Avvia calcolo** per iniziare il viaggio:
    1. spazio umano
    2. spazio reale
    3. spazio latente dell'algoritmo
    """)
    st.stop()

# =====================================================
# PREPROCESSING
# =====================================================
with st.spinner("🔄 Elaborazione dati..."):
    X_log = np.log1p(X)
    X_scaled = StandardScaler().fit_transform(X_log)

    kmeans = KMeans(n_clusters=k, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X_scaled).astype(str)

# =====================================================
# ATTO 1 — SPAZIO UMANO (2D)
# =====================================================
st.header("👀 ATTO 1 — Come vediamo noi i dati (2D)")

col1, col2 = st.columns(2)
x2 = col1.selectbox("Asse X", features, index=0)
y2 = col2.selectbox("Asse Y", features, index=2)

fig_2d_human = px.scatter(
    df,
    x=x2,
    y=y2,
    color="Cluster",
    title="Spazio reale bidimensionale"
)

st.plotly_chart(fig_2d_human, use_container_width=True)

fig.update_layout(
    title_font_size=TITLE_SIZE,
    legend_title_font_size=LEGEND_SIZE,
    legend_font_size=LEGEND_SIZE,
    font=dict(size=PLOT_FONT_SIZE)
)


st.info("""
Scegliamo **solo due dimensioni** perché il nostro cervello funziona così.
""")



# =====================================================
# ATTO 1 — SPAZIO UMANO (3D)
# =====================================================
st.subheader("👀 Spazio umano tridimensionale")

x3 = st.selectbox("Asse X (3D)", features, index=0)
y3 = st.selectbox("Asse Y (3D)", features, index=1)
z3 = st.selectbox("Asse Z (3D)", features, index=2)

fig_3d_human = px.scatter_3d(
    df,
    x=x3,
    y=y3,
    z=z3,
    color="Cluster",
    title="Spazio reale tridimensionale"
)

st.plotly_chart(fig_3d_human, use_container_width=True)

st.warning("""
Già con 3 dimensioni iniziamo a perdere intuizione.
""")

# =====================================================
# ATTO 2 — PCA
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
# ATTO 3 — PCA 2D
# =====================================================
st.header("🧠 ATTO 2 — L'algoritmo crea nuovi assi (PCA 2D)")

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
    title="Spazio latente bidimensionale"
)

st.plotly_chart(fig_pca_2d, use_container_width=True)

st.success("""
Questi assi **non esistevano nei dati originali**.
L’algoritmo li ha creati per capire meglio.
""")

# =====================================================
# ATTO 4 — PCA 3D
# =====================================================
st.header("🚀 ATTO 3 — Lo spazio dell'algoritmo (PCA 3D)")

st.dataframe(
    loadings.rename(columns={
        "PCA1": pca1_name,
        "PCA2": pca2_name,
        "PCA3": pca3_name
    }).style.format("{:.3f}")
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
    title="Spazio latente tridimensionale"
)

st.plotly_chart(fig_pca_3d, use_container_width=True)

st.warning("""
Questo è il massimo che possiamo visualizzare.
L’algoritmo però **non ha limiti dimensionali**.
""")

# =====================================================
# CHIUSURA
# =====================================================
st.markdown("---")
st.markdown("""
### 🎯 Messaggio finale

> L’intelligenza artificiale  
> non vede i dati come noi.  
>  
> Li **ricombina**,  
> li **riproietta**,  
> li **comprende**.
""")
