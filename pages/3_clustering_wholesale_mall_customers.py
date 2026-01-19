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
    "Comportamenti di nicchia"
)

# =====================================================
# PREPROCESSING
# =====================================================
X_log = np.log1p(X)
X_scaled = StandardScaler().fit_transform(X_log)

kmeans = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled).astype(str)

# =====================================================
# ATTO 1 — SPAZIO UMANO (2D)
# =====================================================
st.header("👀 ATTO 1 — Come vediamo noi i dati (2D)")

x2 = "Milk"
y2 = "Grocery"

fig_2d_human = px.scatter(
    df,
    x=x2,
    y=y2,
    color="Cluster",
    labels={
        x2: feature_names_it[x2],
        y2: feature_names_it[y2]
    },
    title="Clienti nello spazio reale (2D)"
)

fig_2d_human.update_layout(
    title_font_size=TITLE_SIZE,
    font=dict(size=PLOT_FONT_SIZE)
)

st.plotly_chart(fig_2d_human, use_container_width=True)

st.info("""
Guardiamo solo **due aspetti della spesa**.
Sembra tutto piuttosto confuso.
""")

# =====================================================
# ATTO 1 — SPAZIO UMANO (3D)
# =====================================================
st.subheader("👀 Spazio umano tridimensionale")

x3, y3, z3 = "Fresh", "Milk", "Delicassen"

fig_3d_human = px.scatter_3d(
    df,
    x=x3,
    y=y3,
    z=z3,
    color="Cluster",
    labels={
        x3: feature_names_it[x3],
        y3: feature_names_it[y3],
        z3: feature_names_it[z3]
    },
    title="Clienti nello spazio reale (3D)"
)

fig_3d_human.update_layout(
    title_font_size=TITLE_SIZE,
    font=dict(size=PLOT_FONT_SIZE)
)

st.plotly_chart(fig_3d_human, use_container_width=True)

st.warning("""
Con 3 dimensioni iniziamo già a perdere orientamento.
""")

# =====================================================
# PCA
# =====================================================
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]
df["PCA3"] = X_pca[:, 2]

loadings = pd.DataFrame(
    pca.components_.T,
    columns=["PCA1", "PCA2", "PCA3"],
    index=[feature_names_it[f] for f in features]
)

# =====================================================
# ATTO 2 — PCA 2D
# =====================================================
st.header("🧠 ATTO 2 — L'algoritmo inventa nuovi assi (PCA 2D)")

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

fig_pca_2d.update_layout(
    title_font_size=TITLE_SIZE,
    font=dict(size=PLOT_FONT_SIZE)
)

st.plotly_chart(fig_pca_2d, use_container_width=True)

st.success("""
Questi assi **non esistevano**.
Sono stati creati per rappresentare **stili di acquisto**.
""")

# =====================================================
# ATTO 3 — PCA 3D
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

fig_pca_3d.update_layout(
    title_font_size=TITLE_SIZE,
    font=dict(size=PLOT_FONT_SIZE)
)

st.plotly_chart(fig_pca_3d, use_container_width=True)

st.warning("""
Qui arriviamo al limite della visualizzazione umana.
L'algoritmo, però, può ragionare in molte più dimensioni.
""")

# =====================================================
# CHIUSURA
# =====================================================
st.markdown("---")
st.markdown("""
### 🎯 Messaggio finale

> L’intelligenza artificiale  
> non guarda i clienti come noi.  
>  
> Li **proietta** in uno spazio  
> dove emergono schemi invisibili.
""")
