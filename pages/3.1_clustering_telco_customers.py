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
    page_title="Come ragiona un algoritmo – TELCO",
    layout="wide"
)

PLOT_FONT_SIZE = 18
AXIS_TITLE_SIZE = 22
TITLE_SIZE = 26
LEGEND_SIZE = 20

# =====================================================
# TITLE
# =====================================================
st.title("📡 Come ragiona un algoritmo di Clustering (caso TELCO)")

st.markdown("""
Un viaggio visivo:
dal **punto di vista umano**  
al **punto di vista dell’algoritmo**  
applicato ai **clienti Telco**.
""")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv("telco_customers_synthetic.csv")

df = load_data()

feature_map = {
    "Tipo di prodotto": "product_complexity",
    "Consumo dati medio (GB/mese)": "avg_data_usage",
    "Segnalazioni aperte": "open_tickets",
    "Densità area geografica": "region_density",
    "Anzianità cliente (mesi)": "customer_tenure",
    "Sensibilità al prezzo": "price_sensitivity"
}

features = list(feature_map.values())
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
    "Profilo di Valore e Utilizzo"
)

pca2_name = st.sidebar.text_input(
    "PCA 2",
    "Stabilità e Complessità Operativa"
)

pca3_name = st.sidebar.text_input(
    "PCA 3",
    "Comportamento Tattico di Nicchia"
)

# =====================================================
# PREPROCESSING
# =====================================================
X_log = np.log1p(X)
X_scaled = StandardScaler().fit_transform(X_log)

kmeans = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled).astype(str)

# =====================================================
# ATTO 1 — SPAZIO UMANO 2D
# =====================================================
st.header("👀 ATTO 1 — Come vediamo noi i clienti (2D)")

st.info("""
Stiamo osservando **solo due dimensioni alla volta**.
Il cervello umano funziona così.
""")

col1, col2 = st.columns(2)

x2_label = col1.selectbox(
    "Asse X",
    list(feature_map.keys()),
    index=1
)
y2_label = col2.selectbox(
    "Asse Y",
    list(feature_map.keys()),
    index=2
)

fig_2d = px.scatter(
    df,
    x=feature_map[x2_label],
    y=feature_map[y2_label],
    color="Cluster",
    labels={
        feature_map[x2_label]: x2_label,
        feature_map[y2_label]: y2_label
    },
    title="Vista bidimensionale dei clienti Telco"
)

fig_2d.update_layout(font=dict(size=PLOT_FONT_SIZE))
st.plotly_chart(fig_2d, use_container_width=True)

# =====================================================
# ATTO 1 — SPAZIO UMANO 3D
# =====================================================
st.subheader("👀 Spazio umano tridimensionale")

x3_label = st.selectbox("Asse X (3D)", list(feature_map.keys()), index=1)
y3_label = st.selectbox("Asse Y (3D)", list(feature_map.keys()), index=0)
z3_label = st.selectbox("Asse Z (3D)", list(feature_map.keys()), index=3)

fig_3d = px.scatter_3d(
    df,
    x=feature_map[x3_label],
    y=feature_map[y3_label],
    z=feature_map[z3_label],
    color="Cluster",
    labels={
        feature_map[x3_label]: x3_label,
        feature_map[y3_label]: y3_label,
        feature_map[z3_label]: z3_label
    },
    title="Vista tridimensionale dei clienti Telco"
)

st.plotly_chart(fig_3d, use_container_width=True)

st.warning("""
Con 3 dimensioni iniziamo già a perdere intuizione.
E ne stiamo ignorando altre 3.
""")

# =====================================================
# ATTO 2 — PCA
# =====================================================
st.header("🧠 ATTO 2 — L'algoritmo cambia punto di vista (PCA)")

st.info("""
La PCA **non crea i gruppi**.  
Crea uno spazio migliore per farli emergere.
""")

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]
df["PCA3"] = X_pca[:, 2]

loadings = pd.DataFrame(
    pca.components_.T,
    columns=["PCA1", "PCA2", "PCA3"],
    index=feature_map.keys()
)

# =====================================================
# PCA 2D
# =====================================================
st.subheader("🔍 Nuovi assi creati dall'algoritmo (PCA 2D)")

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

fig_pca_2d.update_layout(font=dict(size=PLOT_FONT_SIZE))
st.plotly_chart(fig_pca_2d, use_container_width=True)

st.success("""
Ora i gruppi iniziano a emergere chiaramente.
""")

# =====================================================
# PCA 3D
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
Questo è il limite della visualizzazione umana.
L’algoritmo, invece, non ha limiti dimensionali.
""")

# =====================================================
# CHIUSURA
# =====================================================
st.markdown("---")

st.markdown("""
### 🎯 Messaggio finale

> L’algoritmo non scopre i gruppi perché li vede.  
> Li scopre perché **cambia punto di vista**.
""")
