import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import plotly.express as px

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Clustering & PCA Demo",
    layout="wide"
)

st.title("🔍 Clustering e Riduzione Dimensionale")
st.markdown("""
Questa demo mostra come un algoritmo **ragiona nello spazio multidimensionale**
e come possiamo **visualizzarlo** con tecniche di riduzione dimensionale.
""")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
    return pd.read_csv(url)

df = load_data()

features = [
    "Fresh", "Milk", "Grocery",
    "Frozen", "Detergents_Paper", "Delicassen"
]

X = df[features]

# -----------------------------
# PREPROCESSING
# -----------------------------
X_log = np.log1p(X)  # log scaling per effetto visivo + ML corretto
X_scaled = StandardScaler().fit_transform(X_log)

# -----------------------------
# CLUSTERING
# -----------------------------
k = st.sidebar.slider("Numero di cluster (KMeans)", 2, 6, 3)

kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters.astype(str)

# ============================================================
# 1️⃣ 2D LOG SPACE
# ============================================================
st.header("1️⃣ Spazio reale (2D – scala log)")

col1, col2 = st.columns(2)
x_feat = col1.selectbox("Asse X", features, index=0)
y_feat = col2.selectbox("Asse Y", features, index=2)

fig_2d_log = px.scatter(
    df,
    x=np.log1p(df[x_feat]),
    y=np.log1p(df[y_feat]),
    color="Cluster",
    title=f"{x_feat} vs {y_feat} (log scale)",
    labels={"x": x_feat, "y": y_feat}
)

st.plotly_chart(fig_2d_log, use_container_width=True)

st.info("""
Ogni punto è un cliente.  
Gli assi sono **dimensioni reali**, ma trasformate per rendere visibile la struttura.
""")

# ============================================================
# 2️⃣ 3D LOG SPACE
# ============================================================
st.header("2️⃣ Spazio reale (3D – scala log)")

x3 = st.selectbox("Asse X (3D)", features, index=0)
y3 = st.selectbox("Asse Y (3D)", features, index=1)
z3 = st.selectbox("Asse Z (3D)", features, index=2)

fig_3d_log = px.scatter_3d(
    df,
    x=np.log1p(df[x3]),
    y=np.log1p(df[y3]),
    z=np.log1p(df[z3]),
    color="Cluster",
    title="Spazio tridimensionale reale (log)",
)

st.plotly_chart(fig_3d_log, use_container_width=True)

st.warning("""
Già con 3 dimensioni iniziamo a **perdere intuizione visiva**.  
In realtà l'algoritmo lavora in **6 dimensioni contemporaneamente**.
""")

# ============================================================
# PCA
# ============================================================
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]
df["PCA3"] = X_pca[:, 2]

# ============================================================
# 3️⃣ PCA 2D
# ============================================================
st.header("3️⃣ Riduzione dimensionale (PCA 2D)")

fig_pca_2d = px.scatter(
    df,
    x="PCA1",
    y="PCA2",
    color="Cluster",
    title="PCA – 2 dimensioni latenti"
)

st.plotly_chart(fig_pca_2d, use_container_width=True)

st.success("""
Qui gli assi **non sono variabili reali**  
ma **combinazioni intelligenti** che spiegano il comportamento dei clienti.
""")


# -----------------------------
# PCA + LOADINGS
# -----------------------------
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

loadings = pd.DataFrame(
    pca.components_.T,
    columns=["PCA1", "PCA2", "PCA3"],
    index=features
)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]
df["PCA3"] = X_pca[:, 2]

# ============================================================
# 4️⃣ PCA 3D
# ============================================================
st.header("4️⃣ PCA 3D – Spazio latente")

fig_pca_3d = px.scatter_3d(
    df,
    x="PCA1",
    y="PCA2",
    z="PCA3",
    color="Cluster",
    title="PCA – Spazio tridimensionale latente"
)

st.plotly_chart(fig_pca_3d, use_container_width=True)

st.info("""
Questo è il massimo che possiamo **vedere**.  
L'algoritmo però **non ha limiti di dimensione**.
""")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
**Messaggio chiave:**  
> L'intelligenza artificiale non semplifica i dati.  
> Li **riorganizza** per renderli comprensibili.
""")
