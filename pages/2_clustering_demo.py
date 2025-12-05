import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Configurazione pagina
st.set_page_config(page_title="Clustering Clienti Telco", layout="wide")

st.title("📊 Clustering Clienti Telco")

# ------------------------
# SIDEBAR DI NAVIGAZIONE
# ------------------------
st.sidebar.title("Navigazione")
page = st.sidebar.radio(
    "Seleziona fase:",
    [
        "1. Caricamento Dati",
        "2. Pulizia e Preparazione",
        "3. Training Modello",
        "4. Visualizzazione Cluster",
        "5. Insight Business"
    ]
)

# Variabili di sessione per mantenere dataset e modello
if "data" not in st.session_state:
    st.session_state.data = None
if "processed" not in st.session_state:
    st.session_state.processed = None
if "model" not in st.session_state:
    st.session_state.model = None


# ----------------------------------------------------------
# 1. CARICAMENTO DATI
# ----------------------------------------------------------
if page == "1. Caricamento Dati":
    st.header("📊 Caricamento Dataset")

    option = st.radio("Scegli un'opzione:",
                      ["Usa dataset di esempio", "Carica un tuo CSV"])

    if option == "Usa dataset di esempio":
        if st.button("Genera Dataset di Esempio"):
            np.random.seed(42)
            n = 300

            data = pd.DataFrame({
                "ticket_aperti": np.random.poisson(2, n),
                "anzianita_mesi": np.random.normal(30, 10, n).clip(1, 60).astype(int),
                "spesa_mensile": np.random.normal(35, 15, n).clip(10, 100),
                "offerta": np.random.choice(["Base", "Plus", "Premium"], n, p=[0.5, 0.3, 0.2])
            })

            st.session_state.data = data
            st.success("Dataset generato!")

    else:
        uploaded = st.file_uploader("Carica un file CSV", type="csv")
        if uploaded:
            st.session_state.data = pd.read_csv(uploaded)
            st.success("File caricato!")

    if st.session_state.data is not None:
        st.subheader("Anteprima Dataset")
        st.dataframe(st.session_state.data.head())


# ----------------------------------------------------------
# 2. PULIZIA E PREPARAZIONE
# ----------------------------------------------------------
elif page == "2. Pulizia e Preparazione":
    st.header("🧹 Pulizia & Feature Engineering")

    if st.session_state.data is None:
        st.warning("Carica prima un dataset!")
        st.stop()

    data = st.session_state.data.copy()

    st.write("➕ Codifica variabile categorica 'offerta'")
    processed = pd.get_dummies(data, columns=["offerta"], drop_first=True)

    st.write("📏 Standardizzazione")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(processed)

    st.session_state.processed = X_scaled
    st.success("Preparazione completata!")


# ----------------------------------------------------------
# 3. TRAINING MODELLO
# ----------------------------------------------------------
elif page == "3. Training Modello":
    st.header("🤖 Training KMeans")

    if st.session_state.processed is None:
        st.warning("Prima esegui la preparazione dati!")
        st.stop()

    k = st.slider("Numero di cluster (k)", min_value=2, max_value=10, value=4)

    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(st.session_state.processed)

    st.session_state.data["cluster"] = clusters
    st.session_state.model = kmeans

    st.success("Modello addestrato con successo!")
    st.write("Distribuzione cluster:")
    st.bar_chart(st.session_state.data["cluster"].value_counts().sort_index())


# ----------------------------------------------------------
# 4. VISUALIZZAZIONE CLUSTER
# ----------------------------------------------------------
elif page == "4. Visualizzazione Cluster":
    st.header("📈 Visualizzazione Cluster")

    if st.session_state.model is None:
        st.warning("Prima addestra il modello!")
        st.stop()

    data = st.session_state.data

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        data["anzianita_mesi"],
        data["spesa_mensile"],
        c=data["cluster"],
        alpha=0.7
    )
    ax.set_xlabel("Anzianità (mesi)")
    ax.set_ylabel("Spesa mensile (€)")
    ax.set_title("Cluster dei clienti")
    ax.grid(True)

    st.pyplot(fig)


# ----------------------------------------------------------
# 5. INSIGHT BUSINESS
# ----------------------------------------------------------
elif page == "5. Insight Business":
    st.header("💡 Insight Business")

    if st.session_state.model is None:
        st.warning("Prima addestra il modello!")
        st.stop()

    data = st.session_state.data

    st.subheader("📦 Distribuzione clienti per cluster")
    st.bar_chart(data["cluster"].value_counts().sort_index())

    st.subheader("📊 Statistiche dei cluster")
    st.dataframe(data.groupby("cluster").mean(numeric_only=True))

    st.write("""
    ### Come usare questi cluster nel business:
    - Identificare clienti ad **alto valore** → campagne di upselling  
    - Individuare clienti **a rischio churn** (ticket alti, bassa spesa)  
    - Personalizzare offerte e bundle in base al cluster  
    - Migliorare il prioritization dei ticket  
    """)

