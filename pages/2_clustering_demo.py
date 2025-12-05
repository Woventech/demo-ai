import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(page_title="Clustering Clienti Telco", layout="wide")

st.title("📊 Clustering Clienti Telco – Procedura Guidata")

# ----------------------------
# INIZIALIZZAZIONE SESSION STATE
# ----------------------------
for key in ["data", "processed", "model", "clusters"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("Navigazione")
page = st.sidebar.radio(
    "Fasi del modello:",
    [
        "1️⃣ Caricamento Dati",
        "2️⃣ Pulizia & Feature Engineering",
        "3️⃣ Training",
        "4️⃣ Visualizzazione Cluster",
        "5️⃣ Insight Business"
    ]
)


# ----------------------------------------------------------
# 1. CARICAMENTO DATI
# ----------------------------------------------------------
if page == "1️⃣ Caricamento Dati":
    st.header("📥 1. Caricamento Dataset")

    option = st.radio("Scegli come ottenere i dati:",
                      ["Usa dataset di esempio", "Carica un tuo CSV"])

    if option == "Usa dataset di esempio":
        if st.button("📌 Genera Dataset di Esempio"):
            np.random.seed(42)
            n = 300

            data = pd.DataFrame({
                "ticket_aperti": np.random.poisson(2, n),
                "anzianita_mesi": np.random.normal(30, 10, n).clip(1, 60).astype(int),
                "spesa_mensile": np.random.normal(35, 15, n).clip(10, 100),
                "offerta": np.random.choice(["Base", "Plus", "Premium"], n, p=[0.5, 0.3, 0.2])
            })

            st.session_state.data = data
            st.session_state.processed = None
            st.session_state.model = None
            st.session_state.clusters = None

            st.success("Dataset generato correttamente!")

    else:
        uploaded = st.file_uploader("Carica un file CSV", type="csv")
        if uploaded:
            st.session_state.data = pd.read_csv(uploaded)
            st.session_state.processed = None
            st.session_state.model = None
            st.session_state.clusters = None
            st.success("File caricato!")

    if st.session_state.data is not None:
        st.subheader("📄 Anteprima")
        st.dataframe(st.session_state.data.head())


# ----------------------------------------------------------
# 2. PULIZIA E PREPARAZIONE
# ----------------------------------------------------------
elif page == "2️⃣ Pulizia & Feature Engineering":
    st.header("🧹 2. Pulizia e Preparazione Dati")

    if st.session_state.data is None:
        st.warning("⚠ Prima carica un dataset!")
        st.stop()

    st.write("""
    In questa fase eseguiamo:
    - Codifica variabili categoriche (one-hot encoding)
    - Standardizzazione delle variabili numeriche
    """)

    if st.button("⚙️ Esegui Pulizia & Preparazione"):
        data = st.session_state.data.copy()

        # Encoding
        processed = pd.get_dummies(data, columns=["offerta"], drop_first=True)

        # Scaling
        scaler = StandardScaler()
        processed_scaled = scaler.fit_transform(processed)

        st.session_state.processed = processed_scaled

        st.success("Pulizia completata! Ora puoi passare al training.")

    if st.session_state.processed is not None:
        st.info("✔ Dati pronti per il training!")


# ----------------------------------------------------------
# 3. TRAINING
# ----------------------------------------------------------
elif page == "3️⃣ Training":
    st.header("🤖 3. Addestramento Modello KMeans")

    if st.session_state.processed is None:
        st.warning("⚠ Prima esegui la pulizia dei dati!")
        st.stop()

    k = st.slider("Scegli il numero di cluster", 2, 10, 4)

    if st.button("🚀 Avvia Training"):
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(st.session_state.processed)

        st.session_state.model = kmeans
        st.session_state.clusters = clusters
        st.session_state.data["cluster"] = clusters

        st.success("Modello addestrato correttamente!")

        st.write("Distribuzione cluster:")
        st.bar_chart(st.session_state.data["cluster"].value_counts().sort_index())

    if st.session_state.model is not None:
        st.info("✔ Training completato!")


# ----------------------------------------------------------
# 4. VISUALIZZAZIONE
# ----------------------------------------------------------
elif page == "4️⃣ Visualizzazione Cluster":
    st.header("📈 4. Visualizzazione")

    if st.session_state.model is None:
        st.warning("⚠ Prima addestra il modello!")
        st.stop()

    data = st.session_state.data

    st.write("Scatter plot dei cluster basato su due variabili chiave.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        data["anzianita_mesi"],
        data["spesa_mensile"],
        c=data["cluster"],
        alpha=0.7
    )

    ax.set_xlabel("Anzianità (mesi)")
    ax.set_ylabel("Spesa mensile (€)")
    ax.set_title("Cluster Clienti")
    ax.grid(True)

    st.pyplot(fig)


# ----------------------------------------------------------
# 5. INSIGHT BUSINESS
# ----------------------------------------------------------
elif page == "5️⃣ Insight Business":
    st.header("💡 5. Insight Business")

    if st.session_state.model is None:
        st.warning("⚠ Prima addestra il modello!")
        st.stop()

    data = st.session_state.data

    st.subheader("Distribuzione cluster")
    st.bar_chart(data["cluster"].value_counts().sort_index())

    st.subheader("Statistiche medie per cluster")
    st.dataframe(data.groupby("cluster").mean(numeric_only=True))

    st.write("""
    ### 🎯 Utilizzi Business del Clustering
    - Identificazione clienti ad **alto potenziale** per upselling  
    - Rilevazione clienti **a rischio churn** (ticket alti, bassa spesa)  
    - Creazione di offerte personalizzate  
    - Segmentazione per priorità nella gestione ticket  
    """)

