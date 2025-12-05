import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def run():
    st.title("📊 Clustering Clienti Telco")
    st.write("""
    Questo esempio mostra come segmentare i clienti Telco per supportare strategie
    di **upselling**, **retention** e **anti-churn**.
    """)

    # ------------------------
    # 1. Dataset sintetico
    # ------------------------
    st.header("1. Dataset di esempio")

    np.random.seed(42)

    n = 300
    data = pd.DataFrame({
        "ticket_aperti": np.random.poisson(2, n),
        "anzianita_mesi": np.random.normal(30, 10, n).clip(1, 60).astype(int),
        "spesa_mensile": np.random.normal(35, 15, n).clip(10, 100),
        "offerta": np.random.choice(["Base", "Plus", "Premium"], n, p=[0.5, 0.3, 0.2])
    })

    st.dataframe(data.head())

    # Codifica delle variabili categoriche
    df_model = pd.get_dummies(data, columns=["offerta"], drop_first=True)

    # ------------------------
    # 2. Selezione dei parametri
    # ------------------------
    st.header("2. Parametri del modello")

    k = st.slider("Numero di cluster (k)", min_value=2, max_value=8, value=4)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_model)

    # ------------------------
    # 3. Training del modello
    # ------------------------
    st.header("3. Training")

    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    data["cluster"] = clusters
    st.success("Modello addestrato con successo!")

    # ------------------------
    # 4. Visualizzazione dei cluster
    # ------------------------
    st.header("4. Visualizzazione dei cluster")
    st.write("Grafico 2D: Spesa Mensile vs Anzianità")

    fig, ax = plt.subplots(figsize=(8,5))
    scatter = ax.scatter(
        data["anzianita_mesi"],
        data["spesa_mensile"],
        c=data["cluster"],
        alpha=0.7
    )
    plt.xlabel("Anzianità (mesi)")
    plt.ylabel("Spesa mensile (€)")
    plt.title("Cluster dei clienti")
    plt.grid(True)

    st.pyplot(fig)

    # ------------------------
    # 5. Insight business
    # ------------------------
    st.header("5. Insight Business")

    st.write("""
    Segmentare i clienti Telco permette di:
    - Identificare clienti **ad alto valore** → campagne di upselling mirate  
    - Individuare clienti **a rischio churn** (molti ticket, bassa spesa, bassa fedeltà)  
    - Personalizzare **offerta commerciale**, sconti e bundle  
    - Migliorare la gestione del servizio clienti  
    """)

    st.subheader("Distribuzione clienti per cluster")
    st.bar_chart(data["cluster"].value_counts().sort_index())

    st.subheader("Statistiche di cluster")
    st.dataframe(data.groupby("cluster").mean(numeric_only=True))
