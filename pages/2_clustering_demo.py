import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.title("📊 Clustering Demo")
st.markdown("Esempio semplice di clustering con K-Means su un dataset numerico.")

# Upload dataset
uploaded = st.file_uploader("Carica un CSV numerico", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Anteprima Dataset")
    st.dataframe(df.head())

    if st.button("Esegui Clustering"):
        # Preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))

        # K-Means
        k = st.slider("Numero cluster (k)", 2, 10, 3)
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X_scaled)

        # Output
        df["cluster"] = labels
        st.success("Clustering completato!")

        st.subheader("📌 Risultati")
        st.dataframe(df)

        # Plot
        fig, ax = plt.subplots()
        ax.scatter(X_scaled[:,0], X_scaled[:,1], c=labels)
        ax.set_title("Visualizzazione Cluster (prime 2 features)")
        st.pyplot(fig)
