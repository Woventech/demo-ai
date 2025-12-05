import streamlit as st

st.set_page_config(
    page_title="AI & ML Telco Demo",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI & Machine Learning Lab - Telco")

st.markdown("""
Benvenuto nell’ambiente di test per modelli ML e AI dedicati alla Telco.

Usa il menu a sinistra per navigare tra i vari business case:
- **Classificazione Ticket**
- **Clustering**
- **Forecasting**
- **Anomaly Detection**
- e molto altro...
""")

st.info("Seleziona un esperimento dal menu a sinistra per iniziare 👈")
