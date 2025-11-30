import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Configurazione pagina
st.set_page_config(page_title="Classificatore Ticket Telco", layout="wide")

# Titolo
st.title("🎫 Sistema di Classificazione Ticket Clienti - Telco")
st.markdown("---")

# Inizializzazione session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'categories' not in st.session_state:
    st.session_state.categories = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}

# Sidebar per navigazione
st.sidebar.title("Navigazione")
page = st.sidebar.radio("Seleziona fase:", 
                        ["1. Caricamento Dati", 
                         "2. Pulizia e Preparazione", 
                         "3. Training Modello", 
                         "4. Valutazione", 
                         "5. Classificazione"])

# Pagina 1: Caricamento Dati
if page == "1. Caricamento Dati":
    st.header("📊 Caricamento Dataset")
    
    option = st.radio("Scegli un'opzione:", 
                      ["Usa dataset di esempio", "Carica il tuo CSV"])
    
    if option == "Usa dataset di esempio":
        if st.button("Genera Dataset di Esempio"):
            # Dataset di esempio per telco
            data = {
                'ticket_id': range(1, 201),
                'descrizione': [
                    # Internet (50)
                    "internet lento non riesco a navigare", "connessione wifi non funziona",
                    "velocità internet molto bassa", "router non si connette", 
                    "problema con fibra ottica", "internet si disconnette continuamente",
                    "non riesco ad accedere a internet", "velocità download insufficiente",
                    "wifi instabile in casa", "modem non risponde",
                    "internet non funziona da ieri", "connessione molto lenta",
                    "impossibile collegarsi alla rete", "problema di connettività",
                    "velocità non corrisponde al contratto", "wifi non raggiunge alcune stanze",
                    "fibra non funzionante", "internet cade spesso",
                    "problemi con router fornito", "segnale wifi debole",
                    "non posso lavorare da casa per internet lento", "streaming impossibile",
                    "videochiamate interrotte per connessione", "ping molto alto nei giochi",
                    "impossibile caricare pagine web", "timeout continui",
                    "router si riavvia da solo", "luci router rosse",
                    "configurazione wifi non funziona", "password wifi non accettata",
                    "non vedo la rete wifi", "connessione limitata",
                    "ethernet non funziona", "porte router non rispondono",
                    "QoS non configurato correttamente", "bandwidth limitato",
                    "latenza elevata", "packet loss continuo",
                    "DNS non risolvono", "IP non assegnato",
                    "DHCP non funzionante", "bridge mode non attivo",
                    "port forwarding non funziona", "firewall blocca tutto",
                    "VPN non si connette", "double NAT problem",
                    "IPv6 non configurato", "routing table errata",
                    "MTU size sbagliato", "ISP throttling sospetto",
                ] + [
                    # Telefonia (50)
                    "non riesco a chiamare", "telefono senza linea",
                    "chiamate interrotte", "non sento l'interlocutore",
                    "eco durante le chiamate", "volume basso",
                    "impossibile ricevere chiamate", "linea telefonica morta",
                    "disturbi sulla linea", "fruscio continuo",
                    "caduta chiamate", "qualità audio pessima",
                    "non funziona il telefono fisso", "squilli ma non passa chiamata",
                    "linea sempre occupata", "toni di chiamata assenti",
                    "identificativo chiamante non funziona", "segreteria non attiva",
                    "trasferimento chiamata non funziona", "conferenza telefonica impossibile",
                    "chiamate internazionali bloccate", "numeri verdi non raggiungibili",
                    "deviazione chiamata non funziona", "avviso di chiamata assente",
                    "chiamate verso cellulari impossibili", "linea con rumori strani",
                    "telefono non suona", "cornetta non funziona",
                    "tasti telefono non rispondono", "display telefono spento",
                    "voicemail non accessibile", "chiamate verso fissi non passano",
                    "linea telefonica gracchia", "interferenze sulla linea",
                    "manca il segnale di libero", "tono di occupato continuo",
                    "numero non raggiungibile", "servizio temporaneamente non disponibile",
                    "codec audio non supportato", "protocollo SIP non funziona",
                    "registrazione VoIP fallita", "jitter elevato su VoIP",
                    "quality of service voce bassa", "echo cancellation non attivo",
                    "DTMF tones non funzionano", "caller ID spoofing",
                    "trunk SIP disconnesso", "peer SIP unreachable",
                    "authentication VoIP fallita", "RTP stream interrotto",
                ] + [
                    # Fatturazione (50)
                    "addebito non corretto sulla fattura", "importo fattura sbagliato",
                    "costo superiore al previsto", "servizi non richiesti in fattura",
                    "doppio addebito", "fattura non arrivata",
                    "richiesta rimborso", "promozione non applicata",
                    "sconto non presente in fattura", "costi nascosti",
                    "fattura precedente errata", "pagamento non registrato",
                    "bolletta troppo alta", "consumi non reali",
                    "canone errato", "extra non autorizzati",
                    "fattura duplicata", "credito non applicato",
                    "piano tariffario sbagliato", "costo attivazione non dovuto",
                    "penale non giustificata", "interessi non dovuti",
                    "dettaglio chiamate errato", "traffico dati non corrispondente",
                    "roaming non richiesto addebitato", "servizi premium non attivati",
                    "abbonamento non cancellato", "periodo di fatturazione sbagliato",
                    "IVA calcolata male", "arrotondamenti strani",
                    "contributi non spiegati", "tasse non dovute",
                    "costo recesso non previsto", "deposito cauzionale non restituito",
                    "bonus non accreditato", "cashback non ricevuto",
                    "fattura in formato sbagliato", "domiciliazione non attiva",
                    "metodo di pagamento non funziona", "carta di credito rifiutata",
                    "addebito diretto fallito", "bollettino non valido",
                    "QR code fattura non leggibile", "PagoPA non funzionante",
                    "split payment non applicato", "reverse charge errato",
                    "ritenuta d'acconto sbagliata", "partita IVA non riconosciuta",
                    "codice fiscale errato in fattura", "fattura elettronica non pervenuta",
                ] + [
                    # Contratto/Attivazione (50)
                    "attivazione nuova linea", "migrazione da altro operatore",
                    "cambio piano tariffario", "disdetta contratto",
                    "modifica dati contrattuali", "aggiunta servizi al contratto",
                    "richiesta portabilità numero", "stato pratica attivazione",
                    "problemi con attivazione", "contratto non ancora attivo",
                    "ritardo nell'attivazione", "documentazione contrattuale",
                    "recesso anticipato", "rinnovo contratto",
                    "condizioni contrattuali", "vincolo contrattuale",
                    "penali recesso", "modifica intestatario",
                    "aggiunta secondo intestatario", "cambio indirizzo fatturazione",
                    "upgrade piano", "downgrade piano",
                    "numero seriale SIM", "cambio tecnologia connessione",
                    "passaggio da ADSL a fibra", "verifica copertura indirizzo",
                    "preventivo servizi", "offerte disponibili",
                    "promozione in corso", "portabilità da mobile a fisso",
                    "numero temporaneo", "attivazione servizi opzionali",
                    "disattivazione servizi non richiesti", "modifica dati anagrafici",
                    "cambio numero di telefono", "richiesta duplicato SIM",
                    "PUK bloccato", "PIN dimenticato",
                    "contratto business", "passaggio da privato a business",
                    "cessione contratto", "subentro intestatario",
                    "voltura utenza", "sospensione temporanea servizio",
                    "riattivazione dopo sospensione", "migrazione tecnologia",
                    "cambio profilo contrattuale", "adeguamento GDPR",
                    "consensi privacy", "opt-out marketing",
                    "richiesta copia contratto", "modifica clausole contrattuali",
                    "rateizzazione costi attivazione", "conferma ordine",
                    "cancellazione ordine", "stato lavorazione pratica",
                    "documenti identità richiesti", "firma digitale contratto",
                    "accettazione condizioni", "periodo di ripensamento",
                    "diritto di recesso", "preavviso disdetta",
                    "trasferimento contratto", "cessione linea",
                    "subentro contrattuale", "cambio ragione sociale",
                    "variazione sede legale", "aggiornamento PEC",
                    "modifica IBAN addebito", "carta di credito non accettata",
                    "domiciliazione bancaria", "pagamento anticipato",
                    "verifica identità cliente", "procedura antiriciclaggio",
                    "documenti contrattuali mancanti", "condizioni generali servizio",
                    "informativa privacy incompleta", "consenso trattamento dati",
                    "richiesta portabilità dati", "esercizio diritti GDPR",
                    "opposizione profilazione", "cancellazione dati personali",
                    "rettifica informazioni", "limitazione trattamento",
                    "trasferimento numero verde", "attivazione numerazione dedicata",
                    "richiesta numero premium", "blocco numerazione",
                    "sblocco servizi voce", "attivazione roaming internazionale",
                    "disattivazione servizi estero", "configurazione APN dati",
                    "parametri configurazione modem", "credenziali accesso account",
                ]
            }
            
            categories = ['Internet'] * 50 + ['Telefonia'] * 50 + ['Fatturazione'] * 50 + ['Contratto'] * 50
            
            data['categoria'] = categories
            
            st.session_state.df = pd.DataFrame(data)
            st.success("✅ Dataset di esempio generato!")
    
    else:
        uploaded_file = st.file_uploader("Carica file CSV", type=['csv'])
        if uploaded_file:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success("✅ File caricato con successo!")
    
    if st.session_state.df is not None:
        st.subheader("Anteprima Dataset")
        st.dataframe(st.session_state.df.head(10))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Totale Ticket", len(st.session_state.df))
        with col2:
            st.metric("Colonne", len(st.session_state.df.columns))
        with col3:
            if 'categoria' in st.session_state.df.columns:
                st.metric("Categorie", st.session_state.df['categoria'].nunique())

# Pagina 2: Pulizia e Preparazione
elif page == "2. Pulizia e Preparazione":
    st.header("🧹 Pulizia e Preparazione Dati")
    
    if st.session_state.df is None:
        st.warning("⚠️ Carica prima un dataset dalla sezione 'Caricamento Dati'")
    else:
        st.subheader("Dataset Originale")
        st.write(f"Shape: {st.session_state.df.shape}")
        
        # Verifica valori mancanti
        st.subheader("Analisi Valori Mancanti")
        missing = st.session_state.df.isnull().sum()
        if missing.sum() > 0:
            st.dataframe(missing[missing > 0])
        else:
            st.success("✅ Nessun valore mancante!")
        
        # Distribuzione categorie
        if 'categoria' in st.session_state.df.columns:
            st.subheader("Distribuzione Categorie")
            fig, ax = plt.subplots(figsize=(10, 5))
            st.session_state.df['categoria'].value_counts().plot(kind='bar', ax=ax)
            ax.set_title("Distribuzione Ticket per Categoria")
            ax.set_xlabel("Categoria")
            ax.set_ylabel("Numero Ticket")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        # Pulizia dati
        if st.button("Pulisci e Prepara Dati"):
            df_clean = st.session_state.df.copy()
            
            # Rimozione duplicati
            before = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            after = len(df_clean)
            
            # Rimozione valori mancanti
            df_clean = df_clean.dropna()
            
            # Conversione testo in minuscolo
            if 'descrizione' in df_clean.columns:
                df_clean['descrizione'] = df_clean['descrizione'].str.lower().str.strip()
            
            st.session_state.df = df_clean
            
            st.success(f"✅ Dati puliti! Rimossi {before - after} duplicati")
            st.write(f"Dataset finale: {df_clean.shape}")

# Pagina 3: Training
elif page == "3. Training Modello":
    st.header("🤖 Training del Modello")
    
    if st.session_state.df is None:
        st.warning("⚠️ Carica e prepara prima un dataset")
    else:
        st.subheader("Configurazione Training")
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Percentuale Test Set", 10, 40, 20) / 100
        with col2:
            random_state = st.number_input("Random State", 0, 100, 42)
        
        if st.button("Avvia Training", type="primary"):
            with st.spinner("Training in corso..."):
                df = st.session_state.df
                
                # Preparazione dati
                X = df['descrizione']
                y = df['categoria']
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
                
                # Vectorization
                vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
                X_train_vec = vectorizer.fit_transform(X_train)
                X_test_vec = vectorizer.transform(X_test)
                
                # Training
                model = MultinomialNB()
                model.fit(X_train_vec, y_train)
                
                # Predizioni
                y_pred = model.predict(X_test_vec)
                y_pred_proba = model.predict_proba(X_test_vec)
                
                # Salvataggio in session state
                st.session_state.model = model
                st.session_state.vectorizer = vectorizer
                st.session_state.categories = model.classes_
                st.session_state.metrics = {
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                st.success("✅ Training completato!")
                
                # Mostra accuracy
                accuracy = st.session_state.metrics['report']['accuracy']
                st.metric("Accuracy", f"{accuracy:.2%}")

# Pagina 4: Valutazione
elif page == "4. Valutazione":
    st.header("📈 Valutazione del Modello")
    
    if st.session_state.model is None:
        st.warning("⚠️ Effettua prima il training del modello")
    else:
        metrics = st.session_state.metrics
        
        # Metriche generali
        st.subheader("Metriche di Performance")
        col1, col2, col3 = st.columns(3)
        
        report = metrics['report']
        with col1:
            st.metric("Accuracy", f"{report['accuracy']:.2%}")
        with col2:
            st.metric("Macro Avg F1", f"{report['macro avg']['f1-score']:.2%}")
        with col3:
            st.metric("Weighted Avg F1", f"{report['weighted avg']['f1-score']:.2%}")
        
        # Report dettagliato
        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'])
        st.dataframe(report_df.style.format("{:.2%}"))
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(metrics['y_test'], metrics['y_pred'])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=st.session_state.categories,
                    yticklabels=st.session_state.categories, ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_ylabel("Vera Categoria")
        ax.set_xlabel("Categoria Predetta")
        st.pyplot(fig)
        
        # ROC Curve (per classificazione multi-classe)
        st.subheader("ROC Curves")
        from sklearn.preprocessing import label_binarize
        
        y_test_bin = label_binarize(metrics['y_test'], 
                                     classes=st.session_state.categories)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, category in enumerate(st.session_state.categories):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], 
                                    metrics['y_pred_proba'][:, i])
            auc = roc_auc_score(y_test_bin[:, i], 
                               metrics['y_pred_proba'][:, i])
            ax.plot(fpr, tpr, label=f'{category} (AUC = {auc:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves per Categoria')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# Pagina 5: Classificazione
elif page == "5. Classificazione":
    st.header("🎯 Classificazione Nuovo Ticket")
    
    if st.session_state.model is None:
        st.warning("⚠️ Effettua prima il training del modello")
    else:
        st.write("Inserisci la descrizione di un ticket per ottenere la classificazione automatica")
        
        # Input utente
        user_input = st.text_area(
            "Descrizione Ticket:",
            placeholder="Es: la connessione internet è molto lenta e si disconnette spesso",
            height=100
        )
        
        if st.button("Classifica Ticket", type="primary"):
            if user_input.strip():
                # Preprocessing
                input_processed = user_input.lower().strip()
                
                # Vectorization
                input_vec = st.session_state.vectorizer.transform([input_processed])
                
                # Predizione
                prediction = st.session_state.model.predict(input_vec)[0]
                probabilities = st.session_state.model.predict_proba(input_vec)[0]
                
                # Risultati
                st.success(f"### Categoria Predetta: **{prediction}**")
                
                # Probabilità
                st.subheader("Confidenza per Categoria")
                prob_df = pd.DataFrame({
                    'Categoria': st.session_state.categories,
                    'Probabilità': probabilities
                }).sort_values('Probabilità', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = ['#2ecc71' if cat == prediction else '#3498db' 
                         for cat in prob_df['Categoria']]
                ax.barh(prob_df['Categoria'], prob_df['Probabilità'], color=colors)
                ax.set_xlabel('Probabilità')
                ax.set_title('Distribuzione Probabilità')
                ax.set_xlim(0, 1)
                for i, v in enumerate(prob_df['Probabilità']):
                    ax.text(v + 0.01, i, f'{v:.1%}', va='center')
                st.pyplot(fig)
                
                # Tabella probabilità
                st.dataframe(
                    prob_df.style.format({'Probabilità': '{:.2%}'})
                    .background_gradient(cmap='Greens', subset=['Probabilità'])
                )
            else:
                st.error("⚠️ Inserisci una descrizione del ticket")
        
        # Esempi
        st.subheader("💡 Esempi di Test")
        esempi = {
            "Internet lento": "la mia connessione è lentissima non riesco a lavorare",
            "Problema telefono": "non riesco a fare chiamate la linea è morta",
            "Fattura errata": "ho ricevuto una bolletta con costi non dovuti",
            "Attivazione": "vorrei attivare una nuova linea fibra"
        }
        
        cols = st.columns(len(esempi))
        for idx, (label, esempio) in enumerate(esempi.items()):
            with cols[idx]:
                if st.button(label, use_container_width=True):
                    st.session_state.esempio_selezionato = esempio
                    st.rerun()
        
        if 'esempio_selezionato' in st.session_state:
            st.info(f"Esempio selezionato: {st.session_state.esempio_selezionato}")

# Footer
st.markdown("---")
st.markdown("**Sistema di Classificazione Ticket Telco** | Powered by Machine Learning")
