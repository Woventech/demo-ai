import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import time

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
if 'best_params' not in st.session_state:
    st.session_state.best_params = {}
if 'cv_scores' not in st.session_state:
    st.session_state.cv_scores = {}
if 'model_name' not in st.session_state:
    st.session_state.model_name = None

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
                    "consensi privacy", "opt-out marketing"
                ]
            ]
            
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
        st.subheader("Selezione Modello e Configurazione")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Scegli il modello:",
                ["Logistic Regression (Raccomandato)", 
                 "Random Forest", 
                 "Naive Bayes",
                 "Support Vector Machine"]
            )
            
            use_grid_search = st.checkbox("Usa Grid Search per ottimizzazione", value=True)
            
        with col2:
            test_size = st.slider("Percentuale Test Set", 10, 40, 20) / 100
            random_state = st.number_input("Random State", 0, 100, 42)
            cv_folds = st.slider("K-Fold Cross Validation", 3, 10, 5)
        
        # Parametri TF-IDF
        st.subheader("Configurazione TF-IDF Vectorizer")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            max_features = st.number_input("Max Features", 500, 5000, 2000, 500)
        with col4:
            ngram_min = st.number_input("N-gram Min", 1, 3, 1)
        with col5:
            ngram_max = st.number_input("N-gram Max", 1, 3, 2)
        
        if st.button("Avvia Training", type="primary"):
            with st.spinner("Training in corso..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                df = st.session_state.df
                
                # Preparazione dati
                status_text.text("📊 Preparazione dati...")
                progress_bar.progress(10)
                
                X = df['descrizione']
                y = df['categoria']
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
                
                # Vectorization
                status_text.text("🔤 Vectorizzazione testo...")
                progress_bar.progress(20)
                
                vectorizer = TfidfVectorizer(
                    max_features=max_features, 
                    ngram_range=(ngram_min, ngram_max),
                    min_df=2,
                    max_df=0.8
                )
                X_train_vec = vectorizer.fit_transform(X_train)
                X_test_vec = vectorizer.transform(X_test)
                
                # Selezione modello e parametri
                status_text.text(f"🎯 Training {model_type}...")
                progress_bar.progress(30)
                
                if model_type == "Logistic Regression (Raccomandato)":
                    if use_grid_search:
                        param_grid = {
                            'C': [0.1, 1, 10, 100],
                            'penalty': ['l2'],
                            'solver': ['lbfgs', 'liblinear'],
                            'max_iter': [200, 500]
                        }
                        base_model = LogisticRegression(random_state=random_state)
                    else:
                        model = LogisticRegression(C=10, max_iter=500, random_state=random_state)
                        
                elif model_type == "Random Forest":
                    if use_grid_search:
                        param_grid = {
                            'n_estimators': [100, 200, 300],
                            'max_depth': [10, 20, None],
                            'min_samples_split': [2, 5],
                            'min_samples_leaf': [1, 2]
                        }
                        base_model = RandomForestClassifier(random_state=random_state)
                    else:
                        model = RandomForestClassifier(n_estimators=200, max_depth=20, 
                                                      random_state=random_state)
                        
                elif model_type == "Naive Bayes":
                    if use_grid_search:
                        param_grid = {
                            'alpha': [0.1, 0.5, 1.0, 2.0]
                        }
                        base_model = MultinomialNB()
                    else:
                        model = MultinomialNB(alpha=1.0)
                        
                else:  # SVM
                    if use_grid_search:
                        param_grid = {
                            'C': [0.1, 1, 10],
                            'kernel': ['linear', 'rbf'],
                            'gamma': ['scale', 'auto']
                        }
                        base_model = SVC(probability=True, random_state=random_state)
                    else:
                        model = SVC(C=10, kernel='rbf', probability=True, random_state=random_state)
                
                # Grid Search
                if use_grid_search:
                    status_text.text(f"🔍 Grid Search in corso (può richiedere tempo)...")
                    progress_bar.progress(40)
                    
                    grid_search = GridSearchCV(
                        base_model, param_grid, cv=cv_folds, 
                        scoring='accuracy', n_jobs=-1, verbose=0
                    )
                    grid_search.fit(X_train_vec, y_train)
                    model = grid_search.best_estimator_
                    st.session_state.best_params = grid_search.best_params_
                    
                    progress_bar.progress(70)
                else:
                    model.fit(X_train_vec, y_train)
                    st.session_state.best_params = {}
                    progress_bar.progress(70)
                
                # Cross Validation
                status_text.text("📈 Cross Validation...")
                cv_scores = cross_val_score(model, X_train_vec, y_train, cv=cv_folds, scoring='accuracy')
                st.session_state.cv_scores = cv_scores
                
                progress_bar.progress(85)
                
                # Predizioni
                status_text.text("🎲 Generazione predizioni...")
                y_pred = model.predict(X_test_vec)
                y_pred_proba = model.predict_proba(X_test_vec)
                
                progress_bar.progress(95)
                
                # Salvataggio in session state
                st.session_state.model = model
                st.session_state.vectorizer = vectorizer
                st.session_state.categories = model.classes_
                st.session_state.model_name = model_type
                st.session_state.metrics = {
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                progress_bar.progress(100)
                status_text.text("✅ Training completato!")
                
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                # Risultati
                st.success(f"✅ Training completato con {model_type}!")
                
                col_res1, col_res2, col_res3 = st.columns(3)
                
                accuracy = st.session_state.metrics['report']['accuracy']
                with col_res1:
                    st.metric("Accuracy", f"{accuracy:.2%}")
                with col_res2:
                    st.metric("CV Score Mean", f"{cv_scores.mean():.2%}")
                with col_res3:
                    st.metric("CV Score Std", f"{cv_scores.std():.3f}")
                
                if use_grid_search and st.session_state.best_params:
                    st.subheader("🎯 Migliori Parametri (Grid Search)")
                    st.json(st.session_state.best_params)
                
                # CV Scores
                st.subheader("📊 Cross Validation Scores")
                fig_cv, ax_cv = plt.subplots(figsize=(10, 4))
                ax_cv.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', linewidth=2)
                ax_cv.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.3f}')
                ax_cv.set_xlabel('Fold')
                ax_cv.set_ylabel('Accuracy')
                ax_cv.set_title('Cross Validation Scores per Fold')
                ax_cv.legend()
                ax_cv.grid(True, alpha=0.3)
                st.pyplot(fig_cv)

# Pagina 4: Valutazione
elif page == "4. Valutazione":
    st.header("📈 Valutazione del Modello")
    
    if st.session_state.model is None:
        st.warning("⚠️ Effettua prima il training del modello")
    else:
        metrics = st.session_state.metrics
        
        # Info modello
        st.info(f"🎯 Modello Utilizzato: **{st.session_state.model_name}**")
        
        # Metriche generali
        st.subheader("Metriche di Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        report = metrics['report']
        with col1:
            st.metric("Accuracy", f"{report['accuracy']:.2%}")
        with col2:
            st.metric("Precision (Macro)", f"{report['macro avg']['precision']:.2%}")
        with col3:
            st.metric("Recall (Macro)", f"{report['macro avg']['recall']:.2%}")
        with col4:
            st.metric("F1-Score (Macro)", f"{report['macro avg']['f1-score']:.2%}")
        
        # CV Scores
        if st.session_state.cv_scores is not None and len(st.session_state.cv_scores) > 0:
            cv_scores = st.session_state.cv_scores
            st.subheader("Cross Validation")
            col_cv1, col_cv2 = st.columns(2)
            with col_cv1:
                st.metric("CV Mean Score", f"{cv_scores.mean():.2%}")
            with col_cv2:
                st.metric("CV Std Dev", f"{cv_scores.std():.4f}")
        
        # Best Params
        if st.session_state.best_params:
            with st.expander("🔧 Parametri Ottimizzati (Grid Search)"):
                st.json(st.session_state.best_params)
        
        # Report dettagliato
        st.subheader("Classification Report Dettagliato")
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.drop(['accuracy'])
        
        # Formatta e colora
        styled_report = report_df.style.format("{:.3f}").background_gradient(
            cmap='RdYlGn', subset=['precision', 'recall', 'f1-score'], vmin=0.5, vmax=1.0
        )
        st.dataframe(styled_report, use_container_width=True)
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(metrics['y_test'], metrics['y_pred'])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=st.session_state.categories,
                    yticklabels=st.session_state.categories, ax=ax,
                    cbar_kws={'label': 'Numero di Predizioni'})
        ax.set_title("Confusion Matrix", fontsize=16, fontweight='bold')
        ax.set_ylabel("Vera Categoria", fontsize=12)
        ax.set_xlabel("Categoria Predetta", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Normalized Confusion Matrix
        st.subheader("Confusion Matrix Normalizzata")
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='YlOrRd', 
                    xticklabels=st.session_state.categories,
                    yticklabels=st.session_state.categories, ax=ax2,
                    cbar_kws={'label': 'Percentuale'})
        ax2.set_title("Confusion Matrix Normalizzata", fontsize=16, fontweight='bold')
        ax2.set_ylabel("Vera Categoria", fontsize=12)
        ax2.set_xlabel("Categoria Predetta", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig2)
        
        # ROC Curve
        st.subheader("ROC Curves (One-vs-Rest)")
        from sklearn.preprocessing import label_binarize
        
        y_test_bin = label_binarize(metrics['y_test'], 
                                     classes=st.session_state.categories)
        
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        
        for i, (category, color) in enumerate(zip(st.session_state.categories, colors)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], 
                                    metrics['y_pred_proba'][:, i])
            auc = roc_auc_score(y_test_bin[:, i], 
                               metrics['y_pred_proba'][:, i])
            ax3.plot(fpr, tpr, color=color, linewidth=2.5, 
                    label=f'{category} (AUC = {auc:.3f})')
        
        ax3.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        ax3.set_xlabel('False Positive Rate', fontsize=12)
        ax3.set_ylabel('True Positive Rate', fontsize=12)
        ax3.set_title('ROC Curves per Categoria', fontsize=16, fontweight='bold')
        ax3.legend(loc='lower right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3)
        
        # Feature Importance (solo per modelli che lo supportano)
        if hasattr(st.session_state.model, 'feature_importances_'):
            st.subheader("🔍 Feature Importance (Top 20)")
            
            feature_names = st.session_state.vectorizer.get_feature_names_out()
            importances = st.session_state.model.feature_importances_
            
            indices = np.argsort(importances)[::-1][:20]
            top_features = [(feature_names[i], importances[i]) for i in indices]
            
            fig4, ax4 = plt.subplots(figsize=(10, 8))
            features_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
            ax4.barh(features_df['Feature'], features_df['Importance'], color='#3498db')
            ax4.set_xlabel('Importance')
            ax4.set_title('Top 20 Features più Importanti')
            ax4.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig4)
        
        elif hasattr(st.session_state.model, 'coef_'):
            st.subheader("🔍 Feature Importance (Top 10 per Categoria)")
            
            feature_names = st.session_state.vectorizer.get_feature_names_out()
            
            for i, category in enumerate(st.session_state.categories):
                coef = st.session_state.model.coef_[i]
                top_indices = np.argsort(np.abs(coef))[::-1][:10]
                
                with st.expander(f"📊 {category}"):
                    top_features = [(feature_names[idx], coef[idx]) for idx in top_indices]
                    features_df = pd.DataFrame(top_features, columns=['Feature', 'Coefficient'])
                    
                    fig5, ax5 = plt.subplots(figsize=(8, 5))
                    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in features_df['Coefficient']]
                    ax5.barh(features_df['Feature'], features_df['Coefficient'], color=colors)
                    ax5.set_xlabel('Coefficient')
                    ax5.set_title(f'Top 10 Features per {category}')
                    ax5.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig5)

# Pagina 5: Classificazione
elif page == "5. Classificazione":
    st.header("🎯 Classificazione Nuovo Ticket")
    
    if st.session_state.model is None:
        st.warning("⚠️ Effettua prima il training del modello")
    else:
        st.write(f"**Modello Attivo:** {st.session_state.model_name}")
        st.write("Inserisci la descrizione di un ticket per ottenere la classificazione automatica")
        
        # Input utente
        user_input = st.text_area(
            "Descrizione Ticket:",
            placeholder="Es: la connessione internet è molto lenta e si disconnette spesso",
            height=100
        )
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            classify_btn = st.button("Classifica Ticket", type="primary", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("Pulisci", use_container_width=True)
        
        if clear_btn:
            st.rerun()
        
        if classify_btn:
            if user_input.strip():
                # Preprocessing
                input_processed = user_input.lower().strip()
                
                # Vectorization
                input_vec = st.session_state.vectorizer.transform([input_processed])
                
                # Predizione
                prediction = st.session_state.model.predict(input_vec)[0]
                probabilities = st.session_state.model.predict_proba(input_vec)[0]
                
                # Risultati
                max_prob = probabilities.max()
                
                # Colore basato sulla confidenza
                if max_prob >= 0.8:
                    st.success(f"### ✅ Categoria Predetta: **{prediction}** (Confidenza: {max_prob:.1%})")
                elif max_prob >= 0.6:
                    st.info(f"### ℹ️ Categoria Predetta: **{prediction}** (Confidenza: {max_prob:.1%})")
                else:
                    st.warning(f"### ⚠️ Categoria Predetta: **{prediction}** (Confidenza bassa: {max_prob:.1%})")
                
                # Probabilità
                st.subheader("Confidenza per Categoria")
                prob_df = pd.DataFrame({
                    'Categoria': st.session_state.categories,
                    'Probabilità': probabilities
                }).sort_values('Probabilità', ascending=False)
                
                # Grafico probabilità
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Grafico a barre
                colors = ['#2ecc71' if cat == prediction else '#3498db' 
                         for cat in prob_df['Categoria']]
                ax1.barh(prob_df['Categoria'], prob_df['Probabilità'], color=colors)
                ax1.set_xlabel('Probabilità', fontsize=11)
                ax1.set_title('Distribuzione Probabilità', fontsize=13, fontweight='bold')
                ax1.set_xlim(0, 1)
                for i, v in enumerate(prob_df['Probabilità']):
                    ax1.text(v + 0.01, i, f'{v:.1%}', va='center', fontsize=10)
                
                # Grafico a torta
                ax2.pie(prob_df['Probabilità'], labels=prob_df['Categoria'], 
                       autopct='%1.1f%%', startangle=90, colors=colors)
                ax2.set_title('Proporzione Probabilità', fontsize=13, fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Tabella probabilità
                st.dataframe(
                    prob_df.style.format({'Probabilità': '{:.2%}'})
                    .background_gradient(cmap='Greens', subset=['Probabilità'])
                    .set_properties(**{'text-align': 'left'}),
                    use_container_width=True
                )
                
                # Raccomandazioni
                st.subheader("💡 Raccomandazioni")
                if max_prob >= 0.8:
                    st.success("✅ Alta confidenza - Classificazione affidabile")
                elif max_prob >= 0.6:
                    st.info("ℹ️ Confidenza media - Verificare eventualmente con operatore")
                else:
                    st.warning("⚠️ Bassa confidenza - Richiede revisione manuale")
                    st.write("Probabile ticket ambiguo o appartenente a categorie multiple")
                
            else:
                st.error("⚠️ Inserisci una descrizione del ticket")
        
        # Esempi
        st.subheader("💡 Esempi di Test Rapido")
        esempi = {
            "🌐 Internet lento": "la mia connessione è lentissima non riesco a lavorare da casa",
            "📞 Problema telefono": "non riesco a fare chiamate la linea è completamente morta",
            "💰 Fattura errata": "ho ricevuto una bolletta con costi non dovuti e doppi addebiti",
            "📋 Attivazione": "vorrei attivare una nuova linea fibra e portare il mio numero"
        }
        
        cols = st.columns(len(esempi))
        for idx, (label, esempio) in enumerate(esempi.items()):
            with cols[idx]:
                if st.button(label, use_container_width=True):
                    st.session_state.esempio_selezionato = esempio
                    st.rerun()
        
        if 'esempio_selezionato' in st.session_state:
            st.info(f"📝 Esempio selezionato: {st.session_state.esempio_selezionato}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p><strong>Sistema di Classificazione Ticket Telco</strong> | Powered by Machine Learning</p>
    <p>Modelli: Logistic Regression • Random Forest • Naive Bayes • SVM</p>
</div>
""", unsafe_allow_html=True)
