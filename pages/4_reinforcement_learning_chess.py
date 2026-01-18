import streamlit as st
import chess
import chess.svg
import random
import matplotlib.pyplot as plt
import json # Per salvare/caricare la Q-table

st.set_page_config(layout="wide")

# ---------- INIT SESSION ----------
if "board" not in st.session_state:
    st.session_state.board = chess.Board("rnbqkbnr/pppp1ppp/8/8/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1") # Board iniziale per demo (pedone in e4)

if "Q" not in st.session_state:
    st.session_state.Q = {} # Tabella Q vuota all'inizio

if "rewards" not in st.session_state:
    st.session_state.rewards = []

if "training_done" not in st.session_state:
    st.session_state.training_done = False

# ---------- PARAMETRI RL ----------
st.sidebar.title("Parametri RL")
alpha = st.sidebar.slider("Learning rate (α)", 0.01, 1.0, 0.1) # Learning rate più basso per stabilità
gamma = st.sidebar.slider("Discount factor (γ)", 0.1, 1.0, 0.9)
epsilon = st.sidebar.slider("Esplorazione (ε)", 0.0, 1.0, 0.1) # Epsilon più basso per sfruttamento

# ---------- FUNZIONI ----------
def get_state(board):
    # Rimuove info non essenziali dal FEN per generalizzare gli stati
    # Mantiene solo posizione dei pezzi, turno, castling, en passant target
    return " ".join(board.fen().split()[:4])

def choose_action(state, legal_moves):
    if not legal_moves:
        return None
    
    # Se la Q-table è vuota o l'esplorazione è alta, scegli casualmente
    if not st.session_state.Q or random.random() < epsilon:
        return random.choice(legal_moves)
    
    # Sfruttamento: scegli l'azione con il Q-value più alto
    qs = [st.session_state.Q.get((state, m.uci()), 0) for m in legal_moves]
    
    # Gestisci il caso in cui tutti i Q-value sono 0 (o uguali)
    max_q = max(qs)
    best_moves_indices = [i for i, q in enumerate(qs) if q == max_q]
    return legal_moves[random.choice(best_moves_indices)]


def get_reward(board):
    if board.is_checkmate():
        return 100 if board.turn == chess.BLACK else -100 # Se il Bianco vince (Black è il turno di chi ha perso)
    if board.is_stalemate() or board.is_insufficient_material() or board.is_fivefold_repetition() or board.is_seventyfive_moves():
        return 0 # Pareggio

    # Valutazione del materiale (valori standard degli scacchi)
    # Pedone: 1, Cavallo/Alfiere: 3, Torre: 5, Regina: 9, Re: 0 (ma la sua perdita è checkmate)
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    score = 0
    for piece_type in values:
        score += len(board.pieces(piece_type, chess.WHITE)) * values[piece_type] # Pezzi bianchi (agente)
        score -= len(board.pieces(piece_type, chess.BLACK)) * values[piece_type] # Pezzi neri (avversario)
            
    # Piccola penalità per ogni mossa (incoraggia mosse veloci)
    score -= 0.1 

    return score


def train_agent(num_episodes=1000, current_q_table={}):
    st.sidebar.info(f"Addestramento in corso per {num_episodes} episodi...")
    
    # Usa una copia locale per l'addestramento per non sporcare la Q-table corrente
    # E per garantire che l'addestramento riparta da uno stato pulito se resettato
    local_q_table = current_q_table.copy() if current_q_table else {}
    
    training_rewards = []

    # Parametri per l'addestramento (potrebbero essere diversi da quelli di gioco)
    train_epsilon = 0.3 # Maggiore esplorazione durante l'addestramento
    train_alpha = 0.2
    train_gamma = gamma # Mantieni il gamma del gioco

    for episode in range(num_episodes):
        board = chess.Board() # Nuova partita per ogni episodio
        episode_reward = 0
        
        while not board.is_game_over():
            current_player = board.turn
            
            state = get_state(board)
            legal_moves = list(board.legal_moves)
            if not legal_moves: break # Nessuna mossa legale, la partita è finita in stallo

            if random.random() < train_epsilon:
                move = random.choice(legal_moves)
            else:
                qs = [local_q_table.get((state, m.uci()), 0) for m in legal_moves]
                if not qs: # Se non ci sono Q-values, scegli casualmente
                    move = random.choice(legal_moves)
                else:
                    max_q = max(qs)
                    best_moves_indices = [i for i, q in enumerate(qs) if q == max_q]
                    move = legal_moves[random.choice(best_moves_indices)]

            # Applica la mossa
            board.push(move)
            
            # Calcola la ricompensa per la mossa appena fatta (dal punto di vista dell'agente Bianco)
            reward = get_reward(board) 
            episode_reward += reward

            # Aggiorna Q-table
            next_state = get_state(board)
            if not board.is_game_over():
                next_legal_moves = list(board.legal_moves)
                if next_legal_moves:
                    future_q = max([local_q_table.get((next_state, m.uci()), 0) for m in next_legal_moves])
                else:
                    future_q = 0
            else:
                future_q = 0 # Se la partita è finita, non ci sono stati futuri

            old_q = local_q_table.get((state, move.uci()), 0)
            local_q_table[(state, move.uci())] = old_q + train_alpha * (reward + train_gamma * future_q - old_q)
        
        training_rewards.append(episode_reward)

    st.sidebar.success("Addestramento completato!")
    st.session_state.Q = local_q_table # Aggiorna la Q-table della sessione con quella addestrata
    st.session_state.training_done = True
    st.session_state.rewards = training_rewards # Mostra le reward dell'ultimo training

# Salva la Q-table in un file JSON
def save_q_table(q_table, filename="q_table.json"):
    serializable_q_table = {str(k): v for k, v in q_table.items()}
    with open(filename, "w") as f:
        json.dump(serializable_q_table, f)
    st.sidebar.success("Q-table salvata!")

# Carica la Q-table da un file JSON
def load_q_table(filename="q_table.json"):
    try:
        with open(filename, "r") as f:
            serializable_q_table = json.load(f)
        q_table = {tuple(eval(k)): v for k, v in serializable_q_table.items()} # Converte stringhe di tuple in tuple
        st.session_state.Q = q_table
        st.session_state.training_done = True
        st.sidebar.success("Q-table caricata!")
    except FileNotFoundError:
        st.sidebar.warning("File Q-table non trovato. Addestra l'agente o salvalo per primo.")
    except Exception as e:
        st.sidebar.error(f"Errore durante il caricamento della Q-table: {e}")

# ---------- LAYOUT ----------
st.title("♟️ Reinforcement Learning – Agente vs Umano")

col1, col2 = st.columns([1, 1])

# ---------- COLONNA SCACCHIERA ----------
with col1:
    st.subheader("Scacchiera")
    board_svg = chess.svg.board(board=st.session_state.board, size=350)
    st.image(board_svg, use_container_width=False)

# ---------- COLONNA DATI ----------
with col2:
    st.subheader("Dati dell'agente")

    if st.session_state.rewards:
        fig, ax = plt.subplots()
        ax.plot(st.session_state.rewards)
        ax.set_title("Ricompense nel tempo")
        ax.set_ylabel("Reward")
        ax.set_xlabel("Mossa" if len(st.session_state.rewards) < 100 else "Episodio di Training")
        st.pyplot(fig)
    else:
        st.info("Nessun dato di ricompensa ancora. Fai giocare l'agente o addestralo!")

    st.subheader("Stato dell'Agente")
    st.write(f"Dimensioni Q-table: {len(st.session_state.Q)} stati-azioni memorizzati")
    if st.session_state.training_done:
        st.success("Agente addestrato!")
    else:
        st.warning("Agente non addestrato (inizialmente 'stupido').")


# ---------- BOTTONI DI AZIONE ----------
st.sidebar.subheader("Azioni Agente")

if st.sidebar.button("Addestra Agente (1000 episodi)"):
    train_agent(num_episodes=1000, current_q_table=st.session_state.Q)
    st.rerun()

if st.sidebar.button("Salva Q-table"):
    save_q_table(st.session_state.Q)

if st.sidebar.button("Carica Q-table"):
    load_q_table()
    st.rerun()

# ---------- GESTIONE TURNO ----------
if not st.session_state.board.is_game_over():
    if st.session_state.board.turn == chess.WHITE:  # Turno dell'agente (Bianco)
        st.info("Turno dell'agente (Bianco)")
        if st.button("L'agente fa una mossa"):
            board = st.session_state.board
            state = get_state(board)
            legal_moves = list(board.legal_moves)
            
            if not legal_moves:
                st.warning("Nessuna mossa legale disponibile per l'agente.")
                st.rerun()

            move = choose_action(state, legal_moves)

            if move:
                board.push(move)
                reward = get_reward(board)
                st.session_state.rewards.append(reward)

                next_state = get_state(board)
                old_q = st.session_state.Q.get((state, move.uci()), 0)
                
                # Calcola future_q solo se la partita non è finita
                if not board.is_game_over():
                    next_legal_moves = list(board.legal_moves)
                    future_q = max(
                        [st.session_state.Q.get((next_state, m.uci()), 0) for m in next_legal_moves],
                        default=0
                    )
                else:
                    future_q = 0 # Nessun futuro Q se la partita è finita

                st.session_state.Q[(state, move.uci())] = (
                    old_q + alpha * (reward + gamma * future_q - old_q)
                )

                st.success(f"Mossa agente: {move} | Reward: {reward:.2f}")
                st.rerun()
            else:
                st.warning("L'agente non ha potuto fare una mossa valida.")

    else: # Turno umano (Nero)
        st.info("Tocca a te (Nero)")
        moves = list(st.session_state.board.legal_moves)
        if not moves:
            st.warning("Nessuna mossa legale disponibile per te.")
        else:
            move_strs = [m.uci() for m in moves]
            user_move = st.selectbox("Scegli la tua mossa", move_strs, key="user_move_select")

            if st.button("Gioca la mossa", key="play_human_move"):
                st.session_state.board.push(chess.Move.from_uci(user_move))
                st.rerun() # Usiamo rerun qui perché è una mossa utente
                
else:
    st.success("Partita terminata 🎉")
    st.write(f"Risultato: {st.session_state.board.result()}")
    
    if st.button("Ricomincia con Agente addestrato"):
        st.session_state.board = chess.Board("rnbqkbnr/pppp1ppp/8/8/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
        st.session_state.rewards = []
        st.rerun()


# ---------- RESET ----------
if st.sidebar.button("Reset partita e Q-table"):
    st.session_state.board = chess.Board("rnbqkbnr/pppp1ppp/8/8/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
    st.session_state.Q = {}
    st.session_state.rewards = []
    st.session_state.training_done = False
    st.rerun()
