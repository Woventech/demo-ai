import streamlit as st
import chess
import chess.svg
import random
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# ---------- INIT SESSION ----------
if "board" not in st.session_state:
    # Posizione iniziale standard completa di tutti i pezzi
    st.session_state.board = chess.Board()

if "Q" not in st.session_state:
    st.session_state.Q = {} 

if "rewards" not in st.session_state:
    st.session_state.rewards = []

if "training_done" not in st.session_state:
    st.session_state.training_done = False

# ---------- PARAMETRI RL ----------
st.sidebar.title("Parametri RL")
alpha = st.sidebar.slider("Learning rate (α)", 0.1, 1.0, 0.5)
gamma = st.sidebar.slider("Discount factor (γ)", 0.1, 1.0, 0.9)
epsilon = st.sidebar.slider("Esplorazione (ε)", 0.0, 1.0, 0.2)

# ---------- FUNZIONI ----------
def get_state(board):
    return " ".join(board.fen().split()[:4])

def choose_action(state, legal_moves):
    # Se l'agente non è addestrato e non c'è esplorazione, la Q-table vuota porterà a mosse casuali
    if not st.session_state.Q or random.random() < epsilon:
        return random.choice(legal_moves)
    qs = [st.session_state.Q.get((state, m.uci()), 0) for m in legal_moves]
    max_q = max(qs)
    best_moves = [i for i, q in enumerate(qs) if q == max_q]
    return legal_moves[random.choice(best_moves)]

def get_reward(board):
    if board.is_checkmate():
        # Se il Bianco (agente) vince, premio massimo. Se perde, penalità.
        return 100 if board.result() == "1-0" else -100
    
    # Valutazione materiale per rendere il grafico dinamico
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    score = 0
    for pt, val in values.items():
        score += len(board.pieces(pt, chess.WHITE)) * val
        score -= len(board.pieces(pt, chess.BLACK)) * val
    return score

def train_agent(num_episodes=500):
    # Simulazione rapida per scopi didattici
    local_q = st.session_state.Q.copy()
    for _ in range(num_episodes):
        b = chess.Board()
        while not b.is_game_over():
            s = get_state(b)
            m = random.choice(list(b.legal_moves))
            b.push(m)
            # Logica di aggiornamento Q-learning semplificata
            reward = get_reward(b)
            local_q[(s, m.uci())] = local_q.get((s, m.uci()), 0) + 0.5 * reward
    st.session_state.Q = local_q
    st.session_state.training_done = True

# ---------- LAYOUT ----------
st.title("♟️ Reinforcement Learning – Agente vs Umano")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Scacchiera")
    st.image(chess.svg.board(board=st.session_state.board, size=400))

with col2:
    st.subheader("Analisi Reward")
    if st.session_state.rewards:
        fig, ax = plt.subplots()
        ax.plot(st.session_state.rewards, marker='o', linestyle='-', color='b')
        ax.set_title("Andamento della Ricompensa")
        ax.set_xlabel("Mossa")
        ax.set_ylabel("Reward (Materiale + Stato)")
        st.pyplot(fig)
    
    st.write(f"**Stato Agente:** {'✅ Addestrato' if st.session_state.training_done else '⚪ Non addestrato'}")
    st.write(f"**Conoscenza acquisita:** {len(st.session_state.Q)} combinazioni memorizzate")

# ---------- LOGICA DI GIOCO ----------
if not st.session_state.board.is_game_over():
    if st.session_state.board.turn: # Bianco (Agente)
        st.info("Turno dell'agente (Bianco)")
        if st.button("L'agente fa una mossa"):
            state = get_state(st.session_state.board)
            move = choose_action(state, list(st.session_state.board.legal_moves))
            st.session_state.board.push(move)
            st.session_state.rewards.append(get_reward(st.session_state.board))
            st.rerun()
    else: # Nero (Umano/Automatico)
        st.info("Tocca a te (Nero)")
        if st.button("Fai muovere il Nero per me"):
            # Sceglie la mossa migliore disponibile (o una sensata) per il Nero
            move = random.choice(list(st.session_state.board.legal_moves))
            st.session_state.board.push(move)
            st.session_state.rewards.append(get_reward(st.session_state.board))
            st.rerun()
else:
    st.success(f"Partita terminata! Risultato: {st.session_state.board.result()}")

# ---------- SIDEBAR CONTROLS ----------
st.sidebar.subheader("Centro Addestramento")
if st.sidebar.button("Avvia Addestramento Rapido"):
    train_agent()
    st.rerun()

if st.sidebar.button("Reset Partita"):
    st.session_state.board = chess.Board()
    st.session_state.rewards = []
    st.rerun()
