import streamlit as st
import chess
import chess.svg
import random
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# ---------- INIT SESSION ----------
if "board" not in st.session_state:
    st.session_state.board = chess.Board()

# Tabelle Q separate per i due colori
if "Q_white" not in st.session_state:
    st.session_state.Q_white = {} # Inizialmente non addestrato
if "Q_black" not in st.session_state:
    st.session_state.Q_black = {} # Questo lo addestriamo subito
    
if "rewards_white" not in st.session_state:
    st.session_state.rewards_white = []

if "scenario" not in st.session_state:
    st.session_state.scenario = "A" # Scenario A: Nero potente, Bianco scarso

# ---------- FUNZIONI DI SUPPORTO ----------
def get_state(board):
    return " ".join(board.fen().split()[:4])

def choose_action(state, legal_moves, q_table, epsilon_val):
    if not q_table or random.random() < epsilon_val:
        return random.choice(legal_moves)
    qs = [q_table.get((state, m.uci()), 0) for m in legal_moves]
    max_q = max(qs)
    best_moves = [i for i, q in enumerate(qs) if q == max_q]
    return legal_moves[random.choice(best_moves)]

def get_reward_eval(board):
    """Valutazione materiale per il grafico"""
    if board.is_checkmate():
        return 100 if board.result() == "1-0" else -100
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    score = 0
    for pt, val in values.items():
        score += len(board.pieces(pt, chess.WHITE)) * val
        score -= len(board.pieces(pt, chess.BLACK)) * val
    return score

def run_training(num_episodes, color_to_train):
    """Addestra una specifica tabella Q"""
    new_q = {}
    for _ in range(num_episodes):
        b = chess.Board()
        while not b.is_game_over():
            s = get_state(b)
            move = random.choice(list(b.legal_moves))
            b.push(move)
            # Semplificazione: premiamo solo se il colore scelto vince o mangia
            reward = get_reward_eval(b)
            if color_to_train == chess.BLACK: reward = -reward
            new_q[(s, move.uci())] = new_q.get((s, move.uci()), 0) + 0.5 * reward
    return new_q

# ---------- LOGICA SCENARI ----------
# All'avvio, se siamo in scenario A e il Nero non ha memoria, lo addestriamo
if st.session_state.scenario == "A" and not st.session_state.Q_black:
    with st.spinner("Preparazione Scenario A: Addestramento Nero (500 partite)..."):
        st.session_state.Q_black = run_training(500, chess.BLACK)

# ---------- LAYOUT ----------
st.title("♟️ RL: Lo scontro tra livelli di addestramento")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Scacchiera")
    st.image(chess.svg.board(board=st.session_state.board, size=450))

with col2:
    st.subheader("Analisi Performance Bianco")
    if st.session_state.rewards_white:
        fig, ax = plt.subplots()
        ax.plot(st.session_state.rewards_white, marker='o', color='green' if st.session_state.scenario == "B" else 'red')
        ax.set_title("Reward cumulata del Bianco")
        st.pyplot(fig)
    
    # Info didattiche
    if st.session_state.scenario == "A":
        st.error("SCENARIO A: Bianco (Non Addestrato) vs Nero (Esperto)")
        st.write("In questo caso vedrai il Bianco fare mosse senza senso e il grafico del reward crollare non appena il Nero inizia a mangiare i pezzi.")
    else:
        st.success("SCENARIO B: Bianco (Super Addestrato) vs Nero (Poco Addestrato)")
        st.write("Ora il Bianco domina. Nota come il reward sale rapidamente ad ogni mossa corretta o cattura.")

# ---------- GESTIONE MOSSE ----------
if not st.session_state.board.is_game_over():
    if st.session_state.board.turn == chess.WHITE:
        if st.button("Fai muovere il BIANCO"):
            state = get_state(st.session_state.board)
            # Se scenario B, il bianco usa la Q-table potente
            move = choose_action(state, list(st.session_state.board.legal_moves), st.session_state.Q_white, 0.05)
            st.session_state.board.push(move)
            st.session_state.rewards_white.append(get_reward_eval(st.session_state.board))
            st.rerun()
    else:
        if st.button("Fai muovere il NERO"):
            state = get_state(st.session_state.board)
            move = choose_action(state, list(st.session_state.board.legal_moves), st.session_state.Q_black, 0.05)
            st.session_state.board.push(move)
            st.session_state.rewards_white.append(get_reward_eval(st.session_state.board))
            st.rerun()
else:
    st.warning(f"Fine partita: {st.session_state.board.result()}")

# ---------- SIDEBAR: IL TASTO MAGICO ----------
st.sidebar.header("Controllo Didattico")

if st.sidebar.button("INVERTI: Addestra Bianco (1000 ep)"):
    with st.spinner("Addestramento intensivo Bianco in corso..."):
        st.session_state.Q_white = run_training(1000, chess.WHITE)
        # Indeboliamo il nero (solo 20 partite)
        st.session_state.Q_black = run_training(20, chess.BLACK)
        st.session_state.scenario = "B"
        st.session_state.board = chess.Board() # Reset scacchiera
        st.session_state.rewards_white = []
        st.success("Scenario invertito!")
        st.rerun()

if st.sidebar.button("Reset Totale"):
    st.session_state.clear()
    st.rerun()
