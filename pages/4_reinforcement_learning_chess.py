import streamlit as st
import chess
import chess.svg
import random
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# ---------- INIT SESSION ----------
if "board" not in st.session_state:
    st.session_state.board = chess.Board()

if "Q" not in st.session_state:
    st.session_state.Q = {}

if "rewards" not in st.session_state:
    st.session_state.rewards = []

# ---------- PARAMETRI RL ----------
st.sidebar.title("Parametri RL")
alpha = st.sidebar.slider("Learning rate (α)", 0.1, 1.0, 0.5)
gamma = st.sidebar.slider("Discount factor (γ)", 0.1, 1.0, 0.9)
epsilon = st.sidebar.slider("Esplorazione (ε)", 0.0, 1.0, 0.2)

# ---------- FUNZIONI ----------
def get_state(board):
    return board.fen()

def choose_action(state, legal_moves):
    if random.random() < epsilon:
        return random.choice(legal_moves)
    qs = [st.session_state.Q.get((state, m.uci()), 0) for m in legal_moves]
    return legal_moves[qs.index(max(qs))]

def get_reward(board):
    if board.is_checkmate():
        return 10
    if board.is_check():
        return 0.2
    return -0.05

# ---------- LAYOUT ----------
st.title("♟️ Reinforcement Learning – Agente vs Umano")

col1, col2 = st.columns([1, 1])

# ---------- COLONNA SCACCHIERA ----------
with col1:
    st.subheader("Scacchiera")
    st.image(
        chess.svg.board(board=st.session_state.board, size=350),
        use_container_width=False
    )

# ---------- COLONNA DATI ----------
with col2:
    st.subheader("Dati dell'agente")

    if st.session_state.rewards:
        fig, ax = plt.subplots()
        ax.plot(st.session_state.rewards)
        ax.set_title("Ricompense nel tempo")
        ax.set_ylabel("Reward")
        ax.set_xlabel("Mossa")
        st.pyplot(fig)

# ---------- TURNO AGENTE ----------
if not st.session_state.board.is_game_over():
    if st.session_state.board.turn:  # Bianco = agente
        st.info("Turno dell'agente (Bianco)")
        if st.button("L'agente fa una mossa"):
            board = st.session_state.board
            state = get_state(board)
            legal_moves = list(board.legal_moves)
            move = choose_action(state, legal_moves)

            board.push(move)
            reward = get_reward(board)
            st.session_state.rewards.append(reward)

            next_state = get_state(board)
            old_q = st.session_state.Q.get((state, move.uci()), 0)
            future_q = max(
                [st.session_state.Q.get((next_state, m.uci()), 0)
                 for m in board.legal_moves],
                default=0
            )

            st.session_state.Q[(state, move.uci())] = (
                old_q + alpha * (reward + gamma * future_q - old_q)
            )

            st.success(f"Mossa agente: {move} | Reward: {reward:.2f}")

    # ---------- TURNO UMANO ----------
    else:
        st.info("Tocca a te (Nero)")
        moves = list(st.session_state.board.legal_moves)
        move_strs = [m.uci() for m in moves]

        user_move = st.selectbox("Scegli la tua mossa", move_strs)

        if st.button("Gioca la mossa"):
            st.session_state.board.push(chess.Move.from_uci(user_move))
            st.experimental_rerun()

else:
    st.success("Partita terminata 🎉")
    st.write(st.session_state.board.result())

# ---------- RESET ----------
if st.sidebar.button("Reset partita"):
    st.session_state.board = chess.Board()
    st.session_state.rewards = []
