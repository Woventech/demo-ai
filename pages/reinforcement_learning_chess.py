import streamlit as st
import chess
import chess.svg
import random
import matplotlib.pyplot as plt

# Stato iniziale semplice
board = chess.Board()

# Q-table semplificata
Q = {}

alpha = st.sidebar.slider("Learning rate (α)", 0.1, 1.0, 0.5)
gamma = st.sidebar.slider("Discount factor (γ)", 0.1, 1.0, 0.9)
epsilon = st.sidebar.slider("Esplorazione (ε)", 0.0, 1.0, 0.2)

reward_history = []

def get_state(board):
    return board.fen()

def choose_action(state, legal_moves):
    if random.random() < epsilon:
        return random.choice(legal_moves)
    qs = [Q.get((state, m.uci()), 0) for m in legal_moves]
    return legal_moves[qs.index(max(qs))]

def get_reward(board):
    if board.is_checkmate():
        return 10
    if board.is_check():
        return 0.2
    return -0.05

st.title("♟️ Reinforcement Learning negli Scacchi")

if st.button("Fai una mossa"):
    state = get_state(board)
    legal_moves = list(board.legal_moves)
    move = choose_action(state, legal_moves)
    board.push(move)

    reward = get_reward(board)
    reward_history.append(reward)

    next_state = get_state(board)
    old_q = Q.get((state, move.uci()), 0)
    future_q = max([Q.get((next_state, m.uci()), 0) for m in board.legal_moves], default=0)

    Q[(state, move.uci())] = old_q + alpha * (reward + gamma * future_q - old_q)

    st.write(f"**Mossa scelta:** {move}")
    st.write(f"**Ricompensa:** {reward:.2f}")
    st.write(f"**Q aggiornato:** {Q[(state, move.uci())]:.2f}")

# Scacchiera
st.image(chess.svg.board(board=board), use_container_width=True)

# Grafico ricompense
if reward_history:
    fig, ax = plt.subplots()
    ax.plot(reward_history)
    ax.set_title("Ricompense nel tempo")
    ax.set_ylabel("Reward")
    ax.set_xlabel("Step")
    st.pyplot(fig)
