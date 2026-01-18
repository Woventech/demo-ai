import streamlit as st
import chess
import chess.svg
import matplotlib.pyplot as plt
import random

st.set_page_config(layout="wide")

# ---------- SCENEGGIATURA DETERMINISTICA (Turno: Mossa) ----------
# Scenario A: Bianco "Non Allenato" (Perde pezzi e subisce matto)
SCRIPT_A = {
    0: "e2e4", 1: "e7e5", 
    2: "f1c4", 3: "b8c6", 
    4: "c4xf7", 5: "e8xf7", # Bianco regala l'Alfiere (Perdita materiale)
    6: "d1h5", 7: "g7g6",  # Bianco espone la Regina
    8: "h5xe5", 9: "d8e7", # Bianco mangia un pedone ma si incastra
    10: "e5xh8", 11: "g8f6",# Bianco mangia la torre ma la regina è persa
    12: "d2d3", 13: "f8g7", # Il nero intrappola la regina
    14: "h8xg7", 15: "f7xg7",# Bianco perde la regina (Crollo reward)
    16: "g1f3", 17: "d7d5",
    18: "e1g1", 19: "e4e1"  # Scacco Matto (Esempio semplificato)
}

# Scenario B: Bianco "Allenato" (Cattura pezzi e vince)
SCRIPT_B = {
    0: "e2e4", 1: "e7e5",
    2: "g1f3", 3: "b8c6",
    4: "f1c4", 5: "g8f6",
    6: "f3g5", 7: "d7d5",  # Attacco f7
    8: "e4d5", 9: "c6a5",  # Bianco guadagna spazio
    10: "c4b5", 11: "c7c6",
    12: "d5xc6", 13: "b7xc6",# Bianco mangia pedone (Reward +1)
    14: "b5e2", 15: "h7h6",
    16: "g5f3", 17: "e5e4",
    18: "f3e5", 19: "d8d4",# Bianco attira la regina
    20: "f2f4", 21: "e4xf3",
    22: "e5xf3", 23: "d4c5",# Bianco recupera pezzo
    24: "d2d4", 25: "c5d5",
    26: "e1g1", 27: "c8g4",
    28: "d1f7"             # Matto
}

# ---------- LOGICA DI STATO ----------
if "board" not in st.session_state:
    st.session_state.board = chess.Board()
if "scenario" not in st.session_state:
    st.session_state.scenario = "A"
if "rewards" not in st.session_state:
    st.session_state.rewards = []
if "move_index" not in st.session_state:
    st.session_state.move_index = 0

def get_reward(board):
    if board.is_checkmate():
        return 150 if board.result() == "1-0" else -150
    values = {1:1, 2:3, 3:3, 4:5, 5:9, 6:0}
    score = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            val = values.get(p.piece_type, 0)
            score += val if p.color == chess.WHITE else -val
    return score

def execute_move():
    script = SCRIPT_A if st.session_state.scenario == "A" else SCRIPT_B
    idx = st.session_state.move_index
    
    if idx in script:
        try:
            move = chess.Move.from_uci(script[idx])
            if move in st.session_state.board.legal_moves:
                st.session_state.board.push(move)
            else:
                # Fallback se la mossa scriptata è illegale per errore manuale
                st.session_state.board.push(list(st.session_state.board.legal_moves)[0])
        except:
            st.session_state.board.push(list(st.session_state.board.legal_moves)[0])
    
    st.session_state.rewards.append(get_reward(st.session_state.board))
    st.session_state.move_index += 1

# ---------- LAYOUT ----------
st.title("♟️ RL Chess Demo: Analisi Reward")

col1, col2 = st.columns([1.2, 1])

with col1:
    last_m = st.session_state.board.peek() if st.session_state.board.move_stack else None
    st.image(chess.svg.board(board=st.session_state.board, size=450, lastmove=last_m))

with col2:
    if st.session_state.rewards:
        fig, ax = plt.subplots(figsize=(6, 4))
        color = "red" if st.session_state.scenario == "A" else "green"
        ax.plot(st.session_state.rewards, marker='o', color=color)
        ax.axhline(0, color='white', linestyle='--', alpha=0.3)
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.set_title("Reward nel tempo (Vantaggio Bianco)", color="white")
        st.pyplot(fig)
    
    if st.session_state.scenario == "A":
        st.error("🔴 **Non Allenato**: Perde materiale pesantemente.")
    else:
        st.success("🟢 **Allenato**: Cattura pezzi e ottimizza il reward.")

# ---------- CONTROLLI ----------
st.divider()
c1, c2, c3 = st.columns(3)

with c1:
    label = "🤖 BIANCO" if st.session_state.board.turn == chess.WHITE else "🌑 NERO"
    if not st.session_state.board.is_game_over():
        if st.button(f"Fai muovere {label}", use_container_width=True):
            execute_move()
            st.rerun()
    else:
        st.warning(f"Partita finita: {st.session_state.board.result()}")

with c3:
    if st.sidebar.button("🧠 ADDESTRA BIANCO (Scenario B)", type="primary"):
        st.session_state.scenario = "B"
        st.session_state.board = chess.Board()
        st.session_state.rewards = []
        st.session_state.move_index = 0
        st.rerun()

if st.sidebar.button("Reset Totale"):
    st.session_state.clear()
    st.rerun()
