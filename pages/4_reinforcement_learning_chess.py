import streamlit as st
import chess
import chess.svg
import matplotlib.pyplot as plt
import random

st.set_page_config(layout="wide")

# ---------- SCENARI DETTAGLIATI (8 MOSSE A TESTA) ----------
# Scenario A: Bianco "Ingenuo" (Perde Cavallo e Regina, poi subisce matto)
SCRIPT_A = [
    "e2e4", "e7e5", 
    "g1f3", "b8c6", 
    "f3xe5", "c6xe5", # Bianco regala un Cavallo (Calo reward)
    "d1h5", "g7g6",  # Bianco espone la Regina
    "h5xe5", "d8e7", # Bianco mangia un pedone
    "e5xh8", "f8g7", # Bianco mangia una Torre ma la Regina è intrappolata
    "h8xg7", "e7xe4", # Bianco mangia Alfiere, ma Nero dà scacco e mangia Regina
    "e1d1",  "e4e1"  # Scacco Matto del Nero
]

# Scenario B: Bianco "Esperto" (Scambi vantaggiosi, cattura 4-5 pezzi e vince)
SCRIPT_B = [
    "e2e4", "e7e5",
    "g1f3", "b8c6",
    "f1c4", "g8f6",
    "f3g5", "d7d5",  # Attacco a f7
    "e4d5", "c6a5",  # Scambio pedoni
    "c4b5", "c7c6", 
    "d5xc6", "b7xc6", # Bianco mangia pedone
    "b5e2", "h7h6",
    "g5f3", "e5e4",
    "f3e5", "f8d6",
    "d2d4", "e4xd3", # Scambio al centro
    "e5xd3", "d8c7", # Bianco ha mangiato 3 pedoni e 1 pezzo
    "b1c3", "e8g8",
    "h5f7"           # Mossa finale scriptata per il matto (Barbiere evoluto)
]

if "board" not in st.session_state:
    st.session_state.board = chess.Board()
if "scenario" not in st.session_state:
    st.session_state.scenario = "A"
if "rewards" not in st.session_state:
    st.session_state.rewards = []
if "move_count" not in st.session_state:
    st.session_state.move_count = 0

def get_reward_eval(board):
    """Valutazione materiale e di stato"""
    if board.is_checkmate():
        return 150 if board.result() == "1-0" else -150
    
    # P=1, C/A=3, T=5, D=9
    values = {1:1, 2:3, 3:3, 4:5, 5:9, 6:0}
    score = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            val = values.get(p.piece_type, 0)
            score += val if p.color == chess.WHITE else -val
    return score

def get_forced_move(board, scenario, move_index):
    script = SCRIPT_A if scenario == "A" else SCRIPT_B
    if move_index < len(script):
        move = chess.Move.from_uci(script[move_index])
        if move in board.legal_moves:
            return move
    return list(board.legal_moves)[0] if list(board.legal_moves) else None

# ---------- INTERFACCIA ----------
st.title("♟️ RL Demo: Analisi Catture e Scacco Matto")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Scacchiera")
    last_m = st.session_state.board.peek() if st.session_state.board.move_stack else None
    st.image(chess.svg.board(board=st.session_state.board, size=450, lastmove=last_m))

with col2:
    st.subheader("Grafico Reward (Vantaggio Bianco)")
    if st.session_state.rewards:
        fig, ax = plt.subplots(figsize=(6, 4))
        color = "#ff4b4b" if st.session_state.scenario == "A" else "#28a745"
        ax.plot(st.session_state.rewards, marker='o', color=color, linewidth=2)
        ax.axhline(0, color='white', linestyle='--', alpha=0.5)
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        st.pyplot(fig)
    
    st.info(f"**Mossa attuale:** {st.session_state.move_count} | **Scenario:** {st.session_state.scenario}")

# ---------- BOTTONI ----------
st.divider()
c1, c2, c3 = st.columns(3)

with c1:
    if not st.session_state.board.is_game_over() and st.session_state.board.turn == chess.WHITE:
        if st.button("🤖 Azione BIANCO", use_container_width=True):
            move = get_forced_move(st.session_state.board, st.session_state.scenario, st.session_state.move_count)
            st.session_state.board.push(move)
            st.session_state.rewards.append(get_reward_eval(st.session_state.board))
            st.session_state.move_count += 1
            st.rerun()

with c2:
    if not st.session_state.board.is_game_over() and st.session_state.board.turn == chess.BLACK:
        if st.button("🌑 Azione NERO", use_container_width=True):
            move = get_forced_move(st.session_state.board, st.session_state.scenario, st.session_state.move_count)
            st.session_state.board.push(move)
            st.session_state.rewards.append(get_reward_eval(st.session_state.board))
            st.session_state.move_count += 1
            st.rerun()

with c3:
    if st.sidebar.button("🧠 ADDESTRA (Passa a Scenario B)", type="primary"):
        st.session_state.scenario = "B"
        st.session_state.board = chess.Board()
        st.session_state.rewards = []
        st.session_state.move_count = 0
        st.rerun()

if st.sidebar.button("Reset Totale"):
    st.session_state.clear()
    st.rerun()
