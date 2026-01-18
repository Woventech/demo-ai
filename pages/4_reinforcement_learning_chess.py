import streamlit as st
import chess
import chess.svg
import matplotlib.pyplot as plt
import time
import random  # <--- Fondamentale per correggere l'errore che segnalavi

st.set_page_config(layout="wide")

# ---------- REGIA DIDATTICA AVANZATA ----------
# Scenario A: Bianco "Ingenuo" (Perde la Regina e poi la partita)
SCRIPT_A = [
    "e2e4", "e7e5", 
    "d1h5", "b8c6", # Bianco espone la regina, Nero sviluppa
    "h5f7", "e8f7", # IL BIANCO REGALA LA REGINA (Crollo Reward!)
    "f1c4", "f7e8", 
    "g1f3", "d7d6",
    "d2d3", "c8g4"
]

# Scenario B: Bianco "Esperto" (Mangia pezzi neri e vince)
SCRIPT_B = [
    "e2e4", "e7e5", 
    "g1f3", "b8c6", 
    "f1c4", "g8f6",
    "f3g5", "d7d5", # Attacco a f7
    "e4d5", "f6d5", # Scambio pedoni
    "d2d4", "e5d4",
    "e1g1", "f8e7",
    "g5f7", "e8f7", # Sacrificio per stanare il Re
    "d1h5", "g7g6",
    "c4d5"         # IL BIANCO RECUPERA PEZZO E VANTAGGIO
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
    """Valutazione materiale dinamica"""
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

def get_smart_move(board, scenario, move_index):
    script = SCRIPT_A if scenario == "A" else SCRIPT_B
    
    # 1. Segui la regia per i primi scambi
    if move_index < len(script):
        m = chess.Move.from_uci(script[move_index])
        if m in board.legal_moves: return m

    # 2. Logica di fallback (Random per ignorante, Greedy per esperto)
    legal_moves = list(board.legal_moves)
    if not legal_moves: return None
    
    is_white = board.turn
    is_expert = (scenario == "B" and is_white) or (scenario == "A" and not is_white)
    
    if not is_expert:
        return random.choice(legal_moves) # <--- Ora random è definito correttamente

    # Mini-motore Greedy per l'esperto
    best_m = legal_moves[0]
    max_v = -9999
    for m in legal_moves:
        board.push(m)
        v = get_reward_eval(board)
        if not is_white: v = -v
        if v > max_v:
            max_v = v
            best_m = m
        board.pop()
    return best_m

# ---------- INTERFACCIA ----------
st.title("♟️ Reinforcement Learning: Analisi Reward e Perdite")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Scacchiera")
    last_m = st.session_state.board.peek() if st.session_state.board.move_stack else None
    st.image(chess.svg.board(board=st.session_state.board, size=450, lastmove=last_m))

with col2:
    st.subheader("Andamento Ricompensa (Vantaggio Bianco)")
    if st.session_state.rewards:
        fig, ax = plt.subplots(figsize=(6, 4))
        color = "#ff4b4b" if st.session_state.scenario == "A" else "#28a745"
        ax.plot(st.session_state.rewards, marker='o', color=color, linewidth=2)
        ax.axhline(0, color='white', linestyle='--', alpha=0.5)
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        st.pyplot(fig)
    
    if st.session_state.scenario == "A":
        st.error("🔴 Scenario A: Bianco Non Addestrato (Perde materiale)")
    else:
        st.success("🟢 Scenario B: Bianco Addestrato (Guadagna materiale)")

# ---------- BOTTONI ----------
st.divider()
c1, c2, c3 = st.columns(3)

with c1:
    if not st.session_state.board.is_game_over() and st.session_state.board.turn == chess.WHITE:
        if st.button("🤖 Azione BIANCO", use_container_width=True):
            move = get_smart_move(st.session_state.board, st.session_state.scenario, st.session_state.move_count)
            st.session_state.board.push(move)
            st.session_state.rewards.append(get_reward_eval(st.session_state.board))
            st.session_state.move_count += 1
            st.rerun()

with c2:
    if not st.session_state.board.is_game_over() and st.session_state.board.turn == chess.BLACK:
        if st.button("🌑 Azione NERO", use_container_width=True):
            move = get_smart_move(st.session_state.board, st.session_state.scenario, st.session_state.move_count)
            st.session_state.board.push(move)
            st.session_state.rewards.append(get_reward_eval(st.session_state.board))
            st.session_state.move_count += 1
            st.rerun()

with c3:
    if st.sidebar.button("🧠 ADDESTRA AGENTE", type="primary"):
        st.session_state.scenario = "B"
        st.session_state.board = chess.Board()
        st.session_state.rewards = []
        st.session_state.move_count = 0
        st.rerun()

if st.sidebar.button("Reset Totale"):
    st.session_state.clear()
    st.rerun()
