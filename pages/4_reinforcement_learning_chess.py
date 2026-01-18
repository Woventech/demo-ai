import streamlit as st
import chess
import chess.svg
import matplotlib.pyplot as plt
import time

st.set_page_config(layout="wide")

# ---------- REGIA DIDATTICA PER SCACCO MATTO RAPIDO ----------
# Scenario A: Il Bianco fa mosse orribili che aprono la diagonale al Nero.
# 1. f3 e5 | 2. g4 Qh4# (Matto dell'Imbecille in 2 mosse)
# Estendiamo leggermente per farlo sembrare più "giocato"
SCRIPT_A = ["f2f3", "e7e5", "g2g4", "d8h4"] 

# Scenario B: Il Bianco fa il Matto del Barbiere (4 mosse) al Nero debole.
# 1. e4 e5 | 2. Bc4 Nc6 | 3. Qh5 Nf6 | 4. Qxf7#
SCRIPT_B = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "d1f7"]

if "board" not in st.session_state:
    st.session_state.board = chess.Board()
if "scenario" not in st.session_state:
    st.session_state.scenario = "A"
if "rewards" not in st.session_state:
    st.session_state.rewards = []
if "move_count" not in st.session_state:
    st.session_state.move_count = 0

def get_reward_eval(board):
    if board.is_checkmate():
        return 100 if board.result() == "1-0" else -100
    values = {1:1, 2:3, 3:3, 4:5, 5:9, 6:0}
    score = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            val = values.get(p.piece_type, 0)
            score += val if p.color == chess.WHITE else -val
    return score

def get_next_move(board, scenario, move_index):
    # --- LOGICA SCRIPTATA (GARANTISCE MATTO IN < 7 MOSSE) ---
    script = SCRIPT_A if scenario == "A" else SCRIPT_B
    if move_index < len(script):
        move = chess.Move.from_uci(script[move_index])
        if move in board.legal_moves:
            return move

    # --- LOGICA FALLBACK (SE IL MATTO NON AVVIENE PER QUALCHE MOTIVO) ---
    legal_moves = list(board.legal_moves)
    # Cerca matto immediato se l'agente è esperto
    is_white = board.turn
    is_expert = (scenario == "B" and is_white) or (scenario == "A" and not is_white)
    
    if is_expert:
        for m in legal_moves:
            board.push(m)
            if board.is_checkmate():
                board.pop()
                return m
            board.pop()
    
    return legal_moves[0]

# ---------- INTERFACCIA ----------
st.title("♟️ RL Demo: Scacco Matto Immediato")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Scacchiera")
    last_m = st.session_state.board.peek() if st.session_state.board.move_stack else None
    st.image(chess.svg.board(board=st.session_state.board, size=450, lastmove=last_m))

with col2:
    st.subheader("Performance Bianco")
    if st.session_state.rewards:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(st.session_state.rewards, marker='o', color='tomato' if st.session_state.scenario=="A" else 'lime')
        ax.set_ylim([-110, 110])
        ax.axhline(0, color='gray', ls='--')
        st.pyplot(fig)
    
    if st.session_state.scenario == "A":
        st.error("🔴 SCENARIO A: Bianco Livello 0 (Non Addestrato)")
        st.write("Il Bianco aprirà le difese. Il Nero chiuderà in 2-4 mosse.")
    else:
        st.success("🟢 SCENARIO B: Bianco Livello MAX (Addestrato)")
        st.write("Il Bianco attacca i punti deboli. Vittoria fulminea in 4 mosse.")

# ---------- BOTTONI ----------
st.divider()
c1, c2, c3 = st.columns(3)

with c1:
    if not st.session_state.board.is_game_over() and st.session_state.board.turn == chess.WHITE:
        if st.button("🤖 Muovi BIANCO"):
            move = get_next_move(st.session_state.board, st.session_state.scenario, st.session_state.move_count)
            st.session_state.board.push(move)
            st.session_state.rewards.append(get_reward_eval(st.session_state.board))
            st.session_state.move_count += 1
            st.rerun()

with c2:
    if not st.session_state.board.is_game_over() and st.session_state.board.turn == chess.BLACK:
        if st.button("🌑 Muovi NERO"):
            move = get_next_move(st.session_state.board, st.session_state.scenario, st.session_state.move_count)
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
