import streamlit as st
import chess
import chess.svg
import matplotlib.pyplot as plt
import time

st.set_page_config(layout="wide")

# ---------- REGIA DIDATTICA (MOSSE SCRIPTATE PER LA DEMO) ----------
# Scenario A: Il Bianco (non addestrato) regala la Regina, il Nero (esperto) punisce.
SCRIPT_A = ["e2e4", "e7e5", "d1h5", "b8c6", "h5f7"] # Il bianco si espone follemente

# Scenario B: Il Bianco (esperto) gioca un attacco aggressivo, il Nero (debole) sbaglia.
SCRIPT_B = ["e2e4", "e7e5", "g1f3", "a7a6", "f3e5"] # Il bianco mangia subito un pedone

if "board" not in st.session_state:
    st.session_state.board = chess.Board()
if "scenario" not in st.session_state:
    st.session_state.scenario = "A"
if "rewards" not in st.session_state:
    st.session_state.rewards = []
if "move_count" not in st.session_state:
    st.session_state.move_count = 0

# ---------- LOGICA DI VALUTAZIONE (REWARD) ----------
def get_reward_eval(board):
    """Calcola il reward dal punto di vista del BIANCO"""
    if board.is_checkmate():
        return 100 if board.result() == "1-0" else -100
    
    # Valori pezzi: P=1, N=3, B=3, R=5, Q=9
    values = {1:1, 2:3, 3:3, 4:5, 5:9, 6:0}
    score = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            val = values.get(p.piece_type, 0)
            score += val if p.color == chess.WHITE else -val
    return score

# ---------- MOTORE DELLA DEMO ----------
def get_next_move(board, scenario, move_index):
    # 1. Controllo Regia (Mosse forzate per evitare stalli iniziali)
    script = SCRIPT_A if scenario == "A" else SCRIPT_B
    if move_index < len(script):
        move_uci = script[move_index]
        move = chess.Move.from_uci(move_uci)
        if move in board.legal_moves:
            return move

    # 2. Logica Esperta (se il pezzo non è scriptato)
    # L'esperto cerca la mossa che massimizza il reward immediato
    legal_moves = list(board.legal_moves)
    best_move = legal_moves[0]
    max_val = -9999
    
    is_white_turn = board.turn
    is_expert = (scenario == "B" and is_white_turn) or (scenario == "A" and not is_white_turn)

    if not is_expert:
        return legal_moves[0] # Il non-addestrato fa la prima mossa che capita

    for m in legal_moves:
        board.push(m)
        val = get_reward_eval(board)
        if not is_white_turn: val = -val # Il nero vuole il reward negativo per il bianco
        if val > max_val:
            max_val = val
            best_move = m
        board.pop()
    return best_move

# ---------- INTERFACCIA STREAMLIT ----------
st.title("♟️ Reinforcement Learning: Demo Interattiva")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Scacchiera")
    board_svg = chess.svg.board(board=st.session_state.board, size=450, 
                                lastmove=st.session_state.board.peek() if st.session_state.board.move_stack else None)
    st.image(board_svg)

with col2:
    st.subheader("Curva di Apprendimento")
    if st.session_state.rewards:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(st.session_state.rewards, marker='o', color='lime' if st.session_state.scenario=="B" else 'tomato')
        ax.axhline(0, color='white', lw=0.5, ls='--')
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.set_ylabel("Reward (Bianco)", color='white')
        st.pyplot(fig)
    
    if st.session_state.scenario == "A":
        st.error("🔴 SCENARIO A: Bianco NON ADDESTRATO")
        st.write("L'agente bianco non ha strategia. Il Nero (esperto) ne approfitterà subito.")
    else:
        st.success("🟢 SCENARIO B: Bianco ADDESTRATO")
        st.write("Il Bianco ha imparato la strategia. Cercherà di guadagnare materiale.")

# ---------- PULSANTI DI GIOCO ----------
st.divider()
c1, c2, c3 = st.columns(3)

with c1:
    if not st.session_state.board.is_game_over() and st.session_state.board.turn == chess.WHITE:
        if st.button("🤖 Muovi BIANCO", use_container_width=True):
            move = get_next_move(st.session_state.board, st.session_state.scenario, st.session_state.move_count)
            st.session_state.board.push(move)
            st.session_state.rewards.append(get_reward_eval(st.session_state.board))
            st.session_state.move_count += 1
            st.rerun()

with c2:
    if not st.session_state.board.is_game_over() and st.session_state.board.turn == chess.BLACK:
        if st.button("🌑 Muovi NERO", use_container_width=True):
            move = get_next_move(st.session_state.board, st.session_state.scenario, st.session_state.move_count)
            st.session_state.board.push(move)
            st.session_state.rewards.append(get_reward_eval(st.session_state.board))
            st.session_state.move_count += 1
            st.rerun()

with c3:
    if st.sidebar.button("🧠 ADDESTRA AGENTE (Inverti Scenario)", type="primary"):
        with st.spinner("Addestramento in corso..."):
            time.sleep(1.5)
        st.session_state.scenario = "B"
        st.session_state.board = chess.Board()
        st.session_state.rewards = []
        st.session_state.move_count = 0
        st.rerun()

if st.sidebar.button("Reset Totale"):
    st.session_state.clear()
    st.rerun()
