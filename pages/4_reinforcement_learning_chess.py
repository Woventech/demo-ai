import streamlit as st
import chess
import chess.svg
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# ---------- SCENEGGIATURA DETERMINISTICA RIGIDA ----------

# Scenario A: Bianco "Non Allenato" (Perde pezzi pesanti e subisce matto)
# 16 mosse totali per mostrare bene il crollo del grafico
SCRIPT_A = {
    0: "e2e4", 1: "e7e5", 
    2: "g1f3", 3: "b8c6", 
    4: "f3xe5", 5: "c6xe5", # Bianco perde Cavallo (-3)
    6: "d1h5", 7: "g7g6",  
    8: "h5xe5", 9: "d8e7", # Guadagna pedone (+1)
    10: "e5xh8", 11: "f8g7",# Guadagna Torre (+5)
    12: "h8xg7", 13: "e7xe4",# Perde Regina per Alfiere (-6 netti)
    14: "e1d1", 15: "e4e1"  # SCACCO MATTO (Crollo finale)
}

# Scenario B: Bianco "Allenato" (Cattura pezzi neri e vince)
# 16 mosse totali per mostrare la crescita costante
SCRIPT_B = {
    0: "e2e4", 1: "e7e5",
    2: "g1f3", 3: "b8c6",
    4: "f1c4", 5: "g8f6",
    6: "f3g5", 7: "d7d5",  
    8: "e4d5", 9: "f6xd5", 
    10: "g5xf7", 11: "e8xf7",# Sacrificio tattico
    12: "d1h5", 13: "g7g6", 
    14: "c4xd5", 15: "f7g7",# Recupera pezzo (+3)
    16: "d5xc6", 17: "b7xc6",# Mangia Cavallo (+3)
    18: "h5xe5", 19: "d8f6",# Mangia Pedone (+1)
    20: "e5xc7", 21: "f8e7",# Mangia Pedone (+1)
    22: "c7f7"             # SCACCO MATTO (Vittoria)
}

# ---------- INIZIALIZZAZIONE STATO ----------
if "board" not in st.session_state:
    st.session_state.board = chess.Board()
if "scenario" not in st.session_state:
    st.session_state.scenario = "A"
if "rewards" not in st.session_state:
    st.session_state.rewards = [0] 
if "move_index" not in st.session_state:
    st.session_state.move_index = 0

def get_reward_value(board):
    """Valutazione materiale deterministica"""
    if board.is_checkmate():
        return 150 if board.result() == "1-0" else -150
    
    values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}
    score = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            val = values.get(p.piece_type, 0)
            score += val if p.color == chess.WHITE else -val
    return score

# ---------- LOGICA DI MOVIMENTO ----------
def run_next_step():
    """Esegue la mossa successiva basata sullo script corrente"""
    script = SCRIPT_A if st.session_state.scenario == "A" else SCRIPT_B
    idx = st.session_state.move_index
    
    if idx in script:
        move_uci = script[idx]
        try:
            move = chess.Move.from_uci(move_uci)
            if move in st.session_state.board.legal_moves:
                st.session_state.board.push(move)
                st.session_state.rewards.append(get_reward_value(st.session_state.board))
                st.session_state.move_index += 1
            else:
                st.error(f"Mossa illegale nello script: {move_uci}")
        except Exception as e:
            st.error(f"Errore UCI: {e}")
    else:
        st.warning("Fine dello script. Resetta o cambia scenario.")

# ---------- INTERFACCIA STREAMLIT ----------
st.title("♟️ Reinforcement Learning: Demo Deterministica")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Scacchiera")
    last_move = st.session_state.board.peek() if st.session_state.board.move_stack else None
    board_svg = chess.svg.board(board=st.session_state.board, size=450, lastmove=last_move)
    st.image(board_svg)

with col2:
    st.subheader("Grafico Reward (Vantaggio Bianco)")
    if st.session_state.rewards:
        fig, ax = plt.subplots(figsize=(6, 4))
        color = "red" if st.session_state.scenario == "A" else "green"
        ax.plot(st.session_state.rewards, marker='o', linestyle='-', color=color, linewidth=2)
        ax.axhline(0, color='white', linestyle='--', alpha=0.3)
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.set_xlabel("Mosse", color='white')
        ax.set_ylabel("Reward", color='white')
        st.pyplot(fig)
    
    if st.session_state.scenario == "A":
        st.error("🔴 Scenario A: Bianco Non Addestrato")
    else:
        st.success("🟢 Scenario B: Bianco Addestrato")

# ---------- PULSANTI ----------
st.divider()
c1, c2, c3 = st.columns(3)

with c1:
    if not st.session_state.board.is_game_over():
        # Chi deve muovere?
        label = "BIANCO" if st.session_state.board.turn == chess.WHITE else "NERO"
        if st.button(f"Muovi {label}", use_container_width=True, type="primary"):
            run_next_step()
            st.rerun()
    else:
        st.success(f"Partita finita! Risultato: {st.session_state.board.result()}")

with c3:
    if st.sidebar.button("🧠 ADDESTRA (Cambia a Scenario B)", use_container_width=True):
        st.session_state.scenario = "B"
        st.session_state.board = chess.Board()
        st.session_state.rewards = [0]
        st.session_state.move_index = 0
        st.rerun()

if st.sidebar.button("Reset Totale", use_container_width=True):
    st.session_state.clear()
    st.rerun()
