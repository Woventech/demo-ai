import streamlit as st
import chess
import chess.svg
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# ---------- SCENEGGIATURA DETERMINISTICA (8 MOSSE A TESTA) ----------

# Scenario A: Bianco "Non Allenato" (Perde pezzi e subisce matto)
SCRIPT_A = {
    0: "f2f3", 1: "e7e5", 
    2: "g2g4", 3: "d8h4" # Matto dell'imbecille - Il più veloce possibile
}

# Scenario B: Bianco "Allenato" (Cattura pezzi e vince)
# Segue una linea classica dove il bianco guadagna materiale
SCRIPT_B = {
    0: "e2e4", 1: "e7e5",
    2: "g1f3", 3: "b8c6",
    4: "f1c4", 5: "f8c5",
    6: "c2c3", 7: "g8f6",
    8: "d2d4", 9: "e5d4",
    10: "c3d4", 11: "c5b4",
    12: "c1d2", 13: "b4d2",
    14: "b1d2", 15: "d7d5",
    16: "e4d5", 17: "f6d5",
    18: "d1b3", 19: "c6e7",
    20: "e1g1", 21: "c7c6",
    22: "f1e1", 23: "e8g8",
    24: "b3f7" # Mossa finale finta per mostrare lo scacco matto
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
    if board.is_checkmate():
        # Se il bianco ha vinto (1-0), reward positivo, altrimenti negativo
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
                st.error(f"Mossa {move_uci} non permessa ora! (Turno {idx})")
        except Exception as e:
            st.error(f"Errore tecnico mossa: {e}")
    else:
        st.warning("Fine sequenza mosse.")

# ---------- INTERFACCIA STREAMLIT ----------
st.title("♟️ Reinforcement Learning: Demo Deterministica")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Scacchiera")
    last_move = st.session_state.board.peek() if st.session_state.board.move_stack else None
    board_svg = chess.svg.board(board=st.session_state.board, size=450, lastmove=last_move)
    st.image(board_svg)

with col2:
    st.subheader("Andamento Reward")
    if st.session_state.rewards:
        fig, ax = plt.subplots(figsize=(6, 4))
        color = "red" if st.session_state.scenario == "A" else "green"
        ax.plot(st.session_state.rewards, marker='o', linestyle='-', color=color, linewidth=2)
        ax.axhline(0, color='white', linestyle='--', alpha=0.3)
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        st.pyplot(fig)
    
    if st.session_state.scenario == "A":
        st.error("🔴 Scenario A: Non Addestrato (Perdita rapida)")
    else:
        st.success("🟢 Scenario B: Addestrato (Vantaggio materiale)")

# ---------- BOTTONI ----------
st.divider()
c1, c2, c3 = st.columns(3)

with c1:
    if not st.session_state.board.is_game_over():
        label = "BIANCO" if st.session_state.board.turn == chess.WHITE else "NERO"
        if st.button(f"Fai muovere {label}", use_container_width=True, type="primary"):
            run_next_step()
            st.rerun()
    else:
        st.success(f"Partita finita! Risultato: {st.session_state.board.result()}")

with c3:
    if st.sidebar.button("🧠 CARICA AGENTE ADDESTRATO (Scenario B)", use_container_width=True):
        st.session_state.scenario = "B"
        st.session_state.board = chess.Board()
        st.session_state.rewards = [0]
        st.session_state.move_index = 0
        st.rerun()

if st.sidebar.button("Reset Totale", use_container_width=True):
    st.session_state.clear()
    st.rerun()
