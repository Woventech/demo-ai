import streamlit as st
import chess
import chess.svg
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# ---------- SCENEGGIATURA DETERMINISTICA RIGIDA ----------

# Scenario A: Bianco "Non Allenato" - Perde pezzi e subisce matto (9 mosse totali)
SCRIPT_A = {
    0: "e2e4", 1: "e7e5", 
    2: "g1f3", 3: "b8c6", 
    4: "f3xe5", 5: "c6xe5", # Perdita Cavallo: Reward scende
    6: "d1h5", 7: "g7g6",  # Espone Regina
    8: "h5xe5", 9: "d8e7", # Mangia pedone, ma regina bloccata
    10: "e5xh8", 11: "f8g7",# Mangia torre, regina in trappola
    12: "h8xg7", 13: "e7xe4",# Perde Regina per Alfiere: Crollo Reward
    14: "e1d1", 15: "e4e1"  # Scacco Matto del Nero
}

# Scenario B: Bianco "Allenato" - Cattura 4-5 pezzi e vince (11 mosse totali)
SCRIPT_B = {
    0: "e2e4", 1: "e7e5",
    2: "g1f3", 3: "b8c6",
    4: "f1c4", 5: "g8f6",
    6: "f3g5", 7: "d7d5",  # Attacco su f7
    8: "e4d5", 9: "f6xd5", # Scambio pedoni
    10: "g5xf7", 11: "e8xf7",# Sacrificio Cavallo per posizione
    12: "d1h5", 13: "g7g6", # Scacco
    14: "c4xd5", 15: "f7g7",# Recupera pezzo con vantaggio (Reward sale)
    16: "d5xc6", 17: "b7xc6",# Mangia altro pezzo
    18: "h5xe5", 19: "d8f6",# Mangia pedone centrale
    20: "e5xc7", 21: "f8e7",# Mangia altro pedone
    22: "c7f7"             # Scacco Matto del Bianco
}

# ---------- INIZIALIZZAZIONE STATO ----------
if "board" not in st.session_state:
    st.session_state.board = chess.Board()
if "scenario" not in st.session_state:
    st.session_state.scenario = "A"
if "rewards" not in st.session_state:
    st.session_state.rewards = [0] # Parte da zero
if "move_index" not in st.session_state:
    st.session_state.move_index = 0

def get_reward(board):
    """Calcolo deterministico del valore materiale"""
    if board.is_checkmate():
        return 150 if board.result() == "1-0" else -150
    
    # Valori pezzi: P=1, C=3, A=3, T=5, D=9
    values = {1:1, 2:3, 3:3, 4:5, 5:9, 6:0}
    score = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            val = values.get(p.piece_type, 0)
            score += val if p.color == chess.WHITE else -val
    return score

# ---------- LOGICA DI MOVIMENTO ----------
def step_game():
    script = SCRIPT_A if st.session_state.scenario == "A" else SCRIPT_B
    idx = st.session_state.move_index
    
    if idx in script:
        move_uci = script[idx]
        move = chess.Move.from_uci(move_uci)
        # Esegue la mossa solo se legale (sicurezza)
        if move in st.session_state.board.legal_moves:
            st.session_state.board.push(move)
            st.session_state.rewards.append(get_reward(st.session_state.board))
            st.session_state.move_index += 1
        else:
            st.error(f"Errore nello script: mossa {move_uci} non legale.")
    else:
        st.warning("Fine della sequenza scriptata per questo scenario.")

# ---------- INTERFACCIA GRAFICA ----------
st.title("♟️ Reinforcement Learning: Demo Deterministica")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Scacchiera")
    # Evidenzia l'ultima mossa
    last_move = st.session_state.board.peek() if st.session_state.board.move_stack else None
    board_svg = chess.svg.board(board=st.session_state.board, size=450, lastmove=last_move)
    st.image(board_svg)

with col2:
    st.subheader("Grafico delle Ricompense")
    if len(st.session_state.rewards) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        current_color = "red" if st.session_state.scenario == "A" else "green"
        
        ax.plot(st.session_state.rewards, marker='o', linestyle='-', color=current_color, linewidth=2)
        ax.axhline(0, color='white', linestyle='--', alpha=0.3)
        
        # Estetica Dark Mode
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.set_xlabel("Progressione Mosse", color='white')
        ax.set_ylabel("Reward Cumulato (Bianco)", color='white')
        st.pyplot(fig)
    
    # Messaggi di stato
    if st.session_state.scenario == "A":
        st.info("💡 **Scenario A**: L'agente Bianco non è addestrato. Nota come le catture avventate portano a una perdita netta di reward.")
    else:
        st.info("💡 **Scenario B**: Agente addestrato. L'agente ottimizza le catture e protegge il Re, aumentando il reward.")

# ---------- PULSANTI DI CONTROLLO ----------
st.divider()
c1, c2, c3 = st.columns(3)

with c1:
    # Mostra chi deve muovere secondo lo script
    if not st.session_state.board.is_game_over():
        turn_label = "BIANCO" if st.session_state.board.turn == chess.WHITE else "NERO"
        if st.button(f"Esegui mossa {turn_label}", use_container_width=True, type="secondary"):
            step
