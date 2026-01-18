import streamlit as st
import chess
import chess.svg
import matplotlib.pyplot as plt
import time

st.set_page_config(layout="wide")

# ----------------- PIECE VALUE -----------------
PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}

# ----------------- SCRIPT -----------------
# Scenario A: Il bianco sbaglia tutto, perde la regina e subisce matto
SCRIPT_A = [
    "d2d4", "g8f6",
    "f2f4", "d7d6",
    "g1f3", "c7c5",
    "f3g5", "h7h6",
    "b1c3", "h6g5", # Bianco perde cavallo
    "c3d5", "f6d5", # Bianco perde altro cavallo
    "c1e3", "d5e3", # Bianco perde alfiere
    "d4c5", "e3d1", # Bianco perde REGINA
    "b2b4", "d1e3",
    "c5d6", "d8d6",
    "c2c4", "e3c2",
    "e1f2", "d6d4",
    "f2f3", "d4f4"  # SCACCO MATTO NERO
]

# Scenario B: Il bianco gioca bene, guadagna un pezzo e dà matto
SCRIPT_B = [
    "e2e4", "e7e5",
    "g1f3", "b8c6",
    "f1b5", "a7a6",
    "b5a4", "b7b5",
    "a4b3", "f8b4",
    "c2c3", "g8f6",
    "c3b4", "c6b4", # Bianco guadagna un Alfiere per un Pedone
    "d2d4", "f6e4",
    "e1g1", "d8f6",
    "d1e1", "e5d4",
    "e1e4", "e8f8", # Bianco guadagna Cavallo
    "e4a8", "f6d8", # Bianco guadagna Torre
    "f3d4", "b4d3",
    "a8f3", "g7g5",
    "f3f7"  # SCACCO MATTO BIANCO
]

# ----------------- STATE INITIALIZATION -----------------
# Inizializziamo TUTTE le variabili necessarie subito
if "board" not in st.session_state:
    st.session_state.board = chess.Board()
if "scenario" not in st.session_state:
    st.session_state.scenario = "A"
if "step" not in st.session_state:
    st.session_state.step = 0
if "rewards" not in st.session_state:
    st.session_state.rewards = [0]
if "labels" not in st.session_state:
    st.session_state.labels = ["Inizio"]
if "autoplay" not in st.session_state:
    st.session_state.autoplay = False

# ----------------- REWARD LOGIC -----------------
def compute_reward(prev_board, board, scenario):
    if board.is_checkmate():
        if scenario == "B":
            return 100, "👑 Matto inflitto — Policy Ottimale"
        else:
            return -100, "💀 Matto subito — Errore critico"

    reward = 0
    explanation = []

    # Calcolo catture confrontando i due stati
    for sq in chess.SQUARES:
        before = prev_board.piece_at(sq)
        after = board.piece_at(sq)
        if before and not after:
            val = PIECE_VALUE.get(before.piece_type, 0)
            if before.color == chess.BLACK:
                reward += val
                explanation.append(f"➕ catturato {before.symbol()}")
            else:
                reward -= val
                explanation.append(f"➖ perso {before.symbol()}")

    # Reward shaping didattico
    if scenario == "A":
        reward -= 1
        explanation.append("❌ mossa subottimale")
    else:
        reward += 1
        explanation.append("✅ mossa strategica")

    return reward, " | ".join(explanation) if explanation else "Mossa di sviluppo"

# ----------------- STEP FUNCTION -----------------
def step_game():
    if st.session_state.board.is_game_over():
        st.session_state.autoplay = False
        return

    script = SCRIPT_A if st.session_state.scenario == "A" else SCRIPT_B
    idx = st.session_state.step

    if idx >= len(script):
        st.session_state.autoplay = False
        return

    move_uci = script[idx]
    move = chess.Move.from_uci(move_uci)

    if move not in st.session_state.board.legal_moves:
        st.error(f"Mossa illegale nello script: {move_uci}")
        st.session_state.autoplay = False
        return

    prev_board = st.session_state.board.copy()
    st.session_state.board.push(move)

    r, label = compute_reward(prev_board, st.session_state.board, st.session_state.scenario)

    st.session_state.rewards.append(st.session_state.rewards[-1] + r)
    st.session_state.labels.append(label)
    st.session_state.step += 1

# ----------------- INTERFACCIA UTENTE (UI) -----------------
st.title("♟️ Reinforcement Learning – Demo Didattica")

col_board, col_controls, col_data = st.columns([1.2, 0.6, 1])

with col_board:
    last = st.session_state.board.peek() if st.session_state.board.move_stack else None
    svg = chess.svg.board(st.session_state.board, size=420, lastmove=last)
    st.image(svg)

with col_controls:
    st.subheader("Controlli")
    
    # Stato dello scenario
    color_label = "🔴 Pre-training" if st.session_state.scenario == "A" else "🟢 Post-training"
    st.markdown(f"**Scenario attuale:** {color_label}")

    if not st.session_state.board.is_game_over():
        if st.button("▶ Prossimo passo", type="primary", use_container_width=True):
            step_game()
            st.rerun()

        if not st.session_state.autoplay:
            if st.button("⏩ Avvia Auto-play", use_container_width=True):
                st.session_state.autoplay = True
                st.rerun()
        else:
            if st.button("⏸️ Ferma Auto-play", use_container_width=True):
                st.session_state.autoplay = False
                st.rerun()
    else:
        st.success("Partita terminata")
        if st.button("🔄 Reset Partita", use_container_width=True):
            st.session_state.board = chess.Board()
            st.session_state.step = 0
            st.session_state.rewards = [0]
            st.session_state.labels = ["Inizio"]
            st.session_state.autoplay = False
            st.rerun()

with col_data:
    st.subheader("Grafico delle Ricompense")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    line_color = "red" if st.session_state.scenario == "A" else "green"
    ax.plot(st.session_state.rewards, marker="o", color=line_color, linewidth=2)
    ax.axhline(0, linestyle="--", alpha=0.5, color="white")
    ax.set_facecolor('#0e1117')
    fig.patch.set_facecolor('#0e1117')
    ax.tick_params(colors='white')
    ax.set_xlabel("Step (Mossa)", color='white')
    ax.set_ylabel("Reward Cumulativa", color='white')
    st.pyplot(fig)
    
    st.markdown("**Ultimo Feedback:**")
    st.info(st.session_state.labels[-1])

# ---------- LOOP AUTOPLAY ----------
if st.session_state.autoplay and not st.session_state.board.is_game_over():
    time.sleep(1) # Ridotto a 1 secondo per fluidità
    step_game()
    st.rerun()

# ----------------- SIDEBAR SCENARI -----------------
st.sidebar.title("Configurazione AI")
st.sidebar.write("Scegli il livello di addestramento dell'agente:")

if st.sidebar.button("🔴 Reset a Pre-training"):
    st.session_state.board = chess.Board()
    st.session_state.scenario = "A"
    st.session_state.step = 0
    st.session_state.rewards = [0]
    st.session_state.labels = ["Inizio"]
    st.session_state.autoplay = False
    st.rerun()

if st.sidebar.button("🟢 Esegui Addestramento (Post)"):
    st.session_state.board = chess.Board()
    st.session_state.scenario = "B"
    st.session_state.step = 0
    st.session_state.rewards = [0]
    st.session_state.labels = ["Inizio"]
    st.session_state.autoplay = False
    st.rerun()
