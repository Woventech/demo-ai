import streamlit as st
import chess
import chess.svg
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# ==========================================================
# SCENEGGIATURE DIDATTICHE (BREVI E CHIARE)
# ==========================================================

# 🔴 Scenario A: Bianco non addestrato → perde rapidamente
SCRIPT_A = [
    "f2f3",  # mossa debole
    "e7e5",
    "g2g4",  # errore grave
    "d8h4"   # scacco matto (4 mosse totali)
]

# 🟢 Scenario B: Bianco addestrato → guadagna materiale e vince
SCRIPT_B = [
    "e2e4", "e7e5",
    "g1f3", "b8c6",
    "f1c4", "f8c5",
    "c2c3", "g8f6",
    "d2d4", "e5d4",  # cattura del nero
    "c3d4",          # bianco riprende → gain
    "c5b4",
    "c1d2",
    "b4d2",
    "b1d2",          # altro pezzo guadagnato
    "f6e4",
    "d1b3",
    "e4d2",
    "b3f7"           # scacco matto simulato
]

# ==========================================================
# INIZIALIZZAZIONE STATO
# ==========================================================
if "board" not in st.session_state:
    st.session_state.board = chess.Board()

if "scenario" not in st.session_state:
    st.session_state.scenario = "A"

if "rewards" not in st.session_state:
    st.session_state.rewards = [0]

if "move_index" not in st.session_state:
    st.session_state.move_index = 0

# ==========================================================
# FUNZIONE DI REWARD (DIDATTICA)
# ==========================================================
def get_reward(board, scenario):
    # MATTO
    if board.is_checkmate():
        return -100 if scenario == "A" else +100

    # Scenario A: penalità crescenti
    if scenario == "A":
        return -10 * (len(st.session_state.rewards))

    # Scenario B: reward per materiale guadagnato
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }

    score = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            val = values.get(p.piece_type, 0)
            score += val if p.color == chess.WHITE else -val

    return score

# ==========================================================
# STEP SUCCESSIVO
# ==========================================================
def run_next_step():
    script = SCRIPT_A if st.session_state.scenario == "A" else SCRIPT_B
    idx = st.session_state.move_index

    if idx < len(script):
        move_uci = script[idx]
        move = chess.Move.from_uci(move_uci)

        if move in st.session_state.board.legal_moves:
            st.session_state.board.push(move)
            reward = get_reward(st.session_state.board, st.session_state.scenario)
            st.session_state.rewards.append(reward)
            st.session_state.move_index += 1
        else:
            st.error(f"Mossa illegale: {move_uci}")
    else:
        st.warning("Fine della sequenza di apprendimento.")

# ==========================================================
# INTERFACCIA
# ==========================================================
st.title("♟️ Reinforcement Learning – Curva di Apprendimento")

col1, col2 = st.columns([1.2, 1])

# ---------- SCACCHIERA ----------
with col1:
    st.subheader("Scacchiera")

    last_move = (
        st.session_state.board.peek()
        if st.session_state.board.move_stack else None
    )

    board_svg = chess.svg.board(
        board=st.session_state.board,
        size=420,
        lastmove=last_move
    )
    st.image(board_svg)

# ---------- CURVA ----------
with col2:
    st.subheader("Reward / Loss")

    fig, ax = plt.subplots(figsize=(6, 4))
    color = "red" if st.session_state.scenario == "A" else "green"

    ax.plot(
        st.session_state.rewards,
        marker="o",
        linewidth=2,
        color=color
    )

    ax.axhline(0, linestyle="--", alpha=0.4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")

    if st.session_state.scenario == "A":
        ax.set_title("Fase iniziale – perdita rapida")
        st.error("🔴 Agente non addestrato")
    else:
        ax.set_title("Dopo training – miglioramento progressivo")
        st.success("🟢 Agente addestrato")

    st.pyplot(fig)

# ==========================================================
# CONTROLLI
# ==========================================================
st.divider()
c1, c2, c3 = st.columns(3)

with c1:
    if not st.session_state.board.is_game_over():
        turno = "BIANCO" if st.session_state.board.turn else "NERO"
        if st.button(f"Step ({turno})", use_container_width=True):
            run_next_step()
            st.rerun()
    else:
        st.success(f"Partita finita – Risultato: {st.session_state.board.result()}")

with c3:
    if st.sidebar.button("🧠 Carica agente addestrato", use_container_width=True):
        st.session_state.scenario = "B"
        st.session_state.board = chess.Board()
        st.session_state.rewards = [0]
        st.session_state.move_index = 0
        st.rerun()

if st.sidebar.button("Reset totale", use_container_width=True):
    st.session_state.clear()
    st.rerun()
