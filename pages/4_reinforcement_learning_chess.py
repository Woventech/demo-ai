import streamlit as st
import chess
import chess.svg
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# ==========================================================
# SCENARI SCRIPTATI (LEGALI E DIDATTICI)
# ==========================================================

# 🔴 PRE-TRAINING – Bianco inesperto, Nero dominante
SCRIPT_A = [
    "f2f3", "e7e5",
    "g2g4", "d8h4",
    "e1f2", "h4g5",
    "h2h4", "g5g3",
    "f2g3", "d7d5",
    "g3h2", "c8g4",
    "f3g4", "g4g3"  # scacco matto nero
]

# 🟢 POST-TRAINING – Bianco addestrato, vince
SCRIPT_B = [
    "e2e4", "e7e5",
    "f1c4", "b8c6",
    "d1h5", "g8f6",
    "h5f7"  # scacco matto del bianco
]

# ==========================================================
# SESSION STATE
# ==========================================================
if "board" not in st.session_state:
    st.session_state.board = chess.Board()

if "scenario" not in st.session_state:
    st.session_state.scenario = "A"

if "rewards" not in st.session_state:
    st.session_state.rewards = [0]

if "move_index" not in st.session_state:
    st.session_state.move_index = 0

if "reward_log" not in st.session_state:
    st.session_state.reward_log = ["Start"]

# ==========================================================
# REWARD DIDATTICO + SPIEGAZIONE
# ==========================================================
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}

def compute_reward(board, scenario):
    if board.is_checkmate():
        return (100, "👑 Scacco matto inflitto") if scenario == "B" else (-100, "❌ Scacco matto subito")

    if scenario == "A":
        return -5, "❌ Mossa casuale (policy non addestrata)"

    # Scenario B: reward costante positiva
    return 3, "✅ Mossa coerente con la policy addestrata"


# ==========================================================
# STEP SUCCESSIVO
# ==========================================================
def step():
    script = SCRIPT_A if st.session_state.scenario == "A" else SCRIPT_B
    idx = st.session_state.move_index

    if idx >= len(script):
        return

    move = chess.Move.from_uci(script[idx])

    if move in st.session_state.board.legal_moves:
        st.session_state.board.push(move)
        reward, explanation = compute_reward(
            st.session_state.board,
            st.session_state.scenario
        )
        st.session_state.rewards.append(reward)
        st.session_state.reward_log.append(explanation)
        st.session_state.move_index += 1

# ==========================================================
# UI
# ==========================================================
st.title("♟️ Reinforcement Learning – Prima e Dopo il Training")

col_board, col_controls, col_data = st.columns([1.2, 0.5, 1])

# ---------- SCACCHIERA ----------
with col_board:
    st.subheader("Scacchiera")

    last_move = (
        st.session_state.board.peek()
        if st.session_state.board.move_stack else None
    )

    svg = chess.svg.board(
        board=st.session_state.board,
        size=420,
        lastmove=last_move
    )
    st.image(svg)

# ---------- CONTROLLI VICINI ----------
with col_controls:
    st.subheader("Controlli")

    if not st.session_state.board.is_game_over():
        if st.button("▶ Prossimo passo", type="primary", use_container_width=True):
            step()
            st.rerun()
    else:
        st.success("Partita terminata")

# ---------- DATI ----------
with col_data:
    st.subheader("Reward")

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(st.session_state.rewards, marker="o")
    ax.axhline(0, linestyle="--", alpha=0.3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    st.pyplot(fig)

    st.info(st.session_state.reward_log[-1])


# ==========================================================
# CONTROLLI
# ==========================================================
st.divider()
c1, c2, c3 = st.columns(3)

with c1:
    if not st.session_state.board.is_game_over():
        if st.button("▶️ Step successivo", use_container_width=True, type="primary"):
            step()
            st.rerun()
    else:
        st.success(f"Partita finita – Risultato: {st.session_state.board.result()}")

with c3:
    if st.sidebar.button("🧠 Carica agente addestrato", use_container_width=True):
        st.session_state.scenario = "B"
        st.session_state.board = chess.Board()
        st.session_state.rewards = [0]
        st.session_state.reward_log = ["Start"]
        st.session_state.move_index = 0
        st.rerun()

if st.sidebar.button("Reset totale", use_container_width=True):
    st
