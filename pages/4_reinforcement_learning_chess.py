import streamlit as st
import chess
import chess.svg
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")


# -----------------PIECE VALUE--------------

PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}


# ----------------- SCRIPT -----------------
SCRIPT_A = [
    "d2d4", "g8f6",
    "f2f4", "d7d6",
    "g1f3", "c7c5",
    "f3g5", "h7h6",
    "b1c3", "h6g5",
    "c3d5", "f6d5",
    "c1e3", "d5e3",
    "d4c5", "e3d1",
    "b2b4", "d1e3",
    "c5d6", "d8d6",
    "c2c4", "e3c2",
    "e1f2", "d6d4",
    "f2f3", "d4f4"
    # scacco matto finale
]

SCRIPT_B = [
    "e2e4", "e7e5",
    "g1f3", "b8c6",
    "f1b5", "a7a6",
    "b5a4", "b7b5",
    "a4b3", "f8b4",
    "c2c3", "g8f6",
    "c3b4", "c6b4",
    "d2d4", "f6e4",
    "e1g1", "d8f6",
    "d1e1", "e5d3",
    "e1e4", "f8e7",
    "e4a8", "f6d8",
    "f3d4", "c6d4",
    "a8f3", "g7g5",
    "f3f7"  # ♕ SCACCO MATTO DEL BIANCO
]


# ----------------- STATE -----------------
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

# ----------------- REWARD -----------------
def compute_reward(prev_board, board, scenario):
    # scacco matto
    if board.is_checkmate():
        if scenario == "B":
            return 100, "👑 Scacco matto — policy convergente"
        else:
            return -100, "💀 Scacco matto subito — policy fallimentare"

    reward = 0
    explanation = []

    # catture
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

    # shaping didattico
    if scenario == "A":
        reward -= 1
        explanation.append("❌ mossa non strategica")
    else:
        reward += 1
        explanation.append("✅ mossa coerente")

    return reward, " | ".join(explanation)


# ----------------- STEP -----------------
def step_game():
    if st.session_state.board.is_game_over():
        return

    script = SCRIPT_A if st.session_state.scenario == "A" else SCRIPT_B
    idx = st.session_state.step

    if idx >= len(script):
        return

    move_uci = script[idx]
    move = chess.Move.from_uci(move_uci)

    if move not in st.session_state.board.legal_moves:
        st.error(f"Mossa illegale: {move_uci}")
        return

    # ⬇️ SALVIAMO LO STATO PRECEDENTE
    prev_board = st.session_state.board.copy()

    # ⬇️ APPLICHIAMO LA MOSSA
    st.session_state.board.push(move)

    # ⬇️ ORA LA FIRMA È CORRETTA
    r, label = compute_reward(
        prev_board,
        st.session_state.board,
        st.session_state.scenario
    )

    st.session_state.rewards.append(st.session_state.rewards[-1] + r)
    st.session_state.labels.append(label)
    st.session_state.step += 1


# ----------------- UI -----------------
st.title("♟️ Reinforcement Learning – Demo Didattica")

col_board, col_controls, col_data = st.columns([1.2, 0.5, 1])

with col_board:
    last = st.session_state.board.peek() if st.session_state.board.move_stack else None
    svg = chess.svg.board(st.session_state.board, size=420, lastmove=last)
    st.image(svg)

with col_controls:
    st.subheader("Controlli")
    if not st.session_state.board.is_game_over():
        if st.button("▶ Prossimo passo", type="primary", use_container_width=True):
            step_game()
            st.rerun()
    else:
        st.success("Partita terminata")

with col_data:
    st.subheader("Reward")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(st.session_state.rewards, marker="o")
    ax.axhline(0, linestyle="--", alpha=0.3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward cumulativa")
    st.pyplot(fig)
    st.info(st.session_state.labels[-1])

# ----------------- SIDEBAR -----------------
st.sidebar.subheader("Scenari")
if st.sidebar.button("🔴 Pre-training"):
    st.session_state.clear()
    st.session_state.scenario = "A"
    st.rerun()

if st.sidebar.button("🟢 Post-training"):
    st.session_state.clear()
    st.session_state.scenario = "B"
    st.rerun()
