import streamlit as st
import chess
import chess.svg
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# ----------------- SCRIPT -----------------
SCRIPT_A = [
    "f2f3", "e7e5",
    "g2g4", "b8c6",
    "h2h4", "f8c5",
    "f3f4", "d8e7",
    "g1f3", "e5f4",
    "f3h2", "c5g1",
    "h2g4", "e7e3"   # scacco matto finale
]

SCRIPT_B = [
    "e2e4", "e7e5",
    "g1f3", "b8c6",
    "f1c4", "f8c5",
    "d1e2", "g8f6",
    "c2c3", "d7d6",
    "d2d4", "c5b6",
    "e1g1", "e8g8",
    "f3e5", "d6e5",
    "c4f7"
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
def compute_reward(board, scenario):
    if board.is_checkmate():
        return (120, "👑 SCACCO MATTO — Policy ottimale") if scenario == "B" else (-120, "💀 SCACCO MATTO — Policy pessima")

    if scenario == "A":
        return -5, "❌ Bad move (policy non addestrata)"

    return 4, "✅ Good move (policy addestrata)"

# ----------------- STEP -----------------
def step_game():
    script = SCRIPT_A if st.session_state.scenario == "A" else SCRIPT_B
    if st.session_state.step < len(script):
        move = chess.Move.from_uci(script[st.session_state.step])
        if move in st.session_state.board.legal_moves:
            st.session_state.board.push(move)
            r, label = compute_reward(st.session_state.board, st.session_state.scenario)
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
