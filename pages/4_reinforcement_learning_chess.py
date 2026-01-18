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
    "f2f3",
    "e7e5",
    "g2g4",
    "d8h4",   # scacco
    "e1f2",
    "h4g5",   # cattura pedone
    "h2h4",
    "g5g3"    # scacco matto del nero
]

# 🟢 POST-TRAINING – Bianco addestrato, vince
SCRIPT_B = [
    "e2e4", "e7e5",
    "g1f3", "b8c6",
    "f1c4", "f8c5",
    "c4f7",   # + cattura alfiere
    "e8f7",
    "f3e5",   # + cattura pedone
    "c6e5",
    "d1h5",
    "g7g6",
    "h5e5",   # + cattura cavallo
    "f7g8",
    "e5e6"    # scacco matto del bianco
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
    # Scacco matto
    if board.is_checkmate():
        return (-120, "❌ Scacco matto subito") if scenario == "A" else (120, "👑 Scacco matto inflitto")

    # Scenario A: penalità crescente
    if scenario == "A":
        penalty = -15 * len(st.session_state.rewards)
        return penalty, "❌ Mossa inefficiente"

    # Scenario B: reward per materiale
    score = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            v = PIECE_VALUES.get(p.piece_type, 0)
            score += v if p.color == chess.WHITE else -v

    return score, "✅ Vantaggio materiale"

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

col_board, col_data = st.columns([1.3, 1])

# ---------- SCACCHIERA ----------
with col_board:
    st.subheader("Scacchiera")

    last_move = (
        st.session_state.board.peek()
        if st.session_state.board.move_stack else None
    )

    svg = chess.svg.board(
        board=st.session_state.board,
        size=440,
        lastmove=last_move
    )
    st.image(svg)

# ---------- DATI ----------
with col_data:
    st.subheader("Curva Reward / Loss")

    fig, ax = plt.subplots(figsize=(6, 4))
    color = "red" if st.session_state.scenario == "A" else "green"

    ax.plot(
        st.session_state.rewards,
        marker="o",
        linewidth=2,
        color=color
    )
    ax.axhline(0, linestyle="--", alpha=0.3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")

    st.pyplot(fig)

    st.markdown("### Ultimo feedback dell’agente")
    st.info(st.session_state.reward_log[-1])

    if st.session_state.scenario == "A":
        st.error("🔴 Policy iniziale – l’agente sbaglia ed esplora")
    else:
        st.success("🟢 Policy addestrata – l’agente sfrutta ciò che ha imparato")

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
