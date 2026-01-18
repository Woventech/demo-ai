import streamlit as st
import chess
import chess.svg
import matplotlib.pyplot as plt
import time

st.set_page_config(layout="wide")

# ---------- REGIA DIDATTICA AVANZATA ----------
# Scenario A: Bianco "Ingenuo" (Perde pezzi e poi la partita)
# 1. e4 e5 | 2. d4 (pedone indifeso) exd4 | 3. Nf3 Nc6 | 4. Nxd4 (scambio) Nxd4 | 5. Qxd4 (espone regina) c5 | 6. Qe5+ (mossa inutile) Be7 | 7. Qxg7 (avidità punita) Bf6 | 8. Qg3 d5 | 9. exd5 Qxd5... e via verso il matto del Nero.
SCRIPT_A = ["e2e4", "e7e5", "d2d4", "e5d4", "g1f3", "b8c6", "f3d4", "c6d4", "d1d4", "c7c5", "d1e5", "f8e7", "e5g7", "e7f6", "g7g3", "d7d5"]

# Scenario B: Bianco "Esperto" (Sacrifica, cattura e vince)
# 1. e4 e5 | 2. Nf3 Nc6 | 3. Bc4 d6 | 4. d4 (attacco al centro) exd4 | 5. Nxd4 (riprende) Nf6 | 6. Nc3 Be7 | 7. O-O O-O | 8. f4 (espansione) Bg4 | 9. Qd3 Nxd4 | 10. Qxd4... il Bianco domina e cattura.
SCRIPT_B = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "d7d6", "d2d4", "e5d4", "f3d4", "g8f6", "b1c3", "f8e7", "e1g1", "e8g8", "f2f4", "c8g4", "d1d3", "c6d4", "d1d4"]

if "board" not in st.session_state:
    st.session_state.board = chess.Board()
if "scenario" not in st.session_state:
    st.session_state.scenario = "A"
if "rewards" not in st.session_state:
    st.session_state.rewards = []
if "move_count" not in st.session_state:
    st.session_state.move_count = 0

def get_reward_eval(board):
    """Calcolo dinamico del valore della posizione per il Bianco"""
    if board.is_checkmate():
        return 150 if board.result() == "1-0" else -150
    
    # Valori standard: P=1, C/A=3, T=5, D=9
    values = {1:1, 2:3, 3:3, 4:5, 5:9, 6:0}
    material = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            val = values.get(p.piece_type, 0)
            material += val if p.color == chess.WHITE else -val
    
    # Bonus per lo scacco (incoraggia l'aggressività)
    if board.is_check():
        material += 0.5 if board.turn == chess.BLACK else -0.5
        
    return material

def get_smart_move(board, scenario, move_index):
    # 1. Segui lo script per creare la situazione didattica
    script = SCRIPT_A if scenario == "A" else SCRIPT_B
    if move_index < len(script):
        m = chess.Move.from_uci(script[move_index])
        if m in board.legal_moves: return m

    # 2. Dopo lo script, l'agente esperto cerca di vincere
    is_white = board.turn
    is_expert = (scenario == "B" and is_white) or (scenario == "A" and not is_white)
    
    legal_moves = list(board.legal_moves)
    if not is_expert:
        return random.choice(legal_moves) if legal_moves else None

    # Mini-motore per l'esperto: cerca catture e scacchi
    best_m = legal_moves[0]
    max_v = -9999
    for m in legal_moves:
        board.push(m)
        v = get_reward_eval(board)
        if not is_white: v = -v
        if v > max_v:
            max_v = v
            best_m = m
        board.pop()
    return best_m

# ---------- INTERFACCIA ----------
st.title("♟️ Analisi RL: Rewards, Perdite e Strategia")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Campo di Battaglia")
    last_m = st.session_state.board.peek() if st.session_state.board.move_stack else None
    board_svg = chess.svg.board(board=st.session_state.board, size=480, lastmove=last_m)
    st.image(board_svg)

with col2:
    st.subheader("Grafico della Funzione Valore (Q-Value)")
    if st.session_state.rewards:
        fig, ax = plt.subplots(figsize=(6, 4))
        color = '#ff4b4b' if st.session_state.scenario == "A" else '#28a745'
        ax.plot(st.session_state.rewards, marker='o', color=color, linewidth=2, label="Vantaggio Bianco")
        ax.fill_between(range(len(st.session_state.rewards)), st.session_state.rewards, color=color, alpha=0.1)
        ax.axhline(0, color='white', linestyle='--', alpha=0.5)
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.set_xlabel("Numero Mosse", color='white')
        ax.set_ylabel("Reward Cumulativo", color='white')
        st.pyplot(fig)
    
    # Box informativo
    status_text = {
        "A": "🔴 **Agente Bianco NON Addestrato**: commette errori tattici, perde materiale e subisce l'iniziativa del Nero.",
        "B": "🟢 **Agente Bianco ADDESTRATO**: protegge i pezzi, cerca scambi vantaggiosi e accumula reward positivo."
    }
    st.info(status_text[st.session_state.scenario])

# ---------- GESTIONE TURNI ----------
st.divider()
c1, c2, c3 = st.columns(3)

with c1:
    if not st.session_state.board.is_game_over() and st.session_state.board.turn == chess.WHITE:
        if st.button("🤖 Azione BIANCO", use_container_width=True):
            move = get_smart_move(st.session_state.board, st.session_state.scenario, st.session_state.move_count)
            st.session_state.board.push(move)
            st.session_state.rewards.append(get_reward_eval(st.session_state.board))
            st.session_state.move_count += 1
            st.rerun()

with c2:
    if not st.session_state.board.is_game_over() and st.session_state.board.turn == chess.BLACK:
        if st.button("🌑 Azione NERO", use_container_width=True):
            move = get_smart_move(st.session_state.board, st.session_state.scenario, st.session_state.move_count)
            st.session_state.board.push(move)
            st.session_state.rewards.append(get_reward_eval(st.session_state.board))
            st.session_state.move_count += 1
            st.rerun()

with c3:
    if st.session_state.board.is_game_over():
        st.warning(f"Fine Partita: {st.session_state.board.result()}")
        if st.button("Ricomincia Demo"):
            st.session_state.board = chess.Board()
            st.session_state.rewards = []
            st.session_state.move_count = 0
            st.rerun()

# ---------- SIDEBAR CONTROLLI ----------
st.sidebar.header("Configurazione AI")
if st.sidebar.button("Simula Addestramento (1000 Episodi)", type="primary"):
    with st.status("Aggiornamento Q-Table...") as s:
        time.sleep(1)
        st.session_state.scenario = "B"
        st.session_state.board = chess.Board()
        st.session_state.rewards = []
        st.session_state.move_count = 0
        s.update(label="Agente Ottimizzato!", state="complete")
    st.rerun()

if st.sidebar.button("Reset a Scenario A"):
    st.session_state.clear()
    st.rerun()
