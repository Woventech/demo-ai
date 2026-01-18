import streamlit as st
import chess
import chess.svg
import random
import matplotlib.pyplot as plt
import time

st.set_page_config(layout="wide")

# ---------- CONFIGURAZIONE SCENARI ----------
# Posizione iniziale con sviluppo pezzi per accelerare la partita (Apertura Italiana/Gambetto)
STARTING_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"

if "board" not in st.session_state:
    st.session_state.board = chess.Board(STARTING_FEN)

if "scenario" not in st.session_state:
    st.session_state.scenario = "A"  # A: Nero Esperto, B: Bianco Esperto

if "rewards" not in st.session_state:
    st.session_state.rewards = []

# ---------- "INTELLIGENZA" PRE-CARICATA (ZERO ATTESA) ----------
def get_pro_table(color):
    """Simula una Q-Table già addestrata con migliaia di partite"""
    # In un'app reale qui caricheresti un file JSON. 
    # Per la demo, questo dizionario 'finge' di avere memoria.
    return {"status": "trained", "color": color}

if "Q_white" not in st.session_state:
    st.session_state.Q_white = {} # Bianco inizia ignorante
if "Q_black" not in st.session_state:
    st.session_state.Q_black = get_pro_table(chess.BLACK) # Nero inizia esperto

# ---------- FUNZIONI LOGICHE ----------
def get_state(board):
    return " ".join(board.fen().split()[:4])

def engine_move(board, q_table, is_expert):
    """Sceglie la mossa: casuale se ignorante, 'ragionata' se esperto"""
    legal_moves = list(board.legal_moves)
    if not legal_moves: return None
    
    if not is_expert:
        return random.choice(legal_moves)
    
    # Simula la scelta della mossa migliore (Heuristic/Greedy)
    # Per scopi didattici, l'esperto cerca lo scacco o la cattura
    best_move = legal_moves[0]
    max_eval = -9999
    
    for move in legal_moves:
        board.push(move)
        # Valutazione semplice: materiale + scacco
        curr_eval = 0
        if board.is_checkmate(): curr_eval += 100
        if board.is_check(): curr_eval += 0.5
        
        # Valore pezzi
        values = {1:1, 2:3, 3:3, 4:5, 5:9, 6:0}
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p:
                val = values.get(p.piece_type, 0)
                curr_eval += val if p.color == board.turn else -val
        
        if curr_eval > max_eval:
            max_eval = curr_eval
            best_move = move
        board.pop()
    return best_move

def get_reward_display(board):
    """Calcola il reward dal punto di vista del BIANCO"""
    if board.is_checkmate():
        return 100 if board.result() == "1-0" else -100
    
    values = {1:1, 2:3, 3:3, 4:5, 5:9, 6:0}
    score = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            val = values.get(p.piece_type, 0)
            score += val if p.color == chess.WHITE else -val
    return score

# ---------- INTERFACCIA ----------
st.title("♟️ Reinforcement Learning Demo: Lo Scontro tra Agenti")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Scacchiera")
    # Mostra l'ultima mossa fatta per chiarezza
    last_move = st.session_state.board.peek() if st.session_state.board.move_stack else None
    board_svg = chess.svg.board(board=st.session_state.board, size=450, lastmove=last_move)
    st.image(board_svg)

with col2:
    st.subheader("Analisi della Conoscenza")
    
    # Grafico Reward
    if st.session_state.rewards:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(st.session_state.rewards, marker='o', color='royalblue', linewidth=2)
        ax.axhline(0, color='black', lw=0.5, ls='--')
        ax.set_ylabel("Reward (Bianco)")
        ax.set_xlabel("Mosse")
        st.pyplot(fig)
    
    # Stato didattico
    if st.session_state.scenario == "A":
        st.error("🔵 Scenario A: BIANCO (Non addestrato) vs NERO (Esperto)")
        st.write("Il Bianco muove a caso. Il Nero cercherà il matto in 8-10 mosse.")
    else:
        st.success("🔴 Scenario B: BIANCO (Addestrato 1000 ep) vs NERO (Debole)")
        st.write("Il Bianco ha 'imparato'. Vedrai la curva del reward salire.")

# ---------- CONTROLLI GIOCO ----------
st.divider()
c1, c2, c3 = st.columns(3)

with c1:
    if not st.session_state.board.is_game_over() and st.session_state.board.turn == chess.WHITE:
        if st.button("🤖 Fai muovere il BIANCO", use_container_width=True):
            is_exp = (st.session_state.scenario == "B")
            move = engine_move(st.session_state.board, st.session_state.Q_white, is_exp)
            st.session_state.board.push(move)
            st.session_state.rewards.append(get_reward_display(st.session_state.board))
            st.rerun()

with c2:
    if not st.session_state.board.is_game_over() and st.session_state.board.turn == chess.BLACK:
        if st.button("🌑 Fai muovere il NERO", use_container_width=True):
            is_exp = (st.session_state.scenario == "A")
            move = engine_move(st.session_state.board, st.session_state.Q_black, is_exp)
            st.session_state.board.push(move)
            st.session_state.rewards.append(get_reward_display(st.session_state.board))
            st.rerun()

with c3:
    if st.session_state.board.is_game_over():
        st.warning(f"PARTITA FINITA: {st.session_state.board.result()}")
        if st.button("Nuova Partita", use_container_width=True):
            st.session_state.board = chess.Board(STARTING_FEN)
            st.session_state.rewards = []
            st.rerun()

# ---------- SIDEBAR DIDATTICA (IL TASTO MAGICO) ----------
st.sidebar.header("Laboratorio AI")

if st.sidebar.button("🧠 ADDESTRA BIANCO (Simulazione)", type="primary"):
    with st.status("Esecuzione 1000 episodi di Reinforcement Learning...", expanded=True) as status:
        st.write("Generazione partite casuali...")
        time.sleep(0.8)
        st.write("Aggiornamento pesi Q-Table (Bellman Equation)...")
        time.sleep(0.8)
        st.write("Ottimizzazione policy...")
        time.sleep(0.4)
        status.update(label="Addestramento Completato!", state="complete", expanded=False)
    
    # Swap dei ruoli
    st.session_state.scenario = "B"
    st.session_state.Q_white = get_pro_table(chess.WHITE)
    st.session_state.Q_black = {} # Il nero perde la sua forza
    st.session_state.board = chess.Board(STARTING_FEN)
    st.session_state.rewards = []
    st.rerun()

if st.sidebar.button("Reset Totale"):
    st.session_state.clear()
    st.rerun()
