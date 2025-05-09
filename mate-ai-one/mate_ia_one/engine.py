import chess, chess.pgn
import random
import collections

import numpy as np

import chess_engine # our custom c++ engine
import torch
from stockfish import Stockfish



class ChessEngine:
  def __init__(self, board):
    self.board = board

  def computer_move(self):
    raise NotImplementedError("Sottoclasse deve implementare computer_move")

  def search_alphabeta(self, depth, alpha, beta):
    raise NotImplementedError("Sottoclasse deve implementare search_alphabeta")

  def evaluate_position(self):
    raise NotImplementedError("Sottoclasse deve implementare evaluate_position")

  def get_board_pgn(self):
    # Implementazione rimane la stessa di ClassicEngine
    game = chess.pgn.Game()
    switchyard = collections.deque()
    while self.board.move_stack:
      switchyard.append(self.board.pop())
    game.setup(self.board)
    node = game
    while switchyard:
      move = switchyard.pop()
      node = node.add_variation(move)
      self.board.push(move)
    game.headers["Result"] = self.board.result()
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    return game.accept(exporter)


class RandomEngine(ChessEngine):
  def computer_move(self):
    return random.choice(list(self.board.legal_moves))

  def search_alphabeta(self, depth, alpha, beta):
    return 0  # Non usato

  def evaluate_position(self):
    return 0  # Valutazione casuale, non necessaria


class AlphaBetaEngine(ChessEngine):
  def __init__(self, board):
    super().__init__(board)

  def computer_move(self):
    engine = chess_engine.ChessEngine(self.board.fen())
    best_move = engine.computer_move_alphabeta_mvvlva(12)
    return chess.Move.from_uci(best_move)

  def evaluate_position(self):
    engine = chess_engine.ChessEngine(self.board.fen())
    evaluation = engine.search_alphabeta_mvvlva(12)
    print(f"Valutazione della posizione (AlfaBeta CPP): {evaluation}")
    return evaluation/100


class MLEngine(AlphaBetaEngine):
  def __init__(self, board, model):
    super().__init__(board)
    self.model = model

  def evaluate_position(self):
    input_vector = self.board_to_input_vector(self.board)
    evaluation = self.model.evaluate(input_vector)
    #print(f"Valutazione della posizione (NNUE): {evaluation}")
    return float(evaluation[0])

  def board_to_input_vector(self, board):
    """
    Crea un vettore di input di dimensione 768 (senza feature del turno).
    Le 768 componenti sono organizzate come 12 (pezzi) x 64 (caselle).
    """
    input_vector = np.zeros(768)  # Dimensione: 12*64 = 768
    for square in range(64):
      piece = board.piece_at(square)
      if piece is not None:
        # Per i pezzi bianchi l'indice parte da 0, per i neri da 6
        color_index = 0 if piece.color == chess.WHITE else 6
        # piece_type in chess va da 1 a 6: sottraiamo 1 per avere un indice da 0 a 5
        piece_type_index = piece.piece_type - 1
        feature_index = (color_index + piece_type_index) * 64 + square
        input_vector[feature_index] = 1
    return input_vector

  def minimax_beam_alpha_beta(self, depth, alpha, beta, beam_width, is_maximizing):
    """
    Esegue una ricerca minimax con potatura alpha-beta espandendo solo i beam_width migliori rami.
    - depth: profondità residua.
    - alpha, beta: valori per la potatura.
    - beam_width: numero massimo di mosse espanse per ogni nodo.
    - is_maximizing: True se il nodo corrente deve massimizzare, False se minimizzare.
    Restituisce il valore (dal punto di vista del giocatore massimizzante) della posizione.
    """
    # Caso terminale: profondità zero o partita conclusa.
    if depth == 0 or self.board.is_game_over():
      val = self.evaluate_position()
      return val if is_maximizing else -val

    legal_moves = list(self.board.legal_moves)
    if not legal_moves:
      return self.evaluate_position()

    # Calcola una valutazione one-ply per ogni mossa per ordinare l'espansione.
    move_evals = []
    for move in legal_moves:
      self.board.push(move)
      heuristic = self.evaluate_position()
      self.board.pop()
      move_evals.append((heuristic, move))

    sorted_moves = sorted(move_evals, key=lambda x: x[0])
    moves_to_expand = [move for (_, move) in sorted_moves[:beam_width]]

    if is_maximizing:
      max_eval = -np.inf
      for move in moves_to_expand:
        self.board.push(move)
        eval_value = self.minimax_beam_alpha_beta(depth - 1, alpha, beta, beam_width, False)
        self.board.pop()
        max_eval = max(max_eval, eval_value)
        alpha = max(alpha, eval_value)
        if beta <= alpha:
          break  # Potatura alpha-beta
      return max_eval
    else:
      min_eval = np.inf
      for move in moves_to_expand:
        self.board.push(move)
        eval_value = self.minimax_beam_alpha_beta(depth - 1, alpha, beta, beam_width, True)
        self.board.pop()
        min_eval = min(min_eval, eval_value)
        beta = min(beta, eval_value)
        if beta <= alpha:
          break
      return min_eval

  def computer_move(self, depth=5, beam_width=3, temperature=1.0):
    """
    Sceglie la mossa tramite una ricerca minimax con potatura alpha-beta e beam search.
      - depth: profondità della ricerca (inclusa la mossa corrente).
      - beam_width: numero massimo di mosse espanse per nodo.
      - temperature: parametro per la scelta softmax; se ≤ 0 si sceglie in modo greedy.
    """
    legal_moves = list(self.board.legal_moves)
    if not legal_moves:
      return None

    move_values = []
    # Determina se il nodo radice è massimizzante basandosi sul turno corrente
    is_maximizing = (self.board.turn == chess.WHITE)
    for move in legal_moves:
      self.board.push(move)
      value = self.minimax_beam_alpha_beta(depth - 1, -np.inf, np.inf, beam_width, not is_maximizing)
      self.board.pop()
      move_values.append(value)

    move_values = np.array(move_values)
    if temperature <= 0:
      best_idx = int(np.argmax(move_values))
      return legal_moves[best_idx]

    # Distribuzione softmax
    exp_values = np.exp(move_values / temperature)
    probs = exp_values / np.sum(exp_values)
    selected_idx = np.random.choice(len(legal_moves), p=probs)
    return legal_moves[selected_idx]

class PyTorchMLEngine(MLEngine):  # <-- Nuova classe

  def __init__(self, board, model):
    super().__init__(board,model)
    self.model = model

  def board_to_input_vector(self, board):
    """
    Crea un vettore di input di dimensione 769.
    Le prime 768 componenti sono quelle relative alla presenza dei pezzi (12x64) e l'ultima componente rappresenta il turno.
    """
    input_vector = np.zeros(769)  # 768 per i pezzi + 1 per il turno
    for square in range(64):
      piece = board.piece_at(square)
      if piece is not None:
        color_index = 0 if piece.color == chess.WHITE else 6
        piece_type_index = piece.piece_type - 1
        feature_index = (color_index + piece_type_index) * 64 + square
        input_vector[feature_index] = 1
    # Aggiungi il turno: 1 se bianco, 0 se nero
    input_vector[768] = 1 if board.turn == chess.WHITE else 0
    return input_vector

  def evaluate_position(self):
    input_vector = self.board_to_input_vector(self.board)
    with torch.no_grad():
      tensor_input = torch.tensor(input_vector, dtype=torch.float32)
      evaluation = self.model(tensor_input).item()
    # print(f"Valutazione della posizione (NNUEP): {evaluation}")
    return evaluation


class StockfishEngine:
  def __init__(self, board, path="/opt/homebrew/Cellar/stockfish/17/bin/stockfish"):
    self.board = board
    self.stockfish = Stockfish(path=path)
    self.stockfish.set_fen_position(board.fen())

  def computer_move(self, depth=15, **kwargs):
    self.stockfish.set_depth(depth)
    best_move = self.stockfish.get_best_move()
    return chess.Move.from_uci(best_move) if best_move else None