import chess, chess.pgn, chess.polyglot
import random
import collections

class ClassicEngine:
  PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
  }
  INF = 9999
  MAX_SEARCH_DEPTH = 3

  evaluations_counter = 0
  board = chess.Board()
  transposition_table = {}
  
  def __init__(self):
    self.board.reset()
    self.evaluations_counter = 0

  def computer_move_alphabeta(self, depth=MAX_SEARCH_DEPTH):
    self.evaluations_counter = 0
    best_eval = -self.INF
    legal_moves = list(self.board.generate_legal_moves())
    
    for move in legal_moves:
      self.board.push(move)
      self.evaluations_counter += 1
      current_eval = -self.search_alphabeta(depth - 1, -self.INF, self.INF)
      self.board.pop()
      if current_eval > best_eval:
        best_eval = current_eval
        best_move = move
    print(f"### EVALUATED {self.evaluations_counter} MOVES ###")

    # case where opponent found checkmate
    if best_move == None: return random.choice(list(self.board.generate_legal_moves()))

    return best_move


  def search_alphabeta(self, depth=MAX_SEARCH_DEPTH, alpha=-INF, beta=INF):
    if depth == 0: return self.search_captures(alpha, beta)
    moves = list(self.board.generate_legal_moves())
    if len(moves) == 0:
      if self.board.is_check(): return -self.INF
      return 0
    for move in moves:
      self.board.push(move)
      self.evaluations_counter += 1
      evaluation = -self.search_alphabeta(depth - 1, -beta, -alpha)
      self.board.pop()
      if evaluation >= beta: return beta
      alpha = max(alpha, evaluation)
    return alpha


  def search_alphabeta_mvv_lva_transposition(self, depth=MAX_SEARCH_DEPTH, alpha=-INF, beta=INF):
    '''
    We use zobrist_hash() as the key.
    The entry types are:
    - exact: value is precisely known between original alpha/beta
    - lower: value is at least this good (cutoff occurred)
    - upper: value is at most this bad (all moves were worse)
    Implementation Notes:

    Implementation notes:
    1. Depth Comparison: Only uses entries if their stored depth is >= current search depth
    2. Window Adjustment: Uses transposition entries to narrow the alpha-beta window
    3. Entry Storage: Saves results with proper flags after search completion
    4. Beta Cutoff Handling: Stores lower bounds immediately on beta cutoffs
    '''

    original_alpha = alpha
    original_beta = beta

    # zobrist_key = self.board.zobrist_hash()
    zobrist_key = chess.polyglot.zobrist_hash(self.board)

    # check the transposition table. return the exact score if available, else adjust the stored alpha/beta bounds
    tt_entry = self.transposition_table.get(zobrist_key)
    if tt_entry and tt_entry["depth"] >= depth:
      if tt_entry["flag"] == "exact": return tt_entry["value"]
      elif tt_entry["flag"] == "lower": alpha = max(alpha, tt_entry["value"])
      elif tt_entry["flag"] == "upper": beta = min(beta, tt_entry["value"])
      # if adjusted windows collide, return the stored value
      if alpha >= beta: return tt_entry["value"]

    if depth == 0: return self.search_captures(alpha, beta)

    ordered_moves = self.order_moves(list(self.board.generate_legal_moves()))
    if not ordered_moves: return -self.INF if self.board.is_check() else 0

    best_value = -self.INF
    for move in ordered_moves:
      self.board.push(move)
      self.evaluations_counter += 1
      evaluation = -self.search_alphabeta_mvv_lva_transposition(depth - 1, -beta, -alpha)
      self.board.pop()
      
      # beta cutoff (pruning)
      if evaluation >= beta:
        # store lower bound in transposition table
        self.transposition_table[zobrist_key] = {
          "value": beta,
          "depth": depth,
          "flag": "lower"
        }
        return beta
      
      if evaluation > best_value:
        best_value = evaluation
        alpha = max(alpha, evaluation)
      
      if best_value <= original_alpha: flag = "upper"
      elif best_value >= original_beta: flag = "lower"
      else: flag = "exact"
    
    self.transposition_table[zobrist_key] = {
      "value": best_value,
      "depth": depth,
      "flag": flag
    }

    return best_value

  
  def search_alphabeta_mvv_lva(self, depth=MAX_SEARCH_DEPTH, alpha=-INF, beta=INF):
    if depth == 0: return self.search_captures(alpha, beta)
    
    ordered_moves = self.order_moves(list(self.board.generate_legal_moves()))
    if not ordered_moves: return -self.INF if self.board.is_check() else 0
    
    for move in ordered_moves:
      self.board.push(move)
      self.evaluations_counter += 1
      evaluation = -self.search_alphabeta_mvv_lva(depth - 1, -beta, -alpha)
      self.board.pop()
      if evaluation >= beta: return beta
      alpha = max(alpha, evaluation)
    
    return alpha


  def search_captures(self, alpha, beta):
    evaluation = self.basic_evaluation()
    if evaluation >= beta: return evaluation
    alpha = max(alpha, evaluation)

    capture_moves = list(self.board.generate_legal_captures())
    
    for move in capture_moves:
      self.board.push(move)
      self.evaluations_counter += 1
      evaluation = -self.search_captures(-beta, -alpha)
      self.board.pop()
      if evaluation >= beta: return beta
      alpha = max(alpha, evaluation)

    return alpha


  def basic_evaluation(self):
    self.evaluations_counter += 1
    white_score = 0
    black_score = 0
    for square in chess.SQUARES:
      piece = self.board.piece_at(square)
      if piece is not None:
        if piece.color == chess.WHITE: white_score += self.PIECE_VALUES[piece.piece_type]
        else: black_score += self.PIECE_VALUES[piece.piece_type]
    return white_score - black_score


  def computer_move_random(self):
    return random.choice(list(self.board.generate_legal_moves()))

  
  def get_board_pgn(self):
    game = chess.pgn.Game()
    switchyard = collections.deque()
    while self.board.move_stack: switchyard.append(self.board.pop())
    game.setup(self.board)
    node = game
    while switchyard:
      move = switchyard.pop()
      node = node.add_variation(move)
      self.board.push(move)
    game.headers["Result"] = self.board.result()
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    return game.accept(exporter)


  def order_moves(self, moves):
    scored_moves = []
    for move in moves:
      self.evaluations_counter += 1
      score = 0
      # Prioritize captures using MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
      # heuristic https://www.chessprogramming.org/MVV-LVA
      if self.board.is_capture(move):
        victim_piece = self.board.piece_at(move.to_square)
        victim_value = self.PIECE_VALUES[victim_piece.piece_type] if victim_piece else 0
        
        aggressor_piece = self.board.piece_at(move.from_square)
        aggressor_value = self.PIECE_VALUES[aggressor_piece.piece_type] if aggressor_piece else 0
        
        # MVV-LVA: victim * 10 - aggressor
        score += victim_value * 10 - aggressor_value
      # Prioritize promotions, especially to queen
      if move.promotion:
        promo_value = self.PIECE_VALUES.get(move.promotion, 0)
        score += promo_value * 5  # Higher multiplier for promotion
      # Add bonus for checks
      if self.board.gives_check(move):
        score += 20 # Arbitrary bonus for checks
      scored_moves.append((score, move))
    
    # Sort moves by descending score
    scored_moves.sort(key=lambda x: -x[0])
    return [move for _, move in scored_moves]
