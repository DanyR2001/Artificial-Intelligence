import chess
from stockfish import Stockfish
from mate_ia_one.minimax_engine import ClassicEngine

INF = 9999
MAX_SEARCH_DEPTH = 3

def test_engine():
  while not engine.board.is_game_over():
    engine.board.push(engine.computer_move_alphabeta(MAX_SEARCH_DEPTH))
    stockfish.set_fen_position(engine.board.fen())
    engine.board.push_uci(stockfish.get_best_move_time(100))
    print(engine.board)
    print(engine.board.fen())

  board_outcome = engine.board.outcome()
  print(f"The game outcome is: {board_outcome}")

  if board_outcome.winner == chess.WHITE:
    print("Python engine wins!")
  elif board_outcome.winner == chess.BLACK:
    print("Stockfish wins!")
  else:
    print("The game is a draw.")

if __name__ == "__main__":
  board = chess.Board()
  # stockfish = Stockfish(path="/opt/homebrew/Cellar/stockfish/17/bin/stockfish", depth=3, parameters={"Skill Level": 1})
  stockfish = Stockfish(path="/opt/homebrew/Cellar/stockfish/17/bin/stockfish")
  stockfish.update_engine_parameters({"Skill Level":1})
  engine = ClassicEngine()
  test_engine()

