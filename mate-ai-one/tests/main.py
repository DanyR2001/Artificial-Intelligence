from mate_ia_one.minimax_engine import ClassicEngine
import time

# board = chess.Board()
# while not board.is_game_over():
#   board.push(minimax_engine.computer_move_alphabeta(board, MAX_SEARCH_DEPTH))
#   stockfish.set_fen_position(board.fen())
#   board.push_uci(stockfish.get_best_move_time(100))
#   print(board)
#   print(board.fen())
# board_outcome = board.outcome()
# print(f"The game outcome is: {board_outcome}")
# if board_outcome.winner == chess.WHITE:
#   print("wins!")
# elif board_outcome.winner == chess.BLACK:
#   print("Stockfish wins!")
# else:
#   print("The game is a draw.")

def test_mvv_lva_evaluation():
  depth = 3

  # fen = "r1bqk2r/2pp1ppp/p4n2/1pb1N3/3nP3/1B1P4/PPP2PPP/RNBQK2R w KQkq - 1 8"
  # engine.board.set_fen(fen)
  # engine.evaluations_counter = 0
  # start_time = time.time()
  # minimax_evaluation = engine.search_alphabeta(depth=depth)
  # minimax_time = time.time() - start_time
  # print(f"fen evaluated: {fen}")
  # print(f"minimax evaluation: {minimax_evaluation}")
  # print(f"evaluated {engine.evaluations_counter} moves in {minimax_time:.4f} seconds")

  # fen = "rnbqkbnr/pppp1p1p/8/4p3/P3P1p1/5N2/1PPP1PPP/RNBQKB1R b KQkq - 0 4"
  # engine.board.set_fen(fen)
  # engine.evaluations_counter = 0
  # start_time = time.time()
  # minimax_evaluation = engine.search_alphabeta(depth=depth)
  # minimax_time = time.time() - start_time
  # print(f"fen evaluated: {fen}")
  # print(f"minimax evaluation: {minimax_evaluation}")
  # print(f"evaluated {engine.evaluations_counter} moves in {minimax_time:.4f} seconds")

  # fen = "r3kb1r/pppb1ppp/5n2/3Pp3/3n4/3QBN2/PPP2PPP/2KR1B1R w kq - 0 11"
  # engine.board.set_fen(fen)
  # engine.evaluations_counter = 0
  # start_time = time.time()
  # minimax_evaluation = engine.search_alphabeta(depth=depth)
  # minimax_time = time.time() - start_time
  # print(f"fen evaluated: {fen}")
  # print(f"minimax evaluation: {minimax_evaluation}")
  # print(f"evaluated {engine.evaluations_counter} moves in {minimax_time:.4f} seconds")

  # fen = "8/QP4bk/6pp/8/8/4r1P1/7P/1q3RK1 b - - 1 32"
  # engine.board.set_fen(fen)
  # engine.evaluations_counter = 0
  # start_time = time.time()
  # minimax_evaluation = engine.search_alphabeta(depth=depth)
  # minimax_time = time.time() - start_time
  # print(f"fen evaluated: {fen}")
  # print(f"minimax evaluation: {minimax_evaluation}")
  # print(f"evaluated {engine.evaluations_counter} moves in {minimax_time:.4f} seconds")

  # fen = "r1bqk2r/2pp1ppp/p4n2/1pb1N3/3nP3/1B1P4/PPP2PPP/RNBQK2R w KQkq - 1 8"
  # engine.board.set_fen(fen)
  # engine.evaluations_counter = 0
  # start_time = time.time()
  # move = engine.computer_move_alphabeta(depth)
  # minimax_time = time.time() - start_time
  # print(f"searched fen {fen}")
  # print(f"best move: {move}")
  # print(f"evaluated {engine.evaluations_counter} moves in {minimax_time:.4f} seconds")

  fen = "r1bqk1r1/1p1p1n2/p1n2pN1/2p1b2Q/2P1Pp2/1PN5/PB4PP/R4RK1"
  engine.board.set_fen(fen)
  engine.evaluations_counter = 0
  start_time = time.time()
  move = engine.computer_move_alphabeta(depth)
  minimax_time = time.time() - start_time
  print(f"searched fen {fen}")
  print(f"best move is Rxf4, engine found: {move}")
  print(f"evaluated {engine.evaluations_counter} moves in {minimax_time:.4f} seconds")




if __name__ == "__main__":
  # board = chess.Board()
  # stockfish = Stockfish(path="/opt/homebrew/Cellar/stockfish/17/bin/stockfish")

  engine = ClassicEngine()
  test_mvv_lva_evaluation()
