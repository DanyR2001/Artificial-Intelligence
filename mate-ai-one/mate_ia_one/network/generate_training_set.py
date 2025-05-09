import chess.pgn
from state import State
import numpy as np

dataset_folder = "../../dataset/"
dataset_name = "lichess_elite_2023-11.pgn"

dataset_path = dataset_folder + dataset_name

y_values = {'1/2-1/2':0, '0-1':-1, '1-0':1}

if __name__ == "__main__":
  X, Y = [], []
  games_counter = 0

  with open(dataset_path) as pgn_file:
    while games_counter < 25:
      game = chess.pgn.read_game(pgn_file)
      if game is None: break
      
      game_result = game.headers["Result"]

      y_value = y_values[game_result]
      board = game.board()

      for i, move in enumerate(game.mainline_moves()):
        board.push(move)
        ser = State(board).serialize()
        X.append(ser)
        Y.append(y_value)

      games_counter += 1
      
  X = np.array(X)
  Y = np.array(Y)
  np.savez("dataset_25.npz", X, Y)