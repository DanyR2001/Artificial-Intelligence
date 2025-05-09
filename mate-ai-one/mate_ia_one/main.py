import pickle

from flask import Flask, request, jsonify
import chess
import os

from networkx.algorithms.shortest_paths.dense import reconstruct_path

from engine import RandomEngine, AlphaBetaEngine, MLEngine, PyTorchMLEngine

from modelNNUE import NNUE
from modelNNUEpythorch import NNUEP
import torch


app = Flask(__name__)
engine = None

ENGINE_TYPE = int(os.getenv("ENGINE", 3))  # 0=Random, 1=AlphaBeta, 2=ML, 3=Pythorch
MODEL_PATH = "/weight/nnue_model.npz"
PYTORCH_MODEL_PATH = "../weight/weight12/1500000_100_769.pth"

@app.route("/")
def newgame():
    global engine
    board = chess.Board()

    if ENGINE_TYPE == 0:
        engine = RandomEngine(board)
    elif ENGINE_TYPE == 1:
        engine = AlphaBetaEngine(board)
    elif ENGINE_TYPE == 2:
        model = NNUE.load(MODEL_PATH)
        engine = MLEngine(board, model)
    elif ENGINE_TYPE == 3:  # <-- Nuovo caso
        input_size = 769
        hidden_sizes = [769, 384, 192, 96]
        dropouts = 0.1
        model = NNUEP.load(PYTORCH_MODEL_PATH,input_size=input_size,hidden_sizes=hidden_sizes,dropout_rate=dropouts)
        engine = PyTorchMLEngine(board, model)

    ret = open("./index.html").read()
    return ret


@app.route("/move")
def move():
    if engine.board.is_game_over():
        print(f"Game over prima della mossa: Checkmate={engine.board.is_checkmate()}, Stalemate={engine.board.is_stalemate()}, Repetition={engine.board.is_repetition()}, Fifty moves={engine.board.is_fifty_moves()}, Can claim draw={engine.board.can_claim_draw()}")
        return game_over_response()
    try:
        user_move = engine.board.parse_uci(request.args['move'])
        engine.board.push(user_move)
        print(f"Dopo mossa utente, FEN: {engine.board.fen()}")
        if engine.board.is_game_over():
            print(f"Game over dopo mossa utente: Checkmate={engine.board.is_checkmate()}, Stalemate={engine.board.is_stalemate()}, Repetition={engine.board.is_repetition()}, Fifty moves={engine.board.is_fifty_moves()}, Can claim draw={engine.board.can_claim_draw()}")
            return game_over_response()
        ai_move = engine.computer_move(depth=2, beam_width=4, temperature=1)
        if ai_move is None:
            print("ai not moves")
            return game_over_response()
        print(f"Mosse legali per il nero: {engine.board.legal_moves}")
        print(f"Mossa scelta dal motore: {ai_move.uci()}")
        engine.board.push(ai_move)
        print(f"Dopo mossa AI, FEN: {engine.board.fen()}")
        return json_response()     
    except Exception as e:
        print(f"Errore: {e}")
        return json_response(error=str(e))
    
def who_is_in_check():
    if engine.board.is_attacked_by(chess.BLACK, engine.board.king(chess.WHITE)):
        print("check detected white king is in check")
        return "w"
    elif engine.board.is_attacked_by(chess.WHITE, engine.board.king(chess.BLACK)):
        print("check detected black king is in check")
        return "b"

def json_response(error=None):
    return jsonify({
        'fen': engine.board.fen(),
        'pgn': engine.get_board_pgn(),
        'evaluation': float(engine.evaluate_position()),
        'game_over': engine.board.is_game_over(),
        'is_check' : engine.board.is_check(), #restituisce se c'è un matto
        'who_is_in_check': who_is_in_check(),
        'white_king_position': chess.square_name(engine.board.king(chess.WHITE)),
        'black_king_position': chess.square_name(engine.board.king(chess.BLACK)),
        'error': error
    })


def game_over_response():
    if engine.board.is_checkmate():
        message = "Checkmate"
    elif engine.board.is_stalemate():
        message = "Stalemate"
    elif engine.board.can_claim_draw():
        if engine.board.is_fifty_moves():
            message = "Draw by 50-move rule"
        elif engine.board.is_repetition():
            message = "Draw by threefold repetition"
        else:
            message = "Draw by insufficient material or other rules"
    else:
        message = "Game Over"
    return jsonify({
        'game_over': True,
        'message': message
    })




if __name__ == "__main__":
    app.run(debug=True)