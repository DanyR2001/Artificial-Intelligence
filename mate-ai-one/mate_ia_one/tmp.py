import chess_engine 

# Create an instance of the C++ class
engine = chess_engine.ChessEngine("r3kb1r/pppb1ppp/5n2/3Pp3/3n4/3QBN2/PPP2PPP/2KR1B1R w kq - 0 11")

# Call the C++ method
evaluation = engine.search_alphabeta_mvvlva(12)
best_move = engine.computer_move_alphabeta_mvvlva(12)
print(f"Best move: {best_move} with evaluation {evaluation}")