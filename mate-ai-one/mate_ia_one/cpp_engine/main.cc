#include <chrono>
#include <iostream>

#include "chess-library/include/chess.hpp"
#include "chess_engine.h"

int main() {
  ChessEngine engine;
  int depth = 8;

  std::string fen = "r1bqk2r/2pp1ppp/p4n2/1pb1N3/3nP3/1B1P4/PPP2PPP/RNBQK2R w KQkq - 1 8";
  engine.board.setFen(fen);
  engine.evaluations_counter = 0;
  engine.cache_hits = 0;
  auto start_time = std::chrono::high_resolution_clock::now();
  int minimax_evaluation = engine.search_alphabeta_mvvlva(depth);
  std::chrono::duration<double> minimax_time = std::chrono::high_resolution_clock::now() - start_time;
  std::cout << "fen evaluated: " << fen << std::endl;
  std::cout << "minimax evaluation: " << minimax_evaluation << std::endl;
  std::cout << "evaluated " << engine.evaluations_counter << " moves in " << minimax_time.count() << " seconds" << std::endl;
  std::cout << "retrieved " << engine.cache_hits << " moves" << std::endl;

  // engine.evaluations_counter = 0;
  // start_time = std::chrono::high_resolution_clock::now();
  // minimax_evaluation = engine.search_alphabeta_mvvlva(depth, -engine.INF, engine.INF);
  // minimax_time = std::chrono::high_resolution_clock::now() - start_time;
  // std::cout << "fen evaluated: " << fen << std::endl;
  // std::cout << "mvvlva evaluation: " << minimax_evaluation << std::endl;
  // std::cout << "evaluated " << engine.evaluations_counter << " moves in " << minimax_time.count() << " seconds" << std::endl;

  // fen = "rnbqkbnr/pppp1p1p/8/4p3/P3P1p1/5N2/1PPP1PPP/RNBQKB1R b KQkq - 0 4";
  // engine.board.setFen(fen);
  // engine.evaluations_counter = 0;
  // start_time = std::chrono::high_resolution_clock::now();
  // minimax_evaluation = engine.search_alphabeta(depth, -engine.INF, engine.INF);
  // minimax_time = std::chrono::high_resolution_clock::now() - start_time;
  // std::cout << "fen evaluated: " << fen << std::endl;
  // std::cout << "minimax evaluation: " << minimax_evaluation << std::endl;
  // std::cout << "evaluated " << engine.evaluations_counter << " moves in " << minimax_time.count() << " seconds" << std::endl;

  // fen = "r3kb1r/pppb1ppp/5n2/3Pp3/3n4/3QBN2/PPP2PPP/2KR1B1R w kq - 0 11";
  // engine.board.setFen(fen);
  // engine.evaluations_counter = 0;
  // start_time = std::chrono::high_resolution_clock::now();
  // minimax_evaluation = engine.search_alphabeta(depth, -engine.INF, engine.INF);
  // minimax_time = std::chrono::high_resolution_clock::now() - start_time;
  // std::cout << "fen evaluated: " << fen << std::endl;
  // std::cout << "minimax evaluation: " << minimax_evaluation << std::endl;
  // std::cout << "evaluated " << engine.evaluations_counter << " moves in " << minimax_time.count() << " seconds" << std::endl;

  // fen = "8/QP4bk/6pp/8/8/4r1P1/7P/1q3RK1 b - - 1 32";
  // engine.board.setFen(fen);
  // engine.evaluations_counter = 0;
  // start_time = std::chrono::high_resolution_clock::now();
  // minimax_evaluation = engine.search_alphabeta(depth, -engine.INF, engine.INF);
  // minimax_time = std::chrono::high_resolution_clock::now() - start_time;
  // std::cout << "fen evaluated: " << fen << std::endl;
  // std::cout << "minimax evaluation: " << minimax_evaluation << std::endl;
  // std::cout << "evaluated " << engine.evaluations_counter << " moves in " << minimax_time.count() << " seconds" << std::endl;

  // // random position evaluation
  // fen = "r1bqk2r/2pp1ppp/p4n2/1pb1N3/3nP3/1B1P4/PPP2PPP/RNBQK2R w KQkq - 1 8";
  // engine.board.setFen(fen);
  // engine.evaluations_counter = 0;
  // start_time = std::chrono::high_resolution_clock::now();
  // chess::Move computer_move = engine.computer_move_alphabeta(depth);
  // minimax_time = std::chrono::high_resolution_clock::now() - start_time;
  // std::cout << "searched fen: " << fen << std::endl;
  // std::cout << "best move: " << chess::uci::moveToUci(computer_move) << std::endl;
  // std::cout << "evaluated " << engine.evaluations_counter << " moves in " << minimax_time.count() << " seconds" << std::endl;

  // // Rxf4
  // std::string fen = "r1bqk1r1/1p1p1n2/p1n2pN1/2p1b2Q/2P1Pp2/1PN5/PB4PP/R4RK1";
  // engine.board.setFen(fen);
  // engine.evaluations_counter = 0;
  // auto start_time = std::chrono::high_resolution_clock::now();
  // chess::Move computer_move = engine.computer_move_alphabeta_mvvlva(depth);
  // auto end_time = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  // std::cout << "searched fen: " << fen << std::endl;
  // std::cout << "best move is Rxf4, engine found: " << chess::uci::moveToUci(computer_move) << std::endl;
  // std::cout << "evaluated " << engine.evaluations_counter << " moves in " << duration << " ms" << std::endl;
  
  // fen = "r1n2N1k/2n2K1p/3pp3/5Pp1/b5R1/8/1PPP4/8";
  // engine.board.setFen(fen);
  // engine.evaluations_counter = 0;
  // start_time = std::chrono::high_resolution_clock::now();
  // computer_move = engine.computer_move_alphabeta_mvvlva(depth);
  // end_time = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  // std::cout << "searched fen: " << fen << std::endl;
  // std::cout << "best move is Rxf4, engine found: " << chess::uci::moveToUci(computer_move) << std::endl;
  // std::cout << "evaluated " << engine.evaluations_counter << " moves in " << duration << " ms" << std::endl;
  
  return 0;
}