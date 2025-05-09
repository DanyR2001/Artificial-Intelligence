#include "chess_engine.h"

#include <algorithm>
#include <chrono>

std::string ChessEngine::computer_move_alphabeta_mvvlva(int depth) {
  evaluations_counter = 0;
  chess::Move best_move;
  int best_evaluation = -INF;

  // Clear the transposition table at the start of a new search
  tt.clear();

  chess::Movelist legal_moves = generate_legal_moves();

  search_timed_out = false;
  search_start_time = std::chrono::steady_clock::now();
  search_time_limit = 3;  // 3 seconds limit

  for (const auto &move : legal_moves) {
    // Check time before processing this move
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - search_start_time).count();
    if (elapsed >= search_time_limit) {
      break;
    }

    board.makeMove(move);
    evaluations_counter++;
    int current_evaluation = -1 * search_alphabeta_mvvlva_rec(depth - 1, -INF, INF);
    board.unmakeMove(move);

    // Abort if search timed out
    if (search_timed_out) {
      break;
    }

    // Check time again after the move
    current_time = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - search_start_time).count();
    if (elapsed >= search_time_limit) {
      break;
    }

    if (current_evaluation > best_evaluation) {
      best_evaluation = current_evaluation;
      best_move = move;
    }
  }

  // Fallback if no move was evaluated in time
  if (best_move == chess::Move()) {  // Assuming default is invalid
    return chess::uci::moveToUci(legal_moves[0]);
  }

  return chess::uci::moveToUci(best_move);
}

int ChessEngine::search_alphabeta_mvvlva(int max_depth) {
  evaluations_counter = 0;
  int best_evaluation = -INF;
  chess::Move best_move;
  bool have_valid_result = false;

  // Clear the transposition table at the start of a new search
  tt.clear();

  search_timed_out = false;
  search_start_time = std::chrono::steady_clock::now();
  search_time_limit = 3;  // 3 seconds limit

  for (int current_depth = 1; current_depth <= max_depth; current_depth++) {
    int depth_best_eval = -INF;
    chess::Move depth_best_move;
    chess::Movelist legal_moves = generate_legal_moves();

    for (const auto &move : legal_moves) {
      auto current_time = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - search_start_time).count();
      if (elapsed >= search_time_limit) {
        search_timed_out = true;
        break;
      }
      board.makeMove(move);
      evaluations_counter++;
      int current_evaluation = -1 * search_alphabeta_mvvlva_rec(current_depth - 1, -INF, INF);
      board.unmakeMove(move);

      if (search_timed_out) break;

      if (current_evaluation > depth_best_eval) {
        depth_best_eval = current_evaluation;
        depth_best_move = move;
      }
    }

    // If we completed this depth without timeout, update our best result
    if (!search_timed_out) {
      best_evaluation = depth_best_eval;
      best_move = depth_best_move;
      have_valid_result = true;
    } else {
      break;  // Exit the iterative deepening loop
    }
  }

  // If we never completed even depth 1, return a fallback evaluation
  if (!have_valid_result) {
    // Simple evaluation of current position
    best_evaluation = search_captures(-INF, INF);
  }

  return best_evaluation;
}

int ChessEngine::search_alphabeta_mvvlva_rec(int depth, int alpha, int beta) {
  // Check for timeout
  auto current_time = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - search_start_time).count();
  if (elapsed >= search_time_limit) {
    search_timed_out = true;
    return 0;  // Dummy value, flag indicates timeout
  }

  // Get the current position hash
  uint64_t position_hash = board.hash();

  // Check if this position is already in the transposition table
  TTEntry *tt_entry = tt.probe(position_hash);
  if (tt_entry != nullptr && tt_entry->depth >= depth) {
    cache_hits++;
    // We found a valid entry with sufficient depth
    switch (tt_entry->flag) {
      case TT_EXACT:
        return tt_entry->score;
      case TT_LOWER:
        if (tt_entry->score >= beta) return tt_entry->score;
        break;
      case TT_UPPER:
        if (tt_entry->score <= alpha) return tt_entry->score;
        break;
    }
    // If we can't use the exact score, we might still be able to use the best move
  }

  // If at leaf node, use quiescence search
  if (depth == 0) return search_captures(alpha, beta);

  // Get and order legal moves (if we have a TT hit, try the TT move first)
  chess::Movelist legal_moves = generate_legal_moves();
  chess::Move tt_move = (tt_entry != nullptr) ? tt_entry->best_move : chess::Move();
  order_moves_with_tt_move(legal_moves, tt_move);

  if (legal_moves.size() == 0) {
    if (board.inCheck()) return -INF;  // Checkmate
    return 0;                          // Stalemate
  }

  chess::Move best_move;
  int original_alpha = alpha;

  for (const auto &move : legal_moves) {
    board.makeMove(move);
    evaluations_counter++;
    int evaluation = -1 * search_alphabeta_mvvlva_rec(depth - 1, -beta, -alpha);
    board.unmakeMove(move);

    if (search_timed_out) return 0;  // Early return on timeout

    if (evaluation >= beta) {
      // Store a lower bound in the transposition table
      tt.store(position_hash, depth, beta, TT_LOWER, move);
      return beta;  // Beta cutoff
    }

    if (evaluation > alpha) {
      alpha = evaluation;
      best_move = move;
    }
  }

  // Store the result in the transposition table
  TTEntryFlag flag = (alpha > original_alpha) ? TT_EXACT : TT_UPPER;
  tt.store(position_hash, depth, alpha, flag, best_move);

  return alpha;
}

void ChessEngine::order_moves_with_tt_move(chess::Movelist &moves, chess::Move tt_move) {
  if (tt_move != chess::Move()) {
    for (size_t i = 0; i < moves.size(); i++) {
      if (moves[i] == tt_move) {
        // Swap with the first move
        if (i > 0) {
          chess::Move temp = moves[0];
          moves[0] = moves[i];
          moves[i] = temp;
        }
        break;
      }
    }
  }

  // order the rest of the moves using MVV-LVA
  if (moves.size() > 1) {
    std::vector<std::pair<chess::Move, int>> scored_moves;
    size_t start_idx = (tt_move != chess::Move()) ? 1 : 0;
    for (size_t i = start_idx; i < moves.size(); i++) {
      int score = score_move(moves[i]);
      scored_moves.emplace_back(moves[i], score);
    }
    std::sort(scored_moves.begin(), scored_moves.end(),
              [](const auto &a, const auto &b) {
                return a.second > b.second;
              });
    for (size_t i = 0; i < scored_moves.size(); i++) {
      moves[start_idx + i] = scored_moves[i].first;
    }
  }
}

int ChessEngine::score_move(const chess::Move &move) {
  int score = 0;
  if (board.isCapture(move)) {
    chess::Piece victim_piece = board.at(move.to());
    chess::Piece aggressor_piece = board.at(move.from());
    int victim_value = piece_value(victim_piece.type());
    int aggressor_value = piece_value(aggressor_piece.type());
    score += victim_value * 10 - aggressor_value;
  }
  if (move.typeOf() == chess::Move::PROMOTION) {
    score += piece_value(move.promotionType());
  }
  chess::Board temp_board = board;
  temp_board.makeMove(move);
  if (temp_board.inCheck()) score += 50;
  return score;
}

int ChessEngine::piece_value(chess::PieceType type) {
  switch (static_cast<int>(type)) {
    case static_cast<int>(chess::PieceType::PAWN):
      return PAWN_VALUE;
    case static_cast<int>(chess::PieceType::KNIGHT):
      return KNIGHT_VALUE;
    case static_cast<int>(chess::PieceType::BISHOP):
      return BISHOP_VALUE;
    case static_cast<int>(chess::PieceType::ROOK):
      return ROOK_VALUE;
    case static_cast<int>(chess::PieceType::QUEEN):
      return QUEEN_VALUE;
    case static_cast<int>(chess::PieceType::KING):
      return KING_VALUE;
    default:
      return 0;
  }
}

chess::Movelist ChessEngine::generate_legal_moves() {
  chess::Movelist legal_moves;
  chess::movegen::legalmoves(legal_moves, board);
  return legal_moves;
}

int ChessEngine::search_captures(int alpha, int beta) {
  uint64_t position_hash = board.hash();

  TTEntry *tt_entry = tt.probe(position_hash);
  if (tt_entry != nullptr) {
    cache_hits++;
    switch (tt_entry->flag) {
      case TT_EXACT:
        return tt_entry->score;
      case TT_LOWER:
        if (tt_entry->score >= beta) return tt_entry->score;
        break;
      case TT_UPPER:
        if (tt_entry->score <= alpha) return tt_entry->score;
        break;
    }
  }

  int evaluation = basic_evaluation();
  if (evaluation >= beta) return evaluation;

  int original_alpha = alpha;
  alpha = std::max(alpha, evaluation);

  chess::Movelist legal_moves = generate_legal_moves();
  chess::Movelist capture_moves;
  for (const auto &move : legal_moves) {
    if (board.isCapture(move)) {
      capture_moves.add(move);
    }
  }

  // Order captures by MVV-LVA
  std::vector<std::pair<chess::Move, int>> scored_captures;
  for (const auto &move : capture_moves) {
    int score = score_move(move);
    scored_captures.emplace_back(move, score);
  }
  std::sort(scored_captures.begin(), scored_captures.end(),
            [](const auto &a, const auto &b) {
              return a.second > b.second;
            });

  chess::Move best_move;
  for (const auto &[move, score] : scored_captures) {
    board.makeMove(move);
    evaluations_counter++;
    evaluation = -1 * search_captures(-beta, -alpha);
    board.unmakeMove(move);
    if (evaluation >= beta) {
      tt.store(position_hash, 0, beta, TT_LOWER, move);
      return beta;
    }
    if (evaluation > alpha) {
      alpha = evaluation;
      best_move = move;
    }
  }

  TTEntryFlag flag = (alpha > original_alpha) ? TT_EXACT : TT_UPPER;
  tt.store(position_hash, 0, alpha, flag, best_move);

  return alpha;
}

int ChessEngine::basic_evaluation() {
  evaluations_counter++;
  int evaluation = 0;

  for (int sq = 0; sq < 64; ++sq) {
    chess::Square square = chess::Square(sq);
    chess::Piece piece = board.at(square);

    if (piece == chess::Piece::NONE) {
      continue;
    }

    chess::Color color = piece.color();
    chess::PieceType type = piece.type();

    int pieceValue = 0;
    int pstValue = 0;

    switch (static_cast<int>(type)) {
      case static_cast<int>(chess::PieceType::PAWN):
        pieceValue = PAWN_VALUE;
        pstValue = get_pst_value(square, color, PAWN_PST);
        break;
      case static_cast<int>(chess::PieceType::KNIGHT):
        pieceValue = KNIGHT_VALUE;
        pstValue = get_pst_value(square, color, KNIGHT_PST);
        break;
      case static_cast<int>(chess::PieceType::BISHOP):
        pieceValue = BISHOP_VALUE;
        pstValue = get_pst_value(square, color, BISHOP_PST);
        break;
      case static_cast<int>(chess::PieceType::ROOK):
        pieceValue = ROOK_VALUE;
        pstValue = get_pst_value(square, color, ROOK_PST);
        break;
      case static_cast<int>(chess::PieceType::QUEEN):
        pieceValue = QUEEN_VALUE;
        pstValue = get_pst_value(square, color, QUEEN_PST);
        break;
      case static_cast<int>(chess::PieceType::KING):
        pieceValue = KING_VALUE;
        pstValue = get_pst_value(square, color, KING_PST);
        break;
      default:
        break;
    }

    if (color == chess::Color::WHITE) {
      evaluation += pieceValue + pstValue;
    } else {
      evaluation -= pieceValue + pstValue;
    }
  }

  return evaluation;
}

int ChessEngine::get_pst_value(chess::Square square, chess::Color color, const int *pst_table) const {
  int rank = square.rank();
  int file = square.file();
  if (color == chess::Color::BLACK) {
    rank = 7 - rank;  // Mirror rank for black pieces
  }
  return pst_table[rank * 8 + file];
}

ChessEngine::ChessEngine(std::string fen) : tt(64) {  // Initialize with 64MB table
  board.setFen(fen);
}

ChessEngine::ChessEngine() : tt(64) {}