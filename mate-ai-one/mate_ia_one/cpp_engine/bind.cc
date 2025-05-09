#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "chess-library/include/chess.hpp"  // Include the chess library
#include "chess_engine.h"                   // Include the ChessEngine header
#include "transposition_table.h"

namespace py = pybind11;

PYBIND11_MODULE(chess_engine, m) {
  py::class_<chess::Board>(m, "Board")
    .def(py::init<>())
    .def("fen", &chess::Board::getFen)                // Use the correct method name
    .def("is_game_over", &chess::Board::isGameOver);  // Use the correct method name

  py::class_<TranspositionTable>(m, "TranspositionTable")
    // .def(py::init<>()) // not sure if needed
    .def(py::init<size_t>())
    .def("clear", &TranspositionTable::clear)
    .def("probe", &TranspositionTable::probe, py::arg("hash"))
    .def("store", &TranspositionTable::store, py::arg("hash"), py::arg("depth"), py::arg("score"), py::arg("flag"), py::arg("best_move"));

  // Bind ChessEngine
  py::class_<ChessEngine>(m, "ChessEngine")
    .def(py::init<>())
    .def(py::init<std::string>())
    .def("computer_move_alphabeta_mvvlva", &ChessEngine::computer_move_alphabeta_mvvlva, py::arg("depth"))
    .def("search_alphabeta_mvvlva", &ChessEngine::search_alphabeta_mvvlva, py::arg("depth"))

    // // TODO: ask why is this needed to claude
    // .def("generate_legal_moves", [](ChessEngine &self) {
    //   chess::Movelist moves = self.generate_legal_moves();
    //   py::list result;
    //   for (size_t i = 0; i < moves.size(); ++i) {
    //     result.append(py::str(chess::uci::moveToUci(moves[i])));  // Use the correct function
    //   }
    //   return result;
    // })

    .def("generate_legal_moves", &ChessEngine::generate_legal_moves)
    .def("score_move", &ChessEngine::score_move, py::arg("move"))
    .def("piece_value", &ChessEngine::piece_value, py::arg("type"))
    .def("search_captures", &ChessEngine::search_captures, py::arg("alpha"), py::arg("beta"))
    .def("basic_evaluation", &ChessEngine::basic_evaluation)

    .def_readwrite("board", &ChessEngine::board)
    .def_readwrite("evaluations_counter", &ChessEngine::evaluations_counter)
    .def_readwrite("cache_hits", &ChessEngine::cache_hits)

    .def_readonly("PAWN_VALUE", &ChessEngine::PAWN_VALUE)
    .def_readonly("KNIGHT_VALUE", &ChessEngine::KNIGHT_VALUE)
    .def_readonly("BISHOP_VALUE", &ChessEngine::BISHOP_VALUE)
    .def_readonly("ROOK_VALUE", &ChessEngine::ROOK_VALUE)
    .def_readonly("QUEEN_VALUE", &ChessEngine::QUEEN_VALUE)
    .def_readonly("KING_VALUE", &ChessEngine::KING_VALUE)

    .def("search_alphabeta_mvvlva_rec", &ChessEngine::search_alphabeta_mvvlva_rec, py::arg("depth"), py::arg("alpha"), py::arg("beta"))
    .def("order_moves_with_tt_move", &ChessEngine::order_moves_with_tt_move, py::arg("moves"), py::arg("tt_move"));
}