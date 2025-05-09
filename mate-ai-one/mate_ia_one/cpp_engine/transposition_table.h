#ifndef TRANSPOSITIONTABLE_H
#define TRANSPOSITIONTABLE_H

#include <cstdint>
#include <vector>

#include "chess-library/include/chess.hpp"

// Constants for transposition table
enum TTEntryFlag {
  TT_EXACT,  // Exact evaluation
  TT_LOWER,  // Lower bound (failed high)
  TT_UPPER   // Upper bound (failed low)
};

// Transposition table entry
struct TTEntry {
  uint64_t hash;          // Position hash
  int depth;              // Search depth
  int score;              // Evaluation score
  TTEntryFlag flag;       // Type of node
  chess::Move best_move;  // Best move found
};

// Transposition table
class TranspositionTable {
 private:
  std::vector<TTEntry> table;
  size_t size;

 public:
  TranspositionTable(size_t mb_size = 256);
  void clear();
  TTEntry* probe(uint64_t hash);
  void store(uint64_t hash, int depth, int score, TTEntryFlag flag, chess::Move best_move);
};

#endif  // TRANSPOSITIONTABLE_H