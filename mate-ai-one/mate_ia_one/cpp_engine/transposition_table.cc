#include "transposition_table.h"

#include <cstdint>
#include <vector>

#include "chess-library/include/chess.hpp"

TranspositionTable::TranspositionTable(size_t mb_size) {
  // Size in entries (approximately 16 bytes per entry)
  size = (mb_size * 1024 * 1024) / sizeof(TTEntry);
  table.resize(size);
  clear();
}
void TranspositionTable::clear() {
  for (size_t i = 0; i < size; i++) {
    table[i].hash = 0;
    table[i].depth = -1;
  }
}

TTEntry* TranspositionTable::probe(uint64_t hash) {
  size_t index = hash % size;
  TTEntry* entry = &table[index];

  if (entry->hash == hash) {
    return entry;
  }
  return nullptr;
}

void TranspositionTable::store(uint64_t hash, int depth, int score, TTEntryFlag flag, chess::Move best_move) {
  size_t index = hash % size;
  TTEntry* entry = &table[index];

  // Replacement strategy: always replace with deeper searches
  // or if empty or same position
  if (entry->hash == 0 || entry->hash == hash || depth >= entry->depth) {
    entry->hash = hash;
    entry->depth = depth;
    entry->score = score;
    entry->flag = flag;
    entry->best_move = best_move;
  }
}