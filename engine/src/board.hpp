// Go board engine: rules, liberties, ladders and AlphaGo-style input features.
//
// Coordinates exposed to Python are "actions": 0 .. N*N-1 for board points
// (index = y * N + x) and N*N for pass. Internally the board is padded with a
// one-point border so neighbor lookups never go out of bounds.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace go {

constexpr int MAX_N = 19;
constexpr int MAX_W = MAX_N + 2;
constexpr int MAX_P = MAX_W * MAX_W;
constexpr int HISTORY = 8;

enum Color : uint8_t { EMPTY = 0, BLACK = 1, WHITE = 2, BORDER = 3 };
inline Color opponent(Color c) { return c == BLACK ? WHITE : BLACK; }

constexpr int PASS = -1;
constexpr int NO_POINT = -2;

// Feature plane layout (all from the perspective of the player to move).
namespace plane {
constexpr int OWN = 0;
constexpr int OPP = 1;
constexpr int EMPTY_ = 2;
constexpr int ONES = 3;
constexpr int HISTORY_ = 4;            // 8 planes: move played k+1 plies ago
constexpr int OWN_LIBS = 12;           // 4 planes: 1, 2, 3, >=4 liberties
constexpr int OPP_LIBS = 16;           // 4 planes
constexpr int CAPTURE_SIZE = 20;       // 3 planes: move captures 1, 2, >=3 stones
constexpr int SELF_ATARI_SIZE = 23;    // 3 planes: move self-ataris 1, 2, >=3 stones
constexpr int LIBS_AFTER = 26;         // 4 planes: liberties after move 1, 2, 3, >=4
constexpr int LADDER_CAPTURE = 30;
constexpr int LADDER_ESCAPE = 31;
constexpr int SENSIBLE = 32;           // legal and does not fill own eye
constexpr int BLACK_TO_MOVE = 33;
constexpr int COUNT = 34;
}  // namespace plane

class Board {
 public:
  explicit Board(int n = 19, float komi = 7.5f, bool superko = false);

  int size() const { return n_; }
  float komi() const { return komi_; }
  bool superko() const { return superko_; }
  Color to_move() const { return to_move_; }
  int move_number() const { return move_num_; }
  int consecutive_passes() const { return passes_; }
  bool game_over() const { return passes_ >= 2; }
  uint64_t hash() const { return hash_; }
  int captures(Color c) const { return captures_[c == BLACK ? 0 : 1]; }
  int ko_action() const { return ko_ == NO_POINT ? -1 : to_action(ko_); }
  int last_action() const;

  void clear();
  void set_to_move(Color c) { to_move_ = c; }

  // Action-based API (what Python uses). With superko enabled these also forbid
  // recreating any earlier whole-board position (positional superko); the
  // point-based internals below and the SENSIBLE feature plane ignore it.
  bool is_legal_action(int a) const;
  void play_action(int a);                    // throws on illegal move
  void place_stone_action(int a, Color c);    // setup stones (handicap / SGF AB, AW)
  Color color_at_action(int a) const { return color_[to_point(a)]; }

  void stones(int8_t* out) const;             // N*N, 0 empty / 1 black / 2 white
  void legal_mask(uint8_t* out) const;        // N*N+1
  void sensible_mask(uint8_t* out) const;     // N*N+1, legal and not filling own eye
  void features(uint8_t* out) const;          // plane::COUNT * N * N
  float tromp_taylor_score() const;           // black - white - komi
  // Per point: 0 stone, 1/2 empty region bordered only by Black/White, 3 bordered by both (or neither).
  void region_owner(int8_t* out) const;
  std::string to_string() const;

  // Point-based internals, also used by ladder search.
  int to_point(int a) const;
  int to_action(int p) const;
  bool is_legal(int p, Color c) const;
  void play(int p);
  bool is_eye(int p, Color c) const;
  int count_libs(int p, int limit, int* out = nullptr) const;

 private:
  void merge(int a, int b);
  void remove_chain(int p);
  void push_history(int p);
  // Zobrist hash of the position after c plays the (legal) stone move p.
  uint64_t hash_after(int p, Color c) const;
  bool repeats_position(int p) const;
  void record_position();
  // Liberty count and capture size the move at p would produce, without playing it.
  void move_effect(int p, Color c, int lib_limit, int* libs_after, int* captured) const;
  bool ladder_escapes(int chain, int depth, int* budget) const;
  bool ladder_captures(int chain, int depth, int* budget) const;
  void ladder_features(uint8_t* capture, uint8_t* escape) const;

  int n_, w_;
  float komi_;
  bool superko_;
  // Hashes of every position reached through the action API, shared between
  // copies and cloned on write so internal board copies (ladders) stay cheap.
  std::shared_ptr<std::vector<uint64_t>> seen_;
  std::array<int, 4> dirs_;
  std::array<Color, MAX_P> color_;
  std::array<int16_t, MAX_P> head_;   // chain representative
  std::array<int16_t, MAX_P> next_;   // circular list of stones in a chain
  std::array<int16_t, MAX_P> csize_;  // chain size, valid at head
  std::array<int16_t, MAX_P> plib_;   // pseudo-liberties, valid at head (0 <=> captured)
  std::array<int16_t, HISTORY> history_;
  Color to_move_;
  int ko_;
  int passes_;
  int move_num_;
  std::array<int, 2> captures_;
  uint64_t hash_;
};

}  // namespace go
