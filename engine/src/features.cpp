// AlphaGo-style input features and ladder reading.
#include <algorithm>
#include <cstring>
#include <memory>

#include "board.hpp"

namespace go {

namespace {

constexpr int kLadderDepth = 120;
constexpr int kLadderBudget = 3000;  // nodes per root query; on overflow assume escape
constexpr int kMaxCandidates = 32;

struct Marks {
  std::array<uint32_t, MAX_P> mark{};
  uint32_t stamp = 0;
  uint32_t next() {
    if (++stamp == 0) {
      mark.fill(0);
      stamp = 1;
    }
    return stamp;
  }
};
// Separate from board.cpp's marks: move_effect calls count_libs while its own set is live.
thread_local Marks g_effect_marks;

inline bool is_stone(Color c) { return c == BLACK || c == WHITE; }

}  // namespace

void Board::move_effect(int p, Color c, int lib_limit, int* libs_after, int* captured) const {
  Color o = opponent(c);
  int friends[4], nf = 0, caps[4], nc = 0;
  for (int d : dirs_) {
    int r = p + d;
    if (!is_stone(color_[r])) continue;
    int h = head_[r];
    if (color_[r] == c) {
      if (std::find(friends, friends + nf, h) == friends + nf) friends[nf++] = h;
    } else if (color_[r] == o && std::find(caps, caps + nc, h) == caps + nc &&
               count_libs(h, 2) == 1) {
      caps[nc++] = h;
    }
  }
  *captured = 0;
  for (int i = 0; i < nc; ++i) *captured += csize_[caps[i]];

  uint32_t s = g_effect_marks.next();
  auto& m = g_effect_marks.mark;
  m[p] = s;
  int libs = 0;
  auto visit = [&](int q) {
    for (int d : dirs_) {
      int r = q + d;
      if (m[r] == s) continue;
      bool lib = color_[r] == EMPTY ||
                 (color_[r] == o && std::find(caps, caps + nc, head_[r]) != caps + nc);
      if (lib) {
        m[r] = s;
        ++libs;
      }
    }
  };
  visit(p);
  for (int i = 0; i < nf && libs < lib_limit; ++i) {
    int q = friends[i];
    do {
      visit(q);
      q = next_[q];
    } while (q != friends[i] && libs < lib_limit);
  }
  *libs_after = std::min(libs, lib_limit);
}

// Defender to move; the chain at `chain` has exactly one liberty.
bool Board::ladder_escapes(int chain, int depth, int* budget) const {
  if (depth > kLadderDepth || --*budget <= 0) return true;
  Color def = color_[chain];
  int candidates[kMaxCandidates], nc = 0;
  count_libs(chain, 1, candidates);
  nc = 1;
  // Capturing an adjacent attacker chain in atari is also an escape attempt.
  int q = chain;
  do {
    for (int d : dirs_) {
      int r = q + d;
      int lib;
      if (color_[r] == opponent(def) && count_libs(r, 2, nullptr) == 1) {
        count_libs(r, 1, &lib);
        if (nc < kMaxCandidates && std::find(candidates, candidates + nc, lib) == candidates + nc)
          candidates[nc++] = lib;
      }
    }
    q = next_[q];
  } while (q != chain);

  for (int i = 0; i < nc; ++i) {
    int mv = candidates[i];
    if (!is_legal(mv, def)) continue;
    auto nb = std::make_unique<Board>(*this);
    nb->play(mv);
    int libs = nb->count_libs(chain, 3);
    if (libs >= 3) return true;
    if (libs == 2 && !nb->ladder_captures(chain, depth + 1, budget)) return true;
  }
  return false;
}

// Attacker to move; the chain at `chain` has exactly two liberties.
bool Board::ladder_captures(int chain, int depth, int* budget) const {
  if (depth > kLadderDepth || --*budget <= 0) return false;
  int libs[2];
  count_libs(chain, 2, libs);
  for (int mv : libs) {
    if (!is_legal(mv, to_move_)) continue;
    auto nb = std::make_unique<Board>(*this);
    nb->play(mv);
    if (nb->color_[chain] == EMPTY) return true;
    if (nb->count_libs(chain, 2) != 1) continue;
    if (!nb->ladder_escapes(chain, depth + 1, budget)) return true;
  }
  return false;
}

void Board::ladder_features(uint8_t* capture, uint8_t* escape) const {
  Color c = to_move_, o = opponent(c);
  const int nn = n_ * n_;
  std::memset(capture, 0, nn);
  std::memset(escape, 0, nn);
  // Visit each chain once via its head.
  for (int a = 0; a < nn; ++a) {
    int h = to_point(a);
    if (!is_stone(color_[h]) || head_[h] != h) continue;
    int libs[2];
    int nl = count_libs(h, 3, nullptr);
    if (color_[h] == o && nl == 2) {
      // Ladder capture: we atari with one liberty and the defender cannot escape.
      count_libs(h, 2, libs);
      for (int mv : libs) {
        if (capture[to_action(mv)] || !is_legal(mv, c)) continue;
        Board nb = *this;
        nb.play(mv);
        if (nb.count_libs(h, 2) != 1) continue;
        int budget = kLadderBudget;
        if (!nb.ladder_escapes(h, 0, &budget)) capture[to_action(mv)] = 1;
      }
    } else if (color_[h] == c && nl == 1) {
      // Ladder escape: extending (or capturing) gets out of the ladder.
      int candidates[kMaxCandidates], nc = 0;
      count_libs(h, 1, candidates);
      nc = 1;
      int q = h;
      do {
        for (int d : dirs_) {
          int r = q + d;
          int lib;
          if (color_[r] == o && count_libs(r, 2) == 1) {
            count_libs(r, 1, &lib);
            if (nc < kMaxCandidates && std::find(candidates, candidates + nc, lib) == candidates + nc)
          candidates[nc++] = lib;
          }
        }
        q = next_[q];
      } while (q != h);
      for (int i = 0; i < nc; ++i) {
        int mv = candidates[i];
        if (escape[to_action(mv)] || !is_legal(mv, c)) continue;
        Board nb = *this;
        nb.play(mv);
        int l = nb.count_libs(h, 3);
        int budget = kLadderBudget;
        if (l >= 3 || (l == 2 && !nb.ladder_captures(h, 0, &budget))) escape[to_action(mv)] = 1;
      }
    }
  }
}

void Board::features(uint8_t* out) const {
  const int nn = n_ * n_;
  std::memset(out, 0, static_cast<size_t>(plane::COUNT) * nn);
  auto P = [&](int pl) { return out + static_cast<size_t>(pl) * nn; };
  Color c = to_move_, o = opponent(c);

  // Liberties per chain, computed once at each head.
  std::array<int8_t, MAX_P> chain_libs;
  for (int a = 0; a < nn; ++a) {
    int p = to_point(a);
    if (is_stone(color_[p]) && head_[p] == p) chain_libs[p] = static_cast<int8_t>(count_libs(p, 4));
  }

  for (int a = 0; a < nn; ++a) {
    int p = to_point(a);
    Color cp = color_[p];
    P(plane::ONES)[a] = 1;
    if (c == BLACK) P(plane::BLACK_TO_MOVE)[a] = 1;
    if (cp == EMPTY) {
      P(plane::EMPTY_)[a] = 1;
      if (!is_legal(p, c)) continue;
      if (!is_eye(p, c)) P(plane::SENSIBLE)[a] = 1;
      int libs, captured;
      move_effect(p, c, 4, &libs, &captured);
      if (captured > 0) P(plane::CAPTURE_SIZE + std::min(captured, 3) - 1)[a] = 1;
      if (libs >= 1) P(plane::LIBS_AFTER + libs - 1)[a] = 1;
      if (libs == 1) {
        // Size of the resulting chain = 1 + adjacent friendly chain sizes.
        int size = 1, seen[4], ns = 0;
        for (int d : dirs_) {
          int r = p + d;
          if (color_[r] == c && std::find(seen, seen + ns, head_[r]) == seen + ns) {
            seen[ns++] = head_[r];
            size += csize_[head_[r]];
          }
        }
        P(plane::SELF_ATARI_SIZE + std::min(size, 3) - 1)[a] = 1;
      }
    } else {
      int libs = chain_libs[head_[p]];
      if (cp == c) {
        P(plane::OWN)[a] = 1;
        P(plane::OWN_LIBS + libs - 1)[a] = 1;
      } else if (cp == o) {
        P(plane::OPP)[a] = 1;
        P(plane::OPP_LIBS + libs - 1)[a] = 1;
      }
    }
  }
  for (int i = 0; i < HISTORY; ++i)
    if (history_[i] >= 0) P(plane::HISTORY_ + i)[to_action(history_[i])] = 1;
  ladder_features(P(plane::LADDER_CAPTURE), P(plane::LADDER_ESCAPE));
}

}  // namespace go
