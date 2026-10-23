#include "board.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace go {

namespace {

// Stamp-based visited set so liberty walks never need to clear an array.
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
thread_local Marks g_marks;

struct Zobrist {
  std::array<std::array<uint64_t, 3>, MAX_P> table;
  Zobrist() {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (auto& row : table)
      for (auto& v : row) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        v = s;
      }
  }
};
const Zobrist g_zobrist;

inline bool is_stone(Color c) { return c == BLACK || c == WHITE; }

}  // namespace

Board::Board(int n, float komi, bool superko) : n_(n), w_(n + 2), komi_(komi), superko_(superko) {
  if (n < 2 || n > MAX_N) throw std::invalid_argument("board size must be in [2, 19]");
  dirs_ = {-1, 1, -w_, w_};
  clear();
}

void Board::clear() {
  color_.fill(BORDER);
  for (int y = 0; y < n_; ++y)
    for (int x = 0; x < n_; ++x) color_[(y + 1) * w_ + x + 1] = EMPTY;
  for (int p = 0; p < MAX_P; ++p) {
    head_[p] = next_[p] = static_cast<int16_t>(p);
    csize_[p] = plib_[p] = 0;
  }
  history_.fill(NO_POINT);
  to_move_ = BLACK;
  ko_ = NO_POINT;
  passes_ = 0;
  move_num_ = 0;
  captures_ = {0, 0};
  hash_ = 0;
  seen_.reset();
  record_position();
}

void Board::record_position() {
  if (!superko_) return;
  if (!seen_)
    seen_ = std::make_shared<std::vector<uint64_t>>();
  else if (seen_.use_count() > 1)
    seen_ = std::make_shared<std::vector<uint64_t>>(*seen_);
  seen_->push_back(hash_);
}

uint64_t Board::hash_after(int p, Color c) const {
  uint64_t h = hash_ ^ g_zobrist.table[p][c];
  Color o = opponent(c);
  std::array<int, 4> taken;
  int n_taken = 0;
  for (int d : dirs_) {
    int r = p + d;
    if (color_[r] != o || count_libs(r, 2) != 1) continue;
    int hd = head_[r];
    if (std::find(taken.begin(), taken.begin() + n_taken, hd) != taken.begin() + n_taken) continue;
    taken[n_taken++] = hd;
    int q = hd;
    do {
      h ^= g_zobrist.table[q][o];
      q = next_[q];
    } while (q != hd);
  }
  return h;
}

bool Board::repeats_position(int p) const {
  if (!superko_ || p == PASS || !seen_) return false;
  uint64_t h = hash_after(p, to_move_);
  return std::find(seen_->begin(), seen_->end(), h) != seen_->end();
}

int Board::to_point(int a) const {
  if (a == n_ * n_) return PASS;
  if (a < 0 || a > n_ * n_) throw std::out_of_range("action out of range");
  return (a / n_ + 1) * w_ + a % n_ + 1;
}

int Board::to_action(int p) const {
  if (p == PASS) return n_ * n_;
  return (p / w_ - 1) * n_ + p % w_ - 1;
}

int Board::last_action() const {
  return history_[0] == NO_POINT ? -1 : to_action(history_[0]);
}

int Board::count_libs(int p, int limit, int* out) const {
  uint32_t s = g_marks.next();
  auto& m = g_marks.mark;
  int count = 0;
  int q = p;
  do {
    for (int d : dirs_) {
      int r = q + d;
      if (color_[r] == EMPTY && m[r] != s) {
        m[r] = s;
        if (out) out[count] = r;
        if (++count >= limit) return count;
      }
    }
    q = next_[q];
  } while (q != p);
  return count;
}

bool Board::is_legal(int p, Color c) const {
  if (p == PASS) return true;
  if (color_[p] != EMPTY) return false;
  if (p == ko_ && c == to_move_) return false;
  for (int d : dirs_) {
    int r = p + d;
    Color cr = color_[r];
    if (cr == EMPTY) return true;
    // p itself is one of the chain's liberties, so >= 2 means one survives.
    if (cr == c && count_libs(r, 2) >= 2) return true;
    if (cr == opponent(c) && count_libs(r, 2) == 1) return true;
  }
  return false;
}

bool Board::is_legal_action(int a) const {
  int p = to_point(a);
  return is_legal(p, to_move_) && !repeats_position(p);
}

void Board::merge(int a, int b) {
  if (csize_[a] < csize_[b]) std::swap(a, b);
  int q = b;
  do {
    head_[q] = static_cast<int16_t>(a);
    q = next_[q];
  } while (q != b);
  std::swap(next_[a], next_[b]);
  csize_[a] += csize_[b];
  plib_[a] += plib_[b];
}

void Board::remove_chain(int p) {
  int h = head_[p];
  int q = h;
  do {
    int nq = next_[q];
    hash_ ^= g_zobrist.table[q][color_[q]];
    color_[q] = EMPTY;
    for (int d : dirs_) {
      int r = q + d;
      if (is_stone(color_[r])) plib_[head_[r]]++;
    }
    head_[q] = next_[q] = static_cast<int16_t>(q);
    csize_[q] = plib_[q] = 0;
    q = nq;
  } while (q != h);
}

void Board::push_history(int p) {
  for (int i = HISTORY - 1; i > 0; --i) history_[i] = history_[i - 1];
  history_[0] = static_cast<int16_t>(p);
}

void Board::play(int p) {
  Color c = to_move_;
  Color o = opponent(c);
  push_history(p);
  ++move_num_;
  to_move_ = o;
  if (p == PASS) {
    ++passes_;
    ko_ = NO_POINT;
    return;
  }
  passes_ = 0;
  ko_ = NO_POINT;

  color_[p] = c;
  hash_ ^= g_zobrist.table[p][c];
  head_[p] = next_[p] = static_cast<int16_t>(p);
  csize_[p] = 1;
  plib_[p] = 0;
  for (int d : dirs_) {
    int r = p + d;
    if (color_[r] == EMPTY)
      plib_[p]++;
    else if (is_stone(color_[r]))
      plib_[head_[r]]--;
  }
  for (int d : dirs_) {
    int r = p + d;
    if (color_[r] == c && head_[r] != head_[p]) merge(head_[p], head_[r]);
  }
  int captured = 0, cap_point = NO_POINT;
  for (int d : dirs_) {
    int r = p + d;
    if (color_[r] == o && plib_[head_[r]] == 0) {
      captured += csize_[head_[r]];
      cap_point = r;
      remove_chain(r);
    }
  }
  captures_[c == BLACK ? 0 : 1] += captured;
  if (captured == 1 && csize_[head_[p]] == 1 && count_libs(p, 2) == 1) ko_ = cap_point;
}

void Board::play_action(int a) {
  int p = to_point(a);
  if (!is_legal(p, to_move_) || repeats_position(p)) throw std::invalid_argument("illegal move");
  play(p);
  if (p != PASS) record_position();
}

void Board::place_stone_action(int a, Color c) {
  Color saved = to_move_;
  int saved_moves = move_num_;
  auto saved_history = history_;
  to_move_ = c;
  int p = to_point(a);
  if (p == PASS || !is_legal(p, c)) throw std::invalid_argument("cannot place stone");
  play(p);
  to_move_ = saved;
  move_num_ = saved_moves;
  history_ = saved_history;
  record_position();
}

bool Board::is_eye(int p, Color c) const {
  if (color_[p] != EMPTY) return false;
  for (int d : dirs_)
    if (color_[p + d] != c && color_[p + d] != BORDER) return false;
  int bad = 0, border = 0;
  for (int d : {-w_ - 1, -w_ + 1, w_ - 1, w_ + 1}) {
    Color cd = color_[p + d];
    if (cd == BORDER)
      border = 1;
    else if (cd == opponent(c))
      ++bad;
  }
  return bad + border < 2;
}

void Board::stones(int8_t* out) const {
  for (int a = 0; a < n_ * n_; ++a) out[a] = static_cast<int8_t>(color_[to_point(a)]);
}

void Board::legal_mask(uint8_t* out) const {
  for (int a = 0; a < n_ * n_; ++a) out[a] = is_legal_action(a);
  out[n_ * n_] = 1;
}

void Board::sensible_mask(uint8_t* out) const {
  for (int a = 0; a < n_ * n_; ++a) {
    int p = to_point(a);
    out[a] = is_legal(p, to_move_) && !is_eye(p, to_move_) && !repeats_position(p);
  }
  out[n_ * n_] = 1;
}

void Board::region_owner(int8_t* out) const {
  // Flood-fill each empty region and record which colours border it.
  uint32_t s = g_marks.next();
  auto& m = g_marks.mark;
  std::array<int16_t, MAX_P> stack, region;
  for (int a = 0; a < n_ * n_; ++a) {
    int p = to_point(a);
    if (color_[p] != EMPTY) {
      out[a] = 0;
      continue;
    }
    if (m[p] == s) continue;
    int top = 0, size = 0;
    bool sees_black = false, sees_white = false;
    m[p] = s;
    stack[top++] = static_cast<int16_t>(p);
    while (top) {
      int q = stack[--top];
      region[size++] = static_cast<int16_t>(q);
      for (int d : dirs_) {
        int r = q + d;
        if (color_[r] == EMPTY && m[r] != s) {
          m[r] = s;
          stack[top++] = static_cast<int16_t>(r);
        } else if (color_[r] == BLACK) {
          sees_black = true;
        } else if (color_[r] == WHITE) {
          sees_white = true;
        }
      }
    }
    int8_t owner = sees_black && !sees_white ? 1 : sees_white && !sees_black ? 2 : 3;
    for (int i = 0; i < size; ++i) out[to_action(region[i])] = owner;
  }
}

float Board::tromp_taylor_score() const {
  std::array<int8_t, MAX_N * MAX_N> owner;
  region_owner(owner.data());
  int score = 0;  // black - white
  for (int a = 0; a < n_ * n_; ++a) {
    Color c = color_[to_point(a)];
    if (c == BLACK || owner[a] == 1) ++score;
    if (c == WHITE || owner[a] == 2) --score;
  }
  return static_cast<float>(score) - komi_;
}

std::string Board::to_string() const {
  static const char* cols = "ABCDEFGHJKLMNOPQRST";
  std::string s;
  for (int y = n_ - 1; y >= 0; --y) {
    char row[4];
    std::snprintf(row, sizeof row, "%2d", y + 1);
    s += row;
    for (int x = 0; x < n_; ++x) {
      int p = (y + 1) * w_ + x + 1;
      s += ' ';
      s += color_[p] == BLACK ? 'X' : color_[p] == WHITE ? 'O' : (p == ko_ ? '*' : '.');
    }
    s += '\n';
  }
  s += "  ";
  for (int x = 0; x < n_; ++x) {
    s += ' ';
    s += cols[x];
  }
  s += '\n';
  return s;
}

}  // namespace go
