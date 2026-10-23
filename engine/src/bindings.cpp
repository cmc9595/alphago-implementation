#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

#include "board.hpp"

namespace py = pybind11;
using go::Board;

namespace {

py::array_t<uint8_t> features(const Board& b) {
  const int n = b.size();
  py::array_t<uint8_t> out({go::plane::COUNT, n, n});
  b.features(out.mutable_data());
  return out;
}

// Feature extraction for many boards, split across threads with the GIL released.
py::array_t<uint8_t> features_batch(const std::vector<const Board*>& boards, int threads) {
  if (boards.empty()) throw std::invalid_argument("empty batch");
  const int n = boards[0]->size();
  for (auto* b : boards)
    if (b->size() != n) throw std::invalid_argument("mixed board sizes");
  const size_t stride = static_cast<size_t>(go::plane::COUNT) * n * n;
  py::array_t<uint8_t> out({static_cast<int>(boards.size()), go::plane::COUNT, n, n});
  uint8_t* data = out.mutable_data();
  if (threads <= 0) threads = static_cast<int>(std::thread::hardware_concurrency());
  threads = std::max(1, std::min<int>(threads, static_cast<int>(boards.size())));
  {
    py::gil_scoped_release release;
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
      pool.emplace_back([&, t] {
        for (size_t i = t; i < boards.size(); i += threads) boards[i]->features(data + i * stride);
      });
    for (auto& th : pool) th.join();
  }
  return out;
}

// Training-data sampler: sample i is the position after replaying moves[starts[i] .. starts[i]+plies[i])
// on an empty board (Black first). Moves must already be validated as legal.
py::array_t<uint8_t> replay_features(py::array_t<int16_t, py::array::c_style> moves,
                                     py::array_t<int64_t, py::array::c_style> starts,
                                     py::array_t<int32_t, py::array::c_style> plies, int size, int threads) {
  const size_t count = static_cast<size_t>(starts.size());
  if (static_cast<size_t>(plies.size()) != count) throw std::invalid_argument("starts/plies length mismatch");
  const int16_t* mv = moves.data();
  const int64_t* st = starts.data();
  const int32_t* pl = plies.data();
  const int64_t total = moves.size();
  for (size_t i = 0; i < count; ++i)
    if (st[i] < 0 || pl[i] < 0 || st[i] + pl[i] > total) throw std::out_of_range("sample outside move array");
  const size_t stride = static_cast<size_t>(go::plane::COUNT) * size * size;
  py::array_t<uint8_t> out({static_cast<int>(count), go::plane::COUNT, size, size});
  uint8_t* data = out.mutable_data();
  if (threads <= 0) threads = static_cast<int>(std::thread::hardware_concurrency());
  threads = std::max(1, std::min<int>(threads, static_cast<int>(count)));
  std::atomic<bool> failed{false};
  {
    py::gil_scoped_release release;
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
      pool.emplace_back([&, t] {
        for (size_t i = t; i < count; i += threads) {
          Board b(size);
          for (int k = 0; k < pl[i]; ++k) {
            int p = b.to_point(mv[st[i] + k]);
            if (!b.is_legal(p, b.to_move())) {
              failed = true;
              return;
            }
            b.play(p);
          }
          b.features(data + i * stride);
        }
      });
    for (auto& th : pool) th.join();
  }
  if (failed) throw std::invalid_argument("illegal move during replay");
  return out;
}

}  // namespace

PYBIND11_MODULE(_go, m) {
  m.doc() = "Go board engine with AlphaGo-style features";

  py::enum_<go::Color>(m, "Color")
      .value("EMPTY", go::EMPTY)
      .value("BLACK", go::BLACK)
      .value("WHITE", go::WHITE);

  m.attr("NUM_FEATURES") = go::plane::COUNT;

  py::class_<Board>(m, "Board")
      .def(py::init<int, float, bool>(), py::arg("size") = 19, py::arg("komi") = 7.5f,
           py::arg("superko") = false)
      .def_property_readonly("superko", &Board::superko)
      .def("copy", [](const Board& b) { return Board(b); })
      .def("__copy__", [](const Board& b) { return Board(b); })
      .def("__deepcopy__", [](const Board& b, py::dict) { return Board(b); })
      .def_property_readonly("size", &Board::size)
      .def_property_readonly("komi", &Board::komi)
      .def_property_readonly("to_move", &Board::to_move)
      .def_property_readonly("move_number", &Board::move_number)
      .def_property_readonly("consecutive_passes", &Board::consecutive_passes)
      .def_property_readonly("game_over", &Board::game_over)
      .def_property_readonly("hash", &Board::hash)
      .def_property_readonly("ko_action", &Board::ko_action)
      .def_property_readonly("last_action", &Board::last_action)
      .def_property_readonly("pass_action", [](const Board& b) { return b.size() * b.size(); })
      .def("captures", &Board::captures)
      .def("set_to_move", &Board::set_to_move)
      .def("is_legal", &Board::is_legal_action)
      .def("play", &Board::play_action)
      .def("place_stone", &Board::place_stone_action)
      .def("color_at", &Board::color_at_action)
      .def("stones",
           [](const Board& b) {
             py::array_t<int8_t> out({b.size(), b.size()});
             b.stones(out.mutable_data());
             return out;
           })
      .def("region_owner",
           [](const Board& b) {
             py::array_t<int8_t> out(b.size() * b.size());
             b.region_owner(out.mutable_data());
             return out;
           })
      .def("legal_mask",
           [](const Board& b) {
             py::array_t<uint8_t> out(b.size() * b.size() + 1);
             b.legal_mask(out.mutable_data());
             return out;
           })
      .def("sensible_mask",
           [](const Board& b) {
             py::array_t<uint8_t> out(b.size() * b.size() + 1);
             b.sensible_mask(out.mutable_data());
             return out;
           })
      .def("features", &features)
      .def("score", &Board::tromp_taylor_score)
      .def("__str__", &Board::to_string);

  m.def("features_batch", &features_batch, py::arg("boards"), py::arg("threads") = 0);
  m.def("replay_features", &replay_features, py::arg("moves"), py::arg("starts"), py::arg("plies"),
        py::arg("size") = 19, py::arg("threads") = 0);
}
