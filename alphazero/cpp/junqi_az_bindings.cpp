#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "az_state.h"
#include "heuristic_084_evaluator.h"
#include "neural_puct.h"
#include <cstring>

namespace py = pybind11;
using junqi::az::AZState;

namespace {

py::dict leaf_dict(const junqi::az::PUCTLeafRequest& request) {
    py::dict result;
    result["terminal"] = request.terminal;
    result["terminal_value"] = request.terminal_value;
    result["state_key"] = request.state_key;
    result["position_key"] = request.position_key;
    result["player"] = request.player;
    result["legal_actions"] = request.legal_actions;
    result["repetition_contexts"] = request.repetition_contexts;
    if (!request.terminal) {
        py::array_t<float> features(std::vector<py::ssize_t>{AZState::kFeatureChannels, 12, 5});
        std::memcpy(features.mutable_data(), request.features.data(), request.features.size() * sizeof(float));
        result["features"] = std::move(features);
    } else result["features"] = py::none();
    return result;
}

py::array_t<float> feature_array(const AZState& state) {
    const std::vector<float> values = state.features();
    py::array_t<float> result(
        std::vector<py::ssize_t>{AZState::kFeatureChannels, 12, 5});
    std::memcpy(result.mutable_data(), values.data(), values.size() * sizeof(float));
    return result;
}

} // namespace

PYBIND11_MODULE(JunqiAZ, m) {
    m.doc() = "Deterministic, snapshot-safe Chinese Junqi backend for AlphaZero";
    m.attr("BOARD_ROWS") = junqi::BOARD_ROWS;
    m.attr("BOARD_COLS") = junqi::BOARD_COLS;
    m.attr("BOARD_SIZE") = junqi::BOARD_SIZE;
    m.attr("ACTION_SPACE_SIZE") = junqi::JunqiEnvironment::kActionSpaceSize;
    m.attr("FEATURE_CHANNELS") = AZState::kFeatureChannels;
    m.attr("LEGACY_FEATURE_CHANNELS") = junqi::JunqiEnvironment::kFeatureChannels;
    m.attr("HEURISTIC_084_SOURCE_SHA256") =
        "E87CC6ABBF9F61B073697E067659E12F221A7AF3DEE93C8D646A0E7293A86C95";

    m.def("heuristic_084_raw_batch", [](py::array_t<int,
            py::array::c_style | py::array::forcecast> boards, int perspective) {
        const auto input = boards.request();
        if (input.ndim != 2 || input.shape[1] != junqi::BOARD_SIZE) {
            throw std::invalid_argument("boards must have shape [batch, 60]");
        }
        if (perspective != 1 && perspective != -1) {
            throw std::invalid_argument("perspective must be 1 or -1");
        }
        py::array_t<double> values(input.shape[0]);
        auto output = values.mutable_unchecked<1>();
        const int* data = static_cast<const int*>(input.ptr);
        {
            py::gil_scoped_release release;
            for (py::ssize_t row = 0; row < input.shape[0]; ++row) {
                output(row) = junqi::az::heuristic_084_raw_value(
                    data + row * junqi::BOARD_SIZE, perspective);
            }
        }
        return values;
    }, py::arg("boards"), py::arg("perspective") = 1);

    py::enum_<junqi::CombatResult>(m, "CombatResult", py::module_local())
        .value("MOVE_ONLY", junqi::MOVE_ONLY)
        .value("WIN", junqi::WIN)
        .value("LOSS", junqi::LOSS)
        .value("DRAW", junqi::DRAW)
        .value("GAME_OVER_WIN", junqi::GAME_OVER_WIN)
        .value("GAME_OVER_LOSS", junqi::GAME_OVER_LOSS);

    py::class_<AZState>(m, "JunqiAZState")
        .def(py::init<>())
        .def("reset", &AZState::reset, py::arg("red_setup"), py::arg("black_setup"))
        .def("reset_random", &AZState::reset_random, py::arg("seed"))
        .def("load_board_state", &AZState::load_board_state,
             py::arg("board_state"), py::arg("current_player"))
        .def("clone", &AZState::clone)
        .def("current_player", &AZState::current_player)
        .def("opponent", &AZState::opponent, py::arg("player") = 0)
        .def("feature_planes", &feature_array)
        .def("features", &feature_array)
        .def("legal_actions", &AZState::legal_actions)
        .def("repetition_contexts", py::overload_cast<>(
                 &AZState::repetition_contexts, py::const_))
        .def("apply", [](AZState& self, int action) {
            return static_cast<int>(self.step_action(action, false));
        }, py::arg("action"))
        .def("apply_with_undo", [](AZState& self, int action) {
            return static_cast<int>(self.step_action(action, true));
        }, py::arg("action"))
        .def("undo", &AZState::undo)
        .def("clear_undo", &AZState::clear_undo)
        .def("is_terminal", &AZState::terminal)
        .def("terminal", &AZState::terminal)
        .def("terminal_value", [](const AZState& self, int perspective) {
            return self.terminal_value(perspective == 0 ? self.current_player() : perspective);
        }, py::arg("perspective") = 0)
        .def("terminal_value_for", &AZState::terminal_value, py::arg("perspective"))
        .def("winner", &AZState::winner)
        .def("terminal_reason", &AZState::terminal_reason)
        .def("turn_count", &AZState::turn_count)
        .def("key", &AZState::key)
        .def("position_key", &AZState::position_key)
        .def("board", &AZState::board)
        .def("no_capture_ply", &AZState::no_capture_ply)
        .def("repetition_count", &AZState::repetition_count)
        .def("history_size", &AZState::history_size)
        .def("undo_depth", &AZState::undo_depth)
        .def_static("flip_position", &AZState::flip_position, py::arg("position"))
        .def_static("flip_action", &AZState::flip_action, py::arg("action"));
    // Compatibility alias used by early Python adapters.
    m.attr("AZState") = m.attr("JunqiAZState");

    py::class_<junqi::az::NeuralPUCTTree>(m, "NeuralPUCTTree")
        .def(py::init<const AZState&, float, bool>(),
             py::arg("root"), py::arg("cpuct") = 1.5f, py::arg("root_tactical_guard") = false)
        .def("root_tactical_stats", &junqi::az::NeuralPUCTTree::root_tactical_stats)
        .def("select_leaf", [](junqi::az::NeuralPUCTTree& self) {
            const auto request = self.select_leaf();
            py::dict result;
            result["terminal"] = request.terminal;
            result["terminal_value"] = request.terminal_value;
            result["state_key"] = request.state_key;
            result["position_key"] = request.position_key;
            result["player"] = request.player;
            result["legal_actions"] = request.legal_actions;
            result["repetition_contexts"] = request.repetition_contexts;
            if (!request.terminal) {
                py::array_t<float> features(
                    std::vector<py::ssize_t>{
                        AZState::kFeatureChannels, 12, 5});
                std::memcpy(
                    features.mutable_data(), request.features.data(),
                    request.features.size() * sizeof(float));
                result["features"] = std::move(features);
            } else {
                result["features"] = py::none();
            }
            return result;
        })
        .def("backup", &junqi::az::NeuralPUCTTree::backup,
             py::arg("legal_priors"), py::arg("leaf_value"))
        .def("select_leaves", [](junqi::az::NeuralPUCTTree& self, int count) {
            py::list result;
            for (const auto& request : self.select_leaves(count)) result.append(leaf_dict(request));
            return result;
        }, py::arg("count"))
        .def("backup_leaves", &junqi::az::NeuralPUCTTree::backup_leaves,
             py::arg("priors"), py::arg("values"))
        .def("cancel_pending", &junqi::az::NeuralPUCTTree::cancel_pending)
        .def("advance_root", &junqi::az::NeuralPUCTTree::advance_root)
        .def("root_key", &junqi::az::NeuralPUCTTree::root_key)
        .def("add_root_dirichlet",
             &junqi::az::NeuralPUCTTree::add_root_dirichlet,
             py::arg("alpha"), py::arg("epsilon"), py::arg("seed"))
        .def("root_edges", [](const junqi::az::NeuralPUCTTree& self) {
            py::list result;
            for (const auto& edge : self.root_edges()) {
                py::dict row;
                row["action"] = edge.action;
                row["prior"] = edge.prior;
                row["visits"] = edge.visits;
                row["q"] = edge.q;
                result.append(std::move(row));
            }
            return result;
        })
        .def("root_visits", &junqi::az::NeuralPUCTTree::root_visits)
        .def("has_pending_leaf",
             &junqi::az::NeuralPUCTTree::has_pending_leaf);
}
