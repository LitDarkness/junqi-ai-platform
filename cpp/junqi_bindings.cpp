#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include <stdexcept>
#include "junqi_types.h"
#include "junqi_environment.h"
#include "junqi_batch_environment.h"

namespace py = pybind11;

PYBIND11_MODULE(JunqiEnv, m) {
    m.doc() = "Chinese Junqi (Dark Chess) Environment for Reinforcement Learning";
    
    // ==================== 命名空间 ====================
    py::module junqi = m.def_submodule("junqi", "Core junqi types and enums");
    
    // ==================== 枚举类型 ====================
    // PieceType
    py::enum_<junqi::PieceType>(junqi, "PieceType")
        .value("EMPTY", junqi::EMPTY)
        .value("MINE", junqi::MINE)
        .value("BOMB", junqi::BOMB)
        .value("FLAG", junqi::FLAG)
        .value("SAPPER", junqi::SAPPER)
        .value("LIEUTENANT", junqi::LIEUTENANT)
        .value("CAPTAIN", junqi::CAPTAIN)
        .value("MAJOR", junqi::MAJOR)
        .value("COLONEL", junqi::COLONEL)
        .value("BRIGADIER", junqi::BRIGADIER)
        .value("GENERAL", junqi::GENERAL)
        .value("ARMY_CMDR", junqi::ARMY_CMDR)
        .value("FIELD_MARSHAL", junqi::FIELD_MARSHAL)
        .value("UNKNOWN", junqi::UNKNOWN)
        .export_values();
    
    // PlayerId
    py::enum_<junqi::PlayerId>(junqi, "PlayerId")
        .value("NONE", junqi::NONE)
        .value("RED", junqi::RED)
        .value("BLACK", junqi::BLACK)
        .export_values();
    
    // CombatResult
    py::enum_<junqi::CombatResult>(junqi, "CombatResult")
        .value("MOVE_ONLY", junqi::MOVE_ONLY)
        .value("WIN", junqi::WIN)
        .value("LOSS", junqi::LOSS)
        .value("DRAW", junqi::DRAW)
        .value("GAME_OVER_WIN", junqi::GAME_OVER_WIN)
        .value("GAME_OVER_LOSS", junqi::GAME_OVER_LOSS)
        .export_values();
    
    // 常量
    junqi.attr("BOARD_ROWS") = junqi::BOARD_ROWS;
    junqi.attr("BOARD_COLS") = junqi::BOARD_COLS;
    junqi.attr("BOARD_SIZE") = junqi::BOARD_SIZE;
    
    // ==================== 结构体绑定 ====================
    py::class_<junqi::PieceInfo>(m, "PieceInfo")
        .def_readwrite("type", &junqi::PieceInfo::type, "Piece type")
        .def_readwrite("owner", &junqi::PieceInfo::owner, "Piece owner (1=red, -1=black, 0=empty)")
        .def("__repr__", [](const junqi::PieceInfo& self) {
            return std::string("PieceInfo(type=") + std::to_string(self.type) + 
                   ", owner=" + std::to_string(self.owner) + ")";
        });
    
    py::class_<junqi::MoveRecord>(m, "MoveRecord")
        .def_readwrite("from_pos", &junqi::MoveRecord::from_pos, "From position")
        .def_readwrite("to_pos", &junqi::MoveRecord::to_pos, "To position")
        .def_readwrite("player_id", &junqi::MoveRecord::player_id, "Player ID")
        .def_readwrite("result", &junqi::MoveRecord::result, "Combat result")
        .def("__repr__", [](const junqi::MoveRecord& self) {
            return std::string("MoveRecord(from=") + std::to_string(self.from_pos) + 
                   ", to=" + std::to_string(self.to_pos) + 
                   ", player=" + std::to_string(self.player_id) + ")";
        });

    py::class_<junqi::LegalityResult>(m, "LegalityResult")
        .def_readonly("legal", &junqi::LegalityResult::legal)
        .def_readonly("code", &junqi::LegalityResult::code)
        .def_readonly("message_key", &junqi::LegalityResult::message_key)
        .def_readonly("blocking_square", &junqi::LegalityResult::blocking_square)
        .def_readonly("path", &junqi::LegalityResult::path);
    
    // ==================== JunqiEnvironment 类 ====================
    py::class_<junqi::JunqiEnvironment>(m, "JunqiEnvironment")
        .def(py::init<>())
        
        // 初始化
        .def("reset", py::overload_cast<const std::vector<int>&, const std::vector<int>&>(
            &junqi::JunqiEnvironment::reset),
            "Reset game with custom setups",
            py::arg("red_setup"), py::arg("black_setup"))
        .def("reset_random", &junqi::JunqiEnvironment::reset_random,
            "Reset game with random setups")
        
        // 核心交互
        .def("step", &junqi::JunqiEnvironment::step,
            "Execute a move and return combat result",
            py::arg("from_pos"), py::arg("to_pos"))
        
        // 状态获取
        .def("get_perfect_state", &junqi::JunqiEnvironment::get_perfect_state,
            "Get perfect observation (all pieces visible)")
        .def("get_observation", &junqi::JunqiEnvironment::get_observation,
            "Get observation from player's perspective",
            py::arg("player_id"))
        .def("get_legal_action_mask", &junqi::JunqiEnvironment::get_legal_action_mask,
            "Get legal action mask (3600-dim bool array)",
            py::arg("player_id"))
        .def("get_legal_actions", &junqi::JunqiEnvironment::get_legal_actions,
            "Get legal action indices",
            py::arg("player_id"))
        .def("check_move", &junqi::JunqiEnvironment::check_move,
            "Check a move and return a stable structured reason",
            py::arg("from_pos"), py::arg("to_pos"))
        .def("get_feature_planes", &junqi::JunqiEnvironment::get_feature_planes,
            "Get flattened 14x12x5 feature planes",
            py::arg("player_id"))
        .def("get_material_score", &junqi::JunqiEnvironment::get_material_score,
            "Get on-board material score",
            py::arg("player_id"))
        
        // 辅助信息
        .def("is_field_marshal_dead", &junqi::JunqiEnvironment::is_field_marshal_dead,
            "Check if field marshal is dead",
            py::arg("player_id"))
        .def("get_turn_count", &junqi::JunqiEnvironment::get_turn_count,
            "Get current turn count")
        .def("get_current_player", &junqi::JunqiEnvironment::get_current_player,
            "Get current player")
        .def("is_game_over", &junqi::JunqiEnvironment::is_game_over,
            "Check if game is over")
        .def("get_winner", &junqi::JunqiEnvironment::get_winner,
            "Get winner (0 = none, 1 = red, -1 = black)")
        .def("get_terminal_reason", &junqi::JunqiEnvironment::get_terminal_reason,
            py::return_value_policy::reference_internal, "Get stable terminal reason code")
        .def("get_state_key", &junqi::JunqiEnvironment::get_state_key,
            "Get deterministic 64-bit state key")
        
        // 布阵相关 - 使用lambda包装以处理异常
        .def("set_red_setup", [](junqi::JunqiEnvironment& self, const std::vector<int>& setup) {
            std::string error_message;
            if (!self.set_red_setup(setup, error_message)) {
                std::string msg("Invalid red setup: ");
                msg += error_message;
                throw std::runtime_error(msg);
            }
        }, "Set red player's custom setup", py::arg("setup"))
        .def("set_black_setup", [](junqi::JunqiEnvironment& self, const std::vector<int>& setup) {
            std::string error_message;
            if (!self.set_black_setup(setup, error_message)) {
                std::string msg("Invalid black setup: ");
                msg += error_message;
                throw std::runtime_error(msg);
            }
        }, "Set black player's custom setup", py::arg("setup"))
        .def("get_red_setup", &junqi::JunqiEnvironment::get_red_setup,
            "Get red player's current setup")
        .def("get_black_setup", &junqi::JunqiEnvironment::get_black_setup,
            "Get black player's current setup")
        .def("generate_red_setup", &junqi::JunqiEnvironment::generate_red_setup,
            "Generate a random setup for red player")
        .def("generate_black_setup", &junqi::JunqiEnvironment::generate_black_setup,
            "Generate a random setup for black player")
        
        // ==================== 便利工具函数 ====================
        
        // 位置转换（静态方法）
        .def_static("pos_to_index", &junqi::JunqiEnvironment::pos_to_index,
            "Convert (row, col) to index", py::arg("row"), py::arg("col"))
        .def_static("index_to_pos", &junqi::JunqiEnvironment::index_to_pos,
            "Convert index to (row, col)", py::arg("index"))
        .def_static("get_action_index", &junqi::JunqiEnvironment::get_action_index,
            "Get action index from (from, to)", py::arg("from"), py::arg("to"))
        .def_static("decode_action_index", &junqi::JunqiEnvironment::decode_action_index,
            "Decode action index to (from, to)", py::arg("action_index"))
        
        // 棋子信息查询
        .def("get_piece_at", &junqi::JunqiEnvironment::get_piece_at,
            "Get piece info at position", py::arg("pos"))
        .def("get_player_pieces", &junqi::JunqiEnvironment::get_player_pieces,
            "Get all piece positions for a player", py::arg("player_id"))
        .def("get_piece_count", &junqi::JunqiEnvironment::get_piece_count,
            "Get count of specific piece type for a player", 
            py::arg("player_id"), py::arg("piece_type"))
        
        // 移动验证
        .def("can_move", &junqi::JunqiEnvironment::can_move,
            "Check if a move is valid", py::arg("from_pos"), py::arg("to_pos"))
        .def("get_possible_moves", &junqi::JunqiEnvironment::get_possible_moves,
            "Get all possible moves from a position", py::arg("from_pos"))
        .def("get_movable_pieces", &junqi::JunqiEnvironment::get_movable_pieces,
            "Get all movable piece positions for a player", py::arg("player_id"))
        
        // 棋盘渲染
        .def("render", &junqi::JunqiEnvironment::render,
            "Get board as formatted string")
        .def("print_board", &junqi::JunqiEnvironment::print_board,
            "Print board to console")
        
        // 游戏历史
        .def("get_last_move", &junqi::JunqiEnvironment::get_last_move,
            "Get last move record")
        .def("get_game_history", &junqi::JunqiEnvironment::get_game_history,
            "Get complete game history")
        .def("clear_history", &junqi::JunqiEnvironment::clear_history,
            "Clear move history")
        
        // 从给定局面加载
        .def("load_board_state", &junqi::JunqiEnvironment::load_board_state,
            "Load board state from given state array",
            py::arg("board_state"), py::arg("current_player"));

    py::class_<junqi::BatchJunqiEnvironment>(m, "BatchJunqiEnvironment")
        .def(py::init<int>(), py::arg("num_envs"))
        .def("size", &junqi::BatchJunqiEnvironment::size)
        .def("reset_random_all", &junqi::BatchJunqiEnvironment::reset_random_all)
        .def("reset_indices", &junqi::BatchJunqiEnvironment::reset_indices, py::arg("indices"))
        .def("get_current_players", &junqi::BatchJunqiEnvironment::get_current_players)
        .def("get_turn_counts", &junqi::BatchJunqiEnvironment::get_turn_counts)
        .def("get_winners", &junqi::BatchJunqiEnvironment::get_winners)
        .def("get_done_flags", [](const junqi::BatchJunqiEnvironment& self) {
            std::vector<std::uint8_t> flags = self.get_done_flags();
            py::array_t<std::uint8_t> out(flags.size());
            std::memcpy(out.mutable_data(), flags.data(), flags.size() * sizeof(std::uint8_t));
            return out;
        })
        .def("get_perfect_states_batch", [](const junqi::BatchJunqiEnvironment& self) {
            py::array_t<int> out(std::vector<py::ssize_t>{
                static_cast<py::ssize_t>(self.size()), junqi::BOARD_SIZE
            });
            self.fill_perfect_states_batch(out.mutable_data());
            return out;
        })
        .def("get_feature_planes_batch", [](junqi::BatchJunqiEnvironment& self, const std::vector<int>& player_ids) {
            py::array_t<float> out(std::vector<py::ssize_t>{
                static_cast<py::ssize_t>(self.size()), 14, 12, 5
            });
            self.fill_feature_planes_batch(out.mutable_data(), player_ids);
            return out;
        }, py::arg("player_ids"))
        .def("get_legal_action_masks_batch", [](junqi::BatchJunqiEnvironment& self, const std::vector<int>& player_ids) {
            py::array_t<std::uint8_t> out(std::vector<py::ssize_t>{
                static_cast<py::ssize_t>(self.size()), 3600
            });
            self.fill_legal_action_masks_batch(out.mutable_data(), player_ids);
            return out;
        }, py::arg("player_ids"))
        .def("get_material_scores_batch", [](const junqi::BatchJunqiEnvironment& self, const std::vector<int>& player_ids) {
            std::vector<float> scores = self.get_material_scores_batch(player_ids);
            py::array_t<float> out(scores.size());
            std::memcpy(out.mutable_data(), scores.data(), scores.size() * sizeof(float));
            return out;
        }, py::arg("player_ids"))
        .def("step_actions", &junqi::BatchJunqiEnvironment::step_actions, py::arg("action_indices"))
        .def("step_indexed_actions", &junqi::BatchJunqiEnvironment::step_indexed_actions,
            py::arg("indices"), py::arg("action_indices"));
}
