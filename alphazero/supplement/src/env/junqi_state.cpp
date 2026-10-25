#include "junqi_state.h"
#include "move_generator.h"
#include <stdexcept>

namespace junqi {

JunqiState::JunqiState()
    : current_player_(RED), turn_count_(0), game_over_(false), winner_(NONE) {
    pieces_.resize(BOARD_SIZE);
    
    // 初始化所有位置为空
    for (int i = 0; i < BOARD_SIZE; ++i) {
        pieces_[i] = Piece(EMPTY, NONE);
    }
    
    // 初始化司令和军旗状态
    field_marshal_dead_[0] = false; // RED
    field_marshal_dead_[1] = false; // BLACK
    flag_revealed_[0] = false;      // RED
    flag_revealed_[1] = false;      // BLACK
}

void JunqiState::reset(const std::vector<int>& red_setup, const std::vector<int>& black_setup) {
    // 清空棋盘
    for (int i = 0; i < BOARD_SIZE; ++i) {
        pieces_[i] = Piece(EMPTY, NONE);
    }
    
    // 设置红方棋子
    for (size_t i = 0; i < red_setup.size(); ++i) {
        int setup_row = i / 5;    // 布阵内的行 (0-5)
        int setup_col = i % 5;     // 布阵内的列 (0-4)
        int board_row = RED_AREA_START + setup_row;  // 棋盘上的实际行 (0-5)
        int pos = pos_to_index(board_row, setup_col);
        
        PieceType type = static_cast<PieceType>(red_setup[i]);
        if (type != EMPTY) {
            pieces_[pos] = Piece(type, RED);
        }
    }
    
    // 设置黑方棋子
    for (size_t i = 0; i < black_setup.size(); ++i) {
        int setup_row = i / 5;    // 布阵内的行 (0-5)
        int setup_col = i % 5;     // 布阵内的列 (0-4)
        int board_row = BLACK_AREA_START + setup_row;  // 棋盘上的实际行 (6-11)
        int pos = pos_to_index(board_row, setup_col);
        
        PieceType type = static_cast<PieceType>(black_setup[i]);
        if (type != EMPTY) {
            pieces_[pos] = Piece(type, BLACK);
        }
    }
    
    // 重置游戏状态
    current_player_ = RED;
    turn_count_ = 0;
    game_over_ = false;
    winner_ = NONE;
    
    // 重置司令和军旗状态
    field_marshal_dead_[0] = false;
    field_marshal_dead_[1] = false;
    flag_revealed_[0] = false;
    flag_revealed_[1] = false;
}

const Piece& JunqiState::get_piece(int index) const {
    if (index < 0 || index >= BOARD_SIZE) {
        static Piece empty_piece(EMPTY, NONE);
        return empty_piece;
    }
    return pieces_[index];
}

const Piece& JunqiState::get_piece(int row, int col) const {
    int index = pos_to_index(row, col);
    return get_piece(index);
}

void JunqiState::set_piece(int index, const Piece& piece) {
    if (index >= 0 && index < BOARD_SIZE) {
        pieces_[index] = piece;
    }
}

void JunqiState::set_piece(int row, int col, const Piece& piece) {
    int index = pos_to_index(row, col);
    set_piece(index, piece);
}

void JunqiState::remove_piece(int index) {
    if (index >= 0 && index < BOARD_SIZE) {
        pieces_[index] = Piece(EMPTY, NONE);
    }
}

void JunqiState::remove_piece(int row, int col) {
    int index = pos_to_index(row, col);
    remove_piece(index);
}

void JunqiState::move_piece(int from, int to) {
    if (from >= 0 && from < BOARD_SIZE && to >= 0 && to < BOARD_SIZE) {
        pieces_[to] = pieces_[from];
        pieces_[from] = Piece(EMPTY, NONE);
    }
}

void JunqiState::switch_turn() {
    current_player_ = (current_player_ == RED) ? BLACK : RED;
}

bool JunqiState::is_field_marshal_dead(PlayerId player) const {
    int idx = player_to_index(player);
    if (idx < 0) return false;
    return field_marshal_dead_[idx];
}

void JunqiState::set_field_marshal_dead(PlayerId player) {
    int idx = player_to_index(player);
    if (idx >= 0) {
        field_marshal_dead_[idx] = true;
        // 司令死亡后，揭示军旗
        flag_revealed_[idx] = true;
    }
}

void JunqiState::set_flag_revealed(PlayerId player) {
    int idx = player_to_index(player);
    if (idx >= 0) {
        flag_revealed_[idx] = true;
    }
}

bool JunqiState::has_movable_pieces(PlayerId player) const {
    MoveGenerator move_generator(board_);

    for (int i = 0; i < BOARD_SIZE; ++i) {
        const Piece& piece = pieces_[i];
        
        // 跳过空位和敌方棋子
        if (piece.is_empty() || !piece.is_owned_by(player)) {
            continue;
        }
        
        // 跳过不可移动的棋子（地雷、军旗）
        if (!is_movable_piece(piece.type)) {
            continue;
        }
        
        // 检查大本营内的棋子（不能移出）
        if (board_.is_hq(i)) {
            continue;
        }
        
        if (!move_generator.get_possible_moves(*this, i).empty()) {
            return true;
        }
    }
    
    return false;
}

std::vector<int> JunqiState::get_board_state() const {
    std::vector<int> state(BOARD_SIZE);
    
    for (int i = 0; i < BOARD_SIZE; ++i) {
        const Piece& piece = pieces_[i];
        if (piece.is_empty()) {
            state[i] = EMPTY;
        } else if (piece.owner == RED) {
            state[i] = static_cast<int>(piece.type);
        } else {
            // 黑方棋子用负数表示（用于区分）
            state[i] = -static_cast<int>(piece.type);
        }
    }
    
    return state;
}

JunqiState JunqiState::clone() const {
    JunqiState new_state;
    new_state.pieces_ = pieces_;
    new_state.current_player_ = current_player_;
    new_state.turn_count_ = turn_count_;
    new_state.field_marshal_dead_[0] = field_marshal_dead_[0];
    new_state.field_marshal_dead_[1] = field_marshal_dead_[1];
    new_state.flag_revealed_[0] = flag_revealed_[0];
    new_state.flag_revealed_[1] = flag_revealed_[1];
    new_state.game_over_ = game_over_;
    new_state.winner_ = winner_;
    
    return new_state;
}

} // namespace junqi
