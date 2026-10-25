#include "setup_validator.h"
#include <sstream>

namespace junqi {

SetupValidator::SetupValidator(const JunqiBoard& board) : board_(board) {}

bool SetupValidator::validate_setup(const std::vector<int>& setup, PlayerId player,
                                     std::string& error_message) const {
    // 检查布阵大小（6行×5列=30个位置，5个行营为空）
    if (setup.size() != 30) {
        error_message = "Setup must have exactly 30 positions (25 pieces + 5 camps)";
        return false;
    }
    
    // 检查棋子数量
    if (!check_piece_count(setup, error_message)) {
        return false;
    }
    
    // 检查位置限制
    if (!check_position_restrictions(setup, player, error_message)) {
        return false;
    }
    
    return true;
}

bool SetupValidator::validate_both_setups(const std::vector<int>& red_setup,
                                           const std::vector<int>& black_setup,
                                           std::string& error_message) const {
    // 验证红方布阵
    if (!validate_setup(red_setup, RED, error_message)) {
        error_message = "Red setup invalid: " + error_message;
        return false;
    }
    
    // 验证黑方布阵
    if (!validate_setup(black_setup, BLACK, error_message)) {
        error_message = "Black setup invalid: " + error_message;
        return false;
    }
    
    return true;
}

bool SetupValidator::check_piece_count(const std::vector<int>& setup,
                                        std::string& error_message) const {
    std::map<PieceType, int> count_map = get_piece_count_map(setup);
    
    // 定义期望的棋子数量
    std::map<PieceType, int> expected_counts = {
        {MINE, 3},
        {BOMB, 2},
        {FLAG, 1},
        {SAPPER, 3},
        {LIEUTENANT, 3},
        {CAPTAIN, 3},
        {MAJOR, 2},
        {COLONEL, 2},
        {BRIGADIER, 2},
        {GENERAL, 2},
        {ARMY_CMDR, 1},
        {FIELD_MARSHAL, 1}
    };
    
    // 检查每种棋子的数量
    for (const auto& pair : expected_counts) {
        PieceType type = pair.first;
        int expected = pair.second;
        int actual = count_map[type];
        
        if (actual != expected) {
            std::ostringstream oss;
            oss << "Invalid piece count: expected " << expected 
                << " for piece type " << type << ", got " << actual;
            error_message = oss.str();
            return false;
        }
    }
    
    // 检查总数量（应该恰好25个）
    int total = 0;
    for (const auto& pair : count_map) {
        total += pair.second;
    }
    
    if (total != 25) {
        error_message = "Total piece count must be 25, got " + std::to_string(total);
        return false;
    }
    
    return true;
}

bool SetupValidator::check_position_restrictions(const std::vector<int>& setup,
                                                  PlayerId player,
                                                  std::string& error_message) const {
    // 检查军旗位置
    if (!check_flag_position(setup, player, error_message)) {
        return false;
    }
    
    // 检查地雷位置
    if (!check_mine_position(setup, player, error_message)) {
        return false;
    }
    
    // 检查炸弹位置
    if (!check_bomb_position(setup, player, error_message)) {
        return false;
    }
    
    // 检查行营是否为空
    if (!check_camp_empty(setup, player, error_message)) {
        return false;
    }
    
    return true;
}

bool SetupValidator::check_flag_position(const std::vector<int>& setup,
                                          PlayerId player,
                                          std::string& error_message) const {
    int start_row = get_start_row(player);
    
    // 军旗必须在大本营（HQ）
    // 红方：row 0, col 1 或 3
    // 黑方：row 11, col 1 或 3
    
    std::vector<int> hq_positions = board_.get_hq_positions();
    int flag_count = 0;
    
    for (size_t i = 0; i < setup.size(); ++i) {
        int row = start_row + i / 5;
        int col = i % 5;
        int pos = pos_to_index(row, col);
        
        if (setup[i] == FLAG) {
            // 检查是否在大本营
            bool is_hq = false;
            for (int hq_pos : hq_positions) {
                if (pos == hq_pos) {
                    is_hq = true;
                    break;
                }
            }
            
            if (!is_hq) {
                std::ostringstream oss;
                oss << "Flag must be in HQ (row " << (player == RED ? 0 : 11)
                    << ", col 1 or 3)";
                error_message = oss.str();
                return false;
            }
            
            flag_count++;
        }
    }
    
    if (flag_count != 1) {
        error_message = "Must have exactly 1 flag";
        return false;
    }
    
    return true;
}

bool SetupValidator::check_mine_position(const std::vector<int>& setup,
                                          PlayerId player,
                                          std::string& error_message) const {
    int start_row = get_start_row(player);
    
    // 地雷只能在最后两排
    // 红方：row 0, 1
    // 黑方：row 10, 11
    
    for (size_t i = 0; i < setup.size(); ++i) {
        int row = start_row + i / 5;
        int col = i % 5;
        
        if (setup[i] == MINE) {
            if (player == RED) {
                // 红方地雷只能在 row 0 或 1
                if (row > 1) {
                    error_message = "Mines can only be in rows 0-1 for Red player";
                    return false;
                }
            } else {
                // 黑方地雷只能在 row 10 或 11
                if (row < 10) {
                    error_message = "Mines can only be in rows 10-11 for Black player";
                    return false;
                }
            }
        }
    }
    
    return true;
}

bool SetupValidator::check_bomb_position(const std::vector<int>& setup,
                                          PlayerId player,
                                          std::string& error_message) const {
    int start_row = get_start_row(player);
    
    // 炸弹不能在第一排（最前线）
    // 红方：不能在 row 5
    // 黑方：不能在 row 6
    
    int front_row = (player == RED) ? 5 : 6;
    
    for (size_t i = 0; i < setup.size(); ++i) {
        int row = start_row + i / 5;
        int col = i % 5;
        
        if (setup[i] == BOMB) {
            if (row == front_row) {
                error_message = "Bombs cannot be in the front row (row " + 
                               std::to_string(front_row) + ")";
                return false;
            }
        }
    }
    
    return true;
}

bool SetupValidator::check_camp_empty(const std::vector<int>& setup,
                                       PlayerId player,
                                       std::string& error_message) const {
    int start_row = get_start_row(player);
    
    // 行营内不能有棋子
    const std::vector<int>& camp_positions = board_.get_camp_positions();
    
    for (int camp_pos : camp_positions) {
        int r, c;
        index_to_pos(camp_pos, r, c);
        
        // 检查是否在该阵营的区域内
        if (player == RED && (r < RED_AREA_START || r > RED_AREA_END)) {
            continue;
        }
        if (player == BLACK && (r < BLACK_AREA_START || r > BLACK_AREA_END)) {
            continue;
        }
        
        // 计算在setup中的索引
        int setup_index = (r - start_row) * 5 + c;
        
        if (setup_index >= 0 && setup_index < static_cast<int>(setup.size())) {
            if (setup[setup_index] != EMPTY) {
                std::ostringstream oss;
                oss << "Camp at (" << r << ", " << c << ") must be empty";
                error_message = oss.str();
                return false;
            }
        }
    }
    
    return true;
}

std::map<PieceType, int> SetupValidator::get_piece_count_map(
    const std::vector<int>& setup) const {
    std::map<PieceType, int> count_map;
    
    for (int piece : setup) {
        if (piece != EMPTY) {
            PieceType type = static_cast<PieceType>(piece);
            count_map[type]++;
        }
    }
    
    return count_map;
}

int SetupValidator::get_start_row(PlayerId player) const {
    return (player == RED) ? RED_AREA_START : BLACK_AREA_START;
}

} // namespace junqi