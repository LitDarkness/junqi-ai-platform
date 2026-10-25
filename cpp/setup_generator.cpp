#include "setup_generator.h"
#include "setup_validator.h"
#include <algorithm>
#include <random>
#include <ctime>
#include <stdexcept>
#include <iostream>

namespace junqi {

SetupGenerator::SetupGenerator(const JunqiBoard& board) : board_(board) {
    red_setup_.resize(30, EMPTY);
    black_setup_.resize(30, EMPTY);
}

std::vector<int> SetupGenerator::generate_random_setup(PlayerId player) {
    // 每方6行×5列=30个位置，放置25个棋子，5个空位（行营）
    std::vector<int> setup(30, EMPTY);
    
    // 生成棋子列表
    std::vector<PieceType> pieces = generate_piece_list();
    
    // 1. 放置军旗
    place_flag(setup, player, pieces);
    
    // 2. 放置地雷
    place_mines(setup, player, pieces);
    
    // 3. 放置炸弹
    place_bombs(setup, player, pieces);
    
    // 4. 放置剩余棋子
    place_remaining_pieces(setup, player, pieces);
    
    // 保存布阵
    if (player == RED) {
        red_setup_ = setup;
    } else {
        black_setup_ = setup;
    }
    
    return setup;
}

bool SetupGenerator::set_custom_setup(const std::vector<int>& setup, PlayerId player,
                                       std::string& error_message) {
    SetupValidator validator(board_);
    
    // 验证布阵
    if (!validator.validate_setup(setup, player, error_message)) {
        return false;
    }
    
    // 保存布阵
    if (player == RED) {
        red_setup_ = setup;
    } else {
        black_setup_ = setup;
    }
    
    return true;
}

const std::vector<int>& SetupGenerator::get_setup(PlayerId player) const {
    return (player == RED) ? red_setup_ : black_setup_;
}

std::vector<PieceType> SetupGenerator::generate_piece_list() {
    std::vector<PieceType> pieces;
    
    // 按照需求定义的棋子数量
    for (int i = 0; i < 3; ++i) pieces.push_back(MINE);
    for (int i = 0; i < 2; ++i) pieces.push_back(BOMB);
    pieces.push_back(FLAG);
    for (int i = 0; i < 3; ++i) pieces.push_back(SAPPER);
    for (int i = 0; i < 3; ++i) pieces.push_back(LIEUTENANT);
    for (int i = 0; i < 3; ++i) pieces.push_back(CAPTAIN);
    for (int i = 0; i < 2; ++i) pieces.push_back(MAJOR);
    for (int i = 0; i < 2; ++i) pieces.push_back(COLONEL);
    for (int i = 0; i < 2; ++i) pieces.push_back(BRIGADIER);
    for (int i = 0; i < 2; ++i) pieces.push_back(GENERAL);
    pieces.push_back(ARMY_CMDR);
    pieces.push_back(FIELD_MARSHAL);
    
    // 随机打乱
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(pieces.begin(), pieces.end(), g);
    
    return pieces;
}

void SetupGenerator::place_flag(std::vector<int>& setup, PlayerId player,
                             std::vector<PieceType>& pieces) {
    std::vector<int> hq_positions = get_hq_positions(player);
    
    if (hq_positions.empty()) {
        throw std::runtime_error("No HQ positions available");
    }
    
    // 随机选择一个大本营放置军旗
    // get_hq_positions 已经返回setup数组的索引
    int setup_index = select_random_position(hq_positions);
    
    setup[setup_index] = FLAG;
    
    // 从pieces列表中移除军旗
    auto it = std::find(pieces.begin(), pieces.end(), FLAG);
    if (it != pieces.end()) {
        pieces.erase(it);
    }
}

void SetupGenerator::place_mines(std::vector<int>& setup, PlayerId player,
                                  std::vector<PieceType>& pieces) {
    int start_row = (player == RED) ? 0 : 10;
    int end_row = (player == RED) ?1 : 11;
    
    std::vector<int> available_positions = get_positions_in_rows(start_row, end_row);
    
    // 过滤掉已占用的位置和行营
    std::vector<int> valid_positions;
    for (int pos : available_positions) {
        int r, c;
        index_to_pos(pos, r, c);
        int setup_index = (r - ((player == RED) ? 0 : 6)) * 5 + c;
        
        if (is_position_available(setup, setup_index) && !board_.is_camp(r, c)) {
            valid_positions.push_back(setup_index);
        }
    }
    
    // 放置3个地雷
    for (int i = 0; i < 3; ++i) {
        // 从pieces中找到地雷
        auto it = std::find(pieces.begin(), pieces.end(), MINE);
        if (it != pieces.end() && !valid_positions.empty()) {
            pieces.erase(it);
            
            // 随机选择一个位置
            int index = select_random_position(valid_positions);
            
            if (index >= 0) {
                // 放置地雷
                setup[index] = MINE;
                
                // 从可用位置中移除
                valid_positions.erase(std::find(valid_positions.begin(), valid_positions.end(), index));
            }
        }
    }
}

void SetupGenerator::place_bombs(std::vector<int>& setup, PlayerId player,
                                  std::vector<PieceType>& pieces) {
    int front_row = (player == RED) ? 5 : 6;
    int start_row = (player == RED) ? 0 :6;
    int end_row = (player == RED) ? 5 : 11;
    
    std::vector<int> available_positions;
    
    // 遍历所有位置
    for (int row = start_row; row <= end_row; ++row) {
        for (int col = 0; col < BOARD_COLS; ++col) {
            // 跳过前线
            if (row == front_row) {
                continue;
            }
            
            // 跳过行营（炸弹不能放在行营）
            if (board_.is_camp(row, col)) {
                continue;
            }
            
            int setup_index = (row - start_row) * 5 + col;
            
            if (is_position_available(setup, setup_index)) {
                available_positions.push_back(setup_index);
            }
        }
    }
    
    // 放置2个炸弹
    for (int i = 0; i < 2; ++i) {
        // 从pieces中找到炸弹
        auto it = std::find(pieces.begin(), pieces.end(), BOMB);
        if (it != pieces.end() && !available_positions.empty()) {
            pieces.erase(it);
            
            // 随机选择一个位置
            int index = select_random_position(available_positions);
            
            if (index >= 0) {
                // 放置炸弹
                setup[index] = BOMB;
                
                // 从可用位置中移除
                available_positions.erase(std::find(available_positions.begin(), available_positions.end(), index));
            }
        }
    }
}

void SetupGenerator::place_remaining_pieces(std::vector<int>& setup, PlayerId player,
                                             std::vector<PieceType>& pieces) {
    std::vector<int> available_positions;
    
    int start_row = (player == RED) ? 0 : 6;
    int end_row = (player == RED) ? 5 : 11;
    
    // 获取所有可用位置（排除行营，但包含大本营）
    for (int row = start_row; row <= end_row; ++row) {
        for (int col = 0; col < BOARD_COLS; ++col) {
            int setup_index = (row - start_row) * 5 + col;
            
            // 行营不能放任何棋子
            if (board_.is_camp(row, col)) {
                continue;
            }
            
            // 普通棋子可以放大本营（只要位置为空）
            if (is_position_available(setup, setup_index)) {
                available_positions.push_back(setup_index);
            }
        }
    }
    
    // 放置剩余棋子
    while (!pieces.empty()) {
        // 检查是否还有可用位置
        if (available_positions.empty()) {
            throw std::runtime_error("No available positions for remaining pieces");
        }
        
        // 随机选择一个棋子
        int piece_index = rand() % pieces.size();
        PieceType piece = pieces[piece_index];
        pieces.erase(pieces.begin() + piece_index);
        
        // 随机选择一个位置
        int index = select_random_position(available_positions);
        
        if (index < 0) {
            throw std::runtime_error("Failed to select valid position");
        }
        
        // 放置棋子
        setup[index] = piece;
        
        // 从可用位置中移除
        auto it = std::find(available_positions.begin(), available_positions.end(), index);
        if (it != available_positions.end()) {
            available_positions.erase(it);
        }
    }
}

std::vector<int> SetupGenerator::get_available_positions(const std::vector<int>& setup,
                                                        PlayerId player) {
    std::vector<int> available;
    
    int start_row = (player == RED) ? 0 : 6;
    int end_row = (player == RED) ? 5 : 11;
    
    for (int row = start_row; row <= end_row; ++row) {
        for (int col = 0; col < BOARD_COLS; ++col) {
            int setup_index = (row - start_row) * 5 + col;
            
            if (is_position_available(setup, setup_index) && !board_.is_camp(row, col)) {
                available.push_back(setup_index);
            }
        }
    }
    
    return available;
}

std::vector<int> SetupGenerator::get_positions_in_rows(int start_row, int end_row) {
    std::vector<int> positions;
    
    for (int row = start_row; row <= end_row; ++row) {
        for (int col = 0; col < BOARD_COLS; ++col) {
            positions.push_back(pos_to_index(row, col));
        }
    }
    
    return positions;
}

std::vector<int> SetupGenerator::get_hq_positions(PlayerId player) {
    const std::vector<int>& hq_positions = board_.get_hq_positions();
    std::vector<int> player_hq;
    
    int start_row = (player == RED) ? 0 : 6;
    int end_row = (player == RED) ? 5 : 11;
    
    for (int pos : hq_positions) {
        int r, c;
        index_to_pos(pos, r, c);
        
        if (r >= start_row && r <= end_row) {
            // 转换为setup数组索引
            int setup_index = (r - start_row) * 5 + c;
            player_hq.push_back(setup_index);
        }
    }
    
    return player_hq;
}

std::vector<int> SetupGenerator::get_camp_positions(PlayerId player) {
    const std::vector<int>& camp_positions = board_.get_camp_positions();
    std::vector<int> player_camp;
    
    int start_row = (player == RED) ? 0 : 6;
    int end_row = (player == RED) ? 5 : 11;
    
    for (int pos : camp_positions) {
        int r, c;
        index_to_pos(pos, r, c);
        
        if (r >= start_row && r <= end_row) {
            // 转换为setup数组索引
            int setup_index = (r - start_row) * 5 + c;
            player_camp.push_back(setup_index);
        }
    }
    
    return player_camp;
}

int SetupGenerator::select_random_position(const std::vector<int>& positions) {
    if (positions.empty()) {
        return -1;
    }
    
    int index = rand() % positions.size();
    return positions[index];
}

bool SetupGenerator::is_position_available(const std::vector<int>& setup, int index) {
    if (index < 0 || index >= static_cast<int>(setup.size())) {
        return false;
    }
    
    return setup[index] == EMPTY;
}

} // namespace junqi