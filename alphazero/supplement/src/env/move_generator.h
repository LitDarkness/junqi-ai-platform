#ifndef MOVE_GENERATOR_H
#define MOVE_GENERATOR_H

#include "junqi_types.h"
#include "junqi_board.h"
#include "junqi_state.h"
#include <cstdint>
#include <vector>

namespace junqi {

// 移动生成器 - 生成所有合法动作
class MoveGenerator {
public:
    MoveGenerator(const JunqiBoard& board);
    
    // 生成所有合法动作
    // 返回动作列表，每个动作是 <from_pos, to_pos> 的对
    std::vector<std::pair<int, int>> generate_legal_moves(const JunqiState& state) const;
    
    // 获取合法动作掩码（3600维 bool 数组）
    std::vector<bool> get_legal_action_mask(const JunqiState& state, PlayerId player) const;
    void fill_legal_action_mask(const JunqiState& state, PlayerId player, std::uint8_t* out) const;
    
    // 检查单个动作是否合法
    bool is_move_valid(const JunqiState& state, int from, int to) const;
    
    // 生成某个棋子的所有合法移动（公共接口）
    std::vector<int> get_possible_moves(const JunqiState& state, int from) const;
    
private:
    const JunqiBoard& board_;
    
    // 生成某个棋子的所有合法移动
    std::vector<int> generate_piece_moves(const JunqiState& state, int from) const;
    
    // 生成公路移动
    std::vector<int> generate_highway_moves(const JunqiState& state, int from) const;
    
    // 生成铁路移动（非工兵）
    std::vector<int> generate_railway_moves(const JunqiState& state, int from) const;
    
    // 生成工兵的铁路移动（可以拐弯）
    std::vector<int> generate_sapper_railway_moves(const JunqiState& state, int from) const;
    
    // BFS搜索铁路路径
    std::vector<int> bfs_railway_search(const JunqiState& state, int start) const;
    
    // 检查铁路路径是否畅通
    bool is_railway_path_clear(const JunqiState& state, int from, int to) const;
    
    // 检查铁路直线路径是否畅通（同一直线）
    bool is_railway_line_clear(const JunqiState& state, int from, int to) const;
    
    // 检查目标位置是否可以进入
    bool can_enter(const JunqiState& state, int from, int to, PlayerId player) const;
    
    // 获取两个位置之间的中间位置
    std::vector<int> get_positions_between(int from, int to) const;
};

} // namespace junqi

#endif // MOVE_GENERATOR_H
