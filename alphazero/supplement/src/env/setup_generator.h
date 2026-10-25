#ifndef SETUP_GENERATOR_H
#define SETUP_GENERATOR_H

#include "junqi_types.h"
#include "junqi_board.h"
#include <vector>
#include <map>
#include <string>

namespace junqi {

// 自动布阵器 - 生成合法的初始布阵
class SetupGenerator {
public:
    SetupGenerator(const JunqiBoard& board);
    
    // 生成随机合法布阵
    std::vector<int> generate_random_setup(PlayerId player);
    
    // 从外部布阵设置（验证并设置）
    bool set_custom_setup(const std::vector<int>& setup, PlayerId player, 
                          std::string& error_message);
    
    // 获取当前布阵
    const std::vector<int>& get_setup(PlayerId player) const;
    
private:
    const JunqiBoard& board_;
    
    // 红方和黑方的布阵
    std::vector<int> red_setup_;
    std::vector<int> black_setup_;
    
    // 生成棋子列表
    std::vector<PieceType> generate_piece_list();
    
    // 放置军旗（在大本营）
    void place_flag(std::vector<int>& setup, PlayerId player, 
                  std::vector<PieceType>& pieces);
    
    // 放置地雷（在后两排）
    void place_mines(std::vector<int>& setup, PlayerId player, 
                     std::vector<PieceType>& pieces);
    
    // 放置炸弹（不在第一排）
    void place_bombs(std::vector<int>& setup, PlayerId player,
                      std::vector<PieceType>& pieces);
    
    // 随机放置剩余棋子
    void place_remaining_pieces(std::vector<int>& setup, PlayerId player,
                                 std::vector<PieceType>& pieces);
    
    // 获取可用位置列表（排除行营）
    std::vector<int> get_available_positions(const std::vector<int>& setup,
                                              PlayerId player);
    
    // 获取特定行范围的位置列表
    std::vector<int> get_positions_in_rows(int start_row, int end_row);
    
    // 获取大本营位置
    std::vector<int> get_hq_positions(PlayerId player);
    
    // 获取行营位置
    std::vector<int> get_camp_positions(PlayerId player);
    
    // 随机选择一个位置
    int select_random_position(const std::vector<int>& positions);
    
    // 检查位置是否可用
    bool is_position_available(const std::vector<int>& setup, int index);
};

} // namespace junqi

#endif // SETUP_GENERATOR_H