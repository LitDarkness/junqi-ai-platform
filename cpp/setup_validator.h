#ifndef SETUP_VALIDATOR_H
#define SETUP_VALIDATOR_H

#include "junqi_types.h"
#include "junqi_board.h"
#include "junqi_state.h"
#include <vector>
#include <map>
#include <string>

namespace junqi {

// 布阵校验器 - 验证初始布阵的合法性
class SetupValidator {
public:
    SetupValidator(const JunqiBoard& board);
    
    // 验证布阵是否合法
    // 返回true表示合法，false表示非法
    // 如果非法，error_message会被填充错误信息
    bool validate_setup(const std::vector<int>& setup, PlayerId player, 
                        std::string& error_message) const;
    
    // 验证双方布阵
    bool validate_both_setups(const std::vector<int>& red_setup, 
                             const std::vector<int>& black_setup,
                             std::string& error_message) const;
    
private:
    const JunqiBoard& board_;
    
    // 检查棋子数量
    bool check_piece_count(const std::vector<int>& setup, std::string& error_message) const;
    
    // 检查位置限制（军旗、地雷、炸弹、行营）
    bool check_position_restrictions(const std::vector<int>& setup, PlayerId player,
                                    std::string& error_message) const;
    
    // 检查军旗位置
    bool check_flag_position(const std::vector<int>& setup, PlayerId player,
                            std::string& error_message) const;
    
    // 检查地雷位置
    bool check_mine_position(const std::vector<int>& setup, PlayerId player,
                            std::string& error_message) const;
    
    // 检查炸弹位置
    bool check_bomb_position(const std::vector<int>& setup, PlayerId player,
                            std::string& error_message) const;
    
    // 检查行营是否为空
    bool check_camp_empty(const std::vector<int>& setup, PlayerId player,
                         std::string& error_message) const;
    
    // 获取棋子数量映射
    std::map<PieceType, int> get_piece_count_map(const std::vector<int>& setup) const;
    
    // 获取起始行
    int get_start_row(PlayerId player) const;
};

} // namespace junqi

#endif // SETUP_VALIDATOR_H