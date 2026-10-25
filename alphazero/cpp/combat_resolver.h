#ifndef COMBAT_RESOLVER_H
#define COMBAT_RESOLVER_H

#include "junqi_types.h"
#include "junqi_state.h"
#include <string>

namespace junqi {

// 战斗结算器 - 处理战斗结果
class CombatResolver {
public:
    CombatResolver(JunqiState& state);
    
    // 解析战斗结果
    // 返回 CombatResult 并更新游戏状态
    CombatResult resolve_combat(int from, int to);
    
    // 检查当前玩家是否有可移动的棋子
    bool has_movable_pieces(PlayerId player);
    
private:
    JunqiState& state_;
    
    // 执行战斗并返回结果
    CombatResult execute_combat(const Piece& attacker, const Piece& defender);
    
    // 检查游戏是否结束
    void check_game_over(PlayerId current_player);
    
    // 检查是否扛旗
    bool check_flag_captured(PlayerId player);
    
    // 检查是否全歼
    bool check_all_pieces_destroyed(PlayerId player);
};

} // namespace junqi

#endif // COMBAT_RESOLVER_H