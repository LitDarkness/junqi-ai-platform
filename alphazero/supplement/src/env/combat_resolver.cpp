#include "combat_resolver.h"
#include "move_generator.h"
#include <iostream>

namespace junqi {

CombatResolver::CombatResolver(JunqiState& state) : state_(state) {}

CombatResult CombatResolver::resolve_combat(int from, int to) {
    const Piece& attacker = state_.get_piece(from);
    const Piece& defender = state_.get_piece(to);
    
    // 保存攻击方的owner（因为move_piece会改变引用）
    PlayerId attacker_owner = attacker.owner;
    
    // 执行战斗
    CombatResult result = execute_combat(attacker, defender);
    
    // 根据结果更新棋盘
    if (result == WIN) {
        // 攻击方胜：移除防守方，攻击方移动到目标位置
        state_.remove_piece(to);
        state_.move_piece(from, to);
    } else if (result == LOSS) {
        // 攻击方负：移除攻击方
        state_.remove_piece(from);
    } else if (result == DRAW) {
        // 同归于尽：都移除
        state_.remove_piece(from);
        state_.remove_piece(to);
    } else if (result == GAME_OVER_WIN || result == GAME_OVER_LOSS) {
        // 游戏结束：处理扛旗情况
        // 移除防守方（军旗），攻击方移动到目标位置
        state_.remove_piece(to);
        state_.move_piece(from, to);
        
        // 扛旗成功，立即设置游戏结束
        state_.set_game_over(true);
        // // 使用保存的attacker_owner
        // std::cout << "DEBUG: attacker.owner = " << static_cast<int>(attacker_owner) << std::endl;
        state_.set_winner(attacker_owner);
        // // 调试：打印winner
        // std::cout << "DEBUG: winner = " << static_cast<int>(state_.get_winner()) << std::endl;
        return result;
    }
    
    // 检查游戏是否结束（全歼）
    check_game_over(attacker_owner);
    
    // 切换回合
    state_.switch_turn();
    state_.increment_turn_count();
    
    // 检查当前玩家是否有可移动的棋子
    PlayerId current_player = state_.get_current_player();
    if (!has_movable_pieces(current_player)) {
        state_.set_game_over(true);
        state_.set_winner(opposite_player(current_player));
    }
    
    return result;
}

CombatResult CombatResolver::execute_combat(const Piece& attacker, const Piece& defender) {
    // 优先级1：攻击军旗 -> 游戏结束，攻击方胜
    if (defender.type == FLAG) {
        return GAME_OVER_WIN;
    }
    
    // 优先级2：炸弹参与 -> 同归于尽
    if (attacker.type == BOMB || defender.type == BOMB) {
        return DRAW;
    }
    
    // 优先级3：工兵挖地雷 -> 工兵胜
    if (attacker.type == SAPPER && defender.type == MINE) {
        return WIN;
    }
    
    // 优先级4：非工兵挖地雷 -> 地雷胜（攻击方负）
    if (defender.type == MINE) {
        return LOSS;
    }
    
    // 优先级5：等级比较
    int attacker_rank = get_piece_rank(attacker.type);
    int defender_rank = get_piece_rank(defender.type);
    
    if (attacker_rank > defender_rank) {
        return WIN;
    } else if (attacker_rank < defender_rank) {
        return LOSS;
    } else {
        return DRAW;
    }
}

void CombatResolver::check_game_over(PlayerId current_player) {
    // 检查是否全歼
    if (check_all_pieces_destroyed(opposite_player(current_player))) {
        state_.set_game_over(true);
        state_.set_winner(current_player);
        return;
    }
}

bool CombatResolver::check_flag_captured(PlayerId player) {
    // 这个函数不再使用
    // 扛旗判断直接在execute_combat中处理，返回GAME_OVER_WIN
    return false;
}

bool CombatResolver::check_all_pieces_destroyed(PlayerId player) {
    // 遍历所有位置，检查该阵营是否还有棋子
    int start_row = (player == RED) ? RED_AREA_START : BLACK_AREA_START;
    int end_row = (player == RED) ? RED_AREA_END : BLACK_AREA_END;
    
    for (int row = start_row; row <= end_row; ++row) {
        for (int col = 0; col < BOARD_COLS; ++col) {
            const Piece& piece = state_.get_piece(row, col);
            
            // 如果有该阵营的棋子，说明没有全歼
            if (piece.is_owned_by(player)) {
                return false;
            }
        }
    }
    
    return true;
}

bool CombatResolver::has_movable_pieces(PlayerId player) {
    MoveGenerator move_gen(state_.get_board());
    
    // 遍历该阵营的所有棋子
    for (int from = 0; from < BOARD_SIZE; ++from) {
        const Piece& piece = state_.get_piece(from);
        
        // 跳过空位和敌方棋子
        if (piece.is_empty() || !piece.is_owned_by(player)) {
            continue;
        }
        
        // 跳过不可移动的棋子（地雷、军旗）
        if (!is_movable_piece(piece.type)) {
            continue;
        }
        
        // 跳过大本营内的棋子
        if (state_.get_board().is_hq(from)) {
            continue;
        }
        
        if (!move_gen.get_possible_moves(state_, from).empty()) {
            return true;
        }
    }
    
    return false;
}

} // namespace junqi
