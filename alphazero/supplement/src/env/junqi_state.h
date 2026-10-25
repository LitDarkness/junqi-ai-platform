#ifndef JUNQI_STATE_H
#define JUNQI_STATE_H

#include "junqi_types.h"
#include "junqi_board.h"
#include <vector>

namespace junqi {

// 棋子信息：包含类型和阵营
struct Piece {
    PieceType type;
    PlayerId owner;
    
    Piece() : type(EMPTY), owner(NONE) {}
    Piece(PieceType t, PlayerId o) : type(t), owner(o) {}
    
    bool is_empty() const { return type == EMPTY; }
    bool is_owned_by(PlayerId player) const { return owner == player; }
};

// 游戏状态类
class JunqiState {
public:
    JunqiState();
    
    // 重置游戏状态
    void reset(const std::vector<int>& red_setup, const std::vector<int>& black_setup);
    
    // 获取指定位置的棋子
    const Piece& get_piece(int index) const;
    const Piece& get_piece(int row, int col) const;
    
    // 设置指定位置的棋子
    void set_piece(int index, const Piece& piece);
    void set_piece(int row, int col, const Piece& piece);
    
    // 移除指定位置的棋子
    void remove_piece(int index);
    void remove_piece(int row, int col);
    
    // 移动棋子
    void move_piece(int from, int to);
    
    // 获取当前回合方
    PlayerId get_current_player() const { return current_player_; }
    
    // 设置当前回合方
    void set_current_player(PlayerId player) { current_player_ = player; }
    
    // 切换回合
    void switch_turn();
    
    // 获取回合计数
    int get_turn_count() const { return turn_count_; }
    
    // 增加回合计数
    void increment_turn_count() { ++turn_count_; }
    
    // 检查司令是否死亡
    bool is_field_marshal_dead(PlayerId player) const;
    
    // 设置司令死亡状态
    void set_field_marshal_dead(PlayerId player);
    
    // 检查军旗是否被揭示
    bool is_flag_revealed(PlayerId player) const { return flag_revealed_[player_to_index(player)]; }
    
    // 设置军旗揭示状态
    void set_flag_revealed(PlayerId player);
    
    // 获取游戏是否结束
    bool is_game_over() const { return game_over_; }
    
    // 设置游戏结束
    void set_game_over(bool over) { game_over_ = over; }
    
    // 获取获胜方
    PlayerId get_winner() const { return winner_; }
    
    // 设置获胜方
    void set_winner(PlayerId winner) { winner_ = winner; }
    
    // 检查某个阵营是否还有可移动的棋子
    bool has_movable_pieces(PlayerId player) const;
    
    // 获取棋盘（用于访问JunqiBoard）
    const JunqiBoard& get_board() const { return board_; }
    
    // 获取完整棋盘状态（用于调试）
    std::vector<int> get_board_state() const;
    
    // 深拷贝状态
    JunqiState clone() const;
    
    // 清空历史记录
    void clear_history() { move_history_.clear(); }
    
    // 获取移动历史
    const std::vector<std::pair<int, int>>& get_move_history() const { return move_history_; }
    
    // 添加移动到历史
    void add_move_to_history(int from, int to) { move_history_.push_back({from, to}); }
    
private:
    // 将 PlayerId 转换为数组索引
    int player_to_index(PlayerId player) const {
        return (player == RED) ? 0 : ((player == BLACK) ? 1 : -1);
    }
    
    // 棋盘（静态信息）
    JunqiBoard board_;
    
    // 棋子状态（60个位置）
    std::vector<Piece> pieces_;
    
    // 当前回合方
    PlayerId current_player_;
    
    // 回合计数
    int turn_count_;
    
    // 司令死亡状态
    bool field_marshal_dead_[2]; // [RED, BLACK]
    
    // 军旗揭示状态
    bool flag_revealed_[2]; // [RED, BLACK]
    
    // 游戏是否结束
    bool game_over_;
    
    // 获胜方
    PlayerId winner_;
    
    // 移动历史记录
    std::vector<std::pair<int, int>> move_history_;
};

} // namespace junqi

#endif // JUNQI_STATE_H
