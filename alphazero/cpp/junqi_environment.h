#ifndef JUNQI_ENVIRONMENT_H
#define JUNQI_ENVIRONMENT_H

#include "junqi_types.h"
#include "junqi_board.h"
#include "junqi_state.h"
#include "setup_validator.h"
#include "move_generator.h"
#include "combat_resolver.h"
#include "setup_generator.h"
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <cstdint>
#include <unordered_map>

namespace junqi {

// 棋子信息结构
struct PieceInfo {
    int type;   // 棋子类型
    int owner;  // 所有者 (1=红, -1=黑, 0=空)
};

// 移动记录结构
struct MoveRecord {
    int from_pos;
    int to_pos;
    int player_id;
    int result;
};

struct LegalityResult {
    bool legal = false;
    std::string code = "STATE_INVALID";
    std::string message_key = "move.state_invalid";
    int blocking_square = -1;
    std::vector<int> path;
};

// 主环境类 - 提供完整的游戏接口
class JunqiEnvironment {
public:
    JunqiEnvironment();
    JunqiEnvironment(const JunqiEnvironment& other);
    JunqiEnvironment& operator=(const JunqiEnvironment&) = delete;
    JunqiEnvironment(JunqiEnvironment&&) = delete;
    JunqiEnvironment& operator=(JunqiEnvironment&&) = delete;
    void seed(std::uint64_t value) { setup_generator_.seed(value); }
    
    // ==================== 初始化 ====================
    // 重置游戏状态
    // 输入双方的初始布阵 (5x5矩阵 或 List格式)
    // 自动校验布阵合法性，非法则抛出异常
    void reset(const std::vector<int>& red_setup, const std::vector<int>& black_setup);
    
    // 使用自动生成的随机布阵重置
    void reset_random();
    
    // ==================== 核心交互 ====================
    // 执行动作
    // 参数: from_pos (0-59), to_pos (0-59)
    // 返回: CombatResult (这是给Agent唯一的反馈!)
    CombatResult step(int from_pos, int to_pos);
    
    // ==================== 状态获取 ====================
    
    // 1. 获取全知视角 (用于 Teacher Training / Debug)
    // 返回: 12x5 矩阵，包含所有真实棋子ID
    std::vector<int> get_perfect_state() const;
    
    // 2. 获取观测视角 (用于 Student Inference)
    // 参数: player_id (RED/BLACK)
    // 返回: 12x5 矩阵
    //   - 己方: 真实ID
    //   - 敌方暗棋: UNKNOWN (-1)
    std::vector<int> get_observation(int player_id);
    
    // 3. 获取合法动作掩码
    // 返回: 长度 3600 (60x60) 的 Bool 数组
    std::vector<bool> get_legal_action_mask(int player_id);
    std::vector<int> get_legal_actions(int player_id);
    LegalityResult check_move(int from_pos, int to_pos) const;
    std::vector<float> get_feature_planes(int player_id);
    float get_material_score(int player_id) const;
    void fill_perfect_state(int* out) const;
    void fill_feature_planes(float* out, int player_id);
    void fill_legal_action_mask(std::uint8_t* out, int player_id);
    
    // ==================== 辅助信息 ====================
    // 司令是否已死？(用于 Transformer 作为一个全局特征输入)
    bool is_field_marshal_dead(int player_id) const;
    
    // 当前回合数
    int get_turn_count() const;
    
    // 当前行动方
    int get_current_player() const;
    
    // 游戏是否结束
    bool is_game_over() const;
    
    // 获取获胜方
    int get_winner() const;
    std::uint64_t get_state_key() const;
    std::uint64_t get_position_key() const { return position_key(); }
    int get_no_capture_ply() const { return no_capture_ply_; }
    int get_current_repetition_count() const;
    // Return the number of previous occurrences (0, 1, or 2) of the
    // successor position.  Value 2 means the legal move immediately triggers
    // the configured threefold-repetition draw.  Non-repetition terminal
    // moves return 0 because draw tracking is not applied after a decisive
    // terminal transition.
    std::uint8_t get_successor_repetition_context(int from_pos, int to_pos) const;
    std::size_t get_history_size() const { return state_.get_move_history().size(); }
    // Stable terminal reason code defined by standard_open_v1.
    const std::string& get_terminal_reason() const { return terminal_reason_; }
    
    // ==================== 布阵相关 ====================
    // 设置红方布阵
    bool set_red_setup(const std::vector<int>& setup, std::string& error_message);
    
    // 设置黑方布阵
    bool set_black_setup(const std::vector<int>& setup, std::string& error_message);
    
    // 获取红方布阵
    const std::vector<int>& get_red_setup();
    
    // 获取黑方布阵
    const std::vector<int>& get_black_setup();
    
    // 生成随机红方布阵
    std::vector<int> generate_red_setup();
    
    // 生成随机黑方布阵
    std::vector<int> generate_black_setup();
    
    // 获取布阵生成器（用于更高级的布阵策略）
    SetupGenerator& get_setup_generator() { return setup_generator_; }
    
    // ==================== 便利工具函数 ====================
    
    // 位置转换函数
    static int pos_to_index(int row, int col);
    static std::pair<int, int> index_to_pos(int index);
    static int get_action_index(int from, int to);
    static std::pair<int, int> decode_action_index(int action_index);
    
    // 棋子信息查询
    PieceInfo get_piece_at(int pos) const;
    std::vector<int> get_player_pieces(int player_id) const;
    int get_piece_count(int player_id, int piece_type) const;
    
    // 移动验证
    bool can_move(int from_pos, int to_pos) const;
    std::vector<int> get_possible_moves(int from_pos) const;
    std::vector<int> get_movable_pieces(int player_id) const;
    
    // 棋盘渲染
    std::string render() const;
    void print_board() const;
    
    // 游戏历史
    MoveRecord get_last_move() const;
    std::vector<MoveRecord> get_game_history() const;
    void clear_history();
    
    // 从给定局面加载
    void load_board_state(const std::vector<int>& board_state, int current_player);

    static constexpr int kFeatureChannels = 14;
    static constexpr int kActionSpaceSize = BOARD_SIZE * BOARD_SIZE;
    
private:
    // 检查战斗结果是否需要揭示军旗
    void check_reveal_flag(CombatResult result, PlayerId player);
    
    void reset_draw_tracking();
    void update_draw_tracking(bool capture);
    std::uint64_t position_key() const;
    std::uint64_t preview_position_key(int from_pos, int to_pos) const;
    std::uint64_t full_state_key() const;
    void set_draw(const char* reason);
    // 游戏状态
    JunqiState state_;
    
    // 布阵校验器
    SetupValidator validator_;
    
    // 移动生成器
    MoveGenerator move_generator_;
    
    // 布阵生成器
    SetupGenerator setup_generator_;
    
    // 战斗结算器（需要state的引用）
    std::unique_ptr<CombatResolver> combat_resolver_;

    std::unordered_map<std::uint64_t, int> repetition_counts_;
    int no_capture_ply_ = 0;
    int repetition_limit_ = 3;
    int no_capture_ply_limit_ = 200;
    int max_ply_ = 1000;
    std::string terminal_reason_ = "NONE";
};

} // namespace junqi

#endif // JUNQI_ENVIRONMENT_H
