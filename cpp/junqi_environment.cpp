#include "junqi_environment.h"
#include <cstring>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <iostream>

namespace junqi {

namespace {

float piece_material_value(PieceType type) {
    switch (type) {
        case FIELD_MARSHAL: return 9.0f;
        case ARMY_CMDR: return 8.0f;
        case GENERAL: return 7.0f;
        case BRIGADIER: return 6.0f;
        case COLONEL: return 5.0f;
        case MAJOR: return 4.0f;
        case CAPTAIN: return 3.0f;
        case LIEUTENANT: return 2.0f;
        case SAPPER: return 5.0f;
        case BOMB: return 8.0f;
        case MINE: return 8.0f;
        case FLAG: return 15.0f;
        default: return 0.0f;
    }
}

int feature_channel_for_piece(PieceType type) {
    if (type == FIELD_MARSHAL || type == ARMY_CMDR || type == GENERAL) {
        return 0;
    }
    if (type == BRIGADIER || type == COLONEL || type == MAJOR) {
        return 1;
    }
    if (type == CAPTAIN || type == LIEUTENANT || type == SAPPER) {
        return 2;
    }
    if (type == BOMB) {
        return 3;
    }
    if (type == MINE) {
        return 4;
    }
    if (type == FLAG) {
        return 5;
    }
    return -1;
}

}  // namespace

// ==================== 位置转换函数 ====================
int JunqiEnvironment::pos_to_index(int row, int col) {
    return row * BOARD_COLS + col;
}

std::pair<int, int> JunqiEnvironment::index_to_pos(int index) {
    return std::make_pair(index / BOARD_COLS, index % BOARD_COLS);
}

int JunqiEnvironment::get_action_index(int from, int to) {
    return from * BOARD_SIZE + to;
}

std::pair<int, int> JunqiEnvironment::decode_action_index(int action_index) {
    return std::make_pair(action_index / BOARD_SIZE, action_index % BOARD_SIZE);
}

JunqiEnvironment::JunqiEnvironment()
    : validator_(state_.get_board())
    , move_generator_(state_.get_board())
    , setup_generator_(state_.get_board()) {
    combat_resolver_ = std::make_unique<CombatResolver>(state_);
}

void JunqiEnvironment::reset(const std::vector<int>& red_setup,
                               const std::vector<int>& black_setup) {
    // 验证布阵
    std::string error_message;
    if (!validator_.validate_both_setups(red_setup, black_setup, error_message)) {
        throw std::runtime_error("Invalid setup: " + error_message);
    }
    
    // 重置游戏状态
    state_.reset(red_setup, black_setup);
    reset_draw_tracking();
    
    // 重新创建战斗结算器（因为state已经重置）
    combat_resolver_ = std::make_unique<CombatResolver>(state_);
}

void JunqiEnvironment::reset_random() {
    // 生成随机布阵
    std::vector<int> red_setup = setup_generator_.generate_random_setup(RED);
    std::vector<int> black_setup = setup_generator_.generate_random_setup(BLACK);
    
    // 重置游戏
    reset(red_setup, black_setup);
}

CombatResult JunqiEnvironment::step(int from_pos, int to_pos) {
    // 检查游戏是否已经结束
    if (state_.is_game_over()) {
        throw std::runtime_error("Game is already over");
    }
    
    // 检查移动是否合法
    if (!move_generator_.is_move_valid(state_, from_pos, to_pos)) {
        throw std::runtime_error("Invalid move");
    }
    
    // 记录当前玩家
    PlayerId current_player = state_.get_current_player();
    
    const Piece& to_piece = state_.get_piece(to_pos);
    
    CombatResult result;
    
    // 检查目标位置是否是空位
    if (to_piece.is_empty()) {
        // 空位：直接移动，不触发战斗
        state_.move_piece(from_pos, to_pos);
        state_.add_move_to_history(from_pos, to_pos);
        state_.switch_turn();
        state_.increment_turn_count();
        result = static_cast<CombatResult>(MOVE_ONLY); // 使用MOVE_ONLY表示成功移动到空位
    } else {
        // 有敌方棋子：触发战斗
        state_.add_move_to_history(from_pos, to_pos);
        result = combat_resolver_->resolve_combat(from_pos, to_pos);
        if (state_.is_game_over()) {
            // CombatResolver returns early on a flag capture.
            state_.increment_turn_count();
        }
    }
    
    // 检查是否需要揭示军旗（司令死亡）
    check_reveal_flag(result, current_player);
    
    // 检查当前玩家是否有可移动的棋子
    PlayerId new_current_player = state_.get_current_player();
    if (!state_.is_game_over() && !combat_resolver_->has_movable_pieces(new_current_player)) {
        state_.set_game_over(true);
        state_.set_winner(opposite_player(new_current_player));
    }
    
    if (state_.is_game_over() && terminal_reason_ == "NONE") {
        terminal_reason_ = (result == GAME_OVER_WIN || result == GAME_OVER_LOSS)
            ? "FLAG_CAPTURE"
            : "NO_LEGAL_MOVE";
    }
    if (!state_.is_game_over()) {
        update_draw_tracking(result != MOVE_ONLY);
    }
    return result;
}

std::vector<int> JunqiEnvironment::get_perfect_state() const {
    std::vector<int> result(BOARD_SIZE);
    fill_perfect_state(result.data());
    return result;
}

void JunqiEnvironment::fill_perfect_state(int* out) const {
    for (int i = 0; i < BOARD_SIZE; ++i) {
        const Piece& piece = state_.get_piece(i);
        out[i] = piece.is_empty() ? EMPTY :
            (piece.owner == RED ? static_cast<int>(piece.type) : -static_cast<int>(piece.type));
    }
}

std::vector<int> JunqiEnvironment::get_observation(int player_id) {
    // The SRS default profile is full-information (open Junqi).
    (void)player_id;
    return state_.get_board_state();
    std::vector<int> observation(BOARD_SIZE);
    PlayerId player = static_cast<PlayerId>(player_id);
    
    for (int i = 0; i < BOARD_SIZE; ++i) {
        const Piece& piece = state_.get_piece(i);
        
        if (piece.is_empty()) {
            observation[i] = EMPTY;
        } else if (piece.is_owned_by(player)) {
            // 己方棋子：显示真实类型
            observation[i] = static_cast<int>(piece.type);
        } else {
            // 敌方棋子
            if (piece.type == FLAG && state_.is_flag_revealed(piece.owner)) {
                // 军旗已被揭示
                observation[i] = FLAG;
            } else {
                // 暗棋：显示UNKNOWN
                observation[i] = UNKNOWN;
            }
        }
    }
    
    return observation;
}

std::vector<bool> JunqiEnvironment::get_legal_action_mask(int player_id) {
    PlayerId player = static_cast<PlayerId>(player_id);
    
    // 如果游戏已经结束，返回全false
    if (state_.is_game_over()) {
        return std::vector<bool>(BOARD_SIZE * BOARD_SIZE, false);
    }
    
    // 如果不是当前玩家的回合，返回全false
    if (state_.get_current_player() != player) {
        return std::vector<bool>(BOARD_SIZE * BOARD_SIZE, false);
    }
    
    return move_generator_.get_legal_action_mask(state_, player);
}

void JunqiEnvironment::fill_legal_action_mask(std::uint8_t* out, int player_id) {
    const PlayerId player = static_cast<PlayerId>(player_id);
    if (state_.is_game_over() || state_.get_current_player() != player) {
        std::memset(out, 0, kActionSpaceSize * sizeof(std::uint8_t));
        return;
    }
    move_generator_.fill_legal_action_mask(state_, player, out);
}

std::vector<int> JunqiEnvironment::get_legal_actions(int player_id) {
    std::vector<int> actions;
    PlayerId player = static_cast<PlayerId>(player_id);
    if (state_.is_game_over() || state_.get_current_player() != player) {
        return actions;
    }

    const auto moves = move_generator_.generate_legal_moves(state_);
    actions.reserve(moves.size());
    for (const auto& move : moves) {
        actions.push_back(get_action_index(move.first, move.second));
    }
    return actions;
}

LegalityResult JunqiEnvironment::check_move(int from_pos, int to_pos) const {
    auto invalid = [](const char* code, const char* message_key) {
        LegalityResult result;
        result.code = code;
        result.message_key = message_key;
        return result;
    };

    if (state_.is_game_over()) {
        return invalid("GAME_ALREADY_FINISHED", "move.game_already_finished");
    }
    if (from_pos < 0 || from_pos >= BOARD_SIZE || to_pos < 0 || to_pos >= BOARD_SIZE) {
        return invalid("SQUARE_OUT_OF_RANGE", "move.square_out_of_range");
    }
    if (from_pos == to_pos) {
        return invalid("SAME_SOURCE_DESTINATION", "move.same_source_destination");
    }

    const Piece& source = state_.get_piece(from_pos);
    const Piece& target = state_.get_piece(to_pos);
    if (source.is_empty()) {
        return invalid("NO_PIECE_AT_SOURCE", "move.no_piece_at_source");
    }
    if (!source.is_owned_by(state_.get_current_player())) {
        return invalid("NOT_SIDE_TO_MOVE", "move.not_side_to_move");
    }
    if (!is_movable_piece(source.type)) {
        return invalid("PIECE_IMMOBILE", "move.piece_immobile");
    }
    if (state_.get_board().is_hq(from_pos)) {
        return invalid("HEADQUARTERS_EXIT_FORBIDDEN", "move.headquarters_exit_forbidden");
    }
    if (target.is_owned_by(source.owner)) {
        return invalid("DEST_OWN_OCCUPIED", "move.dest_own_occupied");
    }
    if (!target.is_empty() && state_.get_board().is_camp(to_pos)) {
        return invalid("CAMP_ATTACK_FORBIDDEN", "move.camp_attack_forbidden");
    }
    if (!move_generator_.is_move_valid(state_, from_pos, to_pos)) {
        return invalid("NO_BOARD_EDGE", "move.no_board_edge");
    }

    LegalityResult result;
    result.legal = true;
    result.code = "OK";
    result.message_key = "move.ok";
    return result;
}

std::vector<float> JunqiEnvironment::get_feature_planes(int player_id) {
    std::vector<float> planes(kFeatureChannels * BOARD_SIZE);
    fill_feature_planes(planes.data(), player_id);
    return planes;
}

void JunqiEnvironment::fill_feature_planes(float* planes, int player_id) {
    std::fill(planes, planes + kFeatureChannels * BOARD_SIZE, 0.0f);
    PlayerId player = static_cast<PlayerId>(player_id);

    static constexpr int camps[] = {
        11, 13, 17, 21, 23,
        36, 38, 42, 46, 48
    };
    for (int pos : camps) {
        planes[13 * BOARD_SIZE + pos] = 1.0f;
    }

    for (int i = 0; i < BOARD_SIZE; ++i) {
        const Piece& piece = state_.get_piece(i);
        if (piece.is_empty()) {
            continue;
        }

        int piece_channel = feature_channel_for_piece(piece.type);
        if (piece_channel < 0) {
            continue;
        }

        int channel = (piece.owner == player) ? piece_channel : piece_channel + 6;
        planes[channel * BOARD_SIZE + i] = 1.0f;
    }

}

float JunqiEnvironment::get_material_score(int player_id) const {
    PlayerId player = static_cast<PlayerId>(player_id);
    float total = 0.0f;
    for (int i = 0; i < BOARD_SIZE; ++i) {
        const Piece& piece = state_.get_piece(i);
        if (piece.is_owned_by(player)) {
            total += piece_material_value(piece.type);
        }
    }
    return total;
}

bool JunqiEnvironment::is_field_marshal_dead(int player_id) const {
    PlayerId player = static_cast<PlayerId>(player_id);
    return state_.is_field_marshal_dead(player);
}

int JunqiEnvironment::get_turn_count() const {
    return state_.get_turn_count();
}

int JunqiEnvironment::get_current_player() const {
    return static_cast<int>(state_.get_current_player());
}

bool JunqiEnvironment::is_game_over() const {
    return state_.is_game_over();
}

int JunqiEnvironment::get_winner() const {
    return static_cast<int>(state_.get_winner());
}

bool JunqiEnvironment::set_red_setup(const std::vector<int>& setup,
                                       std::string& error_message) {
    return setup_generator_.set_custom_setup(setup, RED, error_message);
}

bool JunqiEnvironment::set_black_setup(const std::vector<int>& setup,
                                        std::string& error_message) {
    return setup_generator_.set_custom_setup(setup, BLACK, error_message);
}

const std::vector<int>& JunqiEnvironment::get_red_setup() {
    return setup_generator_.get_setup(RED);
}

const std::vector<int>& JunqiEnvironment::get_black_setup() {
    return setup_generator_.get_setup(BLACK);
}

std::vector<int> JunqiEnvironment::generate_red_setup() {
    return setup_generator_.generate_random_setup(RED);
}

std::vector<int> JunqiEnvironment::generate_black_setup() {
    return setup_generator_.generate_random_setup(BLACK);
}

void JunqiEnvironment::check_reveal_flag(CombatResult result, PlayerId player) {
    // 在战斗后，检查双方玩家是否都有司令
    // 如果某个玩家的司令已经死亡，就揭示该玩家的军旗
    
    PlayerId players[] = {RED, BLACK};
    
    for (PlayerId p : players) {
        // 检查该玩家是否有司令
        bool has_field_marshal = false;
        
        int start_row = (p == RED) ? RED_AREA_START : BLACK_AREA_START;
        int end_row = (p == RED) ? RED_AREA_END : BLACK_AREA_END;
        
        for (int row = start_row; row <= end_row; ++row) {
            for (int col = 0; col < BOARD_COLS; ++col) {
                const Piece& piece = state_.get_piece(row, col);
                
                if (piece.is_owned_by(p) && piece.type == FIELD_MARSHAL) {
                    has_field_marshal = true;
                    break;
                }
            }
            
            if (has_field_marshal) {
                break;
            }
        }
        
        // 如果司令已经死亡，设置状态
        if (!has_field_marshal) {
            state_.set_field_marshal_dead(p);
            // 揭示该玩家的军旗
            state_.set_flag_revealed(p);
        }
    }
}

// ==================== 棋子信息查询 ====================
PieceInfo JunqiEnvironment::get_piece_at(int pos) const {
    const Piece& piece = state_.get_piece(pos);
    PieceInfo info;
    info.type = static_cast<int>(piece.type);
    info.owner = static_cast<int>(piece.owner);
    return info;
}

std::vector<int> JunqiEnvironment::get_player_pieces(int player_id) const {
    PlayerId player = static_cast<PlayerId>(player_id);
    std::vector<int> positions;
    
    for (int i = 0; i < BOARD_SIZE; ++i) {
        const Piece& piece = state_.get_piece(i);
        if (piece.is_owned_by(player)) {
            positions.push_back(i);
        }
    }
    
    return positions;
}

int JunqiEnvironment::get_piece_count(int player_id, int piece_type) const {
    PlayerId player = static_cast<PlayerId>(player_id);
    PieceType type = static_cast<PieceType>(piece_type);
    int count = 0;
    
    for (int i = 0; i < BOARD_SIZE; ++i) {
        const Piece& piece = state_.get_piece(i);
        if (piece.is_owned_by(player) && piece.type == type) {
            ++count;
        }
    }
    
    return count;
}

// ==================== 移动验证 ====================
bool JunqiEnvironment::can_move(int from_pos, int to_pos) const {
    return move_generator_.is_move_valid(state_, from_pos, to_pos);
}

std::vector<int> JunqiEnvironment::get_possible_moves(int from_pos) const {
    return move_generator_.get_possible_moves(state_, from_pos);
}

std::vector<int> JunqiEnvironment::get_movable_pieces(int player_id) const {
    PlayerId player = static_cast<PlayerId>(player_id);
    std::vector<int> movable_pieces;
    
    for (int from = 0; from < BOARD_SIZE; ++from) {
        const Piece& piece = state_.get_piece(from);
        if (!piece.is_owned_by(player) || piece.is_empty()) {
            continue;
        }
        
        // 检查是否有可移动的位置
        std::vector<int> possible_moves = move_generator_.get_possible_moves(state_, from);
        if (!possible_moves.empty()) {
            movable_pieces.push_back(from);
        }
    }
    
    return movable_pieces;
}

// ==================== 棋盘渲染 ====================
std::string JunqiEnvironment::render() const {
    std::ostringstream oss;
    oss << "\n";
    oss << "   0  1  2  3  4\n";
    
    for (int row = BOARD_ROWS - 1; row >= 0; --row) {
        oss << std::setw(2) << row << ": ";
        for (int col = 0; col < BOARD_COLS; ++col) {
            int pos = row * BOARD_COLS + col;
            const Piece& piece = state_.get_piece(pos);
            
            if (piece.is_empty()) {
                oss << " .  ";
            } else if (piece.type == UNKNOWN) {
                oss << " ?  ";
            } else {
                int val = static_cast<int>(piece.type);
                if (piece.owner == BLACK) {
                    val = -val;
                }
                oss << std::setw(3) << val << " ";
            }
        }
        oss << "\n";
    }
    
    oss << "\n图例: . = 空位, ? = 暗棋, 正数 = 红方, 负数 = 黑方\n";
    oss << "      1=地雷 2=炸弹 3=军旗 10=工兵 20=排长 30=连长\n";
    oss << "      40=营长 50=团长 60=旅长 70=师长 80=军长 90=司令\n";
    
    return oss.str();
}

void JunqiEnvironment::print_board() const {
    std::cout << render();
}

// ==================== 游戏历史 ====================
MoveRecord JunqiEnvironment::get_last_move() const {
    const auto& history = state_.get_move_history();
    if (history.empty()) {
        MoveRecord record = { -1, -1, 0, 0 };
        return record;
    }
    
    const auto& last = history.back();
    MoveRecord record;
    record.from_pos = last.first;
    record.to_pos = last.second;
    record.player_id = static_cast<int>(state_.get_current_player());
    record.result = 0; // 简化处理
    return record;
}

std::vector<MoveRecord> JunqiEnvironment::get_game_history() const {
    const auto& history = state_.get_move_history();
    std::vector<MoveRecord> records;
    records.reserve(history.size());
    
    for (const auto& move : history) {
        MoveRecord record;
        record.from_pos = move.first;
        record.to_pos = move.second;
        record.player_id = 0; // 需要追踪
        record.result = 0;
        records.push_back(record);
    }
    
    return records;
}

void JunqiEnvironment::clear_history() {
    state_.clear_history();
}

// ==================== 从给定局面加载 ====================
void JunqiEnvironment::load_board_state(const std::vector<int>& board_state, int current_player) {
    state_.reset(std::vector<int>(30, EMPTY), std::vector<int>(30, EMPTY));

    // 设置棋盘状态
    for (int i = 0; i < BOARD_SIZE && i < static_cast<int>(board_state.size()); ++i) {
        int val = board_state[i];
        Piece piece;
        
        if (val == 0) {
            // 空位
            piece = Piece(EMPTY, NONE);
        } else if (val > 0) {
            // 红方棋子（正数）
            piece = Piece(static_cast<PieceType>(val), RED);
        } else {
            // 黑方棋子（负数）
            piece = Piece(static_cast<PieceType>(-val), BLACK);
        }
        
        state_.set_piece(i, piece);
    }
    
    // 设置当前玩家
    if (current_player == 1) {
        state_.set_current_player(RED);
    } else if (current_player == -1) {
        state_.set_current_player(BLACK);
    }
    
    // 重置其他状态
    state_.set_game_over(false);
    state_.set_winner(NONE);
    state_.clear_history();
    reset_draw_tracking();
}


std::uint64_t JunqiEnvironment::position_key() const {
    const std::vector<int> board = state_.get_board_state();
    std::uint64_t hash = 1469598103934665603ULL;
    for (int value : board) {
        hash ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(value));
        hash *= 1099511628211ULL;
    }
    hash ^= static_cast<std::uint64_t>(state_.get_current_player() + 2);
    hash *= 1099511628211ULL;
    return hash;
}

void JunqiEnvironment::reset_draw_tracking() {
    no_capture_ply_ = 0;
    terminal_reason_ = "NONE";
    repetition_counts_.clear();
    repetition_counts_[position_key()] = 1;
}

void JunqiEnvironment::set_draw(const char* reason) {
    state_.set_game_over(true);
    state_.set_winner(NONE);
    terminal_reason_ = reason;
}

void JunqiEnvironment::update_draw_tracking(bool capture) {
    no_capture_ply_ = capture ? 0 : no_capture_ply_ + 1;

    if (state_.get_turn_count() >= max_ply_) {
        set_draw("MAX_PLY");
        return;
    }
    if (no_capture_ply_limit_ > 0 && no_capture_ply_ >= no_capture_ply_limit_) {
        set_draw("NO_CAPTURE_LIMIT");
        return;
    }

    int& count = repetition_counts_[position_key()];
    ++count;
    if (repetition_limit_ > 0 && count >= repetition_limit_) {
        set_draw("REPETITION");
    }
}

} // namespace junqi
