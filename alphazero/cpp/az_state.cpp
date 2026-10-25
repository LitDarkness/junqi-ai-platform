#include "az_state.h"
#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace junqi::az {

AZState::AZState() : env_(std::make_unique<JunqiEnvironment>()) {}

AZState::AZState(const JunqiEnvironment& env)
    : env_(std::make_unique<JunqiEnvironment>(env)) {}

AZState::AZState(const AZState& other)
    : env_(std::make_unique<JunqiEnvironment>(*other.env_)),
      initial_board_(other.initial_board_) {
    undo_.reserve(other.undo_.size());
    for (const auto& snapshot : other.undo_) {
        undo_.push_back(std::make_unique<JunqiEnvironment>(*snapshot));
    }
}

AZState& AZState::operator=(const AZState& other) {
    if (this == &other) return *this;
    auto replacement = std::make_unique<JunqiEnvironment>(*other.env_);
    std::vector<std::unique_ptr<JunqiEnvironment>> history;
    history.reserve(other.undo_.size());
    for (const auto& snapshot : other.undo_) {
        history.push_back(std::make_unique<JunqiEnvironment>(*snapshot));
    }
    env_ = std::move(replacement);
    undo_ = std::move(history);
    initial_board_ = other.initial_board_;
    return *this;
}

void AZState::reset(const std::vector<int>& red_setup, const std::vector<int>& black_setup) {
    env_->reset(red_setup, black_setup);
    initial_board_ = env_->get_perfect_state();
    undo_.clear();
}

void AZState::reset_random(std::uint64_t seed) {
    env_->seed(seed);
    env_->reset_random();
    initial_board_ = env_->get_perfect_state();
    undo_.clear();
}

void AZState::load_board_state(const std::vector<int>& board_state, int current_player) {
    env_->load_board_state(board_state, current_player);
    initial_board_ = board_state;
    undo_.clear();
}

CombatResult AZState::step_action(int action, bool save_undo) {
    if (action < 0 || action >= JunqiEnvironment::kActionSpaceSize) {
        throw std::out_of_range("action must be in [0, 3600)");
    }
    if (save_undo) {
        undo_.push_back(std::make_unique<JunqiEnvironment>(*env_));
    }
    const int raw_action = current_player() == BLACK ? flip_action(action) : action;
    const auto decoded = JunqiEnvironment::decode_action_index(raw_action);
    try {
        return env_->step(decoded.first, decoded.second);
    } catch (...) {
        if (save_undo) undo_.pop_back();
        throw;
    }
}

bool AZState::undo() {
    if (undo_.empty()) return false;
    env_ = std::move(undo_.back());
    undo_.pop_back();
    return true;
}

void AZState::clear_undo() { undo_.clear(); }

std::vector<int> AZState::legal_actions() const {
    std::vector<int> actions = env_->get_legal_actions(current_player());
    if (current_player() == BLACK) {
        for (int& action : actions) action = flip_action(action);
    }
    // Railway generation can reach the same destination along multiple paths.
    std::sort(actions.begin(), actions.end());
    actions.erase(std::unique(actions.begin(), actions.end()), actions.end());
    return actions;
}

std::vector<std::uint8_t> AZState::repetition_contexts() const {
    const std::vector<int> actions = legal_actions();
    return repetition_contexts(actions);
}

std::vector<std::uint8_t> AZState::repetition_contexts(
    const std::vector<int>& canonical_actions) const {
    std::vector<std::uint8_t> result;
    result.reserve(canonical_actions.size());
    for (int action : canonical_actions) {
        const int raw_action = current_player() == BLACK ? flip_action(action) : action;
        const auto decoded = JunqiEnvironment::decode_action_index(raw_action);
        result.push_back(
            env_->get_successor_repetition_context(decoded.first, decoded.second));
    }
    return result;
}

std::vector<float> AZState::features() const {
    std::vector<float> source = env_->get_feature_planes(current_player());
    std::vector<float> result(kFeatureChannels * BOARD_SIZE, 0.0f);
    for (int channel = 0; channel < JunqiEnvironment::kFeatureChannels; ++channel) {
        for (int raw_position = 0; raw_position < BOARD_SIZE; ++raw_position) {
            const int canonical_position = current_player() == BLACK
                ? flip_position(raw_position) : raw_position;
            result[channel * BOARD_SIZE + canonical_position] =
                source[channel * BOARD_SIZE + raw_position];
        }
    }

    static constexpr int piece_types[12] = {
        MINE, BOMB, FLAG, SAPPER, LIEUTENANT, CAPTAIN,
        MAJOR, COLONEL, BRIGADIER, GENERAL, ARMY_CMDR, FIELD_MARSHAL
    };
    const std::vector<int> canonical_board = board();
    for (int position = 0; position < BOARD_SIZE; ++position) {
        const int value = canonical_board[position];
        if (value == 0) continue;
        for (int type_index = 0; type_index < 12; ++type_index) {
            if (std::abs(value) == piece_types[type_index]) {
                const int channel = (value > 0 ? 14 : 26) + type_index;
                result[channel * BOARD_SIZE + position] = 1.0f;
                break;
            }
        }
    }

    static const JunqiBoard terrain;
    for (int raw_position = 0; raw_position < BOARD_SIZE; ++raw_position) {
        const int canonical_position = current_player() == BLACK
            ? flip_position(raw_position) : raw_position;
        if (terrain.is_railway(raw_position)) {
            result[38 * BOARD_SIZE + canonical_position] = 1.0f;
        }
        if (terrain.is_hq(raw_position)) {
            result[39 * BOARD_SIZE + canonical_position] = 1.0f;
        }
    }
    const float no_capture = std::min(1.0f, no_capture_ply() / 200.0f);
    const float repetitions = std::min(1.0f, repetition_count() / 3.0f);
    const float turns = std::min(1.0f, turn_count() / 1000.0f);
    std::fill(result.begin() + 40 * BOARD_SIZE, result.begin() + 41 * BOARD_SIZE, no_capture);
    std::fill(result.begin() + 41 * BOARD_SIZE, result.begin() + 42 * BOARD_SIZE, repetitions);
    std::fill(result.begin() + 42 * BOARD_SIZE, result.begin() + 43 * BOARD_SIZE, turns);

    // Explicit absolute side-to-move.  All spatial planes remain canonical,
    // but fixed-layout experiments still need to distinguish the opening side.
    const int raw_player = current_player();
    const int side_channel = raw_player == RED ? 43 : 44;
    std::fill(result.begin() + side_channel * BOARD_SIZE,
              result.begin() + (side_channel + 1) * BOARD_SIZE, 1.0f);

    // Initial layout in the same side-to-move canonical frame as the live
    // board: 12 own and 12 opponent piece-type planes.
    std::vector<int> canonical_initial(BOARD_SIZE, 0);
    if (initial_board_.size() == BOARD_SIZE) {
        for (int raw_position = 0; raw_position < BOARD_SIZE; ++raw_position) {
            const int canonical_position =
                raw_player == BLACK ? flip_position(raw_position) : raw_position;
            const int raw_value = initial_board_[raw_position];
            canonical_initial[canonical_position] =
                raw_player == BLACK ? -raw_value : raw_value;
        }
    }
    int initial_counts[2][12] = {};
    int current_counts[2][12] = {};
    for (int position = 0; position < BOARD_SIZE; ++position) {
        const int initial_value = canonical_initial[position];
        const int current_value = canonical_board[position];
        for (int type_index = 0; type_index < 12; ++type_index) {
            if (std::abs(initial_value) == piece_types[type_index]) {
                const int owner_index = initial_value > 0 ? 0 : 1;
                ++initial_counts[owner_index][type_index];
                const int channel = 45 + owner_index * 12 + type_index;
                result[channel * BOARD_SIZE + position] = 1.0f;
            }
            if (std::abs(current_value) == piece_types[type_index]) {
                const int owner_index = current_value > 0 ? 0 : 1;
                ++current_counts[owner_index][type_index];
            }
        }
    }

    // Broadcast normalized captured-piece counts.  These planes make exchange
    // history explicit even when two different games reach the same board.
    for (int owner_index = 0; owner_index < 2; ++owner_index) {
        for (int type_index = 0; type_index < 12; ++type_index) {
            const int initial_count = initial_counts[owner_index][type_index];
            const int dead_count =
                std::max(0, initial_count - current_counts[owner_index][type_index]);
            const float fraction = initial_count > 0
                ? static_cast<float>(dead_count) / static_cast<float>(initial_count)
                : 0.0f;
            const int channel = 69 + owner_index * 12 + type_index;
            std::fill(result.begin() + channel * BOARD_SIZE,
                      result.begin() + (channel + 1) * BOARD_SIZE, fraction);
        }
    }

    const auto flag_alive = [&](int owner_index) {
        return current_counts[owner_index][2] > 0 ? 1.0f : 0.0f;
    };
    std::fill(result.begin() + 93 * BOARD_SIZE,
              result.begin() + 94 * BOARD_SIZE, flag_alive(0));
    std::fill(result.begin() + 94 * BOARD_SIZE,
              result.begin() + 95 * BOARD_SIZE, flag_alive(1));
    std::fill(result.begin() + 95 * BOARD_SIZE,
              result.begin() + 96 * BOARD_SIZE,
              env_->is_field_marshal_dead(raw_player) ? 1.0f : 0.0f);
    std::fill(result.begin() + 96 * BOARD_SIZE,
              result.begin() + 97 * BOARD_SIZE,
              env_->is_field_marshal_dead(-raw_player) ? 1.0f : 0.0f);

    // Five most recent plies, newest first.  Each age uses from/to planes;
    // +1 denotes the current canonical side, -1 its opponent.
    const std::vector<MoveRecord> history = env_->get_game_history();
    const int available = std::min<int>(kRecentMoveCount, history.size());
    for (int age = 0; age < available; ++age) {
        const MoveRecord& move = history[history.size() - 1 - age];
        const int from = raw_player == BLACK
            ? flip_position(move.from_pos) : move.from_pos;
        const int to = raw_player == BLACK
            ? flip_position(move.to_pos) : move.to_pos;
        const float mover = move.player_id == raw_player ? 1.0f : -1.0f;
        result[(97 + age * 2) * BOARD_SIZE + from] = mover;
        result[(98 + age * 2) * BOARD_SIZE + to] = mover;
    }
    return result;
}

std::vector<int> AZState::board() const {
    std::vector<int> source = env_->get_perfect_state();
    if (current_player() != BLACK) return source;
    std::vector<int> canonical(BOARD_SIZE);
    for (int position = 0; position < BOARD_SIZE; ++position) {
        canonical[flip_position(position)] = -source[position];
    }
    return canonical;
}

int AZState::flip_action(int action) {
    if (action < 0 || action >= JunqiEnvironment::kActionSpaceSize) {
        throw std::out_of_range("action must be in [0, 3600)");
    }
    const auto decoded = JunqiEnvironment::decode_action_index(action);
    return JunqiEnvironment::get_action_index(
        flip_position(decoded.first), flip_position(decoded.second));
}

float AZState::terminal_value(int perspective) const {
    if (!terminal()) throw std::logic_error("terminal_value requested for non-terminal state");
    const int win = winner();
    if (win == 0) return 0.0f;
    if (perspective != 1 && perspective != -1) {
        throw std::invalid_argument("perspective must be RED (1) or BLACK (-1)");
    }
    return win == perspective ? 1.0f : -1.0f;
}

} // namespace junqi::az
