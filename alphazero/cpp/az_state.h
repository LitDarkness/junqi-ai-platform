#ifndef JUNQI_AZ_STATE_H
#define JUNQI_AZ_STATE_H

#include "junqi_environment.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace junqi::az {

// Search-friendly owner of a complete Junqi environment snapshot.
class AZState {
public:
    static constexpr int kFeatureChannels = 107;
    static constexpr int kRecentMoveCount = 5;

    AZState();
    AZState(const AZState& other);
    AZState& operator=(const AZState& other);
    AZState(AZState&&) noexcept = default;
    AZState& operator=(AZState&&) noexcept = default;

    void reset(const std::vector<int>& red_setup, const std::vector<int>& black_setup);
    void reset_random(std::uint64_t seed);
    void load_board_state(const std::vector<int>& board_state, int current_player);

    // All public actions/features use the side-to-move canonical coordinates.
    // BLACK is rotated 180 degrees so the network always sees itself as RED.
    CombatResult step_action(int action, bool save_undo = true);
    bool undo();
    void clear_undo();
    AZState clone() const { return AZState(*this); }

    std::vector<int> legal_actions() const;
    std::vector<std::uint8_t> repetition_contexts() const;
    std::vector<std::uint8_t> repetition_contexts(
        const std::vector<int>& canonical_actions) const;
    std::vector<float> features() const;
    std::vector<int> board() const;

    int current_player() const { return env_->get_current_player(); }
    int opponent(int player = 0) const { return -(player == 0 ? current_player() : player); }
    int turn_count() const { return env_->get_turn_count(); }
    bool terminal() const { return env_->is_game_over(); }
    int winner() const { return env_->get_winner(); }
    std::string terminal_reason() const { return env_->get_terminal_reason(); }
    std::uint64_t key() const { return env_->get_state_key(); }
    std::uint64_t position_key() const { return env_->get_position_key(); }
    int no_capture_ply() const { return env_->get_no_capture_ply(); }
    int repetition_count() const { return env_->get_current_repetition_count(); }
    std::size_t history_size() const { return env_->get_history_size(); }
    std::size_t undo_depth() const { return undo_.size(); }
    float terminal_value(int perspective) const;

    static int flip_position(int position) { return BOARD_SIZE - 1 - position; }
    static int flip_action(int action);

private:
    explicit AZState(const JunqiEnvironment& env);
    std::unique_ptr<JunqiEnvironment> env_;
    std::vector<std::unique_ptr<JunqiEnvironment>> undo_;
    // Immutable episode context used by the policy/value encoder.  The rule
    // kernel owns the live board; this copy only describes where every piece
    // type started and therefore never affects legality or combat.
    std::vector<int> initial_board_;
};

} // namespace junqi::az

#endif
