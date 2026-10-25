#include "junqi_batch_environment.h"

#include <stdexcept>
#include <algorithm>

namespace junqi {

BatchJunqiEnvironment::BatchJunqiEnvironment(int num_envs) {
    if (num_envs <= 0) {
        throw std::runtime_error("num_envs must be positive");
    }
    envs_.reserve(num_envs);
    for (int i = 0; i < num_envs; ++i) {
        envs_.emplace_back();
    }
}

void BatchJunqiEnvironment::reset_random_all() {
    const int count = static_cast<int>(envs_.size());
    for (int i = 0; i < count; ++i) {
        envs_[i].reset_random();
    }
}

void BatchJunqiEnvironment::reset_indices(const std::vector<int>& indices) {
    for (int index : indices) {
        if (index < 0 || index >= static_cast<int>(envs_.size())) {
            throw std::runtime_error("reset index out of range");
        }
        envs_[index].reset_random();
    }
}

std::vector<int> BatchJunqiEnvironment::get_current_players() const {
    std::vector<int> players(envs_.size());
    const int count = static_cast<int>(envs_.size());
    for (int i = 0; i < count; ++i) {
        players[i] = envs_[i].get_current_player();
    }
    return players;
}

std::vector<int> BatchJunqiEnvironment::get_turn_counts() const {
    std::vector<int> turn_counts(envs_.size());
    const int count = static_cast<int>(envs_.size());
    for (int i = 0; i < count; ++i) {
        turn_counts[i] = envs_[i].get_turn_count();
    }
    return turn_counts;
}

std::vector<int> BatchJunqiEnvironment::get_winners() const {
    std::vector<int> winners(envs_.size());
    const int count = static_cast<int>(envs_.size());
    for (int i = 0; i < count; ++i) {
        winners[i] = envs_[i].get_winner();
    }
    return winners;
}

std::vector<std::uint8_t> BatchJunqiEnvironment::get_done_flags() const {
    std::vector<std::uint8_t> done_flags(envs_.size());
    const int count = static_cast<int>(envs_.size());
    for (int i = 0; i < count; ++i) {
        done_flags[i] = envs_[i].is_game_over() ? 1 : 0;
    }
    return done_flags;
}

std::vector<int> BatchJunqiEnvironment::get_perfect_states_batch() const {
    std::vector<int> states(envs_.size() * BOARD_SIZE);
    fill_perfect_states_batch(states.data());
    return states;
}

void BatchJunqiEnvironment::fill_perfect_states_batch(int* out) const {
    const int count = static_cast<int>(envs_.size());
    for (int i = 0; i < count; ++i) {
        envs_[i].fill_perfect_state(out + i * BOARD_SIZE);
    }
}

std::vector<float> BatchJunqiEnvironment::get_feature_planes_batch(const std::vector<int>& player_ids) {
    if (player_ids.size() != envs_.size()) {
        throw std::runtime_error("player_ids size must match num_envs");
    }

    constexpr int feature_size = JunqiEnvironment::kFeatureChannels * BOARD_SIZE;
    std::vector<float> features(envs_.size() * feature_size);
    fill_feature_planes_batch(features.data(), player_ids);
    return features;
}

void BatchJunqiEnvironment::fill_feature_planes_batch(float* out, const std::vector<int>& player_ids) {
    if (player_ids.size() != envs_.size()) {
        throw std::runtime_error("player_ids size must match num_envs");
    }
    constexpr int feature_size = JunqiEnvironment::kFeatureChannels * BOARD_SIZE;
    const int count = static_cast<int>(envs_.size());
    for (int i = 0; i < count; ++i) {
        envs_[i].fill_feature_planes(out + i * feature_size, player_ids[i]);
    }
}

std::vector<std::uint8_t> BatchJunqiEnvironment::get_legal_action_masks_batch(const std::vector<int>& player_ids) {
    if (player_ids.size() != envs_.size()) {
        throw std::runtime_error("player_ids size must match num_envs");
    }

    const int action_size = JunqiEnvironment::kActionSpaceSize;
    std::vector<std::uint8_t> masks(envs_.size() * action_size);
    fill_legal_action_masks_batch(masks.data(), player_ids);
    return masks;
}

void BatchJunqiEnvironment::fill_legal_action_masks_batch(
    std::uint8_t* out,
    const std::vector<int>& player_ids
) {
    if (player_ids.size() != envs_.size()) {
        throw std::runtime_error("player_ids size must match num_envs");
    }
    const int action_size = JunqiEnvironment::kActionSpaceSize;
    const int count = static_cast<int>(envs_.size());
    for (int i = 0; i < count; ++i) {
        envs_[i].fill_legal_action_mask(out + i * action_size, player_ids[i]);
    }
}

std::vector<float> BatchJunqiEnvironment::get_material_scores_batch(const std::vector<int>& player_ids) const {
    if (player_ids.size() != envs_.size()) {
        throw std::runtime_error("player_ids size must match num_envs");
    }

    std::vector<float> scores(envs_.size());
    const int count = static_cast<int>(envs_.size());
    for (int i = 0; i < count; ++i) {
        scores[i] = envs_[i].get_material_score(player_ids[i]);
    }
    return scores;
}

std::vector<int> BatchJunqiEnvironment::step_actions(const std::vector<int>& action_indices) {
    if (action_indices.size() != envs_.size()) {
        throw std::runtime_error("action_indices size must match num_envs");
    }

    std::vector<int> results(envs_.size());
    const int count = static_cast<int>(envs_.size());
    for (int i = 0; i < count; ++i) {
        int action = action_indices[i];
        auto move = JunqiEnvironment::decode_action_index(action);
        CombatResult result = envs_[i].step(move.first, move.second);
        results[i] = static_cast<int>(result);
    }
    return results;
}

std::vector<int> BatchJunqiEnvironment::step_indexed_actions(
    const std::vector<int>& indices,
    const std::vector<int>& action_indices
) {
    if (indices.size() != action_indices.size()) {
        throw std::runtime_error("indices and action_indices must have the same size");
    }

    std::vector<int> results(indices.size());
    const int count = static_cast<int>(indices.size());
    for (int i = 0; i < count; ++i) {
        int env_index = indices[i];
        if (env_index < 0 || env_index >= static_cast<int>(envs_.size())) {
            throw std::runtime_error("step index out of range");
        }
        auto move = JunqiEnvironment::decode_action_index(action_indices[i]);
        CombatResult result = envs_[env_index].step(move.first, move.second);
        results[i] = static_cast<int>(result);
    }
    return results;
}

}  // namespace junqi
