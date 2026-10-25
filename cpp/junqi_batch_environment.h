#ifndef JUNQI_BATCH_ENVIRONMENT_H
#define JUNQI_BATCH_ENVIRONMENT_H

#include "junqi_environment.h"
#include <cstdint>
#include <vector>

namespace junqi {

class BatchJunqiEnvironment {
public:
    explicit BatchJunqiEnvironment(int num_envs);

    int size() const { return static_cast<int>(envs_.size()); }

    void reset_random_all();
    void reset_indices(const std::vector<int>& indices);
    std::vector<int> get_current_players() const;
    std::vector<int> get_turn_counts() const;
    std::vector<int> get_winners() const;
    std::vector<std::uint8_t> get_done_flags() const;
    std::vector<int> get_perfect_states_batch() const;

    std::vector<float> get_feature_planes_batch(const std::vector<int>& player_ids);
    std::vector<std::uint8_t> get_legal_action_masks_batch(const std::vector<int>& player_ids);
    std::vector<float> get_material_scores_batch(const std::vector<int>& player_ids) const;
    std::vector<int> step_actions(const std::vector<int>& action_indices);
    std::vector<int> step_indexed_actions(const std::vector<int>& indices, const std::vector<int>& action_indices);

    void fill_perfect_states_batch(int* out) const;
    void fill_feature_planes_batch(float* out, const std::vector<int>& player_ids);
    void fill_legal_action_masks_batch(std::uint8_t* out, const std::vector<int>& player_ids);

private:
    std::vector<JunqiEnvironment> envs_;
};

}  // namespace junqi

#endif
