#ifndef JUNQI_NEURAL_PUCT_H
#define JUNQI_NEURAL_PUCT_H

#include "az_state.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace junqi::az {

struct PUCTLeafRequest {
    bool terminal = false;
    float terminal_value = 0.0f;
    std::uint64_t state_key = 0;
    std::uint64_t position_key = 0;
    int player = 0;
    std::vector<float> features;
    std::vector<int> legal_actions;
    std::vector<std::uint8_t> repetition_contexts;
};

struct PUCTRootEdge {
    int action = -1;
    float prior = 0.0f;
    int visits = 0;
    float q = 0.0f;
};

class NeuralPUCTTree {
public:
    explicit NeuralPUCTTree(const AZState& root, float cpuct = 1.5f, bool root_tactical_guard = false);
    PUCTLeafRequest select_leaf();
    std::vector<PUCTLeafRequest> select_leaves(int count);
    void backup(const std::vector<float>& legal_priors, float leaf_value);
    void backup_leaves(const std::vector<std::vector<float>>& priors,
                       const std::vector<float>& values);
    void cancel_pending();
    bool advance_root(int action);
    std::uint64_t root_key() const { return root_->state.key(); }
    void add_root_dirichlet(float alpha, float epsilon, std::uint64_t seed);
    std::vector<PUCTRootEdge> root_edges() const;
    int root_visits() const;
    bool has_pending_leaf() const { return !pending_paths_.empty(); }
    std::vector<int> root_tactical_stats() const { return tactical_stats_; }

private:
    struct Node;
    struct Edge {
        int action = -1;
        float prior = 0.0f;
        int visits = 0;
        int virtual_visits = 0;
        double value_sum = 0.0;
        bool disabled = false;
        std::unique_ptr<Node> child;
    };
    struct Node {
        explicit Node(const AZState& source) : state(source) {}
        AZState state;
        bool expanded = false;
        int visits = 0;
        int virtual_visits = 0;
        bool reserved = false;
        double value_sum = 0.0;
        std::vector<Edge> edges;
    };

    struct PendingPath {
        std::vector<Node*> nodes;
        std::vector<Edge*> edges;
    };
    bool select_path(Node& node, PendingPath& path);
    PUCTLeafRequest request_for(const Node& node) const;
    void finish_backup(const PendingPath& path, float leaf_value);
    void guard_root();

    std::unique_ptr<Node> root_;
    float cpuct_;
    bool root_tactical_guard_;
    // [roots examined, native transitions, immediate wins, disabled actions]
    std::vector<int> tactical_stats_ = {0, 0, 0, 0};
    std::vector<PendingPath> pending_paths_;
};

}  // namespace junqi::az

#endif
