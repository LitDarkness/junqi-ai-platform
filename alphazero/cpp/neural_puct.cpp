#include "neural_puct.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
namespace junqi::az {
NeuralPUCTTree::NeuralPUCTTree(const AZState& root, float cpuct, bool root_tactical_guard)
    : root_(std::make_unique<Node>(root)), cpuct_(cpuct), root_tactical_guard_(root_tactical_guard) {
    if (!(cpuct > 0.0f) || !std::isfinite(cpuct))
        throw std::invalid_argument("cpuct must be finite and positive");
}
bool NeuralPUCTTree::select_path(Node& node, PendingPath& path) {
    if (node.reserved) return false;
    path.nodes.push_back(&node);
    if (!node.expanded || node.state.terminal()) return true;
    const double scale = cpuct_ * std::sqrt(double(std::max(1, node.visits + node.virtual_visits)));
    if (node.virtual_visits == 0) {
        // Preserve the linear selection cost of sequential PUCT. Ranking all
        // edges is only needed when other leaves have reserved paths.
        Edge* best = nullptr;
        double best_score = -std::numeric_limits<double>::infinity();
        for (Edge& edge : node.edges) {
            if (edge.disabled) continue;
            const double q = edge.visits ? edge.value_sum / edge.visits : 0.0;
            const double score = q + scale * edge.prior / (1 + edge.visits);
            if (!best || score > best_score || (score == best_score && edge.action < best->action)) {
                best = &edge;
                best_score = score;
            }
        }
        if (best) {
            if (!best->child) {
                AZState child = node.state.clone();
                child.step_action(best->action, false);
                child.clear_undo();
                best->child = std::make_unique<Node>(child);
            }
            path.edges.push_back(best);
            return select_path(*best->child, path);
        }
        path.nodes.pop_back();
        return false;
    }
    std::vector<std::pair<double, Edge*>> ranked;
    ranked.reserve(node.edges.size());
    for (Edge& edge : node.edges) {
        if (edge.disabled) continue;
        const int visits = edge.visits + edge.virtual_visits;
        const double q = visits ? (edge.value_sum - edge.virtual_visits) / visits : 0.0;
        ranked.push_back({q + scale * edge.prior / (1 + visits), &edge});
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first > b.first : a.second->action < b.second->action;
    });
    for (auto& candidate : ranked) {
        Edge& edge = *candidate.second;
        if (!edge.child) {
            AZState child = node.state.clone();
            child.step_action(edge.action, false);
            child.clear_undo();
            edge.child = std::make_unique<Node>(child);
        }
        path.edges.push_back(&edge);
        if (select_path(*edge.child, path)) return true;
        path.edges.pop_back();
    }
    path.nodes.pop_back();
    return false;
}
PUCTLeafRequest NeuralPUCTTree::request_for(const Node& node) const {
    PUCTLeafRequest request;
    request.terminal = node.state.terminal();
    request.state_key = node.state.key();
    request.position_key = node.state.position_key();
    request.player = node.state.current_player();
    if (request.terminal) request.terminal_value = node.state.terminal_value(request.player);
    else {
        request.features = node.state.features();
        request.legal_actions = node.state.legal_actions();
        request.repetition_contexts = node.state.repetition_contexts(request.legal_actions);
    }
    return request;
}
std::vector<PUCTLeafRequest> NeuralPUCTTree::select_leaves(int count) {
    if (has_pending_leaf()) throw std::logic_error("backup or cancellation is required before selection");
    if (count < 1 || count > 256) throw std::invalid_argument("leaf batch must be in [1,256]");
    std::vector<PUCTLeafRequest> requests;
    for (int index = 0; index < count; ++index) {
        PendingPath path;
        if (!select_path(*root_, path)) break;
        Node* leaf = path.nodes.back();
        leaf->reserved = true;
        for (Node* node : path.nodes) ++node->virtual_visits;
        for (Edge* edge : path.edges) ++edge->virtual_visits;
        pending_paths_.push_back(std::move(path));
        requests.push_back(request_for(*leaf));
    }
    return requests;
}
PUCTLeafRequest NeuralPUCTTree::select_leaf() { return select_leaves(1).at(0); }
void NeuralPUCTTree::cancel_pending() {
    for (const PendingPath& path : pending_paths_) {
        path.nodes.back()->reserved = false;
        for (Node* node : path.nodes) --node->virtual_visits;
        for (Edge* edge : path.edges) --edge->virtual_visits;
    }
    pending_paths_.clear();
}
void NeuralPUCTTree::finish_backup(const PendingPath& path, float leaf_value) {
    const int leaf_player = path.nodes.back()->state.current_player();
    path.nodes.back()->reserved = false;
    for (Node* node : path.nodes) {
        --node->virtual_visits;
        ++node->visits;
        node->value_sum += node->state.current_player() == leaf_player ? leaf_value : -leaf_value;
    }
    for (std::size_t index = 0; index < path.edges.size(); ++index) {
        Edge* edge = path.edges[index];
        --edge->virtual_visits;
        ++edge->visits;
        edge->value_sum += path.nodes[index]->state.current_player() == leaf_player ? leaf_value : -leaf_value;
    }
}
void NeuralPUCTTree::backup_leaves(const std::vector<std::vector<float>>& priors,
                                 const std::vector<float>& values) {
    if (pending_paths_.empty()) throw std::logic_error("no pending PUCT leaves");
    if (priors.size() != pending_paths_.size() || values.size() != pending_paths_.size())
        throw std::invalid_argument("backup count must equal pending leaf count");
    std::vector<std::vector<int>> legals;
    std::vector<double> sums;
    // Validate the entire batch before changing visits or expanding a node.
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i]) || values[i] < -1.0001f || values[i] > 1.0001f)
            throw std::invalid_argument("leaf value must be finite and in [-1,1]");
        Node* leaf = pending_paths_[i].nodes.back();
        legals.push_back(leaf->state.terminal() ? std::vector<int>{} : leaf->state.legal_actions());
        if (priors[i].size() != legals.back().size())
            throw std::invalid_argument("prior count must equal legal action count");
        double sum = 0.0;
        for (float prior : priors[i]) {
            if (!std::isfinite(prior) || prior < 0.0f)
                throw std::invalid_argument("priors must be finite and non-negative");
            sum += prior;
        }
        if (!leaf->state.terminal() && !(sum > 0.0))
            throw std::invalid_argument("at least one prior must be positive");
        sums.push_back(sum);
    }
    for (std::size_t i = 0; i < values.size(); ++i) {
        Node* leaf = pending_paths_[i].nodes.back();
        if (!leaf->state.terminal()) {
            leaf->edges.clear();
            leaf->edges.reserve(legals[i].size());
            for (std::size_t j = 0; j < legals[i].size(); ++j) {
                Edge edge;
                edge.action = legals[i][j];
                edge.prior = float(priors[i][j] / sums[i]);
                leaf->edges.push_back(std::move(edge));
            }
            leaf->expanded = true;
            if (leaf == root_.get() && root_tactical_guard_) guard_root();
        }
        finish_backup(pending_paths_[i], values[i]);
    }
    pending_paths_.clear();
}
void NeuralPUCTTree::backup(const std::vector<float>& priors, float value) {
    if (pending_paths_.size() != 1)
        throw std::logic_error("single backup requires exactly one pending leaf");
    backup_leaves({priors}, {value});
}
bool NeuralPUCTTree::advance_root(int action) {
    if (has_pending_leaf()) throw std::logic_error("cannot advance with pending leaves");
    if (root_tactical_guard_) throw std::logic_error("tactical guarded roots cannot be reused");
    for (Edge& edge : root_->edges) {
        if (edge.action != action) continue;
        if (!edge.child) {
            AZState child = root_->state.clone();
            child.step_action(action, false);
            child.clear_undo();
            edge.child = std::make_unique<Node>(child);
        }
        auto next = std::move(edge.child);
        root_ = std::move(next);
        return true;
    }
    return false;
}
void NeuralPUCTTree::add_root_dirichlet(float alpha, float epsilon, std::uint64_t seed) {
    if (has_pending_leaf()) throw std::logic_error("cannot add noise with pending leaves");
    if (!root_->expanded || root_->edges.empty()) throw std::logic_error("root must be expanded before adding noise");
    if (!std::isfinite(alpha) || !std::isfinite(epsilon) || !(alpha > 0.0f) || epsilon < 0.0f || epsilon > 1.0f)
        throw std::invalid_argument("invalid Dirichlet parameters");
    std::mt19937_64 generator(seed);
    std::gamma_distribution<double> gamma(alpha, 1.0);
    std::vector<double> noise(root_->edges.size(), 0.0);
    for (std::size_t i = 0; i < noise.size(); ++i)
        if (!root_->edges[i].disabled) noise[i] = gamma(generator);
    const double total = std::accumulate(noise.begin(), noise.end(), 0.0);
    for (std::size_t index = 0; index < root_->edges.size(); ++index)
        root_->edges[index].prior = root_->edges[index].disabled ? 0.0f : float((1.0 - epsilon) * root_->edges[index].prior +
                                       epsilon * noise[index] / std::max(total, 1e-30));
}
void NeuralPUCTTree::guard_root() {
    // Only formal native outcomes are used. No heuristic evaluation or draw penalty.
    ++tactical_stats_[0];
    const int player = root_->state.current_player();
    std::vector<bool> wins(root_->edges.size(), false), safe(root_->edges.size(), false);
    int win_count = 0, safe_count = 0;
    for (std::size_t i = 0; i < root_->edges.size(); ++i) {
        AZState child = root_->state.clone();
        child.step_action(root_->edges[i].action, false); child.clear_undo();
        ++tactical_stats_[1];
        if (child.terminal()) {
            const float value = child.terminal_value(player);
            wins[i] = value > 0.0f;
            safe[i] = value >= 0.0f;
        } else {
            safe[i] = true;
            const auto board = child.board();
            for (int reply : child.legal_actions()) {
                if (board[reply % 60] != -3) continue;
                AZState end = child.clone(); end.step_action(reply, false); end.clear_undo();
                ++tactical_stats_[1];
                if (end.terminal() && end.terminal_value(player) < 0.0f) {
                    safe[i] = false; break;
                }
            }
        }
        win_count += int(wins[i]); safe_count += int(safe[i]);
    }
    tactical_stats_[2] += win_count;
    double sum = 0.0;
    int allowed_count = 0;
    for (std::size_t i = 0; i < root_->edges.size(); ++i) {
        Edge& edge = root_->edges[i];
        edge.disabled = win_count ? !wins[i] : (safe_count ? !safe[i] : false);
        if (edge.disabled) { edge.prior = 0.0f; ++tactical_stats_[3]; }
        else { sum += edge.prior; ++allowed_count; }
    }
    for (Edge& edge : root_->edges) if (!edge.disabled)
        edge.prior = sum > 0.0 ? float(edge.prior / sum) : float(1.0 / allowed_count);
}
std::vector<PUCTRootEdge> NeuralPUCTTree::root_edges() const {
    std::vector<PUCTRootEdge> result;
    for (const Edge& edge : root_->edges)
        result.push_back({edge.action, edge.prior, edge.visits,
                         edge.visits ? float(edge.value_sum / edge.visits) : 0.0f});
    return result;
}
int NeuralPUCTTree::root_visits() const { return root_->visits; }
}  // namespace junqi::az
