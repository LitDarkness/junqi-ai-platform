#pragma once

#include <cstdint>
#include <utility>
#include <vector>

enum class JunqiHeuristicKind {
    RootOnly,
    EveryNode,
    WideTactical,          // v1.5/v1.6 self-play SOTA
    WideResourceSafety,    // v1.7.2: v2 + necessary-last-bomb root tie-break
    WideQuiescence,        // v1.7.3: v2 + bounded capture quiescence
    WideDrawAware,         // v1.8: qsearch + exact repetition/no-capture/max-ply draws
    WideStrategicDraw,      // v1.8 experimental: qsearch + bounded dominance + exact draw history
    WideTacticalLegacy,    // v1.4.0 policy, retained for direct comparison
    WideFullPly2,          // diagnostic: root, reply, and ply 2 full width
    WideFullAll,           // diagnostic: full width at every node
};

enum JunqiTacticalMask : std::uint32_t {
    TacticalNone = 0,
    TacticalFlagCapture = 1u << 0,
    TacticalFlagApproach = 1u << 1,
    TacticalFlagZoneCombat = 1u << 2,
    TacticalMineClear = 1u << 3,
    TacticalFlagDefense = 1u << 4,
    TacticalImmediateDefense = 1u << 5,
    TacticalLongRail = 1u << 6,
    TacticalBombDefense = 1u << 7,
    TacticalBombExposure = 1u << 8,
    TacticalResourceTieBreak = 1u << 9,
};


struct JunqiSearchHistory {
    int turn_count = 0;
    int no_capture_ply = 0;
    int repetition_limit = 3;
    int no_capture_ply_limit = 200;
    int max_ply = 1000;
    std::vector<std::pair<std::uint64_t, int>> repetition_counts;
};

struct JunqiRootCandidateInfo {
    int action = -1;
    double static_score = 0.0;
    double search_score = 0.0;
    bool has_search_score = false;
    int static_rank = 0;
    int search_rank = 0;
    bool retained = false;
    bool selected = false;
    std::uint32_t tactical_mask = TacticalNone;
};

struct JunqiRootPVInfo {
    int action = -1;
    std::vector<int> actions;
};

int heuristic_select_action(const std::vector<int>& board,
                            const std::vector<int>& legal_actions,
                            int player, int max_nodes, int max_depth,
                            JunqiHeuristicKind kind);

int heuristic_select_action_with_history(const std::vector<int>& board,
                                         const std::vector<int>& legal_actions,
                                         int player, int max_nodes, int max_depth,
                                         JunqiHeuristicKind kind,
                                         const JunqiSearchHistory& history);

int root_only_select_action(const std::vector<int>& board,
                            const std::vector<int>& legal_actions,
                            int player, int max_nodes, int max_depth);
int every_node_select_action(const std::vector<int>& board,
                             const std::vector<int>& legal_actions,
                             int player, int max_nodes, int max_depth);
int wide_tactical_select_action(const std::vector<int>& board,
                                const std::vector<int>& legal_actions,
                                int player, int max_nodes, int max_depth);

int root_only_last_nodes();
int root_only_last_depth();
bool root_only_last_budget_hit();
const std::vector<JunqiRootCandidateInfo>& heuristic_last_candidates();
const std::vector<JunqiRootPVInfo>& heuristic_last_pvs();
std::uint64_t heuristic_position_key(const std::vector<int>& board, int player);
