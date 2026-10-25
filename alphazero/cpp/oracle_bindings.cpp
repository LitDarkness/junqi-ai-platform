#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../supplement/src/root_only_runtime.h"

#include <chrono>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {

JunqiSearchHistory parse_history(const py::dict& source) {
    JunqiSearchHistory history;
    auto integer = [&](const char* key, int fallback) {
        return source.contains(key) ? py::cast<int>(source[key]) : fallback;
    };
    history.turn_count = integer("turn_count", 0);
    history.no_capture_ply = integer("no_capture_ply", 0);
    history.repetition_limit = integer("repetition_limit", 3);
    history.no_capture_ply_limit = integer("no_capture_ply_limit", 200);
    history.max_ply = integer("max_ply", 1000);
    if (source.contains("repetition_counts")) {
        history.repetition_counts =
            py::cast<std::vector<std::pair<std::uint64_t, int>>>(
                source["repetition_counts"]);
    }
    return history;
}

py::dict run_search(
    const std::vector<int>& board,
    const std::vector<int>& legal,
    int player,
    int max_nodes,
    int max_depth,
    const py::dict& history_payload) {
    if (player != 1 && player != -1)
        throw std::invalid_argument("player must be +1 or -1");
    const JunqiSearchHistory history = parse_history(history_payload);
    const auto started = std::chrono::steady_clock::now();
    int action;
    {
        py::gil_scoped_release release;
        action = heuristic_select_action_with_history(
            board, legal, player, max_nodes, max_depth,
            JunqiHeuristicKind::WideQuiescence, history);
    }
    const double milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
    py::list candidates;
    std::unordered_map<int, std::vector<int>> pvs;
    for (const auto& row : heuristic_last_pvs()) pvs[row.action] = row.actions;
    for (const auto& candidate : heuristic_last_candidates()) {
        py::dict row;
        row["action"] = candidate.action;
        row["static_score"] = candidate.static_score;
        row["search_score"] = candidate.search_score;
        row["has_search_score"] = candidate.has_search_score;
        row["static_rank"] = candidate.static_rank;
        row["search_rank"] = candidate.search_rank;
        row["retained"] = candidate.retained;
        row["selected"] = candidate.selected;
        row["tactical_mask"] = candidate.tactical_mask;
        const auto pv = pvs.find(candidate.action);
        row["pv"] = pv == pvs.end()
            ? std::vector<int>{candidate.action} : pv->second;
        candidates.append(std::move(row));
    }
    py::dict result;
    result["action"] = action;
    result["nodes"] = root_only_last_nodes();
    result["completed_depth"] = root_only_last_depth();
    result["budget_hit"] = root_only_last_budget_hit();
    result["milliseconds"] = milliseconds;
    result["candidates"] = std::move(candidates);
    return result;
}

}  // namespace

PYBIND11_MODULE(JunqiOracle, module) {
    module.doc() =
        "Independent supplement_wide_quiescence adapter primitives";
    module.attr("ORACLE_NAME") = "supplement_wide_quiescence";
    module.def(
        "position_key", &heuristic_position_key,
        py::arg("board"), py::arg("player") = 1);
    module.def(
        "search_once", &run_search,
        py::arg("board"), py::arg("legal_actions"),
        py::arg("player"), py::arg("max_nodes"),
        py::arg("max_depth"), py::arg("history"));
}
