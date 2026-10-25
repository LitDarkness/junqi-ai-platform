#include "root_only_runtime.h"
#include "env/junqi_environment.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using junqi::JunqiEnvironment;

static const std::vector<int> kPosition0 = {
20,3,1,1,70,1,2,40,30,10,20,0,50,0,50,60,90,0,20,30,30,0,2,0,10,40,60,80,10,70,
-20,-40,-20,-60,-60,-30,0,-30,0,-10,-2,-10,0,-40,-10,-70,0,-80,0,-50,-30,-1,-70,-50,-20,-1,-3,-1,-2,-90};

static std::uint64_t child_key(const std::vector<int>& board, int player, int action) {
    JunqiEnvironment env;
    env.load_board_state(board, player);
    auto [from,to] = JunqiEnvironment::decode_action_index(action);
    env.step(from,to);
    return env.get_state_key();
}

static JunqiSearchHistory base_history(const JunqiEnvironment& env) {
    JunqiSearchHistory h;
    h.turn_count = env.get_turn_count();
    h.no_capture_ply = env.get_no_capture_ply();
    h.repetition_limit = env.get_repetition_limit();
    h.no_capture_ply_limit = env.get_no_capture_ply_limit();
    h.max_ply = env.get_max_ply();
    h.repetition_counts = env.get_repetition_counts();
    return h;
}

static void require(bool ok, const char* msg) {
    if (!ok) throw std::runtime_error(msg);
}

static void test_all_children_third_repetition() {
    JunqiEnvironment env;
    env.load_board_state(kPosition0, 1);
    const auto legal = env.get_legal_actions(1);
    require(!legal.empty(), "position0 has no legal action");
    JunqiSearchHistory h = base_history(env);
    for (int a : legal) h.repetition_counts.emplace_back(child_key(kPosition0, 1, a), 2);
    heuristic_select_action_with_history(kPosition0, legal, 1, 50000, 4,
                                         JunqiHeuristicKind::WideDrawAware, h);
    int checked = 0;
    for (const auto& c : heuristic_last_candidates()) {
        if (!c.has_search_score) continue;
        ++checked;
        if (std::abs(c.search_score) > 1e-12) {
            std::cerr << "nonzero repetition score action=" << c.action
                      << " score=" << c.search_score << '\n';
            throw std::runtime_error("third repetition was not scored as exact draw");
        }
    }
    require(checked > 0, "no completed root action was searched");
    std::cout << "PASS all_children_third_repetition candidates=" << checked << '\n';
}

static void test_max_ply_draw() {
    JunqiEnvironment env;
    env.load_board_state(kPosition0, 1);
    const auto legal = env.get_legal_actions(1);
    JunqiSearchHistory h = base_history(env);
    h.turn_count = 199;
    h.max_ply = 200;
    heuristic_select_action_with_history(kPosition0, legal, 1, 50000, 4,
                                         JunqiHeuristicKind::WideDrawAware, h);
    int checked = 0;
    for (const auto& c : heuristic_last_candidates()) {
        if (!c.has_search_score) continue;
        ++checked;
        require(std::abs(c.search_score) <= 1e-12, "max-ply child was not exact draw");
    }
    require(checked > 0, "max-ply test missed all root actions");
    std::cout << "PASS max_ply_exact_draw candidates=" << checked << '\n';
}


static std::vector<int> lopsided_board() {
    std::vector<int> board(60, 0);
    board[1] = 90;
    board[13] = 20;
    board[56] = -90;
    board[52] = -80;
    board[47] = -70;
    board[42] = -60;
    return board;
}

static void test_no_capture_exact_draw() {
    const auto board = lopsided_board();
    JunqiEnvironment env;
    env.load_board_state(board, 1);
    const auto legal = env.get_legal_actions(1);
    require(!legal.empty(), "no-capture test has no legal action");
    for (int action : legal) {
        const auto [from, to] = JunqiEnvironment::decode_action_index(action);
        require(board[to] == 0, "no-capture test unexpectedly contains a capture");
    }
    JunqiSearchHistory h = base_history(env);
    h.no_capture_ply = 199;
    h.no_capture_ply_limit = 200;
    heuristic_select_action_with_history(board, legal, 1, 50000, 4,
                                         JunqiHeuristicKind::WideDrawAware, h);
    int checked = 0;
    for (const auto& c : heuristic_last_candidates()) {
        if (!c.has_search_score) continue;
        ++checked;
        require(std::abs(c.search_score) <= 1e-12,
                "no-capture deadline child was not exact draw");
    }
    require(checked > 0, "no-capture test missed all root actions");
    std::cout << "PASS no_capture_exact_draw candidates=" << checked << '\n';
}

static void inspect_losing_draw_preference() {
    // Minimal, deliberately lopsided open position: red has only a sapper against
    // black's marshal/general/major. Flags remain on valid headquarters squares.
    const auto board = lopsided_board();

    JunqiEnvironment env;
    env.load_board_state(board, 1);
    const auto legal = env.get_legal_actions(1);
    require(legal.size() >= 2, "lopsided test needs at least two legal actions");

    // First obtain exact baseline root scores without historical repetition.
    JunqiSearchHistory plain = base_history(env);
    heuristic_select_action_with_history(board, legal, 1, 120000, 5,
                                         JunqiHeuristicKind::WideDrawAware, plain);
    std::vector<std::pair<int,double>> baseline;
    for (const auto& c : heuristic_last_candidates())
        if (c.has_search_score) baseline.emplace_back(c.action, c.search_score);
    require(!baseline.empty(), "no baseline candidate scores");
    const auto best_it = std::max_element(baseline.begin(), baseline.end(),
        [](auto a, auto b){ return a.second < b.second; });
    require(best_it->second < -1e-9, "lopsided position is not evaluated as losing");

    // Mark the baseline-worst legal continuation as an immediate third repetition.
    const auto worst_it = std::min_element(baseline.begin(), baseline.end(),
        [](auto a, auto b){ return a.second < b.second; });
    JunqiSearchHistory draw = plain;
    draw.repetition_counts.emplace_back(child_key(board, 1, worst_it->first), 2);
    const int selected = heuristic_select_action_with_history(
        board, legal, 1, 120000, 5, JunqiHeuristicKind::WideDrawAware, draw);
    if (selected != worst_it->first) {
        std::cerr << "baseline_best=" << best_it->first << ':' << best_it->second
                  << " marked_draw=" << worst_it->first << ':' << worst_it->second
                  << " selected=" << selected << '\n';
        throw std::runtime_error("losing side did not choose an available exact draw");
    }
    const auto& candidates = heuristic_last_candidates();
    auto selected_it = std::find_if(candidates.begin(), candidates.end(),
        [&](const auto& c){ return c.action == selected; });
    require(selected_it != candidates.end() && selected_it->has_search_score,
            "selected draw has no search score");
    require(std::abs(selected_it->search_score) <= 1e-12,
            "selected repetition was not scored zero");
    std::cout << "PASS losing_side_prefers_repetition action=" << selected
              << " baseline=" << worst_it->second << '\n';
}


static void test_winning_side_avoids_repetition() {
    const auto board = lopsided_board();
    JunqiEnvironment env;
    env.load_board_state(board, -1);
    const auto legal = env.get_legal_actions(-1);
    require(legal.size() >= 2, "winning-side test needs at least two legal actions");

    JunqiSearchHistory plain = base_history(env);
    heuristic_select_action_with_history(board, legal, -1, 120000, 5,
                                         JunqiHeuristicKind::WideDrawAware, plain);
    std::vector<std::pair<int,double>> baseline;
    for (const auto& c : heuristic_last_candidates())
        if (c.has_search_score) baseline.emplace_back(c.action, c.search_score);
    std::sort(baseline.begin(), baseline.end(),
        [](auto a, auto b){ return a.second > b.second; });
    require(baseline.size() >= 2 && baseline[1].second > 1e-9,
            "winning-side position lacks a second positive continuation");

    JunqiSearchHistory draw = plain;
    draw.repetition_counts.emplace_back(child_key(board, -1, baseline[0].first), 2);
    const int selected = heuristic_select_action_with_history(
        board, legal, -1, 120000, 5, JunqiHeuristicKind::WideDrawAware, draw);
    require(selected != baseline[0].first,
            "winning side chose an immediate repetition over a positive continuation");
    const auto& candidates = heuristic_last_candidates();
    auto selected_it = std::find_if(candidates.begin(), candidates.end(),
        [&](const auto& c){ return c.action == selected; });
    require(selected_it != candidates.end() && selected_it->has_search_score,
            "winning-side selected action has no search score");
    require(selected_it->search_score > 1e-9,
            "winning side did not preserve a positive continuation");
    std::cout << "PASS winning_side_avoids_repetition marked=" << baseline[0].first
              << " selected=" << selected << " score=" << selected_it->search_score << '\n';
}

int main() {
    try {
        test_all_children_third_repetition();
        test_max_ply_draw();
        test_no_capture_exact_draw();
        inspect_losing_draw_preference();
        test_winning_side_avoids_repetition();
        std::cout << "ALL DRAW SEARCH TESTS PASSED\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
