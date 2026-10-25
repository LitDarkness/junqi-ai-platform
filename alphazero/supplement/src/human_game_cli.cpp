#include "env/junqi_environment.h"
#include "root_only_runtime.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using junqi::JunqiEnvironment;
namespace fs = std::filesystem;

namespace {

struct Config {
    std::string positions = "data/positions_512.json";
    int position_index = 0;
    std::string moves_text;
    std::string op = "status";      // status | apply | ai | hint
    int action = -1;
    std::string heuristic = "root_only";
    int visits = 16000;
    int max_depth = 7;
    std::uint64_t seed = 0;
};

struct HistoryEntry {
    int ply = 0;
    int player = 0;
    int action = -1;
    int from = -1;
    int to = -1;
    int combat_result = 0;
};

struct SearchInfo {
    bool present = false;
    int action = -1;
    int nodes = 0;
    int completed_depth = 0;
    bool budget_hit = false;
    double milliseconds = 0.0;
    std::string heuristic;
    int visits = 0;
    int max_depth = 0;
    std::vector<int> board_before;
    std::vector<JunqiRootCandidateInfo> candidates;
};

std::string json_escape(const std::string& s) {
    std::ostringstream out;
    for (unsigned char c : s) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c)
                        << std::dec << std::setfill(' ');
                } else {
                    out << c;
                }
        }
    }
    return out.str();
}

void write_ints(std::ostream& out, const std::vector<int>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) out << (i ? "," : "") << values[i];
    out << ']';
}

std::vector<std::array<int, 60>> load_positions(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open positions file: " + path);
    std::string text((std::istreambuf_iterator<char>(f)), {});
    const auto key = text.find("\"positions\"");
    if (key == std::string::npos) throw std::runtime_error("positions key not found");
    text = text.substr(key);
    std::vector<int> nums;
    long long value = 0;
    int sign = 1;
    bool in_number = false;
    for (char ch : text) {
        if (ch == '-' && !in_number) { sign = -1; value = 0; in_number = true; }
        else if (ch >= '0' && ch <= '9') {
            if (!in_number) { sign = 1; value = 0; in_number = true; }
            value = value * 10 + (ch - '0');
        } else if (in_number) {
            nums.push_back(static_cast<int>(sign * value));
            in_number = false; sign = 1; value = 0;
        }
    }
    if (in_number) nums.push_back(static_cast<int>(sign * value));
    if (nums.empty() || nums.size() % 60 != 0)
        throw std::runtime_error("positions integer count is not divisible by 60");
    std::vector<std::array<int, 60>> out(nums.size() / 60);
    for (std::size_t i = 0; i < out.size(); ++i) {
        std::copy(nums.begin() + static_cast<long>(i * 60),
                  nums.begin() + static_cast<long>((i + 1) * 60), out[i].begin());
    }
    return out;
}

std::vector<int> parse_moves(const std::string& text) {
    std::vector<int> moves;
    if (text.empty()) return moves;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        const int action = std::stoi(token);
        if (action < 0 || action >= 3600) throw std::invalid_argument("move action out of range");
        moves.push_back(action);
    }
    return moves;
}

Config parse_args(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (a == "--positions") c.positions = need("--positions");
        else if (a == "--position-index") c.position_index = std::stoi(need("--position-index"));
        else if (a == "--moves") c.moves_text = need("--moves");
        else if (a == "--op") c.op = need("--op");
        else if (a == "--action") c.action = std::stoi(need("--action"));
        else if (a == "--heuristic") c.heuristic = need("--heuristic");
        else if (a == "--visits") c.visits = std::stoi(need("--visits"));
        else if (a == "--max-depth") c.max_depth = std::stoi(need("--max-depth"));
        else if (a == "--seed") c.seed = static_cast<std::uint64_t>(std::stoull(need("--seed")));
        else if (a == "--help" || a == "-h") {
            std::cout << "Usage: human_game_cli --positions PATH --position-index N --moves CSV "
                         "--op status|apply|ai|hint [--action N] "
                         "[--heuristic root_only|wide_tactical|wide_quiescence|wide_draw_aware|wide_strategic_draw|wide_resource|wide_tactical_v140|every_node|static|random] "
                         "[--visits N --max-depth N --seed N]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + a);
        }
    }
    if (c.position_index < 0 || c.visits <= 0 || c.max_depth <= 0)
        throw std::invalid_argument("invalid numeric option");
    if (c.op != "status" && c.op != "apply" && c.op != "ai" && c.op != "hint")
        throw std::invalid_argument("invalid op");
    if (c.heuristic != "root_only" && c.heuristic != "wide_tactical" &&
        c.heuristic != "wide_quiescence" && c.heuristic != "wide_draw_aware" && c.heuristic != "wide_strategic_draw" && c.heuristic != "wide_resource" && c.heuristic != "wide_tactical_v140" &&
        c.heuristic != "every_node" &&
        c.heuristic != "static" && c.heuristic != "random")
        throw std::invalid_argument("invalid heuristic");
    return c;
}

int choose_action(const Config& c, JunqiEnvironment& env, const std::vector<int>& legal, SearchInfo& info) {
    if (legal.empty()) throw std::runtime_error("no legal actions");
    const int player = env.get_current_player();
    const auto board = env.get_perfect_state();
    info.present = true;
    info.heuristic = c.heuristic;
    info.visits = c.visits;
    info.max_depth = c.max_depth;
    info.board_before = board;

    const auto start = std::chrono::steady_clock::now();
    int action = legal.front();
    if (c.heuristic == "random") {
        std::uint64_t seed = c.seed;
        if (seed == 0) {
            seed = env.get_state_key() ^ (static_cast<std::uint64_t>(env.get_turn_count() + 1) * 0x9e3779b97f4a7c15ULL);
        }
        std::mt19937_64 rng(seed);
        action = legal[static_cast<std::size_t>(rng() % legal.size())];
        info.nodes = 0;
        info.completed_depth = 0;
        info.budget_hit = false;
    } else if (c.heuristic == "static") {
        action = root_only_select_action(board, legal, player, 1, 1);
        info.nodes = root_only_last_nodes();
        info.completed_depth = root_only_last_depth();
        info.budget_hit = root_only_last_budget_hit();
    } else if (c.heuristic == "every_node") {
        action = every_node_select_action(board, legal, player, c.visits, c.max_depth);
        info.nodes = root_only_last_nodes();
        info.completed_depth = root_only_last_depth();
        info.budget_hit = root_only_last_budget_hit();
        info.candidates = heuristic_last_candidates();
    } else if (c.heuristic == "wide_tactical" || c.heuristic == "wide_quiescence" || c.heuristic == "wide_draw_aware" || c.heuristic == "wide_strategic_draw" ||
               c.heuristic == "wide_resource" || c.heuristic == "wide_tactical_v140") {
        JunqiHeuristicKind kind = JunqiHeuristicKind::WideTactical;
        if (c.heuristic == "wide_quiescence") kind = JunqiHeuristicKind::WideQuiescence;
        else if (c.heuristic == "wide_draw_aware") kind = JunqiHeuristicKind::WideDrawAware;
        else if (c.heuristic == "wide_strategic_draw") kind = JunqiHeuristicKind::WideStrategicDraw;
        else if (c.heuristic == "wide_resource") kind = JunqiHeuristicKind::WideResourceSafety;
        else if (c.heuristic == "wide_tactical_v140") kind = JunqiHeuristicKind::WideTacticalLegacy;
        if (kind == JunqiHeuristicKind::WideDrawAware || kind == JunqiHeuristicKind::WideStrategicDraw) {
            JunqiSearchHistory history;
            history.turn_count = env.get_turn_count();
            history.no_capture_ply = env.get_no_capture_ply();
            history.repetition_limit = env.get_repetition_limit();
            history.no_capture_ply_limit = env.get_no_capture_ply_limit();
            history.max_ply = env.get_max_ply();
            history.repetition_counts = env.get_repetition_counts();
            action = heuristic_select_action_with_history(board, legal, player, c.visits, c.max_depth, kind, history);
        } else {
            action = heuristic_select_action(board, legal, player, c.visits, c.max_depth, kind);
        }
        info.nodes = root_only_last_nodes();
        info.completed_depth = root_only_last_depth();
        info.budget_hit = root_only_last_budget_hit();
        info.candidates = heuristic_last_candidates();
    } else {
        action = root_only_select_action(board, legal, player, c.visits, c.max_depth);
        info.nodes = root_only_last_nodes();
        info.completed_depth = root_only_last_depth();
        info.budget_hit = root_only_last_budget_hit();
        info.candidates = heuristic_last_candidates();
    }
    if (c.heuristic == "static") info.candidates = heuristic_last_candidates();
    info.milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    info.action = action;
    return action;
}

void write_state(const Config& c, JunqiEnvironment& env, const std::vector<HistoryEntry>& history,
                 const SearchInfo& search, bool action_applied, int applied_combat_result) {
    const auto board = env.get_perfect_state();
    const auto legal = env.get_legal_actions(env.get_current_player());
    std::vector<int> movable;
    movable.reserve(legal.size());
    for (int action : legal) movable.push_back(action / 60);
    std::sort(movable.begin(), movable.end());
    movable.erase(std::unique(movable.begin(), movable.end()), movable.end());

    std::cout << "{\n";
    std::cout << "  \"ok\":true,\n";
    std::cout << "  \"op\":\"" << json_escape(c.op) << "\",\n";
    std::cout << "  \"position_index\":" << c.position_index << ",\n";
    std::cout << "  \"board\":"; write_ints(std::cout, board); std::cout << ",\n";
    std::cout << "  \"current_player\":" << env.get_current_player() << ",\n";
    std::cout << "  \"turn_count\":" << env.get_turn_count() << ",\n";
    std::cout << "  \"game_over\":" << (env.is_game_over() ? "true" : "false") << ",\n";
    std::cout << "  \"winner\":" << env.get_winner() << ",\n";
    std::cout << "  \"terminal_reason\":\"" << json_escape(env.get_terminal_reason()) << "\",\n";
    std::cout << "  \"state_key\":\"" << env.get_state_key() << "\",\n";
    std::cout << "  \"legal_actions\":"; write_ints(std::cout, legal); std::cout << ",\n";
    std::cout << "  \"movable_from\":"; write_ints(std::cout, movable); std::cout << ",\n";
    std::cout << "  \"action_applied\":" << (action_applied ? "true" : "false") << ",\n";
    std::cout << "  \"applied_combat_result\":" << applied_combat_result << ",\n";
    std::cout << "  \"history\":[";
    for (std::size_t i = 0; i < history.size(); ++i) {
        const auto& h = history[i];
        std::cout << (i ? "," : "")
                  << "{\"ply\":" << h.ply
                  << ",\"player\":" << h.player
                  << ",\"action\":" << h.action
                  << ",\"from\":" << h.from
                  << ",\"to\":" << h.to
                  << ",\"combat_result\":" << h.combat_result << "}";
    }
    std::cout << "],\n";
    if (search.present) {
        std::cout << "  \"search\":{\"heuristic\":\"" << json_escape(search.heuristic)
                  << "\",\"visits\":" << search.visits
                  << ",\"max_depth\":" << search.max_depth
                  << ",\"action\":" << search.action
                  << ",\"from\":" << search.action / 60
                  << ",\"to\":" << search.action % 60
                  << ",\"nodes\":" << search.nodes
                  << ",\"completed_depth\":" << search.completed_depth
                  << ",\"budget_hit\":" << (search.budget_hit ? "true" : "false")
                  << ",\"milliseconds\":" << std::fixed << std::setprecision(6) << search.milliseconds
                  << ",\"board_before\":";
        write_ints(std::cout, search.board_before);
        std::cout << ",\"candidates\":[";
        for (std::size_t i = 0; i < search.candidates.size(); ++i) {
            const auto& candidate = search.candidates[i];
            std::cout << (i ? "," : "")
                      << "{\"action\":" << candidate.action
                      << ",\"from\":" << candidate.action / 60
                      << ",\"to\":" << candidate.action % 60
                      << ",\"static_score\":" << std::setprecision(10) << candidate.static_score
                      << ",\"static_rank\":" << candidate.static_rank
                      << ",\"search_score\":";
            if (candidate.has_search_score) std::cout << std::setprecision(10) << candidate.search_score;
            else std::cout << "null";
            std::cout << ",\"search_rank\":" << candidate.search_rank
                      << ",\"retained\":" << (candidate.retained ? "true" : "false")
                      << ",\"selected\":" << (candidate.selected ? "true" : "false")
                      << ",\"tactical_mask\":" << candidate.tactical_mask
                      << "}";
        }
        std::cout << "]}\n";
    } else {
        std::cout << "  \"search\":null\n";
    }
    std::cout << "}\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Config c = parse_args(argc, argv);
        const auto positions = load_positions(c.positions);
        if (static_cast<std::size_t>(c.position_index) >= positions.size())
            throw std::runtime_error("position index out of range");
        const std::vector<int> moves = parse_moves(c.moves_text);

        JunqiEnvironment env;
        const auto& initial_arr = positions[static_cast<std::size_t>(c.position_index)];
        env.load_board_state(std::vector<int>(initial_arr.begin(), initial_arr.end()), 1);

        std::vector<HistoryEntry> history;
        history.reserve(moves.size() + 1);
        for (int action : moves) {
            if (env.is_game_over()) throw std::runtime_error("move list continues after game over");
            const auto legal = env.get_legal_actions(env.get_current_player());
            if (std::find(legal.begin(), legal.end(), action) == legal.end())
                throw std::runtime_error("illegal action in move list: " + std::to_string(action));
            const int player = env.get_current_player();
            const auto [from, to] = JunqiEnvironment::decode_action_index(action);
            const int combat_result = static_cast<int>(env.step(from, to));
            history.push_back({env.get_turn_count(), player, action, from, to, combat_result});
        }

        SearchInfo search;
        bool action_applied = false;
        int applied_combat_result = 0;
        if (c.op == "apply") {
            if (env.is_game_over()) throw std::runtime_error("game already over");
            const auto legal = env.get_legal_actions(env.get_current_player());
            if (std::find(legal.begin(), legal.end(), c.action) == legal.end())
                throw std::runtime_error("requested action is illegal");
            const int player = env.get_current_player();
            const auto [from, to] = JunqiEnvironment::decode_action_index(c.action);
            applied_combat_result = static_cast<int>(env.step(from, to));
            history.push_back({env.get_turn_count(), player, c.action, from, to, applied_combat_result});
            action_applied = true;
        } else if (c.op == "ai" || c.op == "hint") {
            if (env.is_game_over()) throw std::runtime_error("game already over");
            const auto legal = env.get_legal_actions(env.get_current_player());
            const int action = choose_action(c, env, legal, search);
            if (c.op == "ai") {
                const int player = env.get_current_player();
                const auto [from, to] = JunqiEnvironment::decode_action_index(action);
                applied_combat_result = static_cast<int>(env.step(from, to));
                history.push_back({env.get_turn_count(), player, action, from, to, applied_combat_result});
                action_applied = true;
            }
        }

        write_state(c, env, history, search, action_applied, applied_combat_result);
        return 0;
    } catch (const std::exception& e) {
        std::cout << "{\"ok\":false,\"error\":\"" << json_escape(e.what()) << "\"}\n";
        return 1;
    }
}
