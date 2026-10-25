#include "env/junqi_environment.h"
#include "root_only_runtime.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using junqi::JunqiEnvironment;
namespace fs = std::filesystem;

struct Config {
    std::string positions = "data/positions_512.json";
    std::string out = "results/replay.json";
    int position_index = 0;
    int red_visits = 4000;
    int black_visits = 4000;
    std::string red_heuristic = "root_only";
    std::string black_heuristic = "root_only";
    int max_depth = 7;
    int max_steps = 200;
};

static std::string json_escape(const std::string& s) {
    std::ostringstream out;
    for (unsigned char c : s) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
                else out << c;
        }
    }
    return out.str();
}

static void write_board(std::ostream& out, const std::vector<int>& board) {
    out << '[';
    for (std::size_t i = 0; i < board.size(); ++i) out << (i ? "," : "") << board[i];
    out << ']';
}

static std::vector<std::array<int, 60>> load_positions(const std::string& path) {
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
    for (std::size_t i = 0; i < out.size(); ++i)
        std::copy(nums.begin() + static_cast<long>(i * 60),
                  nums.begin() + static_cast<long>((i + 1) * 60), out[i].begin());
    return out;
}


static JunqiHeuristicKind parse_heuristic(const std::string& name) {
    if (name == "root_only") return JunqiHeuristicKind::RootOnly;
    if (name == "every_node") return JunqiHeuristicKind::EveryNode;
    if (name == "wide_tactical") return JunqiHeuristicKind::WideTactical;
    if (name == "wide_quiescence" || name == "wide_qsearch") return JunqiHeuristicKind::WideQuiescence;
    if (name == "wide_draw_aware" || name == "wide_draw") return JunqiHeuristicKind::WideDrawAware;
    if (name == "wide_strategic_draw" || name == "wide_v4") return JunqiHeuristicKind::WideStrategicDraw;
    if (name == "wide_resource") return JunqiHeuristicKind::WideResourceSafety;
    if (name == "wide_tactical_v140") return JunqiHeuristicKind::WideTacticalLegacy;
    throw std::invalid_argument("invalid heuristic: " + name);
}

static Config parse_args(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (a == "--positions") c.positions = need("--positions");
        else if (a == "--out") c.out = need("--out");
        else if (a == "--position-index") c.position_index = std::stoi(need("--position-index"));
        else if (a == "--red-visits") c.red_visits = std::stoi(need("--red-visits"));
        else if (a == "--black-visits") c.black_visits = std::stoi(need("--black-visits"));
        else if (a == "--red-heuristic") c.red_heuristic = need("--red-heuristic");
        else if (a == "--black-heuristic") c.black_heuristic = need("--black-heuristic");
        else if (a == "--max-depth") c.max_depth = std::stoi(need("--max-depth"));
        else if (a == "--max-steps") c.max_steps = std::stoi(need("--max-steps"));
        else if (a == "--help" || a == "-h") {
            std::cout << "Usage: play_game --positions PATH --position-index N --red-visits N --black-visits N "
                         "[--red-heuristic root_only|wide_tactical|wide_quiescence|wide_draw_aware|wide_strategic_draw|wide_resource|wide_tactical_v140|every_node] "
                         "[--black-heuristic root_only|wide_tactical|wide_quiescence|wide_draw_aware|wide_strategic_draw|wide_resource|wide_tactical_v140|every_node] --out FILE\n";
            std::exit(0);
        } else throw std::invalid_argument("unknown option: " + a);
    }
    if (c.position_index < 0 || c.red_visits <= 0 || c.black_visits <= 0 ||
        c.max_depth <= 0 || c.max_steps <= 0)
        throw std::invalid_argument("invalid numeric option");
    (void)parse_heuristic(c.red_heuristic);
    (void)parse_heuristic(c.black_heuristic);
    return c;
}

int main(int argc, char** argv) {
    try {
        const Config c = parse_args(argc, argv);
        const auto positions = load_positions(c.positions);
        if (static_cast<std::size_t>(c.position_index) >= positions.size())
            throw std::runtime_error("position index out of range");
        fs::create_directories(fs::path(c.out).parent_path());
        std::ofstream out(c.out);
        if (!out) throw std::runtime_error("cannot create output file");

        JunqiEnvironment env;
        const auto& initial_arr = positions[static_cast<std::size_t>(c.position_index)];
        std::vector<int> initial(initial_arr.begin(), initial_arr.end());
        env.load_board_state(initial, 1);

        out << "{\n  \"position_index\":" << c.position_index
            << ",\n  \"red_visits\":" << c.red_visits
            << ",\n  \"black_visits\":" << c.black_visits
            << ",\n  \"red_heuristic\":\"" << json_escape(c.red_heuristic) << "\""
            << ",\n  \"black_heuristic\":\"" << json_escape(c.black_heuristic) << "\""
            << ",\n  \"max_depth\":" << c.max_depth
            << ",\n  \"max_steps\":" << c.max_steps
            << ",\n  \"initial_board\":";
        write_board(out, initial);
        out << ",\n  \"moves\":[\n";

        bool first = true;
        while (!env.is_game_over() && env.get_turn_count() < c.max_steps) {
            const int player = env.get_current_player();
            const auto legal = env.get_legal_actions(player);
            if (legal.empty()) break;
            const int visits = player == 1 ? c.red_visits : c.black_visits;
            const std::string& heuristic_name = player == 1 ? c.red_heuristic : c.black_heuristic;
            const auto before = env.get_perfect_state();
            const JunqiHeuristicKind heuristic_kind = parse_heuristic(heuristic_name);
            JunqiSearchHistory history;
            history.turn_count = env.get_turn_count();
            history.no_capture_ply = env.get_no_capture_ply();
            history.repetition_limit = env.get_repetition_limit();
            history.no_capture_ply_limit = env.get_no_capture_ply_limit();
            history.max_ply = std::min(env.get_max_ply(), c.max_steps);
            history.repetition_counts = env.get_repetition_counts();
            const auto start = std::chrono::steady_clock::now();
            const int action = (heuristic_kind == JunqiHeuristicKind::WideDrawAware || heuristic_kind == JunqiHeuristicKind::WideStrategicDraw)
                ? heuristic_select_action_with_history(before, legal, player, visits, c.max_depth, heuristic_kind, history)
                : heuristic_select_action(before, legal, player, visits, c.max_depth, heuristic_kind);
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            if (std::find(legal.begin(), legal.end(), action) == legal.end())
                throw std::runtime_error("search returned illegal action");
            const auto [from, to] = JunqiEnvironment::decode_action_index(action);
            const int combat_result = static_cast<int>(env.step(from, to));
            const auto after = env.get_perfect_state();
            if (!first) out << ",\n";
            first = false;
            out << "    {\"ply\":" << env.get_turn_count()
                << ",\"player\":" << player
                << ",\"visits\":" << visits
                << ",\"heuristic\":\"" << json_escape(heuristic_name) << "\""
                << ",\"action\":" << action
                << ",\"from\":" << from
                << ",\"to\":" << to
                << ",\"combat_result\":" << combat_result
                << ",\"legal_count\":" << legal.size()
                << ",\"nodes\":" << root_only_last_nodes()
                << ",\"completed_depth\":" << root_only_last_depth()
                << ",\"budget_hit\":" << (root_only_last_budget_hit() ? "true" : "false")
                << ",\"milliseconds\":" << std::fixed << std::setprecision(6) << ms
                << ",\"board\":";
            write_board(out, after);
            out << '}';
        }

        const int winner = env.is_game_over() ? env.get_winner() : 0;
        std::string reason = env.is_game_over() ? env.get_terminal_reason() : "MAX_STEPS";
        out << "\n  ],\n  \"winner\":" << winner
            << ",\n  \"turns\":" << env.get_turn_count()
            << ",\n  \"game_over\":" << (env.is_game_over() ? "true" : "false")
            << ",\n  \"terminal_reason\":\"" << json_escape(reason) << "\"\n}\n";
        std::cout << fs::absolute(c.out).string() << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
