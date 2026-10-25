#include "env/junqi_environment.h"
#include "root_only_runtime.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using junqi::JunqiEnvironment;
namespace fs = std::filesystem;

struct Config {
    std::string positions = "data/positions_512.json";
    std::string out_dir = "results/manual";
    std::string mode = "anchor";
    std::vector<int> visits{500, 1000, 2000, 4000, 8000, 16000};
    int anchor = 1000;
    int limit = 256;
    int max_steps = 200;
    int max_depth = 7;
    int repeats = 1;
    int threads = 0;  // 0 = auto
    std::string heuristic = "root_only";
    std::string candidate_heuristic;
    std::string opponent_heuristic;
    bool allow_identical = false;
};

struct BotSpec {
    int visits = 1000;
    int max_depth = 7;
    JunqiHeuristicKind kind = JunqiHeuristicKind::RootOnly;
};

struct SearchAcc {
    std::uint64_t decisions = 0;
    std::uint64_t nodes = 0;
    double depth_sum = 0.0;
    double milliseconds = 0.0;
    std::uint64_t budget_hits = 0;

    void add(double ms) {
        ++decisions;
        nodes += static_cast<std::uint64_t>(root_only_last_nodes());
        depth_sum += root_only_last_depth();
        milliseconds += ms;
        budget_hits += root_only_last_budget_hit() ? 1 : 0;
    }

    void merge(const SearchAcc& src) {
        decisions += src.decisions;
        nodes += src.nodes;
        depth_sum += src.depth_sum;
        milliseconds += src.milliseconds;
        budget_hits += src.budget_hits;
    }
};

struct GameResult {
    int winner = 0;
    int turns = 0;
    bool illegal = false;
    std::uint64_t trace_hash = 1469598103934665603ULL;
    std::string terminal_reason = "NONE";
};

struct PairResult {
    int position_index = 0;
    double as_red = 0.0;
    double as_black = 0.0;
    int red_winner = 0;
    int black_winner = 0;
    int red_turns = 0;
    int black_turns = 0;
    std::uint64_t red_hash = 0;
    std::uint64_t black_hash = 0;
    std::string red_reason;
    std::string black_reason;
};

struct PairWorkResult {
    PairResult pair;
    SearchAcc candidate_acc;
    SearchAcc opponent_acc;
    int wins = 0;
    int losses = 0;
    int draws = 0;
    int illegal = 0;
    int total_turns = 0;
    int repetition_draws = 0;
    int no_capture_draws = 0;
    int max_steps_draws = 0;
    int other_draws = 0;
};

struct MatchSummary {
    int wins = 0, losses = 0, draws = 0, illegal = 0;
    double score = 0.0, red_score = 0.0, black_score = 0.0;
    int total_turns = 0;
    double elapsed_s = 0.0;
    std::pair<double, double> ci{0.0, 1.0};
    SearchAcc candidate_acc, opponent_acc;
    int repetition_draws = 0, no_capture_draws = 0, max_steps_draws = 0, other_draws = 0;
    std::vector<PairResult> pairs;
    std::uint64_t result_hash = 1469598103934665603ULL;
};

static void hash_mix(std::uint64_t& h, std::uint64_t x) {
    for (int i = 0; i < 8; ++i) {
        h ^= static_cast<unsigned char>((x >> (i * 8)) & 0xffU);
        h *= 1099511628211ULL;
    }
}

static std::string hex64(std::uint64_t x) {
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << x;
    return out.str();
}

static std::vector<int> parse_visits(const std::string& text) {
    std::vector<int> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        int v = std::stoi(item);
        if (v <= 0) throw std::invalid_argument("visits must be positive");
        out.push_back(v);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    if (out.empty()) throw std::invalid_argument("empty visits list");
    return out;
}


static JunqiHeuristicKind parse_heuristic(const std::string& name) {
    if (name == "root_only") return JunqiHeuristicKind::RootOnly;
    if (name == "wide_tactical") return JunqiHeuristicKind::WideTactical;
    if (name == "wide_resource" || name == "wide_tactical_v3") return JunqiHeuristicKind::WideResourceSafety;
    if (name == "wide_quiescence" || name == "wide_qsearch") return JunqiHeuristicKind::WideQuiescence;
    if (name == "wide_draw_aware" || name == "wide_draw") return JunqiHeuristicKind::WideDrawAware;
    if (name == "wide_strategic_draw" || name == "wide_v4") return JunqiHeuristicKind::WideStrategicDraw;
    if (name == "wide_tactical_v140" || name == "wide_v140") return JunqiHeuristicKind::WideTacticalLegacy;
    if (name == "wide_full_ply2") return JunqiHeuristicKind::WideFullPly2;
    if (name == "wide_full_all") return JunqiHeuristicKind::WideFullAll;
    if (name == "every_node") return JunqiHeuristicKind::EveryNode;
    throw std::invalid_argument("unknown heuristic profile");
}

static const char* heuristic_name(JunqiHeuristicKind kind) {
    switch (kind) {
        case JunqiHeuristicKind::RootOnly: return "root_only";
        case JunqiHeuristicKind::WideTactical: return "wide_tactical";
        case JunqiHeuristicKind::WideResourceSafety: return "wide_resource";
        case JunqiHeuristicKind::WideQuiescence: return "wide_quiescence";
        case JunqiHeuristicKind::WideDrawAware: return "wide_draw_aware";
        case JunqiHeuristicKind::WideStrategicDraw: return "wide_strategic_draw";
        case JunqiHeuristicKind::WideTacticalLegacy: return "wide_tactical_v140";
        case JunqiHeuristicKind::WideFullPly2: return "wide_full_ply2";
        case JunqiHeuristicKind::WideFullAll: return "wide_full_all";
        case JunqiHeuristicKind::EveryNode: return "every_node";
    }
    return "unknown";
}

static void usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "  --positions PATH       fixed JSON corpus\n"
        << "  --out-dir PATH         output directory\n"
        << "  --mode anchor|adjacent|both|matrix\n"
        << "  --visits A,B,C         alpha-beta node budgets\n"
        << "  --anchor N             fixed opponent budget in anchor mode\n"
        << "  --limit N              use first N positions (0 = all)\n"
        << "  --max-steps N          draw after N plies\n"
        << "  --max-depth N          iterative-deepening cap (original = 7)\n"
        << "  --repeats N            exact repeats for reproducibility checks\n"
        << "  --threads N            parallel position pairs (0 = all logical cores)\n"
        << "  --heuristic NAME       set both sides: root_only|wide_tactical|wide_quiescence|wide_draw_aware|wide_strategic_draw|wide_resource|wide_tactical_v140|wide_full_ply2|wide_full_all|every_node\n"
        << "  --candidate-heuristic NAME  override candidate Heuristic\n"
        << "  --opponent-heuristic NAME   override opponent Heuristic\n"
        << "  --allow-identical          allow same heuristic and visits (draw-rate self-play)\n";
}

static Config parse_args(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (a == "--positions") c.positions = need("--positions");
        else if (a == "--out-dir") c.out_dir = need("--out-dir");
        else if (a == "--mode") c.mode = need("--mode");
        else if (a == "--visits") c.visits = parse_visits(need("--visits"));
        else if (a == "--anchor") c.anchor = std::stoi(need("--anchor"));
        else if (a == "--limit") c.limit = std::stoi(need("--limit"));
        else if (a == "--max-steps") c.max_steps = std::stoi(need("--max-steps"));
        else if (a == "--max-depth") c.max_depth = std::stoi(need("--max-depth"));
        else if (a == "--repeats") c.repeats = std::stoi(need("--repeats"));
        else if (a == "--threads") c.threads = std::stoi(need("--threads"));
        else if (a == "--heuristic") c.heuristic = need("--heuristic");
        else if (a == "--candidate-heuristic") c.candidate_heuristic = need("--candidate-heuristic");
        else if (a == "--opponent-heuristic") c.opponent_heuristic = need("--opponent-heuristic");
        else if (a == "--allow-identical") c.allow_identical = true;
        else if (a == "--help" || a == "-h") { usage(argv[0]); std::exit(0); }
        else throw std::invalid_argument("unknown option: " + a);
    }
    if (c.candidate_heuristic.empty()) c.candidate_heuristic = c.heuristic;
    if (c.opponent_heuristic.empty()) c.opponent_heuristic = c.heuristic;
    if (c.mode != "anchor" && c.mode != "adjacent" && c.mode != "both" && c.mode != "matrix")
        throw std::invalid_argument("mode must be anchor, adjacent, both, or matrix");
    if (c.anchor <= 0 || c.limit < 0 || c.max_steps <= 0 || c.max_depth <= 0 ||
        c.repeats <= 0 || c.threads < 0)
        throw std::invalid_argument("numeric options must be positive (limit/threads may be zero)");
    return c;
}

static int effective_threads(int requested, std::size_t jobs) {
    unsigned int hw = std::thread::hardware_concurrency();
    int n = requested == 0 ? static_cast<int>(hw ? hw : 1U) : requested;
    n = std::max(1, n);
    if (jobs > 0) n = std::min<int>(n, static_cast<int>(jobs));
    return n;
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

static double score_for(int winner, int primary_colour) {
    return winner == primary_colour ? 1.0 : (winner == 0 ? 0.5 : 0.0);
}

static GameResult play(const std::array<int, 60>& initial,
                       const BotSpec& red, const BotSpec& black,
                       int max_steps, SearchAcc& red_acc, SearchAcc& black_acc) {
    JunqiEnvironment env;
    env.load_board_state(std::vector<int>(initial.begin(), initial.end()), 1);
    GameResult result;
    hash_mix(result.trace_hash, static_cast<std::uint64_t>(red.visits));
    hash_mix(result.trace_hash, static_cast<std::uint64_t>(black.visits));
    hash_mix(result.trace_hash, static_cast<std::uint64_t>(red.kind));
    hash_mix(result.trace_hash, static_cast<std::uint64_t>(black.kind));
    while (!env.is_game_over() && env.get_turn_count() < max_steps) {
        const int player = env.get_current_player();
        const auto legal = env.get_legal_actions(player);
        if (legal.empty()) {
            result.winner = -player;
            result.turns = env.get_turn_count();
            result.terminal_reason = "NO_LEGAL_ACTION";
            hash_mix(result.trace_hash, static_cast<std::uint64_t>(result.winner + 2));
            return result;
        }
        const auto board = env.get_perfect_state();
        const BotSpec& bot = player == 1 ? red : black;
        JunqiSearchHistory history;
        history.turn_count = env.get_turn_count();
        history.no_capture_ply = env.get_no_capture_ply();
        history.repetition_limit = env.get_repetition_limit();
        history.no_capture_ply_limit = env.get_no_capture_ply_limit();
        history.max_ply = std::min(env.get_max_ply(), max_steps);
        history.repetition_counts = env.get_repetition_counts();
        const auto start = std::chrono::steady_clock::now();
        const int action = (bot.kind == JunqiHeuristicKind::WideDrawAware || bot.kind == JunqiHeuristicKind::WideStrategicDraw)
            ? heuristic_select_action_with_history(board, legal, player, bot.visits, bot.max_depth, bot.kind, history)
            : heuristic_select_action(board, legal, player, bot.visits, bot.max_depth, bot.kind);
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        (player == 1 ? red_acc : black_acc).add(ms);
        hash_mix(result.trace_hash, static_cast<std::uint64_t>(action));
        hash_mix(result.trace_hash, static_cast<std::uint64_t>(player + 2));
        if (std::find(legal.begin(), legal.end(), action) == legal.end()) {
            result.winner = -player;
            result.turns = env.get_turn_count();
            result.illegal = true;
            result.terminal_reason = "ILLEGAL";
            hash_mix(result.trace_hash, 0xBADULL);
            return result;
        }
        const auto [from, to] = JunqiEnvironment::decode_action_index(action);
        env.step(from, to);
    }
    result.winner = env.is_game_over() ? env.get_winner() : 0;
    result.turns = env.get_turn_count();
    result.terminal_reason = env.is_game_over() ? env.get_terminal_reason() : "MAX_STEPS";
    hash_mix(result.trace_hash, static_cast<std::uint64_t>(result.winner + 2));
    hash_mix(result.trace_hash, static_cast<std::uint64_t>(result.turns));
    return result;
}

static PairWorkResult run_pair(const std::array<int, 60>& position, int position_index,
                               const BotSpec& candidate, const BotSpec& opponent,
                               int max_steps) {
    PairWorkResult w;
    SearchAcc red1, black1, red2, black2;
    const GameResult gr = play(position, candidate, opponent, max_steps, red1, black1);
    const GameResult gb = play(position, opponent, candidate, max_steps, red2, black2);
    const double sr = score_for(gr.winner, 1);
    const double sb = score_for(gb.winner, -1);
    w.pair = {position_index, sr, sb, gr.winner, gb.winner,
              gr.turns, gb.turns, gr.trace_hash, gb.trace_hash,
              gr.terminal_reason, gb.terminal_reason};
    w.total_turns = gr.turns + gb.turns;
    w.illegal = (gr.illegal ? 1 : 0) + (gb.illegal ? 1 : 0);
    for (double x : {sr, sb}) {
        if (x == 1.0) ++w.wins;
        else if (x == 0.0) ++w.losses;
        else ++w.draws;
    }
    auto count_draw_reason = [&](const GameResult& game) {
        if (game.winner != 0) return;
        if (game.terminal_reason == "REPETITION") ++w.repetition_draws;
        else if (game.terminal_reason == "NO_CAPTURE_LIMIT") ++w.no_capture_draws;
        else if (game.terminal_reason == "MAX_STEPS" || game.terminal_reason == "MAX_PLY") ++w.max_steps_draws;
        else ++w.other_draws;
    };
    count_draw_reason(gr);
    count_draw_reason(gb);
    // Candidate is red in game 1 and black in game 2.
    w.candidate_acc.merge(red1);
    w.opponent_acc.merge(black1);
    w.opponent_acc.merge(red2);
    w.candidate_acc.merge(black2);
    return w;
}

static std::pair<double, double> paired_bootstrap_ci(const std::vector<double>& pair_scores,
                                                      std::uint64_t seed) {
    if (pair_scores.empty()) return {0.0, 1.0};
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> pick(0, pair_scores.size() - 1);
    constexpr int samples = 20000;
    std::vector<double> means;
    means.reserve(samples);
    for (int s = 0; s < samples; ++s) {
        double sum = 0.0;
        for (std::size_t i = 0; i < pair_scores.size(); ++i) sum += pair_scores[pick(rng)];
        means.push_back(sum / pair_scores.size());
    }
    std::sort(means.begin(), means.end());
    return {means[static_cast<std::size_t>(0.025 * samples)],
            means[static_cast<std::size_t>(0.975 * samples)]};
}

static MatchSummary run_match(const std::vector<std::array<int, 60>>& positions,
                              const BotSpec& candidate, const BotSpec& opponent,
                              int max_steps, int threads,
                              int matchup_index, int matchup_count) {
    MatchSummary m;
    const auto start = std::chrono::steady_clock::now();
    std::vector<PairWorkResult> work(positions.size());
    std::atomic<std::size_t> next{0};
    std::atomic<std::size_t> completed{0};
    std::mutex progress_mutex;
    const int workers = effective_threads(threads, positions.size());
    const std::size_t progress_stride = std::max<std::size_t>(1, positions.size() / 100);

    auto worker = [&]() {
        while (true) {
            const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= positions.size()) break;
            work[i] = run_pair(positions[i], static_cast<int>(i), candidate, opponent, max_steps);
            const std::size_t done = completed.fetch_add(1, std::memory_order_relaxed) + 1;
            if (done == positions.size() || done % progress_stride == 0) {
                std::lock_guard<std::mutex> lock(progress_mutex);
                std::cerr << "PROGRESS matchup=" << matchup_index << '/' << matchup_count
                          << " pairs=" << done << '/' << positions.size()
                          << " threads=" << workers << '\n';
            }
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(workers));
    for (int t = 0; t < workers; ++t) pool.emplace_back(worker);
    for (auto& t : pool) t.join();

    std::vector<double> pair_scores;
    pair_scores.reserve(positions.size());
    m.pairs.reserve(positions.size());
    // Aggregate strictly in position order. This keeps hashes and CSV deterministic
    // regardless of worker scheduling.
    for (const PairWorkResult& w : work) {
        m.pairs.push_back(w.pair);
        pair_scores.push_back((w.pair.as_red + w.pair.as_black) / 2.0);
        m.red_score += w.pair.as_red;
        m.black_score += w.pair.as_black;
        m.wins += w.wins;
        m.losses += w.losses;
        m.draws += w.draws;
        m.illegal += w.illegal;
        m.total_turns += w.total_turns;
        m.repetition_draws += w.repetition_draws;
        m.no_capture_draws += w.no_capture_draws;
        m.max_steps_draws += w.max_steps_draws;
        m.other_draws += w.other_draws;
        m.candidate_acc.merge(w.candidate_acc);
        m.opponent_acc.merge(w.opponent_acc);
        hash_mix(m.result_hash, w.pair.red_hash);
        hash_mix(m.result_hash, w.pair.black_hash);
    }
    m.elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const int games = static_cast<int>(positions.size() * 2);
    m.score = (m.wins + 0.5 * m.draws) / games;
    m.red_score /= positions.size();
    m.black_score /= positions.size();
    m.ci = paired_bootstrap_ci(pair_scores,
        20260722ULL ^ static_cast<std::uint64_t>(candidate.visits * 1000003ULL + opponent.visits));
    return m;
}

static bool run_tactics(int visits, int max_depth, JunqiHeuristicKind kind) {
    std::vector<std::tuple<std::array<int, 60>, int, int>> cases;
    auto add = [&](std::initializer_list<std::pair<int, int>> pieces, int player, int expected) {
        std::array<int, 60> b{};
        for (auto [p, v] : pieces) b[p] = v;
        cases.emplace_back(b, player, expected);
    };
    add({{1,3},{27,40},{32,-3},{58,-40}}, 1, 27*60+32);
    add({{1,40},{27,3},{32,-40},{58,-3}}, -1, 32*60+27);
    add({{1,3},{26,10},{31,-1},{58,-3}}, 1, 26*60+31);
    add({{1,3},{27,70},{28,-30},{32,-2},{58,-3}}, 1, 27*60+28);
    JunqiEnvironment env;
    bool all = true;
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto& [board, player, expected] = cases[i];
        env.load_board_state(std::vector<int>(board.begin(), board.end()), player);
        const auto legal = env.get_legal_actions(player);
        const int chosen = heuristic_select_action(env.get_perfect_state(), legal, player, visits, max_depth, kind);
        const bool ok = chosen == expected;
        all = all && ok;
        std::cout << "TACTIC case=" << i << ' ' << (ok ? "PASS" : "FAIL")
                  << " chosen=" << chosen << " expected=" << expected << "\n";
    }
    return all;
}

static std::vector<std::tuple<std::string, int, int>> make_matchups(const Config& c) {
    std::vector<std::tuple<std::string, int, int>> out;
    std::set<std::tuple<std::string, int, int>> seen;
    auto push = [&](std::string mode, int candidate, int opponent) {
        if (!c.allow_identical && candidate == opponent && c.candidate_heuristic == c.opponent_heuristic) return;
        auto x = std::make_tuple(mode, candidate, opponent);
        if (seen.insert(x).second) out.push_back(x);
    };
    if (c.mode == "anchor" || c.mode == "both")
        for (int v : c.visits) push("anchor", v, c.anchor);
    if (c.mode == "adjacent" || c.mode == "both")
        for (std::size_t i = 1; i < c.visits.size(); ++i)
            push("adjacent", c.visits[i], c.visits[i - 1]);
    if (c.mode == "matrix")
        for (std::size_t hi = 1; hi < c.visits.size(); ++hi)
            for (std::size_t lo = 0; lo < hi; ++lo)
                push("matrix", c.visits[hi], c.visits[lo]);
    return out;
}

static double mean_nodes(const SearchAcc& a) {
    return a.decisions ? static_cast<double>(a.nodes) / a.decisions : 0.0;
}
static double mean_depth(const SearchAcc& a) {
    return a.decisions ? a.depth_sum / a.decisions : 0.0;
}
static double mean_ms(const SearchAcc& a) {
    return a.decisions ? a.milliseconds / a.decisions : 0.0;
}
static double budget_rate(const SearchAcc& a) {
    return a.decisions ? static_cast<double>(a.budget_hits) / a.decisions : 0.0;
}

int main(int argc, char** argv) {
    try {
        const Config c = parse_args(argc, argv);
        auto positions = load_positions(c.positions);
        if (c.limit > 0) {
            if (static_cast<std::size_t>(c.limit) > positions.size())
                throw std::runtime_error("--limit exceeds corpus size");
            positions.resize(static_cast<std::size_t>(c.limit));
        }
        const int threads = effective_threads(c.threads, positions.size());
        fs::create_directories(c.out_dir);
        std::ofstream summary(fs::path(c.out_dir) / "summary.csv");
        std::ofstream pairs(fs::path(c.out_dir) / "pairs.csv");
        std::ofstream config(fs::path(c.out_dir) / "config.txt");
        if (!summary || !pairs || !config) throw std::runtime_error("cannot create output files");

        config << "positions=" << c.positions << "\npositions_used=" << positions.size()
               << "\nmode=" << c.mode << "\nanchor=" << c.anchor << "\nmax_steps=" << c.max_steps
               << "\nmax_depth=" << c.max_depth << "\nrepeats=" << c.repeats
               << "\nthreads_requested=" << c.threads << "\nthreads=" << threads
               << "\ncandidate_heuristic=" << c.candidate_heuristic
               << "\nopponent_heuristic=" << c.opponent_heuristic
               << "\nallow_identical=" << (c.allow_identical ? 1 : 0) << "\nvisits=";
        for (std::size_t i = 0; i < c.visits.size(); ++i) config << (i ? "," : "") << c.visits[i];
        config << "\n";

        summary << "mode,candidate_visits,opponent_visits,repeat,threads,positions,games,wins,losses,draws,score,score_delta,ci_low,ci_high,red_score,black_score,avg_turns,elapsed_s,illegal,result_hash,"
                << "candidate_decisions,candidate_mean_nodes,candidate_mean_depth,candidate_mean_ms,candidate_budget_hit_rate,"
                << "opponent_decisions,opponent_mean_nodes,opponent_mean_depth,opponent_mean_ms,opponent_budget_hit_rate,candidate_heuristic,opponent_heuristic,repetition_draws,no_capture_draws,max_steps_draws,other_draws\n";
        pairs << "mode,candidate_visits,opponent_visits,repeat,position_index,as_red,as_black,pair_score,red_winner,black_winner,red_turns,black_turns,red_trace_hash,black_trace_hash,red_terminal_reason,black_terminal_reason,candidate_heuristic,opponent_heuristic\n";

        const int tactic_visits = std::max(1000, *std::max_element(c.visits.begin(), c.visits.end()));
        const JunqiHeuristicKind candidate_kind = parse_heuristic(c.candidate_heuristic);
        const JunqiHeuristicKind opponent_kind = parse_heuristic(c.opponent_heuristic);
        if (!run_tactics(tactic_visits, c.max_depth, candidate_kind)) {
            std::cerr << "tactical gate failed\n";
            return 2;
        }

        const auto matchups = make_matchups(c);
        if (matchups.empty()) throw std::runtime_error("no non-trivial matchups were generated");
        const int total_runs = static_cast<int>(matchups.size()) * c.repeats;
        int run_index = 0;
        for (int rep = 1; rep <= c.repeats; ++rep) {
            for (const auto& [mode, candidate_visits, opponent_visits] : matchups) {
                ++run_index;
                std::cerr << "MATCHUP " << run_index << '/' << total_runs << ' ' << mode
                          << ' ' << c.candidate_heuristic << '-' << candidate_visits << " vs "
                          << c.opponent_heuristic << '-' << opponent_visits
                          << " threads=" << threads << "\n";
                const BotSpec candidate{candidate_visits, c.max_depth, candidate_kind};
                const BotSpec opponent{opponent_visits, c.max_depth, opponent_kind};
                const MatchSummary m = run_match(positions, candidate, opponent, c.max_steps,
                                                  threads, run_index, total_runs);
                const int games = static_cast<int>(positions.size() * 2);
                summary << std::fixed << std::setprecision(9)
                        << mode << ',' << candidate_visits << ',' << opponent_visits << ',' << rep << ','
                        << threads << ',' << positions.size() << ',' << games << ',' << m.wins << ','
                        << m.losses << ',' << m.draws << ',' << m.score << ',' << (m.score - 0.5) << ','
                        << m.ci.first << ',' << m.ci.second << ',' << m.red_score << ',' << m.black_score << ','
                        << static_cast<double>(m.total_turns) / games << ',' << m.elapsed_s << ','
                        << m.illegal << ',' << hex64(m.result_hash) << ','
                        << m.candidate_acc.decisions << ',' << mean_nodes(m.candidate_acc) << ','
                        << mean_depth(m.candidate_acc) << ',' << mean_ms(m.candidate_acc) << ','
                        << budget_rate(m.candidate_acc) << ',' << m.opponent_acc.decisions << ','
                        << mean_nodes(m.opponent_acc) << ',' << mean_depth(m.opponent_acc) << ','
                        << mean_ms(m.opponent_acc) << ',' << budget_rate(m.opponent_acc) << ','
                        << c.candidate_heuristic << ',' << c.opponent_heuristic << ','
                        << m.repetition_draws << ',' << m.no_capture_draws << ','
                        << m.max_steps_draws << ',' << m.other_draws << '\n';
                summary.flush();
                for (const PairResult& p : m.pairs) {
                    pairs << mode << ',' << candidate_visits << ',' << opponent_visits << ',' << rep << ','
                          << p.position_index << ',' << p.as_red << ',' << p.as_black << ','
                          << (p.as_red + p.as_black) / 2.0 << ',' << p.red_winner << ','
                          << p.black_winner << ',' << p.red_turns << ',' << p.black_turns << ','
                          << hex64(p.red_hash) << ',' << hex64(p.black_hash) << ','
                          << p.red_reason << ',' << p.black_reason << ','
                          << c.candidate_heuristic << ',' << c.opponent_heuristic << '\n';
                }
                pairs.flush();
                std::cout << "RESULT mode=" << mode << " candidate=" << candidate_visits
                          << " opponent=" << opponent_visits << " repeat=" << rep
                          << " W=" << m.wins << " L=" << m.losses << " D=" << m.draws
                          << " score=" << std::setprecision(6) << m.score
                          << " ci=[" << m.ci.first << ',' << m.ci.second << "]"
                          << " elapsed_s=" << m.elapsed_s << " hash=" << hex64(m.result_hash) << '\n';
            }
        }
        std::cout << "DONE out_dir=" << fs::absolute(c.out_dir).string() << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
