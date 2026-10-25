#include "root_only_runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr int kRows = 12, kCols = 5, kSize = 60;
constexpr int kRed = 1, kBlack = -1;
constexpr int kMine = 1, kBomb = 2, kFlag = 3, kSapper = 10;
constexpr int kMajor = 40, kColonel = 50, kGeneral = 70;
constexpr double kMate = 1'000'000.0;
thread_local int g_last_nodes = 0;
thread_local int g_last_completed_depth = 1;
thread_local bool g_last_budget_hit = false;
thread_local std::vector<JunqiRootCandidateInfo> g_last_candidates;
thread_local std::vector<JunqiRootPVInfo> g_last_pvs;

struct State {
    std::array<int, kSize> board{};
    int player = kRed;
    int winner = 0;
    bool terminal = false;
    int turn_count = 0;
    int no_capture_ply = 0;
    std::uint64_t key = 0;
};

struct Context {
    int root_player = kRed;
    int nodes = 0;
    bool stopped = false;
    int max_nodes = 1;
    JunqiHeuristicKind kind = JunqiHeuristicKind::RootOnly;
    std::unordered_map<int, double> root_resource_liability;
    std::unordered_map<std::uint64_t, int> historical_repetition_counts;
    std::vector<std::uint64_t> path_keys;
    int repetition_limit = 0;
    int no_capture_ply_limit = 0;
    int max_ply = 0;
    bool historical_repeat_pressure = false;
    bool draw_tracking_active = false;
    std::unordered_set<int> root_allowed_actions;
    std::unordered_map<std::uint64_t, int> pv_actions;
};

struct Phase {
    int own_flag = -1;
    int enemy_flag = -1;
    bool emergency = false;
    bool contact = false;
};

struct WideScoredAction {
    int action = -1;
    double score = 0.0;
    std::uint32_t tactical_mask = TacticalNone;
    bool forced = false;
};

bool is_wide_kind(JunqiHeuristicKind kind) {
    switch (kind) {
        case JunqiHeuristicKind::WideTactical:
        case JunqiHeuristicKind::WideResourceSafety:
        case JunqiHeuristicKind::WideQuiescence:
        case JunqiHeuristicKind::WideDrawAware:
        case JunqiHeuristicKind::WideStrategicDraw:
        case JunqiHeuristicKind::WideTacticalLegacy:
        case JunqiHeuristicKind::WideFullPly2:
        case JunqiHeuristicKind::WideFullAll:
            return true;
        default:
            return false;
    }
}

bool uses_resource_safety(JunqiHeuristicKind kind) {
    return kind == JunqiHeuristicKind::WideResourceSafety;
}

bool uses_capture_quiescence(JunqiHeuristicKind kind) {
    return kind == JunqiHeuristicKind::WideQuiescence || kind == JunqiHeuristicKind::WideDrawAware || kind == JunqiHeuristicKind::WideStrategicDraw;
}

bool uses_strategic_draw(JunqiHeuristicKind kind) {
    return kind == JunqiHeuristicKind::WideStrategicDraw;
}

bool uses_exact_draw(JunqiHeuristicKind kind) {
    if (kind != JunqiHeuristicKind::WideQuiescence &&
        kind != JunqiHeuristicKind::WideDrawAware &&
        kind != JunqiHeuristicKind::WideStrategicDraw) return false;
    static const bool value = [] {
        if (const char* text = std::getenv("JUNQI_EXACT_DRAW"))
            return std::string(text) != "0";
        return true;
    }();
    return value;
}

int capture_quiescence_depth() {
    static const int value = [] {
        if (const char* text = std::getenv("JUNQI_CAPTURE_QDEPTH")) {
            try { return std::clamp(std::stoi(text), 0, 4); }
            catch (...) {}
        }
        return 2;
    }();
    return value;
}

int capture_quiescence_width() {
    static const int value = [] {
        if (const char* text = std::getenv("JUNQI_CAPTURE_QWIDTH")) {
            try { return std::clamp(std::stoi(text), 1, 24); }
            catch (...) {}
        }
        return 4;
    }();
    return value;
}

double strategic_weight() {
    static const double value = [] {
        if (const char* text = std::getenv("JUNQI_STRATEGIC_WEIGHT")) {
            try { return std::clamp(std::stod(text), 0.0, 2.0); }
            catch (...) {}
        }
        return 1.0;
    }();
    return value;
}

double draw_risk_weight() {
    static const double value = [] {
        if (const char* text = std::getenv("JUNQI_DRAW_RISK_WEIGHT")) {
            try { return std::clamp(std::stod(text), 0.0, 1.0); }
            catch (...) {}
        }
        return 0.0;
    }();
    return value;
}

double projection_discount() {
    static const double value = [] {
        if (const char* text = std::getenv("JUNQI_PROJECTION_DISCOUNT")) {
            try { return std::clamp(std::stod(text), 0.0, 0.5); }
            catch (...) {}
        }
        return 0.10;
    }();
    return value;
}

int dominance_min_rank() {
    static const int value = [] {
        if (const char* text = std::getenv("JUNQI_DOM_MIN_RANK")) {
            try { return std::clamp(std::stoi(text), 40, 90); }
            catch (...) {}
        }
        return 70;
    }();
    return value;
}

bool owned(int piece, int player) { return piece != 0 && ((piece > 0) == (player > 0)); }
int type_of(int piece) { return std::abs(piece); }
int row(int p) { return p / kCols; }
int col(int p) { return p % kCols; }
int manhattan(int a, int b) { return std::abs(row(a) - row(b)) + std::abs(col(a) - col(b)); }

std::uint64_t mix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

std::uint64_t piece_key(int pos, int piece) {
    const std::uint64_t code = static_cast<std::uint64_t>(static_cast<std::uint32_t>(piece));
    return mix64((static_cast<std::uint64_t>(pos + 1) << 32) ^ code ^ 0x4a554e5149504945ULL);
}

std::uint64_t player_key(int player) {
    return player == kRed ? 0x3c79ac492ba7b653ULL : 0x1c69b3f74ac4ae35ULL;
}

std::uint64_t position_key(const State& s) {
    std::uint64_t hash = player_key(s.player);
    for (int pos = 0; pos < kSize; ++pos)
        if (s.board[pos] != 0) hash ^= piece_key(pos, s.board[pos]);
    return hash;
}

bool camp(int p) {
    switch (p) {
        case 11: case 13: case 17: case 21: case 23:
        case 36: case 38: case 42: case 46: case 48: return true;
        default: return false;
    }
}

bool hq(int p) { return p == 1 || p == 3 || p == 56 || p == 58; }

bool railway(int p) {
    const int r = row(p), c = col(p);
    return r == 1 || r == 5 || r == 6 || r == 10 ||
           ((r >= 2 && r <= 4 || r >= 7 && r <= 9) && (c == 0 || c == 4));
}

bool mountain_blocked(int a, int b) {
    return (a == 26 && b == 31) || (a == 31 && b == 26) ||
           (a == 28 && b == 33) || (a == 33 && b == 28);
}

double value_of(int piece) {
    switch (type_of(piece)) {
        case 90: return 9.0; case 80: return 8.0; case 70: return 7.0;
        case 60: return 6.0; case 50: return 5.0; case 40: return 4.0;
        case 30: return 3.0; case 20: return 2.0; case 10: return 4.5;
        case 2: return 7.0; case 1: return 8.0; case 3: return 25.0;
        default: return 0.0;
    }
}

// Exact project combat precedence: flag, bomb, sapper-mine, ordinary mine,
// then rank. Return 1 attacker wins, -1 loses, 0 both removed.
int combat(int attacker, int defender) {
    const int a = type_of(attacker), d = type_of(defender);
    if (d == kFlag) return 1;
    if (a == kBomb || d == kBomb) return 0;
    if (d == kMine) return a == kSapper ? 1 : -1;
    if (a == kMine || a == kFlag) return -1;
    return a == d ? 0 : (a > d ? 1 : -1);
}

bool can_enter(const State& s, int from, int to) {
    const int target = s.board[to];
    return target == 0 || (!owned(target, s.player) && !camp(to));
}

void add_highway_moves(const State& s, int from, std::vector<int>& out) {
    static constexpr int dr[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    static constexpr int dc[8] = {0, 0, -1, 1, -1, 1, -1, 1};
    for (int d = 0; d < 8; ++d) {
        const int nr = row(from) + dr[d], nc = col(from) + dc[d];
        if (nr < 0 || nr >= kRows || nc < 0 || nc >= kCols) continue;
        if (d >= 4 && !camp(from) && !camp(nr * kCols + nc)) continue;
        const int to = nr * kCols + nc;
        if (!mountain_blocked(from, to) && can_enter(s, from, to)) out.push_back(to);
    }
}

void add_straight_railway_moves(const State& s, int from, std::vector<int>& out) {
    static constexpr int dr[4] = {-1, 1, 0, 0};
    static constexpr int dc[4] = {0, 0, -1, 1};
    for (int d = 0; d < 4; ++d) {
        int nr = row(from) + dr[d], nc = col(from) + dc[d], previous = from;
        while (nr >= 0 && nr < kRows && nc >= 0 && nc < kCols) {
            const int to = nr * kCols + nc;
            if (!railway(to) || mountain_blocked(previous, to)) break;
            if (s.board[to] != 0) {
                if (can_enter(s, from, to)) out.push_back(to);
                break;
            }
            out.push_back(to);
            previous = to;
            nr += dr[d]; nc += dc[d];
        }
    }
}

void add_sapper_railway_moves(const State& s, int from, std::vector<int>& out) {
    std::array<bool, kSize> visited{};
    std::array<int, kSize> queue{};
    int head = 0, tail = 0;
    queue[tail++] = from; visited[from] = true;
    static constexpr int dr[4] = {-1, 1, 0, 0};
    static constexpr int dc[4] = {0, 0, -1, 1};
    while (head < tail) {
        const int current = queue[head++];
        for (int d = 0; d < 4; ++d) {
            const int nr = row(current) + dr[d], nc = col(current) + dc[d];
            if (nr < 0 || nr >= kRows || nc < 0 || nc >= kCols) continue;
            const int to = nr * kCols + nc;
            if (!railway(to) || visited[to] || mountain_blocked(current, to)) continue;
            if (s.board[to] != 0) {
                if (can_enter(s, from, to)) { out.push_back(to); visited[to] = true; }
                continue;
            }
            visited[to] = true;
            out.push_back(to);
            queue[tail++] = to;
        }
    }
}

std::vector<int> legal_actions(const State& s) {
    std::vector<int> actions;
    actions.reserve(96);
    for (int from = 0; from < kSize; ++from) {
        const int piece = s.board[from], type = type_of(piece);
        if (!owned(piece, s.player) || type == kMine || type == kFlag || hq(from)) continue;
        std::vector<int> destinations;
        destinations.reserve(24);
        if (railway(from)) {
            if (type == kSapper) add_sapper_railway_moves(s, from, destinations);
            else add_straight_railway_moves(s, from, destinations);
        }
        add_highway_moves(s, from, destinations);
        for (int to : destinations) actions.push_back(from * kSize + to);
    }
    return actions;
}

std::vector<int> unique_actions(std::vector<int> actions) {
    std::unordered_set<int> seen;
    std::vector<int> out;
    out.reserve(actions.size());
    for (int action : actions) {
        if (seen.insert(action).second) out.push_back(action);
    }
    return out;
}

bool has_any_piece(const State& s, int player) {
    for (int piece : s.board) if (owned(piece, player)) return true;
    return false;
}

State apply(const State& s, int action) {
    State next = s;
    const int from = action / kSize, to = action % kSize;
    const int attacker = s.board[from], defender = s.board[to];
    ++next.turn_count;
    next.no_capture_ply = defender == 0 ? s.no_capture_ply + 1 : 0;
    next.key = s.key ^ piece_key(from, attacker);
    if (defender != 0) next.key ^= piece_key(to, defender);
    if (defender == 0) {
        next.board[to] = attacker; next.board[from] = 0;
        next.key ^= piece_key(to, attacker);
    } else {
        const int result = combat(attacker, defender);
        if (type_of(defender) == kFlag) {
            next.board[to] = attacker; next.board[from] = 0;
            next.key ^= piece_key(to, attacker);
            next.terminal = true; next.winner = s.player;
            return next;
        }
        if (result > 0) {
            next.board[to] = attacker; next.board[from] = 0;
            next.key ^= piece_key(to, attacker);
        } else if (result < 0) {
            next.board[from] = 0;
        } else {
            next.board[from] = 0; next.board[to] = 0;
        }
        if (!has_any_piece(next, -s.player)) {
            next.terminal = true; next.winner = s.player;
            return next;
        }
    }
    next.key ^= player_key(s.player) ^ player_key(-s.player);
    next.player = -s.player;
    return next;
}

int find_flag(const State& s, int player) {
    for (int p = 0; p < kSize; ++p) if (s.board[p] == player * kFlag) return p;
    return -1;
}

int hq_distance(int pos, int side) {
    const int a = side == kRed ? 1 : 56;
    const int b = side == kRed ? 3 : 58;
    return std::min(manhattan(pos, a), manhattan(pos, b));
}

bool potential_flag_reacher(const State& s, int player, int target) {
    for (int from = 0; from < kSize; ++from) {
        const int piece = s.board[from];
        if (!owned(piece, player)) continue;
        const int type = type_of(piece);
        if (type == kMine || type == kFlag || hq(from)) continue;
        if (manhattan(from, target) <= 2) return true;
        if (railway(from) && railway(target)) {
            if (type == kSapper || row(from) == row(target) || col(from) == col(target)) return true;
        }
    }
    return false;
}

bool highway_reaches_target(const State& s, int from, int target) {
    const int dr = std::abs(row(from) - row(target));
    const int dc = std::abs(col(from) - col(target));
    if (dr > 1 || dc > 1 || (dr == 0 && dc == 0)) return false;
    if (dr == 1 && dc == 1 && !camp(from) && !camp(target)) return false;
    return !mountain_blocked(from, target) && can_enter(s, from, target);
}

bool straight_railway_reaches_target(const State& s, int from, int target) {
    if (!railway(from) || !railway(target)) return false;
    int dr = 0, dc = 0;
    if (row(from) == row(target)) dc = col(target) > col(from) ? 1 : -1;
    else if (col(from) == col(target)) dr = row(target) > row(from) ? 1 : -1;
    else return false;
    int current = from;
    int nr = row(from) + dr, nc = col(from) + dc;
    while (nr >= 0 && nr < kRows && nc >= 0 && nc < kCols) {
        const int to = nr * kCols + nc;
        if (!railway(to) || mountain_blocked(current, to)) return false;
        if (to == target) return can_enter(s, from, target);
        if (s.board[to] != 0) return false;
        current = to;
        nr += dr; nc += dc;
    }
    return false;
}

bool sapper_railway_reaches_target(const State& s, int from, int target) {
    if (!railway(from) || !railway(target)) return false;
    std::array<bool, kSize> visited{};
    std::array<int, kSize> queue{};
    int head = 0, tail = 0;
    queue[tail++] = from;
    visited[from] = true;
    static constexpr int dr[4] = {-1, 1, 0, 0};
    static constexpr int dc[4] = {0, 0, -1, 1};
    while (head < tail) {
        const int current = queue[head++];
        for (int d = 0; d < 4; ++d) {
            const int nr = row(current) + dr[d], nc = col(current) + dc[d];
            if (nr < 0 || nr >= kRows || nc < 0 || nc >= kCols) continue;
            const int to = nr * kCols + nc;
            if (!railway(to) || visited[to] || mountain_blocked(current, to)) continue;
            if (to == target) return can_enter(s, from, target);
            visited[to] = true;
            if (s.board[to] == 0) queue[tail++] = to;
        }
    }
    return false;
}

bool player_can_capture_flag(const State& s, int player) {
    const int enemy_flag = find_flag(s, -player);
    if (enemy_flag < 0 || !potential_flag_reacher(s, player, enemy_flag)) return false;
    State view = s;
    view.player = player;
    for (int from = 0; from < kSize; ++from) {
        const int piece = view.board[from];
        if (!owned(piece, player)) continue;
        const int type = type_of(piece);
        if (type == kMine || type == kFlag || hq(from)) continue;
        if (highway_reaches_target(view, from, enemy_flag)) return true;
        if (type == kSapper) {
            if (sapper_railway_reaches_target(view, from, enemy_flag)) return true;
        } else if (straight_railway_reaches_target(view, from, enemy_flag)) {
            return true;
        }
    }
    return false;
}


// v1.7.2 resource-safety model ---------------------------------------------
//
// This is deliberately not a global material bonus.  It is consulted only at
// a high-budget root where the side to move has exactly one bomb, that bomb is
// immediately capturable, and removing it would leave at least a colonel-level
// enemy piece without any abstract counter.  In that state, a safer root move
// may win a 0.025-point numerical tie.  The normal v2 tree, leaf evaluator and
// deep move widths are unchanged.
constexpr std::array<int, 9> kOrdinaryRanks = {10, 20, 30, 40, 50, 60, 70, 80, 90};
constexpr double kResourceTieWindow = 0.025;
constexpr double kResourceMinImprovement = 0.20;
constexpr int kResourceMinNodes = 262144;

struct MaterialProfile {
    std::array<int, 9> ordinary{};
    int bombs = 0;
};

MaterialProfile material_profile(const State& s, int player) {
    MaterialProfile p;
    for (int piece : s.board) {
        if (!owned(piece, player)) continue;
        const int type = type_of(piece);
        if (type == kBomb) {
            ++p.bombs;
            continue;
        }
        for (std::size_t i = 0; i < kOrdinaryRanks.size(); ++i) {
            if (type == kOrdinaryRanks[i]) {
                ++p.ordinary[i];
                break;
            }
        }
    }
    return p;
}

double unmatched_rank_weight(int rank_type) {
    switch (rank_type) {
        case 90: return 6.5;
        case 80: return 4.8;
        case 70: return 3.2;
        case 60: return 2.0;
        case 50: return 1.2;
        case 40: return 0.70;
        case 30: return 0.35;
        case 20: return 0.20;
        case 10: return 0.12;
        default: return 0.0;
    }
}

// Exact maximum matching in the rank-only abstraction.  Ordinary counters are
// totally ordered; strongest targets can therefore greedily consume the
// weakest sufficient ordinary counter, falling back to one bomb if needed.
double one_side_unmatched_value(const MaterialProfile& targets,
                                const MaterialProfile& counters) {
    std::array<int, 9> available = counters.ordinary;
    int bombs = counters.bombs;
    double value = 0.0;
    int unmatched_count = 0;
    for (int i = 8; i >= 0; --i) {
        for (int copy = 0; copy < targets.ordinary[static_cast<std::size_t>(i)]; ++copy) {
            int chosen = -1;
            for (int j = i; j < 9; ++j) {
                if (available[static_cast<std::size_t>(j)] > 0) {
                    chosen = j;
                    break;
                }
            }
            if (chosen >= 0) --available[static_cast<std::size_t>(chosen)];
            else if (bombs > 0) --bombs;
            else {
                value += unmatched_rank_weight(kOrdinaryRanks[static_cast<std::size_t>(i)]);
                ++unmatched_count;
            }
        }
    }
    if (unmatched_count >= 2) value += 0.22 * (unmatched_count - 1);
    return value;
}

double bomb_marginal_counter_value(const MaterialProfile& owner,
                                   const MaterialProfile& enemy) {
    if (owner.bombs <= 0) return 0.0;
    MaterialProfile without = owner;
    --without.bombs;
    const double with_bomb = one_side_unmatched_value(enemy, owner);
    const double without_bomb = one_side_unmatched_value(enemy, without);
    return std::max(0.0, without_bomb - with_bomb);
}

// Exact one-move reachability against one of the at most two bombs.  This is
// used only at root and for root successors, never at every Alpha-Beta node.
double cheapest_direct_attacker_value(const State& s, int attacker_side, int target) {
    State view = s;
    view.player = attacker_side;
    double cheapest = std::numeric_limits<double>::infinity();
    for (int from = 0; from < kSize; ++from) {
        const int piece = view.board[from];
        if (!owned(piece, attacker_side)) continue;
        const int type = type_of(piece);
        if (type == kMine || type == kFlag || hq(from)) continue;
        bool reaches = highway_reaches_target(view, from, target);
        if (!reaches && type == kSapper)
            reaches = sapper_railway_reaches_target(view, from, target);
        else if (!reaches)
            reaches = straight_railway_reaches_target(view, from, target);
        if (reaches) cheapest = std::min(cheapest, value_of(piece));
    }
    return cheapest;
}

struct BombThreatSummary {
    double worst_loss = 0.0;
    int threatened = 0;
};

BombThreatSummary immediate_bomb_threat_summary(const State& s, int owner_side) {
    BombThreatSummary out;
    const MaterialProfile owner = material_profile(s, owner_side);
    if (owner.bombs == 0) return out;
    const MaterialProfile enemy = material_profile(s, -owner_side);
    const double marginal_counter = bomb_marginal_counter_value(owner, enemy);
    for (int pos = 0; pos < kSize; ++pos) {
        if (s.board[pos] != owner_side * kBomb) continue;
        const double attacker = cheapest_direct_attacker_value(s, -owner_side, pos);
        if (!std::isfinite(attacker)) continue;
        ++out.threatened;
        const double exchange_loss = std::max(0.0, value_of(kBomb) - attacker);
        out.worst_loss = std::max(out.worst_loss, 0.8 + exchange_loss + marginal_counter);
    }
    return out;
}

double resource_policy_cost(const State& s, int owner_side) {
    const MaterialProfile owner = material_profile(s, owner_side);
    if (owner.bombs != 1) return 0.0;
    const MaterialProfile enemy = material_profile(s, -owner_side);
    const double marginal_counter = bomb_marginal_counter_value(owner, enemy);
    if (marginal_counter + 1e-12 < unmatched_rank_weight(kColonel)) return 0.0;
    const BombThreatSummary threat = immediate_bomb_threat_summary(s, owner_side);
    return threat.threatened > 0 ? threat.worst_loss : 0.0;
}

double resource_liability_after_action(const State& s, int action) {
    const int mover = s.player;
    const State next = apply(s, action);
    if (next.terminal) return 0.0;
    return resource_policy_cost(next, mover);
}

std::uint32_t classify_resource_safety_action(const State& s, int action,
                                              double before_liability) {
    const double after = resource_liability_after_action(s, action);
    std::uint32_t mask = TacticalNone;
    if (after + kResourceMinImprovement < before_liability) mask |= TacticalBombDefense;
    if (after > before_liability + kResourceMinImprovement) mask |= TacticalBombExposure;
    return mask;
}


// v1.8 strategic dominance ---------------------------------------------------
// Ordinary pieces are reusable sweepers: an unmatched higher rank can defeat
// arbitrarily many lower ordinary pieces. Equal ranks cancel one-for-one and
// each bomb can neutralize one excess higher piece. The first unresolved rank
// is the strategic frontier. The result is deliberately bounded far below mate.
struct DominanceFrontier {
    int side = 0;
    int rank = 0;
    int excess = 0;
    int bombs_spent = 0;
};

DominanceFrontier immediate_dominance_frontier(const MaterialProfile& red,
                                               const MaterialProfile& black) {
    int red_max = -1, black_max = -1;
    for (int i = 8; i >= 3; --i) {
        if (red_max < 0 && red.ordinary[static_cast<std::size_t>(i)] > 0) red_max = i;
        if (black_max < 0 && black.ordinary[static_cast<std::size_t>(i)] > 0) black_max = i;
    }
    if (red_max == black_max) return {};
    const int side = red_max > black_max ? kRed : kBlack;
    const MaterialProfile& strong = side == kRed ? red : black;
    const MaterialProfile& weak = side == kRed ? black : red;
    const int weak_max = side == kRed ? black_max : red_max;
    int bombs = weak.bombs;
    for (int i = 8; i > weak_max && i >= 3; --i) {
        int copies = strong.ordinary[static_cast<std::size_t>(i)];
        const int used = std::min(copies, bombs);
        copies -= used;
        bombs -= used;
        if (copies > 0)
            return {side, kOrdinaryRanks[static_cast<std::size_t>(i)], copies, used};
    }
    return {};
}

DominanceFrontier projected_dominance_frontier(const MaterialProfile& red,
                                               const MaterialProfile& black) {
    int red_bombs = red.bombs, black_bombs = black.bombs;
    for (int i = 8; i >= 3; --i) { // major and above; lower pieces are tactical noise
        int diff = red.ordinary[static_cast<std::size_t>(i)] -
                   black.ordinary[static_cast<std::size_t>(i)];
        if (diff > 0) {
            const int used = std::min(diff, black_bombs);
            diff -= used; black_bombs -= used;
            if (diff > 0) return {kRed, kOrdinaryRanks[static_cast<std::size_t>(i)], diff, used};
        } else if (diff < 0) {
            int excess = -diff;
            const int used = std::min(excess, red_bombs);
            excess -= used; red_bombs -= used;
            if (excess > 0) return {kBlack, kOrdinaryRanks[static_cast<std::size_t>(i)], excess, used};
        }
    }
    return {};
}

double dominance_rank_cap(int rank) {
    switch (rank) {
        case 90: return 10.0;
        case 80: return 7.5;
        case 70: return 5.5;
        case 60: return 4.0;
        case 50: return 2.8;
        case 40: return 1.8;
        default: return 0.0;
    }
}

double flag_conversion_factor(const State& s, int dominant_side) {
    const int own_flag = find_flag(s, dominant_side);
    if (own_flag < 0) return 0.0;
    int nearest_enemy = 99;
    int movable = 0;
    for (int p = 0; p < kSize; ++p) {
        const int piece = s.board[p];
        if (!piece) continue;
        const int t = type_of(piece);
        if (owned(piece, -dominant_side) && t != kMine && t != kFlag)
            nearest_enemy = std::min(nearest_enemy, manhattan(p, own_flag));
        if (t != kMine && t != kFlag) ++movable;
    }
    double safety = 1.0;
    if (nearest_enemy <= 1) safety = 0.10;
    else if (nearest_enemy == 2) safety = 0.28;
    else if (nearest_enemy == 3) safety = 0.50;
    else if (nearest_enemy == 4) safety = 0.72;
    const double phase = std::clamp(0.35 + 0.65 * (1.0 - movable / 50.0), 0.35, 1.0);
    return safety * phase;
}

double strategic_dominance_from_summary(const MaterialProfile& red,
                                          const MaterialProfile& black,
                                          int nearest_to_red, int nearest_to_black,
                                          int movable, int perspective) {
    DominanceFrontier f = immediate_dominance_frontier(red, black);
    double projection_discount_value = 1.0;
    if (f.side == 0) {
        int red_max = -1, black_max = -1;
        for (int i = 8; i >= 3; --i) {
            if (red_max < 0 && red.ordinary[static_cast<std::size_t>(i)] > 0) red_max = i;
            if (black_max < 0 && black.ordinary[static_cast<std::size_t>(i)] > 0) black_max = i;
        }
        const bool top_pair_can_cancel = red_max >= 3 && red_max == black_max &&
            red.ordinary[static_cast<std::size_t>(red_max)] ==
            black.ordinary[static_cast<std::size_t>(black_max)];
        if (top_pair_can_cancel) {
            const DominanceFrontier projected = projected_dominance_frontier(red, black);
            const bool defender_has_no_bomb = projected.side == kRed
                ? black.bombs == 0 : projected.side == kBlack ? red.bombs == 0 : false;
            if (defender_has_no_bomb) {
                f = projected;
                projection_discount_value = ::projection_discount();
            }
        }
    }
    if (f.side == 0 || f.rank < dominance_min_rank()) return 0.0;
    const int nearest_enemy = f.side == kRed ? nearest_to_red : nearest_to_black;
    double safety = 1.0;
    if (nearest_enemy <= 1) safety = 0.10;
    else if (nearest_enemy == 2) safety = 0.28;
    else if (nearest_enemy == 3) safety = 0.50;
    else if (nearest_enemy == 4) safety = 0.72;
    const double phase = std::clamp(0.30 + 0.70 * (1.0 - movable / 50.0), 0.30, 1.0);
    const double cap = dominance_rank_cap(f.rank);
    const double multiplicity = 1.0 - std::exp(-0.85 * f.excess);
    const double raw = projection_discount_value * cap * multiplicity * safety * phase;
    const double bounded = 12.0 * std::tanh(raw / 12.0);
    return (f.side == perspective ? 1.0 : -1.0) * bounded;
}

// Preserve the incumbent's material/initiative evaluation. WideTactical changes
// which continuations reach the leaf; it intentionally does not replace the
// reference static preference in this hotfix.
double evaluate_unscaled(const State& s, const Context& ctx) {
    const int perspective = ctx.root_player;
    const int enemy = -perspective;
    double score = 0.0;
    int own_mobility = 0, enemy_mobility = 0;
    MaterialProfile red, black;
    int red_flag = -1, black_flag = -1;
    for (int p : {1, 3, 56, 58}) {
        if (s.board[p] == kRed * kFlag) red_flag = p;
        else if (s.board[p] == kBlack * kFlag) black_flag = p;
    }
    int nearest_to_red = 99, nearest_to_black = 99;
    int movable = 0;
    for (int pos = 0; pos < kSize; ++pos) {
        const int piece = s.board[pos];
        if (!piece) continue;
        const int type = type_of(piece);
        if (uses_strategic_draw(ctx.kind)) {
            MaterialProfile& profile = piece > 0 ? red : black;
            if (type == kBomb) ++profile.bombs;
            else {
                for (std::size_t i = 0; i < kOrdinaryRanks.size(); ++i)
                    if (type == kOrdinaryRanks[i]) { ++profile.ordinary[i]; break; }
            }
            if (type != kMine && type != kFlag) {
                ++movable;
                if (piece < 0 && red_flag >= 0)
                    nearest_to_red = std::min(nearest_to_red, manhattan(pos, red_flag));
                if (piece > 0 && black_flag >= 0)
                    nearest_to_black = std::min(nearest_to_black, manhattan(pos, black_flag));
            }
        }
        const bool ours = owned(piece, perspective);
        const double sign = ours ? 1.0 : -1.0;
        score += sign * value_of(piece);
        if (type != kMine && type != kFlag) {
            const int advance = ours ? (perspective == kRed ? row(pos) : 11 - row(pos))
                                    : (enemy == kRed ? row(pos) : 11 - row(pos));
            score += sign * advance * (type >= kColonel ? 0.11 : 0.055);
            const int d = hq_distance(pos, enemy);
            score += sign * std::max(0, 12 - d) * (type >= kColonel ? 0.06 : 0.025);
            if (camp(pos)) score += sign * 0.45;
            if (ours) ++own_mobility; else ++enemy_mobility;
        }
        if (type == kFlag) {
            static constexpr int dr[4] = {-1, 1, 0, 0};
            static constexpr int dc[4] = {0, 0, -1, 1};
            for (int d = 0; d < 4; ++d) {
                const int nr = row(pos) + dr[d], nc = col(pos) + dc[d];
                if (nr < 0 || nr >= kRows || nc < 0 || nc >= kCols) continue;
                const int neighbor = s.board[nr * kCols + nc];
                if (owned(neighbor, piece > 0 ? kRed : kBlack)) {
                    if (type_of(neighbor) == kMine) score += sign * 2.5;
                    else if (type_of(neighbor) != kFlag) score += sign * 0.7;
                }
            }
        }
    }
    score += (own_mobility - enemy_mobility) * 0.08;
    const double sw = strategic_weight();
    if (uses_strategic_draw(ctx.kind) && sw > 0.0) {
        score += sw * strategic_dominance_from_summary(
            red, black, nearest_to_red, nearest_to_black, movable, perspective);
    }
    // draw-risk scaling is applied below after the helper definition.
    return score;
}

double draw_risk_factor(const State& s, const Context& ctx) {
    double risk = 0.0;
    if (ctx.no_capture_ply_limit > 0) {
        const double x = static_cast<double>(s.no_capture_ply) / ctx.no_capture_ply_limit;
        if (x > 0.82) {
            const double t = std::clamp((x - 0.82) / 0.18, 0.0, 1.0);
            risk = std::max(risk, 0.60 * t * t * (3.0 - 2.0 * t));
        }
    }
    if (ctx.max_ply > 0) {
        const double x = static_cast<double>(s.turn_count) / ctx.max_ply;
        if (x > 0.75) {
            const double t = std::clamp((x - 0.75) / 0.25, 0.0, 1.0);
            risk = std::max(risk, 0.65 * t * t * (3.0 - 2.0 * t));
        }
    }
    if (ctx.repetition_limit > 1) {
        const std::uint64_t key = s.key;
        int count = 0;
        if (const auto it = ctx.historical_repetition_counts.find(key);
            it != ctx.historical_repetition_counts.end()) count += it->second;
        count += static_cast<int>(std::count(ctx.path_keys.begin(), ctx.path_keys.end(), key));
        if (count == ctx.repetition_limit - 1) risk = std::max(risk, 0.35);
    }
    return std::clamp(risk, 0.0, 0.90);
}

double evaluate(const State& s, const Context& ctx) {
    double score = evaluate_unscaled(s, ctx);
    const double rw = draw_risk_weight();
    if (uses_strategic_draw(ctx.kind) && rw > 0.0)
        score *= 1.0 - rw * draw_risk_factor(s, ctx);
    return score;
}

Phase analyse_phase(const State& s, bool is_root, bool exact_reach_every_node) {
    Phase phase;
    phase.own_flag = find_flag(s, s.player);
    phase.enemy_flag = find_flag(s, -s.player);
    for (int p = 0; p < kSize; ++p) {
        const int piece = s.board[p];
        if (!piece || type_of(piece) == kMine || type_of(piece) == kFlag) continue;
        if (owned(piece, -s.player) && phase.own_flag >= 0 && manhattan(p, phase.own_flag) <= 5) phase.emergency = true;
        if (owned(piece, s.player) && phase.enemy_flag >= 0 && manhattan(p, phase.enemy_flag) <= 5) phase.contact = true;
    }
    if (!is_root && !exact_reach_every_node) return phase;
    if (phase.own_flag >= 0) {
        State enemy = s;
        enemy.player = -s.player;
        for (int action : legal_actions(enemy)) {
            if (manhattan(action % kSize, phase.own_flag) <= 3) {
                phase.emergency = true;
                break;
            }
        }
    }
    if (phase.enemy_flag >= 0) {
        State ours = s;
        ours.player = s.player;
        for (int action : legal_actions(ours)) {
            if (manhattan(action % kSize, phase.enemy_flag) <= 3) {
                phase.contact = true;
                break;
            }
        }
    }
    return phase;
}

double action_score(const State& s, int action, const Phase& phase) {
    const int from = action / kSize, to = action % kSize;
    const int attacker = s.board[from], defender = s.board[to], type = type_of(attacker);
    double score = 0.0;
    if (defender) {
        if (type_of(defender) == kFlag) return kMate;
        const int result = combat(attacker, defender);
        if (result > 0) score += 80.0 + value_of(defender) * 8.0 - value_of(attacker) * 0.1;
        else if (result == 0) score += value_of(defender) * 1.5 - value_of(attacker) * 0.4;
        else score -= value_of(attacker) * 5.0;
        if (type_of(defender) == kMine && type == kSapper) score += 18.0;
        if ((type == kBomb || type_of(defender) == kBomb) && std::max(type, type_of(defender)) >= kGeneral) score += 9.0;
    } else {
        score += 0.2;
        if (camp(to)) score += 0.8;
        if (railway(from) && manhattan(from, to) > 1) score += type == kSapper ? 1.1 : 0.35;
    }
    const int forward = s.player == kRed ? row(to) - row(from) : row(from) - row(to);
    score += forward * (type >= kMajor ? 0.70 : 0.35);
    if (phase.enemy_flag >= 0) {
        const int gain = manhattan(from, phase.enemy_flag) - manhattan(to, phase.enemy_flag);
        if (gain > 0) score += gain * (type >= kMajor ? 0.45 : 0.12);
    }
    if (phase.emergency && phase.own_flag >= 0) {
        const int gain = manhattan(from, phase.own_flag) - manhattan(to, phase.own_flag);
        if (gain > 0) score += gain * (type >= kMajor ? 2.0 : 0.60);
    }
    return score;
}

// Original ordering path, kept byte-for-byte in behaviour for root_only and
// every_node conformance.
std::vector<int> ordered_actions_original(const State& s, int limit, bool is_root, bool exact_reach_every_node) {
    std::vector<int> actions = legal_actions(s);
    const Phase phase = analyse_phase(s, is_root, exact_reach_every_node);
    std::stable_sort(actions.begin(), actions.end(), [&](int a, int b) {
        return action_score(s, a, phase) > action_score(s, b, phase);
    });
    if (static_cast<int>(actions.size()) > limit) actions.resize(limit);
    return actions;
}

std::uint32_t classify_tactical(const State& s, int action, const Phase& phase,
                                bool under_immediate_flag_threat) {
    const int from = action / kSize, to = action % kSize;
    const int attacker = s.board[from], defender = s.board[to];
    const int type = type_of(attacker);
    std::uint32_t mask = TacticalNone;

    if (defender && type_of(defender) == kFlag) mask |= TacticalFlagCapture;
    if (phase.enemy_flag >= 0) {
        const int before = manhattan(from, phase.enemy_flag);
        const int after = manhattan(to, phase.enemy_flag);
        if (after <= 1 || (after <= 3 && after < before)) mask |= TacticalFlagApproach;
        if (defender && after <= 3) mask |= TacticalFlagZoneCombat;
    }
    if (phase.own_flag >= 0) {
        const int before = manhattan(from, phase.own_flag);
        const int after = manhattan(to, phase.own_flag);
        if (defender && after <= 3) mask |= TacticalFlagZoneCombat;
        if ((phase.emergency || after <= 3) && after < before) mask |= TacticalFlagDefense;
    }
    if (defender && type_of(defender) == kMine && type == kSapper) mask |= TacticalMineClear;
    if (railway(from) && manhattan(from, to) > 1) mask |= TacticalLongRail;

    if (under_immediate_flag_threat) {
        const State next = apply(s, action);
        if (next.terminal || !player_can_capture_flag(next, next.player)) {
            mask |= TacticalImmediateDefense;
        }
    }
    return mask;
}

bool tactical_forced(std::uint32_t mask) {
    constexpr std::uint32_t forced = TacticalFlagCapture | TacticalFlagApproach |
        TacticalFlagZoneCombat | TacticalMineClear | TacticalFlagDefense |
        TacticalImmediateDefense;
    return (mask & forced) != 0;
}

std::vector<WideScoredAction> wide_scored_actions(const State& s, bool is_root, JunqiHeuristicKind kind) {
    std::vector<int> actions = unique_actions(legal_actions(s));
    const Phase phase = analyse_phase(s, is_root, false);
    const bool under_threat = player_can_capture_flag(s, -s.player);
    // v1.4 classified every candidate by applying it and regenerating the
    // opponent's flag-capture moves. Preserve that only for the legacy profile.
    // New profiles make immediate-threat nodes full width, so no candidate can
    // be lost and the O(actions * move_generation) classification is unnecessary.
    const bool classify_immediate_defense = under_threat &&
        (is_root || kind == JunqiHeuristicKind::WideTacticalLegacy);
    std::vector<WideScoredAction> scored;
    scored.reserve(actions.size());
    for (int action : actions) {
        const std::uint32_t mask = classify_tactical(s, action, phase, classify_immediate_defense);
        double score = action_score(s, action, phase);
        // These bonuses are ordering-only. Minimax leaf scores remain unchanged.
        if (mask & TacticalFlagCapture) score += 500000.0;
        if (mask & TacticalImmediateDefense) score += 220.0;
        if (mask & TacticalFlagApproach) score += 28.0;
        if (mask & TacticalFlagZoneCombat) score += 16.0;
        if (mask & TacticalFlagDefense) score += 14.0;
        if (mask & TacticalMineClear) score += 10.0;
        scored.push_back({action, score, mask, tactical_forced(mask)});
    }
    std::stable_sort(scored.begin(), scored.end(), [](const WideScoredAction& a, const WideScoredAction& b) {
        return a.score > b.score;
    });
    return scored;
}

std::vector<int> wide_ordered_actions(const State& s, bool is_root, int depth, int ply,
                                      JunqiHeuristicKind kind) {
    std::vector<WideScoredAction> scored = wide_scored_actions(s, is_root, kind);

    bool full_width = is_root || ply <= 1;
    if (kind == JunqiHeuristicKind::WideFullAll) full_width = true;
    if (kind == JunqiHeuristicKind::WideFullPly2 && ply <= 2) full_width = true;
    if (full_width) {
        std::vector<int> all;
        all.reserve(scored.size());
        for (const auto& x : scored) all.push_back(x.action);
        return all;
    }

    const Phase phase = analyse_phase(s, false, false);
    const bool under_threat = player_can_capture_flag(s, -s.player);
    // In the new profiles, an immediate flag threat is a proof-critical node:
    // search every legal defence instead of simulating and labelling each move.
    if (under_threat && kind != JunqiHeuristicKind::WideTacticalLegacy) {
        std::vector<int> all;
        all.reserve(scored.size());
        for (const auto& x : scored) all.push_back(x.action);
        return all;
    }
    int limit = 16;
    switch (kind) {
        case JunqiHeuristicKind::WideResourceSafety:
        case JunqiHeuristicKind::WideTactical:
            // Ablation winner: root and first reply remain full width, but the
            // old ply-2=40 special case is removed. Near-horizon and tactical
            // nodes still widen, while forced flag actions are always retained.
            if (depth <= 2) limit = 28;
            else if (phase.emergency || phase.contact) limit = 24;
            if (under_threat) limit = std::max(limit, 32);
            break;
        case JunqiHeuristicKind::WideDrawAware:
        case JunqiHeuristicKind::WideStrategicDraw:
            // Same bounded deep width as the validated qsearch profile.
            break;
        case JunqiHeuristicKind::WideTacticalLegacy:
            if (ply == 2) limit = 40;
            else if (depth <= 2) limit = 28;
            else if (phase.emergency || phase.contact) limit = 24;
            if (under_threat) limit = std::max(limit, 32);
            break;
        case JunqiHeuristicKind::WideFullPly2:
            if (depth <= 2) limit = 28;
            else if (phase.emergency || phase.contact) limit = 24;
            if (under_threat) limit = std::max(limit, 32);
            break;
        case JunqiHeuristicKind::WideFullAll:
            break;
        default:
            break;
    }

    int forced_count = 0;
    for (const auto& x : scored) forced_count += x.forced ? 1 : 0;
    const int target = std::max(limit, forced_count);
    std::unordered_set<int> retained;
    retained.reserve(static_cast<std::size_t>(target * 2 + 1));
    for (const auto& x : scored) if (x.forced) retained.insert(x.action);
    for (const auto& x : scored) {
        if (static_cast<int>(retained.size()) >= target) break;
        retained.insert(x.action);
    }

    std::vector<int> out;
    out.reserve(retained.size());
    for (const auto& x : scored) if (retained.count(x.action)) out.push_back(x.action);
    return out;
}

bool exhausted(Context& ctx) {
    if (++ctx.nodes >= ctx.max_nodes) {
        ctx.stopped = true;
        return true;
    }
    return false;
}

bool draw_due(const State& s, const Context& ctx) {
    if (ctx.max_ply > 0 && s.turn_count >= ctx.max_ply) return true;
    if (ctx.no_capture_ply_limit > 0 && s.no_capture_ply >= ctx.no_capture_ply_limit) return true;
    if (ctx.repetition_limit > 0) {
        const std::uint64_t key = s.key;
        int count = 0;
        if (const auto it = ctx.historical_repetition_counts.find(key);
            it != ctx.historical_repetition_counts.end()) count += it->second;
        count += static_cast<int>(std::count(ctx.path_keys.begin(), ctx.path_keys.end(), key));
        if (count >= ctx.repetition_limit) return true;
    }
    return false;
}

double terminal_value(const State& s, const Context& ctx, int ply) {
    if (s.terminal) {
        if (s.winner == 0) return 0.0;
        return s.winner == ctx.root_player ? kMate - ply : -kMate + ply;
    }
    if (ctx.draw_tracking_active && draw_due(s, ctx)) {
        // Environment adjudicates no-legal-move before draw counters.
        if (legal_actions(s).empty())
            return s.player == ctx.root_player ? -kMate + ply : kMate - ply;
        return 0.0;
    }
    return std::numeric_limits<double>::quiet_NaN();
}

std::vector<int> capture_quiescence_actions(const State& s) {
    const Phase phase = analyse_phase(s, false, false);
    struct Item { int action; double score; };
    std::vector<Item> items;
    for (int action : unique_actions(legal_actions(s))) {
        const int from = action / kSize, to = action % kSize;
        const int attacker = s.board[from], defender = s.board[to];
        if (!defender || owned(defender, s.player)) continue;
        const int a = type_of(attacker), d = type_of(defender);
        const int result = combat(attacker, defender);
        const bool forcing = d == kFlag || a == kBomb || d == kBomb ||
                             (d == kMine && a == kSapper) || result >= 0;
        if (!forcing) continue;
        items.push_back({action, action_score(s, action, phase)});
    }
    std::stable_sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
        return a.score > b.score;
    });
    if (static_cast<int>(items.size()) > capture_quiescence_width())
        items.resize(static_cast<std::size_t>(capture_quiescence_width()));
    std::vector<int> actions;
    actions.reserve(items.size());
    for (const Item& item : items) actions.push_back(item.action);
    return actions;
}


template <class SearchFn>
double search_child_with_history(const State& child, Context& ctx, SearchFn&& fn) {
    if (!ctx.draw_tracking_active) return fn();
    ctx.path_keys.push_back(child.key);
    const double score = fn();
    ctx.path_keys.pop_back();
    return score;
}


double capture_quiescence(const State& s, double alpha, double beta, int ply,
                          int qdepth, Context& ctx) {
    const double terminal = terminal_value(s, ctx, ply);
    if (!std::isnan(terminal)) return terminal;

    const bool maximizing = s.player == ctx.root_player;
    double best = evaluate(s, ctx);
    if (maximizing) alpha = std::max(alpha, best);
    else beta = std::min(beta, best);
    if (qdepth <= 0 || alpha >= beta) return best;

    const std::vector<int> actions = capture_quiescence_actions(s);
    for (int action : actions) {
        if (exhausted(ctx)) break;
        const State child = apply(s, action);
        const double score = search_child_with_history(child, ctx, [&] {
            return capture_quiescence(child, alpha, beta, ply + 1, qdepth - 1, ctx);
        });
        if (maximizing) {
            best = std::max(best, score);
            alpha = std::max(alpha, best);
        } else {
            best = std::min(best, score);
            beta = std::min(beta, best);
        }
        if (ctx.stopped || alpha >= beta) break;
    }
    return best;
}

double alphabeta(const State& s, int depth, double alpha, double beta, int ply,
                 int flag_extension_left, Context& ctx) {
    const double terminal = terminal_value(s, ctx, ply);
    if (!std::isnan(terminal)) return terminal;
    if (exhausted(ctx)) return evaluate(s, ctx);

    if (depth <= 0) {
        if (uses_capture_quiescence(ctx.kind)) {
            return capture_quiescence(s, alpha, beta, ply,
                                      capture_quiescence_depth(), ctx);
        }
        if (is_wide_kind(ctx.kind)) {
            if (player_can_capture_flag(s, s.player)) {
                return s.player == ctx.root_player ? kMate - ply - 1 : -kMate + ply + 1;
            }
            // If the opponent has a one-move flag capture threat, give the side
            // to move one exact defensive ply instead of evaluating at the
            // horizon. This is bounded to one extension per line.
            if (flag_extension_left > 0 && player_can_capture_flag(s, -s.player)) {
                depth = 1;
                --flag_extension_left;
            } else {
                return evaluate(s, ctx);
            }
        } else {
            return evaluate(s, ctx);
        }
    }

    std::vector<int> actions;
    if (is_wide_kind(ctx.kind)) {
        actions = wide_ordered_actions(s, false, depth, ply, ctx.kind);
    } else {
        const bool exact = ctx.kind == JunqiHeuristicKind::EveryNode;
        const Phase phase = analyse_phase(s, false, exact);
        const int width = phase.emergency || phase.contact ? 13 : 10;
        actions = ordered_actions_original(s, width, false, exact);
    }
    if (actions.empty()) return s.player == ctx.root_player ? -kMate + ply : kMate - ply;

    const bool maximizing = s.player == ctx.root_player;
    double best = maximizing ? -kMate : kMate;
    int best_action = -1;
    for (int action : actions) {
        const State child = apply(s, action);
        const double score = search_child_with_history(child, ctx, [&] {
            return alphabeta(child, depth - 1, alpha, beta,
                             ply + 1, flag_extension_left, ctx);
        });
        if ((maximizing && score > best) || (!maximizing && score < best)) {
            best = score;
            best_action = action;
        }
        if (maximizing) alpha = std::max(alpha, best);
        else beta = std::min(beta, best);
        if (ctx.stopped || alpha >= beta) break;
    }
    if (!ctx.stopped && best_action >= 0) ctx.pv_actions[s.key] = best_action;
    return best;
}

void initialize_root_candidates(const State& root, JunqiHeuristicKind kind,
                                const std::vector<int>& retained_actions) {
    const bool wide = is_wide_kind(kind);
    const bool exact = kind == JunqiHeuristicKind::EveryNode;
    std::vector<int> legal = unique_actions(legal_actions(root));
    const Phase phase = analyse_phase(root, true, exact);
    const bool under_threat = wide && player_can_capture_flag(root, -root.player);
    const double resource_before = uses_resource_safety(kind)
        ? resource_policy_cost(root, root.player) : 0.0;
    std::unordered_set<int> retained(retained_actions.begin(), retained_actions.end());

    g_last_candidates.clear();
    g_last_candidates.reserve(legal.size());
    for (int action : legal) {
        std::uint32_t mask = TacticalNone;
        double score = action_score(root, action, phase);
        if (wide) {
            mask = classify_tactical(root, action, phase, under_threat);
            if (uses_resource_safety(kind) && resource_before > 0.0)
                mask |= classify_resource_safety_action(root, action, resource_before);
            if (mask & TacticalFlagCapture) score += 500000.0;
            if (mask & TacticalImmediateDefense) score += 220.0;
            if (mask & TacticalFlagApproach) score += 28.0;
            if (mask & TacticalFlagZoneCombat) score += 16.0;
            if (mask & TacticalFlagDefense) score += 14.0;
            if (mask & TacticalMineClear) score += 10.0;
        }
        JunqiRootCandidateInfo info;
        info.action = action;
        info.static_score = score;
        info.retained = retained.count(action) != 0;
        info.tactical_mask = mask;
        g_last_candidates.push_back(info);
    }
    std::stable_sort(g_last_candidates.begin(), g_last_candidates.end(), [](const auto& a, const auto& b) {
        return a.static_score > b.static_score;
    });
    for (std::size_t i = 0; i < g_last_candidates.size(); ++i)
        g_last_candidates[i].static_rank = static_cast<int>(i + 1);
}

void apply_completed_root_scores(const std::vector<std::pair<int, double>>& scores, int selected) {
    std::unordered_map<int, double> by_action;
    for (const auto& [action, score] : scores) by_action[action] = score;
    for (auto& c : g_last_candidates) {
        const auto it = by_action.find(c.action);
        c.has_search_score = it != by_action.end();
        if (c.has_search_score) c.search_score = it->second;
        c.search_rank = 0;
        c.selected = c.action == selected;
    }
    std::vector<JunqiRootCandidateInfo*> ranked;
    for (auto& c : g_last_candidates) if (c.has_search_score) ranked.push_back(&c);
    std::stable_sort(ranked.begin(), ranked.end(), [](const auto* a, const auto* b) {
        return a->search_score > b->search_score;
    });
    for (std::size_t i = 0; i < ranked.size(); ++i) ranked[i]->search_rank = static_cast<int>(i + 1);
}

bool root_iteration(const State& root, int depth, int previous_numeric_best,
                    int& numeric_best_action, int& selected_action,
                    std::vector<std::pair<int, double>>& completed_scores, Context& ctx) {
    ctx.pv_actions.clear();
    if (uses_exact_draw(ctx.kind)) {
        const int horizon = depth + capture_quiescence_depth();
        const bool near_no_capture = ctx.no_capture_ply_limit > 0 &&
            root.no_capture_ply + horizon >= ctx.no_capture_ply_limit;
        const bool near_max_ply = ctx.max_ply > 0 &&
            root.turn_count + horizon >= ctx.max_ply;
        const bool path_can_repeat_thrice = horizon >= 8;
        ctx.draw_tracking_active = ctx.historical_repeat_pressure ||
            near_no_capture || near_max_ply || path_can_repeat_thrice;
    } else {
        ctx.draw_tracking_active = false;
    }
    std::vector<int> actions;
    if (is_wide_kind(ctx.kind)) {
        actions = wide_ordered_actions(root, true, depth, 0, ctx.kind);
    } else {
        actions = ordered_actions_original(root, 18, true, ctx.kind == JunqiHeuristicKind::EveryNode);
    }
    if (!ctx.root_allowed_actions.empty()) {
        actions.erase(
            std::remove_if(
                actions.begin(), actions.end(),
                [&](int action) {
                    return ctx.root_allowed_actions.find(action) ==
                           ctx.root_allowed_actions.end();
                }),
            actions.end());
    }
    if (actions.empty()) return false;
    if (is_wide_kind(ctx.kind)) {
        const auto pv = std::find(actions.begin(), actions.end(), previous_numeric_best);
        if (pv != actions.end() && pv != actions.begin()) std::rotate(actions.begin(), pv, pv + 1);
    }

    const bool policy_active = uses_resource_safety(ctx.kind) &&
                               !ctx.root_resource_liability.empty();
    struct RootResult { int action; double score; double liability; };
    std::vector<RootResult> results;
    results.reserve(actions.size());
    double numeric_best_score = -kMate;
    int numeric_best = actions.front();

    for (int action : actions) {
        double liability = 0.0;
        if (policy_active) {
            const auto it = ctx.root_resource_liability.find(action);
            if (it != ctx.root_resource_liability.end()) liability = it->second;
        }
        // A move that actually removes the critical last-bomb threat is given
        // a lower alpha only by the tie window.  If it still fails low, it is
        // provably outside the lexicographic tie and cannot be selected.
        const bool safer = policy_active && liability <= 1e-12;
        const double child_alpha = safer
            ? std::max(-kMate, numeric_best_score - kResourceTieWindow)
            : numeric_best_score;
        const State child = apply(root, action);
        const double score = search_child_with_history(child, ctx, [&] {
            return alphabeta(child, depth - 1, child_alpha, kMate, 1, 1, ctx);
        });
        if (ctx.stopped) return false;
        results.push_back({action, score, liability});
        if (score > numeric_best_score) {
            numeric_best_score = score;
            numeric_best = action;
        }
    }

    int chosen = numeric_best;
    double chosen_score = numeric_best_score;
    double chosen_liability = policy_active
        ? ctx.root_resource_liability.at(numeric_best) : 0.0;
    if (policy_active) {
        for (const RootResult& result : results) {
            if (result.score + kResourceTieWindow + 1e-12 < numeric_best_score) continue;
            if (result.liability + kResourceMinImprovement < chosen_liability - 1e-12 ||
                (std::abs(result.liability - chosen_liability) <= 1e-12 &&
                 result.score > chosen_score)) {
                chosen = result.action;
                chosen_score = result.score;
                chosen_liability = result.liability;
            }
        }
    }

    numeric_best_action = numeric_best;
    selected_action = chosen;
    completed_scores.clear();
    completed_scores.reserve(results.size());
    for (const RootResult& result : results)
        completed_scores.emplace_back(result.action, result.score);
    std::vector<JunqiRootPVInfo> completed_pvs;
    completed_pvs.reserve(results.size());
    for (const RootResult& result : results) {
        JunqiRootPVInfo info;
        info.action = result.action;
        info.actions.push_back(result.action);
        State cursor = apply(root, result.action);
        for (int remaining = depth - 1;
             remaining > 0 && !cursor.terminal; --remaining) {
            const auto found = ctx.pv_actions.find(cursor.key);
            if (found == ctx.pv_actions.end()) break;
            const std::vector<int> legal = unique_actions(legal_actions(cursor));
            if (std::find(legal.begin(), legal.end(), found->second) == legal.end())
                break;
            info.actions.push_back(found->second);
            cursor = apply(cursor, found->second);
        }
        completed_pvs.push_back(std::move(info));
    }
    g_last_pvs = std::move(completed_pvs);
    return true;
}

int select_action(const std::vector<int>& board, const std::vector<int>& supplied_legal,
                  int player, int max_nodes, int max_depth, JunqiHeuristicKind kind,
                  const JunqiSearchHistory* history) {
    if (board.size() != kSize || supplied_legal.empty() || max_nodes <= 0 || max_depth < 1)
        throw std::invalid_argument("invalid search input");
    State root;
    std::copy(board.begin(), board.end(), root.board.begin());
    root.player = player;
    root.key = position_key(root);
    Context ctx{player, 0, false, max_nodes, kind};
    ctx.root_allowed_actions.insert(supplied_legal.begin(), supplied_legal.end());
    if (history && uses_exact_draw(kind)) {
        root.turn_count = history->turn_count;
        root.no_capture_ply = history->no_capture_ply;
        ctx.repetition_limit = history->repetition_limit;
        ctx.no_capture_ply_limit = history->no_capture_ply_limit;
        ctx.max_ply = history->max_ply;
        for (const auto& [key, count] : history->repetition_counts) {
            ctx.historical_repetition_counts[key] = count;
            if (count >= 2) ctx.historical_repeat_pressure = true;
        }
    } else if (uses_exact_draw(kind)) {
        ctx.repetition_limit = 3;
        ctx.no_capture_ply_limit = 200;
        ctx.max_ply = 1000;
        ctx.historical_repetition_counts[root.key] = 1;
    }
    if (uses_resource_safety(kind) && max_nodes >= kResourceMinNodes) {
        const double before = resource_policy_cost(root, player);
        if (before > 0.0) {
            for (int action : unique_actions(legal_actions(root)))
                ctx.root_resource_liability.emplace(action, resource_liability_after_action(root, action));
        }
    }

    std::vector<int> fallback;
    if (is_wide_kind(kind))
        fallback = wide_ordered_actions(root, true, 1, 0, kind);
    else
        fallback = ordered_actions_original(root, 18, true, kind == JunqiHeuristicKind::EveryNode);
    initialize_root_candidates(root, kind, fallback);
    g_last_pvs.clear();
    for (int action : supplied_legal)
        g_last_pvs.push_back({action, std::vector<int>{action}});

    int numeric_best = fallback.empty() ? supplied_legal.front() : fallback.front();
    int best = numeric_best;
    int completed_depth = 1;
    std::vector<std::pair<int, double>> last_scores;
    for (int depth = 2; depth <= max_depth; ++depth) {
        int candidate_numeric = numeric_best;
        int candidate_selected = best;
        std::vector<std::pair<int, double>> scores;
        if (!root_iteration(root, depth, numeric_best, candidate_numeric, candidate_selected, scores, ctx)) break;
        numeric_best = candidate_numeric;
        best = candidate_selected;
        completed_depth = depth;
        last_scores = std::move(scores);
        apply_completed_root_scores(last_scores, best);
        for (auto& c : g_last_candidates) c.tactical_mask &= ~TacticalResourceTieBreak;
        if (best != numeric_best) {
            for (auto& c : g_last_candidates)
                if (c.action == best) c.tactical_mask |= TacticalResourceTieBreak;
        }
    }
    if (last_scores.empty()) {
        for (auto& c : g_last_candidates) c.selected = c.action == best;
    }

    g_last_nodes = ctx.nodes;
    g_last_completed_depth = completed_depth;
    g_last_budget_hit = ctx.nodes >= max_nodes;
    if (std::find(supplied_legal.begin(), supplied_legal.end(), best) == supplied_legal.end()) {
        best = supplied_legal.front();
        for (auto& c : g_last_candidates) c.selected = c.action == best;
    }
    return best;
}

}  // namespace

int heuristic_select_action(const std::vector<int>& board, const std::vector<int>& legal,
                            int player, int max_nodes, int max_depth, JunqiHeuristicKind kind) {
    return select_action(board, legal, player, max_nodes, max_depth, kind, nullptr);
}

int heuristic_select_action_with_history(const std::vector<int>& board, const std::vector<int>& legal,
                                         int player, int max_nodes, int max_depth,
                                         JunqiHeuristicKind kind,
                                         const JunqiSearchHistory& history) {
    return select_action(board, legal, player, max_nodes, max_depth, kind, &history);
}

int root_only_select_action(const std::vector<int>& board, const std::vector<int>& legal,
                            int player, int max_nodes, int max_depth) {
    return heuristic_select_action(board, legal, player, max_nodes, max_depth, JunqiHeuristicKind::RootOnly);
}

int every_node_select_action(const std::vector<int>& board, const std::vector<int>& legal,
                             int player, int max_nodes, int max_depth) {
    return heuristic_select_action(board, legal, player, max_nodes, max_depth, JunqiHeuristicKind::EveryNode);
}

int wide_tactical_select_action(const std::vector<int>& board, const std::vector<int>& legal,
                                int player, int max_nodes, int max_depth) {
    return heuristic_select_action(board, legal, player, max_nodes, max_depth, JunqiHeuristicKind::WideTactical);
}

int root_only_last_nodes() { return g_last_nodes; }
int root_only_last_depth() { return g_last_completed_depth; }
bool root_only_last_budget_hit() { return g_last_budget_hit; }
const std::vector<JunqiRootCandidateInfo>& heuristic_last_candidates() { return g_last_candidates; }
const std::vector<JunqiRootPVInfo>& heuristic_last_pvs() { return g_last_pvs; }

std::uint64_t heuristic_position_key(const std::vector<int>& board, int player) {
    if (board.size() != kSize || (player != kRed && player != kBlack))
        throw std::invalid_argument("invalid position-key input");
    State state;
    std::copy(board.begin(), board.end(), state.board.begin());
    state.player = player;
    return position_key(state);
}
