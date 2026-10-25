#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
namespace {

constexpr int kRows = 12, kCols = 5, kSize = 60;
constexpr int kRed = 1, kBlack = -1;
constexpr int kMine = 1, kBomb = 2, kFlag = 3, kSapper = 10;
constexpr int kMajor = 40, kColonel = 50, kGeneral = 70;
constexpr double kMate = 1'000'000.0;
constexpr auto kBudget = std::chrono::milliseconds(50); // Hard safety cap
constexpr int kMaxNodes = 4000;

struct Zobrist {
    uint64_t table[60][256];
    uint64_t player_turn;
    Zobrist() {
        uint64_t seed = 123456789ULL;
        auto rand64 = [&]() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            return seed;
        };
        for (int i = 0; i < 60; ++i) {
            for (int j = 0; j < 256; ++j) {
                table[i][j] = rand64();
            }
        }
        player_turn = rand64();
    }
};
const Zobrist z;

struct State {
    std::array<int, kSize> board{};
    int player = kRed;
    int winner = 0;
    bool terminal = false;
    
    uint64_t hash = 0;
    
    uint8_t red_pieces[30];
    uint8_t num_red_pieces = 0;
    uint8_t black_pieces[30];
    uint8_t num_black_pieces = 0;
    uint8_t piece_idx[kSize];
};

void init_state(State& s) {
    s.hash = 0;
    s.num_red_pieces = 0;
    s.num_black_pieces = 0;
    for (int i = 0; i < kSize; ++i) {
        int piece = s.board[i];
        if (piece != 0) {
            s.hash ^= z.table[i][static_cast<uint8_t>(piece)];
            if (piece > 0) {
                s.piece_idx[i] = s.num_red_pieces;
                s.red_pieces[s.num_red_pieces++] = i;
            } else {
                s.piece_idx[i] = s.num_black_pieces;
                s.black_pieces[s.num_black_pieces++] = i;
            }
        }
    }
    if (s.player == kBlack) {
        s.hash ^= z.player_turn;
    }
}

struct TTEntry {
    uint64_t key = 0;
    double score = 0.0;
    int depth = -1;
    int bound = 0; // 0: exact, 1: lower, 2: upper
    int best_action = -1;
};

uint64_t hash_state(const State& s) {
    return s.hash;
}

struct BudgetController {
    int nodes_until_hard_limit;
    int nodes_until_soft_limit;
    int nodes_until_time_check;
    std::chrono::steady_clock::time_point deadline;
    bool aborted;

    BudgetController(int soft, int hard, std::chrono::steady_clock::time_point dl)
        : nodes_until_hard_limit(hard), 
          nodes_until_soft_limit(soft),
          nodes_until_time_check(256),
          deadline(dl), 
          aborted(false) {}

    bool can_start_new_depth() const {
        return nodes_until_soft_limit > 0 && !aborted;
    }

    bool tick() {
        if (aborted) return true;

        --nodes_until_hard_limit;
        --nodes_until_soft_limit;
        --nodes_until_time_check;

        if (nodes_until_hard_limit <= 0) {
            aborted = true;
            return true;
        }

        if (nodes_until_time_check <= 0) {
            nodes_until_time_check = 256;
            if (std::chrono::steady_clock::now() >= deadline) {
                aborted = true;
                return true;
            }
        }

        return false;
    }
};

struct KillerTable {
    static constexpr int kMaxPly = 64;
    uint32_t packed_moves[kMaxPly];

    KillerTable() {
        for (int i = 0; i < kMaxPly; ++i) {
            packed_moves[i] = 0xFFFFFFFF;
        }
    }

    void store(int ply, int action) {
        if (ply >= kMaxPly) return;
        uint32_t current = packed_moves[ply];
        uint32_t k1 = current & 0xFFFF;
        if (k1 == static_cast<uint32_t>(action)) return;
        packed_moves[ply] = (static_cast<uint32_t>(action) & 0xFFFF) | (k1 << 16);
    }

    int get_killer1(int ply) const {
        if (ply >= kMaxPly) return -1;
        uint32_t k1 = packed_moves[ply] & 0xFFFF;
        return k1 == 0xFFFF ? -1 : static_cast<int>(k1);
    }

    int get_killer2(int ply) const {
        if (ply >= kMaxPly) return -1;
        uint32_t k2 = (packed_moves[ply] >> 16) & 0xFFFF;
        return k2 == 0xFFFF ? -1 : static_cast<int>(k2);
    }
};

struct Context {
    int root_player = kRed;
    BudgetController budget;
    std::vector<TTEntry> tt = std::vector<TTEntry>(1 << 18);
    KillerTable killers;
    
    Context(int player, std::chrono::steady_clock::time_point dl)
        : root_player(player), budget(3000, 6000, dl) {}
};

bool owned(int piece, int player) { return piece != 0 && ((piece > 0) == (player > 0)); }
int type_of(int piece) { return std::abs(piece); }
int row(int p) { return p / kCols; }
int col(int p) { return p % kCols; }
int manhattan(int a, int b) { return std::abs(row(a) - row(b)) + std::abs(col(a) - col(b)); }

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
        if (d < 4 && railway(from) && railway(to)) continue;
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
    
    const uint8_t* pieces = (s.player == kRed) ? s.red_pieces : s.black_pieces;
    const int num_pieces = (s.player == kRed) ? s.num_red_pieces : s.num_black_pieces;
    
    for (int i = 0; i < num_pieces; ++i) {
        int from = pieces[i];
        const int piece = s.board[from], type = type_of(piece);
        if (type == kMine || type == kFlag || hq(from)) continue;
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

std::vector<int> capture_actions(const State& s) {
    std::vector<int> actions;
    actions.reserve(32);
    
    const uint8_t* pieces = (s.player == kRed) ? s.red_pieces : s.black_pieces;
    const int num_pieces = (s.player == kRed) ? s.num_red_pieces : s.num_black_pieces;
    
    for (int i = 0; i < num_pieces; ++i) {
        int from = pieces[i];
        const int piece = s.board[from], type = type_of(piece);
        if (type == kMine || type == kFlag || hq(from)) continue;
        std::vector<int> destinations;
        destinations.reserve(24);
        if (railway(from)) {
            if (type == kSapper) add_sapper_railway_moves(s, from, destinations);
            else add_straight_railway_moves(s, from, destinations);
        }
        add_highway_moves(s, from, destinations);
        for (int to : destinations) {
            if (s.board[to] != 0) {
                actions.push_back(from * kSize + to);
            }
        }
    }
    return actions;
}

double mvv_lva(const State& s, int action) {
    const int from = action / kSize, to = action % kSize;
    const int attacker = s.board[from], defender = s.board[to];
    if (defender == 0) return 0;
    return value_of(defender) * 100.0 - value_of(attacker);
}

State apply(const State& s, int action) {
    State next = s;
    const int from = action / kSize, to = action % kSize;
    const int attacker = s.board[from], defender = s.board[to];
    
    auto remove_piece = [&](int pos, int piece) {
        next.hash ^= z.table[pos][static_cast<uint8_t>(piece)];
        next.board[pos] = 0;
        if (piece > 0) {
            uint8_t idx = next.piece_idx[pos];
            next.num_red_pieces--;
            if (idx != next.num_red_pieces) {
                uint8_t moved_pos = next.red_pieces[next.num_red_pieces];
                next.red_pieces[idx] = moved_pos;
                next.piece_idx[moved_pos] = idx;
            }
        } else {
            uint8_t idx = next.piece_idx[pos];
            next.num_black_pieces--;
            if (idx != next.num_black_pieces) {
                uint8_t moved_pos = next.black_pieces[next.num_black_pieces];
                next.black_pieces[idx] = moved_pos;
                next.piece_idx[moved_pos] = idx;
            }
        }
    };
    
    auto move_piece = [&](int from_pos, int to_pos, int piece) {
        next.hash ^= z.table[from_pos][static_cast<uint8_t>(piece)];
        next.hash ^= z.table[to_pos][static_cast<uint8_t>(piece)];
        next.board[from_pos] = 0;
        next.board[to_pos] = piece;
        if (piece > 0) {
            uint8_t idx = next.piece_idx[from_pos];
            next.red_pieces[idx] = to_pos;
            next.piece_idx[to_pos] = idx;
        } else {
            uint8_t idx = next.piece_idx[from_pos];
            next.black_pieces[idx] = to_pos;
            next.piece_idx[to_pos] = idx;
        }
    };

    if (defender == 0) {
        move_piece(from, to, attacker);
    } else {
        const int result = combat(attacker, defender);
        if (type_of(defender) == kFlag) {
            remove_piece(to, defender);
            move_piece(from, to, attacker);
            next.terminal = true; next.winner = s.player;
            next.player = -s.player;
            next.hash ^= z.player_turn;
            return next;
        }
        if (result > 0) { 
            remove_piece(to, defender);
            move_piece(from, to, attacker);
        }
        else if (result < 0) {
            remove_piece(from, attacker);
        }
        else { 
            remove_piece(from, attacker);
            remove_piece(to, defender);
        }
        if (next.num_red_pieces == 0) {
            next.terminal = true; next.winner = kBlack;
            next.player = -s.player;
            next.hash ^= z.player_turn;
            return next;
        }
        if (next.num_black_pieces == 0) {
            next.terminal = true; next.winner = kRed;
            next.player = -s.player;
            next.hash ^= z.player_turn;
            return next;
        }
    }
    next.player = -s.player;
    next.hash ^= z.player_turn;
    return next;
}

int find_flag_fast(const State& s, int player) {
    const uint8_t* pieces = (player == kRed) ? s.red_pieces : s.black_pieces;
    const int num_pieces = (player == kRed) ? s.num_red_pieces : s.num_black_pieces;
    for (int i = 0; i < num_pieces; ++i) {
        if (type_of(s.board[pieces[i]]) == kFlag) return pieces[i];
    }
    return -1;
}

int hq_distance(int pos, int side) {
    const int a = side == kRed ? 1 : 56;
    const int b = side == kRed ? 3 : 58;
    return std::min(manhattan(pos, a), manhattan(pos, b));
}

struct BoardControlMap {
    int red_max_rank[kSize] = {0};
    bool red_has_bomb[kSize] = {false};
    int black_max_rank[kSize] = {0};
    bool black_has_bomb[kSize] = {false};
    
    int global_red_max_rank = 0;
    bool global_red_has_bomb = false;
    int global_black_max_rank = 0;
    bool global_black_has_bomb = false;
};

BoardControlMap build_control_map(const State& s) {
    BoardControlMap map;
    static constexpr int dr[4] = {-1, 1, 0, 0};
    static constexpr int dc[4] = {0, 0, -1, 1};
    static constexpr int dr8[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    static constexpr int dc8[8] = {0, 0, -1, 1, -1, 1, -1, 1};

    for (int i = 0; i < kSize; ++i) {
        int piece = s.board[i];
        if (piece == 0) continue;
        int type = type_of(piece);
        if (type == kMine || type == kFlag || hq(i)) continue;
        int player = piece > 0 ? kRed : kBlack;
        
        if (player == kRed) {
            if (type == kBomb) map.global_red_has_bomb = true;
            else map.global_red_max_rank = std::max(map.global_red_max_rank, type);
        } else {
            if (type == kBomb) map.global_black_has_bomb = true;
            else map.global_black_max_rank = std::max(map.global_black_max_rank, type);
        }
        
        auto add_attack = [&](int to) {
            if (camp(to) && s.board[to] != 0) return; // Immune
            if (player == kRed) {
                if (type == kBomb) map.red_has_bomb[to] = true;
                else map.red_max_rank[to] = std::max(map.red_max_rank[to], type);
            } else {
                if (type == kBomb) map.black_has_bomb[to] = true;
                else map.black_max_rank[to] = std::max(map.black_max_rank[to], type);
            }
        };

        // Highway
        for (int d = 0; d < 8; ++d) {
            const int nr = row(i) + dr8[d], nc = col(i) + dc8[d];
            if (nr < 0 || nr >= kRows || nc < 0 || nc >= kCols) continue;
            if (d >= 4 && !camp(i) && !camp(nr * kCols + nc)) continue;
            const int to = nr * kCols + nc;
            if (!mountain_blocked(i, to)) {
                add_attack(to);
            }
        }

        // Railway
        if (railway(i)) {
            if (type == kSapper) {
                std::array<int, kSize> queue{};
                int head = 0, tail = 0;
                queue[tail++] = i;
                bool bfs_visited[kSize] = {false};
                bfs_visited[i] = true;
                while (head < tail) {
                    const int current = queue[head++];
                    for (int d = 0; d < 4; ++d) {
                        const int nr = row(current) + dr[d], nc = col(current) + dc[d];
                        if (nr < 0 || nr >= kRows || nc < 0 || nc >= kCols) continue;
                        const int to = nr * kCols + nc;
                        if (!railway(to) || bfs_visited[to] || mountain_blocked(current, to)) continue;
                        bfs_visited[to] = true;
                        add_attack(to);
                        if (s.board[to] == 0) queue[tail++] = to;
                    }
                }
            } else {
                for (int d = 0; d < 4; ++d) {
                    int nr = row(i) + dr[d], nc = col(i) + dc[d], previous = i;
                    while (nr >= 0 && nr < kRows && nc >= 0 && nc < kCols) {
                        const int to = nr * kCols + nc;
                        if (!railway(to) || mountain_blocked(previous, to)) break;
                        add_attack(to);
                        if (s.board[to] != 0) break;
                        previous = to;
                        nr += dr[d]; nc += dc[d];
                    }
                }
            }
        }
    }
    return map;
}

struct EvalFeatures {
    double material = 0.0;
    double advance = 0.0;
    double hq_proximity = 0.0;
    double camp_bonus = 0.0;
    double flag_defense = 0.0;
    double mobility = 0.0;
    double threat_score = 0.0;
    double sapper_rail_bonus = 0.0;
    double dominance_bonus = 0.0;
    int flag_pos = -1;
    double enemy_mobility_near_flag = 0.0;
};

EvalFeatures compute_features(const State& s, int player, int enemy_flag_pos, const BoardControlMap& cmap) {
    EvalFeatures f;
    
    static constexpr int dr[4] = {-1, 1, 0, 0};
    static constexpr int dc[4] = {0, 0, -1, 1};
    
    const uint8_t* pieces = (player == kRed) ? s.red_pieces : s.black_pieces;
    const int num_pieces = (player == kRed) ? s.num_red_pieces : s.num_black_pieces;
    
    for (int i = 0; i < num_pieces; ++i) {
        int pos = pieces[i];
        const int piece = s.board[pos];
        const int type = type_of(piece);
        
        if (type == kFlag) {
            f.flag_pos = pos;
        }
        
        if (type != kMine && type != kFlag && !hq(pos)) {
            bool visited[kSize] = {false};
            
            auto eval_target = [&](int to) {
                if (visited[to]) return;
                visited[to] = true;
                const int target = s.board[to];
                
                double mob_val = 0.02;
                if (enemy_flag_pos >= 0 && manhattan(to, enemy_flag_pos) <= 3) {
                    mob_val += 0.03;
                    f.enemy_mobility_near_flag += 1.0;
                }
                
                if (target == 0) {
                    int enemy_max_rank = player == kRed ? cmap.black_max_rank[to] : cmap.red_max_rank[to];
                    bool enemy_has_bomb = player == kRed ? cmap.black_has_bomb[to] : cmap.red_has_bomb[to];
                    
                    if ((enemy_max_rank == 0 && !enemy_has_bomb) || (type >= enemy_max_rank && !enemy_has_bomb)) {
                        f.mobility += mob_val;
                    }
                } else if (owned(target, -player) && !camp(to)) {
                    int result = combat(piece, target);
                    if (result > 0) {
                        int defender_max_rank = player == kRed ? cmap.black_max_rank[to] : cmap.red_max_rank[to];
                        bool defender_has_bomb = player == kRed ? cmap.black_has_bomb[to] : cmap.red_has_bomb[to];
                        
                        if (defender_has_bomb || defender_max_rank >= type) {
                            f.threat_score += std::max(0.0, value_of(target) - value_of(piece)) * 0.5;
                        } else {
                            f.threat_score += value_of(target) * 0.5;
                        }
                    } else if (result == 0) {
                        f.threat_score += std::max(0.0, value_of(target) - value_of(piece)) * 0.5;
                    }
                }
            };

            // Highway moves (8 directions)
            static constexpr int dr8[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
            static constexpr int dc8[8] = {0, 0, -1, 1, -1, 1, -1, 1};
            for (int d = 0; d < 8; ++d) {
                const int nr = row(pos) + dr8[d], nc = col(pos) + dc8[d];
                if (nr < 0 || nr >= kRows || nc < 0 || nc >= kCols) continue;
                if (d >= 4 && !camp(pos) && !camp(nr * kCols + nc)) continue;
                const int to = nr * kCols + nc;
                if (!mountain_blocked(pos, to)) {
                    eval_target(to);
                }
            }

            // Railway moves
            if (railway(pos)) {
                if (type == kSapper) {
                    f.sapper_rail_bonus += 1.5;
                    std::array<int, kSize> queue{};
                    int head = 0, tail = 0;
                    queue[tail++] = pos;
                    bool bfs_visited[kSize] = {false};
                    bfs_visited[pos] = true;
                    
                    while (head < tail) {
                        const int current = queue[head++];
                        for (int d = 0; d < 4; ++d) {
                            const int nr = row(current) + dr[d], nc = col(current) + dc[d];
                            if (nr < 0 || nr >= kRows || nc < 0 || nc >= kCols) continue;
                            const int to = nr * kCols + nc;
                            if (!railway(to) || bfs_visited[to] || mountain_blocked(current, to)) continue;
                            
                            bfs_visited[to] = true;
                            eval_target(to);
                            
                            if (s.board[to] == 0) {
                                queue[tail++] = to;
                            }
                        }
                    }
                } else {
                    for (int d = 0; d < 4; ++d) {
                        int nr = row(pos) + dr[d], nc = col(pos) + dc[d], previous = pos;
                        while (nr >= 0 && nr < kRows && nc >= 0 && nc < kCols) {
                            const int to = nr * kCols + nc;
                            if (!railway(to) || mountain_blocked(previous, to)) break;
                            
                            eval_target(to);
                            
                            if (s.board[to] != 0) break;
                            
                            previous = to;
                            nr += dr[d]; nc += dc[d];
                        }
                    }
                }
            }
        }
        
        f.material += value_of(piece);
        
        if (type != kMine && type != kFlag) {
            int enemy_global_max_rank = (player == kRed) ? cmap.global_black_max_rank : cmap.global_red_max_rank;
            bool enemy_global_has_bomb = (player == kRed) ? cmap.global_black_has_bomb : cmap.global_red_has_bomb;
            
            if (type > enemy_global_max_rank) {
                if (!enemy_global_has_bomb) {
                    f.dominance_bonus += 15.0;
                } else {
                    f.dominance_bonus += 3.0;
                }
            }
            
            const int advance = (player == kRed ? row(pos) : 11 - row(pos));
            f.advance += advance * (type >= kColonel ? 0.11 : 0.055);
            const int d = hq_distance(pos, -player);
            f.hq_proximity += std::max(0, 12 - d) * (type >= kColonel ? 0.06 : 0.025);
            if (camp(pos)) f.camp_bonus += 0.45;
        }
        if (type == kFlag) {
            for (int d = 0; d < 4; ++d) {
                const int nr = row(pos) + dr[d], nc = col(pos) + dc[d];
                if (nr < 0 || nr >= kRows || nc < 0 || nc >= kCols) continue;
                const int neighbor = s.board[nr * kCols + nc];
                if (owned(neighbor, player)) {
                    if (type_of(neighbor) == kMine) f.flag_defense += 2.5;
                    else if (type_of(neighbor) != kFlag) f.flag_defense += 0.7;
                }
            }
        }
    }
    return f;
}

double evaluate(const State& s, int perspective) {
    int red_flag = find_flag_fast(s, kRed);
    int black_flag = find_flag_fast(s, kBlack);
    
    BoardControlMap cmap = build_control_map(s);
    
    EvalFeatures red = compute_features(s, kRed, black_flag, cmap);
    EvalFeatures black = compute_features(s, kBlack, red_flag, cmap);
    
    double red_flag_vulnerability = std::max(0.0, 5.0 - red.flag_defense);
    double black_flag_vulnerability = std::max(0.0, 5.0 - black.flag_defense);
    
    double red_threat_amplified = red.threat_score + red.enemy_mobility_near_flag * black_flag_vulnerability * 0.5;
    double black_threat_amplified = black.threat_score + black.enemy_mobility_near_flag * red_flag_vulnerability * 0.5;
    
    double red_score = red.material + red.advance + red.hq_proximity + red.camp_bonus + red.flag_defense + red.mobility + red_threat_amplified + red.sapper_rail_bonus + red.dominance_bonus;
    double black_score = black.material + black.advance + black.hq_proximity + black.camp_bonus + black.flag_defense + black.mobility + black_threat_amplified + black.sapper_rail_bonus + black.dominance_bonus;
    
    return perspective == kRed ? (red_score - black_score) : (black_score - red_score);
}

struct Phase {
    int own_flag = -1;
    int enemy_flag = -1;
    bool emergency = false;
    bool contact = false;
};

Phase analyse_phase(const State& s, bool is_root) {
    Phase phase;
    phase.own_flag = find_flag_fast(s, s.player);
    phase.enemy_flag = find_flag_fast(s, -s.player);
    
    const uint8_t* enemy_pieces = (s.player == kRed) ? s.black_pieces : s.red_pieces;
    const int num_enemy_pieces = (s.player == kRed) ? s.num_black_pieces : s.num_red_pieces;
    
    for (int i = 0; i < num_enemy_pieces; ++i) {
        int p = enemy_pieces[i];
        const int piece = s.board[p];
        if (type_of(piece) == kMine || type_of(piece) == kFlag) continue;
        if (phase.own_flag >= 0 && manhattan(p, phase.own_flag) <= 5) phase.emergency = true;
    }
    
    const uint8_t* own_pieces = (s.player == kRed) ? s.red_pieces : s.black_pieces;
    const int num_own_pieces = (s.player == kRed) ? s.num_red_pieces : s.num_black_pieces;
    
    for (int i = 0; i < num_own_pieces; ++i) {
        int p = own_pieces[i];
        const int piece = s.board[p];
        if (type_of(piece) == kMine || type_of(piece) == kFlag) continue;
        if (phase.enemy_flag >= 0 && manhattan(p, phase.enemy_flag) <= 5) phase.contact = true;
    }
    
    if (!is_root) return phase;
    
    if (phase.own_flag >= 0) {
        State enemy = s;
        enemy.player = -s.player;
        enemy.hash ^= z.player_turn;
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

std::vector<int> ordered_actions(const State& s, int limit, bool is_root, int tt_best_action = -1, int killer1 = -1, int killer2 = -1) {
    std::vector<int> actions = legal_actions(s);
    const Phase phase = analyse_phase(s, is_root);
    std::stable_sort(actions.begin(), actions.end(), [&](int a, int b) {
        if (a == tt_best_action && b != tt_best_action) return true;
        if (b == tt_best_action && a != tt_best_action) return false;
        if (a == killer1 && b != killer1) return true;
        if (b == killer1 && a != killer1) return false;
        if (a == killer2 && b != killer2) return true;
        if (b == killer2 && a != killer2) return false;
        return action_score(s, a, phase) > action_score(s, b, phase);
    });
    if (static_cast<int>(actions.size()) > limit) {
        bool attacked[kSize] = {false};
        bool attacked_computed = false;
        int swap_idx = limit - 1;

        for (size_t i = limit; i < actions.size() && swap_idx >= 0; ++i) {
            if (!attacked_computed) {
                State enemy_state = s;
                enemy_state.player = -s.player;
                enemy_state.hash ^= z.player_turn;
                for (int a : legal_actions(enemy_state)) {
                    attacked[a % kSize] = true;
                }
                attacked_computed = true;
            }
            int from = actions[i] / kSize;
            if (attacked[from]) {
                while (swap_idx >= 0 && attacked[actions[swap_idx] / kSize]) {
                    swap_idx--;
                }
                if (swap_idx >= 0) {
                    std::swap(actions[i], actions[swap_idx]);
                    swap_idx--;
                }
            }
        }
        actions.resize(limit);
    }
    return actions;
}

bool exhausted(Context& ctx) {
    return ctx.budget.tick();
}

double terminal_value(const State& s, const Context& ctx, int ply) {
    if (!s.terminal) return std::numeric_limits<double>::quiet_NaN();
    if (s.winner == 0) return 0.0;
    return s.winner == ctx.root_player ? kMate - ply : -kMate + ply;
}

double alphabeta(const State& s, int depth, double alpha, double beta, int ply, Context& ctx) {
    const double terminal = terminal_value(s, ctx, ply);
    if (!std::isnan(terminal)) return terminal;
    if (exhausted(ctx)) return evaluate(s, ctx.root_player);

    bool is_qs = (depth <= 0);
    double stand_pat = 0.0;
    const bool maximizing = s.player == ctx.root_player;

    if (is_qs) {
        stand_pat = evaluate(s, ctx.root_player);
        if (maximizing) {
            if (stand_pat >= beta) return beta;
            alpha = std::max(alpha, stand_pat);
        } else {
            if (stand_pat <= alpha) return alpha;
            beta = std::min(beta, stand_pat);
        }
    }

    uint64_t key = hash_state(s);
    size_t tt_index = key % ctx.tt.size();
    TTEntry& tte = ctx.tt[tt_index];
    int tt_best_action = -1;

    if (tte.key == key) {
        tt_best_action = tte.best_action;
        if (tte.depth >= depth) {
            if (tte.bound == 0) return tte.score;
            if (tte.bound == 1) alpha = std::max(alpha, tte.score);
            if (tte.bound == 2) beta = std::min(beta, tte.score);
            if (alpha >= beta) return tte.score;
        }
    }

    std::vector<int> actions;
    Phase phase;
    if (is_qs) {
        actions = capture_actions(s);
        std::stable_sort(actions.begin(), actions.end(), [&](int a, int b) {
            return mvv_lva(s, a) > mvv_lva(s, b);
        });
    } else {
        phase = analyse_phase(s, false);
        const int width = 24;
        int killer1 = ctx.killers.get_killer1(ply);
        int killer2 = ctx.killers.get_killer2(ply);
        actions = ordered_actions(s, width, false, tt_best_action, killer1, killer2);
    }

    if (actions.empty()) {
        if (is_qs) return stand_pat;
        return s.player == ctx.root_player ? -kMate + ply : kMate - ply;
    }
    
    double best = is_qs ? stand_pat : (maximizing ? -kMate : kMate);
    int best_action = -1;
    double original_alpha = alpha;
    double original_beta = beta;

    int move_count = 0;
    for (int action : actions) {
        if (is_qs) {
            const int from = action / kSize, to = action % kSize;
            const int attacker = s.board[from], defender = s.board[to];
            const int a_type = type_of(attacker), d_type = type_of(defender);

            bool is_asymmetric = (a_type == kBomb || d_type == kBomb || 
                                  (a_type == kSapper && d_type == kMine) || 
                                  d_type == kFlag);

            if (!is_asymmetric) {
                double max_gain = value_of(defender) * 1.5 + 3.0;
                if (maximizing) {
                    if (stand_pat + max_gain < alpha) continue;
                } else {
                    if (stand_pat - max_gain > beta) continue;
                }
            }
        }

        move_count++;
        double score;
        
        if (!is_qs) {
            bool do_lmr = (move_count > 4) && !phase.emergency && (depth >= 2);
            
            if (do_lmr) {
                score = alphabeta(apply(s, action), depth - 2, alpha, beta, ply + 1, ctx);
                if (maximizing) {
                    if (score > alpha) {
                        score = alphabeta(apply(s, action), depth - 1, alpha, beta, ply + 1, ctx);
                    }
                } else {
                    if (score < beta) {
                        score = alphabeta(apply(s, action), depth - 1, alpha, beta, ply + 1, ctx);
                    }
                }
            } else {
                score = alphabeta(apply(s, action), depth - 1, alpha, beta, ply + 1, ctx);
            }
        } else {
            score = alphabeta(apply(s, action), depth - 1, alpha, beta, ply + 1, ctx);
        }

        if (maximizing) { 
            if (score > best) { best = score; best_action = action; }
            alpha = std::max(alpha, best); 
        } else { 
            if (score < best) { best = score; best_action = action; }
            beta = std::min(beta, best); 
        }
        if (ctx.budget.aborted || alpha >= beta) {
            if (alpha >= beta && !is_qs) {
                ctx.killers.store(ply, action);
            }
            break;
        }
    }

    if (!ctx.budget.aborted) {
        tte.key = key;
        tte.score = best;
        tte.depth = depth;
        tte.best_action = best_action;
        if (maximizing) {
            if (best <= original_alpha) tte.bound = 2; // UPPER
            else if (best >= beta) tte.bound = 1; // LOWER
            else tte.bound = 0; // EXACT
        } else {
            if (best >= original_beta) tte.bound = 1; // LOWER
            else if (best <= alpha) tte.bound = 2; // UPPER
            else tte.bound = 0; // EXACT
        }
    }

    return best;
}

double root_alphabeta(const State& root, int depth, double alpha, double beta, int& best_action, Context& ctx) {
    uint64_t key = hash_state(root);
    size_t tt_index = key % ctx.tt.size();
    int tt_best_action = ctx.tt[tt_index].key == key ? ctx.tt[tt_index].best_action : -1;

    int killer1 = ctx.killers.get_killer1(0);
    int killer2 = ctx.killers.get_killer2(0);
    std::vector<int> actions = ordered_actions(root, 24, true, tt_best_action, killer1, killer2);
    if (actions.empty()) return -kMate;
    
    int candidate = actions.front();
    double best = -kMate;
    double original_alpha = alpha;

    int move_count = 0;
    const Phase phase = analyse_phase(root, true);
    for (int action : actions) {
        move_count++;
        double score;
        bool do_lmr = (move_count > 4) && !phase.emergency && (depth >= 2);
        
        if (do_lmr) {
            score = alphabeta(apply(root, action), depth - 2, alpha, beta, 1, ctx);
            if (score > alpha) {
                score = alphabeta(apply(root, action), depth - 1, alpha, beta, 1, ctx);
            }
        } else {
            score = alphabeta(apply(root, action), depth - 1, alpha, beta, 1, ctx);
        }

        if (ctx.budget.aborted) {
            best_action = candidate;
            return best;
        }
        if (score > best) { best = score; candidate = action; }
        alpha = std::max(alpha, best);
        if (alpha >= beta) {
            ctx.killers.store(0, action);
            break;
        }
    }
    best_action = candidate;

    TTEntry& tte = ctx.tt[tt_index];
    tte.key = key;
    tte.score = best;
    tte.depth = depth;
    tte.best_action = best_action;
    if (best <= original_alpha) tte.bound = 2; // UPPER
    else if (best >= beta) tte.bound = 1; // LOWER
    else tte.bound = 0; // EXACT

    return best;
}

bool aspiration_window(const State& root, int depth, double& f, int& best_action, Context& ctx) {
    double delta = 25.0;
    double alpha = f - delta;
    double beta = f + delta;
    int current_best_action = best_action;
    
    while (true) {
        int candidate_action = current_best_action;
        double g = root_alphabeta(root, depth, alpha, beta, candidate_action, ctx);
        
        if (ctx.budget.aborted) {
            best_action = candidate_action;
            return false;
        }
        
        current_best_action = candidate_action;
        
        if (g <= alpha) {
            alpha = -kMate;
        } else if (g >= beta) {
            beta = kMate;
        } else {
            f = g;
            best_action = current_best_action;
            return true;
        }
    }
}

int select_action(const std::vector<int>& board, const std::vector<int>& supplied_legal, int player) {
    if (board.size() != kSize || supplied_legal.empty()) throw std::invalid_argument("invalid search input");
    State root;
    std::copy(board.begin(), board.end(), root.board.begin());
    root.player = player;
    init_state(root);
    Context ctx(player, std::chrono::steady_clock::now() + kBudget);
    const std::vector<int> fallback = ordered_actions(root, 18, true);
    int best = fallback.empty() ? supplied_legal.front() : fallback.front();
    
    double f = 0.0;
    
    for (int depth = 2; depth <= 7; ++depth) {
        if (!ctx.budget.can_start_new_depth()) break;
        int candidate = best;
        if (!aspiration_window(root, depth, f, candidate, ctx)) {
            best = candidate;
            break;
        }
        best = candidate;
    }
    if (std::find(supplied_legal.begin(), supplied_legal.end(), best) == supplied_legal.end()) return supplied_legal.front();
    return best;
}

std::vector<int> debug_legal_actions(const std::vector<int>& board, int player) {
    if (board.size() != kSize) throw std::invalid_argument("invalid board");
    State state;
    std::copy(board.begin(), board.end(), state.board.begin());
    state.player = player;
    init_state(state);
    return legal_actions(state);
}

py::tuple debug_transition(const std::vector<int>& board, int player, int action) {
    if (board.size() != kSize) throw std::invalid_argument("invalid board");
    State state;
    std::copy(board.begin(), board.end(), state.board.begin());
    state.player = player;
    init_state(state);
    const State next = apply(state, action);
    return py::make_tuple(std::vector<int>(next.board.begin(), next.board.end()), next.player,
                          next.terminal, next.winner);
}

}  // namespace

PYBIND11_MODULE(fast_rule_ab, module) {
    module.doc() = "Ephemeral exact-rule iterative-deepening alpha-beta solver";
    module.def("select_action", &select_action, py::arg("board"), py::arg("legal_actions"), py::arg("player_id"));
    module.def("debug_legal_actions", &debug_legal_actions, py::arg("board"), py::arg("player_id"));
    module.def("debug_transition", &debug_transition, py::arg("board"), py::arg("player_id"), py::arg("action"));
}
