#include "heuristic_084_evaluator.h"

#include "junqi_board.h"
#include "junqi_types.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace junqi::az {
namespace {

constexpr int kSize = BOARD_SIZE;
constexpr int kRed = 1;
constexpr int kBlack = -1;

int row(int position) { return position / BOARD_COLS; }
int col(int position) { return position % BOARD_COLS; }
int manhattan(int left, int right) {
    return std::abs(row(left) - row(right)) + std::abs(col(left) - col(right));
}
bool owned(int piece, int player) {
    return piece != 0 && ((piece > 0) == (player > 0));
}
int type_of(int piece) { return std::abs(piece); }

const JunqiBoard& terrain() {
    static const JunqiBoard board;
    return board;
}

double value_of(int piece) {
    switch (type_of(piece)) {
        case FIELD_MARSHAL: return 9.0;
        case ARMY_CMDR: return 8.0;
        case GENERAL: return 7.0;
        case BRIGADIER: return 6.0;
        case COLONEL: return 5.0;
        case MAJOR: return 4.0;
        case CAPTAIN: return 3.0;
        case LIEUTENANT: return 2.0;
        case SAPPER: return 4.5;
        case BOMB: return 7.0;
        case MINE: return 8.0;
        case FLAG: return 25.0;
        default: return 0.0;
    }
}

int combat(int attacker, int defender) {
    const int a = type_of(attacker), d = type_of(defender);
    if (d == FLAG) return 1;
    if (a == BOMB || d == BOMB) return 0;
    if (d == MINE) return a == SAPPER ? 1 : -1;
    if (a == MINE || a == FLAG) return -1;
    return a == d ? 0 : (a > d ? 1 : -1);
}

int find_flag(const int* board, int player) {
    for (int position = 0; position < kSize; ++position) {
        if (owned(board[position], player) && type_of(board[position]) == FLAG) {
            return position;
        }
    }
    return -1;
}

int hq_distance(int position, int side) {
    const int first = side == kRed ? 1 : 56;
    const int second = side == kRed ? 3 : 58;
    return std::min(manhattan(position, first), manhattan(position, second));
}

struct BoardControlMap {
    std::array<int, kSize> red_max_rank{};
    std::array<bool, kSize> red_has_bomb{};
    std::array<int, kSize> black_max_rank{};
    std::array<bool, kSize> black_has_bomb{};
    int global_red_max_rank = 0;
    bool global_red_has_bomb = false;
    int global_black_max_rank = 0;
    bool global_black_has_bomb = false;
};

BoardControlMap build_control_map(const int* board) {
    BoardControlMap map;
    static constexpr int dr[4] = {-1, 1, 0, 0};
    static constexpr int dc[4] = {0, 0, -1, 1};
    static constexpr int dr8[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    static constexpr int dc8[8] = {0, 0, -1, 1, -1, 1, -1, 1};

    for (int position = 0; position < kSize; ++position) {
        const int piece = board[position];
        if (piece == 0) continue;
        const int type = type_of(piece);
        if (type == MINE || type == FLAG || terrain().is_hq(position)) continue;
        const int player = piece > 0 ? kRed : kBlack;
        if (player == kRed) {
            if (type == BOMB) map.global_red_has_bomb = true;
            else map.global_red_max_rank = std::max(map.global_red_max_rank, type);
        } else {
            if (type == BOMB) map.global_black_has_bomb = true;
            else map.global_black_max_rank = std::max(map.global_black_max_rank, type);
        }

        auto add_attack = [&](int to) {
            if (terrain().is_camp(to) && board[to] != 0) return;
            if (player == kRed) {
                if (type == BOMB) map.red_has_bomb[to] = true;
                else map.red_max_rank[to] = std::max(map.red_max_rank[to], type);
            } else {
                if (type == BOMB) map.black_has_bomb[to] = true;
                else map.black_max_rank[to] = std::max(map.black_max_rank[to], type);
            }
        };

        for (int direction = 0; direction < 8; ++direction) {
            const int nr = row(position) + dr8[direction];
            const int nc = col(position) + dc8[direction];
            if (nr < 0 || nr >= BOARD_ROWS || nc < 0 || nc >= BOARD_COLS) continue;
            const int to = nr * BOARD_COLS + nc;
            if (direction >= 4 && !terrain().is_camp(position) && !terrain().is_camp(to)) continue;
            if (!terrain().is_mountain_blocked(position, to)) add_attack(to);
        }

        if (!terrain().is_railway(position)) continue;
        if (type == SAPPER) {
            std::array<int, kSize> queue{};
            std::array<bool, kSize> visited{};
            int head = 0, tail = 0;
            queue[tail++] = position;
            visited[position] = true;
            while (head < tail) {
                const int current = queue[head++];
                for (int direction = 0; direction < 4; ++direction) {
                    const int nr = row(current) + dr[direction];
                    const int nc = col(current) + dc[direction];
                    if (nr < 0 || nr >= BOARD_ROWS || nc < 0 || nc >= BOARD_COLS) continue;
                    const int to = nr * BOARD_COLS + nc;
                    if (!terrain().is_railway(to) || visited[to] ||
                        terrain().is_mountain_blocked(current, to)) continue;
                    visited[to] = true;
                    add_attack(to);
                    if (board[to] == 0) queue[tail++] = to;
                }
            }
        } else {
            for (int direction = 0; direction < 4; ++direction) {
                int nr = row(position) + dr[direction];
                int nc = col(position) + dc[direction];
                int previous = position;
                while (nr >= 0 && nr < BOARD_ROWS && nc >= 0 && nc < BOARD_COLS) {
                    const int to = nr * BOARD_COLS + nc;
                    if (!terrain().is_railway(to) ||
                        terrain().is_mountain_blocked(previous, to)) break;
                    add_attack(to);
                    if (board[to] != 0) break;
                    previous = to;
                    nr += dr[direction];
                    nc += dc[direction];
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
    double enemy_mobility_near_flag = 0.0;
};

EvalFeatures compute_features(const int* board, int player, int enemy_flag,
                              const BoardControlMap& control) {
    EvalFeatures result;
    static constexpr int dr[4] = {-1, 1, 0, 0};
    static constexpr int dc[4] = {0, 0, -1, 1};
    static constexpr int dr8[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    static constexpr int dc8[8] = {0, 0, -1, 1, -1, 1, -1, 1};

    for (int position = 0; position < kSize; ++position) {
        const int piece = board[position];
        if (!owned(piece, player)) continue;
        const int type = type_of(piece);
        if (type != MINE && type != FLAG && !terrain().is_hq(position)) {
            std::array<bool, kSize> visited{};
            auto eval_target = [&](int to) {
                if (visited[to]) return;
                visited[to] = true;
                const int target = board[to];
                double mobility = 0.02;
                if (enemy_flag >= 0 && manhattan(to, enemy_flag) <= 3) {
                    mobility += 0.03;
                    result.enemy_mobility_near_flag += 1.0;
                }
                if (target == 0) {
                    const int enemy_rank = player == kRed
                        ? control.black_max_rank[to] : control.red_max_rank[to];
                    const bool enemy_bomb = player == kRed
                        ? control.black_has_bomb[to] : control.red_has_bomb[to];
                    if ((enemy_rank == 0 && !enemy_bomb) ||
                        (type >= enemy_rank && !enemy_bomb)) result.mobility += mobility;
                } else if (owned(target, -player) && !terrain().is_camp(to)) {
                    const int outcome = combat(piece, target);
                    if (outcome > 0) {
                        const int defender_rank = player == kRed
                            ? control.black_max_rank[to] : control.red_max_rank[to];
                        const bool defender_bomb = player == kRed
                            ? control.black_has_bomb[to] : control.red_has_bomb[to];
                        if (defender_bomb || defender_rank >= type) {
                            result.threat_score +=
                                std::max(0.0, value_of(target) - value_of(piece)) * 0.5;
                        } else {
                            result.threat_score += value_of(target) * 0.5;
                        }
                    } else if (outcome == 0) {
                        result.threat_score +=
                            std::max(0.0, value_of(target) - value_of(piece)) * 0.5;
                    }
                }
            };

            for (int direction = 0; direction < 8; ++direction) {
                const int nr = row(position) + dr8[direction];
                const int nc = col(position) + dc8[direction];
                if (nr < 0 || nr >= BOARD_ROWS || nc < 0 || nc >= BOARD_COLS) continue;
                const int to = nr * BOARD_COLS + nc;
                if (direction >= 4 && !terrain().is_camp(position) &&
                    !terrain().is_camp(to)) continue;
                if (!terrain().is_mountain_blocked(position, to)) eval_target(to);
            }

            if (terrain().is_railway(position)) {
                if (type == SAPPER) {
                    result.sapper_rail_bonus += 1.5;
                    std::array<int, kSize> queue{};
                    std::array<bool, kSize> rail_visited{};
                    int head = 0, tail = 0;
                    queue[tail++] = position;
                    rail_visited[position] = true;
                    while (head < tail) {
                        const int current = queue[head++];
                        for (int direction = 0; direction < 4; ++direction) {
                            const int nr = row(current) + dr[direction];
                            const int nc = col(current) + dc[direction];
                            if (nr < 0 || nr >= BOARD_ROWS || nc < 0 || nc >= BOARD_COLS) continue;
                            const int to = nr * BOARD_COLS + nc;
                            if (!terrain().is_railway(to) || rail_visited[to] ||
                                terrain().is_mountain_blocked(current, to)) continue;
                            rail_visited[to] = true;
                            eval_target(to);
                            if (board[to] == 0) queue[tail++] = to;
                        }
                    }
                } else {
                    for (int direction = 0; direction < 4; ++direction) {
                        int nr = row(position) + dr[direction];
                        int nc = col(position) + dc[direction];
                        int previous = position;
                        while (nr >= 0 && nr < BOARD_ROWS && nc >= 0 && nc < BOARD_COLS) {
                            const int to = nr * BOARD_COLS + nc;
                            if (!terrain().is_railway(to) ||
                                terrain().is_mountain_blocked(previous, to)) break;
                            eval_target(to);
                            if (board[to] != 0) break;
                            previous = to;
                            nr += dr[direction];
                            nc += dc[direction];
                        }
                    }
                }
            }
        }

        result.material += value_of(piece);
        if (type != MINE && type != FLAG) {
            const int enemy_max = player == kRed
                ? control.global_black_max_rank : control.global_red_max_rank;
            const bool enemy_bomb = player == kRed
                ? control.global_black_has_bomb : control.global_red_has_bomb;
            if (type > enemy_max) result.dominance_bonus += enemy_bomb ? 3.0 : 15.0;
            const int progress = player == kRed ? row(position) : 11 - row(position);
            result.advance += progress * (type >= COLONEL ? 0.11 : 0.055);
            const int distance = hq_distance(position, -player);
            result.hq_proximity +=
                std::max(0, 12 - distance) * (type >= COLONEL ? 0.06 : 0.025);
            if (terrain().is_camp(position)) result.camp_bonus += 0.45;
        }
        if (type == FLAG) {
            for (int direction = 0; direction < 4; ++direction) {
                const int nr = row(position) + dr[direction];
                const int nc = col(position) + dc[direction];
                if (nr < 0 || nr >= BOARD_ROWS || nc < 0 || nc >= BOARD_COLS) continue;
                const int neighbor = board[nr * BOARD_COLS + nc];
                if (!owned(neighbor, player)) continue;
                if (type_of(neighbor) == MINE) result.flag_defense += 2.5;
                else if (type_of(neighbor) != FLAG) result.flag_defense += 0.7;
            }
        }
    }
    return result;
}

} // namespace

double heuristic_084_raw_value(const int* board, int perspective) {
    const int red_flag = find_flag(board, kRed);
    const int black_flag = find_flag(board, kBlack);
    const BoardControlMap control = build_control_map(board);
    const EvalFeatures red = compute_features(board, kRed, black_flag, control);
    const EvalFeatures black = compute_features(board, kBlack, red_flag, control);
    const double red_vulnerability = std::max(0.0, 5.0 - red.flag_defense);
    const double black_vulnerability = std::max(0.0, 5.0 - black.flag_defense);
    const double red_threat = red.threat_score +
        red.enemy_mobility_near_flag * black_vulnerability * 0.5;
    const double black_threat = black.threat_score +
        black.enemy_mobility_near_flag * red_vulnerability * 0.5;
    const double red_score = red.material + red.advance + red.hq_proximity +
        red.camp_bonus + red.flag_defense + red.mobility + red_threat +
        red.sapper_rail_bonus + red.dominance_bonus;
    const double black_score = black.material + black.advance + black.hq_proximity +
        black.camp_bonus + black.flag_defense + black.mobility + black_threat +
        black.sapper_rail_bonus + black.dominance_bonus;
    return perspective == kRed ? red_score - black_score : black_score - red_score;
}

} // namespace junqi::az
