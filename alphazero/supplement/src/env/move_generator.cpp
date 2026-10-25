#include "move_generator.h"
#include <queue>
#include <algorithm>
#include <cstring>

namespace junqi {

MoveGenerator::MoveGenerator(const JunqiBoard& board) : board_(board) {}

std::vector<std::pair<int, int>> MoveGenerator::generate_legal_moves(
    const JunqiState& state) const {
    std::vector<std::pair<int, int>> moves;
    PlayerId player = state.get_current_player();
    
    // 遍历所有位置
    for (int from = 0; from < BOARD_SIZE; ++from) {
        const Piece& piece = state.get_piece(from);
        
        // 跳过空位
        if (piece.is_empty()) {
            continue;
        }
        
        // 跳过敌方棋子
        if (!piece.is_owned_by(player)) {
            continue;
        }
        
        // 跳过不可移动的棋子（地雷、军旗）
        if (!is_movable_piece(piece.type)) {
            continue;
        }
        
        // 跳过大本营内的棋子
        if (board_.is_hq(from)) {
            continue;
        }
        
        // 生成该棋子的所有合法移动
        std::vector<int> destinations = generate_piece_moves(state, from);
        
        // 添加到动作列表
        for (int to : destinations) {
            moves.push_back(std::make_pair(from, to));
        }
    }
    
    return moves;
}

std::vector<bool> MoveGenerator::get_legal_action_mask(const JunqiState& state,
                                                        PlayerId player) const {
    std::vector<bool> mask(BOARD_SIZE * BOARD_SIZE, false);
    
    // 遍历所有位置
    for (int from = 0; from < BOARD_SIZE; ++from) {
        const Piece& piece = state.get_piece(from);
        
        // 跳过空位
        if (piece.is_empty()) {
            continue;
        }
        
        // 跳过敌方棋子
        if (!piece.is_owned_by(player)) {
            continue;
        }
        
        // 跳过不可移动的棋子（地雷、军旗）
        if (!is_movable_piece(piece.type)) {
            continue;
        }
        
        // 跳过大本营内的棋子
        if (board_.is_hq(from)) {
            continue;
        }
        
        // 生成该棋子的所有合法移动
        std::vector<int> destinations = generate_piece_moves(state, from);
        
        // 更新掩码
        for (int to : destinations) {
            mask[from * BOARD_SIZE + to] = true;
        }
    }
    
    return mask;
}

void MoveGenerator::fill_legal_action_mask(const JunqiState& state,
                                           PlayerId player,
                                           std::uint8_t* out) const {
    std::memset(out, 0, BOARD_SIZE * BOARD_SIZE * sizeof(std::uint8_t));
    for (int from = 0; from < BOARD_SIZE; ++from) {
        const Piece& piece = state.get_piece(from);
        if (piece.is_empty() || !piece.is_owned_by(player) ||
            !is_movable_piece(piece.type) || board_.is_hq(from)) {
            continue;
        }
        const std::vector<int> destinations = generate_piece_moves(state, from);
        for (int to : destinations) {
            out[from * BOARD_SIZE + to] = 1;
        }
    }
}

std::vector<int> MoveGenerator::get_possible_moves(const JunqiState& state, int from) const {
    return generate_piece_moves(state, from);
}

bool MoveGenerator::is_move_valid(const JunqiState& state, int from, int to) const {
    // 检查边界
    if (from < 0 || from >= BOARD_SIZE || to < 0 || to >= BOARD_SIZE) {
        return false;
    }
    
    // 检查起点和终点不同
    if (from == to) {
        return false;
    }
    
    const Piece& from_piece = state.get_piece(from);
    PlayerId player = state.get_current_player();
    
    // 检查起点是否有己方棋子
    if (from_piece.is_empty() || !from_piece.is_owned_by(player)) {
        return false;
    }
    
    // 检查棋子是否可移动
    if (!is_movable_piece(from_piece.type)) {
        return false;
    }
    
    // 检查大本营内的棋子
    if (board_.is_hq(from)) {
        return false;
    }
    
    // 检查是否可以进入目标位置
    if (!can_enter(state, from, to, player)) {
        return false;
    }
    
    // 检查是否是合法的移动类型
    std::vector<int> destinations = generate_piece_moves(state, from);
    
    return std::find(destinations.begin(), destinations.end(), to) != destinations.end();
}

std::vector<int> MoveGenerator::generate_piece_moves(const JunqiState& state, int from) const {
    std::vector<int> moves;
    const Piece& piece = state.get_piece(from);
    
    // 根据棋子类型和位置生成移动
    if (piece.type == SAPPER && board_.is_railway(from)) {
        // 工兵在铁路上：可以拐弯
        moves = generate_sapper_railway_moves(state, from);
        // 加上公路移动（包括到相邻公路位置的移动）
        std::vector<int> highway_moves = generate_highway_moves(state, from);
        moves.insert(moves.end(), highway_moves.begin(), highway_moves.end());
    } else if (board_.is_railway(from)) {
        // 其他棋子在铁路上：只能直冲
        moves = generate_railway_moves(state, from);
        // 加上公路移动（包括到相邻公路位置的移动）
        std::vector<int> highway_moves = generate_highway_moves(state, from);
        moves.insert(moves.end(), highway_moves.begin(), highway_moves.end());
    } else {
        // 不在铁路上：只能公路移动
        moves = generate_highway_moves(state, from);
    }
    
    return moves;
}

std::vector<int> MoveGenerator::generate_highway_moves(const JunqiState& state, int from) const {
    std::vector<int> moves;
    std::vector<int> neighbors = board_.get_neighbors(from);
    PlayerId player = state.get_piece(from).owner;
    
    for (int to : neighbors) {
        if (can_enter(state, from, to, player)) {
            moves.push_back(to);
        }
    }
    
    return moves;
}

std::vector<int> MoveGenerator::generate_railway_moves(const JunqiState& state, int from) const {
    std::vector<int> moves;
    PlayerId player = state.get_piece(from).owner;
    
    // 检查四个方向的铁路直冲
    int r, c;
    index_to_pos(from, r, c);
    
    int dr[] = {-1, 1, 0, 0};
    int dc[] = {0, 0, -1, 1};
    
    for (int d = 0; d < 4; ++d) {
        int nr = r + dr[d];
        int nc = c + dc[d];
        
        // 保存上一个位置，用于山界检查
        int prev_pos = from;
        
        // 沿着铁路方向前进
        while (nr >= 0 && nr < BOARD_ROWS && nc >= 0 && nc < BOARD_COLS) {
            int to = pos_to_index(nr, nc);
            
            // 必须是铁路节点
            if (!board_.is_railway(nr, nc)) {
                break;
            }
            
            // 检查山界阻断（每次移动一步都要检查）
            if (board_.is_mountain_blocked(prev_pos, to)) {
                break;
            }
            
            // 检查该位置是否有棋子
            const Piece& to_piece = state.get_piece(to);
            
            if (!to_piece.is_empty()) {
                // 有棋子，检查是否可以攻击
                if (can_enter(state, from, to, player)) {
                    moves.push_back(to);  // 只能攻击这一个位置
                }
                break;  // 无论是否可以攻击，都停止（不能跳过棋子）
            }
            
            // 空位，可以移动到这个位置
            moves.push_back(to);
            
            // 更新上一个位置为当前位置，用于下一步的山界检查
            prev_pos = to;
            
            // 继续前进
            nr += dr[d];
            nc += dc[d];
        }
    }
    
    return moves;
}

std::vector<int> MoveGenerator::generate_sapper_railway_moves(const JunqiState& state, int start) const {
    // 工兵可以在铁路上自由移动（包括拐弯），使用BFS搜索
    return bfs_railway_search(state, start);
}

std::vector<int> MoveGenerator::bfs_railway_search(const JunqiState& state, int start) const {
    std::vector<int> reachable;
    std::vector<bool> visited(BOARD_SIZE, false);
    std::queue<int> queue;
    
    PlayerId player = state.get_piece(start).owner;
    
    queue.push(start);
    visited[start] = true;
    
    const auto& adj = board_.get_railway_adjacency();
    
    while (!queue.empty()) {
        int current = queue.front();
        queue.pop();
        
        // 遍历所有铁路邻居
        for (int neighbor : adj[current]) {
            if (visited[neighbor]) {
                continue;
            }
            
            // 检查该位置是否有棋子
            const Piece& neighbor_piece = state.get_piece(neighbor);
            
            if (!neighbor_piece.is_empty()) {
                // 有棋子，检查是否可以攻击
                if (can_enter(state, start, neighbor, player)) {
                    reachable.push_back(neighbor);
                    visited[neighbor] = true;
                }
                // 不能继续通过（停止BFS在该方向）
                continue;
            }
            
            // 空位，可以进入并继续BFS
            reachable.push_back(neighbor);
            visited[neighbor] = true;
            queue.push(neighbor);
        }
    }
    
    return reachable;
}

bool MoveGenerator::is_railway_path_clear(const JunqiState& state, int from, int to) const {
    // 检查是否在同一铁路线上
    if (!board_.is_on_same_railway_line(from, to)) {
        return false;
    }
    
    return is_railway_line_clear(state, from, to);
}

bool MoveGenerator::is_railway_line_clear(const JunqiState& state, int from, int to) const {
    std::vector<int> between = get_positions_between(from, to);
    
    for (int pos : between) {
        if (!state.get_piece(pos).is_empty()) {
            return false;
        }
    }
    
    return true;
}

bool MoveGenerator::can_enter(const JunqiState& state, int from, int to, PlayerId player) const {
    const Piece& to_piece = state.get_piece(to);
    
    // 如果目标是空位，可以进入
    if (to_piece.is_empty()) {
        return true;
    }
    
    // 如果目标是敌方棋子，可以攻击
    if (to_piece.owner != player) {
        // 检查行营保护：如果目标在行营内，不可进入
        if (board_.is_camp(to)) {
            return false;
        }
        return true;
    }
    
    // 如果是己方棋子，不能进入
    return false;
}

std::vector<int> MoveGenerator::get_positions_between(int from, int to) const {
    std::vector<int> positions;
    
    int r1, c1, r2, c2;
    index_to_pos(from, r1, c1);
    index_to_pos(to, r2, c2);
    
    // 必须在同一行或同一列
    if (r1 != r2 && c1 != c2) {
        return positions;
    }
    
    if (r1 == r2) {
        // 同一行
        int start = std::min(c1, c2) + 1;
        int end = std::max(c1, c2);
        for (int col = start; col < end; ++col) {
            positions.push_back(pos_to_index(r1, col));
        }
    } else {
        // 同一列
        int start = std::min(r1, r2) + 1;
        int end = std::max(r1, r2);
        for (int row = start; row < end; ++row) {
            positions.push_back(pos_to_index(row, c1));
        }
    }
    
    return positions;
}

} // namespace junqi
