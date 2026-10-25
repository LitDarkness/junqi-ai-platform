#include "junqi_board.h"
#include <algorithm>
#include <queue>

namespace junqi {

JunqiBoard::JunqiBoard() {
    init_terrain();
    init_special_positions();
    init_railway_adjacency();
}

void JunqiBoard::init_terrain() {
    // 初始化 12x5 地形矩阵
    terrain_.resize(BOARD_ROWS, std::vector<Terrain>(BOARD_COLS, NORMAL));
    
    // 设置地形 (根据需求文档的地图布局)
    // Row 11 (Black side top)
    terrain_[11][1] = HQ;
    terrain_[11][3] = HQ;
    
    // Row 10 - 6 (Black side)
    for (int col = 0; col < BOARD_COLS; ++col) {
        terrain_[10][col] = RAILWAY;
        terrain_[6][col] = RAILWAY;
    }
    
    // Row 9
    terrain_[9][0] = RAILWAY;
    terrain_[9][1] = CAMP;
    terrain_[9][2] = NORMAL;
    terrain_[9][3] = CAMP;
    terrain_[9][4] = RAILWAY;
    
    // Row 8
    terrain_[8][0] = RAILWAY;
    terrain_[8][1] = NORMAL;
    terrain_[8][2] = CAMP;
    terrain_[8][3] = NORMAL;
    terrain_[8][4] = RAILWAY;
    
    // Row 7
    terrain_[7][0] = RAILWAY;
    terrain_[7][1] = CAMP;
    terrain_[7][2] = NORMAL;
    terrain_[7][3] = CAMP;
    terrain_[7][4] = RAILWAY;
    
    // Row 5 - 1 (Red side)
    for (int col = 0; col < BOARD_COLS; ++col) {
        terrain_[5][col] = RAILWAY;
        terrain_[1][col] = RAILWAY;
    }
    
    // Row 4
    terrain_[4][0] = RAILWAY;
    terrain_[4][1] = CAMP;
    terrain_[4][2] = NORMAL;
    terrain_[4][3] = CAMP;
    terrain_[4][4] = RAILWAY;
    
    // Row 3
    terrain_[3][0] = RAILWAY;
    terrain_[3][1] = NORMAL;
    terrain_[3][2] = CAMP;
    terrain_[3][3] = NORMAL;
    terrain_[3][4] = RAILWAY;
    
    // Row 2
    terrain_[2][0] = RAILWAY;
    terrain_[2][1] = CAMP;
    terrain_[2][2] = NORMAL;
    terrain_[2][3] = CAMP;
    terrain_[2][4] = RAILWAY;
    
    // Row 0 (Red side bottom)
    terrain_[0][1] = HQ;
    terrain_[0][3] = HQ;
    // Row 0 的其余位置保持 NORMAL
}

void JunqiBoard::init_special_positions() {
    // 初始化行营位置
    // Red side camps: (2,1), (2,3), (3,2), (4,1), (4,3)
    camp_positions_ = {
        pos_to_index(2, 1), pos_to_index(2, 3),
        pos_to_index(3, 2),
        pos_to_index(4, 1), pos_to_index(4, 3),
        // Black side camps: (7,1), (7,3), (8,2), (9,1), (9,3)
        pos_to_index(7, 1), pos_to_index(7, 3),
        pos_to_index(8, 2),
        pos_to_index(9, 1), pos_to_index(9, 3)
    };
    
    // 初始化大本营位置
    // Red side HQ: (0,1), (0,3)
    // Black side HQ: (11,1), (11,3)
    hq_positions_ = {
        pos_to_index(0, 1), pos_to_index(0, 3),
        pos_to_index(11, 1), pos_to_index(11, 3)
    };
    
    // 填充集合以便快速查找
    for (int pos : camp_positions_) {
        camp_set_.insert(pos);
    }
    
    for (int pos : hq_positions_) {
        hq_set_.insert(pos);
    }
}

void JunqiBoard::init_railway_adjacency() {
    railway_adjacency_.resize(BOARD_SIZE);
    
    // 遍历所有铁路节点，建立邻接关系
    for (int row = 0; row < BOARD_ROWS; ++row) {
        for (int col = 0; col < BOARD_COLS; ++col) {
            int from = pos_to_index(row, col);
            
            if (!is_railway(row, col)) {
                continue; // 不是铁路节点，跳过
            }
            
            // 检查四个方向
            int dr[] = {-1, 1, 0, 0};
            int dc[] = {0, 0, -1, 1};
            
            for (int d = 0; d < 4; ++d) {
                int nr = row + dr[d];
                int nc = col + dc[d];
                
                // 边界检查
                if (nr < 0 || nr >= BOARD_ROWS || nc < 0 || nc >= BOARD_COLS) {
                    continue;
                }
                
                int to = pos_to_index(nr, nc);
                
                // 必须是铁路节点
                if (!is_railway(nr, nc)) {
                    continue;
                }
                
                // 检查山界阻断
                if (is_mountain_blocked(from, to)) {
                    continue;
                }
                
                // 添加邻接关系
                railway_adjacency_[from].push_back(to);
            }
        }
    }
}

Terrain JunqiBoard::get_terrain(int row, int col) const {
    if (row < 0 || row >= BOARD_ROWS || col < 0 || col >= BOARD_COLS) {
        return NORMAL;
    }
    return terrain_[row][col];
}

Terrain JunqiBoard::get_terrain(int index) const {
    int row, col;
    index_to_pos(index, row, col);
    return get_terrain(row, col);
}

std::vector<int> JunqiBoard::get_neighbors(int index) const {
    int row, col;
    index_to_pos(index, row, col);
    return get_neighbors(row, col);
}

std::vector<int> JunqiBoard::get_neighbors(int row, int col) const {
    std::vector<int> neighbors;
    
    // 检查当前位置是否在边界内
    if (row < 0 || row >= BOARD_ROWS || col < 0 || col >= BOARD_COLS) {
        return neighbors;
    }
    
    // 判断起点是否是行营
    bool from_is_camp = is_camp(row, col);
    
    // 检查八个方向（前4个是正交，后4个是对角线）
    // 正交: 上(-1,0), 下(1,0), 左(0,-1), 右(0,1)
    // 对角: 左上(-1,-1), 右上(-1,1), 左下(1,-1), 右下(1,1)
    int dr[] = {-1, 1, 0, 0, -1, -1, 1, 1};
    int dc[] = {0, 0, -1, 1, -1, 1, -1, 1};
    
    for (int d = 0; d < 8; ++d) {
        int nr = row + dr[d];
        int nc = col + dc[d];
        
        // 边界检查
        if (nr < 0 || nr >= BOARD_ROWS || nc < 0 || nc >= BOARD_COLS) {
            continue;
        }
        
        // 判断终点是否是行营
        bool to_is_camp = is_camp(nr, nc);
        
        // 规则：如果起点或终点有一个是行营，允许全部8个方向
        // 否则只允许前4个正交方向（上下左右）
        if (!from_is_camp && !to_is_camp && d >= 4) {
            continue;
        }
        
        int from = pos_to_index(row, col);
        int to = pos_to_index(nr, nc);
        
        // 检查山界阻断
        if (is_mountain_blocked(from, to)) {
            continue;
        }
        
        neighbors.push_back(to);
    }
    
    return neighbors;
}

const std::vector<std::vector<int>>& JunqiBoard::get_railway_adjacency() const {
    return railway_adjacency_;
}

bool JunqiBoard::is_camp(int row, int col) const {
    int index = pos_to_index(row, col);
    return camp_set_.find(index) != camp_set_.end();
}

bool JunqiBoard::is_camp(int index) const {
    return camp_set_.find(index) != camp_set_.end();
}

bool JunqiBoard::is_hq(int row, int col) const {
    int index = pos_to_index(row, col);
    return hq_set_.find(index) != hq_set_.end();
}

bool JunqiBoard::is_hq(int index) const {
    return hq_set_.find(index) != hq_set_.end();
}

bool JunqiBoard::is_railway(int row, int col) const {
    if (row < 0 || row >= BOARD_ROWS || col < 0 || col >= BOARD_COLS) {
        return false;
    }
    return terrain_[row][col] == RAILWAY;
}

bool JunqiBoard::is_railway(int index) const {
    int row, col;
    index_to_pos(index, row, col);
    return is_railway(row, col);
}

const std::vector<int>& JunqiBoard::get_camp_positions() const {
    return camp_positions_;
}

const std::vector<int>& JunqiBoard::get_hq_positions() const {
    return hq_positions_;
}

bool JunqiBoard::is_adjacent(int from, int to) const {
    int r1, c1, r2, c2;
    index_to_pos(from, r1, c1);
    index_to_pos(to, r2, c2);
    
    // 计算曼哈顿距离
    int dr = abs(r1 - r2);
    int dc = abs(c1 - c2);
    
    // 检查是否是山界阻断
    if (is_mountain_blocked(from, to)) {
        return false;
    }
    
    // 如果起点或终点是行营，允许对角线移动
    if (is_camp(from) || is_camp(to)) {
        return dr <= 1 && dc <= 1 && (dr + dc > 0);
    } else {
        // 否则只能走上下左右
        return dr + dc == 1;
    }
}

bool JunqiBoard::is_on_same_railway_line(int from, int to) const {
    int r1, c1, r2, c2;
    index_to_pos(from, r1, c1);
    index_to_pos(to, r2, c2);
    
    // 如果不是两个都是铁路节点，返回false
    if (!is_railway(r1, c1) || !is_railway(r2, c2)) {
        return false;
    }
    
    // 检查是否在同一行或同一列
    if (r1 != r2 && c1 != c2) {
        return false; // 不在同一直线上
    }
    
    // 检查路径上是否有山界阻断
    if (r1 == r2) {
        // 同一行，检查中间列
        int start_col = std::min(c1, c2);
        int end_col = std::max(c1, c2);
        for (int col = start_col + 1; col < end_col; ++col) {
            if (!is_railway(r1, col)) {
                return false;
            }
        }
        // 检查山界阻断
        if (is_mountain_blocked(from, to)) {
            return false;
        }
        return true;
    } else {
        // 同一列，检查中间行
        int start_row = std::min(r1, r2);
        int end_row = std::max(r1, r2);
        for (int row = start_row + 1; row < end_row; ++row) {
            if (!is_railway(row, c1)) {
                return false;
            }
        }
        // 检查山界阻断
        if (is_mountain_blocked(from, to)) {
            return false;
        }
        return true;
    }
}

bool JunqiBoard::is_mountain_blocked(int from, int to) const {
    int r1, c1, r2, c2;
    index_to_pos(from, r1, c1);
    index_to_pos(to, r2, c2);
    
    // 山界阻断位置：(5,1)-(6,1) 和 (5,3)-(6,3)
    // 检查是否跨越这些边界
    
    // 检查 (5,1)-(6,1)
    bool is_blocked_1 = ((r1 == 5 && c1 == 1 && r2 == 6 && c2 == 1) ||
                         (r1 == 6 && c1 == 1 && r2 == 5 && c2 == 1));
    
    // 检查 (5,3)-(6,3)
    bool is_blocked_3 = ((r1 == 5 && c1 == 3 && r2 == 6 && c2 == 3) ||
                         (r1 == 6 && c1 == 3 && r2 == 5 && c2 == 3));
    
    return is_blocked_1 || is_blocked_3;
}

} // namespace junqi