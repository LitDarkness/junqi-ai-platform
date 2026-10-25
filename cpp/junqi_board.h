#ifndef JUNQI_BOARD_H
#define JUNQI_BOARD_H

#include "junqi_types.h"
#include <vector>
#include <set>

namespace junqi {

// 棋盘结构 - 管理地形、邻接关系等静态信息
class JunqiBoard {
public:
    JunqiBoard();
    
    // 获取指定位置的地形
    Terrain get_terrain(int row, int col) const;
    Terrain get_terrain(int index) const;
    
    // 获取所有合法的邻居位置（根据地形）
    std::vector<int> get_neighbors(int index) const;
    std::vector<int> get_neighbors(int row, int col) const;
    
    // 获取铁路邻接表（预计算的铁路连通关系）
    const std::vector<std::vector<int>>& get_railway_adjacency() const;
    
    // 判断是否是行营
    bool is_camp(int row, int col) const;
    bool is_camp(int index) const;
    
    // 判断是否是大本营
    bool is_hq(int row, int col) const;
    bool is_hq(int index) const;
    
    // 判断是否是铁路
    bool is_railway(int row, int col) const;
    bool is_railway(int index) const;
    
    // 获取行营位置列表
    const std::vector<int>& get_camp_positions() const;
    
    // 获取大本营位置列表
    const std::vector<int>& get_hq_positions() const;
    
    // 判断两个位置是否相邻（公路连接）
    bool is_adjacent(int from, int to) const;
    
    // 判断两个位置是否在同一铁路线上
    bool is_on_same_railway_line(int from, int to) const;
    
    // 检查山界阻断（公开给 move_generator 使用）
    bool is_mountain_blocked(int from, int to) const;
    
private:
    // 初始化地形矩阵
    void init_terrain();
    
    // 初始化铁路邻接表
    void init_railway_adjacency();
    
    // 初始化特殊位置列表
    void init_special_positions();
    
    // 地形矩阵 (12x5)
    std::vector<std::vector<Terrain>> terrain_;
    
    // 铁路邻接表
    std::vector<std::vector<int>> railway_adjacency_;
    
    // 行营位置列表
    std::vector<int> camp_positions_;
    
    // 大本营位置列表
    std::vector<int> hq_positions_;
    
    // 行营位置的集合（快速查找）
    std::set<int> camp_set_;
    
    // 大本营位置的集合（快速查找）
    std::set<int> hq_set_;
};

} // namespace junqi

#endif // JUNQI_BOARD_H