#ifndef JUNQI_TYPES_H
#define JUNQI_TYPES_H

#include <vector>

namespace junqi {

// ==================== 棋子类型 (Piece Types) ====================
// 使用整数表示，数值越大等级越高（特殊棋子除外）
enum PieceType {
    EMPTY = 0,      // 空位
    
    // 特殊棋子
    MINE = 1,       // 地雷 (3枚)
    BOMB = 2,       // 炸弹 (2枚)
    FLAG = 3,       // 军旗 (1枚) - 胜负条件
    
    // 战斗单位 (按等级排列)
    SAPPER = 10,    // 工兵 (3枚) - 唯一能拐弯飞棋，唯一能挖雷
    LIEUTENANT = 20,// 排长 (3枚)
    CAPTAIN = 30,   // 连长 (3枚)
    MAJOR = 40,     // 营长 (2枚)
    COLONEL = 50,   // 团长 (2枚)
    BRIGADIER = 60, // 旅长 (2枚)
    GENERAL = 70,   // 师长 (2枚)
    ARMY_CMDR = 80, // 军长 (1枚)
    FIELD_MARSHAL = 90, // (总)司令 (1枚) - 死亡后亮旗
    
    // 掩码/状态
    UNKNOWN = -1    // 对敌方暗棋的观测值
};

// ==================== 阵营定义 (Factions) ====================
enum PlayerId {
    NONE = 0,
    RED = 1,        // 下方 (通常先行)
    BLACK = -1      // 上方
};

// ==================== 地形类型 (Terrain Types) ====================
enum Terrain {
    NORMAL = 0,     // 普通兵站 (Road)
    RAILWAY = 1,    // 铁路线 (Rail)
    CAMP = 2,       // 行营 (Protection) - 免战区
    HQ = 3,         // 大本营 (Base) - 只能进不能出，军旗位
    MOUNTAIN = 4    // 山界 (Obstacle) - 不可通行
};

// ==================== 战斗结果 (Combat Results) ====================
// 暗棋的核心，Agent 只能收到这个反馈，而不能直接读取敌方棋子
enum CombatResult {
    MOVE_ONLY = 0,      // 移动到空地，无战斗
    WIN = 1,            // 己方胜 (敌死，我活)
    LOSS = 2,           // 己方负 (我死，敌活)
    DRAW = 3,           // 同归于尽 (都死)
    
    // 游戏结束信号
    GAME_OVER_WIN = 10, // 扛旗或全歼 -> 胜
    GAME_OVER_LOSS = 11 // 被扛旗或被全歼 -> 负
};

// ==================== 常量定义 ====================
constexpr int BOARD_ROWS = 12;
constexpr int BOARD_COLS = 5;
constexpr int BOARD_SIZE = BOARD_ROWS * BOARD_COLS;  // 60

constexpr int RED_AREA_START = 0;
constexpr int RED_AREA_END = 5;
constexpr int BLACK_AREA_START = 6;
constexpr int BLACK_AREA_END = 11;

// ==================== 辅助函数 ====================
// 坐标转换
inline int pos_to_index(int row, int col) {
    return row * BOARD_COLS + col;
}

inline void index_to_pos(int index, int& row, int& col) {
    row = index / BOARD_COLS;
    col = index % BOARD_COLS;
}

// 获取棋子等级（用于比较）
inline int get_piece_rank(PieceType piece) {
    return static_cast<int>(piece);
}

// 判断是否是特殊棋子
inline bool is_special_piece(PieceType piece) {
    return piece == MINE || piece == BOMB || piece == FLAG;
}

// 判断是否是可移动棋子
inline bool is_movable_piece(PieceType piece) {
    return piece != MINE && piece != FLAG && piece != EMPTY;
}

// 获取阵营ID的相反阵营
inline PlayerId opposite_player(PlayerId player) {
    return (player == RED) ? BLACK : ((player == BLACK) ? RED : NONE);
}

} // namespace junqi

#endif // JUNQI_TYPES_H