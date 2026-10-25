**中国军棋（暗棋）环境开发规范书**。


# Project: Junqi-RL-Environment Specification
**Target Language:** C++ (Core Logic) with PyBind11 (Python Interface)
**Game Variant:** 2-Player Land Battle Chess (Dark/Blind Mode)

## 1. 核心常量定义 (Constants & Enums)

首先定义数值映射，以便进行矩阵运算和逻辑比较。

### 1.1 棋子定义 (Piece Types)
使用整数表示，数值越大等级越高（特殊棋子除外）。

```cpp
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
```

### 1.2 阵营定义 (Factions)
```cpp
enum PlayerId {
    NONE = 0,
    RED = 1,    // 下方 (通常先行)
    BLACK = -1  // 上方
};
```

### 1.3 战斗结果 (Combat Results)
这是暗棋的核心，Agent 只能收到这个反馈，而不能直接读取敌方棋子。
```cpp
enum CombatResult {
    MOVE_ONLY = 0,      // 移动到空地，无战斗
    WIN = 1,            // 己方胜 (敌死，我活)
    LOSS = 2,           // 己方负 (我死，敌活)
    DRAW = 3,           // 同归于尽 (都死)
    
    // 游戏结束信号
    GAME_OVER_WIN = 10, // 扛旗或全歼 -> 胜
    GAME_OVER_LOSS = 11 // 被扛旗或被全歼 -> 负
};
```

---

## 2. 棋盘与图结构 (Map & Graph)

军棋棋盘本质上是一个混合图（Grid + Graph）。我们需要定义节点的**类型**和**连通性**。

### 2.1 坐标系
*   尺寸：$12 \text{ (行)} \times 5 \text{ (列)}$。
*   索引：$(row, col)$，其中 $(0,0)$ 为红方左下角，$(11,4)$ 为黑方右上角。
*   区域划分：
    *   红方区域：Row $0 \sim 5$
    *   黑方区域：Row $6 \sim 11$ (注意 Row 5 和 6 之间是前线/山界)

### 2.2 节点类型 (Terrain Type)
每个坐标点 $(r, c)$ 都有一个地形属性：
```cpp
enum Terrain {
    NORMAL = 0,     // 普通兵站 (Road)
    RAILWAY = 1,    // 铁路线 (Rail)
    CAMP = 2,       // 行营 (Protection) - 免战区
    HQ = 3,         // 大本营 (Base) - 只能进不能出，军旗位
    MOUNTAIN = 4    // 山界 (Obstacle) - 不可通行
};
```

**具体的地图布局矩阵 (12x5):**
*(注：R=Rail, N=Normal, C=Camp, H=HQ, M=Mountain)*
```text
(Black Side - Top)
Col:
0  1  2  3  4

N  H  N  H  N  (Row 11) - 只有(11,1)和(11,3)是HQ，其余是兵站
R  R  R  R  R  (Row 10)
R  C  N  C  R  (Row 9)
R  N  C  N  R  (Row 8)
R  C  N  C  R  (Row 7)
R  R  R  R  R  (Row 6) - (6,2)是中间铁路线，但在两山之间
---------------- (Frontline)
R  R  R  R  R  (Row 5)
R  C  N  C  R  (Row 4)
R  N  C  N  R  (Row 3)
R  C  N  C  R  (Row 2)
R  R  R  R  R  (Row 1)
N  N  N  N  N  (Row 0) - 只有(0,1)和(0,3)是HQ
```
*说明：行营 (C) 与周围 8 个格均有**公路**连接。*

### 2.3 连通性逻辑 (Connectivity Logic)
必须实现一个 `GetNeighbors(r, c)` 函数，根据地形返回邻居。
1.  **公路连通 (Highway):** 相邻一格，行营包含斜向，其余的仅包括上下左右（如果有）。
2.  **铁路连通 (Railway):** 只有在都在 Railway 节点上，并且中途没有任何友方或者敌方棋子，才算铁路连通。
3.  **山界:** 红黑两方的前线之间, Col: 1, 3 所对应的位置不联通，也即 (5, 1), (6, 1) 不连通; (5, 3) , (6, 3) 不连通，其余的按照上述规则
---

## 3. 游戏逻辑实现规则 (Game Logic)

这是环境最核心的 `Step` 函数需要处理的部分。

### 3.1 布阵校验 (Init Validation)
在游戏开始前 (`reset` 时)，验证 `board` 是否合法：
1.  **数量检查：** 双方各 25 子，兵种数量符合 1.1 定义，并且所有棋子都在己方对应的场地（如前所述，各自对应半个棋盘内布子）。
2.  **位置限制：**
    *   `FLAG` 必须在 `HQ` (Row 0/11 的 col 1/3)。
    *   `BOMB` 不能在第一排 （最前线 Row 5/6）。
    *   `MINE` 只能在最后两排 (Row 0,1 / 10,11)。
    *   `CAMP` 内不能布子。

### 3.2 移动合法性生成 (Move Generation)
输入当前状态，返回所有合法动作列表 `List<(r1, c1) -> (r2, c2)>`。

1.  **基础约束：**
    *   起点 $(r1, c1)$ 必须是己方棋子。
    *   终点 $(r2, c2)$ 不能是己方棋子。
    *   `MINE` 和 `FLAG` 不能出现在 `From` (不可移动)。
    *   如果是 `HQ` 内的子，**不允许**移出。
    *   如果终点是 `CAMP`，若其内有子（无论敌我），不可进入（Camp保护规则）。

2.  **路径类型判定：**
    *   **工兵 (Sapper) 特权：**
        *   若起点终点都在铁路网：使用 BFS/DFS 搜索铁路路径。只要路径上没有障碍点，即可到达（可拐弯）。
    *   **非工兵 (Others)：**
        *   **公路步：** 距离为 1 (行营可走对角线)。
        *   **铁路直冲：** 起点终点在同一直线铁路，且中间无子。

### 3.3 战斗结算 (Combat Referee)
当 Move 是攻击行为时（目标格有敌子）：

| 攻击方 (Attacker) | 防守方 (Defender) | 结果 | 存活者 |
| :--- | :--- | :--- | :--- |
| 任何 | FLAG | **GAME_OVER_WIN** | Attacker |
| BOMB | 任何 | DRAW | None |
| 任何 | BOMB | DRAW | None |
| SAPPER | MINE | WIN | SAPPER (Sapper lives) |
| 非SAPPER | MINE | LOSS | MINE (Mine lives) |
| Rank A | Rank B | A > B ? WIN : (A < B ? LOSS : DRAW) | ... |

（注意这些判断的优先级）

*   **特殊规则：** 司令 (Field Marshal) 阵亡（即结果为 LOSS 或 DRAW），触发事件 `REVEAL_FLAG`，该阵营的军旗位置对所有人变为可见。
*   在当前回合，如果这一方没有可移动的子，那么这一方立刻判负并且游戏结束（注意：在一个特殊情况下这意味着如果双方最后一个子可移动的子同归于尽，那么 Defender 输）

---
## 3.5 布阵
你同时需要实现一个自动布阵器，并且暴露若干接口，支持：
1. 从外部输入或者给定的库中接收布阵
2. 自动生成**合法**的布阵，在这些布阵中，可能会有一些要求：例如，先生成军棋、地雷；然后如果另一个军旗营空，那么放入排长，然后随机生成（保证合法），当然，也可能比这更复杂，你需要向一个聪明的方式实现。

## 4. 接口定义 (API Specification)

为了适配 Python 和 RL 训练，需要暴露不限于以下的若干 C++ 接口。

```cpp
class JunqiEnvironment {
public:
    // --- 初始化 ---
    // 输入双方的初始布阵 (5x5矩阵 或 List格式)
    // 自动校验布阵合法性，非法则抛出异常
    void reset(const std::vector<int>& red_setup, const std::vector<int>& black_setup);

    // --- 核心交互 ---
    // 执行动作
    // 参数: from_pos (0-59), to_pos (0-59)
    // 返回: CombatResult (这是给Agent唯一的反馈!)
    CombatResult step(int from_pos, int to_pos);

    // --- 状态获取 ---
    
    // 1. 获取全知视角 (用于 Teacher Training / Debug)
    // 返回: 12x5 矩阵，包含所有真实棋子ID
    std::vector<int> get_perfect_state();

    // 2. 获取观测视角 (用于 Student Inference)
    // 参数: player_id (RED/BLACK)
    // 返回: 12x5 矩阵
    //   - 己方: 真实ID
    //   - 敌方暗棋: UNKNOWN (-1)
    std::vector<int> get_observation(int player_id);

    // 3. 获取合法动作掩码
    // 返回: 长度 3600 (60x60) 的 Bool 数组
    std::vector<bool> get_legal_action_mask(int player_id);

    // --- 辅助信息 ---
    // 司令是否已死？(用于 Transformer 作为一个全局特征输入)
    bool is_field_marshal_dead(int player_id);
    
    // 当前回合数
    int get_turn_count();
    
    // 当前行动方
    int get_current_player();
};
```

---

## 5. 关键实现细节 (给开发者的备注)

### 5.1 坐标展平 (Flattening)
为了配合 Transformer 的 Token，建议内部维护一套 `0~59` 的索引：
$$ \text{Index} = \text{Row} \times 5 + \text{Col} $$
反解：
$$ \text{Row} = \text{Index} / 5, \quad \text{Col} = \text{Index} \% 5 $$

### 5.2 铁路线图 (Graph Adjacency)
不要在运行时动态算铁路，直接**硬编码**铁路的邻接表。

### 5.3 动作空间压缩
虽然理论动作是 $60 \times 60 = 3600$，但实际上很多点之间永远无法到达（比如自己家第一排飞到对面家最后一排）。
*   **建议：** 保持 3600 的接口不变，用 Mask 屏蔽即可。这对于 GPU 计算最友好。

### 5.4 观察状态的处理 (Observation State)
需要维护一个 `revealed_map`。
*   初始全为 False。
*   当 `step` 返回结果后，若发生战斗，**只有司令死后，军旗的位置直接返回 FLAG ID。**，其余情况不揭露
