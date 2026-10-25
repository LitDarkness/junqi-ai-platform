# 中国军棋（暗棋）环境 - API 使用文档

## 目录
1. [安装与导入](#安装与导入)
2. [快速开始](#快速开始)
3. [核心API](#核心api)
4. [便利API](#便利api)
5. [枚举与常量](#枚举与常量)
6. [布阵系统](#布阵系统)
7. [游戏流程](#游戏流程)
8. [完整示例](#完整示例)

---

## 安装与导入

### 前置条件
- Python 3.10+
- 已编译的 `JunqiEnv.cp310-win_amd64.pyd` 模块

### 导入模块

```python
import sys
sys.path.insert(0, 'build')  # 或模块所在目录
import JunqiEnv

# 访问子模块
from JunqiEnv import junqi
```

---

## 快速开始

### 最简单的使用

```python
import JunqiEnv
from JunqiEnv.junqi import PieceType, PlayerId, CombatResult

# 创建环境
env = JunqiEnv.JunqiEnvironment()

# 使用随机布阵重置游戏
env.reset_random()

# 获取全知视角状态
state = env.get_perfect_state()
print(f"状态大小: {len(state)}")  # 60 (12x5)

# 获取当前玩家
current_player = env.get_current_player()
print(f"当前玩家: {'红方' if current_player == 1 else '黑方'}")

# 获取合法动作掩码
legal_mask = env.get_legal_action_mask(PlayerId.RED)
print(f"合法动作数: {sum(legal_mask)}")

# 执行一个动作 (从位置12移动到位置11)
result = env.step(12, 11)
print(f"战斗结果: {result}")
```

---

## 核心API

### 1. 环境初始化

#### `JunqiEnvironment()`
创建一个新的军棋环境实例。

```python
env = JunqiEnv.JunqiEnvironment()
```

**参数**: 无

**返回**: 环境对象

---

### 2. 游戏重置

#### `reset(red_setup, black_setup)`
使用自定义布阵重置游戏。

```python
# 创建自定义布阵
red_setup = [10, 20, 30, 40, 50,  # 红方第0行
             60, 70, 80, 90, 1,    # 红方第1行
             ...]                   # 共30个位置

black_setup = [90, 80, 70, 60, 50,  # 黑方第6行
              40, 30, 20, 10, 1,    # 黑方第7行
              ...]                    # 共30个位置

env.reset(red_setup, black_setup)
```

**参数**:
- `red_setup` (List[int]): 红方布阵，长度30
- `black_setup` (List[int]): 黑方布阵，长度30

**异常**: 如果布阵非法，抛出 `RuntimeError`

#### `reset_random()`
使用自动生成的随机布阵重置游戏。

```python
env.reset_random()
```

**参数**: 无

**返回**: 无

---

### 3. 执行动作

#### `step(from_pos, to_pos)`
执行一个移动动作。

```python
# 从位置12移动到位置11
result = env.step(12, 11)

if result == CombatResult.WIN:
    print("胜利！敌方棋子被消灭")
elif result == CombatResult.LOSS:
    print("失败！己方棋子被消灭")
elif result == CombatResult.DRAW:
    print("同归于尽")
elif result == CombatResult.MOVE_ONLY:
    print("移动到空地")
```

**参数**:
- `from_pos` (int): 起始位置，范围 0-59
- `to_pos` (int): 目标位置，范围 0-59

**返回**: `CombatResult` 枚举值

**位置索引说明**:
- 棋盘为 12行×5列，共60个位置
- 索引计算: `index = row * 5 + col`
- 红方区域: 行 0-5 (索引 0-29)
- 黑方区域: 行 6-11 (索引 30-59)

---

### 4. 状态获取

#### `get_perfect_state()`
获取全知视角状态（所有棋子可见）。用于调试或教师训练。

```python
state = env.get_perfect_state()
# state[0] 是位置0的棋子
# state[59] 是位置59的棋子

# 打印棋盘
for row in range(12):
    line = f"Row {row:2d} | "
    for col in range(5):
        idx = row * 5 + col
        piece = state[idx]
        if piece == 0:
            line += "  .  "
        elif piece > 0:
            line += f" {piece:3d} "
        else:
            line += f" {piece:4d} "
    print(line)
```

**参数**: 无

**返回**: `List[int]`, 长度60

**值说明**:
- `0`: 空位
- `正数`: 红方棋子
- `负数`: 黑方棋子（取绝对值为棋子类型）

---

#### `get_observation(player_id)`
获取从指定玩家视角的观测。敌方暗棋显示为 UNKNOWN。

```python
from JunqiEnv.junqi import PlayerId

# 获取红方视角
red_obs = env.get_observation(PlayerId.RED)

# 获取黑方视角
black_obs = env.get_observation(PlayerId.BLACK)

# 统计敌方暗棋数量
unknown_count = sum(1 for p in red_obs if p == PieceType.UNKNOWN)
print(f"敌方暗棋数: {unknown_count}")
```

**参数**:
- `player_id` (PlayerId): 玩家ID (RED=1, BLACK=-1)

**返回**: `List[int]`, 长度60

**值说明**:
- `0`: 空位
- `正数`: 己方棋子
- `-1` (UNKNOWN): 敌方暗棋
- `负数` (其他): 敌方已揭示的棋子

---

#### `get_legal_action_mask(player_id)`
获取合法动作掩码，用于强化学习。

```python
from JunqiEnv.junqi import PlayerId

mask = env.get_legal_action_mask(PlayerId.RED)
print(f"动作空间大小: {len(mask)}")  # 3600 (60x60)

# 获取合法动作
legal_actions = []
for from_pos in range(60):
    for to_pos in range(60):
        action_idx = from_pos * 60 + to_pos
        if mask[action_idx]:
            legal_actions.append((from_pos, to_pos))

print(f"合法动作数: {len(legal_actions)}")

# 随机选择一个合法动作
import random
if legal_actions:
    from_pos, to_pos = random.choice(legal_actions)
    result = env.step(from_pos, to_pos)
```

**参数**:
- `player_id` (PlayerId): 玩家ID

**返回**: `List[bool]`, 长度3600

**掩码索引**: `action_index = from_pos * 60 + to_pos`

---

### 5. 辅助信息

#### `get_current_player()`
获取当前行动的玩家。

```python
player = env.get_current_player()
if player == PlayerId.RED:
    print("红方行动")
else:
    print("黑方行动")
```

**参数**: 无

**返回**: `int` (1=RED, -1=BLACK, 0=NONE)

---

#### `get_turn_count()`
获取当前回合数。

```python
turn = env.get_turn_count()
print(f"第 {turn} 回合")
```

**参数**: 无

**返回**: `int`

---

#### `is_game_over()`
检查游戏是否结束。

```python
if env.is_game_over():
    winner = env.get_winner()
    if winner == 1:
        print("红方获胜！")
    elif winner == -1:
        print("黑方获胜！")
```

**参数**: 无

**返回**: `bool`

---

#### `get_winner()`
获取获胜方。

```python
winner = env.get_winner()
# 0 = 无（游戏未结束或平局）
# 1 = 红方
# -1 = 黑方
```

**参数**: 无

**返回**: `int`

---

#### `is_field_marshal_dead(player_id)`
检查指定玩家的司令是否已阵亡。

```python
from JunqiEnv.junqi import PlayerId

red_marshal_dead = env.is_field_marshal_dead(PlayerId.RED)
black_marshal_dead = env.is_field_marshal_dead(PlayerId.BLACK)

if red_marshal_dead:
    print("红方司令阵亡，军旗位置已揭示")
```

**参数**:
- `player_id` (PlayerId): 玩家ID

**返回**: `bool`

---

## 便利API

便利API提供了一系列简化的接口，方便快速查询游戏状态和执行常见操作。

### 位置转换函数

#### `pos_to_index(row, col)`
将行列坐标转换为线性索引。

```python
row, col = 3, 2
index = JunqiEnv.JunqiEnvironment.pos_to_index(row, col)
print(f"index = {index}")  # 输出: 17
```

**参数**:
- `row` (int): 行号，范围 0-11
- `col` (int): 列号，范围 0-4

**返回**: `int` - 线性索引，范围 0-59

---

#### `index_to_pos(index)`
将线性索引转换为行列坐标。

```python
index = 17
row, col = JunqiEnv.JunqiEnvironment.index_to_pos(index)
print(f"row = {row}, col = {col}")  # 输出: row = 3, col = 2
```

**参数**:
- `index` (int): 线性索引，范围 0-59

**返回**: `Tuple[int, int]` - (row, col)

---

#### `get_action_index(from_pos, to_pos)`
将起始和目标位置转换为动作索引。

```python
from_pos, to_pos = 10, 15
action_idx = JunqiEnv.JunqiEnvironment.get_action_index(from_pos, to_pos)
print(f"action_idx = {action_idx}")  # 输出: 615
```

**参数**:
- `from_pos` (int): 起始位置，范围 0-59
- `to_pos` (int): 目标位置，范围 0-59

**返回**: `int` - 动作索引，范围 0-3599

---

#### `decode_action_index(action_index)`
将动作索引解码为起始和目标位置。

```python
action_idx = 615
from_pos, to_pos = JunqiEnv.JunqiEnvironment.decode_action_index(action_idx)
print(f"from_pos = {from_pos}, to_pos = {to_pos}")  # 输出: from_pos = 10, to_pos = 15
```

**参数**:
- `action_index` (int): 动作索引，范围 0-3599

**返回**: `Tuple[int, int]` - (from_pos, to_pos)

---

### 棋子信息查询

#### `get_piece_at(pos)`
获取指定位置的棋子信息。

```python
env.reset_random()

# 查询位置0的棋子
piece_info = env.get_piece_at(0)
print(f"type = {piece_info.type}, owner = {piece_info.owner}")

# 查询多个位置
for pos in range(5):
    piece = env.get_piece_at(pos)
    if piece.owner != 0:
        print(f"位置{pos}: 棋子类型={piece.type}, 所属={piece.owner}")
```

**参数**:
- `pos` (int): 位置索引，范围 0-59

**返回**: `PieceInfo` 结构体
- `type` (int): 棋子类型（0=空位，1=地雷，2=炸弹，3=军旗，10+=战斗单位）
- `owner` (int): 所属方（0=无，1=红方，-1=黑方）

**注意**: 空位返回 type=0, owner=0，不会报错。

---

#### `get_player_pieces(player_id)`
获取指定玩家的所有棋子位置。

```python
# 获取红方所有棋子位置
red_pieces = env.get_player_pieces(1)
print(f"红方棋子数量: {len(red_pieces)}")
print(f"红方棋子位置: {red_pieces[:10]}")  # 打印前10个

# 获取黑方所有棋子位置
black_pieces = env.get_player_pieces(-1)
print(f"黑方棋子数量: {len(black_pieces)}")

# 遍历红方棋子
for pos in red_pieces:
    piece = env.get_piece_at(pos)
    print(f"位置{pos}: {piece.type}")
```

**参数**:
- `player_id` (int): 玩家ID（1=红方，-1=黑方）

**返回**: `List[int]` - 棋子位置列表

---

#### `get_piece_count(player_id, piece_type)`
获取指定玩家某种棋子的数量。

```python
# 统计红方工兵数量
sapper_count = env.get_piece_count(1, 10)  # 10 = 工兵
print(f"红方工兵数量: {sapper_count}")

# 统计黑方地雷数量
mine_count = env.get_piece_count(-1, 1)  # 1 = 地雷
print(f"黑方地雷数量: {mine_count}")

# 统计双方司令数量
red_marshal = env.get_piece_count(1, 90)  # 90 = 司令
black_marshal = env.get_piece_count(-1, 90)
print(f"红方司令: {red_marshal}, 黑方司令: {black_marshal}")
```

**参数**:
- `player_id` (int): 玩家ID（1=红方，-1=黑方）
- `piece_type` (int): 棋子类型（1=地雷，2=炸弹，3=军旗，10+=战斗单位）

**返回**: `int` - 棋子数量

---

### 移动验证

#### `can_move(from_pos, to_pos)`
检查从from_pos移动到to_pos是否合法。

```python
# 检查特定移动是否合法
is_valid = env.can_move(10, 15)
print(f"从10移动到15是否合法: {is_valid}")

# 检查多个移动
moves_to_check = [(10, 15), (10, 11), (5, 10)]
for from_pos, to_pos in moves_to_check:
    if env.can_move(from_pos, to_pos):
        print(f"✓ {from_pos} -> {to_pos} 合法")
    else:
        print(f"✗ {from_pos} -> {to_pos} 非法")
```

**参数**:
- `from_pos` (int): 起始位置，范围 0-59
- `to_pos` (int): 目标位置，范围 0-59

**返回**: `bool` - True表示合法，False表示非法

---

#### `get_possible_moves(from_pos)`
获取某个位置的所有合法移动目标。

```python
# 获取位置10的所有合法移动
possible_moves = env.get_possible_moves(10)
print(f"位置10的合法移动: {possible_moves}")

# 遍历所有移动
for to_pos in possible_moves:
    print(f"  10 -> {to_pos}")

# 找到移动最多的棋子
env.reset_random()
max_moves = 0
best_pos = -1
for pos in range(60):
    moves = env.get_possible_moves(pos)
    if len(moves) > max_moves:
        max_moves = len(moves)
        best_pos = pos

print(f"位置{best_pos}有最多移动: {max_moves}个")
```

**参数**:
- `from_pos` (int): 起始位置，范围 0-59

**返回**: `List[int]` - 合法目标位置列表

---

#### `get_movable_pieces(player_id)`
获取指定玩家所有可移动棋子的位置。

```python
# 获取红方可移动的棋子
movable_red = env.get_movable_pieces(1)
print(f"红方可移动棋子数: {len(movable_red)}")
print(f"红方可移动棋子: {movable_red}")

# 获取黑方可移动的棋子
movable_black = env.get_movable_pieces(-1)
print(f"黑方可移动棋子数: {len(movable_black)}")

# 选择第一个可移动棋子
if movable_red:
    first_pos = movable_red[0]
    possible_moves = env.get_possible_moves(first_pos)
    print(f"位置{first_pos}可以移动到: {possible_moves}")
```

**参数**:
- `player_id` (int): 玩家ID（1=红方，-1=黑方）

**返回**: `List[int]` - 可移动棋子位置列表

---

### 棋盘渲染

#### `render()`
生成棋盘的文本表示。

```python
# 渲染棋盘
board_str = env.render()
print(board_str)

# 保存到文件
with open('board.txt', 'w', encoding='utf-8') as f:
    f.write(env.render())

# 定期保存棋盘状态
turn = env.get_turn_count()
with open(f'board_turn_{turn}.txt', 'w', encoding='utf-8') as f:
    f.write(f"回合 {turn}\n")
    f.write(env.render())
```

**参数**: 无

**返回**: `str` - 棋盘的文本表示

**输出格式**:
```
   0  1  2  3  4
11:  90  .  .  .  . 
10:  .  .  30  .  . 
...
 0:  10  20  30  40  50 

图例: . = 空位, ? = 暗棋, 正数 = 红方, 负数 = 黑方
      1=地雷 2=炸弹 3=军旗 10=工兵 20=排长 30=连长
      40=营长 50=团长 60=旅长 70=师长 80=军长 90=司令
```

---

#### `print_board()`
直接打印棋盘到控制台。

```python
# 打印当前棋盘
env.print_board()

# 在游戏循环中打印
for turn in range(10):
    if env.is_game_over():
        break
    
    # 执行移动
    # ...
    
    # 打印棋盘
    print(f"\n回合 {turn}:")
    env.print_board()
```

**参数**: 无

**返回**: 无（直接打印到控制台）

---

### 游戏历史

#### `get_last_move()`
获取最后一步移动的记录。

```python
# 获取最后一步移动
last_move = env.get_last_move()
print(f"from_pos = {last_move.from_pos}")
print(f"to_pos = {last_move.to_pos}")
print(f"player_id = {last_move.player_id}")
print(f"result = {last_move.result}")

# 检查是否有移动历史
if last_move.from_pos == -1:
    print("还没有移动历史")
else:
    print(f"最后移动: {last_move.from_pos} -> {last_move.to_pos}")
```

**参数**: 无

**返回**: `MoveRecord` 结构体
- `from_pos` (int): 起始位置（-1表示无历史）
- `to_pos` (int): 目标位置（-1表示无历史）
- `player_id` (int): 执行移动的玩家ID
- `result` (int): 移动结果

---

#### `get_game_history()`
获取完整的移动历史。

```python
# 获取完整历史
history = env.get_game_history()
print(f"历史记录数: {len(history)}")

# 打印所有历史
for i, move in enumerate(history):
    print(f"第{i+1}步: {move.from_pos} -> {move.to_pos}")

# 分析历史
from_positions = [move.from_pos for move in history]
most_frequent = max(set(from_positions), key=from_positions.count)
print(f"最常作为起点的位置: {most_frequent}")
```

**参数**: 无

**返回**: `List[MoveRecord]` - 移动记录列表

---

#### `clear_history()`
清空移动历史。

```python
# 清空历史
env.clear_history()

# 验证历史已清空
history = env.get_game_history()
assert len(history) == 0, "历史应该为空"

# 在开始新游戏前清空历史
env.reset_random()
env.clear_history()
```

**参数**: 无

**返回**: 无

---

### 局面保存与加载

#### `load_board_state(board_state, current_player)`
从给定的棋盘状态加载局面。

```python
# 获取当前状态
state = env.get_perfect_state()
player = env.get_current_player()

# 保存到文件
import json
with open('saved_state.json', 'w') as f:
    json.dump({
        'board_state': state,
        'current_player': player
    }, f)

# 从文件加载
with open('saved_state.json', 'r') as f:
    data = json.load(f)
    env.load_board_state(data['board_state'], data['current_player'])

# 验证加载正确
new_state = env.get_perfect_state()
assert state == new_state, "加载的状态应该相同"
```

**参数**:
- `board_state` (List[int]): 棋盘状态，长度60
  - 0 = 空位
  - 正数 = 红方棋子
  - 负数 = 黑方棋子
- `current_player` (int): 当前玩家（1=红方，-1=黑方）

**返回**: 无

**注意**: 此函数会清空历史并重置游戏结束状态。

---

### 完整示例：使用便利API

```python
import JunqiEnv

# 创建环境
env = JunqiEnv.JunqiEnvironment()
env.reset_random()

# 1. 位置转换
row, col = 3, 2
idx = JunqiEnv.JunqiEnvironment.pos_to_index(row, col)
print(f"位置转换: ({row}, {col}) -> {idx}")

# 2. 查询棋子
piece = env.get_piece_at(0)
if piece.owner != 0:
    print(f"位置0有棋子: type={piece.type}, owner={piece.owner}")

# 3. 查询玩家棋子
red_pieces = env.get_player_pieces(1)
print(f"红方棋子数: {len(red_pieces)}")

# 4. 统计棋子
marshal_count = env.get_piece_count(1, 90)
print(f"红方司令数: {marshal_count}")

# 5. 验证移动
can_move = env.can_move(10, 15)
print(f"10->15合法: {can_move}")

# 6. 获取可移动棋子
movable = env.get_movable_pieces(1)
if movable:
    first = movable[0]
    moves = env.get_possible_moves(first)
    print(f"位置{first}可移动到: {moves}")

# 7. 渲染棋盘
print(env.render())

# 8. 保存和加载
state = env.get_perfect_state()
player = env.get_current_player()

env2 = JunqiEnv.JunqiEnvironment()
env2.load_board_state(state, player)
assert env2.get_perfect_state() == state
print("局面保存和加载成功")
```

---

## 枚举与常量

### PieceType（棋子类型）

```python
from JunqiEnv.junqi import PieceType

# 特殊棋子
PieceType.EMPTY = 0        # 空位
PieceType.MINE = 1         # 地雷 (3枚)
PieceType.BOMB = 2         # 炸弹 (2枚)
PieceType.FLAG = 3         # 军旗 (1枚)

# 战斗单位（等级递增）
PieceType.SAPPER = 10       # 工兵 (3枚)
PieceType.LIEUTENANT = 20  # 排长 (3枚)
PieceType.CAPTAIN = 30     # 连长 (3枚)
PieceType.MAJOR = 40       # 营长 (2枚)
PieceType.COLONEL = 50     # 团长 (2枚)
PieceType.BRIGADIER = 60  # 旅长 (2枚)
PieceType.GENERAL = 70     # 师长 (2枚)
PieceType.ARMY_CMDR = 80    # 军长 (1枚)
PieceType.FIELD_MARSHAL = 90  # 司令 (1枚)

# 观测值
PieceType.UNKNOWN = -1      # 敌方暗棋
```

### PlayerId（玩家ID）

```python
from JunqiEnv.junqi import PlayerId

PlayerId.NONE = 0   # 无
PlayerId.RED = 1    # 红方
PlayerId.BLACK = -1  # 黑方
```

### CombatResult（战斗结果）

```python
from JunqiEnv.junqi import CombatResult

# 移动结果
CombatResult.MOVE_ONLY = 0     # 移动到空地

# 战斗结果
CombatResult.WIN = 1           # 己方胜（敌死，我活）
CombatResult.LOSS = 2          # 己方负（我死，敌活）
CombatResult.DRAW = 3          # 同归于尽（都死）

# 游戏结束
CombatResult.GAME_OVER_WIN = 10    # 胜利（扛旗/全歼）
CombatResult.GAME_OVER_LOSS = 11   # 失败（被扛旗/被全歼）
```

### 棋盘常量

```python
from JunqiEnv.junqi import BOARD_ROWS, BOARD_COLS, BOARD_SIZE

print(f"棋盘行数: {BOARD_ROWS}")   # 12
print(f"棋盘列数: {BOARD_COLS}")   # 5
print(f"总格数: {BOARD_SIZE}")     # 60
```

---

## 布阵系统

### 布阵格式

每方布阵为30个位置的列表（6行×5列）：
- 25个棋子 + 5个空位（行营）
- 布阵索引: `setup[row * 5 + col]` 其中 row=0-5, col=0-4

### 随机布阵

```python
# 生成随机布阵
red_setup = env.generate_red_setup()
black_setup = env.generate_black_setup()

print(f"红方布阵: {red_setup}")
print(f"黑方布阵: {black_setup}")

# 使用随机布阵重置
env.reset_random()
```

### 自定义布阵

```python
try:
    # 设置自定义布阵
    my_setup = [
        10, 20, 30, 40, 50,  # 第0行
        60, 70, 80, 90, 3,   # 第1行 (军旗在最后)
        1, 1, 1, 10, 20,    # 第2行 (3个地雷)
        2, 2, 20, 30, 40,    # 第3行 (2个炸弹)
        50, 60, 10, 20, 30,  # 第4行
        70, 80, 10, 20, 30    # 第5行
    ]
    
    env.set_red_setup(my_setup)
    print("布阵设置成功")
except RuntimeError as e:
    print(f"布阵非法: {e}")
```

### 获取当前布阵

```python
red_setup = env.get_red_setup()
black_setup = env.get_black_setup()

print(f"红方布阵大小: {len(red_setup)}")  # 30
print(f"黑方布阵大小: {len(black_setup)}")  # 30
```

### 布阵规则

1. **军旗**: 必须放在大本营（红方位置1或3，黑方位置31或33）
2. **地雷**: 只能在最后两排（红方行0-1，黑方行10-11）
3. **炸弹**: 不能在前线（红方行5，黑方行6）
4. **行营**: 不能放置任何棋子（红方位置11,13,17,21,23，黑方位置36,38,42,46,48）
5. **棋子数量**: 
   - 地雷: 3枚
   - 炸弹: 2枚
   - 军旗: 1枚
   - 工兵、排长、连长: 各3枚
   - 营长、团长、旅长、师长: 各2枚
   - 军长、司令: 各1枚

---

## 游戏流程

### 完整游戏循环

```python
import JunqiEnv
from JunqiEnv.junqi import PlayerId, CombatResult
import random

# 创建环境
env = JunqiEnv.JunqiEnvironment()

# 重置游戏
env.reset_random()

# 游戏循环
while not env.is_game_over():
    # 获取当前玩家
    player = env.get_current_player()
    
    # 获取合法动作
    mask = env.get_legal_action_mask(player)
    
    # 提取合法动作
    legal_actions = []
    for from_pos in range(60):
        for to_pos in range(60):
            action_idx = from_pos * 60 + to_pos
            if mask[action_idx]:
                legal_actions.append((from_pos, to_pos))
    
    # 随机选择动作
    from_pos, to_pos = random.choice(legal_actions)
    
    # 执行动作
    result = env.step(from_pos, to_pos)
    
    # 输出结果
    turn = env.get_turn_count()
    player_name = "红方" if player == PlayerId.RED else "黑方"
    result_name = {
        CombatResult.MOVE_ONLY: "移动",
        CombatResult.WIN: "胜利",
        CombatResult.LOSS: "失败",
        CombatResult.DRAW: "同归于尽"
    }.get(result, f"游戏结束({result})")
    
    print(f"回合 {turn} {player_name}: {result_name}")

# 游戏结束
winner = env.get_winner()
if winner == 1:
    print("\n游戏结束：红方获胜！")
elif winner == -1:
    print("\n游戏结束：黑方获胜！")
else:
    print("\n游戏结束：平局！")
```

### 强化学习训练循环

```python
import JunqiEnv
from JunqiEnv.junqi import PlayerId
import numpy as np

env = JunqiEnv.JunqiEnvironment()
env.reset_random()

for episode in range(1000):
    state = env.reset_random()
    done = False
    
    while not done:
        # 获取当前玩家观测
        player = env.get_current_player()
        obs = np.array(env.get_observation(player))
        
        # 获取动作掩码
        mask = np.array(env.get_legal_action_mask(player))
        
        # 使用模型预测动作
        # action_probs = model.predict(obs, mask)
        # from_pos, to_pos = select_action(action_probs, mask)
        
        # 随机动作（示例）
        legal_indices = np.where(mask)[0]
        action_idx = np.random.choice(legal_indices)
        from_pos = action_idx // 60
        to_pos = action_idx % 60
        
        # 执行动作
        result = env.step(from_pos, to_pos)
        
        # 检查游戏结束
        if result >= 10:  # GAME_OVER_WIN or GAME_OVER_LOSS
            done = True
            winner = env.get_winner()
            reward = 1.0 if winner == PlayerId.RED else -1.0
        else:
            reward = 0.0
        
        # 存储经验
        # replay_buffer.add(obs, action, reward, next_obs, done)
    
    print(f"Episode {episode + 1} completed, winner: {env.get_winner()}")
```

---

## 完整示例

### 示例1: 基本游戏演示

```python
import JunqiEnv
from JunqiEnv.junqi import PieceType, PlayerId, CombatResult

def print_board(state):
    """打印棋盘状态"""
    print("=" * 50)
    for row in range(12):
        line = f"Row {row:2d} | "
        for col in range(5):
            idx = row * 5 + col
            piece = state[idx]
            if piece == 0:
                line += " .  "
            elif piece > 0:
                line += f" {piece:2d} "
            else:
                line += f"{piece:3d} "
        print(line)
    print("=" * 50)

# 创建环境
env = JunqiEnv.JunqiEnvironment()

# 使用随机布阵重置
env.reset_random()

# 打印初始状态
state = env.get_perfect_state()
print("初始棋盘:")
print_board(state)

# 执行几步动作
for i in range(5):
    player = env.get_current_player()
    mask = env.get_legal_action_mask(player)
    
    # 获取第一个合法动作
    legal_actions = [idx for idx, m in enumerate(mask) if m]
    if legal_actions:
        action_idx = legal_actions[0]
        from_pos = action_idx // 60
        to_pos = action_idx % 60
        
        player_name = "红方" if player == PlayerId.RED else "黑方"
        print(f"\n步骤 {i+1}: {player_name} 从 {from_pos} 移动到 {to_pos}")
        
        result = env.step(from_pos, to_pos)
        result_name = {
            CombatResult.MOVE_ONLY: "移动到空地",
            CombatResult.WIN: "消灭敌方",
            CombatResult.LOSS: "被敌方消灭",
            CombatResult.DRAW: "同归于尽"
        }[result]
        
        print(f"结果: {result_name}")
        
        if result >= 10:
            print("游戏结束！")
            break

# 最终状态
state = env.get_perfect_state()
print("\n最终棋盘:")
print_board(state)

print(f"\n总回合数: {env.get_turn_count()}")
print(f"红方司令存活: {not env.is_field_marshal_dead(PlayerId.RED)}")
print(f"黑方司令存活: {not env.is_field_marshal_dead(PlayerId.BLACK)}")
print(f"获胜方: {env.get_winner()}")
```

### 示例2: 自定义布阵对战

```python
import JunqiEnv
from JunqiEnv.junqi import PieceType, PlayerId

# 创建红方攻击性布阵
red_attack_setup = [
    90, 80, 3, 70, 60,  # 第0行（军旗在中间）
    50, 1, 1, 40, 30,   # 第1行（2个地雷）
    2, 2, 10, 20, 30,   # 第2行（2个炸弹）
    40, 50, 10, 20, 30,  # 第3行
    60, 70, 10, 20, 30,  # 第4行
    80, 90, 10, 20, 1    # 第5行（1个地雷）
]

# 创建黑方防御性布阵
black_defense_setup = [
    10, 20, 30, 40, 50,  # 第6行
    60, 70, 80, 90, 3,   # 第7行（军旗在末尾）
    1, 1, 1, 10, 20,    # 第8行（3个地雷）
    2, 2, 20, 30, 40,    # 第9行（2个炸弹）
    50, 60, 10, 20, 30,  # 第10行
    70, 80, 10, 20, 30   # 第11行
]

# 创建环境
env = JunqiEnv.JunqiEnvironment()

# 使用自定义布阵
env.reset(red_attack_setup, black_defense_setup)

print("红方布阵: " + str(red_attack_setup))
print("黑方布阵: " + str(black_defense_setup))
print(f"\n当前玩家: {'红方' if env.get_current_player() == PlayerId.RED else '黑方'}")
```

### 示例3: 双人对战界面

```python
import JunqiEnv
from JunqiEnv.junqi import PieceType, PlayerId, CombatResult

def get_piece_name(piece):
    """获取棋子名称"""
    names = {
        0: "空位",
        1: "地雷", 2: "炸弹", 3: "军旗",
        10: "工兵", 20: "排长", 30: "连长",
        40: "营长", 50: "团长", 60: "旅长",
        70: "师长", 80: "军长", 90: "司令"
    }
    return names.get(abs(piece), "未知")

def get_piece_symbol(piece):
    """获取棋子符号"""
    if piece == 0:
        return "."
    elif piece > 0:
        return f"r{get_piece_name(piece)[0]}"
    else:
        return f"b{get_piece_name(piece)[0]}"

def display_observation(obs, player):
    """显示观测"""
    print(f"\n{'红方' if player == PlayerId.RED else '黑方'}视角:")
    for row in range(12):
        line = f"Row {row:2d} | "
        for col in range(5):
            idx = row * 5 + col
            piece = obs[idx]
            if piece == 0:
                line += "  .  "
            elif piece == -1:
                line += " 未知 "
            else:
                symbol = get_piece_symbol(piece)
                line += f" {symbol:3s} "
        print(line)
    print()

# 双人对战
env = JunqiEnv.JunqiEnvironment()
env.reset_random()

while not env.is_game_over():
    player = env.get_current_player()
    player_name = "红方" if player == PlayerId.RED else "黑方"
    
    # 显示当前玩家视角
    obs = env.get_observation(player)
    display_observation(obs, player)
    
    # 获取合法动作
    mask = env.get_legal_action_mask(player)
    legal_actions = [idx for idx, m in enumerate(mask) if m]
    
    # 玩家输入
    print(f"{player_name}行动")
    print(f"合法动作数: {len(legal_actions)}")
    
    try:
        from_pos = int(input("输入起始位置 (0-59): "))
        to_pos = int(input("输入目标位置 (0-59): "))
        
        # 检查动作是否合法
        action_idx = from_pos * 60 + to_pos
        if action_idx not in legal_actions:
            print("非法动作！请重新选择")
            continue
        
        # 执行动作
        result = env.step(from_pos, to_pos)
        
        # 显示结果
        result_name = {
            CombatResult.MOVE_ONLY: "移动到空地",
            CombatResult.WIN: "消灭敌方棋子！",
            CombatResult.LOSS: "己方棋子被消灭！",
            CombatResult.DRAW: "同归于尽！"
        }[result]
        
        print(f"结果: {result_name}")
        
    except KeyboardInterrupt:
        print("\n游戏中断")
        break
    except Exception as e:
        print(f"错误: {e}")
        continue

# 游戏结束
winner = env.get_winner()
print(f"\n游戏结束！{'红方' if winner == 1 else '黑方'}获胜！")
print(f"总回合数: {env.get_turn_count()}")
```

---

## 常见问题

### Q: 如何获取某个位置的棋子？
```python
state = env.get_perfect_state()
piece = state[position_index]  # position_index 范围 0-59
```

### Q: 如何计算位置索引？
```python
# 从行列计算索引
row = 2
col = 3
index = row * 5 + col  # 13

# 从索引计算行列
index = 13
row = index // 5  # 2
col = index % 5   # 3
```

### Q: 如何判断棋子类型？
```python
from JunqiEnv.junqi import PieceType

piece = state[index]
if piece == PieceType.MINE:
    print("这是地雷")
elif piece == PieceType.SAPPER:
    print("这是工兵")
elif piece >= PieceType.LIEUTENANT and piece <= PieceType.FIELD_MARSHAL:
    print(f"这是等级{piece//10}的军官")
```

### Q: 如何处理游戏结束？
```python
result = env.step(from_pos, to_pos)

if result >= 10:  # GAME_OVER_WIN or GAME_OVER_LOSS
    winner = env.get_winner()
    if winner == PlayerId.RED:
        print("红方胜利")
    else:
        print("黑方胜利")
```

---

## 性能优化建议

1. **避免频繁创建对象**: 重用列表和数组
2. **使用numpy数组**: 将掩码转换为numpy进行矩阵运算
3. **批量处理**: 在训练时使用批量环境

```python
import numpy as np

# 优化后的动作选择
mask = np.array(env.get_legal_action_mask(player))
legal_indices = np.flatnonzero(mask)  # 比列表推导快
action_idx = np.random.choice(legal_indices)
```

---

## 版本信息

- **版本**: 1.0.0
- **最后更新**: 2026-02-12
- **Python版本**: 3.10+
- **编译器**: VSCode (Windows)

---

## 技术支持

如有问题，请查看：
- `test_junqi.py` - 完整测试示例
- 代码注释 - 每个文件都有详细注释
