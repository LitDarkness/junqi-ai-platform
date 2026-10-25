# JunqiEnv - 陆战棋环境

一个用C++实现、使用pybind11绑定的陆战棋游戏环境，专为强化学习和AI研究设计。

## 目录

- [简介](#简介)
- [特性](#特性)
- [环境要求](#环境要求)
- [安装](#安装)
- [构建](#构建)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [API文档](#api文档)
- [测试](#测试)
- [性能优化](#性能优化)
- [常见问题](#常见问题)
- [许可证](#许可证)

## 简介

JunqiEnv是一个完整的陆战棋游戏实现，包括：
- 完整的游戏规则和移动逻辑
- 支持布阵验证和随机布阵生成
- 暗棋规则（军旗隐藏机制）
- 完善的Python API接口
- 高性能C++实现

## 特性

- **完整的游戏规则**: 包括战斗、扛旗、全歼等胜负判断
- **暗棋机制**: 敌方棋子类型默认隐藏，司令死亡后揭示军旗
- **随机布阵**: 自动生成合法的随机布阵
- **布阵验证**: 确保布阵符合规则
- **高效实现**: C++核心 + pybind11绑定，性能优异
- **易于使用**: 简洁的Python API接口

## 环境要求

### 必需
- **Python**: 3.10 或更高版本
- **CMake**: 3.23 或更高版本
- **C++编译器**: GCC (MinGW) 8.1.0 或更高版本
- **pybind11**: 11.0.2 或更高版本

### 可选
- **Git**: 用于版本控制
- **VSCode**: 推荐的代码编辑器

## 安装

### 1. 克隆项目

```bash
git clone <repository-url>
cd pybindT
```

### 2. 安装pybind11

**重要**: pybind11需要单独安装！

#### 方法A: 使用pip安装（推荐）

```bash
pip install pybind11
```

#### 方法B: 使用Git安装

```bash
git clone https://github.com/pybind/pybind11.git
cd pybind11
git checkout v3.0.2
```

#### 方法C: 使用项目内置的pybind11

本项目已经包含pybind11子模块，可以直接使用（见构建步骤）。

### 3. 安装构建工具

#### Windows (MinGW)

```bash
# 使用 Chocolatey
choco install mingw

# 或手动下载安装 MinGW-w64
# https://www.mingw-w64.org/
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install build-essential cmake python3-dev
```

#### macOS

```bash
brew install cmake python3
```

## 构建

### 前提条件

确保已安装：
- CMake >= 3.23
- C++编译器 (GCC/Clang/MSVC)
- pybind11

### 构建步骤

#### 1. 清理旧构建（可选）

```powershell
# Windows PowerShell
Remove-Item -Recurse -Force build

# Linux/macOS
rm -rf build
```

#### 2. 配置CMake

```powershell
# Windows PowerShell
cmake -G "MinGW Makefiles" -B build -DCMAKE_BUILD_TYPE=Release

# Linux/macOS
cmake -B build -DCMAKE_BUILD_TYPE=Release
```

**配置选项**:
- `-DCMAKE_BUILD_TYPE=Release`: Release模式（优化性能）
- `-DCMAKE_BUILD_TYPE=Debug`: Debug模式（调试用）

#### 3. 编译

```bash
# 进入build目录
cd build

# 编译（使用所有CPU核心）
cmake --build . --config Release -j

# 或者单线程编译
cmake --build . --config Release
```

#### 4. 复制模块

```powershell
# Windows
copy JunqiEnv.cp310-win_amd64.pyd ../

# Linux/macOS
cp JunqiEnv*.so ../
```

### 快速构建脚本

#### Windows (PowerShell)

```powershell
# 一键构建Release版本
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
cmake -G "MinGW Makefiles" -B build -DCMAKE_BUILD_TYPE=Release
cd build
cmake --build . --config Release -j
copy JunqiEnv.cp310-win_amd64.pyd ../
cd ..
```

#### Linux/macOS (Bash)

```bash
#!/bin/bash
# 一键构建Release版本
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cd build
make -j$(nproc)
cp JunqiEnv*.so ..
cd ..
```

### 构建输出

成功构建后，项目根目录会生成：
- `JunqiEnv.cp310-win_amd64.pyd` (Windows) 或
- `JunqiEnv.cpython-310-x86_64-linux-gnu.so` (Linux) 或
- `JunqiEnv.cpython-310-darwin.so` (macOS)

## 快速开始

### 基本使用

```python
import sys
sys.path.insert(0, '.')

import JunqiEnv
import junqi

# 创建环境
env = JunqiEnv.JunqiEnvironment()

# 随机布阵并开始游戏
env.reset_random()

# 游戏循环
while not env.is_game_over():
    # 获取当前玩家
    current_player = env.get_current_player()
    
    # 获取观察（暗棋模式）
    observation = env.get_observation(current_player)
    
    # 获取合法动作
    legal_actions = env.get_legal_action_mask(current_player)
    
    # 选择动作（示例：随机选择）
    # 这里应该是你的AI策略
    # from_pos, to_pos = your_ai(observation, legal_actions)
    
    # 执行移动
    # result = env.step(from_pos, to_pos)
    
    break  # 示例，实际需要完整实现

# 获取结果
winner = env.get_winner()
print(f"游戏结束，获胜方: {winner}")
```

### 自定义布阵

```python
import JunqiEnv

env = JunqiEnv.JunqiEnvironment()

# 生成随机布阵
red_setup = env.generate_red_setup()
black_setup = env.generate_black_setup()

# 或设置自定义布阵
red_setup = [10, 20, 30, ...]  # 25个元素
black_setup = [10, 20, 30, ...]  # 25个元素

# 开始游戏
env.reset(red_setup, black_setup)
```

### 加载特定局面

```python
import JunqiEnv
import junqi

env = JunqiEnv.JunqiEnvironment()

# 创建棋盘状态（60个位置）
board_state = [0] * 60

# 设置红方棋子（正数）
board_state[10 * 5 + 3] = junqi.PieceType.FIELD_MARSHAL
board_state[10 * 5 + 4] = junqi.PieceType.SAPPER

# 设置黑方棋子（负数）
board_state[11 * 5 + 3] = -junqi.PieceType.FLAG
board_state[10 * 5 + 2] = -junqi.PieceType.FIELD_MARSHAL

# 加载局面
env.load_board_state(board_state, junqi.PlayerId.RED)
```

## 项目结构

```
pybindT/
├── README.md                 # 本文件
├── CMakeLists.txt           # CMake构建配置
├── junqi_types.h            # 类型定义和常量
├── junqi_board.h/cpp        # 棋盘管理
├── junqi_state.h/cpp        # 游戏状态管理
├── setup_validator.h/cpp     # 布阵验证
├── setup_generator.h/cpp     # 随机布阵生成
├── move_generator.h/cpp      # 移动生成和验证
├── combat_resolver.h/cpp     # 战斗结算
├── junqi_environment.h/cpp   # 主环境类
├── junqi_bindings.cpp       # Python绑定
├── pybind11/               # pybind11库（子模块）
├── test_*.py              # 测试文件
├── build/                 # 构建目录
└── docs/                  # 文档
    ├── API_使用文档.md
    ├── 代码架构文档.md
    └── BUILD_SUMMARY.md
```

## API文档

详细的API文档请参考 [API_使用文档.md](API_使用文档.md)

### 主要类和方法

#### JunqiEnvironment

```python
env = JunqiEnv.JunqiEnvironment()

# 游戏控制
env.reset(red_setup, black_setup)  # 重置游戏
env.reset_random()                # 随机布阵
result = env.step(from, to)      # 执行移动

# 状态查询
observation = env.get_observation(player_id)          # 获取观察
legal_actions = env.get_legal_action_mask(player_id)   # 获取合法动作
is_over = env.is_game_over()                       # 游戏是否结束
winner = env.get_winner()                          # 获取获胜方

# 棋盘查询
perfect_state = env.get_perfect_state()    # 获取完整状态（上帝视角）
board_render = env.render()                # 渲染棋盘

# 布阵相关
red_setup = env.generate_red_setup()       # 生成红方布阵
black_setup = env.generate_black_setup()   # 生成黑方布阵
```

#### 棋子类型 (PieceType)

```python
import junqi

# 特殊棋子
junqi.PieceType.MINE    # 地雷
junqi.PieceType.BOMB    # 炸弹
junqi.PieceType.FLAG    # 军旗

# 战斗单位
junqi.PieceType.SAPPER         # 工兵
junqi.PieceType.LIEUTENANT     # 排长
junqi.PieceType.CAPTAIN       # 连长
junqi.PieceType.MAJOR         # 营长
junqi.PieceType.COLONEL       # 团长
junqi.PieceType.BRIGADIER     # 旅长
junqi.PieceType.GENERAL       # 师长
junqi.PieceType.ARMY_CMDR     # 军长
junqi.PieceType.FIELD_MARSHAL # 司令
```

#### 战斗结果 (CombatResult)

```python
import junqi

junqi.CombatResult.MOVE_ONLY        # 移动到空地
junqi.CombatResult.WIN             # 胜
junqi.CombatResult.LOSS            # 负
junqi.CombatResult.DRAW            # 同归于尽
junqi.CombatResult.GAME_OVER_WIN    # 游戏结束（我胜）
junqi.CombatResult.GAME_OVER_LOSS   # 游戏结束（我负）
```

#### 阵营 (PlayerId)

```python
import junqi

junqi.PlayerId.NONE   # 无
junqi.PlayerId.RED    # 红方
junqi.PlayerId.BLACK  # 黑方
```

## 测试

项目包含多个测试文件：

```bash
# 运行所有测试
python test_dark_chess.py       # 暗棋规则测试
python test_flag_capture.py     # 扛旗测试
python test_combat.py          # 战斗规则测试
```

### 测试结果

所有测试应该通过：

```
✓ 暗棋特殊规则测试完成！
✓ 所有测试通过
```

## 性能优化

### Release vs Debug

- **Release模式**: 
  - O2优化
  - 静态链接
  - 文件大小: ~150KB
  - 运行速度: 快（推荐生产环境）

- **Debug模式**:
  - 包含调试符号
  - 动态链接
  - 文件大小: ~300KB
  - 运行速度: 慢（仅用于调试）

### 编译优化

```bash
# Release模式（推荐）
cmake -DCMAKE_BUILD_TYPE=Release

# Debug模式（调试用）
cmake -DCMAKE_BUILD_TYPE=Debug
```

### 并行编译

使用所有CPU核心加速编译：

```bash
cmake --build . --config Release -j
```

## 常见问题

### 1. 找不到pybind11

**错误**: `Could not find pybind11`

**解决**:
```bash
# 安装pybind11
pip install pybind11

# 或确保pybind11在项目目录
ls pybind11/include/pybind11
```

### 2. CMake找不到Python

**错误**: `Could NOT find Python`

**解决**:
```bash
# 指定Python路径
cmake -DPYTHON_EXECUTABLE=/path/to/python.exe ..
```

### 3. 编译器错误

**错误**: `C compiler not found`

**解决**:
- Windows: 安装MinGW
- Linux: `sudo apt install build-essential`
- macOS: `xcode-select --install`

### 4. 模块导入失败

**错误**: `ImportError: DLL load failed`

**解决**:
- 确保编译器和Python位数匹配（都是64位或都是32位）
- 检查Python版本是否匹配（3.10）
- 重新编译模块

### 5. 游戏逻辑问题

**问题**: 某些规则不符合预期

**解决**:
- 查看测试文件了解正确用法
- 参考 `API_使用文档.md`
- 查看 `代码架构文档.md` 了解实现细节

## 文档

- [API使用文档](API_使用文档.md) - 详细的API参考
- [代码架构文档](代码架构文档.md) - 系统架构说明
- [构建总结](BUILD_SUMMARY.md) - 构建配置和优化

## 贡献

欢迎贡献！请遵循以下步骤：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 许可证

本项目采用MIT许可证 - 详见LICENSE文件

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送Pull Request

## 致谢

- [pybind11](https://github.com/pybind/pybind11) - C++/Python绑定
- [CMake](https://cmake.org/) - 构建系统
- 陆战棋规则参考

---

**版本**: 1.0.0  
**最后更新**: 2026年2月