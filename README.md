# Junqi AI Platform

双人军棋明棋 AI 的源码工程：C++ 规则核、Python 可视化工作台、启发式对手，以及 Policy/Value 网络与原生 PUCT 搜索学习实验。

本仓库于 **2026-10-25** 从持续重构的本机工程整理为独立源码快照，使用全新 Git 历史。主要研究代码位于 `alphazero/`。它不是包含预训练模型的开箱即用强棋力成品。

## 包含内容

- `cpp/`、`junqi_rl/`、`GUI.py`：原生规则、批环境、基线 AI、PPO 与 GUI。
- `alphazero/cpp/`：107 平面状态编码、原生 PUCT、RootExplore 搜索与规则适配。
- `alphazero/python/`：稀疏策略回放、WDL Value、完整训练状态、独立开发评测、教师隔离与可恢复比赛分片。
- `alphazero/tools/`、`alphazero/tests/`：训练、复核、诊断和测试源码。
- `teachers/082/`、`teachers/084/`：两个独立教师引擎的源码及构建入口，不含编译二进制。
- `docs/RESEARCH_STATUS.md`：已完成实验的口径、实际结果与限制。

**不包含**模型权重、Adam 状态、训练回放、原始棋谱、历史运行目录、机器监督状态、原始 train/dev/final 布局语料及编译产物。旧实验配置保留方法沿革，引用的检查点和原始证据不会随源码自动出现。

## 构建与验证

当前主要支持 **Windows x64、Python 3.10、C++17**。需要 CMake、Ninja，以及 GCC/MinGW 或已激活的 MSVC 编译环境。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python tools/build_native.py --compiler g++
python tools/verify_source.py
```

使用 MSVC 时，在 Developer PowerShell 中运行 `python tools/build_native.py --compiler cl`。省略 `--compiler` 由 CMake 选择环境中的编译器。构建将生成平台、AlphaZero、RootExplore、Oracle 和两个独立教师模块。不会启动训练。

启动工作台：

```powershell
$env:PYTHONPATH="$PWD;$PWD/build"
python GUI.py --red human --black heuristic_v3
```

## 新实验与历史续训的区别

可先读取 [AlphaZero 使用说明](alphazero/README.md)。示例配置 `alphazero/config/examples/fresh_smoke.json` 使用新随机模型和新布局，只用于验证训练链路，不代表历史 738 代模型。

```powershell
cd alphazero
$env:PYTHONPATH="$PWD/build;$PWD/python;$PWD/tools"
python tools/run_open_v4.py --config config/examples/fresh_smoke.json --through-iteration 1
```

这条命令是显式的新实验，会生成本地模型和数据。本次仓库整理没有执行它，也没有恢复原训练。

历史 50/50 主线采用每 256 个自身前段 Learner 新终局后 64 次更新、50% 自博弈／50% 对真实 082、三段回放及完整优化器/RNG 继承。相应阶段入口与契约源码一并保留，但精确续训还需要未发布的完整状态、布局和原生构建身份；不能把新随机初始化冒称为续训。参见 [发布范围](docs/DISTRIBUTION.md)。

## 已有研究结果

2026-10-09 的冻结大横评完成 10,840 场新评测；738 模型在 64 家族、256 模拟的开发集合上：

| 对手 | 家族等权得分 | 95% 家族区间 |
|---|---:|---:|
| Heu082 | 64.58% | 60.74%–68.49% |
| Heu084 | 58.72% | 54.23%–62.96% |
| 第 400 代 | 57.10% | 53.58%–60.74% |
| 第 600 代 | 55.14% | 51.89%–58.53% |

得分按胜 1／和 0.5／负 0 计算。原完整采用门槛仍有一项未通过，因此没有自动晋级。重复开发集合、相关教师和不同搜索预算的比较不能证明独立最终泛化，也不是单一机制的受控因果实验。

## 规则和实验边界

规则源见 [完整规则说明](docs/rules_zh-CN.md)。规则核变更须通过原生一致性测试；回放保存合法动作 ID 与稀疏访问次数；检查点保存模型、优化器、调度、计数和 RNG。最终布局不能用于训练、调参或晋级。

旧文档中引用的 `runs/`、`reports/`、权重和图表属于本机实验档案，未包含在仓库中；历史运行命令不是自动恢复许可。源码归属与迁移见 [来源说明](docs/DISTRIBUTION.md) 和 [逐文件来源摘要](docs/SOURCE_PROVENANCE.json)。本次整理没有替原工程或第三方代码授予新的开源许可。
