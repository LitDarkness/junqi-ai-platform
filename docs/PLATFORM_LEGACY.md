# 中国军棋明棋 AI 平台

这是依据 `docs/SRS_v1.1.md` 整理的统一工程，将原 `pybindT` C++ 规则核与 `JunQi2` 强化学习流水线合并，默认使用 `standard_open_v1` 规则。

## 已实现能力

- C++17 规则核与直接合法着生成。
- 公路、铁路、工兵转弯、行营、大本营、布阵和战斗规则。
- 明棋全信息观测、C++ 批环境和 Python 3.10 pybind11 扩展。
- 夺旗、无合法着、三次重复、200 ply 无吃子和 1000 ply 上限终局。
- 版本化 RuleProfile、规范化 SHA-256、结构化合法性解释和原子对局记录。
- Random、Greedy、Heuristic V1/V2/V3、限时 Alpha-Beta、PPO 模型及对称评测和冠军门禁。
- 可视化 AI 工作台：后台思考、合法着/热力图、搜索统计、时间线、撤销重做和哈希校验记录。
- 有上限的历史对手池、可复现随机种子、uint8 rollout 掩码。
- 低浪费成对策略头：默认模型约 102.6 万参数，策略头从 691.6 万参数降至 6402 参数。

完整规则见 [军棋完整规则说明](docs/rules_zh-CN.md)，需求原文见 [SRS v1.1](docs/SRS_v1.1.md)，实现边界见 [SRS 实现状态](docs/implementation_status.md)。

## 构建

在 Windows PowerShell 中使用 Python 3.10 与系统 MinGW GCC：

```powershell
cmake -G Ninja -S cpp -B build_native `
  -DCMAKE_CXX_COMPILER="g++" `
  -DPython_EXECUTABLE="python" `
  -DPython_ROOT_DIR="<python-prefix>" `
  -DCMAKE_BUILD_TYPE=Release
cmake --build build_native --config Release -j 4
```

扩展输出到 `build/JunqiEnv.cp310-win_amd64.pyd`。构建依赖由 `requirements.txt` 中的 pybind11 提供。

## 验证

```powershell
$env:PYTHONPATH="$PWD;$PWD/build"
python rule_sanity.py
python bot_sanity.py
python -m unittest discover -s tests -v
python performance_gate.py --output reports/performance_gate.json
python benchmark_env.py
```

当前机器最后一次单核门禁结果：C++ 规则核约 182,857 steps/s，Heuristic V1 约 6,400，V2 约 352.3，V3 约 232.7，生产 Alpha-Beta 约 120.7；门槛分别为 50,000 和 100 steps/s。32 环境批量基准约 215,924 transitions/s。

## 可视化 GUI

```powershell
python GUI.py --red human --black heuristic_v3
python GUI.py --red ab --black heuristic_v3
python GUI.py --red runs/ppo_probe/checkpoints/model_40.pth --black heuristic_v3
# PowerShell 5 安全的等价启动器
.\run_gui.ps1 -Red human -Black heuristic_v3
```

空格自动/暂停，右方向键单步，`Ctrl+Z/Ctrl+Y` 撤销重做，`H` 显示策略热力图。自动对局的 AI 在后台线程运行，状态改变后会丢弃过期响应。GUI 验收截图见 `reports/gui_smoke.png`。

## 训练

```powershell
python train.py --backend cpp-batch --num-envs 8 --num-steps 256 `
  --iters 5000 --seed 20260712 --max-history-models 8
```

推荐使用带预检、实时 ETA/GPU/吞吐/损失、停滞告警和独立 CPU 评测进程的启动器：

```powershell
.\run_large_training.ps1 -Profile standard
# 快速验证整套训练监控
.\run_large_training.ps1 -Profile quick
# 从已有权重继续（当前为权重续训，不恢复优化器动量）
.\run_large_training.ps1 -Profile standard -Resume "runs\...\model_250.pth"
# 自动选择当前最新有效 checkpoint（当前默认指向已暂停任务）
.\run_continue_training.ps1 -Profile standard
```

`standard` 为 12 个 rollout 进程、192 个 C++ 环境、5,000 轮；`large` 为 16 个进程、256 环境、20,000 轮。输出在 `runs/<profile_timestamp>`，含原始命令、实时日志、JSONL/CSV、TensorBoard、模型池和周期评测。

训练后端现在使用 `cpp-sharded`：多个轻量 Windows worker 并行执行 C++ batch rollout 和 V3，对模型对手的请求则汇聚到主进程的单一 CUDA 缓存。实测旧单进程约 433–586 rollout steps/s；最终 standard 的 rollout 约 3,268–3,531 steps/s，完整 iteration 约 7.1–7.5 秒但每轮含 12,288 transitions。GPU 在 rollout 阶段约 14–23%、update 阶段 97–99%，综合占空比约 49–52%。延迟导入 Torch/TensorBoard 后，12-shard 工作集约 2.59 GiB、private commit 约 6.55 GiB。完整扩展曲线与内存边界见 `reports/sharded_training_benchmark.md`。可复测：

```powershell
python benchmark_training_throughput.py --pool-dir runs\<run>\pool `
  --num-envs 192 --num-steps 64 --shards 12 --devices cuda
```

## 棋力评测

```powershell
python evaluate_suite.py --bots random greedy heuristic_v1 heuristic_v2 heuristic_v3 ab2 path\to\model.pth `
  --games-per-color 20 --max-steps 200
```

评测总是交换红黑，输出 JSON、Markdown、score 的 95% 区间和端到端吞吐。当前短程 PPO 探针结果见 `reports/evaluation_suite.md`；40 轮只用于验证训练闭环，不能作为成熟模型。

固定布局 Heuristic 基准使用持久化语料、双方换色和配对 bootstrap 区间：

```powershell
python heuristic_benchmark.py --positions 16 --bots heuristic_v3 `
  --opponents heuristic_v1 heuristic_v2 greedy random --latency-repeats 200
```

## 自动进化 Heuristic

独立的 `heuristic_evolver/` 实验室允许 AI/SubAgent 在不修改生产 bot 和 C++ 的情况下提出候选。它强制执行战术检查、单核 100 steps/s 硬门槛、开发集/独立验证集/新随机局面棋力晋升、原子恢复状态和不可变候选/冠军归档。运行前必须阅读 `heuristic_evolver/README.md` 与 `heuristic_evolver/AGENTS.md`。

## 目录

- `cpp/`：唯一规则事实源与 Python 绑定。
- `config/rules/`：不可变规则配置。
- `junqi_rl/`：模型、PPO、环境、基线 AI、记录与规则适配层。
- `docs/`：SRS、完整规则及实现状态。
- `build/`：已验证的 Python 3.10 扩展。
- `GUI.py`：可视化工作台；`train_large.py`：长期训练监督器。
- `train.py`、`evaluate.py`、`evaluate_suite.py`：训练与评测入口。

## 旧模型兼容

旧约 31.8 MB 检查点使用 dense `actor_head`，不能严格加载到新成对策略头。运行 `tools/migrate_checkpoint.py` 可保留 encoder 与 critic 参数，新策略头会重新初始化。迁移结果必须重新通过固定对手的对称评测，不能直接晋升为冠军。

训练池只保留最新 `--max-history-models` 个 `model_*.pth`，同时保留 `latest.pth` 与显式冠军文件，避免模型池无限膨胀。

## 当前边界

本交付完成并验证了 SRS 的规则冻结、核心可信化、GUI、限时 Alpha-Beta 基线、PPO 训练闭环和独立 Heuristic Evolver。Alpha-Beta 目前通过 Python 调度 C++ 合法着/落子内核；尚无 C++ apply/undo，因此深层搜索效率仍有限。独立外部 Engine Host、MCTS Search SDK 及 24 小时稳定性认证仍属于后续里程碑。
