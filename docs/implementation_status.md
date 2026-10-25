# SRS v1.1 实现状态

更新时间：2026-07-13。状态含义：完成表示已有实现和自动验证；部分完成表示入口存在但尚未满足全部验收项；待实现表示本轮未声称交付。

| 范围 | 状态 | 证据或说明 |
|---|---|---|
| M0 规则冻结 | 完成 | `standard_open_v1.json`、规则 SHA-256、完整规则说明和冲突消解 |
| M1 核心可信 | 部分完成 | 直接 movegen、C++ 结构化合法性解释、64 位局面键、战斗、终局、批环境和回归测试完成；apply/undo、稳定 PieceId、增量 Zobrist 尚待结构性重构 |
| M2 GUI 验证 | 完成（工程验收） | AI 工作台含后台思考、结构化合法性、搜索统计、热力图、时间线、分支、保存/载入及状态哈希校验；已有 headless 截图证据 |
| M3 AI Host | 部分完成 | Random、Greedy、Heuristic、AB、PPO 可运行；GUI 有过期响应丢弃，外部独立进程协议和 stop token 待实现 |
| M4 性能基线 | 完成（当前机器单核门禁） | C++ 约 160,000 steps/s；Heuristic v1 约 7,680，v2 约 362.3，生产 AB2 约 120.6，均通过自动门禁 |
| M5 棋力框架 | 部分完成 | 换色评测、95% 区间、冠军门禁和历史池存在；固定随机布局题集待 C++ seed API 支持 |
| M6 训练闭环 | 完成 | C++ batch/PPO 实测、长期训练监督器、实时 ETA/吞吐/GPU/损失、停滞告警、独立评测进程与中断保存入口完成 |
| M7 持续优化 | 部分完成 | 限时迭代加深 Alpha-Beta、TT、走法排序、静态搜索已完成；C++ apply/undo、MCTS、批推理仍待实现 |
| M8 自动演化 | 待实现 | Evolution Lab、候选沙箱、谱系和签名晋升报告不在本轮声明范围 |

## 本轮关键修复

- 统一明棋观测，消除旧 README 暗棋语义与 SRS 的冲突。
- 补齐三次重复、无吃子、最大 ply 及稳定终局原因。
- 修复夺旗手未计 ply。
- 将策略头从 6,915,600 参数降至 6,402 参数；默认全模型约 1,025,731 参数，FP32 约 3.91 MiB。
- rollout 动作掩码由 float32 改为 uint8，该缓冲字段减少 75% 内存。
- 修复配置存在但从未执行的随机对手分支。
- 历史模型池增加硬上限和自动裁剪，避免再次产生大量重复检查点。
- `get_legal_actions()` 改为直接消费 movegen，不再分配并扫描 3600 位 mask。
- 每个环境移除一份重复棋盘拓扑，特征平面的行营表改为静态常量。
- 默认生产 `heuristic` 映射到通过单核门禁的 v2；慢速多层前瞻只能显式使用 `lookahead_experimental`。
- 新增 `performance_gate.py`：裸 C++ 后端最低 50,000 steps/s，所有生产 Heuristic 最低 100 steps/s。
- 新增生产 `ab2`（8 ms 限时）及分析 `ab/ab3`（150 ms），并将生产搜索纳入同一 100 steps/s 门禁。
- 新增统一能力评测和训练监督器；结果分别写入 `reports/evaluation_suite.*` 与每次 `runs/` 目录。

## 清理策略

旧 `JunQi2` 与 `pybindT` 的全部源码已核对迁入统一工程；低体积历史文档和最终 stage6 指标保存在 `docs/legacy`。旧工程、重复检查点、TensorBoard 日志、独立工具链及其 ZIP、重复 SRS 和缓存现已删除。系统 `the configured MinGW toolchain` 的 GCC 15.2 已完成替代构建验证，因此工作目录最终只保留 `junqi_platform` 和交付 ZIP。
