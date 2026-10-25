# 固定布局 RL：重复和棋与零训练泛化诊断

## 1. 诊断字段

每轮训练的 `selfplay_terminal_reasons` 现在包含 `total_games`，并对以下终局同时
给出 `count` 和 `fraction`：

- `capture_flag`
- `no_legal_move`
- `repetition_draw`
- `no_capture_draw`
- `max_ply_cutoff`
- `other`

工程诊断等级为：

| repetition fraction | 等级 | 含义 |
|---:|---|---|
| `< 15%` | `normal` | 通常不紧急 |
| `15%–30%` | `inspect_by_opponent` | 必须按对手拆分 |
| `30%–50%` | `suspicious` | 明显可疑 |
| `>= 50%` | `likely_policy_degradation` | 很可能发生策略退化 |

每个 `selfplay_opponent` 也包含独立的终局分布、重复比例和等级。训练日志中的
`084`、`082` 是宽 teacher ensemble，不是 deterministic exact 084/082；字段
`opponent_implementation` 会明确写出实现。独立评测参数 `-B 084` 使用的才是
exact 084。

## 2. 重复环记录

每个新产生的重复终局都会另写一条 `kind=selfplay_repetition_cycle`，至少包含：

- `first_cycle_ply`
- `cycle_length` 与两次重复间隔
- `side_that_first_repeated`
- 十六进制 `position_hash_sequence`
- `value_at_cycle_start`
- `terminal_value`
- `opponent_type`
- `cycle_actions`
- game key、setup seed、布局 ID、learner 红黑方

`cycle_actions` 还包含 mover、canonical from/to、action ID 和移动棋子类型，
因而可以统计是否集中在同一棋子、同一开局线路和同一 ply。

独立评测 JSON 的每个重复对局在 `games_detail[].repetition_cycle` 保存相同结构；
CSV 把 hash/action 序列编码成紧凑 JSON 字符串。

## 3. Iteration 165 训练终局

总计 1024 局：

| 终局 | 数量 | 比例 |
|---|---:|---:|
| capture flag | 512 | 50.00% |
| no legal move | 4 | 0.39% |
| repetition draw | 508 | 49.61% |
| no-capture draw | 0 | 0% |
| max-ply cutoff | 0 | 0% |
| other | 0 | 0% |

按对手：

| 对手 | Games | W/D/L | Score | Repetition | Avg ply |
|---|---:|---:|---:|---:|---:|
| current/latest | 265 | 0/265/0 | 50.0% | 100% | 94.0 |
| recent checkpoint | 205 | 0/205/0 | 50.0% | 100% | 94.0 |
| champion (iteration 55) | 35 | 0/35/0 | 50.0% | 100% | 94.0 |
| 宽 084 teacher | 228 | 203/2/23 | 89.47% | 0.88% | 44.4 |
| 宽 082 teacher | 98 | 90/1/7 | 92.35% | 1.02% | 40.3 |
| random | 97 | 97/0/0 | 100% | 0% | 32.2 |
| aggressive | 96 | 96/0/0 | 100% | 0% | 26.7 |

旧 iteration 165 的分类行还没有逐对手终局字段，但该轮总 draw 数恰为 508，
aggregate repetition 也为 508，其他 draw reason 全为 0，因此可严格推出上表每个
draw 都是 repetition，而不是估计。

结论：重复集中在网络对网络，并非对强 Heuristic 的合理保和；这是共同策略坍缩。

## 4. Iteration 163 checkpoint 纵向测试

相同固定布局、相同 256 局 paired seed bank、交换红黑、temperature 0.25：

| Baseline | W/D/L | Score | Repetition | Avg ply | Capture flag |
|---|---:|---:|---:|---:|---:|
| iteration 10 | 152/14/90 | 62.11% | 5.47% | 51.2 | 90.63% |
| iteration 50 | 123/129/4 | 73.24% | 50.39% | 91.3 | 48.05% |
| iteration 100 | 0/256/0 | 50.00% | 100% | 94.0 | 0% |
| iteration 150 | 0/256/0 | 50.00% | 100% | 94.0 | 0% |
| exact 084 | 4/1/251 | 1.76% | 0.39% | 47.3 | 98.83% |

对 iteration 50 的高得分并非纯靠和棋：仍有 48.05% 胜、仅 1.56% 负；但从
iteration 100 起，与后期网络的 50% score 已完全由 100% 和棋构成。

环结构高度一致：

- 163 vs 50：129 个重复全部为 4-ply；125 个从 ply 86 开始，4 个从 ply 92；
- 163 vs 100/150：512 个重复全部从 ply 86 开始，全部为 4-ply；
- 所有 cycle 的四步移动棋子均为 `50,50,50,50`，即双方团长各自来回；
- 163 vs 100 的 cycle-start value 平均约 `-9.66e-7`，等价于零价值固定点。

因此退化形成区间可以锁定在 iteration 50–100。

## 5. Iteration 165 零训练泛化

对手均为 exact 084，每组 256 局，不做微调。

同布局双方：

| 布局 | W/D/L | Score | Repetition |
|---|---:|---:|---:|
| 训练原布局 | 26/2/228 | 10.55% | 0.78% |
| 原布局水平镜像 | 0/0/256 | 0% | 0% |
| 稳扎稳打型 | 0/1/255 | 0.20% | 0.39% |
| 猛虎下山型 | 0/0/256 | 0% | 0% |
| 中轴双核型 | 1/0/255 | 0.39% | 0% |
| 双弹中锁型 | 0/1/255 | 0.20% | 0.39% |

红方保持训练布局、黑方使用其他布局；paired game 中再交换 agent 红黑：

| 黑方布局 | W/D/L | Score | Repetition |
|---|---:|---:|---:|
| 稳扎稳打型 | 0/1/255 | 0.20% | 0.39% |
| 猛虎下山型 | 3/1/252 | 1.37% | 0.39% |
| 中轴双核型 | 1/0/255 | 0.39% | 0% |
| 双弹中锁型 | 2/1/253 | 0.98% | 0.39% |

连训练布局的水平镜像都降到 0%，说明当前网络主要记忆单布局线路，没有获得稳健
的一般走子能力。跨布局失败几乎都以被吃旗结束，不是迁移后改成保和。

## 6. 命令与输出

单组评测：

```powershell
.\evaluate_fixed_layout.ps1 `
  -A iteration:163 -B iteration:100 `
  -Games 256 -Mode sample -Temperature 0.25 -Seed 2026072413
```

不同布局：

```powershell
.\evaluate_fixed_layout.ps1 `
  -A iteration:165 -B 084 -Games 256 `
  -LayoutSet config\layouts\fixed_layout_evaluation_v1.json `
  -RedLayoutId public_conservative_advance `
  -BlackLayoutId public_tiger_descent
```

汇总已有 JSON：

```powershell
.\summarize_fixed_layout_diagnostics.ps1 `
  -Pattern "checkpoint_sweep_*.json" `
  -OutputPrefix runs\fixed_layout_rl_v1\diagnostics\checkpoint_sweep_summary
```

结果：

- `runs/fixed_layout_rl_v1/diagnostics/checkpoint_sweep_summary.json/.csv`
- `runs/fixed_layout_rl_v1/diagnostics/generalization_summary.json/.csv`
- `runs/fixed_layout_rl_v1/diagnostics/iteration_165_training_terminal_report.json`

这些评测只加载 model，不加载 optimizer、不写 replay、不推进 global step，并恢复
Python、NumPy、Torch、CUDA RNG。
