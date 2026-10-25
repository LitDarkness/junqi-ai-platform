# 明棋 Policy/Value 与搜索学习实验

## 当前训练入口

2026 年 10 月 3 日的审计修复已加入多布局 v4 训练。新训练使用
[v4 运行说明](docs/OPEN_JUNQI_V4_zh-CN.md)和 `run_open_v4.ps1`。
它分别保存训练模型、出棋模型和冠军，使用独立开发布局评测。
[修复与实测报告](reports/repair_20261003/REPORT_zh-CN.md)记录效率、训练、
1,934 局开发比赛与尚未解决的外部棋力差距。
[长期训练说明](docs/LONG_RUNNING_TRAINING_zh-CN.md)记录后台训练、定期评测、
自动接续、状态查看和停止方式；后台入口为 `start_open_v4_long.ps1`。
原始源码和编译模块保存在 `reports/repair_20261003/baseline_source` 与
`baseline_build`，历史模型不参与覆盖迁移。

v3 的正式延续入口已冻结。以下内容记录历史实验，不能据此继续旧训练。

## 历史实验 open_junqi_search_v3

历史主线是固定布局、严格阶段门控的 Oracle→监督网络→Neural PUCT→搜索蒸馏
闭环。Gate 00–04 已通过，Gate 05 长跑只会在每轮硬门通过后继续：

```powershell
cd .
.\run_open_junqi_search_v3_long.ps1
.\monitor_open_junqi_search_v3.ps1 -Watch -IntervalSeconds 10
```

完整边界、公式、验收条件、当前结果与恢复方式见
[docs/OPEN_JUNQI_SEARCH_V3_zh-CN.md](docs/OPEN_JUNQI_SEARCH_V3_zh-CN.md)。下文保留的是
fixed-layout v1/v2 实验入口和失败基线，不能与 v3 的 Search-PUCT lineage 混用。

## 历史 fixed-layout Policy/Value RL

这是一个独立、可恢复、可扩展的明棋强化学习实验。第一阶段只使用
`optional-local-layout-library.zip` 中的
`public_conservative_advance`（保守突进型），每局红黑双方使用完全相同的
canonical 布局，排除布局差异对胜率的污染。

当前路线不是 AlphaZero/MCTS，也不是 latest-vs-latest PPO：

1. 084、082 和 C++ 静态评价组成宽分布 teacher，先做模仿学习；
2. 网络直接采样动作进行无搜索自博弈；
3. 对手从当前网络、历史网络、当前/历史 Heuristic、随机和激进 exploiter
   组成的人口中抽取；
4. 联合优化 policy、WDL value、熵和相对冻结锚点的 KL。

Teacher 的 084/082 搜索由 16 个持久工作进程并发执行；bootstrap 按 replay
中实际完成的唯一 game key 补足 512 局，不会把 batch size 误当成对局数。

网络输入 107 个平面，动作头为
`logit(i->j)=q_i^T k_j/sqrt(d)+b_from(i)+b_to(j)+b_ij`，再施加合法着
mask；价值头输出 `(P_loss, P_draw, P_win)`。默认模型 6,895,769 参数。

## 直接启动

```powershell
cd .
.\run_fixed_layout_rl.ps1 -Iterations 100
```

脚本自动检测并重建 107 通道 C++ 后端。相同配置和相同
`run_directory` 再次运行会自动从 `latest.pt` 继续。
训练 lineage 永远从 `latest.pt` 连续前进；champion 只用于稳定评测和人口对手，
候选未晋升不会触发训练回滚。按 Ctrl+C 时会丢弃当前未完成 iteration，并恢复
最后一个完整的 `latest.pt`。

监控：

```powershell
.\monitor_fixed_layout_rl.ps1
```

性能测试：

```powershell
.\benchmark_fixed_layout_rl.ps1
```

独立只读评测：

```powershell
# 任意 checkpoint A 对 B；latest/champion/iteration:N 可直接作为别名
.\evaluate_fixed_layout.ps1 `
  -A latest -B champion -Games 512 -Mode sample `
  -Temperature 0.25 -Seed 2026072309

# checkpoint 对规则型或 exploiter 对手
.\evaluate_fixed_layout.ps1 -A iteration:15 -B 084 -Games 512 -Mode greedy
.\evaluate_fixed_layout.ps1 -A latest -B random -Games 512
```

支持的特殊对手名为 `084`、`082`、`random`、`aggressive`。评测强制使用成对
seed 并交换红黑，输出 JSON 与 UTF-8 CSV；它不加载 optimizer、不写 replay、
不推进 global step，也会恢复调用前的 Python/NumPy/Torch/CUDA RNG。

Checkpoint 目录：

```text
runs/fixed_layout_rl_v1/
├─ latest.pt                         # 每轮覆盖，默认恢复点
├─ champion.json                     # 晋升元数据和历史
├─ checkpoints/champion.pt           # 稳定评测/人口基准
├─ checkpoints/archive/iteration_*.pt# 每 5 轮永久保存
└─ checkpoints/recent/iteration_*.pt # 最近 5 个完整 iteration
```

小规模链路验证：

```powershell
.\run_fixed_layout_rl.ps1 `
  -RunDirectory runs\smoke `
  -Iterations 1 `
  -TeacherGames 8 `
  -BootstrapSteps 1 `
  -GamesPerIteration 8 `
  -ParallelGames 8 `
  -TrainSteps 1 `
  -BatchSize 8
```

详细配置、损失、阶段、恢复和扩展方法见
[docs/FIXED_LAYOUT_RL_PIPELINE_zh-CN.md](docs/FIXED_LAYOUT_RL_PIPELINE_zh-CN.md)。
重复和棋、checkpoint 纵向测试和跨布局零训练泛化见
[docs/REPETITION_AND_GENERALIZATION_DIAGNOSTICS_zh-CN.md](docs/REPETITION_AND_GENERALIZATION_DIAGNOSTICS_zh-CN.md)。
完整规则见 [docs/RULES_zh-CN.md](docs/RULES_zh-CN.md)。
