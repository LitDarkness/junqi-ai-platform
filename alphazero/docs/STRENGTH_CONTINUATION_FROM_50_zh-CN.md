# 从 frozen champion iteration 50 继续训练

## 正式长跑命令

```powershell
cd .
.\run_search_v3_strength_continuation.ps1 `
  -ThroughIteration 100 `
  -CpuThreads 12 `
  -ArenaWorkers 12 `
  -ArenaBlockStates 128
```

这会运行 iteration 51–100。中断后重复命令即可恢复。`CpuThreads` 是纯运行时
性能参数，不进入训练配置 hash；恢复时可以在 `8`、`12`、`16` 之间调整。

## 固定不变的训练边界

- 单一固定布局、红方先行、无镜像；
- 普通根 PUCT64，重复敏感根 PUCT128；
- 每轮 1024 局；
- Policy:Value step 为 2:1；
- 最大 2 epoch，每 0.25 epoch 验证；
- epoch 0 是正式候选，允许安全 no-op；
- PUCT、Dirichlet、温度、网络结构和损失比例不变。

iteration 50 的 model、三参数组 optimizer、scheduler、global step、全部 RNG、
replay 状态和 Value calibration 被原子恢复。原始 iteration 50 checkpoint
另外按原文件 SHA-256 冻结保存；iteration 10 和所有旧失败证据不修改。

## 每轮 gate 与记录

内部 incumbent 仍由 held-out Search Policy、WDL family bootstrap、Search64
regret、Value-phase Policy KL、数据多样性和 avoidable repetition 联合决定。
若没有更好的合法候选，epoch 0 成为安全 no-op，本轮仍算完成。

报告额外写入 `round_summary`，包括 selected epoch、update/no-op、Policy/Value
steps、Search excess、Policy/WDL family CI、bare/Search64 regret、KL、
多样性、重复率、平均 ply、终局比例、self-play simulations/s、model hash 和
global step。

## 自动冠军赛

自上次冠军赛后，满足任一条件即阻塞训练并启动正式冠军赛：

- 接受 3 次非 epoch-0 更新；
- 完成 10 轮。

每次生成新的 1024-state sealed suite，并排除原 strength suite、独立 sealed
suite 和所有历史冠军赛 suite 的完整状态。Suite 不进入训练和日常选择。

为正确判断 exact082 是否退化，正式计划包含七组：

1. bare challenger vs bare champion；
2. Search64 challenger vs Search64 champion；
3. bare challenger vs exact082；
4. Search64 challenger vs exact082；
5. Search64 challenger vs bare challenger；
6. bare champion vs exact082；
7. Search64 champion vs exact082。

每组 1024 状态、每状态换边，共 2048 局；七组合计 14,336 局。输出
state-weighted score、family-weighted mean、family 95% CI、W/D/L、重复率、
吃旗率、平均续弈 ply 和速度。冠军赛 transaction 可安全恢复。

只有 challenger 的 bare/Search64 至少一项 family CI 完全高于 50%，另一项
不明确退化、Search64 对 exact082 不明确退化、同代 Search64 显著高于 bare，
并且多样性与重复结构正常时，才更新 frozen champion。

## 存储控制

- 正式 checkpoint：最近 5 个、每 5 轮一个、全部 champion、至多约 8 个
  非 champion competitive；
- candidate：最近 5 个以及 champion/competitive；
- 完整 replay：最近 3 轮、每 10 轮一个、champion 对应轮；
- 所有 iteration report、冠军赛 sealed suite 和晋升证据永久保留；
- 控制台 transcript 保留最近 5 个。

CPU 利用率不是主要效率指标。自博弈在一个进程中同时推进 256 个树，并将叶子
合并成 GPU batch；因此 GPU 等待或短尾 batch 时 CPU 不会自然满载。
`CpuThreads` 控制 Torch/OMP CPU 线程和外部 evaluator 并发，建议本机从
`8`、`12`、`16` 中选择；以控制台 `selfplay-sps` 最高者为准。
