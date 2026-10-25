# Iteration 50 sealed continuation 正式评测

本评测不会继续训练，也不会修改 `championship.json`、optimizer、replay 或
训练 RNG。

## 一条正式命令

```powershell
cd .
.\run_search_v3_sealed_continuation.ps1 -Workers 16 -BlockStates 128
```

这不是 smoke test。命令固定生成并使用 1024 个全新状态，每个 matchup
从每个状态换边续弈两局，共 2048 局。七个 matchup 合计 14,336 局。

## Sealed suite

新 suite 使用独立 seed `2026072701`。冻结的 iteration 10 只负责生成受控
多样轨迹；生成时明确排除原 1024-state suite 的全部完整状态 key。正式评测
开始前写入 `SEALED.json`，固定：

- suite SHA-256；
- generation checkpoint SHA-256；
- 固定布局和完整规则上下文；
- trajectory family；
- 原 suite SHA-256 和零状态重合证明。

Suite 不进入训练、replay 或超参数选择。

## 固定比较

命令不可在同一 run 中改变比较计划：

1. `bare_50 vs bare_10`；
2. `search64_50 vs search64_10`；
3. `bare_50 vs exact082`；
4. `search64_50 vs exact082`；
5. `search64_50 vs bare_50`；
6. `bare_50 vs bare_49`；
7. `search64_50 vs search64_49`。

所有对局均使用完全相同的起始完整状态、正式重复历史和规则计数；双方换边，
greedy，无采样，无 Dirichlet。每组输出 W/D/L、score、trajectory-family
paired bootstrap 95% CI、重复率、吃旗率、平均续弈 ply 和 Search64 速度。

## 恢复与输出

每 128 个状态形成一个完整 transaction。中断后重新运行同一命令，会复用
已经完成并且 contract hash 一致的 transaction。

最终自动汇总到：

```text
runs/open_junqi_search_v3_sealed_continuation_v1/
  sealed_continuation_suite/SEALED.json
  formal_continuation_arena/report.json
  formal_continuation_arena/summary.csv
  console/formal_continuation_*.log
```

`report.json` 仅提供是否替换 frozen champion 的证据，不会自动晋升。
