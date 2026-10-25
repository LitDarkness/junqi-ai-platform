# Iteration 107：Value-step 最小消融

## 结论

在两个已经定位出的 Policy 候选上，唯一满足“相对 iteration 107
incumbent 的 Policy 与 WDL 均不明确退化、且 Value KL 合格”的组合是：

```text
Policy: LR 0.5×，3 steps
Value: 2 steps
```

这并不表示 Value step 越多越好。实验显示了清楚的折衷：

- 第 1 个 Value step 对刚完成的 Policy 候选干扰最小，但尚未完全修复
  Policy 更新引起的 WDL 漂移；
- 第 2 个 Value step 将 WDL 恢复到相对 incumbent 的统计中性，同时
  cycle-aware aggregate Policy 仍为统计中性，ordinary Policy 仍优于
  incumbent；
- 第 3 个 Value step 继续小幅改善 WDL，但已经明确侵蚀 Policy 蒸馏收益。

因此正式训练应采用“最多 2 个 Value steps，并允许在第 0/1/2 step
之间选择”的方式，不能无条件固定执行更多 Value steps。

## 实验隔离

两条分支均从 iteration 107 的完整原子 checkpoint 开始，复现此前
Policy 消融使用的固定 Policy batch key、batch 顺序和 RNG，并验证复现
出的模型 hash 与已保存 Policy candidate 完全一致：

1. `0.5× / 3 Policy steps`
2. `0.25× / 6 Policy steps`

随后仅增加 `Value steps ∈ {0,1,2,3}`。两条分支使用相同的 Value batch
RNG，且每个 Value step 的正式训练逻辑保持不变：

- WDL head：当前 Value LR 的 `1.0×`；
- shared trunk：当前 Value LR 的 `0.10×`；
- Policy head：冻结，并以 hash 验证未改变；
- Oracle scalar Value weight：`0.05`；
- Value-phase Policy KL target：`1e-3`；
- scheduler、global step 不推进。

本实验没有运行 Arena、PUCT 或 probe；没有测试 fresh-only，没有修改
cycle quota、repeat scale、布局、搜索参数或数据。

## 两种“安全”的定义

报告必须区分：

- `value_integration_safe`：相对刚完成的 Policy candidate，Value 是否
  没有明确破坏 Policy，并改善或不破坏 WDL；
- `formal_candidate_eligible`：相对 iteration 107 incumbent，最终
  Policy 与 WDL 是否均不明确退化，且 KL 合格。

前者回答“Value 是否在局部安全”，后者回答“最终候选能否进入正式
候选选择”。只看前者会把 `0.5×/3 + 1 Value step` 误报成可直接使用。

## 结果

所有 delta 均为 `candidate − reference`；CE/excess 越低越好。

### 0.5× / 3 Policy steps

| Value steps | Cycle-aware Policy Δ（相对 Policy 起点） | Adjusted-cycle Δ | Ordinary Δ | WDL CE Δ（相对 Policy 起点） | Value step final/peak KL | 局部安全 | 相对 incumbent 合格 |
|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 0 | 0 | 0 | 0 | 0 | 0 / 0 | 是 | 否 |
| 1 | -0.000225 | +0.001567 | -0.000325 | -0.002592 | 0.000392 / 0.000392 | 是 | 否 |
| 2 | +0.000735 | +0.004167 | +0.000524 | -0.006505 | 0.000859 / 0.000859 | 否 | **是** |
| 3 | +0.001761 | +0.005724 | +0.001500 | -0.007707 | 0.000251 / 0.000859 | 否 | 否 |

`+1` 的 WDL 相对 Policy 起点显著改善，但相对 iteration 107 仍为
`+0.004363`，95% CI `[+0.001172, +0.007537]`，所以仍是明确退化。

`+2` 的 WDL 相对 iteration 107 为 `+0.000450`，95% CI
`[-0.003305, +0.003994]`，恢复为统计中性。其 cycle-aware Policy
相对 iteration 107 为 `-0.001425`，95% CI
`[-0.003086, +0.000024]`，统计中性；ordinary Policy 仍显著更好：
`-0.001988`，95% CI `[-0.004119, -0.000286]`。

### 0.25× / 6 Policy steps

| Value steps | Cycle-aware Policy Δ（相对 Policy 起点） | Adjusted-cycle Δ | Ordinary Δ | WDL CE Δ（相对 Policy 起点） | Value step final/peak KL | 局部安全 | 相对 incumbent 合格 |
|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 0 | 0 | 0 | 0 | 0 | 0 / 0 | 是 | 否 |
| 1 | +0.000756 | +0.002903 | +0.000596 | -0.003059 | 0.000801 / 0.000801 | 否 | 否 |
| 2 | +0.003678 | +0.007897 | +0.003376 | -0.006462 | 0.001140 / 0.001140 | 否 | 否 |
| 3 | +0.006671 | +0.010857 | +0.006339 | -0.007523 | 0.000407 / 0.001140 | 否 | 否 |

该分支的第 1 个 Value step 已使 aggregate 与 ordinary Policy 明确
退化；第 2 个 Value step 的 final KL 又超过 `1e-3`，并进一步损伤
Policy。因此它不适合作为当前正式调度。

## 正式训练建议

下一阶段保持其他参数不变，只修改 Policy/Value 调度：

```text
Policy LR multiplier = 0.5
Policy checkpoints = 1, 2, 3 steps
首选 Policy candidate = step 3（仍由正式 held-out gate 确认）
Value checkpoints = 0, 1, 2 steps
Value hard maximum = 2 steps
```

候选选择顺序：

1. 相对 iteration incumbent，cycle-aware 与 ordinary Policy 不得
   clearly degraded；
2. WDL 不得 clearly degraded；
3. Value step final KL `<=1e-3`，本候选 Value 过程 peak KL
   `<=1.5e-3`；
4. 满足以上条件后，优先选择 WDL 更好且保留更多 Policy 收益的候选；
5. 若 Value 0/1/2 均无合格候选，保留 Policy-only 或 epoch 0，不强行
   接入 Value。

本次 Value 消融并非全部失败：`0.5×/3 + 2 Value steps` 已找到可行
交点。因此现在没有证据要求回到 fresh-only、cycle quota、repeat
scale 或 PUCT 参数。

## 产物

- 机器可读报告：
  `runs/open_junqi_search_v3_strength_continuation/diagnostics/iteration_000107_value_step_ablation/report.json`
- 候选模型：
  `runs/open_junqi_search_v3_strength_continuation/diagnostics/iteration_000107_value_step_ablation/candidates/`

Value 候选为诊断用 model-only checkpoint；正式恢复训练仍应从完整原子
lineage checkpoint 进入，并在训练器内复现选定的 Policy/Value 调度。
