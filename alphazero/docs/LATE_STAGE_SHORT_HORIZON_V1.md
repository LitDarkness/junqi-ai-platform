# Late-stage short-horizon v1

## 历史边界

`iteration_000107` 是旧训练逻辑产生的合法 safe no-op，以下文件保持
不变：

- `checkpoints/loop_iteration_000107.pt`
- `reports/iteration_000107.json`
- iteration 107 的 Policy LR、target semantics 和 Value-step 诊断报告

新逻辑使用独立语义版本：

```text
training_logic_schema = late_stage_short_horizon_v1
```

旧 config hash：

```text
d8ed1e1227d732e38ee0e11aee9b71f9bba0913c31186ba8156b5571e56a1074
```

新 config hash：

```text
8c8daa1b5432d0ad37b755ca606c4fd12259047c8b27ea9b1f8a953fbb4ea27d
```

旧 checkpoint 不迁移、不覆写；新 checkpoint 显式记录新 config、
training-logic schema 和本轮实际选择的 Policy/Value step。

## Iteration 107 原子重放

从 `loop_iteration_000107.pt` 恢复：

- model；
- optimizer；
- scheduler；
- global step；
- Python/NumPy/PyTorch/CUDA RNG；
- alternating batch generator；
- replay 元数据；
- Value calibration。

随后使用固定 batch key 重放：

```text
Policy LR multiplier = 0.5
Policy steps = 3
Value steps = 2
```

完整状态重放得到的模型 hash 必须严格等于：

```text
4feaa5f7b6741c032b8f034e9d465b0391be74b011f354ea6c4979e89bf065b9
```

内部 gate 通过后，完整的 model、optimizer、scheduler、RNG、batch
order、replay 和 calibration 保存在：

```text
checkpoints/late_stage_bootstrap_pending.pt
```

这不是 model-only checkpoint，也不是正式 accepted update。

## Bootstrap 正式门控

pending candidate 使用此前已经 sealed 的同一 1024-state suite，与
iteration 107 进行：

```text
bare_candidate     vs bare_107
Search64_candidate vs Search64_107
bare_candidate     vs exact082
Search64_candidate vs exact082
Search64_candidate vs bare_candidate
bare_107           vs exact082
Search64_107       vs exact082
```

每组 1024 states、每状态换边，共 2048 局；greedy、无 Dirichlet，
使用完整重复历史和规则上下文，以 trajectory family bootstrap 判定。
额外两组 iteration 107 对 exact082 用于在相同 suite 上计算真实的
paired regression 和 repetition noninferiority。

只有正式门控晋升时，candidate 才成为 accepted bootstrap update，并
写入 `loop_iteration_000108.pt`。若门控未通过：

- candidate 不接受；
- iteration 107 仍是模型起点；
- 只创建一个新 schema/config hash 的完整 no-op resume seed；
- 从 iteration 108 生成新 self-play，使用自适应短窗口逻辑。

因此 championship 拒绝不会伪造 accepted update，也不会阻止后续从
合法 safe no-op 继续学习。

## 后续每轮自适应候选

### Policy 阶段

使用：

```text
policy_lr_multiplier = 0.5
policy_candidate_steps = [1, 2, 3, 4, 6]
```

每个窗口输出：

- cycle-aware aggregate family excess；
- adjusted-cycle root family excess；
- ordinary non-cycle family excess；
- paired-family CI 和 classification；
- candidate-level Policy KL。

Cycle-aware、adjusted-cycle 或 ordinary 任一 clearly degraded，或者
candidate Policy KL 超过 `0.006`，均不能进入 Policy shortlist。
其余候选按 family 改善排序，仅保留最多两个。

### Value 阶段

从每个入围 Policy candidate 分别派生：

```text
value_candidate_steps = [0, 1, 2]
```

每个联合候选重新测量 Policy 保留、WDL family CE、Value final/peak
KL 和 candidate-level Policy KL。限制为：

```text
Value final KL <= 0.001
Value peak KL  <= 0.0015
candidate Policy KL <= 0.006
```

系统从合格联合候选中自适应选择。`3+2` 只是 iteration 107 的 bootstrap
默认点，不是永久硬编码；以后可以合法选择 `2+1`、`4+2`、`3+0`、
`6+1` 或 epoch 0。

静态 Oracle regret 仅在最终候选上计算并作为安全诊断。真实 continuation
dev gate 仍决定候选是否优于 incumbent；失败时恢复完整 epoch-0
model、optimizer 和 RNG。

## 长跑命令

```powershell
.\run_search_v3_late_stage_training.ps1 `
  -ThroughIteration 150 `
  -CpuThreads 12 `
  -ArenaWorkers 12 `
  -ArenaBlockStates 128
```

第一次运行自动完成：

1. iteration 107 完整重放；
2. 内部 gate；
3. 正式 1024-state championship；
4. 根据 championship 结果从 accepted candidate 或 immutable
   iteration-107 no-op seed 开始长跑。

之后再次执行相同命令会从最新完整 checkpoint 恢复，不重复已完成的
bootstrap championship。

## 清理

清理只删除可由 iteration 107 原子状态、固定 batch key 和代码重新生成
的未选 model-only 消融权重。保留：

- 全部 JSON/CSV/NPZ 诊断；
- 固定 batch keys；
- 两个进入 Value 消融的 Policy branch；
- 最终选中 Value model-only branch；
- iteration 107 原子 checkpoint；
- 完整 bootstrap checkpoint；
- target-semantics、Pareto、championship 和历史失败证据。

清理清单及每个删除/保留文件的 SHA-256 位于：

```text
runs/open_junqi_search_v3_strength_continuation/
cleanup/iteration107_ablation_weights.json
```

本次共删除 23 个可再生 model-only 权重（605.6 MiB）。另外删除了工程
根目录误留的 `iteration_000015.pt`（79.0 MiB）；它与
`runs/open_junqi_search_v3_scaled_training/candidates/iteration_000015.pt`
的 SHA-256 完全相同，规范位置的原件仍保留。合计回收约 684.6 MiB。
