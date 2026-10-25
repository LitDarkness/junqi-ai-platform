# Search-v3 固定布局循环治理

## 生效边界

本次修改从 `open_junqi_search_v3_strength_continuation` 的 iteration 100
重新开始生效。iteration 99 是最后一个旧 replay 语义的完整 checkpoint，也是
正式对照基线。

保持不变：

- 6.9M Policy/WDL 网络及 107 输入通道；
- 固定布局、红方先行和正式终局规则；
- 普通根 Search64；
- WDL 真实终局目标、和棋价值和合法动作集合；
- PUCT、Dirichlet 和温度的普通状态设置。

## Replay 循环裁剪

对 `REPETITION` 终局，保存入口前全部状态及第一个完整循环，删除第二次及后续
循环。之后再在每局内按以下键去重：

```text
(position_key, current_repetition_count, all_legal_action_repetition_contexts)
```

因此相同棋盘但规则历史不同的状态不会错误合并；完全相同的局面与重复上下文
每局最多贡献一次 Policy 样本。报告字段：

```text
data.replay_cycle_cleanup.cycle_suffix_samples_removed
data.replay_cycle_cleanup.same_position_context_samples_removed
```

未完成的 iteration 100 旧 shard 已删除，下一次运行会从 iteration 99 原子
checkpoint 重新生成 iteration 100，不会混合两种 replay 语义。

历史 99 及以前的 shard 作为可追溯证据保留，但没有
`cycle_cleaned=true` 标记，因此不再进入新训练的 recent replay。iteration
100 的 recent 份额临时回填本轮 fresh 数据；从 iteration 101 起只使用新语义
的最近 replay。这样既不篡改历史证据，也不让旧循环后缀继续贡献梯度。

## 循环入口教师

只要当前局面已经出现至少两次，或任一合法着的 `repetition_context > 0`：

- 根预算从 Search64 提升到 Search256；
- 分别取有访问量的最佳重复着 `Q_repeat` 和最佳退出着 `Q_exit`；
- 当 `Q_exit >= Q_repeat - 0.02` 时，重复着的训练 visits 乘以 `0.25`；
- 当重复着明显更好时，不修改 visits，保留合理保和；
- 实际自博弈采样仍使用原始 Search visits，修改的只是教师 Policy target。

报告字段：

```text
data.cycle_teacher.roots
data.cycle_teacher.eligible
data.cycle_teacher.adjusted
data.cycle_teacher.rational_repeat_preserved
data.cycle_teacher.insufficient_root_coverage
```

## Minibatch

fresh Search 与 recent Search 各自以 18% 为目标抽取
cycle-entry/endgame 样本。循环样本先按 `game_id`（trajectory family）
轮转抽取，单 family 的首轮上限为 batch 的 2%；family 数量不足以填满 batch
时才允许确定性补齐。Oracle anchor 比例、Policy:Value=2:1 和各 loss 不变。

每轮控制台输出 `cycle sampling`，完整记录在：

```text
training_log[*].fresh_search_cycle_endgame_fraction
training_log[*].fresh_search_cycle_family_max_share
training_log[*].recent_search_cycle_endgame_fraction
training_log[*].recent_search_cycle_family_max_share
cycle_endgame_sampling
```

## 规则上下文

网络已得到：

- plane 40：`no_capture_ply / 200`；
- plane 41：当前 `repetition_count / 3`；
- plane 42：`turn_count / 1000`；
- 每个合法动作独立的 `repetition_context`（0/1/2），通过 3600 动作张量进入
  Policy 和 WDL 前向。

规则阈值固定为 repetition=3、no-capture=200、max-ply=1000，所以归一化
计数与“距阈值剩余量”一一对应。配置加载现在会拒绝阈值变化或关闭动作重复
上下文，避免同一盘面在不同历史下被当作相同输入。没有增加通道或改变网络。

## Frozen champion 重复非劣门

正式冠军赛继续使用原棋力条件，并在 candidate 与 incumbent 对 exact082 的
同一 sealed family 上计算重复终局指示量的 paired bootstrap CI。

```text
delta = candidate repetition rate - incumbent repetition rate
clearly degraded iff family CI lower bound > +0.02
```

bare 与 Search64 都不得 clearly degraded。这里允许约 2 个百分点噪声，不把
“重复越低”当作优化目标，也不会惩罚合理和棋。

## 继续训练

从完整 iteration 99 恢复并长期运行到 120：

```powershell
.\run_search_v3_strength_continuation.ps1 `
  -ThroughIteration 120 `
  -CpuThreads 12 `
  -ArenaWorkers 12 `
  -ArenaBlockStates 128
```

checkpoint 的 config hash 已验证与当前配置相同；不需要迁移参数。程序会重新
开始 iteration 100。

## 3–5 次 accepted update 后的正式验证

等待至少三次非 epoch-0 accepted update 后运行：

```powershell
.\run_search_v3_cycle_validation.ps1 `
  -AcceptedUpdates 3 `
  -Workers 12 `
  -BlockStates 128
```

命令会自动找到 iteration 99 之后第 3 次 accepted update，不会把普通
iteration 误算为更新。它固定使用同一 1024-state suite，每状态换边，greedy、
无 Dirichlet、正式历史上下文，并运行：

- new/baseline 的 bare-vs-bare；
- new/baseline 的 Search64-vs-bare；
- new/baseline 的 bare/Search64-vs-exact082；
- bare_new-vs-bare_99；
- Search64_new-vs-Search64_99。

自动汇总 state/family score、family CI、重复率和以下目标：

- bare-vs-bare 重复率下降至少 5 个百分点；
- Search64-vs-bare 重复率下降至少 3 个百分点；
- 对 exact082 得分不下降超过 2 个百分点；
- 新旧模型直接得分不低于 50%。

该评测只读，不写 replay、不改变训练 RNG，也不自动晋升 champion。

## 清理

清理证据保存在：

```text
runs/open_junqi_search_v3_strength_continuation/cleanup/
cycle_aware_20260729_085528.json
```

已释放约 1.50 GiB。所有 frozen champion、正式 checkpoint 保留策略
（最近 5、每 5 轮、冠军、有限 competitive）、iteration 报告、失败证据和
championship suite 均未删除。以后完成的 candidate-dev 临时模型和已由
official checkpoint 替代的 candidate 会由 retention 自动清理。
