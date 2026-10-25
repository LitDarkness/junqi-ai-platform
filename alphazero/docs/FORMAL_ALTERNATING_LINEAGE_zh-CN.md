# Search v3 正式 alternating lineage

## 1. 目的与不可变边界

本 lineage 从已通过 A/B/C/D 因果实验的
`runs/open_junqi_search_v3_iteration3_recovery/checkpoints/iteration_000003_alternating.pt`
开始，不覆盖原 iteration 3 joint failure，也不回写原 `open_junqi_search_v3` lineage。

原失败证据固定在：

```text
runs/open_junqi_search_v3/diagnostics/iteration_000003_joint_failure_original
```

正式 lineage 为：

```text
runs/open_junqi_search_v3_alternating
```

`manifest.json` 保存恢复 checkpoint、selection report、原失败 manifest 的路径和
SHA-256。初始化时会逐项校验复制的 Oracle 数据、probe、iteration 2/3 replay。

固定不变的实验边界：

- 单一固定布局 `public_conservative_advance`；
- 不改变 PUCT、Dirichlet、温度、规则或布局；
- 3600 动作空间与 6.9M Policy/WDL 网络不变；
- iteration 4、5 每轮各生成 1024 局；
- 已完整提交的数据可恢复复用，gate 失败时不重新生成。

## 2. 启动与恢复

从工程目录运行：

```powershell
cd .
.\run_search_v3_formal_alternating.ps1 -ThroughIteration 5
```

脚本是幂等的：

- 已完成的 iteration 直接读取报告和 checkpoint；
- self-play 中断后复用已事务提交的游戏；
- gate 失败后保留 candidate、report 和本轮数据并暂停；
- 不会因再次运行而覆盖原 iteration 3 失败证据。

配置文件：

```text
config/open_junqi_search_v3_alternating.json
```

正式 checkpoint：

```text
runs/open_junqi_search_v3_alternating/checkpoints/loop_iteration_XXXXXX.pt
```

未通过 gate 的完整候选仍保存到：

```text
runs/open_junqi_search_v3_alternating/candidates/iteration_XXXXXX.pt
```

## 3. 训练调度

### Policy phase

实际 optimizer step 比例严格为：

```text
Policy : Value = 2 : 1
```

每个 Policy batch 的监督来源为：

| 来源 | 比例 |
|---|---:|
| 当前轮 fresh Search visits | 70% |
| 紧邻的最近 1–2 轮 Search visits | 15% |
| Oracle Policy anchor | 10% |
| 循环入口 anchor | 5% |

不会把最近五轮旧 Policy replay 均匀混入。Policy step 更新 Policy head 和
shared trunk，冻结 WDL head。

### Value phase

| 来源 | 比例 |
|---|---:|
| 当前轮完整终局 | 60% |
| 最近历史终局 replay | 40% |

Value step 更新 WDL head，并以较低学习率更新 shared trunk；Policy head 冻结。
Oracle scalar Value 辅助权重固定为 `0.05`。

Value step 前冻结一份 Policy 分布，更新后计算：

```text
KL(pi_before_value || pi_after_value)
```

目标及硬门槛均为 `1e-3`。报告同时输出本轮最大 KL 和被选候选之前各 Value
step 的平均 KL。

### epoch 与候选

- 最多 2 个 fresh-Policy epoch；
- 每约 0.25 epoch 验证一次；
- 未训练的 epoch 0 也是合法候选；
- epoch 0 被选中表示“本轮数据有效但所有参数更新均不值得接受”，不是训练失败；
- scheduler 分开记录 `completed_iterations` 与
  `accepted_update_iterations`。

## 4. Gate 与统计单位

held-out split 按完整 trajectory family 划分。Policy 和 WDL 都先在每个
family 内计算 post-pre 差，再对 family 做 paired bootstrap。

Policy/WDL 分类：

- `improved`：95% CI 上界低于 0；
- `neutral`：均值为 0 或满足严格非退化条件；
- `inconclusive`：证据不足但未达到明确退化；
- `clearly_degraded`：置信区间或相对恶化达到明确退化条件。

一轮正式通过要求：

1. held-out Search excess 改善或统计非退化；
2. WDL CE 改善或统计非退化；
3. Search64 Oracle regret 低于 bare；
4. Value-phase 最大 Policy KL 不超过 `1e-3`；
5. unique-state、轨迹与前缀多样性不坍缩；
6. avoidable repetition 的 Search64 退出率不系统恶化。

固定布局外部 Arena 只有当每个 matchup 至少有 32 条不同 deterministic
trajectory 时才参与 gate。重复 128 次同一确定性轨迹仍只算一条独立证据。

## 5. 每轮报告字段

报告路径：

```text
runs/open_junqi_search_v3_alternating/reports/iteration_XXXXXX.json
```

主要字段：

- `fresh_train.policy_excess`：被选模型在本轮 fresh train 上的 Search excess；
- `held_out.policy_excess`：按 family 隔离的 held-out Search excess；
- `paired_policy`：family 级 Policy 差、95% bootstrap CI 与分类；
- `held_out.wdl_ce`、`paired_wdl`：WDL CE 与 family 级变化；
- `value_phase_policy_kl`：目标、最大值及被选候选前的均值；
- `pre_probe`、`post_probe`：bare/Search64 Oracle regret；
- `avoidable_repetition`：Search64 对可避免重复的退出率；
- `data`：unique state、trajectory、8/16/24-ply 前缀、根 visits entropy、
  repetition、cycle 与吞吐；
- `external_deterministic_divergence`：独立确定性轨迹的首次分叉 ply，
  分叉前后动作及 32k Oracle Q 差；
- `external_arena.used_for_gate`：外部 Arena 是否达到 32 条独立轨迹门槛；
- `gradient_cosine`：共享 trunk 上 Search Policy 与 Value 梯度余弦。

## 6. iteration 4、5 实际结果

### iteration 4

iteration 4 选择 `epoch=1.2453`，即 66 个 Policy step 和 33 个 Value
step，实际更新比例为 2:1。

| 指标 | 结果 |
|---|---:|
| fresh train Search excess | 0.644886 |
| held-out Search excess | 0.687884 |
| held-out Policy family mean delta | -0.041642 |
| Policy family 95% CI | [-0.048093, -0.035259] |
| held-out WDL CE | 0.469805 |
| WDL family mean delta | -0.004751 |
| WDL family 95% CI | [-0.017986, 0.009205] |
| Value-phase 最大 Policy KL | 0.000184 |
| bare Oracle regret | 0.017580 |
| Search64 Oracle regret | 0.012143 |
| Search64 相对 bare regret 改善 | 30.93% |
| avoidable repetition 退出率 | 98.36% → 96.72% |
| unique state ratio | 98.56% |
| unique trajectories | 1024 / 1024 |
| repetition | 4 / 1024 = 0.39% |
| self-play 吞吐 | 4818 simulations/s |

Policy 显著改善；WDL 的 paired CI 跨 0，属于统计非退化。退出率下降
1.64 个百分点，未超过 2 个百分点的系统性恶化门槛。

8 条外部确定性诊断轨迹中 7 条发生分叉，并记录了首次分叉点的 32k Oracle
Q 差。由于每个 matchup 仅有 2 条独立轨迹，外部 Arena 未用于 gate。

### iteration 5

iteration 5 的训练后候选均未在满足前置非退化条件的同时取得更低的
Search64 regret，因此词典序选择回到 epoch 0。

具体地，`epoch=0.261` 的 Search64 regret 降到 `0.011692`，但 WDL 被
paired bootstrap 判为 `clearly_degraded`，因此不合格；其余合格训练候选中
最低的 Search64 regret 为 `0.012221`，仍略高于 epoch 0 的 `0.012143`。
gate 因而没有用单项 Search 改善交换明确的 Value 退化。

| 指标 | 结果 |
|---|---:|
| 被选 epoch | 0 |
| 接受的 Policy / Value step | 0 / 0 |
| fresh train Search excess | 0.555564 |
| held-out Search excess | 0.630647 |
| held-out WDL CE | 0.631410 |
| Policy/WDL paired classification | neutral / neutral |
| Value-phase 最大 Policy KL | 0 |
| bare Oracle regret | 0.017580 |
| Search64 Oracle regret | 0.012143 |
| avoidable repetition 退出率 | 96.72% → 96.72% |
| unique state ratio | 98.31% |
| unique trajectories | 1024 / 1024 |
| repetition | 5 / 1024 = 0.49% |
| self-play 吞吐 | 3892 simulations/s |

iteration 5 是“安全拒绝本轮参数更新”，不是“证明新参数更强”。正式
iteration 5 checkpoint 的模型哈希与 iteration 4 相同；global step 也保持
745。scheduler 记录 3 个 completed iterations（3、4、5），其中只有 2 个
accepted update iterations（3、4）。

## 7. 恢复结论与后续约束

`RECOVERY_STATUS.json` 当前为：

```text
iteration_4_passed = true
iteration_5_passed = true
recovery_confirmed = true
```

这确认的是：交替训练和 gate 已连续两轮避免灾难性退化，且 iteration 4
产生了可接受的实际改进。它不等价于“两轮都提升”；iteration 5 明确没有接受
更新。

若后续再次发生 Policy 蒸馏失败：

1. 保留并复用当轮已提交数据，不重新生成；
2. 先检查 fresh/recent/Oracle/cycle 的实际采样计数；
3. 再检查 Value-phase Policy KL 及 shared-trunk 梯度余弦；
4. 在完成上述因果检查前，不修改 PUCT、Dirichlet、温度或布局。
