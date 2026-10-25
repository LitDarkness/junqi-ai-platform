# Advantage RL v2：独立 lineage、训练、评测与验收

## 1. 结论与实验边界

`fixed_layout_rl_v1` 已冻结为
`outcome_weighted_single_layout_failure_baseline`。永久基线是 iteration 165；
它的 checkpoint、两类 replay 和既有诊断结果不得被 v2 写入或迁移。

v1 的主要故障不是“和棋本身”，而是 outcome-weighted imitation 将胜局动作权重设为
2、和局设为 1、负局设为 0；在单布局中，这会反复克隆容易保和的轨迹。扫描显示：

| checkpoint | self-play repetition | 最大单一 cycle / 重复局 | exact 084 score | 镜像 score |
|---:|---:|---:|---:|---:|
| 40 | 1.95% | 20.0% | 56.64% | 0% |
| 50 | 6.25% | 50.0% | 50.39% | 0% |
| 60 | 100% | 100% | 47.27% | 0% |
| 70 | 100% | 100% | 45.70% | 0% |
| 80 | 100% | 100% | 37.70% | 0% |

因此 v2 仅从 iteration 40 加载 `encoder + policy head`，WDL head 重新初始化；
optimizer、scheduler、EMA、anchor、replay 和 RNG 全部重置。禁止从
100、150、163、165 初始化。

本实现借鉴 Ataraxos 的 advantage policy update，但保留本地 Heuristic teacher
作为算力受限的 warm start；这不是对原论文 tabula-rasa 设定的复刻。军棋终局规则
未修改，也没有给 repetition draw 添加固定负奖励。

## 2. 两条独立实验

- `v2a`：`runs/fixed_layout_rl_v2a_advantage_single_layout`。布局、人口权重、
  网络和每轮 1024 局与 v1 相同，只替换 policy update；只跑 20–30 轮做因果验证。
- `v2b`：`runs/fixed_layout_rl_v2b_advantage_multilayout`。四个训练布局，每局双方
  使用同一 canonical 布局，整局以 50% 概率统一水平镜像。
- v2b 密封布局是 `original_open_double_bomb_lock`。它不进入 teacher、自博弈、
  Value replay、训练或超参数选择，只在只读审计中出现。

v2a 与 v2b 的目录、配置 hash、optimizer、scheduler、EMA、anchor、RNG、teacher
bootstrap、fresh rollout 和 checkpoint 历史互不相通。

## 3. 一轮训练的事务顺序

1. 从在线模型复制并冻结 `pi_old`；整轮所有 learner 动作只能由该副本采样。
2. 完成整局后才提交。每个 learner action 保存 canonical observation、legal mask
   的稀疏 action id、所选动作、`pi_old` 原始合法着 logits、真实
   `behavior_log_prob`、温度、uniform mix、行动方、对手、game id、ply、终局结果
   和终局原因。
3. Heuristic/random/aggressive 的动作不进入 policy-gradient；其状态仍进入 Value。
   network-vs-network 时双方动作都是 learner policy 样本。
4. 按行动方分别建立状态序列，即同一方的相邻状态通常相隔两个 ply。计算
   scalar 和 vector λ-return。
5. Value 使用本轮所有完整状态；Policy 只使用本轮、通过 advantage filter 的
   learner action，且最多一个 epoch。旧 rollout 永不再次进入 policy-gradient。
6. 更新 EMA，原子保存本轮 rollout，最后保存完整 `latest.pt`，此时 iteration 才
   标记 completed。`.partial` 文件在恢复时删除。

中断后，未结束游戏只存在内存中，不会进入 rollout；未完成 iteration 没有可恢复
checkpoint。恢复点始终是最后一个 `phase=completed` 的 `latest.pt`。

完整 raw rollout 每轮约 1.0–1.2 GiB，因此事务完成后仅保留最新一轮 raw rollout
供行为概率审计；历史各轮的终局、对手、cycle、advantage 分位数和训练指标永久保存为
`rollouts/summaries/iteration_NNNNNN.json` 与 `metrics.jsonl`。这不会改变训练：
Value 本来就只读取当前轮，Policy 明确禁止复用旧 rollout。

## 4. λ-return 与视角

旧 WDL 标量值：

```text
V_old(s) = P_old(W|s) - P_old(L|s)
```

对于某一行动方自己的状态序列，最后状态直接 bootstrap 到该方终局结果
`z ∈ {-1,0,+1}`；向前递推：

```text
G_t = (1-lambda_advantage) V_old(s_next_same_side)
      + lambda_advantage G_next
delta_t = G_t - V_old(s_t)
```

默认 `gamma=1`、`lambda_advantage=0.5`。WDL 向量顺序严格是
`(P_L,P_D,P_W)`，默认 `lambda_outcome=0.8`：

```text
xi_t = (1-lambda_outcome) WDL_old(s_next_same_side)
       + lambda_outcome xi_next
```

终局 `xi` 是真实结果 one-hot。RED 胜时 RED 的 `z=+1`、BLACK 的 `z=-1`；
和棋双方均为 0。确定性单测覆盖同玩家序列、红黑符号、和棋 one-hot 和终局边界。

## 5. Policy loss

真实采样分布包括温度和探索混合：

```text
b(a|s) = (1-u) softmax(raw_logits / T) + u / |A_legal|
behavior_log_prob = log b(selected|s)
r = exp(log pi_new(selected|s) - behavior_log_prob)
L_pg = -mean(min(r*delta, clip(r,0.8,1.2)*delta))
```

reverse KL 使用同一真实行为分布 `b`：

```text
L_old_kl = 0.1 KL(pi_new || b)
```

magnet policy 不是合法着均匀分布。若当前有 `M` 枚可移动棋子，而棋子 `i`
有 `d_i` 个合法目的地：

```text
rho(i -> j) = 1/M * 1/d_i
alpha_t = 0.05 / (iteration + 1)^0.3
L_magnet = alpha_t KL(pi_new || rho)
```

最终：

```text
L_policy = L_pg + L_old_kl + L_magnet
```

Teacher imitation 单独记录，并以独立辅助权重加到优化目标；默认从 0.30 在 30 轮内
退火到 0。它不属于上述 advantage loss。

## 6. Advantage filter 与日志

只在 learner policy 样本中计算本轮 `abs(delta)` 的第 75 百分位 `q75`：

```text
keep = policy_eligible
       and abs(delta) >= q75
       and abs(delta) >= 0.01
```

每轮记录过滤前后样本数、保留率、正负 advantage 比例、绝对 advantage 的
q0/q25/q50/q75/q90/q95/q99/q100、policy entropy、reverse KL、magnet KL、
importance ratio、PPO clip fraction、梯度范数和学习率。

终局与对手统计包含 W/D/L、score、平均 ply、moves/s、repetition rate；
每个重复局另存 first cycle ply、cycle length、position hash sequence、
cycle actions、起点 Value 和对手类型。

## 7. 配置位置

- v2a：`config/advantage_v2a_single_layout.json`
- v2b：`config/advantage_v2b_multilayout.json`

常改参数：

| 含义 | JSON 路径 | 默认 |
|---|---|---:|
| 每轮局数 | `selfplay.games_per_iteration` | 1024 |
| 并行局数 | `selfplay.parallel_games` | 256 |
| 采样温度 | `selfplay.temperature` | 0.9 |
| 探索均匀混合 | `selfplay.uniform_mix` | 0 |
| λ advantage | `advantage.lambda_advantage` | 0.5 |
| λ outcome | `advantage.lambda_outcome` | 0.8 |
| PPO clip | `advantage.clip_epsilon` | 0.2 |
| reverse KL 权重 | `advantage.old_policy_kl_weight` | 0.1 |
| magnet 初值 | `advantage.magnet_alpha_initial` | 0.05 |
| filter 分位数 | `advantage.filter_percentile` | 75 |
| batch size | `train.batch_size` | 1024 |
| learning rate | `train.learning_rate` | 5e-5 |
| 最大梯度范数 | `train.max_gradient_norm` | 0.5 |
| EMA smoothing | `train.ema_smoothing` | 0.999 |
| teacher 退火轮数 | `train.teacher_anneal_iterations` | 30 |

运行中禁止修改配置；checkpoint 保存 config hash，hash 不一致会拒绝恢复。需要改配置
时应复制 JSON、使用新的 `run_directory` 和 seed，形成新 lineage。

## 8. 启动、恢复与监控

v2a 正式短期 smoke（建议 25 轮）：

```powershell
cd .
.\run_advantage_rl.ps1 -Lineage v2a -Iterations 25
```

同一条命令会自动从该 lineage 的 `latest.pt` 恢复；`-Iterations` 表示再运行多少轮。

v2b 的四布局 teacher/Value bootstrap 可单独执行而不运行 advantage：

```powershell
.\run_advantage_rl.ps1 -Lineage v2b -Iterations 0
```

当前正式 v2b 已完成该步骤并停在 iteration 0。由于 v2a 未通过，不要启动下面命令；
只有重新审查 Value calibration、行为概率和低温循环后，才可继续：

```powershell
.\run_advantage_rl.ps1 -Lineage v2b -Iterations 25
```

监控：

```powershell
.\monitor_advantage_rl.ps1 -Lineage v2a -Tail 10
```

任意只读评测：

```powershell
.\evaluate_advantage_rl.ps1 -Lineage v2a -A latest -B 084 `
  -Games 512 -Mode sample -Temperature 0.25 `
  -LayoutId public_conservative_advance
```

`A/B` 支持 `latest`、`champion`、`iteration:N`、checkpoint 路径、`084`、`082`、
`random`、`aggressive`。v2 checkpoint 的评测权重是 `evaluation_model`，即 EMA。

## 9. 每五轮只读评测与验收

每 5 轮自动评测 EMA：self-play、recent、champion、exact 084、exact 082、random、
aggressive、训练布局镜像；v2b 额外评测其余训练布局、密封布局及镜像。每组输出
W/D/L、score、95% 区间、repetition、最大单一 cycle 占比、capture flag rate 和
平均 ply。评测器保存并恢复 Python/NumPy/Torch/CUDA RNG，不写 rollout 或 optimizer。

密封布局结果仅用于盲审趋势，不能用于修改超参数或晋升。Champion 晋升只使用训练
分布上的独立 champion match，要求 score ≥52% 且 95% 区间下界 >50%；无论是否
晋升，下一轮都从 latest 继续，不回滚。

重复坍缩的自动诊断组合是：

```text
network-vs-network repetition >= 80%
and 单一 cycle / 重复局 >= 50%
and exact 084 没有外部得分提升
```

高和棋率本身不会触发失败，也没有直接惩罚和棋。

v2a 通过：最近 5 轮 network-vs-network repetition 平均 <15%，不再由第 86 ply
固定四步循环主导，对 random/aggressive 不异常重复，且无 NaN、KL 爆炸或 entropy
归零。v2a 不通过时暂停 v2b 长跑。

v2b 第一阶段通过：不形成 v1 固定重复均衡；训练布局镜像不再接近 0%；至少三个训练
布局有可测学习；密封布局趋势总体改善；exact 084 不随自博弈 loss 下降而持续恶化。

完整初始化扫描在
`runs/advantage_initialization_scan/scan_summary.json`，v1 对比诊断见
`docs/REPETITION_AND_GENERALIZATION_DIAGNOSTICS_zh-CN.md`。
