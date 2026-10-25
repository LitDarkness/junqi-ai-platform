# Search v3 第 3 轮失败复现与恢复

## 1. 边界

本恢复工作不生成新对局。唯一输入为：

- `loop_iteration_000002.pt`；
- iteration 3 已完整提交的 1024 局；
- 原 Oracle 训练集和循环入口 anchor。

原 iteration 3 的 checkpoint、报告和四个 replay shard 已复制到
`runs/open_junqi_search_v3/diagnostics/iteration_000003_joint_failure_original`。
`manifest.json` 固定记录每个文件的 SHA-256。任何恢复实验不得覆盖该目录。

## 2. A/B/C/D 因果实验

命令：

```powershell
cd .
.\run_search_v3_iteration3_ablation.ps1
```

四组实验都从 iteration 2 的同一模型和 Adam 状态开始，使用相同
train/validation split、相同 shuffle 和 cursor 顺序、相同尾 batch、相同
训练步数。`contract.json` 固定 checkpoint、replay 和 batch plan 的摘要。
batch plan 从 iteration 2 checkpoint 保存的 RNG 边界按当前 `_fit` 实现重构；
由于旧 iteration 3 没有另外保存“进入 train phase 前”的 batch manifest，且
CUDA kernel 未启用全确定模式，这不是历史运行的位级重放，而是当前 joint
配置可复现的等价重放。

| 实验 | Policy 数据 | Value 数据 | 用途 |
|---|---|---|---|
| A | 仅本轮 fresh Search visits | 无 | 检查 visits 本身能否被拟合 |
| B | 无 | 当前 WDL/Value 路径 | 测量 Value 经共享 trunk 对 Policy 的漂移 |
| C | 旧 replay Policy + 当前 joint Value/anchor | 当前配置 | 重放原失败的前两个被选 epoch |
| D | 仅 fresh Search Policy，删除旧 Policy replay | 与 C 相同 | 隔离旧 Policy 目标的稀释作用 |

共享 trunk 的梯度冲突按下式记录：

```text
rho_s,v = <g_search, g_value> / (||g_search|| * ||g_value||)
```

报告同时给出均值、中位数、P10/P90 和小于零的比例。梯度余弦只做诊断；
实现会保存并恢复模型 buffer 与 Torch RNG，不得因诊断 forward 改变 C 的训练轨迹。

## 3. 正式恢复调度

只有 A/B/C/D 全部成功落盘后，才允许运行：

```powershell
.\run_search_v3_iteration3_recovery.ps1
```

该命令建立独立目录
`runs/open_junqi_search_v3_iteration3_recovery`，不会覆盖原 lineage，也不会调用
self-play。若同一 contract 的完整 `selection.json` 和 checkpoint 已存在，命令
会直接返回已有结果，不会重复训练。

### Policy 阶段

每两个 Policy step 后执行一个 Value phase。每个 Policy batch 的监督比例为：

| 来源 | 比例 |
|---|---:|
| iteration 3 fresh Search visits | 70% |
| iteration 1–2 Search visits | 15% |
| 原 Oracle Policy anchor | 10% |
| 循环入口 Oracle Policy anchor | 5% |

Policy 阶段只更新 Policy head 和 shared trunk；WDL head 冻结。Policy head
学习率为当前基准的 `1.0x`，shared trunk 为 `0.25x`。不计算 WDL loss 和
Oracle scalar Value loss。

### Value 阶段

WDL 数据为本轮完整终局 60%、iteration 1–2 历史终局 40%。WDL head
学习率为 `1.0x`，shared trunk 为 `0.10x`，Policy head 冻结。Oracle scalar
辅助权重为 `0.05`。

每个 Value phase 在更新前冻结 Policy 分布，并用

```text
L_value = L_WDL + 0.05 L_q
          + lambda_KL KL(pi_before_value || pi_after_value)
```

限制 Policy 漂移。目标平均 KL 为 `1e-3`；超过目标后，下一 phase 的
`lambda_KL` 加倍，低于目标一半时逐步降低。正式 alternating lineage
严格采用两个 Policy optimizer step 对一个 Value optimizer step。每个
Value step 都会测量更新前后的 Policy KL；超过 `1e-3` 即不能通过本轮 gate。
这里的 KL 同时是诊断和硬门槛，不再通过额外 Value substep 改变 2:1 的实际更新比例。

## 4. 候选选择和 gate

最多训练 2 个 fresh-Policy epoch，每 0.25 epoch 验证一次，并把未训练的
epoch 0 作为合法候选。外部固定对手得分只作诊断，不参与候选的前置选择。

held-out 单位是 `game_id`，即完整 trajectory family，而不是单个 position。
对每个 family 先求平均差：

```text
Delta_i = metric_post,i - metric_pre,i
```

再进行 paired bootstrap：

- `CI95 upper < 0`：改善；
- CI 包含 0 且平均相对变化不超过 0.5%：中性；
- `CI95 lower > 0` 或平均相对恶化超过 1%：明确退化；
- 其余为证据不足。

候选按以下词典序筛选：

1. fresh held-out Search excess 必须改善或中性；
2. held-out WDL CE 不得明确退化；
3. PUCT64 的 Oracle regret 必须低于 bare；
4. 前三项都满足时，选择 PUCT64 Oracle regret 最低者。

如果所有训练后候选都不合格，选择 epoch 0，即“本轮不更新”，而不是强行提交
退化模型。

## 5. 输出

- A/B/C/D 合同：
  `runs/open_junqi_search_v3/diagnostics/iteration_000003_training_ablation/contract.json`
- A/B/C/D 汇总：
  `runs/open_junqi_search_v3/diagnostics/iteration_000003_training_ablation/summary.json`
- 恢复 lineage 合同：
  `runs/open_junqi_search_v3_iteration3_recovery/manifest.json`
- 词典序选择和 bootstrap 报告：
  `runs/open_junqi_search_v3_iteration3_recovery/reports/selection.json`
- 被选 checkpoint：
  `runs/open_junqi_search_v3_iteration3_recovery/checkpoints/iteration_000003_alternating.pt`

恢复 checkpoint 通过 gate 之前，不得替换
`runs/open_junqi_search_v3/checkpoints/loop_iteration_000003.pt`，也不得继续收集
iteration 4。

## 6. 本次实际结果（2026-07-26）

A/B/C/D 的共同基线为 `Policy excess=0.875192`、`WDL CE=0.712963`。

- A 在 step 132 达到 `Policy excess=0.846387`，说明 fresh Search visits
  可以被学会；继续到 step 350 后反弹到 `0.922630`，证明原训练时长过大。
- B 仅更新 Value 时，step 132 的 WDL CE 曾改善到 `0.667911`，但同一点
  Policy excess 已恶化到 `0.951517`；最终 Policy excess 为 `0.987635`。
  这是 Value 经 shared trunk 破坏 Policy 的直接因果证据。
- C 的当前 joint 重放最终为 `Policy excess=0.875842`、`WDL CE=0.734647`：
  Policy 基本中性，WDL 明显变差，复现了原 iteration 3 的失败方向。
- D 删除旧 Policy replay 后，最终 Policy excess 为 `0.867347`；其最佳点
  step 88 为 `0.841182`。因此旧 Policy 目标确实在稀释 fresh visits。
- 四组 shared-trunk 梯度余弦小于零的比例分别为
  `38.6% / 40.9% / 25.0% / 36.4%`。均值并非长期为负，所以结论不是
  “持续完全冲突”，而是间歇梯度冲突叠加显著的 Value-induced Policy drift。

交替恢复在 2 epoch 的九个候选中选择了 `epoch=1.00`：

| 指标 | iteration 2 基线 | 被选恢复模型 |
|---|---:|---:|
| held-out Policy excess | 0.875192 | 0.845043 |
| held-out WDL CE | 0.712963 | 0.691727 |
| bare Oracle regret | 0.016085 | 0.017361 |
| PUCT64 Oracle regret | 0.014335 | 0.012704 |
| PUCT64 相对 bare regret 改善 | 10.88% | 26.82% |
| avoidable-repetition 搜索退出率 | 96.72% | 98.36% |

以 103 个 held-out trajectory families 为单位：

- Policy mean delta 为 `-0.043516`，95% CI
  `[-0.050890, -0.036375]`，相对改善 `5.82%`；
- WDL mean delta 为 `-0.031123`，95% CI
  `[-0.058396, -0.002802]`，相对改善 `4.58%`。

Value phase 引起的 Policy KL 平均为 `2.45e-4`、最大为 `6.36e-4`，均低于
`1e-3` 目标。PUCT64 probe 速度为约 `7265 simulations/s`。

固定布局外部对局每个 matchup 仍只有两条不同的 greedy 轨迹。其 bare 平均分
从 `0.25` 变为 `0.375`，Search64 平均分从 `0.50` 变为 `0.25`；这项结果保留
为警告，但不参与本次候选选择，不能把名义 128 局误当成 128 个独立样本。
因此恢复模型保存在独立 lineage，状态为
`candidate_selected_not_installed_in_source_lineage`，没有自动覆盖原主线。
