# Iteration 107 平台期因果审计

日期：2026-07-29  
正式 lineage：`runs/open_junqi_search_v3_strength_continuation`  
审计数据：Iteration 107 已提交 replay；初始化为 Iteration 106 原子 checkpoint。

## 结论

这次平台期不是“训练使用 cycle-aware-v1，而 gate 仍使用旧 Search64 target”造成的。

训练和 gate 的实际代码路径分别为：

```text
Policy train:
_policy_step
  -> _policy_ce
  -> search_distill_batch(sample.visits)

Policy gate:
_train_iteration
  -> _candidate
  -> _family_metrics(sample.visits)
```

两者读取同一个已提交的 `sample.visits`。对循环敏感根，该字段已经是
Search256 经 `Q_exit >= Q_repeat - 0.02` 判定、并把重复着 visits 缩放到
25% 后的 cycle-aware-v1 target。

双 target 审计的结果也一致：0.12、0.24、0.48 epoch 在旧 Search64 和
cycle-aware-v1 两套 target 下都退化。因此，gate 语义不一致假设被否定。

当前证据更支持以下原因：

1. Policy 更新本身过大或混合监督方向冲突；
2. cycle-aware target 不是唯一故障点，因为 adjusted cycle roots 自身也没有改善；
3. Value 更新在 0.48 epoch 出现峰值 KL 超限，但不是最早的 Policy 失败原因；
4. 18% cycle/endgame 配额会增加冲突，但数据不支持“cycle 已改善、仅普通策略遗忘”这一较窄解释。

## 可复现性

审计从 Iteration 106 的 model、optimizer、scheduler 和 RNG 恢复，仅重用
Iteration 107 已提交数据，没有生成新对局，没有修改正式 checkpoint/replay。

候选：

| epoch | Policy steps | Value steps | model hash 前缀 |
|---:|---:|---:|---|
| 0.12 | 6 | 3 | `09ccdd2a7e99` |
| 0.24 | 12 | 6 | `26de64b23c15` |
| 0.48 | 24 | 12 | `a7c9218b3610` |

0.24 和 0.48 与正式训练报告的复现误差：

| epoch | Policy excess 绝对误差 | WDL CE 绝对误差 | Top-1 |
|---:|---:|---:|---|
| 0.24 | `2.89e-8` | `7.74e-10` | 完全相同 |
| 0.48 | `3.23e-8` | `3.27e-9` | 完全相同 |

这说明候选权重、batch 顺序和 RNG 重放与正式训练一致。

## 两套 held-out target

`cycle_aware_v1`：

- 普通根：Search64；
- 循环敏感根：Search256；
- 若退出着不比重复着差超过 0.02，则重复着 visits 缩放到 25%。

`legacy_search64`：

- 普通根直接使用已提交的原始 Search64 visits；
- 循环根在完全相同的状态、规则历史、root-noise seed 上补跑 Search64。

补跑 Search64 在普通根上与已提交 visits 的逐元素完全一致率为
`5274/5504 = 95.82%`。普通根报告使用已提交 visits，因此没有这部分数值误差；
补跑只用于历史数据中没有保存旧标签的循环根。

两套 target 在 5970 个 held-out roots 中有 466 个不同。所有根的平均
total variation 为 0.0171；adjusted cycle roots 内为 0.2255，说明循环标签
迁移本身很强，但只影响 held-out 的一部分状态。

## Aggregate Policy 结果

下表的 delta 均为“候选减 incumbent”；Policy excess 越低越好，所以正数表示退化。
CI 按 trajectory family bootstrap。

| epoch | target | state-weighted delta | family mean delta | family 95% CI | 判定 |
|---:|---|---:|---:|---|---|
| 0.12 | cycle-aware-v1 | +0.001705 | +0.002846 | [+0.000482, +0.004969] | 明确退化 |
| 0.12 | legacy Search64 | +0.001047 | +0.002793 | [+0.000299, +0.004991] | 明确退化 |
| 0.24 | cycle-aware-v1 | +0.002683 | +0.002600 | [+0.000338, +0.004790] | 明确退化 |
| 0.24 | legacy Search64 | +0.002617 | +0.002536 | [+0.000152, +0.004716] | 明确退化 |
| 0.48 | cycle-aware-v1 | +0.007262 | +0.006069 | [+0.003344, +0.008839] | 明确退化 |
| 0.48 | legacy Search64 | +0.006810 | +0.005991 | [+0.003413, +0.008692] | 明确退化 |

若 gate 语义错位，应出现“新 target 改善、旧 target 退化”。实际没有出现。

## cycle-aware-v1 分类结果

Policy excess delta 同样以正数表示退化。`Policy KL` 为
`KL(incumbent || candidate)`，它衡量整个候选相对 incumbent 的漂移，不是仅
Value step 前后的 KL。

### 0.12 epoch

| 分类 | 样本 | state delta | family mean / 95% CI | Top-1 | WDL CE delta | Policy KL |
|---|---:|---:|---|---:|---:|---:|
| adjusted cycle | 399 | +0.013273 | -0.000191 / [-0.011282, +0.009406] | 33.83% | +0.047095 | 0.009433 |
| unadjusted cycle | 67 | +0.007648 | +0.008504 / [-0.005882, +0.024039] | 73.13% | +0.035135 | 0.015789 |
| ordinary non-cycle | 5504 | +0.000794 | +0.002329 / [-0.000671, +0.004827] | 75.42% | +0.003988 | 0.014952 |
| endgame | 1079 | +0.004082 | -0.003711 / [-0.014726, +0.005252] | 61.08% | +0.024660 | 0.015357 |
| opening | 2098 | +0.003701 | +0.003843 / [+0.001504, +0.006057] | 86.27% | -0.005319 | 0.014333 |
| midgame | 2793 | -0.000712 | +0.003405 / [-0.001106, +0.007589] | 66.81% | +0.009899 | 0.014492 |
| tactical | 5596 | +0.001074 | +0.002679 / [+0.000201, +0.004719] | 73.57% | +0.003186 | 0.014553 |

### 0.24 epoch

| 分类 | 样本 | state delta | family mean / 95% CI | Top-1 | WDL CE delta | Policy KL |
|---|---:|---:|---|---:|---:|---:|
| adjusted cycle | 399 | +0.009663 | +0.009342 / [-0.002240, +0.024025] | 33.08% | +0.015254 | 0.008450 |
| unadjusted cycle | 67 | +0.006263 | +0.008104 / [-0.006752, +0.022430] | 76.12% | +0.009371 | 0.014769 |
| ordinary non-cycle | 5504 | +0.002133 | +0.002174 / [-0.000455, +0.004621] | 74.02% | +0.002289 | 0.015043 |
| endgame | 1079 | +0.008580 | -0.001138 / [-0.011613, +0.007513] | 59.13% | +0.026279 | 0.014990 |
| opening | 2098 | +0.002489 | +0.002552 / [+0.000061, +0.005103] | 84.37% | -0.007392 | 0.013758 |
| midgame | 2793 | +0.000551 | +0.001870 / [-0.001158, +0.004816] | 66.20% | +0.002316 | 0.015080 |
| tactical | 5596 | +0.002002 | +0.002399 / [+0.000101, +0.004547] | 72.18% | +0.001754 | 0.014646 |

### 0.48 epoch

| 分类 | 样本 | state delta | family mean / 95% CI | Top-1 | WDL CE delta | Policy KL |
|---|---:|---:|---|---:|---:|---:|
| adjusted cycle | 399 | +0.017953 | +0.008029 / [-0.002180, +0.018347] | 34.09% | +0.022610 | 0.016725 |
| unadjusted cycle | 67 | +0.017134 | +0.009384 / [-0.004951, +0.025074] | 74.63% | +0.031047 | 0.021677 |
| ordinary non-cycle | 5504 | +0.006367 | +0.005822 / [+0.002999, +0.008786] | 73.42% | +0.006043 | 0.022581 |
| endgame | 1079 | +0.014929 | -0.003775 / [-0.025978, +0.010252] | 59.31% | +0.029394 | 0.023552 |
| opening | 2098 | +0.006485 | +0.006581 / [+0.002962, +0.010320] | 82.98% | -0.005061 | 0.024196 |
| midgame | 2793 | +0.004884 | +0.003831 / [+0.000036, +0.007588] | 66.09% | +0.008329 | 0.020135 |
| tactical | 5596 | +0.006123 | +0.005659 / [+0.002965, +0.008428] | 71.77% | +0.006586 | 0.021933 |

分类存在重叠：opening/midgame/endgame/tactical 是状态属性；adjusted/unadjusted
cycle 两类互斥。state-weighted delta 和 family mean 可能不同，候选门应以
family paired 统计为主。

## Value 与 KL

| epoch | Value-step peak KL | candidate final KL | final family KL | 全候选 Policy KL |
|---:|---:|---:|---:|---:|
| 0.12 | 0.000528 | 0.000528 | 0.000518 | 0.014592 |
| 0.24 | 0.000839 | 0.000473 | 0.000496 | 0.014599 |
| 0.48 | 0.001625 | 0.000230 | 0.000216 | 0.022180 |

这证实：

- 0.48 的 Value 阶段训练过程峰值超过 0.0015；
- 其最终 KL 已恢复到阈值内，因此不能用早期峰值描述最终候选；
- 更大的异常来自 Policy 阶段和累计更新：仅 6 个 Policy steps 后候选总漂移
  已约 0.0146，而现有 KL 门只约束单次 Value phase。

Aggregate WDL family delta：

| epoch | family mean delta | 95% CI | 结论 |
|---:|---:|---|---|
| 0.12 | +0.006145 | [-0.002080, +0.014001] | 统计不确定 |
| 0.24 | -0.000429 | [-0.011118, +0.010991] | 中性 |
| 0.48 | +0.001574 | [-0.012290, +0.018530] | 中性 |

因此 Value 峰值问题会阻止较晚候选，但 0.12/0.24 的 Policy 已先失败。

## 数据与采样事实

- held-out：5970 states、103 trajectory families；
- replay 实际没有“完全停止旧 Policy replay”：
  Policy loss 仍为 fresh 70%、最近两轮 15%、Oracle anchor 10%、cycle anchor 5%；
- fresh 和 recent Search minibatch 都固定约 18% cycle/endgame；
- Iteration 107 cap 前 cycle roots 4377，adjusted 3764；
- gate 实际读取的 cap 后 held-out 中 adjusted cycle roots 399、unadjusted 67；
- 0.24 时 fresh-train excess 也从 0.218417 变为 0.219430，先轻微恶化；
- 0.48 时 fresh-train excess 改善到 0.217438，但 held-out 恶化到 0.226095，
  表明继续更新开始拟合训练混合分布而伤害 held-out family。

## 下一步实验顺序

不要直接从 107 无条件跑到 120。继续固定 Iteration 107 数据，不生成新对局：

1. Policy-only 因果实验：冻结 Value，分别比较当前混合与 fresh-only；
2. 在同一 batch 顺序下测试 Policy head LR `1.0x / 0.5x / 0.25x`，trunk
   保持更小倍率；同时约束整个候选相对 incumbent 的 Policy KL；
3. 只有 Policy-only 能在 cycle-aware held-out 上改善或统计非退化后，才加入
   Value step；
4. Value 单独测试 trunk LR 和自适应 KL，避免峰值超过 0.0015；
5. 若低 LR 的 current mix 仍失败，再单独测试把 cycle/endgame quota 从 18%
   降到 10%–12%；不要同时修改 repeat scale；
6. 若上述仍失败，再单独比较 repeat visit scale 0.25 与 0.5。

这样每次只改变一个因果因素，避免再次把采样比例、标签强度、Policy LR 和
Value 干扰同时混在一起。

## 产物

- 机器可读总报告：
  `runs/open_junqi_search_v3_strength_continuation/diagnostics/iteration_000107_target_semantics_audit/report.json`
- 冻结证据清单：
  `.../frozen_evidence_manifest.json`
- 旧 Search64 held-out 缓存：
  `.../legacy_search64_targets.npz`
- 候选 checkpoint：
  `.../candidates/epoch_0p12.pt`、`epoch_0p24.pt`、`epoch_0p48.pt`

正式 lineage 的 replay 和 checkpoint 均未被覆盖。
