# v2a Advantage 单布局 20 轮因果验证结果

状态：**未通过，已暂停在 transaction-complete iteration 20。**

## 做对了什么

- 从 v1 iteration 40 仅加载 encoder 与 policy head；WDL、optimizer、scheduler、
  EMA、anchor、RNG、teacher replay 和 rollout 全部重新建立。
- 完成 512 局 teacher bootstrap 和 20 × 1024 局 advantage self-play。
- 没有 NaN，entropy 没有突然归零；iteration 20 为 1.590。
- 未再次形成 v1 的“第 86 ply、单一四步循环占绝对多数”的模式。
- random/aggressive 的得分均为 100%，repetition 均为 0%。
- historical checkpoint inference 批量化后，从修复前 845 moves/s 恢复到通常
  1,500–1,700 moves/s。

## 为什么仍判失败

最近五轮训练分布的 network repetition：

```text
iteration 16..20 = 8.40%, 12.99%, 19.82%, 20.21%, 20.80%
mean = 16.45%
```

超过 v2a 约定的 `<15%` 条件。iteration 20 的低温独立 self-play 更明显：

```text
W/D/L = 18/212/26
score = 48.44%
repetition = 82.81%
capture_flag = 14.45%
average ply = 79.22
```

但是最大单一 cycle 只占重复局 4.72%。因此它不是 v1 的单一固定环复发，而是大量
不同循环共同形成的广义保和策略。

外部强度也不稳定：

| 评测 | iter 5 | iter 10 | iter 15 | iter 20 |
|---|---:|---:|---:|---:|
| exact 084 score | 22.07% | 17.77% | 4.49% | 37.50% |
| exact 082 score | 11.72% | 5.66% | 4.10% | 5.47% |

iteration 20 对 random/aggressive 均为 100%，但水平镜像只有 0.20%。这说明网络
不是全面失去合法走子能力，而是仍严重依赖单布局线路，并对人口分布过拟合。

## 因果结论

Ataraxos-style fresh advantage update **有效消除了 outcome-weighted imitation
造成的单一固定循环放大机制**，但它本身不足以解决：

1. 单布局的多循环保和均衡；
2. wide teacher population 与 exact 084/082 的分布差异；
3. 单布局对水平镜像近乎零泛化；
4. 共享 trunk 中 Value 更新对 policy 的跨阶段漂移。

因此不能把 v2a 的 champion 累积解释为通用棋力提升，也不能直接启动 v2b
advantage 长跑。

## 当前执行决定

- v2a 保留 iteration 20 的 latest、EMA、champion、5/10/15/20 archives、
  最近五个 checkpoints、最新 raw rollout 和所有 compact summaries。
- v2b 已完成 **bootstrap-only**：1024 局、91,838 状态、600 optimizer steps，
  保存为 transaction-complete iteration 0。八个训练 strata（四布局 × 原始/镜像）
  均有数据；密封布局未进入 teacher 或 Value。尚未开始任何 advantage iteration。
- 下一步若继续，应先审查 iteration 20 的 Value calibration、pi_old 行为概率、
  低温循环与训练温度差异，并决定是否让 v2b 只做四布局 teacher/Value bootstrap
  后先评测、暂不启用 advantage。

机器可读判定见
`runs/fixed_layout_rl_v2a_advantage_single_layout/V2A_SMOKE_STATUS.json`。
