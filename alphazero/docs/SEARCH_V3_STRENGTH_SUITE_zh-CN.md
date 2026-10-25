# Open Junqi Search v3 Strength Suite

## 1. 固定边界

本阶段没有进入镜像，也没有改变固定布局。唯一布局仍为：

```text
public_conservative_advance
```

`open_junqi_search_v3_alternating` 已冻结在正式通过的 iteration 10。冻结清单
记录 iteration 10 checkpoint 和报告的 SHA-256，原 runner 在请求 iteration
10 之后的轮次时会拒绝继续运行。

新的只读 lineage：

```text
runs/open_junqi_search_v3_strength
```

incumbent 是 iteration 10 checkpoint 的逐字节副本。该 lineage 禁止训练和
replay 写入。

## 2. single_layout_strength_suite

suite 包含 1024 个唯一、合法、可从固定初始局面完整重放的非终局状态，来自
409 条不同 trajectory family。

| 主分类 | 数量 |
|---|---:|
| 开局 | 160 |
| 中局 | 320 |
| 残局 | 160 |
| 战术 | 320 |
| 循环入口 | 64 |

每个状态保存：

- 固定布局 seed 和完整 action trace；
- 每一 ply 的 position-key history；
- 当前完整 state key、position key、棋盘和行动方；
- turn count、无吃子计数和 repetition count；
- 全部合法动作及逐动作 repetition context；
- Oracle 使用的完整 repetition-count payload；
- 规则上限和固定先手；
- 107×12×5 feature 的 SHA-256。

加载时会从固定初始布局逐步执行完整 action trace，并重新核验 legal action、
完整状态 key、重复历史、repetition context 和 feature hash。1024 个状态均
通过重放校验。

suite 位于：

```text
runs/open_junqi_search_v3_strength/single_layout_strength_suite
```

`training_excluded=true` 且 `replay_write_prohibited=true`，不会进入任何训练
或超参数拟合。

## 3. 评测定义

同一个冻结 iteration 10 网络产生两种确定性动作：

- `bare`：合法动作 mask 后的 Policy argmax；
- `Search64`：无噪声、greedy、64 simulations 的 Neural PUCT。

对每个 suite 状态，分别用
`supplement_wide_quiescence@500/2000/8000/32000` 对全部合法动作评分。

对 agent 所选动作定义：

```text
regret = Q_oracle(best legal action) - Q_oracle(agent action)
```

报告输出 mean/median regret、Oracle top-1 agreement、Search64 相对 bare
regret 变化、better/equal/worse 状态数，并按完整 trajectory family 做 paired
bootstrap，而不是把同一轨迹中的多个状态当成完全独立样本。

exact082 当前接口只返回动作，没有可校准的逐动作 Q，因此对 exact082 输出
top-1 action agreement；不能把该数值解释成胜率或 regret。

## 4. iteration 10 实际结果

| 参考 | bare mean regret | Search64 mean regret | Search64 变化 | family 级结论 |
|---|---:|---:|---:|---|
| Oracle 500 | 0.212665 | 0.236038 | 恶化 10.99% | 明确恶化 |
| Oracle 2000 | 0.151639 | 0.175919 | 恶化 16.01% | 明确恶化 |
| Oracle 8000 | 0.107052 | 0.115726 | 恶化 8.10% | 明确恶化 |
| Oracle 32000 | 0.063261 | 0.063552 | 恶化 0.46% | 统计中性 |

family 级 `Search64 regret - bare regret`：

| Oracle 预算 | mean delta | 95% CI |
|---|---:|---:|
| 500 | +0.027357 | [+0.003566, +0.052640] |
| 2000 | +0.032001 | [+0.012742, +0.052925] |
| 8000 | +0.014818 | [+0.004774, +0.025535] |
| 32000 | +0.002521 | [-0.002584, +0.007836] |

Search64 与 bare 在 81.05% 状态选择相同动作。对 exact082：

| 指标 | bare | Search64 |
|---|---:|---:|
| exact082 top-1 agreement | 24.80% | 27.54% |

在 bare 与 Search64 不同的 194 个状态中，exact082 选择 bare 动作 17 次，
选择 Search64 动作 45 次。

## 5. 分阶段观察

以 Oracle 32000 为参考：

| 分类 | bare regret | Search64 regret | Search64 变化 |
|---|---:|---:|---:|
| 开局 | 0.0778 | 0.0969 | 恶化 24.6% |
| 中局 | 0.0607 | 0.0479 | 改善 21.1% |
| 残局 | 0.0430 | 0.0456 | 恶化 6.0% |
| 战术 | 0.0687 | 0.0710 | 恶化 3.4% |
| 循环入口 | 0.0634 | 0.0662 | 恶化 4.3% |

因此当前 Search64 不是普遍无效：它在中局有清楚的 regret 收益，也更常与
exact082 一致；但开局损失抵消了这些收益。原来的 512 probe 对
`Search64 > bare` 的判断不能推广到这个更广、固定且训练隔离的 1024 状态
suite。

当前结论是：闭环训练 lineage 可以冻结为“流程验证成功”，但 iteration 10
尚不能被认定为“Search64 全局增强的强度版本”。下一步应先在本 suite 上检查
开局 Value calibration、PUCT backup 和 prior/value 权衡，不应修改布局或引入
镜像。

## 6. 命令

完整构建并评测：

```powershell
cd .
.\run_search_v3_strength.ps1 -Workers 16
```

只构建或验证 suite：

```powershell
.\run_search_v3_strength.ps1 -BuildOnly -Workers 16
```

只执行或恢复评测：

```powershell
.\run_search_v3_strength.ps1 -EvaluateOnly -Workers 16
```

Oracle 评测按 32 个状态事务提交；中断后会复用完整事务。最终输出：

```text
runs/open_junqi_search_v3_strength/evaluation/report.json
runs/open_junqi_search_v3_strength/evaluation/per_state.csv
```

## 7. 控制台状态

后续训练控制台增加三类行：

```text
[championship] incumbent=... challenger=... phase=selfplay
[throughput] games=... samples=... selfplay-sps=... unique-state=... repetition=...
[championship] status=PROMOTED|INCUMBENT_RETAINED_EPOCH0|PAUSED ...
```

每个 self-play transaction 同时显示累计和当前块 simulations/s、重复局数；
strength suite 显示 positions/s；Search64 显示 simulations/s；Oracle 显示
states/s 和 ETA。`PROMOTED` 只在本轮实际接受非 epoch-0 参数更新且全部 gate
通过时显示。
