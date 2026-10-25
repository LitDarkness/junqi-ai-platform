# open_junqi_search_v3：固定布局搜索学习链

## 1. 目的与硬边界

本 lineage 用一个固定、双方相同的 `public_conservative_advance` 布局，依次回答五个问题：

1. `supplement_wide_quiescence` 是否能作为可复现 Oracle；
2. 固定布局下能否生成足够多、无 family 泄漏的状态；
3. 6.9M Policy/WDL 网络能否拟合 Oracle；
4. 同一冻结网络上 Neural PUCT 是否强于裸 Policy；
5. `Policy → Search → visits 蒸馏 → 新 Policy` 是否能连续形成正向闭环。

第一代固定红方先行，不启用镜像、多布局、Setup Network、历史人口、PSRO、PPO/advantage、和棋惩罚或重复禁着。规则核仍使用正式的三次重复、200 ply 无吃子和 1000 ply 上限。

主 run 为 `runs/open_junqi_search_v3`，默认完全随机初始化。消融配置
`config/open_junqi_search_v3_v2b_ablation.json` 只复制 v2b bootstrap 的 encoder，
Policy/WDL head、optimizer、scheduler、replay 和 RNG 都重新初始化。

## 2. 阶段门与自动暂停

| Gate | 命令中的 Stage | 通过条件摘要 | 失败行为 |
|---|---|---|---|
| 00 | `boundary` | 固定 layout/rule/config hash；独立随机 checkpoint | 写 `STATUS.json` 并暂停 |
| 01 | `oracle` | 红黑、强制根着、历史重复、终局符号、总预算、PV、稳定性 | 不允许生成数据 |
| 02 | `diversity` | 1024 family；20k–40k 去重状态；family split；轨迹/前缀占比 | 不允许监督训练 |
| 03 | `supervised` | 2048 fixed-batch overfit；held-out Policy/WDL | 不允许 Search Gain |
| 04 | `search-gain` | PUCT64–bare、外部对手、32k probe regret/循环退出 | 不允许搜索自博弈 |
| 05 | `closed-loop` | 5–10 个连续搜索—蒸馏检查轮 | 任一硬故障或连续两次 Search Gain 失效即暂停 |

每个 gate 位于 `runs/open_junqi_search_v3/gates`。失败文件包含定量 `reasons`；
后续 stage 会检查前一 gate 的 `passed=true`，不能靠命令行跳过。

## 3. Oracle 的定义

统一名称为 `supplement_wide_quiescence`。输入是完整 canonical 棋盘、行动方、重复历史、无吃子计数、ply 和正式规则阈值。根总预算仅可取
`500/2000/8000/32000` nodes，含义是整个根局面的节点总数。

Adapter 先浅搜所有合法着，再以 common-depth、前置预算的 successive halving
细化候选。每个合法着保存：

- `raw_score`；
- `Q = tanh(raw_score / score_scale)`，终局着直接为 `-1/0/+1`；
- canonical PV；
- 实际/分配节点；
- 终局原因；
- 是否进入正式重复；
- 根最佳着。

`score_scale` 是独立校准状态上 `|raw_score|` 的冻结 90% 分位数，不按局面动态变化。当前 Oracle 门同时报告：

- 8k/32k exact top-1 一致率；
- 8k 动作进入 32k top-3 的比例；
- 用 32k Q 衡量的 8k 平均 regret。

近似并列着会降低 exact top-1，但不应单独否决 Oracle，因此采用三项复合门。

## 4. 数据与标签

1024 个 `trajectory_family` 的对手权重固定为 40% Oracle、25% exact082、
15% random、10% aggressive、10% 冻结弱网络。分支数分布为
`0/1/2/3 = 10%/40%/35%/15%`。分支着只从 500-node Oracle top-5
近优着按温度 0.25 采样；扰动着本身不成为标签，到达的新状态会重新搜。

红方始终是规则意义上的先手，但 Oracle actor 不能永远控制红方：偶数 family
由 Oracle 控红、奇数 family 由 Oracle 控黑，另一方才使用抽中的混合对手。这样
既保持同一个固定布局/先手实验边界，又避免 WDL 利用绝对红黑 plane 学成
“Oracle 所在一方必然更强”的代理身份脚本。

普通状态用 2k，分支后、循环入口、战斗附近等重要状态用 8k。512 个 held-out
状态另用 32k 写入不可变 `probe_manifest.json`，不进入训练。

Policy 标签为

```text
y(a|s) = exp((Q(a)-max Q)/tau) / Σ_b exp((Q(b)-max Q)/tau)
```

`tau=0.1/0.2/0.5` 分别做短消融，再按 validation CE、excess、top-1 和标签熵选择。
不使用 `target_power=4`、`confidence_exponent=2` 或 one-hot。

Value 为行动方视角 W/D/L，主要标签只来自完整终局。网络自己的 root-Q 不作真值。
额外的 Oracle 标量辅助项使用独立 Oracle 标签

```text
V_theta(s) = P(W|s) - P(L|s)
q_oracle(s) = max_a Q_oracle(s,a)
L_oracle = MSE(V_theta(s), q_oracle(s))
```

正式目标为 `L_policy + L_WDL + 0.1 L_oracle`。这个辅助项只校准 Value 的幅度和
方向，权重不超过 terminal WDL 的 0.1；它不使用网络自己的浅层 root-Q，也不替代
真实终局 WDL。

## 5. 监督指标的定量含义

- `policy_ce = -Σ_a y(a) log p(a)`，越低越好；
- `target_entropy = -Σ_a y(a) log y(a)`，给定标签的理论 CE 下界；
- `excess = policy_ce - target_entropy`，理想为 0；
- `uniform = log(number_of_legal_moves)`，合法着均匀基线；
- `top1`：网络最大概率着与 Oracle 标签最大概率着一致率；
- `illegal_probability`：mask 后非法着总概率，必须为 0；
- `wdl_ce = -Σ_c ξ_c log P(c)`；
- WDL constant baseline：训练集 W/D/L 类别先验在 held-out 集上的 CE。

固定 2048 样本会被反复训练，关闭 weight decay、KL 和 symmetry，并严格分成
Policy-only 与 WDL-only 两段。数据诊断还会计算相同网络输入发生标签冲突时的
Bayes top-1 上限。当前样本中 37.2% 存在 Oracle 第一名并列，54.4% 的第一/第二名
Q 差不超过 0.001；因此 Policy-only 正确性测试以完整软标签 CE 为主，同时加入
仅用于该测试的 `0.05 × top-set CE`，为 float16 量化级近并列提供可辨识的排序梯度。
这项辅助不进入正式训练，正式训练仍只拟合完整 Oracle 软分布。

只有 `excess<0.05`、top-1 达到数据可实现上限的 95% 以内、WDL accuracy≥95%、
WDL CE≤0.15 且非法概率为 0，才允许正式训练。当前固定批次 Policy 在 200 steps
达到 `excess=0.0101, top1=95.2%`，证明动作编码和网络容量可以通过此门。

## 6. Neural PUCT

树遍历、状态 clone/apply、终局判断、full state key 和 backup 在 C++。
Python 只把多个并行根的 leaf 合并为一个 GPU batch。合法 mask 与重复上下文先在
CPU 组成完整 batch，再各做一次 CPU→GPU 传输；Policy/WDL 也整批一次性回传，避免
逐根同步。合法先验来自网络 Policy，叶值为 `V=P(W)-P(L)`。

每个 checkpoint 还保存一个只由 Oracle validation 状态拟合的叶值尺度：

```text
beta = clip(Σ_s V_theta(s) q_oracle(s) / Σ_s V_theta(s)^2, 0, 1)
V_leaf(s) = beta V_theta(s)
```

该尺度不使用 probe/test，也不按单个局面变化。它防止校准尚不充分的 Value 以过大
振幅覆盖 Policy prior；每轮训练前后都会重算并写入 checkpoint 和 iteration 报告。

选择公式为

```text
a = argmax_a [ Q(s,a) + c_puct P(s,a) sqrt(N(s)) / (1+N(s,a)) ].
```

backup 不盲目按 ply 翻号，而是比较每个节点的实际 `current_player` 与 leaf
视角；这也覆盖“吃旗后 native current_player 不切换”的终局细节。

Search Gain 固定关闭 Dirichlet 和采样，比较 bare 与 8/16/32/64/128 simulations。
最低门为：

- PUCT64 对 bare 的 512 局 score >52%，或 95% CI 下界 >50%；
- PUCT64 外部得分不比 PUCT32 低 2pp 以上；
- PUCT64 probe regret 至少比 bare 降 10%；
- `avoidable_repetition` 的退出率不能下降。

由于第一代只有一个固定布局且所有评测均为 greedy，512 局中可能出现完全相同的
确定性轨迹。报告仍按批准的 512 局门输出，同时保留轨迹去重/重复诊断；不得把重复
复制的确定性对局误解为 512 个独立统计样本。真正的跨局统计功效要到镜像/多布局
generation 才会增加。

因此固定布局 greedy 外部赛仍完整报告 W/D/L，但当每个 matchup 少于 32 条真正不同
轨迹时，`pi_(k+1)≈Search(pi_k)` 不使用其 5pp 波动作硬门，而使用本轮新鲜 held-out
Search-visits 的 excess 是否下降。外部 bare 与 Search 同时退化仍作为系统退化诊断；
达到 32 条独立轨迹后，外部 closure 门自动恢复为硬门。

## 7. 搜索自博弈与闭环

只有 Gate 04 通过才会运行。每轮冻结 actor，生成 1024 个完整游戏。每个真实落子根独立加入：

```text
epsilon = 0.25
alpha = clip(10/legal_count, 0.03, 1.0)
```

ply<24 从原始 visits 以温度 1.0 采样，之后温度 0.5。当前状态已经重复或任一着
会触发正式重复终局时，本手升到 128 simulations 且温度恢复 1.0；不禁止或惩罚重复。

Policy target 是原始归一化 visits。同一 full state hash 每轮最多贡献 4 个 Policy
样本。配置的事务下限为 64 局；为填满 GPU 推理 batch，有效收集/原子提交窗口取
`max(transaction_games, inference_batch_size)`（当前为 256 局）。窗口完成前只报告
窗口内 finished/active/ply，不提交不完整游戏；异常恢复会重做整个未提交窗口，已经
提交的 game range 不重复写入。NPZ 每个数组只解压一次，汇总 state hash 时只读取
必要字段，避免过去逐样本重复解压造成的内存峰值。

最近 5 轮 replay 采用 Algorithm R 均匀 reservoir sampling：读取时即限制为 200,000
个样本，不再先载入全部历史后裁剪；被保留的行会复制脱离 NPZ backing array，避免
一个样本切片把整个旧 shard 固定在内存中。`progress.json` 记录 eligible/retained 和
实际采样算法。

每轮最多训练 3 个 epoch，并按独立 validation 的 Search visits、Oracle Policy、WDL
和循环入口 Policy 综合选择最佳 epoch，而不是无条件采用最后一轮。闭环训练同时使用：

- 本轮新鲜 Search visits（主蒸馏目标）；
- 原监督集 Oracle Policy/WDL 小权重 anchor，避免遗忘；
- 8000-node 循环入口 Oracle Policy anchor，保留有价值的退出动作。

Search visits、Oracle anchor 与循环 anchor 的 legal mask、重复上下文和目标分布均先
在 CPU 构成 dense batch，再各自一次性传入 GPU；不再对 256/1024 个样本逐行发起
小型 CUDA copy。

循环 anchor 不禁止重复，也不修改重复终局值，只监督 Oracle 对合法动作的相对评分。

模型、Adam optimizer、常量 scheduler 和闭环 shuffle RNG 必须来自同一个被选中的
epoch，再一起原子写入 checkpoint。审计发现第 1 轮旧实现选择了 epoch 2 的模型、却
保留 epoch 3 的 Adam 动量；第 2 轮恢复边界因此只做一次 optimizer/scheduler/RNG
重置，模型与已提交的 256 局完全保留。迁移证据写在
`loop/ATOMIC_TRAINING_STATE_MIGRATION.json`，后续 checkpoint 不再允许这种混配。

每轮检查：

```text
pi_k < Search(pi_k) ≈ pi_(k+1) < Search(pi_(k+1))
```

并记录完整轨迹、8/16/24 ply 前缀、unique-state ratio、root entropy、终局原因、
重复率和 simulations/s。validation 快速恶化、数据多样性越界或 Search Gain
连续两个窗口失效会自动暂停。

数据门不仅检查 `unique_state_ratio>=0.2` 和任一 8/16/24-ply 前缀不超过 20%，
也检查单一完整轨迹不超过 2%；若 unique-state ratio 单轮下降超过 20 个百分点，
即使尚未跌破绝对下限也会按“多样性快速下降”暂停。

## 8. 命令

构建与测试：

```powershell
cd .
.\build.ps1
.\run_tests.ps1
```

安全清理先 dry-run，再按同一白名单执行：

```powershell
.\clean_open_junqi_search_v3.ps1
.\clean_open_junqi_search_v3.ps1 -Apply
```

逐阶段运行：

```powershell
.\run_open_junqi_search_v3.ps1 -Stage boundary
.\run_open_junqi_search_v3.ps1 -Stage oracle
.\run_open_junqi_search_v3.ps1 -Stage diversity
.\run_open_junqi_search_v3.ps1 -Stage supervised
.\run_open_junqi_search_v3.ps1 -Stage search-gain
.\run_open_junqi_search_v3.ps1 -Stage closed-loop
```

前四门通过后，正式 10-round 长跑建议使用带防休眠的入口：

```powershell
.\run_open_junqi_search_v3_long.ps1
```

批处理优化后，纯 benchmark 在 64 roots×64 simulations 为约 2,628 simulations/s，
256 roots×64 为约 5,375 simulations/s；正式第 1 轮（含长短局、重复敏感节点和数据
落盘）为 3,220 simulations/s。每轮还含前后 probe/Arena 与最多 3 个训练 epoch，实际
通常约 50–65 分钟，10 轮应按约 9–11 小时预留。脚本阻止空闲睡眠，但合上笔记本盖
仍可能触发 Modern Standby。

或让门控链自动执行到指定阶段；任一失败会返回非零并停止：

```powershell
.\run_open_junqi_search_v3_pipeline.ps1 -Through closed-loop
```

监控与独立性能测试：

```powershell
.\monitor_open_junqi_search_v3.ps1
.\monitor_open_junqi_search_v3.ps1 -Watch -IntervalSeconds 10
.\benchmark_open_junqi_search_v3.ps1 -Roots 256 -Simulations 64
```

Gate 04 失败后的只读 leaf diagnostics（不会打开 gate）：

```powershell
python tools\diagnose_search_v3_leaf_value.py
python tools\diagnose_search_v3_external_leaf.py
```

v2b trunk 消融必须使用独立配置：

```powershell
.\run_open_junqi_search_v3_pipeline.ps1 `
  -Config config\open_junqi_search_v3_v2b_ablation.json `
  -Through oracle
```

## 9. 代际扩展

单布局连续 5–10 轮稳定闭环后结束本 lineage，不直接修改当前 config。新建 run
generation，依次加入水平镜像、两个布局、四个布局；每次重新生成 split、probe
manifest 和 config hash。原单布局 checkpoint、probes 和 Search Gain 报告永久作为回归基准。

## 10. 当前已验证结果（2026-07-26）

- Gate 00：通过；layout hash 为
  `2eef21ed...e0c38`，Policy/WDL 网络参数量为 `6,895,769`。
- Gate 01：通过；冻结 `score_scale=23.045`，8k→32k exact top-1 为
  59.375%，top-3 为 75%，平均归一化 regret 为 0.01379。
- Gate 02：第一次因 16-ply 前缀占比 34.28% 自动暂停；修正受控开局分支后重跑通过。
  最终为 1024 families、33,696 个去重状态、512 个不可变 probes；
  最大单 family 占比 1.804%，最大 16-ply 前缀占比 6.641%。
- C++ Neural PUCT：批量传输优化后，256 个并行根、每根 64 simulations 时达到
  约 5,375 simulations/s；64 次 GPU batch、平均 batch=256。
- 第一版 Gate 03 曾因 tau 选择器偏爱接近均匀的 `tau=0.5` 而暂停；修正为“先
  CE 胜过合法均匀，再最大化捕获的 Oracle 信号比例”后，以 `tau=0.1` 通过。
- 第一版 Gate 04 随后正确暂停：PUCT64 probe regret 改善 28.4%，但外部得分
  由 PUCT32 的 0.125 降至 0，avoidable-repetition 退出率由 100% 降到 90.4%。
  leaf ablation 中 zero-value PUCT64 对 wide_quiescence 为 0.75，而 learned-value
  为 0，确认 Value 在分布外误导搜索。
- 根因是第一版生成器始终让 Oracle 控红、混合对手控黑，WDL 因绝对 side plane
  学到了代理身份。该完整 run 已冻结为
  `runs/open_junqi_search_v3_generation_1_asymmetric_value_failure`；当前主 run
  从随机权重和空数据重新开始，使用 50/50 Oracle actor 换边。
- balanced generation 的 Gate 02/03/04 已通过：36,810 训练状态、512/512
  Oracle actor 换边；validation Policy CE `3.0361 < 3.1132`，WDL CE
  `0.2385 < 0.9570`；PUCT64 probe regret 改善 22.44%，32/64/128 regret
  为 `0.01233/0.01212/0.01200`，外部 32/64 得分均为 0.25。
- Gate 05 第 1 轮已通过：1024 局产生 130,928 个 Policy 样本、130,181 个去重状态，
  unique-state ratio 99.43%，重复 6/1024，正式吞吐 3,220 simulations/s。
  蒸馏 excess `1.2219→1.1994`，WDL CE `1.5794→0.7020`；Search64 对 bare 的直接
  score `0.75→1.00`，外部 Search 平均分 `0.25→0.375`。叶值尺度由 `0.6279`
  校准到 `0.8078`。
- 第 1 轮并非所有指标都单调：Search64 probe regret `0.01187→0.01454`，循环退出率
  `58/61→57/61`。Search 仍优于同代 bare，下降未越过 2pp 容差，因此本轮通过；这应
  视为“获得一轮证据”，不是已经证明长期稳定闭环。
- Gate 05 第 2 轮也已通过：1024 局、90,749 个去重状态、unique-state ratio 99.14%、
  重复 2/1024、3,826 simulations/s。Search-target excess `0.7499→0.7284`，WDL CE
  `0.5885→0.5447`；Search64 regret `0.01454→0.01434`，循环退出 `57/61→59/61`，
  外部 Search 均分 `0.375→0.625`，Value calibration MSE `0.1827→0.1565`。
- 第 2 轮曾被旧 external-closure 门暂停：post bare 外部分从 0.5 翻到 0.25。但每个
  matchup 只有 2 条确定性轨迹，名义 128 局是复制而非独立证据；系统 Search 反而显著
  改善。Gate logic v3 保留该诊断，只在至少 32 条独立轨迹时采用 5pp 外部 closure
  硬门，否则使用 8,795 个新鲜 held-out Search target 的 excess。
- Gate 05 当前正在第 3 轮。它会在数据多样性硬故障、validation 恶化，或 Search Gain
  连续两个检查窗口失效时写出原因并自动暂停；在完成 5–10 个连续有效窗口前，不宣称
  整个 lineage 已通过。
