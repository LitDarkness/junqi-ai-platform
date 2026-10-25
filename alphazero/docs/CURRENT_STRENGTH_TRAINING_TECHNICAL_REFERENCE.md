# 当前固定布局 Strength Continuation 技术参考

更新时间：2026-07-29  
适用主线：`runs/open_junqi_search_v3_strength_continuation`  
当前恢复边界：iteration 99 完整 checkpoint，iteration 100 使用循环感知 replay
重新生成  
当前配置 hash：
`d8ed1e1227d732e38ee0e11aee9b71f9bba0913c31186ba8156b5571e56a1074`

本文只描述当前正在训练的 Search-v3 固定布局 strength continuation。旧
AlphaZero、fixed-layout PPO/advantage、Oracle 数据生成和历史消融实验不等于
当前训练流程。

## 1. 配置和代码的权威层级

当前参数来自四个不同层级，不能只看一个 JSON：

| 层级 | 权威位置 | 内容 |
|---|---|---|
| 启动参数 | `run_search_v3_strength_continuation.ps1` | 目标 iteration、CPU 线程、Arena worker、Arena block |
| 当前子配置 | `config/open_junqi_search_v3_strength_continuation.json` | run 目录、初始化方式、最大 iteration |
| 基础配置 | `config/open_junqi_search_v3_alternating.json` | 网络、规则、PUCT、batch、数据和通用 loop 参数 |
| 当前训练契约 | `python/junqi_search_v3/recovery.py::RECOVERY_CONFIG` | Policy/Value 比例、loss 混合、分阶段 LR、KL 和 family bootstrap |
| 循环治理常量 | `python/junqi_search_v3/loop.py` | Search256、epsilon、重复 visits 缩放、replay 裁剪 |
| Gate | `python/junqi_search_v3/alternating_lineage.py`、`strength_continuation.py`、`championship.py` | 候选资格、dev gate、冠军赛和重复非劣门 |

`strength_continuation.json` 使用 `extends` 继承
`open_junqi_search_v3_alternating.json`。同名嵌套字段递归覆盖；未出现的字段
继续使用基础配置。

### 1.1 修改参数前必须知道的限制

JSON 配置参与 checkpoint 的 `config_hash`。直接修改当前主线 JSON 后尝试恢复
iteration 99，会因 hash 不一致而拒绝加载。这是防止实验语义悄悄变化的保护。

以下参数目前写在 Python 训练契约中，不属于 JSON config hash：

- `RECOVERY_CONFIG` 中的 Policy/Value 混合、LR 倍率、KL 和 bootstrap 参数；
- `loop.py` 中的循环 Search256、`epsilon=0.02`、重复 visits 缩放；
- Gate 的固定阈值。

修改这些代码常量虽然不会触发 config-hash 拒绝，但同样改变实验语义。正式
修改应建立新的 lineage/config/run directory，并在 manifest 中记录代码版本，
不能把变化后的结果继续解释为原实验。

## 2. Heuristic082 到底搜索多少

### 2.1 实际加载对象

配置路径：

```text
../teachers/082/candidate.py
```

相对于 `.` 解析为：

```text
../teachers/082\candidate.py
```

`candidate.py` 加载同目录的 `fast_rule_ab.cp310-win_amd64.pyd`。其对应源码为：

```text
../teachers/082\fast_rule_ab.cpp
```

### 2.2 它不是 128/256 次 MCTS

exact082 每次轮到它落子时调用一次：

```text
MTD(f) + iterative-deepening alpha-beta
```

当前编译源码的有效预算是：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| soft node budget | 3000 | 低于该余额才允许开始下一层 iterative deepening |
| hard node budget | 6000 | 正在搜索的一层最多继续到此上限，达到即中断 |
| wall-clock safety | 50 ms | 每 256 个节点检查一次；到期中断 |
| depth | 2 到 7 | 完成一层后才进入下一层 |
| root width | 24 | 根节点最多搜索 24 个排序后的动作 |
| inner width | 24 | 内部节点最多搜索 24 个动作 |
| fallback width | 18 | 搜索未完成时使用静态排序的前 18 个动作产生 fallback |
| TT size | `2^18` entries | 单次落子搜索的置换表 |
| LMR | 第 5 个动作起 | 非 emergency 且 depth≥2 时先少搜一层 |
| move ordering | TT move、两个 killer moves、静态 action score | 提高 alpha-beta 截断率 |

因此准确说法是：

> exact082 每个决策通常在 3000 节点附近停止开启新深度，但当前深度允许继续；
> 每步绝对上限是约 6000 个计数节点，并可能因 50 ms、终局或完成 depth 7
> 更早结束。

源码顶部存在：

```cpp
constexpr int kMaxNodes = 4000;
```

但它没有被任何搜索路径引用，不能把 4000 当成实际预算。

### 2.3 “节点”如何计数

`alphabeta()` 在非终局调用 `BudgetController::tick()`；MTD(f) 的多次零窗口
Alpha–Beta 探测共享同一个预算。根函数本身不 tick，递归访问的子节点和叶节点
会消耗预算。因此：

```text
单步 exact082 节点数 ≤ 6000
```

但不是固定恰好 3000 或 6000。

### 2.4 黑方为什么仍传 `player_id=1`

`AZState.board()` 和 `legal_actions()` 都采用行动方 canonical 坐标：

- 黑方行动时棋盘旋转 180°；
- 棋子符号翻转，使行动方仍为正方；
- 动作也旋转到同一 canonical 坐标。

所以外层始终给 exact082 传 `player_id=1` 是有意的，不代表让黑方按红方原始
坐标走棋。

### 2.5 一次评测总共调用多少次 082

exact082 每个由它控制的 ply 调用一次。若一局总长为 \(L\)，它通常控制约
\(L/2\) 个 ply：

```text
estimated exact082 nodes
≈ games × average_ply / 2 × nodes_per_082_decision
```

`nodes_per_082_decision` 介于提前结束值和 6000 之间，不能只由局数推出精确
总量。相同完整状态在一个批次内会去重，实际 C++ 调用数可能低于逻辑根数。

当前常见评测规模：

| 场景 | exact082 相关对局 |
|---|---:|
| 每轮 post external diagnostic | bare vs 082 共 128 局；Search64 vs 082 共 128 局 |
| 正式冠军赛 | challenger bare/Search64 和 champion bare/Search64 各对 082；4×2048=8192 局 |
| cycle validation | new/baseline 的 bare/Search64 各对 082；同样 8192 局 |

限制：exact082 API 只接收 canonical 棋盘和合法着，不接收完整 repetition table
或 no-capture counter。正式外层环境仍负责判和与终局，但 082 本身不是历史感知
的重复策略网络。因此它是固定战术基线，不是完整规则历史 Oracle。

## 3. 当前网络

| 项目 | 当前值 |
|---|---:|
| 参数量 | 6,895,769 |
| 输入 | 107×12×5 |
| trunk channels | 192 |
| residual blocks | 10 |
| Policy channels | 64 |
| 动作空间 | 3600，即 `from×60+to` |
| Value | W/D/L 三分类 |
| 搜索标量值 | `P(W)-P(L)` |
| mixed precision | CUDA 上 bfloat16 autocast |

Policy 动作头：

```text
logit(i→j) = q_i^T k_j / sqrt(d) + from_bias_i + to_bias_j
             + action_bias_(i,j) + repetition_bias_context
```

随后应用 legal mask。

规则相关输入至少包括：

- 完整明棋棋盘和行动方 canonical 表示；
- 死亡棋子统计、军旗、初始位置和最近 5 步；
- `no_capture_ply/200`；
- 当前 `repetition_count/3`；
- `turn_count/1000`；
- 每个合法动作独立的 `repetition_context=0/1/2`。

完整 repetition table 保存在 C++ state，并进入 `full_state_key`；网络本身只看到
上述当前计数和一步合法着上下文。详细边界见
`docs/CYCLE_AWARE_TRAINING_20260729.md`。

## 4. 一个 iteration 的完整流程

### 4.1 恢复 incumbent

从上一完整 official checkpoint 原子恢复：

- model；
- optimizer；
- scheduler；
- global step；
- Python/NumPy/PyTorch/CUDA RNG；
- closed-loop RNG 和 batch RNG；
- replay 元数据；
- Value calibration / leaf value scale。

Frozen champion 只用于正式评测和历史人口，不是默认训练恢复点。训练一直从
latest official incumbent 累积。

### 4.2 冻结 actor 并生成 1024 局

整轮 self-play 使用同一个 iteration-start incumbent，生成期间不在线更新。

实验边界：

- 一个固定布局；
- 双方相同布局；
- 红方固定先行；
- 双方使用同一个冻结网络；
- 不镜像、不多布局；
- 正式 repetition/no-capture/max-ply 规则。

普通根：

```text
PUCT64
```

循环敏感根（当前局面已经出现至少两次，或任一合法动作
`repetition_context>0`）：

```text
PUCT256
```

注意：基础 JSON 中的：

```json
"repeated_state_simulations": 128
```

在当前循环治理版本中已不再控制循环根；实际值来自
`loop.py::CYCLE_TEACHER_SIMULATIONS=256`。该 JSON 字段目前仅为兼容遗留。

### 4.3 PUCT 参数

当前普通搜索参数：

| 参数 | 值 | 位置 | 含义 |
|---|---:|---|---|
| `selfplay_simulations` | 64 | JSON `puct` | 普通实际落子根预算 |
| cycle simulations | 256 | `loop.py` | 循环敏感根预算 |
| `cpuct` | 1.5 | JSON `puct` | prior 相对 Q 的探索强度 |
| `inference_batch_size` | 256 | JSON `puct` | 一次 GPU leaf inference 最大行数 |
| `dirichlet_epsilon` | 0.25 | JSON `puct` | 根 prior 中噪声混合比例 |
| Dirichlet alpha | `clip(10/legal_count,0.03,1)` | `search.py` | 随合法着数量自适应 |
| leaf value | network WDL | `search.py` | `P(W)-P(L)`，再乘 checkpoint leaf scale |

`puct.simulations=[8,16,32,64,128]` 是通用 benchmark 搜索曲线列表，不控制当前
self-play 的单步预算。

### 4.4 自博弈采样

实际动作从原始 root visits 采样：

| 状态 | temperature |
|---|---:|
| ply < 24 | 1.0 |
| ply ≥ 24 | 0.5 |
| 循环敏感根 | 1.0 |

Policy target 默认是 visits 的归一化分布，不做四次方锐化。

循环根额外比较已访问动作：

```text
Q_repeat = 最佳重复动作 Q
Q_exit   = 最佳非重复动作 Q
```

若：

```text
Q_exit ≥ Q_repeat - 0.02
```

则只把训练 target 中重复动作 visits 乘以 0.25。实际本局采样仍使用原始 visits；
若重复明显更优则保持原 target，允许合理保和。

### 4.5 Replay 提交与裁剪

配置：

```json
"games_per_iteration": 1024,
"transaction_games": 64
```

实际事务 block 为：

```text
max(transaction_games, inference_batch_size) = 256 games
```

所以一轮通常提交 4 个 256-game shard。只有完整结束的游戏才能进入 shard。

循环终局只保留：

```text
循环入口前全部状态 + 第一个完整循环
```

然后每局按：

```text
(position_key, current_repetition_count,
 all_legal_action_repetition_contexts)
```

去重。相同 key 每局最多一个 Policy 样本；每个 state hash 每 iteration 最多
4 个 Policy 样本。

iteration 99 及以前的旧 replay 保留为证据，但因没有 `cycle_cleaned=true`
不会再进入当前 recent training pool。

### 4.6 Train/validation split

按完整 self-play game/trajectory family 划分：

```text
game_id % 10 == 0  → held-out validation
其余               → train
```

约 90% family 用于训练，10% family 用于 held-out。一个 family 的状态不会同时
进入 train 和 validation。

## 5. 每轮训练步数如何计算

令：

```text
N_fresh = 本轮 fresh train Policy 样本数
B       = train.batch_size = 1024
E       = ceil(N_fresh / B)
```

这里 \(E\) 定义为一个 Policy epoch 的 optimizer step 数。

当前最多训练 2 epoch：

```text
maximum_policy_steps = 2E
maximum_value_steps  = E
```

执行顺序：

```text
Policy step
Policy step
Value step
重复直到 Policy 达到 2E
```

因此 Policy:Value optimizer steps = 2:1。每轮不是固定 30 或 90 steps；它随
本轮裁剪后的 fresh 样本数变化。

每约 0.25 Policy epoch 保存并验证一个候选：

```text
validation interval = max(1, round(0.25E))
```

epoch 0 incumbent 始终作为候选，允许安全 no-op。

## 6. Policy step

### 6.1 Batch 和 loss

每个 Policy optimizer step 使用四个独立子 batch：

| 来源 | 样本数 | loss 权重 |
|---|---:|---:|
| 当前轮 fresh Search visits | 717 | 0.70 |
| 最近 cycle-cleaned Search replay | 154 | 0.15 |
| immutable Oracle Policy anchor | 102 | 0.10 |
| Oracle cycle-entry anchor | 51 | 0.05 |

四项样本数合计 1024。各子 loss 先分别求均值，再乘表中权重：

```text
L_policy =
  0.70 CE(fresh_search)
  + 0.15 CE(recent_search)
  + 0.10 CE(oracle_anchor)
  + 0.05 CE(cycle_anchor)
```

Search CE：

```text
CE = -Σ_a π_search(a|s) log π_net(a|s)
```

recent pool 只接受 `cycle_cleaned=true` 的最近 1–2 轮数据。iteration 100 因没有
新格式历史 replay，recent 份额临时从当前 fresh 回填；iteration 101 起使用新
格式 recent replay。

### 6.2 Cycle/endgame 分层

fresh Search 和 recent Search 子 batch 都以 18% 为目标抽取：

- repetition context > 0；
- 当前 repetition count ≥2；
- 已标记 cycle entry；
- 存活棋子≤24或 ply≥80 的 endgame。

特殊样本按 `game_id` round-robin；单个 family 首轮上限为整个子 batch 的 2%。
如果 family 数量不足以填满 batch，才允许确定性补齐，并记录实际最大 share。

### 6.3 Policy phase 学习率

基础 LR：

```text
train.learning_rate = 1e-4
```

Policy step 参数组：

| 参数组 | 倍率 | 实际 LR |
|---|---:|---:|
| Policy head | 1.0 | `1e-4` |
| shared trunk | 0.25 | `2.5e-5` |
| WDL head | 0 | 冻结 |

所有可训练参数做 global norm clipping：

```text
max_gradient_norm = 1.0
```

## 7. Value step

### 7.1 Batch 和 loss

| 来源 | 样本数 | 权重 |
|---|---:|---:|
| 当前轮完整终局 WDL | 614 | 0.60 |
| 最近 cycle-cleaned 终局 | 410 | 0.40 |
| Oracle scalar anchor | 128 | 0.05 auxiliary |
| Policy drift KL | fresh validation 前 256 个状态 | adaptive |

总损失：

```text
L_value =
  0.60 CE_WDL(fresh)
  + 0.40 CE_WDL(recent)
  + 0.05 L_oracle_scalar
  + λ_KL KL(π_before_value || π_after_value)
```

WDL target 是真实终局 one-hot：

```text
loss/draw/win
```

当前网络自己的浅层 root-Q 不是主要 Value 真值。

### 7.2 Value phase 学习率

| 参数组 | 倍率 | 实际 LR |
|---|---:|---:|
| WDL head | 1.0 | `1e-4` |
| shared trunk | 0.10 | `1e-5` |
| Policy head | 0 | 冻结 |

### 7.3 Policy KL 护栏

每个 Value step 前冻结同一批 validation 状态的 Policy 分布，更新后计算：

```text
KL(π_before_value || π_after_value)
```

目标：

```text
1e-3
```

KL 权重初始 1.0：

- observed KL > `1e-3`：下一步权重乘 2，最大 100；
- observed KL < `0.5e-3`：下一步权重乘 0.8，最小 `1e-4`；
- 其余保持。

候选 Gate 同时检查：

```text
candidate_final_kl ≤ 1e-3
value_step_peak_kl ≤ 1.5e-3
```

早期一个低于硬上限的尖峰不会永久代替 final KL；每个候选按自己的 final/peak
判断。

## 8. 候选验证与选择

### 8.1 Policy/WDL family paired bootstrap

对同一 held-out trajectory family，计算：

```text
delta_i = metric_after_i - metric_before_i
```

CE/excess 越低越好，因此 delta<0 表示改善。使用 4000 次 family bootstrap：

| 分类 | 条件 |
|---|---|
| improved | 95% CI upper < 0 |
| clearly degraded | CI lower > 0，或平均相对恶化 >1% |
| neutral | CI 包含 0，且平均相对变化绝对值≤0.5% |
| inconclusive | 其余 |

Policy 必须为 improved 或 neutral；WDL 只禁止 clearly degraded。

### 8.2 静态 Search probe

不可变 probe 上比较 bare 与 Search64：

```text
Oracle regret = Q_oracle(best) - Q_oracle(chosen)
```

越低越好。候选要求：

```text
Search64 regret < bare regret
```

并且 Search64 regret 不得相对 epoch 0 出现“大幅退化”：

```text
candidate_search_regret
≤ baseline_search_regret + max(0.005, 0.5×baseline_search_regret)
```

静态 regret 是安全诊断，不再单独压过真实 continuation 终局结果。

### 8.3 候选排序

eligible 候选使用词典序：

```text
[
  是否为非 epoch0 更新,
  Policy/WDL improved 的项数,
  Policy/WDL family relative gain,
  -Search64 Oracle regret
]
```

先满足资格，再排序；不把多个指标随意合成一个未经标定的浮点总分。

### 8.4 真实 continuation dev gate

静态选择出的最终非零候选还要在固定的 64-state dev suite 上打：

- bare candidate vs bare incumbent；
- Search64 candidate vs Search64 incumbent；
- 每个状态换边；
- 每个 matchup 128 局；
- 总计 256 局。

接受条件：

```text
bare family CI upper ≥ 0.5
AND Search64 family CI upper ≥ 0.5
AND mean(bare family score, Search64 family score) ≥ 0.5
```

它主要排除明确实战退化。失败时恢复 epoch 0 的 model、optimizer 和 RNG，不是
只恢复模型权重。

## 9. iteration 级 Gate

最终 incumbent 必须同时满足：

- held-out Policy 未 clearly degraded；
- WDL 未 clearly degraded；
- Search64 Oracle regret 低于 bare；
- final KL≤`1e-3`；
- Value-step peak KL≤`1.5e-3`；
- avoidable-repetition exit rate 不系统下降超过 2 个百分点；
- unique-state、trajectory 和 prefix 多样性未坍缩；
- continuation dev 通过，或者最终选择 epoch 0 no-op。

没有合格更新时选择 epoch 0。安全 no-op 是一个完整、成功的 iteration，但不计
为 accepted update。

## 10. 日常 external diagnostic

每轮训练后的当前模型运行：

- Search64 vs bare：128 局；
- bare vs supplement_wide_quiescence@500：128 局；
- Search64 vs supplement_wide_quiescence@500：128 局；
- bare vs exact082：128 局；
- Search64 vs exact082：128 局。

固定布局确定性轨迹的独立性可能少于 32，因此这组结果通常标记为 diagnostic，
不能把重复的 128 局当作 128 个独立样本。

## 11. Frozen championship

触发条件：

```text
accepted_since_last_championship ≥ 3
OR
(completed_rounds_since_last_championship ≥ 10
 AND accepted_since_last_championship ≥ 1)
```

`accepted_since=0` 不会触发冠军赛；challenger/champion model hash 相同时直接
`SKIPPED_IDENTICAL_MODEL`。

每次冠军赛生成新的 sealed 1024-state suite，不进入训练或候选选择。7 个
matchup，每个状态换边，共：

```text
7 × 1024 × 2 = 14336 games
```

matchup：

1. bare challenger vs bare champion；
2. Search64 challenger vs Search64 champion；
3. bare challenger vs exact082；
4. Search64 challenger vs exact082；
5. Search64 challenger vs bare challenger；
6. bare champion vs exact082；
7. Search64 champion vs exact082。

晋升要求：

- bare 或 Search64 至少一项对 champion 的 family CI 完全高于 50%；
- 另一项不得 clearly degraded；
- challenger 对 exact082 的 Search64 指标不得明确退化；
- 同代 Search64 必须显著优于 bare；
- paired-family repetition 相对 champion 不得 clearly degraded；
- 多样性和终局结构正常。

重复非劣定义：

```text
delta = candidate repetition rate - champion repetition rate
clearly degraded iff family CI lower bound > +0.02
```

另外：

```text
selfplay repetition ≥80%
AND 单一 cycle 占重复终局≥50%
```

视为重复病态。所有无决胜终局合计必须低于 50%。

冠军赛缓存键包括 challenger/champion model hash、suite hash、规则 hash 和
search configuration hash；相同组合复用报告。

## 12. Retention

Official checkpoint：

- 最近 5 个；
- 每 5 个 iteration 永久 archive；
- 所有 frozen champion；
- 有限数量 competitive；
- latest。

通过 iteration 的普通 candidate 已被 official atomic checkpoint 完整替代，
因此删除。失败/中断且没有 official 的 candidate 才有独立恢复价值。

Replay：

- 最近 3 个完整 iteration；
- 每 10 个 iteration；
- champion iteration；
- 所有报告和 championship suite 永久保留。

已完成 candidate-dev 的临时模型删除，JSON/CSV 决策证据保留。

## 13. 参数位置与“是否真正在当前主线生效”

### 13.1 JSON `boundary`

| 参数 | 当前值 | 当前作用 |
|---|---:|---|
| `layout_set` | `fixed_layout_v1.json` | 布局库 |
| `layout_id` | `public_conservative_advance` | 唯一固定布局 |
| `fixed_first_player` | 1 | 红方先行 |
| `mirror` | false | 当前禁止镜像 |
| `multiple_layouts` | false | 当前禁止多布局 |
| `max_ply` | 1000 | 最大步数判和 |
| `no_capture_ply_limit` | 200 | 无吃子判和 |
| `repetition_limit` | 3 | 三次重复判和 |

107-plane 编码把 3/200/1000 视为冻结阈值，修改会被 config validation 拒绝。

### 13.2 JSON `train`

| 参数 | 当前值 | 当前主线状态 |
|---|---:|---|
| `batch_size` | 1024 | 生效，用于计算 epoch steps |
| `learning_rate` | `1e-4` | 生效，参数组再乘 phase multiplier |
| `weight_decay` | `1e-4` | 生效 |
| `max_gradient_norm` | 1.0 | 生效 |
| `epochs` | 2 | 与恢复契约一致，但实际循环以 `RECOVERY_CONFIG.maximum_epochs` 为权威 |
| `minimum_learning_rate` | `1e-5` | 当前 constant scheduler 未使用 |
| `terminal_wdl_weight` | 1.0 | 当前 alternating loss 显式写 0.6/0.4，非主要控制入口 |
| `oracle_value_aux_weight` | 0.05 | 值相同，但当前权威是 `RECOVERY_CONFIG.oracle_scalar_value_weight` |
| `policy_weight` | 1.0 | 用于配置合法性；当前四路 Policy loss 使用显式 0.70/0.15/0.10/0.05 |
| `overfit_batch_size` | 2048 | 监督 overfit 阶段使用，当前 iteration 不使用 |

### 13.3 JSON `puct`

| 参数 | 当前值 | 当前主线状态 |
|---|---:|---|
| `cpuct` | 1.5 | 生效 |
| `inference_batch_size` | 256 | 生效，也使事务 block 至少 256 games |
| `selfplay_simulations` | 64 | 普通 self-play 根生效 |
| `repeated_state_simulations` | 128 | 当前被 cycle Search256 代码常量替代 |
| `dirichlet_epsilon` | 0.25 | self-play 生效；正式评测关闭噪声 |
| `simulations` | `[8,16,32,64,128]` | benchmark 列表，不控制当前落子 |
| `search_gain_direct_games` | 512 | 初始 Search Gain 阶段参数；当前 strength 日常 gate 不直接使用 |
| `search_gain_external_games` | 64 | 初始阶段参数；当前日常 external 使用 `loop.check_games` |

### 13.4 JSON `loop`

| 参数 | 当前值 | 当前主线状态 |
|---|---:|---|
| `iterations` | 250 | config 允许的 lineage 上限 |
| `games_per_iteration` | 1024 | 生效且被 validation 固定 |
| `transaction_games` | 64 | 生效，但实际 block=max(64,256)=256 |
| `replay_iterations` | 2 | 最近 1–2 轮 Search replay |
| `maximum_replay_samples` | 200000 | generic reservoir 上限；当前 alternating recent 直接按两轮读取，不是主要限制 |
| `state_policy_cap_per_iteration` | 4 | 生效 |
| `check_games` | 128 | 日常 external diagnostic 每 matchup 局数 |
| `minimum_unique_state_ratio` | 0.2 | 多样性 Gate |
| `maximum_prefix_share` | 0.2 | 固定 8/16/24-ply prefix 最大占比 |
| `consecutive_search_gain_failures` | 1 | 旧 closed-loop 自动暂停参数；当前 strength 候选流程主要由当前 Gate 控制 |

### 13.5 JSON `data` 和 `oracle`

这些字段主要描述初始 Oracle dataset/probe 生成，并不代表当前每轮 self-play
仍按 40%/25% 等人口混合。

当前 self-play 是：

```text
latest frozen actor vs same latest frozen actor
```

`data.opponents`、`branch_counts`、`tau_labels`、`normal/important/probe_budget`
是初始数据/Oracle 阶段参数；不会把 exact082 以 25% 概率加入当前 1024 局。

当前仍使用的相关资产：

- immutable Oracle Policy anchor；
- cycle anchor；
- immutable 32k probe manifest；
- supplement_wide_quiescence@500 和 exact082 日常外部诊断。

## 14. 日志指标的定量含义

### 14.1 Policy

```text
policy CE = -Σ π_target log π_net
target entropy H = -Σ π_target log π_target
policy excess = CE - H = KL(π_target || π_net)
```

理论最优 excess 为 0；低于合法动作 uniform baseline 才说明优于均匀策略。

`top1` 是网络最大 logit 动作与 target 最大概率动作一致率，但高 top1 不等于
整体分布已经拟合。

### 14.2 Value

```text
WDL CE = -Σ y_WDL log p_WDL
V = P(W)-P(L)
```

CE 越低越好。Brier 是概率平方误差；draw-Brier 只看和棋概率校准；
EV-MSE 比较标量期望值。

### 14.3 Family delta

```text
mean_delta = after metric - before metric
```

对于 CE/excess，负值是改善。CI 的统计单位是 trajectory family，不是相互相关
的单个状态。

### 14.4 Search regret

```text
regret = Oracle best Q - chosen action Q
search gain = 1 - Search64 regret / bare regret
```

regret 越低越好；gain>0 表示 Search64 降低了静态 Oracle regret。但它只是
诊断，完整终局 continuation 仍是最终实战证据。

### 14.5 KL

- `candidate_final_kl`：该候选最后一个 Value step 后的 state-weighted KL；
- `candidate_final_family_kl`：先按 family 平均，再跨 family 平均；
- `candidate_final_family_kl_max`：最坏 family；
- `value_step_peak_kl`：到该候选为止的 Value-step 峰值。

### 14.6 Self-play

- `selfplay-sps`：PUCT simulations / search wall seconds，不是完整游戏 steps/s；
- `unique_state_ratio`：去重 state 数 / Policy 样本数；
- `repetition_fraction`：重复终局局数 / 总局数；
- `maximum_single_cycle_share_of_repetitions`：最大单一 cycle signature 占全部重复
  终局比例；
- `cycle_teacher.adjusted/eligible`：满足退出比较条件的根中，有多少实际降低了
  重复 visits；
- `cycle sampling fresh/recent`：Policy 子 batch 中 cycle/endgame 实际占比。

## 15. 如何安全调整

### 15.1 可以在同一主线运行时调整

以下只影响运行资源或停止目标，不改变训练语义：

```powershell
.\run_search_v3_strength_continuation.ps1 `
  -ThroughIteration 120 `
  -CpuThreads 12 `
  -ArenaWorkers 12 `
  -ArenaBlockStates 128
```

- `ThroughIteration`：本次运行停在哪一轮；
- `CpuThreads`：PyTorch CPU/OMP/MKL 线程，也作为 external opponent worker 数；
- `ArenaWorkers`：continuation/championship worker；
- `ArenaBlockStates`：评测事务块大小，只影响吞吐、RAM 和恢复粒度。

这些参数不进入 semantic config hash。

### 15.2 不应直接在当前 run 修改

以下属于实验语义：

- learning rate、weight decay、batch size；
- Policy/Value mix；
- PUCT64/256、cpuct、Dirichlet；
- cycle epsilon/visit scale；
- KL target；
- fixed layout 或规则；
- 网络结构。

正式修改方式应是：

1. 复制当前 JSON，修改 `run_directory`；
2. 在新 lineage manifest 中记录 source checkpoint；
3. 明确是否恢复 optimizer/scheduler；
4. 把 Python 常量变化写入独立 contract/hash；
5. 不覆盖当前 checkpoint、replay、championship 和报告。

### 15.3 常用实验方向

| 目标 | 首选参数 | 不建议同时改变 |
|---|---|---|
| Policy 拟合太慢 | Policy head LR multiplier；Policy steps/epoch | trunk LR、PUCT 和 target 同时改 |
| Value 破坏 Policy | value trunk multiplier；KL target/weight | 直接冻结全部 trunk 很久 |
| 数据生成太慢 | ordinary Search64；inference batch；collection并发 | 降低正式 Arena 可信度 |
| 循环仍高 | cycle Search256/512；epsilon；cycle batch fraction | 禁止重复或给和棋固定负奖励 |
| GPU 利用率低 | inference batch、同时活跃 games | 盲目增加 Python 线程 |
| Arena 太慢 | ArenaWorkers、BlockStates | 减少 paired role swap |

一次正式实验只改变一组因果相关参数，并至少等待 3–5 次 accepted update 后再用
同一 sealed benchmark 判断。

## 16. 文件索引

```text
run_search_v3_strength_continuation.ps1
tools/run_search_v3_strength_continuation.py
config/open_junqi_search_v3_strength_continuation.json
config/open_junqi_search_v3_alternating.json
python/junqi_search_v3/config.py
python/junqi_search_v3/loop.py
python/junqi_search_v3/recovery.py
python/junqi_search_v3/alternating_lineage.py
python/junqi_search_v3/strength_continuation.py
python/junqi_search_v3/championship.py
python/junqi_search_v3/retention.py
python/junqi_search_v3/training_retention.py
python/junqi_az/model.py
cpp/az_state.cpp
cpp/junqi_environment.cpp
../teachers/082/fast_rule_ab.cpp
```
