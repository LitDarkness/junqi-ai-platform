# 固定同布局明棋 RL：实现、配置与运行手册

## 1. 实验边界

本阶段只回答一个问题：在双方布局完全相同且固定时，一个不依赖 MCTS 的
Policy/Value 网络能否先模仿强 Heuristic，再通过人口自博弈继续改善。

固定布局来自 `optional-local-layout-library.zip` 的
`public_conservative_advance`。源文件按 A（前线）到 F（后排）记录；
C++ canonical setup 按后排到前线记录，因此导入时反转六行，并将英文棋子名
映射为规则核的整数类型。导入后必须通过库存、行营、军旗、地雷、炸弹约束验证。

每局都满足：

- 红方 canonical layout = 黑方 canonical layout；
- 黑方送入原生棋盘时做 180° 行列变换；
- 第一阶段不随机左右镜像，`mirror_probability=0`；
- 初始布局 ID、哈希和双方布局写入 manifest/provenance；
- 训练、评测若以后增加换手对局，仍以同一个 canonical 布局为成对单位。

这会有意造成单布局专精，不代表跨布局棋力。配置与 backend 已把“布局集合”和
“布局采样器”分开，后续可增加布局，但本阶段不混入第二个布局。

## 2. 规则和状态流

C++ 规则核仍是唯一事实源，负责合法着、铁路、行营、战斗、军旗、无子可动、
重复局面和终局。Python 不重新实现规则，只执行：

```text
C++ state -> 107-plane observation + legal actions + repetition context
          -> batched GPU policy/value
          -> sampled canonical action
          -> C++ apply
```

动作 ID 为 `source * 60 + target`，总空间 3600。黑方状态、动作和历史在
C++ 边界统一旋转到“当前行动方在己方视角下”的 canonical 坐标，因此同一
网络可服务双方。

## 3. 107 个输入平面

空间尺寸固定为 `12 x 5`：

| 通道 | 含义 |
|---:|---|
| 0–13 | 原规则核公共/地形/状态平面 |
| 14–25 | 当前行动方 12 种棋子身份 |
| 26–37 | 对方 12 种棋子身份 |
| 38–42 | 铁路、行营/大本营、无吃子计数、重复和回合进度 |
| 43–44 | 原始 RED/BLACK 当前行动方广播平面 |
| 45–56 | 当前行动方每枚棋子的初始位置 |
| 57–68 | 对方每枚棋子的初始位置 |
| 69–80 | 当前行动方 12 类死亡棋子数（按该类库存归一化） |
| 81–92 | 对方 12 类死亡棋子数 |
| 93–94 | 己方/对方军旗是否存活 |
| 95–96 | 己方/对方司令是否死亡 |
| 97–106 | 最近 5 个 ply 的 from/to；己方走子为 +1，对方为 -1 |

重复/追逐的直接监督还通过每个合法动作的三分类 repetition context
`0/1/2` 输入 policy bias 和 value 汇总分支。当前规则核能准确给出第三次重复
将和棋的动作；“追逐”没有另造一套与规则核冲突的 Python 规则。

## 4. 网络

默认模型：

- 192 个 trunk channels；
- 10 个 residual blocks；
- GroupNorm + SiLU；
- 64 维 from/to embedding；
- WDL-only value head；
- 6,895,769 参数，处于要求的 5M–20M 范围。

动作头：

\[
\ell(i\rightarrow j)=
\frac{q_i^\top k_j}{\sqrt d}
b_{\mathrm{from}}(i)+b_{\mathrm{to}}(j)+b_{ij}.
\]

随后：

\[
\ell_a=-\infty,\quad a\notin A_{\mathrm{legal}}(s),
\qquad
\pi_\theta(a|s)=\operatorname{softmax}(\ell)_a .
\]

`b_ij` 是完整的 3600 项可学习参数，不是只能分解的起点/终点偏置。价值头输出：

\[
(P_L,P_D,P_W),\quad V=P_W-P_L.
\]

## 5. 阶段一：宽 teacher 模仿

默认生成 512 局 teacher-vs-teacher 数据，再做 400 optimizer steps。程序从
replay 的 `game_key` 精确统计已完成局数，中断后只补缺少的局；训练 batch
需要 1024 个 position，但绝不会因此错误地生成 1024 局。
每个状态的 teacher policy 不是唯一最佳着 one-hot：

1. C++ 084 对每个合法后继做静态估值；
2. 用温度 `0.7/1.0/1.5` 分别 softmax 后求平均；
3. 75% 状态采用当前 084、25% 状态采用历史 082 的搜索选择作为投票；
4. 混入 5% 合法着均匀分布。

这种确定性交错避免每个 root 同时支付两次 alpha-beta 成本。084/082 搜索由
`teacher.workers=16` 个持久进程并发运行，绕过 Python GIL；C++ 静态后继评价
仍在主进程批量完成。网络会先学习合法
行动结构、交换、铁路机动、护旗与基础攻防，而不是复制单一预算下一次确定性的
选择。teacher target 在 replay 中以稀疏
`(legal_action_id, integer_weight)` 保存，只有训练 mini-batch 才转成 3600 维。

## 6. 阶段二：无搜索人口自博弈

网络直接按带温度的 `pi_theta` 采样，不运行 MCTS。默认每轮 1024 局，
最多 256 局并行送入 GPU。温度从 0.9 线性降到 0.25。

对手人口默认权重：

| 对手 | 权重 |
|---|---:|
| 当前 checkpoint | 0.25 |
| 最近历史 checkpoint | 0.25 |
| 当前最强 Heuristic | 0.20 |
| 历史 Heuristic | 0.10 |
| 随机 exploiter | 0.10 |
| 激进吃子/前进 exploiter | 0.10 |

每局随机指定 learner 执红或执黑。当前网络动作使用 outcome-weighted behaviour
learning：胜局动作权重 2，和局 1，负局 0；所有状态仍学习真实终局 WDL。
Heuristic 回合保留较低权重的宽 policy 标签。历史/随机/exploiter 回合只供
value 学习，不要求当前 policy 模仿。

这不是朴素 latest-vs-latest PPO；历史和规则型人口降低策略循环与遗忘风险。

## 7. 训练目标

对 batch 中第 n 个状态：

\[
L_{\text{policy}}
=-\frac{\sum_n w_n\sum_a y_{n,a}\log\pi_\theta(a|s_n)}
{\max(1,\sum_n w_n)}
\]

\[
L_{\text{value}}
=-\frac1N\sum_n\log P_\theta(z_n|s_n),
\quad z_n\in\{L,D,W\}.
\]

冻结每轮训练前的网络为 anchor，合法着均匀分布为 `u`：

\[
\rho=(1-\alpha)\pi_{\text{anchor}}+\alpha u,\quad \alpha=0.10,
\]

\[
L=L_{\text{policy}}+\lambda_v L_{\text{value}}
-\lambda_H H(\pi_\theta)
+\lambda_{KL}D_{KL}(\pi_\theta\Vert\rho).
\]

默认 `lambda_v=1`。在 200,000 optimizer steps 内：

- learning rate 余弦从 `2e-4` 降到 `1e-5`；
- entropy 系数从 `0.02` 降到 `0.002`；
- KL 系数从 `0.05` 降到 `0.005`；
- teacher replay 比例从 `0.70` 降到 `0.10`。

这使前期保留探索和 teacher 基础，后期逐渐减少正则与示范依赖。

## 8. replay、checkpoint、事务与恢复

teacher 和 self-play 使用两个独立的 crash-safe NPZ shard replay：

- observation 为 `float16`，因为最近动作平面含 `-1`，不能用旧的 uint8；
- policy target 在磁盘上稀疏保存；
- 默认总容量各 500,000 positions；
- 每条样本有 game key、ply 和 repetition context；
- shard 写入采用临时文件 + 原子替换。

### 8.1 三类 checkpoint

`latest.pt` 每个完整 iteration 原子覆盖一次，是唯一默认连续训练恢复点。它保存：

- model、每轮 anchor；
- optimizer；
- iteration、global step、累计局数；
- 完整配置快照；
- Python、NumPy、Torch、CUDA RNG；
- 两个 replay 的索引/RNG 状态。

另外两类 checkpoint 不改变训练起点：

- `checkpoints/champion.pt`：只有正式评测通过才更新；只作为稳定评测基准和
  人口对手，不是默认恢复点；
- `checkpoints/archive/iteration_NNNNNN.pt`：每 5 轮永久保存，不按棋力删除；
- `checkpoints/recent/iteration_NNNNNN.pt`：始终只保留最近 5 个完整 iteration，
  用于人工故障回滚；
- champion 的来源 checkpoint 会被放入 archive，因此 recent 清理永远不会
  删除 champion 来源。

初始 champion 是 iteration 10。若恢复的是已越过 iteration 10 的旧 lineage，
但 iteration 10 不可用，才使用最近一个完整的 5 倍数 checkpoint。新 run 在
iteration 1–9 恢复时不会提前建立 champion。

### 8.2 iteration 事务边界

每轮严格按以下顺序执行：

```text
写 iteration_transaction.json
→ self-play：只有完整结束的游戏才 append
→ 全部游戏完成并 flush replay
→ train
→ 原子保存 completed latest.pt
→ 注册 recent/archive
→ 提交 replay 事务并清除事务文件
→ champion evaluation / cross-play（只读）
```

事务文件记录 base/target iteration、phase、已完成游戏数、已提交样本数、两个
replay 的 shard/next-id 状态。容量裁剪涉及的旧 shard 在 iteration 提交前只做
延迟删除，因此中断可以精确恢复旧索引。未完成游戏只存在于内存，永不写 replay；
未提交的新 shard 会被删除。game key 从完整 checkpoint 的累计局数继续，因此
恢复后不会重复提交同一个 key。

若程序在 `latest.pt` 原子替换后、事务清理前退出，下一次启动会识别这个完整
checkpoint，完成延迟清理而不会重跑该轮。若在此之前退出，则回到 base
checkpoint，并恢复 model、optimizer、anchor、global step、replay 和 RNG。

相同配置重新执行：

```powershell
.\run_fixed_layout_rl.ps1 -Iterations 100
```

会自动从最后一个完整 `latest.pt` 恢复。Ctrl+C 不保存半轮状态：它丢弃当前
iteration 的未提交数据并回到上一轮完整 checkpoint。

训练和晋升互相独立。评测拒绝 candidate 时，candidate 仍是 `latest`，下一轮
继续在它之上训练；系统不会自动回滚到 champion。checkpoint 损坏、NaN、entropy
坍缩、KL 爆炸或经充分独立评测确认的灾难性退化，只发出证据供人工决定回滚。

若改变任何训练配置，必须二选一：

1. 同时在每次恢复命令中传入完全相同的覆盖参数；
2. 推荐给实验使用新的 `run_directory`，避免把不同实验混进同一 lineage。

例如：

```powershell
.\run_fixed_layout_rl.ps1 `
  -RunDirectory runs\fixed_layout_lr_test `
  -Iterations 20 `
  -GamesPerIteration 2048 `
  -ParallelGames 256 `
  -TrainSteps 300 `
  -BatchSize 1024
```

## 9. 修改配置

唯一正式配置是 `config/fixed_layout_rl_v1.json`。

常改字段：

| JSON 路径 | 默认 | 作用 |
|---|---:|---|
| `teacher.bootstrap_games` | 512 | 首次 teacher 对局数 |
| `teacher.bootstrap_train_steps` | 400 | 首次模仿更新数 |
| `teacher.workers` | 16 | 084/082 搜索工作进程数 |
| `teacher.softmax_temperatures` | 0.7/1/1.5 | teacher policy 宽度 |
| `selfplay.games_per_iteration` | 1024 | 每轮人口自博弈局数 |
| `selfplay.parallel_games` | 256 | 同时推进、一起 GPU 推理的局数 |
| `selfplay.opponent_weights.*` | 见上表 | 人口组成 |
| `train.batch_size` | 1024 | optimizer batch |
| `train.steps_per_iteration` | 200 | 每轮 optimizer steps |
| `train.learning_rate` | 2e-4 | 初始学习率 |
| `train.value_weight` | 1.0 | WDL loss 权重 |
| `train.entropy_start/end` | .02/.002 | 探索正则 |
| `train.kl_start/end` | .05/.005 | 相对每轮 anchor 的信赖域 |
| `train.teacher_fraction_start/end` | .70/.10 | 示范占比退火 |
| `checkpoint.archive_interval` | 5 | 永久 archive 的 iteration 间隔 |
| `checkpoint.recent_keep` | 5 | 最近完整 checkpoint 保留数 |
| `checkpoint.initial_champion_iteration` | 10 | 初始 champion iteration |
| `evaluation.interval` | 5 | 正式 champion evaluation 间隔 |
| `evaluation.initial_games` | 512 | 第一阶段评测局数 |
| `evaluation.extended_games` | 2048 | 模糊区间扩展后的总局数 |
| `evaluation.reject_score` | .52 | 第一阶段低于此值拒绝 |
| `evaluation.direct_promote_score` | .55 | 第一阶段高于此值直接晋升 |
| `evaluation.temperature` | .25 | 双方相同评测温度 |
| `evaluation.mode` | sample | `sample` 或 `greedy` |
| `evaluation.seed` | 2026072309 | 固定成对 seed 列表的根 seed |
| `evaluation.crossplay_interval` | 10 | 完整 cross-play 报告间隔 |
| `evaluation.crossplay_games` | 128 | 每组 cross-play 局数 |

命令行覆盖仅用于局数、并行数、step、batch 和 run 目录；其他变化应复制 JSON，
修改后使用新 run 目录，方便审批和复现。

## 10. 扩展到多个布局

当前 `config/layouts/fixed_layout_v1.json` 的 schema 已是 `layouts[]`。
扩展顺序应是：

1. 从 zip 读取另一个布局；
2. 转换方向并通过 `validate_layout`；
3. 加入稳定 ID、中文名、来源和哈希；
4. 新增 deterministic sampler；
5. 每局仍让双方使用抽中的同一个 canonical 布局；
6. teacher/self-play 和独立评测使用不重叠的布局 ID/seed 集。

不要简单令红黑分别随机抽布局，否则布局强弱会重新污染 policy/value 信号。

## 11. 指标解释

训练日志：

- `policy`：加权 policy 交叉熵，越低越贴近监督分布；
- `target_entropy`：监督分布自身熵，是理论可达到的 CE 下界；
- `excess = policy - target_entropy`：接近 0 表示拟合充分；
- `wdl`：WDL 三分类交叉熵；均匀初始基线约 `ln(3)=1.099`；
- `value_mse`：`P(W)-P(L)` 与实际 `-1/0/+1` 的均方误差；
- `entropy`：当前合法着 policy 熵，过早趋近 0 表示坍缩；
- `KL`：当前 policy 到 `rho` 的平均前向 KL；
- `teacher`：本 step 的 teacher replay 比例；
- `gradient_norm`：裁剪前全局梯度范数。

生成日志：

- `moves_per_second`：端到端含 C++ 状态推进、对手策略、GPU 推理和样本构造；
- `parallel_games`：并发推进局数；
- `temperature`：当前网络采样温度；
- `samples`：写入 replay 的 position 数。

每轮另有七条 `kind=selfplay_opponent`，分别对应：
`current/latest`、`recent_checkpoint`、`champion`、`084`、`082`、
`random_exploiter`、`aggressive_exploiter`。每条均包含 games、W/D/L、
score rate、average ply 和 moves/s；`kind=selfplay` 保留总计。

`kind=selfplay_terminal_reasons` 把终局分成 `capture_flag`、
`no_legal_move`、`repetition_draw`、`no_capture_draw`、
`max_ply_cutoff`、`other`，输出数量和比例。重复和棋或 max-ply 比例超过配置
阈值只会打印 warning，不会停止或回滚训练。

重复率诊断分级为 `<15%` 正常、`15%–30%` 需按对手检查、`30%–50%` 可疑、
`>=50%` 很可能策略退化。每个对手行也输出独立终局分布与等级。每个重复终局
另写 `selfplay_repetition_cycle`，包括 cycle 起点/长度、首次重复方、position
hash 序列、cycle-start value、terminal value、对手类型以及逐步 action/from/to/
棋子类型。完整字段和当前诊断见
`docs/REPETITION_AND_GENERALIZATION_DIAGNOSTICS_zh-CN.md`。

## 12. 独立评测与 champion 晋升

### 12.1 手工只读评测

```powershell
# A/B checkpoint
.\evaluate_fixed_layout.ps1 `
  -A iteration:15 -B iteration:10 `
  -Games 512 -Mode sample -Temperature 0.25 -Seed 2026072309

# checkpoint 对 084/082/random/aggressive
.\evaluate_fixed_layout.ps1 -A latest -B 084 -Games 512 -Mode greedy

# 明确输出路径和布局 ID
.\evaluate_fixed_layout.ps1 `
  -A latest -B champion -Games 2048 `
  -LayoutId public_conservative_advance `
  -OutputPrefix runs\fixed_layout_rl_v1\evaluations\manual_check
```

零训练不同布局评测可另外指定只读 layout set，并让红黑采用不同布局：

```powershell
.\evaluate_fixed_layout.ps1 `
  -A iteration:165 -B 084 -Games 256 `
  -LayoutSet config\layouts\fixed_layout_evaluation_v1.json `
  -RedLayoutId public_conservative_advance `
  -BlackLayoutId public_tiger_descent
```

`A`、`B` 可写绝对 checkpoint 路径，也可写 `latest`、`champion`、
`iteration:N`、`084`、`082`、`random`、`aggressive`。局数必须是偶数；
每个 seed 打两局并交换 A 的红黑方。两边使用同一布局、同一温度和同一决策模式。
输出为逐局 JSON 与带 BOM 的 UTF-8 CSV。

评测器只加载 checkpoint 的 model 权重，不实例化或修改 optimizer，不读取或写入
replay，不推进训练 global step。调用前后的 Python、NumPy、Torch 和 CUDA RNG
完全恢复，因此评测插入训练过程不会改变后续 self-play 序列。

### 12.2 每 5 轮正式评测

候选是当前 `latest`，基线是当前 `champion`：

1. 固定布局、固定 seed 列表、交换红黑、双方同温度，先打 512 局；
2. score rate `< 0.52`：不晋升；
3. score rate `> 0.55`：直接晋升；
4. `[0.52, 0.55]`：用不重复的后续 paired seeds 补到总计 2048 局；
5. 扩展后仅当 95% CI 下界 `> 0.50` 时晋升。

每局分数 \(x_i\) 为胜 `1`、和 `0.5`、负 `0`：

\[
\bar x=\frac1n\sum_i x_i,\qquad
SE=\frac{s(x)}{\sqrt n},\qquad
CI_{95\%}=[\bar x-1.96SE,\bar x+1.96SE]\cap[0,1].
\]

这里直接用实际得分的样本标准差，和棋不会被忽略或当成二项失败。评测 JSON/CSV
位于 `evaluations/champion_iteration_NNNNNN.*`。`champion.json` 记录来源
checkpoint/iteration、晋升时间、W/D/L、score、局数、CI、布局 ID/hash、
seed hash、温度、配置 hash、上一个 champion 和完整评测/晋升历史。

每 10 轮（可改为 20）额外生成 latest、champion、084、082、random、
aggressive 的完整两两 cross-play，写入 `crossplay/`。Archive 不做单败淘汰，
也不因一次平均分低而删除；后续可以在这些永久样本上加入克制性/策略差异筛选。

## 13. 当前性能基线

2026-07-23、当前机器、CUDA、6.90M 网络，使用正式脚本（100,000 次转移、
每个 batch 重复 20 次、256 局 rollout）的测量：

- C++ 随机合法转移：约 97,418 transitions/s；
- GPU batch 64：约 9,918 states/s；
- GPU batch 256：约 11,464 states/s；
- GPU batch 1024：约 9,417 states/s；
- 256 路纯网络无搜索 rollout：约 5,775 moves/s，平均推理 batch 217.0；
- batch 1024 联合训练：约 2.54 optimizer steps/s，即 2,601 samples/s；
- 完整 benchmark 峰值显存约 2,873 MiB，8 GiB GPU 有充分余量；
- iteration 13–15 的 256 路正式混合人口 self-play：约
  1,098–1,154 moves/s，每 1024 局约 61–65 秒；
- 旧逐局 teacher：约 6–12 moves/s；
- 16 worker teacher：32 局实测约 289 moves/s。

正式 benchmark 必须使用：

```powershell
.\benchmark_fixed_layout_rl.ps1
```

它报告 C++ transitions/s、各 GPU batch 的 states/s、无搜索 rollout moves/s、
平均 inference batch、optimizer steps/s 和峰值显存。混合人口训练速度会受
084/082 搜索比例影响，不能拿纯网络 rollout 数字冒充完整训练速度。按当前
测量，iteration 13–15 的 200 个训练 step（含 checkpoint）约 152–167 秒，
完整一轮约 213–232 秒，即约 3.6–3.9 分钟。固定布局、无搜索的 512 局
champion evaluation 约 4 秒、约 6,936 moves/s。

## 14. 当前执行结果与阶段性验收

当前已从 iteration 12 的完整 checkpoint 事务恢复，丢弃旧 iteration 13 的
部分 self-play，连续完成 iteration 13–15：

- latest：iteration 15，global step 3400；
- archive：iteration 5、10、15；
- recent：iteration 11–15；
- champion：iteration 10；
- iteration 15 正式 512 局评测：W/D/L=`265/2/245`，
  score=`0.51953125`，95% CI=`[0.4763, 0.5628]`；
- 因第一阶段 score 低于 52%，未晋升、未扩展到 2048 局；训练 lineage
  仍停在 latest iteration 15，后续从这里继续，不回滚。

第一阶段至少检查：

1. teacher policy excess 持续下降；
2. WDL loss 低于 1.099，且 held-out/独立对局校准随后补齐；
3. policy entropy 不突然坍缩；
4. 人口自博弈结果不只剩重复和棋；
5. archive/champion 独立对局显示长期趋势，而非只看训练 loss；
6. 每 5 轮 gate 只决定 champion，不阻断连续 learner 的小幅积累。
