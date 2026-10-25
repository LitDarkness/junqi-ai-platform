# Single-layout Suite Continuation Arena

## 1. 实验定义

本实验使用 `single_layout_strength_suite` 的全部 1024 个状态。每个状态都从
固定布局和完整 action trace 无损重建，并保留：

- 当前棋盘与行动方；
- 完整 repetition history；
- no-capture、turn、repetition counters；
- 原规则终局条件。

每个 matchup 对同一状态运行两局：

1. 主测 agent 控制红方；
2. 主测 agent 控制黑方。

两局使用相同 suite state、setup seed、历史和规则上下文。所有 agent 均为
greedy，关闭采样和 Dirichlet noise。因此每个 matchup 为：

```text
1024 paired states × 2 role-swapped games = 2048 games
```

统计同时给出：

- 单局 W/D/L；
- 每个 state 换边两局的 paired-state bootstrap；
- 以 409 个 trajectory family 为单位的 family bootstrap。

不能把同一状态的换边两局或同一 family 的多个状态当成完全独立样本。

## 2. 冻结对象

网络为 `open_junqi_search_v3_alternating` iteration 10 正式 incumbent。
iteration 10 选择 epoch 0，因此权重与最后接受更新的 iteration 8 相同。

冻结对手为 exact082，源码 SHA-256 已写入 continuation contract。比较：

```text
bare vs exact082
Search64 vs exact082
Search64 vs bare
```

Search64 与 bare 使用完全相同的网络权重；唯一差别是是否执行无噪声的
64-simulation Neural PUCT。

## 3. 完整结果

| Matchup（前者视角） | W/D/L | 得分率 | family 95% CI |
|---|---:|---:|---:|
| bare vs exact082 | 225/86/1737 | 0.1309 | [0.1074, 0.1402] |
| Search64 vs exact082 | 329/88/1631 | 0.1821 | [0.1615, 0.1983] |
| Search64 vs bare | 1242/231/575 | 0.6628 | [0.6414, 0.6821] |

以完全相同的 exact082、suite state、角色和 seed 做 paired comparison：

```text
Search64 score - bare score = +0.05127
family mean delta            = +0.05666
family 95% CI                = [+0.04302, +0.07073]
```

区间完全高于 0，说明 Search64 对真实终局结果的改善不是换边误差或重复样本造成。

## 4. 红黑角色拆分

| Matchup | 主测为红方 | 主测为黑方 |
|---|---:|---:|
| bare vs exact082 | 0.1538 | 0.1079 |
| Search64 vs exact082 | 0.2183 | 0.1460 |
| Search64 vs bare | 0.6763 | 0.6494 |

Search64 在红黑两侧都优于 bare；直接 Search64 vs bare 的优势没有依赖单一角色。

## 5. Suite 分类结果

| 分类 | bare vs exact082 | Search64 vs exact082 | Search64 vs bare |
|---|---:|---:|---:|
| 开局 | 0.0969 | 0.1547 | 0.6672 |
| 中局 | 0.1180 | 0.1641 | 0.6977 |
| 残局 | 0.2484 | 0.2984 | 0.6328 |
| 战术 | 0.0961 | 0.1492 | 0.6398 |
| 循环入口 | 0.1602 | 0.2148 | 0.6680 |

Search64 在五个分类的最终对局得分上都优于 bare。

这与静态 32k Oracle regret 的开局结果不同：静态 Oracle 曾认为 Search64
开局动作更差，但完整续弈显示 Search64 开局对 exact082 和对 bare 都更强。
因此当前 Oracle action-regret 不能单独替代真实终局对局 gate；它更适合作为
局部诊断。

## 6. 终局与速度

| Matchup | 吃旗 | 重复 | 无合法着 | 平均续弈 ply |
|---|---:|---:|---:|---:|
| bare vs exact082 | 1871 | 86 | 91 | 21.66 |
| Search64 vs exact082 | 1904 | 88 | 56 | 25.20 |
| Search64 vs bare | 1742 | 231 | 75 | 23.33 |

Search64 vs bare 的重复率为 11.28%，没有达到此前使用的 15% 工程警戒线。

累计 Search64 吞吐：

- 对 exact082：4656 simulations/s；
- 对 bare：3985 simulations/s。

大批量活跃阶段通常约 7.5k–8.4k simulations/s；少数超过 100–250 ply 的尾局
会使 batch 缩小，从而拉低累计值。

## 7. 运行和恢复

完整命令：

```powershell
cd .
.\run_search_v3_continuation_arena.ps1 `
  -States 1024 `
  -BlockStates 128 `
  -Workers 16
```

每 128 个 suite states、256 局形成一个完整事务。中断后会复用已完成事务。

也支持将冻结对手替换为指定预算的 supplement Oracle，例如：

```powershell
.\run_search_v3_continuation_arena.ps1 `
  -States 1024 `
  -Matchups "bare_vs_oracle_2000,search64_vs_oracle_2000,search64_vs_bare"
```

最终报告：

```text
runs/open_junqi_search_v3_strength/continuation_arena/report.json
```

每个 matchup 的 `summary.json`、逐局 `games.csv` 和事务文件位于：

```text
runs/open_junqi_search_v3_strength/continuation_arena/states_1024
```
