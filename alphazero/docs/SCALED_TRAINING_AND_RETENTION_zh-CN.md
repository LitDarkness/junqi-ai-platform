# Search v3 稍大规模训练与保留策略

## 1. 新训练 lineage

长期训练不解冻 `open_junqi_search_v3_alternating`，也不修改只读的
`open_junqi_search_v3_strength`。

新的 successor：

```text
runs/open_junqi_search_v3_scaled_training
```

它从 strength iteration 10 incumbent 初始化，并恢复：

- model；
- 三参数组 optimizer；
- constant scheduler；
- Python、NumPy、Torch、CUDA、closed-loop 和 batch RNG；
- global step 805；
- iteration 9、10 Search replay；
- 原 Oracle anchor 和 immutable probe。

下一轮为 iteration 11。仍使用固定布局
`public_conservative_advance`，`mirror=false`，没有修改 PUCT、
Dirichlet、温度、网络或训练比例。

## 2. Checkpoint 保留策略

每个成功 iteration 完整保存 checkpoint 和 report 后，自动执行：

1. 最近 5 个 iteration 的正式 checkpoint 全部保留；
2. 更老 checkpoint 中，每 5 个 iteration 保留一个；
3. championship 当前版本和历史版本永久保留；
4. competitive checkpoint 永久保留；
5. candidate 最近 5 轮保留；较老 candidate 仅保留 championship 或
   competitive；
6. 删除前先把逐文件 dry-run 写入 `retention.json`，再执行删除并回写结果。

competitive 的当前工程定义：

- 是实际非 epoch-0 更新；
- Policy family 未退化；
- WDL family 未明确退化；
- immutable probe 上 Search64 regret 低于 bare；
- Value-phase Policy KL 不超过 `1e-3`。

championship 初始固定为经过完整 1024-state continuation suite 验证的
iteration 10。普通训练 gate 通过不会自动冒充 championship；后续只有独立的
paired continuation Arena 批准后才更新 championship。

## 3. 长期运行命令

从 iteration 10 训练到 iteration 30，即最多 20 个新 iteration：

```powershell
cd .

.\run_search_v3_scaled_training.ps1 -ThroughIteration 30
```

若希望约 40 个新 iteration：

```powershell
.\run_search_v3_scaled_training.ps1 -ThroughIteration 50
```

过去 iteration 6–10 平均约 13–14 分钟一轮。粗略估计：

- 到 iteration 30：约 4–5 小时；
- 到 iteration 50：约 9–10 小时。

具体时间取决于平均对局长度、尾部小 batch 和是否选择 epoch 0。

脚本包含：

- Windows system/away-mode 防休眠；
- UTF-8；
- crash-safe replay transaction；
- 已完成 iteration 自动跳过；
- 每轮自动 retention；
- gate 失败自动暂停；
- 完整控制台 transcript。

日志位于：

```text
runs/open_junqi_search_v3_scaled_training/console
```

控制台持续显示：

```text
[championship] incumbent=... challenger=...
[search-v3 loop] games=... selfplay-sps=... last-block-sps=... repetition=...
[throughput] samples=... unique-state=...
[alternating] epoch=... heldout=... wdl=... kl-max=...
[retention] recent=... archive=... competitive=... reclaimed=...
```

Ctrl+C 后，未完成游戏不会提交；再次运行同一命令会复用完整 replay
transaction，并从最近完整 checkpoint 继续。

## 4. 本次清理

执行清单：

```text
cleanup/cleanup_20260726_scaled.json
```

实际删除 232 个文件，共 2.389 GiB，主要包括：

- v2a 冻结失败线的 value rollout 与 teacher replay；
- 被取代的 asymmetric-value failure 模型与 Oracle 缓存；
- alternating 中不符合保留策略的两个重复 checkpoint；
- 两状态 continuation smoke、Python cache 和空测试目录内容。

明确保留：

- `fixed_layout_rl_v1` 全量失败基线；
- `fixed_layout_rl_v2a` 的失败报告、metrics、final/latest/champion 和原
  checkpoint retention 集；
- `fixed_layout_rl_v2b` 全量 bootstrap 状态和最终诊断；
- 原 Search v3 iteration-3 failure、A/B/C/D 和恢复证据；
- alternating iteration 10 及其必要历史；
- strength suite、静态评测、全部 2048 局 continuation Arena；
- 新 scaled-training lineage。
