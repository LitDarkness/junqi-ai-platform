# Policy/Value 与原生搜索实验

本目录保留 `junqi_az`、`junqi_fixed_rl`、`junqi_adv_rl`、`junqi_search_v3` 与 `junqi_open_v4` 的实现和历史入口。请从仓库根目录的 `tools/build_native.py` 构建，不依赖本机历史 `.pyd` 文件。

核心模块：

- `cpp/az_state.*`：规则适配、107 平面编码、完整终局和历史状态。
- `cpp/neural_puct.*`、`cpp/root_explore/`：普通 PUCT 与受控根探索。
- `python/junqi_az/model.py`：成对策略头、WDL 与辅助输出。
- `python/junqi_open_v4/learner.py`：完整检查点、分阶段 Policy/Value 更新。
- `python/junqi_open_v4/isolated_teachers.py`：真实教师进程身份核验。
- `python/junqi_open_v4/arena_journal.py`、`resumable_arena.py`：精确绑定、不可变分片及比赛恢复。
- `tools/run_open_v4_recipe50_to800.py` 等：历史有限范围训练和正式保护契约，需相应的未发布状态。

## 小规模新实验

从仓库根目录构建后，在本目录执行：

```powershell
$env:PYTHONPATH="$PWD/build;$PWD/python;$PWD/tools"
python tools/run_open_v4.py --config config/examples/fresh_smoke.json --through-iteration 1
```

示例采用 CPU、小模型、2 个终局和 1 次更新，只验证闭环；初始权重为随机状态，教师为编译后的真实 082，使用新生成的布局。它不是 738 代训练配方，也不产生棋力证明。修改预算或方法应使用新的配置和运行目录。

`config/experiments/` 是历史实验配置快照，包含源文件 SHA 与旧检查点引用。发布时做过路径可移植化处理，其摘要不能当作原机器的冻结契约；精确续训必须重新核对原始私有档案。整个源码包不提供训练模型，也不自动启动后台任务。

原文档与方法沿革保存在 `docs/`；涉及未发布证据的链接属于历史记录。最新评测摘要见 `../docs/RESEARCH_STATUS.md`。
