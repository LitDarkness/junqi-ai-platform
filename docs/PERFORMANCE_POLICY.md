# Heuristic 单核性能原则

## 硬性门槛

从本版本开始，任何以 `heuristic`、`heuristic_vN` 或等价生产名称暴露的启发式算法，必须在单个逻辑 CPU 上达到至少 **100 steps/s**。

同一门禁还要求裸 C++ 后端在单个逻辑 CPU 上达到至少 **50,000 steps/s**。该后端项目包含直接 movegen、选取首个合法动作、规则校验、状态更新、终局检查及必要的重置，但不包含 Heuristic 评分。

这里的一个 step 是一个完整 ply，计时范围包含：

1. C++ 直接生成合法着；
2. 从 C++ 取得当前棋盘和双方物质分；
3. Heuristic 选择一个动作；
4. C++ 校验并执行该动作；
5. 必要时重置已经结束的对局。

因此不能只报告纯评分函数速度，也不能排除 movegen、Python/C++ 边界或状态更新时间。

## 标准门禁

```powershell
$env:PYTHONPATH="$PWD;$PWD/build"
python performance_gate.py
```

脚本将进程固定到一个当前允许的逻辑 CPU，预热后重复测量三次，以中位数为准。裸 C++ 后端或任一生产 Heuristic 低于对应门槛，进程以非零状态退出。

建议开发目标不低于 120 steps/s，为操作系统抖动和较慢机器保留余量。正式报告必须记录 CPU、编译类型、Python 版本、测量步数和各次样本。

## 注册规则

- 新生产 Heuristic 必须加入 `performance_gate.py` 的 `PRODUCTION_HEURISTICS`。
- 新算法的规则、战术或棋力测试与性能门禁必须同时通过。
- 低于门槛的搜索实验只能使用 `experimental` 或 `research` 名称，不能成为 GUI、训练或评测的默认 `heuristic`。
- 不允许通过减少规则校验、使用非法着 fast path 或改变终局语义换取速度。
- 性能优化后必须重新运行规则回归、Heuristic 行为测试和最小 RL 闭环。

## 当前命名

- `heuristic`：生产默认值，映射到通过门禁的 `heuristic_v3`。
- `heuristic_v1`：生产快速战术基线。
- `heuristic_v2`：生产战略基线，使用固定 12 候选预算。
- `heuristic_v3`：生产战术基线；在 v2 的候选上用 C++ 精确合法着检查对手最强一手，固定 6 候选预算；对手应手权重为 0.55（由固定开发集选择，并经独立 64 布局与新随机对局验证）。
- `lookahead_experimental`：多层前瞻研究算法，不属于生产 Heuristic，不得作为默认配置。
