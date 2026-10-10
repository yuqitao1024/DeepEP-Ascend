# 当前 main 的 Official SIMT 复测

日期：2026-10-10。基线代码为 main `d78a57e`，远端实验副本额外包含一处
16K token 用例所需的 remote-only workspace 对齐修改：runtime workspace 从
4MiB 提升到 32MiB。该修改只影响容量校验，不改变 kernel 的数据读写路径。

## 结论

当前 main 上，experimental Official SIMT 执行器仍无性能收益。五个操作的平均
延迟都比默认 AICore/URMA 执行器慢约 0.19% 到 0.39%，差异方向一致但幅度很小。
因此 `DEEP_EP_ASCEND_OFFICIAL_SIMT` 继续保持默认关闭。

本轮结果与 2026-09-26 旧基线的结论一致：单纯把 staged command queue 的执行
后端替换为 HCOMM SIMT，不改变 producer/command queue/service boundary，不能
得到稳定收益。

## 测试环境

- 主机：直连 NPU8P；
- 设备：0–7；
- CANN：/usr/local/Ascend/cann-9.3.0；
- Python：/data/disk2/pyptouser/yuqitao/deepep-venv-py311/bin/python；
- AICore/URMA 二进制 SHA256：
  `5a0c8d54caca8b10e36a4a37069356ba33433787adcaf77ff272787fd77af637`；
- Official SIMT 二进制 SHA256：
  `5f9e713f9db8f633d21c6e1d7b4f68d470b68fb93495d09da8a22708e07315a2`。

## 协议

- case：`ep-fp8-align128-bias0-hcopy0-prev0-async1-alloc0`；
- 16384 tokens、hidden 7168、top-k 6、256 experts、EP8；
- `num_sms=64`；
- 10 warmup、30 iteration；
- A/B 顺序：B1、A1、A2、B2；
- 每个操作取 8-rank 最大延迟，每侧 A/B 合并两次运行共 60 个样本；
- 逻辑带宽为 8-rank 聚合口径。

## 结果

| 操作 | AICore/URMA ms | Official SIMT ms | Official SIMT 相对变化 |
|---|---:|---:|---:|
| Dispatch | 22.971089 | 23.061339 | -0.39% |
| Expanded Dispatch | 104.046651 | 104.305081 | -0.25% |
| Cached Dispatch | 18.652391 | 18.704841 | -0.28% |
| Combine | 13.178433 | 13.203214 | -0.19% |
| Reduced Combine | 12.618431 | 12.655182 | -0.29% |

## 任务与数据

| 项 | 任务 |
|---|---|
| B1 | `task_20261010_080959_266699923867` |
| A1 | `task_20261010_081042_267108626504` |
| A2 | `task_20261010_081126_267632222143` |
| B2 | `task_20261010_081749_272867917931` |

原始 JSON 与汇总文件在
`tests/ascend/benchmark/results/official_simt_current/`。

## 优化判断

本轮不能得出“HCOMM SIMT API 上限不佳”的结论。当前 B 分支仍保留：

- GM `TransportCommandQueue`；
- AICore service shell 的 command/channel 预检；
- service 到 SIMT VF 的 `asc_vf_call` 边界；
- 单 lane 顺序消费 command；
- 每次 invocation 独占所有 channel 并 immediate drain。

真正可能改变结论的优化方向是重新设计提交模型，而不是继续微调当前 executor：

1. producer 直接调用 HCOMM SIMT API，消除 command queue；
2. 为每个 channel 指定唯一提交 lane，避免单 lane 消费全局命令批次；
3. 引入 deferred commit / last-commit 批量发布，减少 immediate WQE 发布开销；
4. 将 command contract 校验前移到 producer 或 host preflight；
5. 为 SIMT executor 补充 stage 级 profiling，明确提交和 drain 的比例。

这些是独立的架构实验，不能用本轮数据预测结果。

