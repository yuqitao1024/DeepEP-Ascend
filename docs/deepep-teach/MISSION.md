# Mission: DeepEP v2 and Ascend 950 EP optimization

## Why
帮助参与 DeepEP-Ascend 的工程师建立同一套 MoE、EP、DeepEP v2、NCCL Gin、HCOMM 与 Ascend 950 实现模型。材料用于团队串讲，也用于把后续性能工作拆成边界清楚、能够独立验收的任务。

## Success looks like
- 能画出一个 MoE 层中 router、dispatch、专家计算和 combine 的数据流，并说明 DeepEP 的边界。
- 能解释 H800 上 direct/hybrid 两条 EPv2 路径如何使用 NVLink、RDMA、NCCL Gin、SM 和 QP。
- 能解释 Ascend 950 为什么采用 SIMT 命令生产者加 AICore 通信服务，而不是直接照搬 Gin 调用。
- 能按优先级领取 8 个白盒优化项，写出正确性约束、性能指标和 NPU8P 验收步骤。
- 能在不绕过设备和队列策略的前提下提交、查看和诊断 NPU8P 任务。

## Constraints
- 代码证据以本仓库 `main` 提交 `2c434b3` 为基线。
- 上游 EPv2 证据以 `upstream/epv2-release` 提交 `018f21e` 为基线。
- 性能数字必须标出实测、公开参考或估算，不能把不同工作负载的结果当作正式比值。
- 当前 NPU8P 策略只允许设备 `0,1` 或 `6,7`，单任务最多两卡，并且同一提交者最多一个活动任务。

## Out of scope
- 把 DeepEP 讲成专家 GEMM、完整推理引擎或模型路由算法。
- 宣称 Ascend 950 已通过物理多机 RoCE 性能验收。
- 给出未经当前基准工具生成的 H800/Ascend 精确性能比。
