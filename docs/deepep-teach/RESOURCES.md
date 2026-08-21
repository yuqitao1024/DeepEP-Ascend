# DeepEP v2 and Ascend 950 resources

## 主要资料

- [DeepEP upstream README](https://github.com/deepseek-ai/DeepEP/tree/epv2-release)
  EPv2 的功能边界、公开性能、安装依赖和 `ElasticBuffer` 接口。用于核对 V1/V2 差异、SM/QP 策略和公开基准。
- [DeepEP v1 archived documentation](../legacy.md)
  明确标注 H800、CX7、NVLink/RDMA 的 V1 性能和 normal/low-latency 设计。用于提供 H800 量级参考，不能替代 EPv2 同口径报告。
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
  DeepSeekMoE、node-limited routing、无 token 丢弃、DualPipe、跨节点 all-to-all 和推理部署背景。
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
  细粒度专家、共享专家隔离和 device-limited routing 的模型背景。
- [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066)
  DeepSeekMoE 架构的原始论文。用于理解细粒度专家和共享专家为何改变通信形态。
- [NCCL GIN and Symmetric Memory](https://github.com/NVIDIA/nccl/blob/master/docs/contrib/GIN/NCCL_Gin_and_Symmetric_Memory/NCCL_Gin_and_Symmetric_Memory.md)
  NCCL Device API、LSA、GIN、对称内存和窗口注册的社区技术说明。
- [NCCL Device API AllToAll with GIN](https://github.com/NVIDIA/nccl/tree/master/docs/examples/06_device_api/02_alltoall_gin)
  纯 GIN 多机 all-to-all 示例。用于理解 GPU 发起的 put、signal、flush 与 barrier。
- [SGLang expert parallelism](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/expert_parallelism.mdx)
  社区推理框架把 MoE 前向拆成 router、dispatch、permute、grouped GEMM 和 combine 的实际集成方式。
- [vLLM expert parallel deployment](https://github.com/vllm-project/vllm/blob/main/docs/serving/expert_parallel_deployment.md)
  推理服务中的 DP/EP 部署、DeepEP all-to-all backend 和负载均衡边界。
- [Ascend EPv2 transport contract](../ascend-design/epv2-ascend-transport-contract.md)
  本仓后端中立传输契约、能力位和失败边界。
- [Ascend SIMT transport facade](../ascend-design/epv2-ascend-simt-transport-stub.md)
  NCCL Gin 到 Ascend facade 的语义映射，以及 CANN AIN 的公开能力和限制。
- [Ascend SIMT to AICore URMA transport](../ascend-design/epv2-ascend-simt-urma-transport.md)
  `__simt_vf__` 无法直接敲 URMA doorbell 后采用的命令缓冲区和 AICore 服务方案。
- [Ascend production core operators](../ascend-design/epv2-ascend-production-core-operators.md)
  barrier、BF16 dispatch/combine、资源所有权、窗口布局、generation 和 teardown 的生产路径。
- [Ascend EPv2 roadmap](../ascend-design/epv2-ascend-roadmap.md)
  单机扩展、物理多机 RoCE、hybrid、异步、FP8 和性能成熟度的已完成与未完成边界。
- [Ascend benchmark parity](../ascend-design/epv2-ascend-benchmark-parity.md)
  144 个 case、720 个 operation、同 manifest 比较规则和 NPU8P 运行证据。
- [Ascend performance optimization](../ascend-design/epv2-ascend-performance-optimization.md)
  当前基线、8 个白盒优化项、优先级、验收信号和多 kernel 流水设计。
- [Dispatch SIMT parallelization](../ascend-design/dispatch-simt-parallelization.md)
  当前单 block 内并行化的实现边界和后续多 AI Vector 优化的前置证据。

## 社区资料

- [DeepEP GitHub issues and pull requests](https://github.com/deepseek-ai/DeepEP/issues)
  用于核对上游行为、硬件兼容性和性能回归。提交问题时附 commit、拓扑、shape、日志和可复现命令。
- [SGLang large-scale EP discussion](https://lmsys.org/blog/2025-05-05-large-scale-ep/)
  推理服务中 overlap、EPLB 和大规模 EP 的工程经验。其配置和性能不自动适用于本仓 Ascend 后端。

## 证据缺口

- 尚无由本仓当前 144-case 自动化生成、且 workload fingerprint 相同的 H800 `benchmark.json`。
- NPU8P 只提供单主机环境，物理多机 RoCE、NIC 选择和链路故障语义仍需真实多机环境验证。
- CANN 9.2.0 的公开 SIMT 设备通信接口不能完整覆盖 Gin 语义；当前实现依赖本仓经过 ABI 约束的 staged transport。
