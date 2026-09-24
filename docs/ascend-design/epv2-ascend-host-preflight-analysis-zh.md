# Ascend host preflight 瓶颈分析

日期：2026-09-24

## 结论摘要

本轮 profiling 确认，8-rank 典型 case 中每次 dispatch/combine 前的 Python
preflight 是 host 边界的主要耗时来源。它不是 pybind 参数转换，也不是本地
contract 编码和解码，而是 preflight 内部的分布式对象聚合
dist.all_gather_object。

在 8-rank、4096 tokens、hidden 7168、top-k 6、256 experts 的 FP8 case 中，
normal dispatch 的 Python preflight 平均约 14.225 ms，其中
dist.all_gather_object 约 13.937 ms；pybind/native 调用约 7.409 ms。
preflight 还造成各 rank 进入 C++ runtime 的时间错位，normal dispatch 的
C++ entry spread 约 7.166 ms。这个错位会转化为设备侧 acquire wait。

当前 benchmark 的 logical_gbps 继续使用原有公式：

logical_gbps = total_logical_bytes / device_seconds

其中 device_seconds 是 NPU start event 到 end event 的时间跨度。由于 start
event 在 Python wrapper 调用前记录，而 preflight 发生在 wrapper 内、kernel
启动前，因此当前 logical_gbps 是 API 端到端口径，已经包含 preflight 时间。
本轮只记录该影响，不修改统计公式。

## 什么是 preflight

preflight 是 DeepEP Ascend Python wrapper 在进入 C++ runtime 前做的安全
检查。它不是 DeepEP 数据面 kernel，而是进入数据面前的 host 侧一致性检查。

当前 dispatch/combine 的 Python wrapper 大致流程如下：

1. Python API 被调用；
2. 解包 payload，例如把 FP8 的 x 拆成 payload 和 scale factor；
3. 执行 Ascend preflight；
4. 解包 cached handle；
5. 调用 C++/pybind runtime；
6. C++ runtime 启动并等待 DeepEP kernel；
7. Python wrapper 组装返回值。

Ascend preflight 主要做三类事情：

1. 本地 tensor contract 检查：检查 dtype、shape、contiguous、device、
   rank、capacity、expert alignment 等；
2. handle 与 buffer 的一致性检查：检查 cached handle 是否属于当前
   ElasticBuffer、descriptor generation 是否一致、handle tensor 是否在同一
   NPU 上；
3. 跨 rank contract 聚合：把每个 rank 的本地 contract 通过
   dist.all_gather_object 汇总，确认所有 rank 的关键配置一致，避免某个
   rank 带着不一致配置进入 collective kernel 后挂死或产生难定位的错误。

问题集中在第 3 类：每次公开 dispatch/combine 都执行一次 Python 对象
collective。这个检查有正确性价值，但对固定 buffer、固定 process group、
固定 workload 模式来说，大多数 contract 字段不会逐次变化，因此它不应成为
热路径的固定成本。

## 环境与数据来源

- 节点：NPU8P，设备 0-7；
- 提交方式：task-submit；
- CANN/HCOMM：/data/disk2/cann_version/0916/use_cann/cann-9.3.0；
- workload：8 ranks，每 rank 4096 tokens，hidden 7168，top-k 6，256 experts；
- case：ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0；
- 参数：num_sms=56，warmups=1，iterations=1；
- 8-rank 任务：task_20260924_104007_183834114738；
- 8-rank 结果：
  /home/pyptouser/yuqitao/deepep-93-work/repo/results/host-pybind-boundary-split-8rank/benchmark.json；
- 4-rank 任务：task_20260924_102608_6149851538；
- 2-rank 验证任务：task_20260924_102336_55969220178。

诊断代码临时拆分了以下阶段：

- unpack_payload：解包输入 payload；
- preflight：整个 Python Ascend preflight；
- unpack_handle：解包 cached handle；
- pybind_pre：即将进入 C++/pybind 调用；
- pybind_post：C++/pybind 调用返回；
- wrapper_exit：Python wrapper 收尾完成；
- preflight encode：本地 contract 编码；
- preflight gather：dist.all_gather_object；
- preflight decode：解码并比较所有 rank 的 contract。

这些诊断字段仅用于定位，不属于当前生产路径设计。

## rank 数量扩展趋势

normal dispatch 的平均数据如下：

| world size | Python preflight | pybind/native | wrapper total |
| ---: | ---: | ---: | ---: |
| 2 | 2.886 ms | 1.242 ms | 4.487 ms |
| 4 | 6.640 ms | 3.794 ms | 10.823 ms |
| 8 | 14.225 ms | 7.409 ms | 22.389 ms |

可以看到 preflight 和 pybind/native 都近似随 rank 数增加而放大。8-rank 时
Python wrapper 内部总耗时约 22.389 ms，其中 preflight 占 14.225 ms。

normal dispatch 中各阶段平均耗时约为：

| 阶段 | 耗时 |
| --- | ---: |
| unpack payload | 0.004 ms |
| preflight | 14.225 ms |
| unpack handle | 0.002 ms |
| pybind/native | 7.409 ms |
| wrapper 收尾 | 0.749 ms |

因此，之前观察到的“Python 到 C++ 入口延迟”不是 pybind 参数转换，而是进入
pybind 之前的 Python preflight。

## 8-rank 操作级数据

| operation | reported mean | device event mean | preflight | preflight gather | pybind/native |
| --- | ---: | ---: | ---: | ---: | ---: |
| dispatch | 21.793 ms | 21.101 ms | 14.225 ms | 13.937 ms | 7.409 ms |
| expanded_dispatch | 29.517 ms | 28.923 ms | 13.899 ms | 13.637 ms | 15.026 ms |
| cached_dispatch | 50.095 ms | 49.631 ms | 14.200 ms | 11.813 ms | 34.205 ms |
| combine | 24.999 ms | 24.092 ms | 13.254 ms | 13.016 ms | 6.262 ms |
| reduced_combine | 27.903 ms | 27.254 ms | 13.290 ms | 13.032 ms | 15.338 ms |

normal dispatch 的 preflight 内部拆分为：

| preflight 阶段 | 耗时 |
| --- | ---: |
| encode | 0.051 ms |
| dist.all_gather_object | 13.937 ms |
| decode/compare | 0.127 ms |

因此瓶颈是每次调用的 Python 对象 collective，而不是：

- pybind 参数转换；
- Python payload 解包；
- cached handle 解包；
-本地 contract 编码；
-本地 decode/compare。

## 对 logical_gbps 的影响

当前计时逻辑在 operation 外部记录 NPU start event：

1. record start event；
2. 调用 Python wrapper；
3. wrapper 内执行 preflight；
4. preflight 完成后进入 C++ runtime；
5. DeepEP kernel 执行；
6. record end event。

因此 device_seconds 覆盖了 preflight 和 kernel 启动错位。以 normal dispatch
为例：

| 指标 | 数值 |
| --- | ---: |
| reported device span | 21.101 ms |
| preflight gather | 13.937 ms |
| 近似去掉 preflight 后的时间 | 7.164 ms |
| 当前 logical_gbps | 154.014 GB/s |
| 近似去掉 preflight 后的带宽 | 453.8 GB/s |

453.8 GB/s 只是诊断估算，不是新的正式报告。它说明当前 API 端到端口径会把
post-preflight 数据路径的带宽低估约 3 倍。后续如果需要 kernel-only 口径，
应另增指标，而不是修改现有 logical_gbps 语义。

## 对 kernel 启动时间的影响

preflight 不只是增加固定延迟，还会造成各 rank 进入 C++ runtime 的时间错位。
本轮 8-rank 数据中，C++ entry spread 与 preflight spread 如下：

| operation | C++ entry spread | preflight spread |
| --- | ---: | ---: |
| dispatch | 7.166 ms | 6.284 ms |
| expanded_dispatch | 6.786 ms | 6.311 ms |
| cached_dispatch | 6.001 ms | 4.257 ms |
| combine | 4.050 ms | 6.792 ms |
| reduced_combine | 7.524 ms | 5.407 ms |

normal dispatch 中，各 rank 的 acquire wait 约为：

| rank | acquire wait |
| ---: | ---: |
| 0 | 11.378 ms |
| 1 | 0.030 ms |
| 2 | 0.029 ms |
| 3 | 11.454 ms |
| 4 | 5.848 ms |
| 5 | 3.841 ms |
| 6 | 4.743 ms |
| 7 | 5.044 ms |

早进入设备路径的 rank 需要等待仍停留在 Python preflight 中的 rank，因此
preflight 的 rank skew 会转化为设备侧 acquire wait。这说明它同时影响：

1. 端到端 device event 时间；
2. logical_gbps；
3. kernel 内部 acquire wait；
4. rank tail 分析和 critical path 归因。

## 为什么当前实现会慢

preflight_ascend_contract 的核心流程是：

1. 每个 rank 构造本地 contract；
2. 将 contract 编码成 Python object；
3. 调用 dist.all_gather_object；
4. 解码所有 rank 的结果；
5. 比较所有 rank 的 contract 是否一致。

其中 dist.all_gather_object 使用 Python object 集合通信，涉及对象序列化、
反序列化和 Python 调度。8-rank 时它平均约 13-14 ms，而本地 encode/decode
只有约 0.18 ms。这说明成本几乎全部来自跨 rank Python object collective。

更重要的是，这个检查在每次公开 dispatch/combine 调用中执行。对固定
ElasticBuffer 和固定 workload 模式，contract 中的大部分字段是稳定的，例如：

- world size；
- topology；
- buffer bytes；
- dtype 和布局；
- expert alignment；
- capacity；
- hybrid/direct 模式；
- transport capability；
- descriptor schema。

逐次重复聚合这些稳定字段，收益有限，但成本明显。

## 优化目标与约束

目标不是删除所有校验，而是把正确性检查从热路径移到变化点：

1. 保持非法输入能被明确拒绝；
2. 保持跨 rank 配置不一致能被发现；
3. 避免每次调用都执行 Python object collective；
4. 不改变当前 benchmark 统计公式；
5. 不改变公开 API 语义；
6. 不把错误推迟到设备侧造成难诊断挂死。

## 候选方案

### 方案 A：contract fingerprint 缓存

为每个本地 contract 计算稳定 fingerprint。若 fingerprint 与上次成功校验的
值一致，则跳过 dist.all_gather_object，只执行本地检查。

优点：

- 改动集中；
- 热路径收益直接；
- 对固定 workload 的 benchmark 和训练循环最有效。

风险：

- 所有 rank 可能都认为本地未变，但彼此之间的配置不同；
- 需要一种低成本的跨 rank 变化通知机制。

### 方案 B：一次性校验 stable contract，热路径只做 local check

把 contract 分成 stable 和 dynamic 两类：

- stable：world size、topology、buffer bytes、transport capability、schema；
- dynamic：shape、dtype、capacity、alignment、event/stream 模式。

stable contract 在 buffer 构造或首次调用时全量聚合校验一次；热路径只做本地
dynamic contract 检查。

优点：

- 语义清晰；
- 热路径彻底去掉 all_gather_object；
- 适合固定 buffer 的生产场景。

风险：

- 如果 dynamic contract 仍可能造成跨 rank 不一致，需要确认 C++/设备协议
  是否已有错误报告兜底；
- 需要明确哪些字段允许本地校验，哪些必须跨 rank 校验。

### 方案 C：小型 tensor collective 替代 Python object collective

用预分配的 int32/int64 tensor 交换 contract fingerprint，而不是
all_gather_object。每个 rank 本地比较 fingerprint。

优点：

- 保留每次调用的跨 rank 一致性检查；
- 避免对象序列化和 Python 调度；
- collective 规模小。

风险：

- 仍有一次每调用 collective；
- 可能仍带来 rank skew；
- fingerprint 冲突需要足够强的哈希或版本号策略；
- 该 collective 本身仍在 NPU event 计时范围内。

### 方案 D：只在变化点重新校验

contract 状态显式带 version/epoch。buffer 构造、topology 变化、handle
descriptor 变化、显式配置变化时递增 epoch。只有 epoch 变化时才做跨 rank
聚合。

优点：

- 理论上最符合语义；
- 正常热路径没有 Python collective。

风险：

- 需要证明所有会影响 contract 的路径都会更新 epoch；
- 实现和测试面比方案 A/B 更大。

## 推荐路线

推荐路线改为借鉴 CUDA 的设计原则：Ascend 不照抄 CUDA 实现，但 hot path
尽量薄，安全检查放在正确 seam。也就是说，稳定配置在 construction/topology
等稳定 seam 做完整跨 rank 校验；dispatch/combine 热路径只做本地 contract
检查，不再每次执行 Python object collective。

方案 B 是主线，方案 C 不再作为第一阶段实现。原因是即使换成小 tensor
collective，每次调用仍有一次集合通信，仍可能保留 rank skew；而稳定配置
的正确检查点本来就是 buffer construction/topology，不应重复放在逐 token
dispatch/combine 路径上。

具体设计：

1. 新增 DEEP_EP_ASCEND_PREFLIGHT 模式：
   - stable：默认模式，热路径本地校验；
   - full：保留现有 all_gather_object 逐调用跨 rank 校验，用于测试和诊断；
2. construction、construction_communicator、topology 继续完整跨 rank 校验；
3. dispatch、combine 在 stable 模式下只执行：
   - tensor 类型、dtype、shape、contiguous、device 检查；
   - handle ownership 和 descriptor generation 检查；
   - scalar、capacity、alignment、event/stream 模式检查；
4. 本地非法 contract 在进入 C++ runtime 前直接拒绝；
5. 跨 rank 不一致由稳定 seam 的完整校验和 full 模式诊断覆盖；
6. 不改变公开 API 语义和 benchmark 的 logical_gbps 公式。

### 风险与边界

如果某个 dynamic 字段既无法本地校验，也未被稳定 construction contract 覆盖，
stable 模式可能不会在调用前发现该字段的跨 rank 不一致。因此：

1. 每次引入新的 contract 字段时，必须标记它的校验 seam；
2. 无法归入稳定 seam 的跨 rank 字段，只能加入 full 模式，不能默认依赖
   stable 模式发现；
3. correctness matrix 必须同时覆盖 stable 和 full 模式；
4. NPU 失败排查时优先使用 full 模式复现旧诊断能力。

## 验收指标

优化后应满足：

1. 既有 144-case correctness matrix 不回归；
2. 8-rank normal dispatch 的 Python preflight 从约 14 ms 降至 1 ms 以内；
3. C++ entry spread 从约 4-7.5 ms 明显降低；
4. normal dispatch 的 acquire wait 分布不再呈现 rank 0/3 长等待的形态；
5. 五个 operation 的 device mean/p95 不回退；
6. logical_gbps 继续使用原有公式，可作为端到端改进证据；
7. 额外记录 post-preflight kernel 诊断，用于确认数据面收益。

## 实现与验证结果

2026-09-24 已按上述设计实现：

- 新增 DEEP_EP_ASCEND_PREFLIGHT=stable/full，默认 stable；
- stable 模式下 dispatch/combine 只做本地 contract 拒绝，不再执行
  dist.all_gather_object；
- construction、construction_communicator、topology 仍完整跨 rank 聚合；
- full 模式保留原有逐调用跨 rank object collective，用于诊断。

该实现与 CUDA 的对齐关系是：

- 已对齐的原则：公开 dispatch/combine 热路径保持薄，不在每次调用中执行
  Python object collective；稳定配置检查集中在 construction/topology 等
  变化点；本地非法输入在进入 runtime 前拒绝。
- 不照抄的部分：CUDA 没有 Ascend 的 HCCL symmetric-root/topology 协议，也
  没有对应的跨 rank contract 聚合需求；Ascend 保留这些稳定 seam 的完整
  聚合，并额外提供 full 模式用于复现和定位协议问题。
- 语义边界：stable 模式不再逐调用比较 dynamic contract 的跨 rank 差异。
  这与 CUDA hot path 的行为一致，但意味着如果未来新增无法本地校验、也未
  纳入稳定 seam 的字段，必须在 full 模式下验证，并明确该字段的校验时机。

验证结果：

- 本地 host 测试：tests/ascend/test_python_api.py 与
  tests/ascend/test_benchmark_contract.py 共 141 passed，2 skipped；
- NPU8P host 隔离场景：stable hot-path 与 full collective preflight 场景
  均 exit 0；
- NPU8P 2-rank FP8 runtime matrix：12 cases passed，任务
  task_20260924_124714_5090062284；
- NPU8P 8-rank 典型 case A/B：
  stable 任务 task_20260924_124838_51411112691；
  full 任务 task_20260924_124924_52755128092；
  workload 为 4096 tokens、hidden 7168、top-k 6、256 experts、FP8、56 blocks。

8-rank 五个操作的结果如下：

| operation | stable device mean | full device mean | device 改善 |
| --- | ---: | ---: | ---: |
| dispatch | 5.527 ms | 16.162 ms | 10.635 ms |
| expanded_dispatch | 13.899 ms | 25.608 ms | 11.709 ms |
| cached_dispatch | 35.176 ms | 50.032 ms | 14.856 ms |
| combine | 10.978 ms | 26.261 ms | 15.283 ms |
| reduced_combine | 11.710 ms | 26.734 ms | 15.024 ms |

stable 模式的 logical_gbps 为：

| operation | stable logical_gbps | full logical_gbps |
| --- | ---: | ---: |
| dispatch | 587.982 GB/s | 201.086 GB/s |
| expanded_dispatch | 269.722 GB/s | 146.394 GB/s |
| cached_dispatch | 92.390 GB/s | 64.957 GB/s |
| combine | 422.529 GB/s | 176.628 GB/s |
| reduced_combine | 396.127 GB/s | 173.506 GB/s |

注意：本轮 warmups=1、iterations=1，用于 A/B 快速验证；结果足以证明逐调用
object collective 是主要瓶颈，但正式性能报告仍应使用稳定的多迭代协议。

### 与历史 366 GB/s 的口径对齐

P7 优化文档中的 dispatch 约 366.611 GB/s 来自 30 warmups、30 measured
iterations 的默认路径 qualification，是稳定多采样统计，并且当时的诊断口径
没有把本轮逐调用 Python object preflight 纳入 dispatch 的 device event span。

本节 full 模式的 201.086 GB/s 是为了复现旧 preflight 行为而显式打开
DEEP_EP_ASCEND_PREFLIGHT=full 后的单次 A/B 采样；start event 在 Python
preflight 前记录，因此 device span 包含 object gather 造成的 host 延迟和
rank skew。它不是对 P7 历史 366 GB/s 的同口径复测。

stable 模式的 587.982 GB/s 同样是 warmups=1、iterations=1 的快速 A/B 结果，
只能用于和本轮 full 模式比较收益，不能直接替代历史 366 GB/s 的正式
qualification 结论。后续如需正式对齐，应使用相同 workload、相同提交、相同
30 warmups / 30 measured iterations 协议，并分别报告 stable 与 full 或
kernel-only 口径。

## 后续工作

1. 列出 dispatch 和 combine 的完整 contract 字段；
2. 标记每个字段的校验时机和跨 rank必要性；
3. 实现 stable/full 两种 preflight 模式；
4. 为 stable 热路径、本地错误、full 远端错误和 contract mismatch 增加单元测试；
5. 构造跨 rank contract 不一致的失败测试；
6. 在 NPU8P 上重复 2/4/8-rank profiling；
7. 清理本轮临时 profiling 代码，只保留可维护的诊断字段或文档。

## 当前工作区说明

本轮 profiling 曾临时修改 deep_ep/buffers/elastic.py、
deep_ep/utils/envs.py 和 tests/ascend/benchmark/runtime.py，用于输出
wrapper/preflight 分段时间。分析完成后，这些临时诊断修改已从本地工作区
还原；本文档中的数据来自远端诊断结果文件，不依赖这些临时代码继续存在。
