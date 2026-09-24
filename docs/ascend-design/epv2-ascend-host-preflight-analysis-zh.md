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

不能采用“各 rank 本地 fingerprint 相同就各自跳过 collective”的方案：如果
部分 rank 跳过而另一部分 rank 仍进入 all_gather_object，同一 process group
会因 collective 操作不匹配而挂死。因此优先采用方案 C 的安全变体：
所有 rank 每次都参与同一个轻量级 fingerprint 交换，但只在 fingerprint 变化
时回退到完整的 Python object contract 聚合。

具体路线：

1. 梳理 dispatch/combine contract 字段；
2. 为完整 contract 计算 64-bit fingerprint；
3. 在 process group 上用预分配的小 tensor 做 all_gather，交换 fingerprint；
4. 所有 fingerprint 一致时，直接跳过完整 contract 的 object 聚合；
5. 任一 fingerprint 缺失、非法或不一致时，所有 rank 进入同一个
   all_gather_object 回退路径，交换并比较完整 contract；
6. 本地 error_code 必须编码进 fingerprint 交换结果，不能在本地直接抛出；
7. 用 2/4/8-rank 对比 preflight、C++ entry spread、acquire wait 和五个
   operation 的 device mean/p95。

## 验收指标

优化后应满足：

1. 既有 144-case correctness matrix 不回归；
2. 8-rank normal dispatch 的 Python preflight 从约 14 ms 降至 1 ms 以内；
3. C++ entry spread 从约 4-7.5 ms 明显降低；
4. normal dispatch 的 acquire wait 分布不再呈现 rank 0/3 长等待的形态；
5. 五个 operation 的 device mean/p95 不回退；
6. logical_gbps 继续使用原有公式，可作为端到端改进证据；
7. 额外记录 post-preflight kernel 诊断，用于确认数据面收益。

## 后续工作

1. 列出 dispatch 和 combine 的完整 contract 字段；
2. 标记每个字段的校验时机和跨 rank必要性；
3. 实现 fingerprint 快速交换和完整 contract 回退路径；
4. 为 fingerprint 一致、本地错误、远端错误和 contract mismatch 增加单元测试；
5. 构造跨 rank contract 不一致的失败测试；
6. 在 NPU8P 上重复 2/4/8-rank profiling；
7. 清理本轮临时 profiling 代码，只保留可维护的诊断字段或文档。

## 当前工作区说明

本轮 profiling 曾临时修改 deep_ep/buffers/elastic.py、
deep_ep/utils/envs.py 和 tests/ascend/benchmark/runtime.py，用于输出
wrapper/preflight 分段时间。分析完成后，这些临时诊断修改已从本地工作区
还原；本文档中的数据来自远端诊断结果文件，不依赖这些临时代码继续存在。
