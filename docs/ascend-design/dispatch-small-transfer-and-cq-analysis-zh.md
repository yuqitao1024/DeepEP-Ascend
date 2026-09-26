# Dispatch 小拷贝与 CQ drain 验证

日期：2026-09-27。基线：`f2d6683`，已合并 descriptor 重复回读。
前置记录：[单次回读](dispatch-descriptor-readback-optimization-zh.md)。

## 范围和验收

依次检查 C++ completion、剩余 snapshot 回读、release payload CQ drain。
暂不引入计算/通信 overlap。固定 NPU8P、设备 0–7、8192 tokens/rank、
hidden 7168、top-k 8、256 experts、num_sms 64，FP8 普通 Dispatch。
CANN/HCOMM 使用 `/data/disk2/cann_version/0916/use_cann/cann-9.3.0`
同树环境。正式性能沿用五操作 benchmark 与原 Event 统计，每次独立进程
30 warmups / 30 iterations，以至少三组 ABBA 的稳定收益决定是否保留。

## ACL 调用边界实测

实验脚本 `.scratch/runtime-tail/{native_probe.cpp,probe.py,run.sh}`。
临时动态库只在探针进程中记录 ACL 调用时间，生产二进制未改变。
两次独立运行各 30 warmups / 30 captures，剔除各 rank 第 0 个 capture，
每轮 232 个 rank/capture 样本。不是正式 benchmark 的逐 iteration max-rank。

每个 capture 的调用序列完全一致：

| 所在段 / 操作 | 字节数 | 第 1 轮 mean / μs | 第 2 轮 mean / μs |
| --- | ---: | ---: | ---: |
| C++：stream 同步，包含 kernel 执行等待 | — | 2803.267 | 2831.543 |
| C++：transport diagnostic D2H | 64 | 160.083 | 195.210 |
| C++：count bridge D2H | 2176 | 170.674 | 211.609 |
| C++：descriptor H2D | 160 | 193.262 | 223.528 |
| reconcile：descriptor D2H | 160 | 197.575 | 260.455 |
| 整个 snapshot 查询 | — | 205.492 | 269.793 |

snapshot 的 C++/pybind 非拷贝部分平均仅约 8–9 μs。优化 Python tuple 或
vector 分配不是优先候选。初始 stream 同步包含 kernel 等待，不能直接删掉。
最后显式设备同步未被这些 ACL hook 完整捕获，不由此推断其耗时为零。

任务：`task_20260927_022348_13214229600`，两轮探针成功，随后原 stage
profile 五操作通过；该 profile 运行只在完整结果写出后发生已知 teardown
SIGSEGV。原始数据：`results/runtime-tail/acl{1,2}/rank{0..7}.json`。

## 拷贝机制筛选

单卡微测比较普通内存同步拷贝、固定页内存同步拷贝、固定页内存异步拷贝
加 stream 等待，每项检查实际返回字节。异步路径约 15 μs；稳定后的同步
路径约 24 μs，初始部分同步样本明显更慢。微测存在顺序和状态影响，不将
其倍率外推为端到端收益。

随后使用临时 interposer 做两项独立 ABBA，每次 30/30、全五操作校验：

| 原型作用范围 | A mean / ms | B mean / ms | Dispatch 改善 |
| --- | ---: | ---: | ---: |
| diagnostic + count D2H、descriptor H2D | 4.698327 | 3.813513 | 18.833% |
| 仅 160 字节 descriptor D2H | 4.892277 | 4.436897 | 9.308% |

任务：`task_20260927_022811_1463277601`。这些是各一组机制筛选，不能代替
最终生产实现的三组验收。按字节数筛选的 interposer、其进程级资源持有方式
都不会进入生产实现。

生产实现使用 runtime 拥有的有界固定页缓冲区，在指定 stream 上拷贝并
等待完成后返回；保留字节/generation 校验，等待失败不发布结果，不释放
或复用仍被异步任务引用的内存。大于缓冲区容量的请求保留原路径。

## CQ 细分

临时设备探针分别记录 owner 轮询区间、tail/doorbell 更新区间、重试次数、
完成项数、drain 次数及空 drain 次数。只在 profile 路径启用；带探针的数据
用于归因，不能作为正式收益数据。诊断编译任务：
`task_20260927_023331_16989823640`。

两次完整五操作 profile 均通过，任务
`task_20260927_023459_17923713347`。每轮每 rank 的普通 Dispatch 都有
28 个 CQ 完成项、21 次 drain，其中 7 次为空 drain。

| 轮次 | CQ drain 范围 / cycles | owner 轮询范围 / cycles | tail/doorbell 更新范围 / cycles |
| --- | ---: | ---: | ---: |
| 1 | 790189–811640 | 783477–804682 | 578–712 |
| 2 | 756542–814219 | 749678–807465 | 577–665 |

按同一 rank 配对计算，owner 轮询约占 drain 的 99.1%–99.2%，
tail/doorbell 更新不到 0.1%。这里的轮询时间包含读 owner 与等待硬件完成，
不等于纯链路传输；没有据此宣称 P2P 已打满。该结果不支持优先重写 CQ
完成项处理或删除完成等待。第三个方向以“归因完成、没有新的处理热点”
闭环，保留既有 scalar poll 和协议，临时诊断全部从生产源码移除。

## 生产实现

`SmallHostTransfer` 为每个 runtime 惰性分配并复用 4096 字节 pinned host
buffer。指定 stream 上 enqueue copy，等待成功后才将 D2H 数据复制给调用方；
H2D 先复制到自有缓冲区，等待成功后返回。每次仍是真实设备读写。
enqueue/wait 失败不发布 D2H 数据；未完成的缓冲区禁止复用，destroy 先 drain，
失败时保留资源以供再次清理。超过容量或后端没有异步接口时，先等待指定
stream 再回退原同步拷贝。

分别构建两份可独立验收的二进制：

- `completion`：仅替换 Dispatch diagnostic/count D2H 和 descriptor H2D。
  diagnostic 借助 transport 的可选 readback callback 使用 runtime 缓冲区；
  原接口与其他 completion 调用保持兼容。
- `both`：在上述基础上，snapshot 同样使用该机制。snapshot 获取当前
  stream，并先让 Torch-NPU 提交该 stream 的 host task queue，保证已排队的
  descriptor 修改在回读之前完成；原 tensor 身份、
  大小、内容和 generation 检查保留。

生产二进制不含 interposer、字节数筛选、CQ 诊断或实验环境开关。

新增设备回归去掉了 descriptor mutation/restore 后的显式
`torch.npu.synchronize()`。初版候选在该检查失败：已有 stream 查询使用
`stream(false)`，不会提交 Torch-NPU host 队列，裸 ACL copy 可能先读到旧字节。
修正为 snapshot 专用的 `copy_to_host_on_current_stream`，通过 `stream()`
提交在先的 host 工作，再 enqueue copy 与等待。未修改全局 stream 查询语义，
避免把额外同步扩展到全部 hot path。失败记录：
`task_20260927_024627_21172911195`。

修正后的双卡检查全部通过。第一次在两个 `SNAPSHOT_BOUNDARIES_PASSED`
都输出后出现已知 teardown SIGSEGV；再次运行正常退出，并完成 512 experts、
8192 tokens、hidden 128、top-k 2 的五操作回退检查。任务：
`task_20260927_025043_2219034566`。

本地检查：Python API/platform 加新 pinned-buffer 回归 49 passed、3 skipped、
19 subtests passed；runtime lifecycle 两项通过；transport contract 与
device transport stub 六项通过。新回归覆盖等待失败不发布、禁止复用 pending
缓冲区、销毁前 drain、enqueue 失败清理、重复 destroy，以及回退路径的
stream 顺序和错误传播。

8-rank 正式验收任务：`task_20260927_025200_22619124415`。
先执行 BF16 sync、FP8 async、FP8 previous-event + async + allocate-on-comm-stream
三个 case（每个五操作），均已通过；再分别进行两项独立的三组 ABBA。

| 变体 | extension SHA256 |
| --- | --- |
| baseline，已合并重复 descriptor 回读 | `4d0526f74d9517f7b6b89a8aab102a35896e177f672ec13cf7df5e1d2b9591d6` |
| completion | `56cba2bd108b4dd172e50850393b517b171aefc62f597cdbaee49a600ca71dae` |
| both，含 Torch-NPU 队列衔接修正 | `aaf762e60005f108d500a73d6ff46962dda6f5db9ff1e321a09461879fec57c0` |

Python wrapper 未修改，SHA256 为
`a09e272218174cc9f418104c8c1732b51fa6e02c11c12588c78c1b73c1921409`。

## C++ completion 独立 ABBA：保留

A = baseline，B = completion。每组每实现 60 个正式 max-rank 样本，
合并每实现 180 个；各 run 的原始样本重新汇总，不平均分位数。

| 组 | A mean / ms | B mean / ms | mean 改善 | p50 改善 | p95 改善 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.801853 | 4.081223 | 15.007% | 14.987% | 16.286% |
| 2 | 4.700481 | 4.058609 | 13.655% | 13.278% | 14.748% |
| 3 | 4.760593 | 4.171515 | 12.374% | 13.096% | 5.995% |
| 合并 | 4.754309 | 4.103782 | 13.683% | 13.274% | 13.162% |

合并 Dispatch logical bandwidth：1637.691 → 1897.296 GB/s，仍为
8-rank 聚合逻辑字节 / Event 时间，不是物理链路带宽。
12 次运行五操作正确性全部通过。其他操作的合并 mean 改善：
Expanded Dispatch 3.671%、Cached Dispatch 0.251%、Combine -0.014%、
Reduced Combine 0.473%。不将这些小幅混合变化当作额外稳定收益。

数据：`results/runtime-tail/completion-{1,2,3}-{A1,B1,B2,A2}.json`。
脚本：`.scratch/runtime-tail/{run_candidate.sh,abba_candidate.sh,summarize_candidate.py}`。

## snapshot 增量 ABBA：普通 Dispatch 收益可复现

A = completion，B = both，隔离 snapshot 本身的增量效果。每组每实现
60 个正式 max-rank 样本，合并每实现 180 个。仍逐次执行 30 warmups。

| 组 | A mean / ms | B mean / ms | mean 改善 | p50 改善 | p95 改善 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.148095 | 3.849256 | 7.204% | 3.941% | 5.245% |
| 2 | 4.097695 | 4.023418 | 1.813% | 1.063% | 3.736% |
| 3 | 4.110921 | 4.050596 | 1.467% | 0.898% | 1.697% |
| 合并 | 4.118904 | 3.974423 | 3.508% | 1.161% | 4.191% |

普通 Dispatch 三组 mean/p50/p95 都改善，按本轮普通 Dispatch 的验收目标
保留。聚合逻辑带宽 1890.331 → 1959.049 GB/s。第一组增益较大，不能只用
该组的 7.2% 代表稳定增益。两项独立实验的百分比不直接相加。

当前两项优化在生产实现中默认生效，无需新增宏或环境变量；两份候选
二进制仅用于拆分验收。合并结果为 1.959 TB/s（十进制），尚未达到
2 TB/s。同样逻辑字节数下，达到 2 TB/s 需要将平均延迟从 3.974 ms
降至约 3.893 ms，继续减少约 81 μs。第一组超过 2 TB/s 不代表合并结果
已稳定达到该水平。

12 次运行五操作正确性全部通过。但其他操作没有同步改善：Expanded
Dispatch 三组 mean 分别退化 0.291%、2.779%、1.415%，合并退化 1.490%；
Combine 三组退化 0.586%、0.681%、1.069%，合并退化 0.779%。Cached
Dispatch 合并改善 0.030%，Reduced Combine 合并退化 0.283%。不把这些
变化省略或认定为已排除的噪声，也不宣称全五操作均获益；其他操作的
性能归因不由本轮普通 Dispatch 验收替代。

正式任务 `task_20260927_025200_22619124415` 完整结束，exit=0，24 次
ABBA 运行均有完整五操作结果。少数运行完整输出结果后发生已知 teardown
SIGSEGV，由现有结果验证器确认后继续，不算为干净进程退出。
本地与实测远端的六个生产修改文件及 snapshot 回归脚本 SHA256 一致。

数据：`results/runtime-tail/snapshot-{1,2,3}-{A1,B1,B2,A2}.json`。

## 与 CUDA / H800 的关系

这是 Ascend host/runtime 实现的优化，不修改 CUDA 路径，不等于 kernel
加速，也不通过减少输入规模、关闭校验或改变计时窗口获得收益。

CUDA 的 `csrc/elastic/buffer.hpp` 构造时使用 mapped pinned host workspace
（`cudaMallocHost`、`cudaHostGetDevicePointer`）。普通 Dispatch 的
`do_cpu_sync=True` 路径直接轮询 host workspace 的 rank/expert counters，
而不是像本 Ascend 路径一样执行几次小块同步 D2H。CUDA 也有 host 等待、
输出分配和流依赖，但实现结构已经不同。Python `_reconcile_ascend_handle`
仅由非 CUDA 路径调用，CUDA 没有同样的 descriptor generation/fingerprint
回读链。因此没有依据说 CUDA host 存在同样量级、同样原因的瓶颈。

源码核查位置：`csrc/elastic/buffer.hpp:135`（mapped host workspace）、
`:1019`（计数轮询）、`:558`（event/stream epilogue）；
`deep_ep/buffers/elastic.py:1615`（仅 Ascend 的 handle reconcile）。这些
是设计差异的证据，不能用来量化 H800 host 占比，也不能证明 CUDA host
已经没有优化空间。

比较必须区分两个指标：

1. 端到端公开 API Event span / logical bandwidth：CUDA parity 入口
   `tests/elastic/test_ep.py::run_cuda_parity` 与 Ascend 共用
   `tests/ascend/benchmark/timing.py::NpuEventTimer` 的测量逻辑，只替换
   CUDA/NPU Event backend。两边都是 start event → operation → end event，
   另记 wall time。host 若推迟后续 kernel 或 end event 提交，会影响此指标；
   Event span 是设备上两个事件之间的时间，并不覆盖所有 CPU 开销，不能
   等同于完整 wall time；跨平台报告应同时保留二者。
   优化真实 host 开销属于该指标的有效改善，不要求两边执行相同数量的
   host 指令或采用相同内部协议。
2. 设备 kernel 执行/跨度及通信效率：本轮没有修改生产 kernel，不能用
   端到端带宽增长证明 kernel 比 H800 更快。CUDA 默认非 parity 测试还使用
   `bench_kineto` 统计 kernel；这类历史数字不能直接与 API Event span 混比。

本轮未登录 H800 测量 host 时间。仓库文档中的历史状态也不能替代新 H800
报告；正式跨平台结论仍需要同 workload fingerprint、模式、warmup、迭代数
和 Event 计时的两份实测结果。

补充归因任务 `task_20260927_030850_34578628438` 曾提交固定设备 0–7，
计划分别对 baseline/both 采集 ACL 调用边界和完整 CPU/NPU 时间线。
2026-09-27 再次核查时队列返回 `not_found`，无对应采样产物，也没有
本轮 pending/running 任务；消失原因尚未确认。该补充实验未完成，不能
作为前后时间线验证依据，不影响上面已完成的正式 ABBA 结果。
