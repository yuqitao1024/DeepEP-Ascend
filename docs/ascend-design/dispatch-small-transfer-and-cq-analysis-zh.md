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
完成项处理或删除完成等待。第三个方向在该轮只完成归因，尚未落地新的
性能优化，不等于 CQ 等待问题已经解决。保留既有 scalar poll 和协议，
临时诊断全部从生产源码移除。

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
二进制仅用于拆分验收。该轮三组合并结果为 1.959 TB/s（十进制），尚未达到
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

## 补充定位：优化后的时间线（2026-09-27）

重新提交任务 `task_20260927_062114_282754212823`，固定设备 0–7，
四次采样全部成功退出。A 为原 baseline，B 为两项优化均启用的正式版本
`1d6ba92` 对应二进制，SHA256 与上表一致。每次 30 warmups；ACL
边界采样 30 次，完整 CPU/NPU trace 12 次，各剔除每 rank 第 0 次。
以下分别汇总 232 与 88 个 rank/capture，不是正式 max-rank ABBA。

| ACL 边界 / μs | A mean | B mean |
| --- | ---: | ---: |
| 最初 stream wait，包含设备执行 | 2799.753 | 2800.139 |
| C++ completion 后续拷贝和等待之和 | 744.673 | 47.289 |
| 整个 snapshot 查询 | 335.054 | 23.654 |
| C++ dispatch 调用 | 3819.980 | 3118.872 |
| Python API 调用 | 4330.991 | 3305.392 |

| 完整 trace 区间 / μs | A mean | B mean |
| --- | ---: | ---: |
| 首个至末个 DeepEP kernel 的跨度 | 2820.975 | 3027.285 |
| 末 kernel → C++ 返回 | 542.946 | 215.260 |
| C++ 返回 → Python API 返回 | 284.586 | 159.638 |
| API 返回后的结果释放区间 | 36.843 | 34.359 |
| 释放结束 → 最终同步开始 | 91.562 | 92.323 |
| 最终设备同步 | 412.947 | 339.383 |
| 末 kernel → capture 结束的整个尾部 | 1378.287 | 847.930 |

两种探针独立采样，不能将它们的均值逐段相减拼成同一条时间线。完整
profiler 的 kernel span 本身也出现波动，因此不能将 trace 差值直接作为
性能验收结果。ACL 边界显示初始设备等待几乎不变，小拷贝和 snapshot
明显缩短，支持“主要是 host/runtime 收益”的结论。剩余约 0.85 ms 的
profiler 尾部包含测量框架释放对象、event 记录、最终同步及 profiler
自身影响，不能全部当成生产 API 中可消除的开销。

原始数据：`results/runtime-tail/final-acl-{A,B}/rank*.json`、
`results/dispatch-tail/final-trace-{A,B}/rank*.json`；汇总脚本
`.scratch/runtime-tail/summarize_final.py`。生产 kernel 未改动。

### Expanded Dispatch / Combine 分段采样

A = completion，B = both。每操作独立进程 ABBA，每进程 30 warmups /
30 captures，剔除首个 capture，表中均为 rank/capture mean、单位 μs。
这组探针在每次采样前显式对齐 host entry，用于归因；正式验收继续使用
未修改的五操作 benchmark，不将二者的均值混合。

| 操作 / 区间 | A（A1/A2 合并） | B（B1/B2 合并） |
| --- | ---: | ---: |
| Expanded：snapshot 查询 | 231.326 | 203.851 |
| Expanded：C++ 调用 | 14026.773 | 14005.987 |
| Expanded：Python API | 14464.530 | 14417.894 |
| Combine：snapshot 查询 | 356.446 | 462.534 |
| Combine：后续 fingerprint | 575.872 | 382.754 |
| Combine：整个 preflight | 1083.699 | 995.296 |
| Combine：C++ 调用 | 12568.201 | 12550.703 |
| Combine：Python API | 13715.461 | 13609.385 |

Expanded 同样只回读 160 字节；B 的 snapshot 内实际 ACL copy+wait
约 38–39 μs，其余耗时在调用外层，符合 Torch-NPU host task queue 衔接
开销，而非回读尺寸超过 4096 的回退。Combine 的部分等待转移到 snapshot，
后续 fingerprint 变短，单看 snapshot 函数会错误解释整个 preflight 的
变化。这里还没有单独拦截 `NPUStream::stream()`，不能精确量化其独占耗时。

完整 trace 也表明存在其他区间波动：Expanded kernel span A/B 为
13973.056/13697.343 μs；Combine kernel span 为 11486.301/11457.605 μs，
但 kernel 结束到 C++ 返回为 541.494/608.767 μs。不能把所有端到端变化
归到 snapshot。

Expanded 采样任务：`task_20260927_062404_284373915799`。该任务在随后
Combine 的首次 warmup 被 `invalid_dispatch_handle` 拒绝：临时脚本漏掉
正式 benchmark 的 `prepare_launches` 刷新步骤。修正脚本后重新执行，
未修改生产校验或生产实现；失败样本不计入性能。Combine 和正式复测任务：
`task_20260927_063027_289332319954`。原始分段数据：
`results/snapshot-followup/{expanded_dispatch,combine-fixed}-{A1,B1,B2,A2}`
以及对应的 `*-trace-{A,B}`，每个目录含 8 rank 的 JSON。

### 正式 ABBA 复测及六组合并

继续使用同一对 completion/both 二进制，新增第 4–6 组三组完整五操作
ABBA。每进程仍是 30 warmups / 30 iterations；任务
`task_20260927_063027_289332319954` 正常结束。12 个正式 run 全部产出
完整五操作正确结果。生产代码、计时公式、输入和优化开关均未改变。

下表为 mean 延迟改善，正数表示更快、负数表示更慢：

| 组 | Dispatch | Expanded Dispatch | Cached Dispatch | Combine | Reduced Combine |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 6.481% | -0.679% | -0.098% | 0.873% | 0.570% |
| 5 | -0.915% | 0.333% | -0.160% | -0.255% | -1.747% |
| 6 | 6.244% | -0.282% | -0.035% | -0.655% | 0.452% |
| 新三组合并 | 3.976% | -0.208% | -0.098% | -0.014% | -0.237% |
| 全六组合并 | 3.739% | -0.848% | -0.034% | -0.396% | -0.260% |

新三组合并普通 Dispatch：A 4.008471 ms，B 3.849096 ms，B logical
bandwidth 2022.836 GB/s。全六组合并每实现 360 个正式 max-rank 样本：
A 4.063687 ms，B 3.911760 ms，B logical bandwidth 1990.431 GB/s，
B p50 3.933857 ms、p95 4.341522 ms。单独第 4/5/6 组 B 带宽分别为
2038.197/1955.504/2078.744 GB/s。因此本轮没有新增生产优化，却因
运行波动出现超过 2 TB/s 的批次；不能据此宣称已经稳定达到 2 TB/s。

snapshot 的普通 Dispatch 增量在六组中五组为正，合并 mean/p50/p95
均改善，继续保留现有实现，但“每组都改善”的表述仅适用于最初三组，
不能推广到扩大样本后的结论。第 5 组 B1/B2 mean 分别为
4.126084/3.837171 ms，显示跨进程波动。Expanded 六组中五组方向为负，
仍保留小幅性能代价的记录，不能仅凭单独探针不复现而判为不存在；
Combine 新三组近似持平，也不能抹去旧三组数据。

数据：`results/runtime-tail/snapshot-{4,5,6}-{A1,B1,B2,A2}.json`。
汇总命令：`.scratch/runtime-tail/summarize_candidate.py snapshot 4 5 6`
以及 `snapshot 1 2 3 4 5 6`。六组合并不混入带 profiler/interposer 的数据。

### 保持正式前序操作的 Expanded 归因

补充任务 `task_20260927_064350_300349315294` 按正式顺序先执行普通
Dispatch 的 30 warmups / 30 iterations，再准备 Expanded；取消诊断脚本
额外的 host entry 对齐，保留正式 benchmark 的同步和 Event 顺序。
每个 Expanded 诊断进程仍为 30 warmups / 30 captures，剔除首个 capture。
A = completion，B = both，四个进程均通过校验。

| 运行 | snapshot rank mean / μs | Python API rank mean / μs | max-rank Event mean / μs |
| --- | ---: | ---: | ---: |
| A1 | 223.796 | 14567.253 | 15041.250 |
| B1 | 182.819 | 14566.172 | 15261.129 |
| B2 | 179.597 | 14416.302 | 14842.319 |
| A2 | 225.524 | 14593.772 | 15051.654 |

合并 max-rank Event 为 A 15046.452 μs、B 15051.724 μs，近似持平。
两个 B 进程的 snapshot 都更快，但整个操作的最慢 rank 耗时仍明显波动。
因此尚不能将正式六组 Expanded 的小幅退化定位到 snapshot 拷贝本身，
也不能宣称退化已修复。该实验保留诊断 hook，其结果不并入正式 ABBA；
仍以六组正式数据记录 Expanded -0.848%、Combine -0.396% 的变化。

数据：`results/snapshot-followup/expanded-sequence-{A1,B1,B2,A2}/rank*.json`。
脚本：`.scratch/snapshot-followup/{probe.py,sequence.sh,summarize.py}`。

### CQ：等待未就绪与处理已就绪完成项

在独立诊断构建中逐完成项记录 command、peer、tail、owner 轮询起止与
重试数，并对已就绪 owner word 增加八次读取。诊断 workspace 显式发布，
后续 service kernel 进入时失效诊断计数缓存，避免跨 core 读到旧记录。
协议和超时检查均保留。构建任务 `task_20260927_064243_299428423099`，
诊断二进制 SHA256：
`d569b28ba4f34c3eccfcfe505614d695aab5d7ac65bd4507612c7b0534cf5a79`。

上述 `task_20260927_064350_300349315294` 随后完成两轮五操作 profile，
正确性全部通过，进程正常退出。每轮普通 Dispatch 的每个 rank 都记录
28 个完成项：payload flush（command 7）7 项，control flush（command 29）
21 项。下表的每 rank 指标取 8 rank 均值；已就绪观察区间取所有相应完成项
均值。cycle 指设备计数器读数，未换算为时间或物理带宽。

| 指标 | 第 1 轮 | 第 2 轮 |
| --- | ---: | ---: |
| 每 rank owner 轮询总 cycles | 766522.375 | 776922.125 |
| payload 占全部 owner 轮询 cycles | 99.265% | 99.250% |
| 每 rank 最长单项占轮询总量的比例 | 95.239% | 95.111% |
| 每 rank 首次读取即就绪项数 / 28 | 24.75 | 24.25 |
| 每 rank owner 重试数 | 2841.5 | 2528.5 |
| 已就绪项完整 owner 观察区间 / cycles | 271.591 | 278.619 |
| 八次 ready-word 读取的平均 cycles/次 | 246.517 | 249.160 |

两轮所有 rank 的最长等待均为首个 payload peer 完成项，后续 payload
完成项多数已就绪；control 的 21 项全部首次读取即就绪、零重试。
首个 peer 是遍历顺序中的首项，不代表该 peer 的链路一定最慢。长区间
包含硬件完成进度和 owner 可见性的等待，不能分解为纯传输时间，更不能
据此证明链路饱和。

测量限制：探针额外读取与记录会增加开销，不作为正式性能数据。
单独的 `first_read_cycles` 仅有约 6–7 cycles，时间戳可能早于 load 结果
被实际消费，不能解释为序列化 load 延迟，故不用于结论。上表使用包含
owner 判断的完整就绪观察区间；八次读取均值仅作辅助，不声称是严格的
单次访存延迟。新增诊断版本的 drain 总量也不能直接与旧探针相减。

原始数据：`results/dispatch-tail/cq-followup-{1,2}.json`；本地归档于
`.scratch/cq-followup/results/dispatch-tail/`，汇总脚本
`.scratch/cq-followup/summarize.py` 校验五操作、8 rank 和每 rank 28 项。

### 本轮结论与下一步

本轮补齐时间线、扩大 snapshot 正式样本并细分 CQ，没有新增生产优化。
现有两项 host 优化继续保留；普通 Dispatch 六组合并为 1990.431 GB/s，
仍不能宣称稳定超过 2 TB/s。Expanded 的小幅代价尚未完成因果定位，
Combine 扩大样本后的变化较小，但两者都不能称为已消除退化。

第三项完成了诊断，没有形成可保留的 CQ 优化补丁。当前主要观察到的是
payload 尚未完成时的等待，没有证据支持优先改写已就绪 CQ 的处理或删除
等待。结合此前未获得稳定收益的单点候选，下一阶段建议设计一个最小的
分块计算/通信 overlap 实验：保留完成确认和消费顺序，检验提前提交 payload
能否缩短关键路径，再以同一正式用例 ABBA 验收。本轮不实施该流水改造，
也不宣称其他单点优化空间已被穷尽。

所有任务已结束，远端已恢复正式 `both` 二进制，SHA256：
`aaf762e60005f108d500a73d6ff46962dda6f5db9ff1e321a09461879fec57c0`。
临时替换的 `elastic_buffer.hpp`、`stage_profile.hpp`、
`aicore_transport_service.hpp` 均已恢复，哈希与本地生产源码一致；
诊断代码仅保留在 `.scratch`，不进入生产源码。
