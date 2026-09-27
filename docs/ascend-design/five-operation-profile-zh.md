# 当前五操作性能差异：NPU8P Profiling

日期：2026-09-27。代码：`0fe8976`。本轮只诊断，不修改生产实现。

结论：五操作之间的大幅差距主要来自设备侧不同实现路径。Cached Dispatch
有约 48.53 ms 的 producer plan 单点；Expanded Dispatch 的 producer 打包、
单线程 prefix 和输出搬运明显慢于普通路径；Combine 的主要工作在 producer
搬运、通信 service 和最终归约。Host 开销仍有优化空间，但不能解释这些
十几至几十毫秒的差距。

## 环境与测量口径

- NPU8P，固定设备 `0,1,2,3,4,5,6,7`，通过 `task-submit` 独占申请。
- CANN/HCOMM 同树：`/data/disk2/cann_version/0916/use_cann/cann-9.3.0`。
- Python：`/home/miniconda3/envs/py310/bin/python`。
- `--num-tokens 8192 --hidden 7168 --num-topk 8 --num-experts 256 --num-sms 64`。
  沿用原 manifest，实际每 rank tokens 为 `8192-rank`，即 8192 至 8185。
- Case：`ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0`；
  `allow_multiple_reduction=1`。Dispatch 使用 FP8，Combine 输入为 BF16。
- stable preflight、1 channel；两个 pipeline chunk 环境变量均未设置，overlap 关闭。
  沿用当前优化配置：device prefix、parallel prefix、8192-byte consumer tile、
  token fan-out、Combine direct local placement、512-element vector reduce 等。
  Selector 不适用的路径仍按生产规则回退，没有强行启用。
- 扩展使用当前提交对应的已验证 v3b 构建，SHA256：
  `568a44c2e03fdf14f43a5101a6e037354155b5a5be93adeb37713480c554e3d9`。
  当前源码与构建归档覆盖的生产文件逐一匹配；远端不依赖 Git 元数据识别版本。

正式性能使用未修改的五操作 benchmark，每操作 30 warmups / 30 iterations，
每次取 max-rank NPU Event，逻辑字节数跨 rank 求和。逻辑带宽包含协议约定的
copy/reduction 字节，**不是物理 P2P 链路带宽**。

Profiler 使用 Torch-NPU Level2、关闭 AiC metrics，每操作独立进程。先执行
原准备和正确性校验，并按原 benchmark 顺序运行前置操作；刷新目标 handle 后，
目标操作预热 30 次，先采 30 次无 profiler Event，再采 12 次时间线。排除
首个 capture，每操作保留 8 ranks × 11 = 88 个诊断样本。

Profiler capture 前额外对齐 host entry 并同步设备，因此其时间不能替代正式
性能；本文分段表均为**每次选择 Event 最慢 rank 后，11 次取均值**，不混用
跨 rank 平均值。使用区间并集核算 kernel span，不将嵌套 host 等待重复相加。

## 当前性能

正式复测任务 `task_20260927_115839_289044116335`，完整五操作校验通过，
进程正常退出，未发生 teardown SIGSEGV。

| 操作 | 正式 Event mean / ms | 正式逻辑带宽 / GB/s | 采集进程无 profiler mean / ms | Profiler Event mean / ms |
| --- | ---: | ---: | ---: | ---: |
| Dispatch | 3.745 | 2079.32 | 3.926 | 4.051 |
| Expanded Dispatch | 14.934 | 619.47 | 15.087 | 14.961 |
| Cached Dispatch | 63.720 | 122.19 | 63.980 | 64.068 |
| Combine | 14.195 | 767.96 | 13.879 | 13.989 |
| Reduced Combine | 14.414 | 756.32 | 14.506 | 14.499 |

本轮 Dispatch 单次正式复测超过 2 TB/s。此前六个基线进程合并为
3.896 ms / 1998.70 GB/s；本轮没有改生产代码，也没有做新的收益 ABBA，
所以不能把 2079.32 GB/s 宣称为稳定新基线或新优化收益。

## 设备关键路径

各路径 kernel 之间没有观察到并行执行，kernel 间总空隙只有约 12–19 μs。
大部分 device 时间在 kernel 内；仅合并 launch 间隙不能消除主要差距。

| 指标 / ms | Dispatch | Expanded | Cached | Combine | Reduced Combine |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepEP kernel 首尾跨度 | 2.818 | 13.701 | 58.887 | 11.582 | 11.942 |
| Kernel 累计执行 | 2.805 | 13.688 | 58.874 | 11.568 | 11.923 |
| Kernel 间空隙 | 0.013 | 0.012 | 0.012 | 0.014 | 0.019 |
| Kernel 数量 | 28 | 28 | 26 | 23 | 23 |

### 普通 Dispatch

| 阶段 | 耗时 / ms |
| --- | ---: |
| producer record | 0.120 |
| producer hidden fan-out | 0.166 |
| release command 生成 | 0.173 |
| transport service（包括完成等待） | 0.946 |
| epilogue acquire | 0.194 |
| expert count / parallel prefix | 0.083 / 0.234 |
| metadata / assign destinations | 0.171 / 0.015 |
| 输出 metadata/尾部 copy VF | 0.166 |
| 输出 hidden vector copy | 0.317 |

Service 是最大单项，约占 profiler Event 的 23%。它包括传输服务和完成等待，
不能据此声称 0.946 ms 全是纯 DMA 传输或 P2P 已饱和。输出搬运两段合计
约 0.483 ms；producer record + hidden 合计约 0.286 ms。

### Expanded Dispatch：不是单纯数据变多

| 可对照阶段 | 普通 / ms | Expanded / ms |
| --- | ---: | ---: |
| producer record | 0.120 | 3.776 |
| producer hidden copy | 0.166 | 0.937 |
| epilogue prefix | 0.234 | 2.792 |
| assign destinations | 0.015 | 0.337 |
| 输出 hidden vector copy | 0.317 | 3.862 |
| transport service | 0.946 | 0.952 |

前五项解释了约 10.85 ms 的设备差距；整个 kernel span 差约 10.88 ms。
两者逻辑 scale-up 字节数相同，均为 2,595,363,264 bytes；通信 service 也接近。
Expanded 输出更多槽位，但当前实现路径的差异同样重要：

1. `dispatch_token_fanout.hpp` 明确排除 expanded/cached。普通 FP8 hidden
   7168 bytes 全部走 fan-out 搬运；回退路径按 2048 bytes 切 producer vector
   copy，剩余 1024 bytes 在 `direct_dispatch_write_record` 的逐 byte 循环处理。
   该代码差异与 record 3.776 ms 的热点一致；具体收益仍需单变量 A/B 验证。
2. `select_dispatch_parallel_prefix_config` 排除 expanded。
   `direct_dispatch_epilogue_prefix_vf` 只有 thread 0 扫描 expert/tile，
   当前约 2.792 ms，普通 parallel prefix 约 0.234 ms。
3. `select_dispatch_consumer_tile_config` 排除 expanded/cached，回退为
   512-byte tile；普通路径为 8192 bytes。Expanded 搬运还需按 top-k lane
   定位输出槽位。约 3.862 ms 的 copy 需要结合展开语义重新优化，不能仅把
   普通路径 selector 的排除条件删掉。

### Cached Dispatch：明确的 producer plan 单点

`direct_dispatch_producer_plan_kernel` 为 **48.526 ms**，约占整个 profiler
Event 的 **75.7%**，占 kernel 累计时间约 **82.4%**。

源码 `direct_dispatch_producer_plan.asc` 按 `threadIdx.x` 分配目标 rank，
control launch 为 1 block。在 8-rank case 中仅 8 个线程负责各目标 rank，
每线程串行遍历所有 tokens × top-k，执行 cached slot 解码、合法性、唯一性
bitmap 和 count/max-slot 校验。普通 grouping 路径跳过这个串行 plan。
这是可定位的低并行度算法路径，不是笼统的 host preflight 慢。

其余主要项：record 3.795 ms、producer hidden copy 0.937 ms、consumer hidden
copy 2.866 ms、service 0.907 ms、acquire 0.693 ms、validate 0.572 ms。
Cached 与普通 Dispatch 的逻辑字节数完全相同，无法用数据量解释约 17 倍的
正式耗时差。重用 handle 没有消除当前每次执行的全量 cached slot 校验。

若未来恢复 Cached 专项，应首先把 plan 校验按 token tile 并行化，再归并
计数和错误，保留跨 tile slot 唯一性和完整协议检查；本轮没有实现或关闭检查。

### Combine 与 Reduced Combine

| 阶段 | Combine / ms | Reduced Combine / ms |
| --- | ---: | ---: |
| producer plan | 1.134 | 1.156 |
| producer plan prefix | 0.764 | 0.765 |
| producer record VF | 0.044 | 0.045 |
| producer payload 搬运 / expanded 预归约 | 3.359 | 3.756 |
| release command 生成 | 0.167 | 0.164 |
| transport service | 1.723 | 1.726 |
| acquire | 0.683 | 0.609 |
| validate | 0.632 | 0.639 |
| prepare vector slots | 0.560 | 0.560 |
| 最终 vector reduce | 2.410 | 2.411 |

两者最大的差异在 producer 端，Reduced 多约 0.397 ms。当前 case 的
`allow_multiple_reduction=1`，Expanded 输入先做 producer 预归约；这两项
正式报告的逻辑通信字节数相同，最终归约和 service 也几乎相同。
不应把 Reduced 的增加笼统解释为最后一个 reduce 更慢。

Combine 不是只有一个通信长尾：producer payload、plan/prefix、最终归约
都有毫秒级成本。优先细化这些单项，再讨论更复杂的 overlap，仍有明确依据。

## Host 与此前约 1.3 ms 尾部

| 区间 / ms | Dispatch | Expanded | Cached | Combine | Reduced Combine |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python preflight | 0.097 | 0.100 | 0.961 | 0.816 | 0.876 |
| 最后 kernel 结束 → C++ 返回 | 0.542 | 0.395 | 1.495 | 0.941 | 0.911 |
| C++ 返回 → Python API 返回 | 0.138 | 0.348 | 0.392 | 0.029 | 0.033 |
| 最后 kernel 结束 → capture 结束 | 1.225 | 1.382 | 2.447 | 1.731 | 1.585 |

最后一行**包含**前两段尾部，不能相加。普通 Dispatch 的 1.225 ms 中，
约 0.542 ms 在 C++ 返回之前，0.138 ms 在 Python 返回之前，其余约 0.545 ms
在返回后的对象释放、Event 提交、最终设备同步及 profiler 包装范围内。
它不是一个耗时 1.3 ms 的 kernel，也不能全当成生产 API 可消除的开销。

普通 Dispatch 的 C++ 内 4 次 stream sync 累计约 3.074 ms，其中仅约
0.427 ms 落在最后一个 DeepEP kernel 之后。大部分在等待已有设备工作，
不能再加到 2.818 ms kernel span 上。

Cached/Combine 使用不同的事件完成路径。Cached 在 C++ 内每 capture 平均
约 43,781 次 `aclrtQueryEvent*` 调用，调用区间累计约 27.47 ms；Combine
约 8,686 次、5.35 ms。它们主要与设备执行重叠，不能把这些数值算成额外的
端到端成本。该 CPU 轮询行为可单独研究，但没有证据说明它导致了 Cached
48.53 ms 的 plan kernel；profiler 也会影响调用频率。同步 memcpy 的长区间
同样可能包含 runtime 等待，不能直接解释成小数据传输带宽低。

## 建议顺序与边界

- 如果继续只优化普通 Dispatch：service/CQ 完成路径、consumer 搬运是当前
  较大设备区间；精确区分尾部的可消除 host 时间。不能仅靠消除 launch gap
  获得毫秒级收益。已验证退化的 overlap 仍保持关闭。
- 若扩展到其他操作：Cached plan 是最明确的单点；Expanded 的 producer
  尾部搬运、prefix、consumer copy 有直接路径对照；Combine 优先拆 producer
  payload 和最终归约。尊重此前 Cached 专项暂停的决定，本轮只记录发现。
- 上述为优化方向，尚无修复后的收益数据，不承诺把某个 kernel 耗时完全消除。

## 运行完整性与复现材料

首次任务 `task_20260927_114359_227784228555` 的原 benchmark 在准备过程中
240 秒超时，没有有效报告，排除于性能结果。等待期间对 rank 7 做过一次 GDB
只读栈检查，看到 CANN runtime 内等待，但不足以定位根因；该失败运行本来就
不能进入性能统计，不能仅据此认定与此前 rank 6 → rank 4 flush 超时同因。

随后任务 `task_20260927_114846_231002625319` 的五个独立采集进程均通过
原始准备校验，输出全部 40 份 trace 和 40 份 measurement，正常结束采集。
Profiler 导出后脚本使用 `os._exit(0)`，不将它当作 teardown 回归证据；后续
原始 benchmark 正常退出才是本轮完整进程验证。首次间歇性超时仍未闭环，
不能因重跑通过声称稳定性问题已修复。

远端 workspace：`/home/pyptouser/yuqitao/deepep-official-simt.AaR0IW`。

- 原始 trace：`results/five-op-profile/current-0fe8976-<operation>/rank*.json`。
- 原始正式报告：`results/dispatch-tail/five-op-current-0fe8976-retry.json`。
- 本地归档：`.scratch/five-op-profile/results/`。
- 脚本：`.scratch/five-op-profile/{run.sh,probe.py,analyze.py}`；各操作目录内
  `analysis.json` 保留全部 rank/capture、kernel 顺序和 host API 区间。

完成后确认两个后续任务均为 `completed (exit=0)`。远端安装扩展已恢复为
运行前 SHA256 `aaf762e60005f108d500a73d6ff46962dda6f5db9ff1e321a09461879fec57c0`；
没有修改生产源码、系统环境或设备状态，没有提交或推送代码。
