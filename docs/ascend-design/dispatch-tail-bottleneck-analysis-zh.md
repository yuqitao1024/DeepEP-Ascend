# Normal Dispatch 尾部与单点瓶颈复核

日期：2026-09-26。代码基线：`f569f9b` 的默认自研通信路径。

## 决策

先处理 handle completion 的重复 descriptor 回读，不启动新的计算/通信 overlap
实现。实测仍有明确的单点优化候选，尚不满足“没有单点瓶颈再做 overlap”的条件。
这次只做诊断和记录，没有修改生产代码，也没有测得或宣称修复后的收益。

此前第 7 项把尾部归为未完全覆盖的 runtime/record 收尾，暂时关闭。新增
函数边界测量证明其中有可定位的 descriptor 回读成本；应重新打开该子项。
这不等于整个尾部都可消除，也不说明通信已经没有优化空间。

## 环境与测量边界

- NPU8P-ALT，`task-submit` 显式申请设备 0–7，8 ranks。
- CANN：`/data/disk2/cann_version/0916/use_cann/cann-9.3.0`；
  HCOMM 使用同树 `aarch64-linux`。
- 8192 tokens/rank、hidden 7168、top-k 8、256 experts、num_sms 64。
- `ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0`。
- 沿用已验证性能 selector，stable preflight，官方 SIMT 开关 OFF。
- 二进制 SHA256：
  `7a9b1ba3a2083a18a61af03a9141a73a6970f84a81b5f5a84e5bcd10214e4349`。
- 实测远端 Python wrapper、C++ elastic buffer、benchmark runtime 的 SHA256
  与本地当前版本分别一致；复用此前 ABBA 的 A 二进制，不重新混合构建产物。

三类数据分开报告：

1. 正式基线：原五操作 benchmark，30 warmups / 30 iterations，三次独立运行。
   原 correctness 与 Event 统计逻辑不变。
2. 阶段采样：两次原 `--profile-stages`，同样 30/30；只用于归因。
3. 边界诊断：相同输入，30 次 warmup；两次无 profiler 各 30 次调用，中间
   一次 CPU/NPU profiler 共 12 次 capture。统一剔除各 rank 第 0 次，分别为
   232、232、88 个 rank/capture 样本。入口 HCCL all-reduce 后显式设备同步。

边界诊断通过 Python proxy 和函数 wrapper 测量 C++ dispatch、handle 构造、
generation 校验、fingerprint、结果释放及最终同步。Profiler 模式增加命名
区间；无 profiler 模式只读 host 时钟。未绕过任何业务校验。

诊断改变了入口同步与观测方式，rank/capture 分布不能当作正式 benchmark
的逐 iteration max-rank 延迟，也不能把诊断 Event 均值拿来宣称性能改善。
诊断导出每 rank 结果后直接退出，避免已知的 profiler 后分布式 teardown；
其本身不是五操作全矩阵回归的替代品。

## 最新正式基线

| Run | Dispatch mean / ms | p50 / ms | p95 / ms | logical GB/s |
| --- | ---: | ---: | ---: | ---: |
| baseline1 | 5.431854 | 5.258765 | 6.234952 | 1433.413 |
| baseline2 | 5.340655 | 5.248415 | 6.023768 | 1457.890 |
| baseline3 | 5.243346 | 5.123630 | 6.033838 | 1484.947 |
| 合并 mean | 5.338618 | — | — | 1458.447 |

三个 run 样本数相同，合并 mean 汇总全部 90 个正式样本；带宽由相同逻辑
字节除以合并 mean 得到，不平均各 run 的 GB/s。带宽仍为八 rank 聚合逻辑
带宽，不是单链路 P2P 带宽。

三次基线和两次阶段采样均写出五操作通过的完整结果。baseline3 在完整通过
并写出结果后出现已知 teardown SIGSEGV，validator 识别为
`KNOWN_TEARDOWN_SIGSEGV_AFTER_VALID_RESULT`，不计入 kernel 性能时间。
其余诊断任务正常退出，八个 rank 的结果全部存在。

## 设备侧还有哪些大段

新完整 trace 的同一批 88 个 rank/capture 样本：

| 区间或 kernel | p50 / μs | 说明 |
| --- | ---: | --- |
| release payload 对应通用 dispatch stage | 944.016 | 最大稳定单一设备区间，含通信 service |
| epilogue hidden payload copy | 317.701 | D4 双缓冲后的 AICore 搬运部分 |
| epilogue parallel prefix | 234.928 | 本轮实际值，不套用旧约 175 μs |
| epilogue metadata | 170.539 | 此前候选无稳定收益，已撤回 |
| producer release VF | 168.511 | 命令构造，与上面的 service 区间分开 |
| epilogue copy outputs VF | 165.343 | scale/weight 等辅助输出，不包含 hidden copy |
| epilogue acquire | 151.145 | p95 451.026 μs；有 rank 时序影响 |
| producer record | 119.327 | 已非原先的 multi-ms 热点 |
| epilogue count experts | 82.468 | 并行计数保留后的结果 |
| producer prefix | 76.339 | small-chunk cache 保留后的新实测 |
| epilogue validate records | 46.140 | 并行校验保留后的结果 |

每个 capture 从首个到最后一个 Dispatch kernel 的跨度 p50 为 2773.330 μs，
其间 kernel gap 总和 p50 仅 12.972 μs，p95 19.786 μs。
不能再把旧阶段计时未覆盖的 VF 执行解释为毫秒级 launch idle。

两次独立阶段采样分别取 **service active 最大的同一个 rank**：

| Run / rank | service active / cycles | CQ drain / cycles | drain 占 active |
| --- | ---: | ---: | ---: |
| stages1 / 1 | 1239531 | 789329 | 63.68% |
| stages2 / 6 | 1240287 | 801154 | 64.59% |

CQ drain 包括轮询、CQ 处理和等待，并非纯网络传输。它嵌套在 flush/service
中，不能重复相加。通信完成等待仍值得关注，但不是本轮先改 overlap 的理由。

## “尾部 1.3 ms”究竟是什么

原指标的起点是最后一个 DeepEP Dispatch kernel 完成，终点是
`normal_dispatch_capture` 的 host record function 结束。中间包括 kernel
后的 C++ runtime 操作、Python 返回处理、返回对象释放和显式设备同步。
`profile.export_chrome_trace()` 在 capture 外执行，**导出 trace 的耗时不在
这个尾部里**。Profiler 的观测扰动可能影响数值，但不能把真实业务操作
笼统归为 profiler 开销。

本轮增加边界后，尾部 p50 为 **1.706838 ms**，mean 为 **1.834645 ms**，
p95 为 **3.209918 ms**。历史的约 1.3 ms 不是固定常数。以下为同一批
capture 的互斥分段；各段 mean 可相加，各段 p50/p95 不可相加：

| 分段 | mean / ms | p50 / ms |
| --- | ---: | ---: |
| 最后 Dispatch kernel → C++ dispatch 返回 | 0.709065 | 0.701121 |
| C++ 返回 → Python dispatch API 返回 | 0.626401 | 0.570785 |
| API 返回 → 返回对象释放结束 | 0.036825 | 0.033355 |
| 释放结束 → 最终同步开始（含 end Event record） | 0.092398 | 0.086865 |
| 最终显式设备同步 | 0.362079 | 0.331490 |
| 同步结束 → capture 结束 | 0.007877 | 0.006440 |

C++ 内第一段仍包含 stream completion、transport diagnostic 回读、count
bridge 回读、descriptor 发布、tensor narrow 及返回转换等。现有阶段采样
测到 count bridge 回读各 rank 为 16–350 μs；这是另一批采样，不能从上述
0.709 ms 直接相减来计算剩余开销。第一段尚未全部拆到具体 runtime API。

正式 benchmark 在 `operation()` 返回后记录 end Event，再调用设备同步。
所以 Python handle 确认发生在 end Event 之前，会影响现有 logical bandwidth。
而最终 host `synchronize()` 的全部墙钟耗时不能直接加到 Event 延迟上：
Event 只计到设备实际执行 end Event 的位置。

## 已确认的单点：同一 descriptor 连续两次回读

生产调用关系：

```text
ElasticBuffer.dispatch
  C++ runtime.dispatch 返回
  EPHandle 构造
  _reconcile_ascend_handle
    get_dispatch_handle_generation
      C++ copy_to_host(descriptor) + memcmp(committed snapshot)
    _ascend_descriptor_fingerprint
      descriptor.detach().cpu().reshape(-1).tolist()
```

第一遍用于验证 generation 对应的实际 descriptor 内容；第二遍又读相同
tensor，为 Python handle 保存 fingerprint。两遍都有业务目的，但获取
同一快照不应必然要求两次同步 D2H。

两次无 profiler 诊断的 p50：

| 段 | plain1 / ms | plain2 / ms |
| --- | ---: | ---: |
| C++ get handle generation（含第一遍 D2H） | 0.248 | 0.275 |
| Python fingerprint（含第二遍 D2H 和转换） | 0.327 | 0.404 |
| 整个 handle reconcile | 0.586 | 0.759 |
| C++ 返回后全部 Python 工作 | 0.610 | 0.781 |
| 返回对象释放 | 0.017 | 0.018 |
| 最终显式设备同步 | 0.462 | 0.495 |

这些数据证明约 0.6–0.8 ms 的 handle 收尾确实在无 profiler 时也存在，
主要不是 EPHandle 对象构造（约 3 μs）或对象释放。不同实验条件的 p50
不能相减推导优化收益；合并回读的收益仍需真正实现后的 ABBA 验证。

## 下一步实施边界

优先候选：让 C++ 一次完成 descriptor D2H、内容校验，然后把 **这次实际
观察到的快照和 generation 一起返回**，Python 用该快照建立 fingerprint。

必须保留：

- owner、tensor identity/size、generation 和 committed bytes 校验；
- descriptor 被修改、旧 handle、跨 buffer handle 的拒绝语义；
- cached/combine 的后续变更检查；
- async completion 的发布边界；不能在设备完成前使用待提交快照。

不能直接删除 fingerprint，不能只返回缓存的“预期字节”冒充实际回读结果，
也不能把当前快照永久缓存后跳过后续使用边界的校验。

先做上述单点，完成边界回归和至少三组无 profiler ABBA，稳定收益才保留。
之后重新评估 C++ 收尾与通信等待；没有更明确单点时，再设计最小 chunk
overlap。暂不重开此前收益不稳定的 metadata/prefix 候选，也不宣称两次
回读合并后就能隐藏整个 1.7 ms 尾部。

## 任务与原始数据

远端工作区：`/home/pyptouser/yuqitao/deepep-official-simt.AaR0IW`。

- `task_20260926_192012_330583212881`：baseline1、stages1。
- `task_20260926_192215_33135289640`：plain1、trace1、plain2。
- `task_20260926_192423_332437216438`：baseline2、stages2、baseline3。
- 正式结果：`results/dispatch-tail/baseline{1,2,3}.json`。
- 阶段结果：`results/dispatch-tail/stages{1,2}.json`。
- 边界结果：`results/dispatch-tail/{plain1,plain2,trace1}/rank*-measure.json`。
- 完整 trace：`results/dispatch-tail/trace1/rank{0..7}.json`。
- 日志：远端 `results/dispatch-tail/*.log`。
- 诊断脚本：`.scratch/dispatch-tail/{run.sh,probe.py}`；本地解析脚本
  `.scratch/dispatch-tail/summarize.py`，输出 `summary.json`，含逐 capture 分段。

复现使用新的 name 防止覆盖结果；仍需按节点队列规则、同一时刻只运行一个
自己的设备任务。例如在上述工作区：

```bash
task-submit --device 0,1,2,3,4,5,6,7 --max-time 900 --run \
  'bash .scratch/dispatch-tail/run.sh plain plain-new'
```

本地保留正式 JSON、诊断 JSON 和 trace 副本。
