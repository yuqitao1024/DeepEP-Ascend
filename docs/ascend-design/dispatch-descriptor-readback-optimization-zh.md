# Dispatch descriptor 单次回读

日期：2026-09-27。基线：`f569f9b`。问题定位见
[尾部瓶颈复核](dispatch-tail-bottleneck-analysis-zh.md)。

## 修改边界

原 `_reconcile_ascend_handle` 先调用 C++ generation 查询，内部 D2H 后与
committed descriptor 字节比较；返回 Python 后又调用 `descriptor.cpu()`
获取 fingerprint。两次读取的是同一份设备 descriptor。

新增 Ascend runtime 接口 `get_dispatch_handle_snapshot`，一次返回
`(generation, observed_bytes)`。C++ 在原有锁作用域内完成 tensor 身份、
dtype、长度和实际内容校验，只返回本次成功读取的字节；失败返回 `(0, [])`。
Python 直接将这些字节转为 fingerprint，不再做第二遍 D2H。

原 `get_dispatch_handle_generation` 保留并转调该实现，只返回 generation，
兼容原有调用方。设备 kernel、workspace、transport 协议和 Event 计时口径
均未变化。Python 与 extension 需要配套更新。

没有把预期的 committed bytes 当成实测快照，没有跳过完成边界，也没有
删除 cached/combine 在后续使用边界的 descriptor 检查。一次 reconcile
中的两次读取合为一次；不宣称整个 cached/combine 调用只会读取一次。

## 验证

本地 `tests/ascend/test_python_api.py` 与 `tests/platform`：48 passed、
3 skipped、19 subtests passed。新增回归在修改前因
`completion performed a second descriptor D2H` 失败，修改后通过。
既有 copied/dropped event、失败 completion、hybrid route 内容修改和
owner 检查继续通过。

旧 host probe 的五项失败已在干净的 `f569f9b` 源码副本上复现：其中两项
为过期源码断言，三项因 HCOMM/pybind stub 不完整而不能编译。这些不是
此次引入的失败。本次没有扩展修复那套 stub；新增 C++ 接口的边界直接在
真实 NPU 上验证，不把 host stub 失败记成实机通过。

新增 `tests/ascend/production/run_dispatch_handle_snapshot.py`，双卡检查：

- 正常完成的 generation、字节快照和 Python fingerprint 一致；
- completion 期间禁止调用旧 Python fingerprint 回读 helper；
- cloned tensor、错误长度、错误 dtype 被拒绝；
- 修改设备 descriptor 后拒绝，恢复后重新成功，旧快照不被覆盖；
- 新 generation 发布后拒绝旧 handle；
- async completion 前不发布新快照，copied raw event 完成后正常发布；
- wrapper completion hook 复用快照，销毁后的 native runtime 返回无效快照。

首次设备测试只在最后的销毁后检查失败：测试通过已置空的
`buffer.runtime` 访问 native 方法。修正测试为持有原 native runtime 引用
后两 rank 全部通过；生产实现和二进制未因此改动。

双卡小规模五操作 benchmark（32 tokens、hidden 7168、top-k 2、4 experts）
通过，2 warmups / 5 iterations。

8-rank 典型用例五操作通过；另有 BF16 sync、FP8 async，以及 FP8 同时开启
previous-event、async、allocate-on-comm-stream 三个 case 各五操作通过。这些功能运行
使用 2 warmups / 5 iterations，与下面正式 30/30 ABBA 分开。

## 性能验收协议

NPU8P，CANN/HCOMM 同树 9.3.0，设备 0–7，沿用原性能 selector。
典型输入：8 ranks、8192 tokens/rank、hidden 7168、top-k 8、256 experts、
num_sms 64，`ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0`。

A 为原实现，B 为单次回读实现。每次分别替换匹配的 Python wrapper 和
extension，在新进程中运行；无 profiling、30 warmups / 30 iterations，
按 A1/B1/B2/A2 顺序，至少三组，以普通 Dispatch 端到端收益稳定复现为
保留依据。其余四操作功能检查开启。逻辑带宽仍为八 rank 聚合值。

| 文件 | A SHA256 | B SHA256 |
| --- | --- | --- |
| extension | `7a9b1ba3a2083a18a61af03a9141a73a6970f84a81b5f5a84e5bcd10214e4349` | `4d0526f74d9517f7b6b89a8aab102a35896e177f672ec13cf7df5e1d2b9591d6` |
| elastic.py | `3e4f050250943ff21b860b6a375cc4a8a9c6309474c150b017cee3646751e119` | `a09e272218174cc9f418104c8c1732b51fa6e02c11c12588c78c1b73c1921409` |

## 任务与产物

远端工作区：`/home/pyptouser/yuqitao/deepep-official-simt.AaR0IW`。
构建与对照产物在 `.scratch/descriptor-readback/artifacts/{A,B}/`，
脚本在 `.scratch/descriptor-readback/`，结果在 `results/descriptor-readback/`。

- 构建：`task_20260926_193848_335544418397`，只重编译 host binding 并链接。
- 初次边界脚本失败：`task_20260927_013750_159335816`。
- 修正后双卡边界与 smoke：`task_20260927_013849_190525266`。
- 8-rank 典型、功能变体及 ABBA 第 1 组：`task_20260927_013953_225391377`。
- ABBA 第 2–3 组和尾部函数计时对照：`task_20260927_014417_4199510452`。

## ABBA 结果：保留

三组共 12 次独立运行全部五操作正确性通过，无卡死或结果写出前失败。
`abba1-A1` 在完整成功结果写出后出现已知 teardown SIGSEGV，validator
按既定规则识别；其余 11 次正常退出。两种实现都没有关闭业务校验。

每组每个实现汇总两次运行的 60 个正式样本；合并行每个实现为 180 个样本。
正的改善率表示延迟降低，逻辑带宽按相同逻辑字节除以合并 mean 计算。

| 组 | Dispatch A / ms | Dispatch B / ms | mean 改善 | p50 改善 | p95 改善 | A GB/s | B GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.309103 | 4.700843 | 11.4569% | 12.9966% | 9.6064% | 1466.555 | 1656.318 |
| 2 | 5.226679 | 4.756306 | 8.9995% | 8.9360% | 8.3198% | 1489.682 | 1637.004 |
| 3 | 5.342555 | 4.687341 | 12.2640% | 12.1066% | 13.5386% | 1457.372 | 1661.089 |
| 合并 | 5.292779 | 4.714830 | 10.9196% | 11.0111% | 10.5842% | 1471.078 | 1651.404 |

合并 Dispatch p50 为 5.223957 → 4.648744 ms，p95 为
5.989890 → 5.355910 ms；这里重新汇总原始样本，不平均各 run 的分位数。

五操作合并结果：

| 操作 | A mean / ms | B mean / ms | 改善 | A GB/s | B GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dispatch | 5.292779 | 4.714830 | 10.9196% | 1471.078 | 1651.404 |
| Expanded Dispatch | 16.172550 | 15.674772 | 3.0779% | 572.049 | 590.215 |
| Cached Dispatch | 65.074525 | 63.992272 | 1.6631% | 119.649 | 121.672 |
| Combine | 14.488508 | 14.010712 | 3.2978% | 752.403 | 778.062 |
| Reduced Combine | 14.982596 | 14.467190 | 3.4400% | 727.591 | 753.512 |

五操作各自的三组 mean 改善均为正。普通 Dispatch 的 mean/p50/p95
均稳定复现，符合“稳定收益才保留、不设百分比门槛”的要求，保留实现。
Cached/Combine 也会调用 reconcile，因此这一处共享改动会影响这些操作；
本次没有另行修改它们的 kernel 或额外删除其 use-boundary 检查。

## 无 profiler 函数边界对照

单独按 A/B/B/A 运行四次尾部诊断，每次 30 warmups / 30 captures，入口
HCCL 后显式设备同步。剔除每 rank 第 0 次后，每个实现共 464 个
rank/capture 样本；该分布不等于正式 benchmark 的逐 iteration max-rank
延迟，不与上述 ABBA 混合汇总。

| 区间 | A mean / ms | B mean / ms | A p50 / ms | B p50 / ms |
| --- | ---: | ---: | ---: | ---: |
| generation 查询 / 合并 snapshot 查询 | 0.277277 | 0.294943 | 0.260220 | 0.332635 |
| Python fingerprint 二次回读 | 0.359127 | 不再调用 | 0.372025 | 不再调用 |
| 整个 handle reconcile | 0.672154 | 0.327264 | 0.671225 | 0.367100 |
| C++ 返回后全部 Python 工作 | 0.694957 | 0.351349 | 0.695815 | 0.390605 |
| C++ dispatch 本体 | 3.763309 | 3.769080 | 3.802035 | 3.792890 |
| 最终显式设备同步 | 0.518340 | 0.519821 | 0.500480 | 0.519010 |

B 的每个计时样本都包含一次 `get_handle_snapshot`，不包含旧
`get_handle_generation` 或 `descriptor_fingerprint` 调用。handle reconcile
平均缩短约 0.345 ms（51.31%），C++ dispatch 与最终同步均值基本不变。
这支持收益来自减少尾部回读的判断；不将 0.345 ms 与正式 max-rank
ABBA 的 0.578 ms 改善机械相减、或把差额直接归因于某项同步优化。

本次没有重新采集完整 NPU trace，因此不宣称历史整个约 1.3–1.7 ms
尾部已消除。剩余 C++ completion/readback 和 runtime 同步仍存在。

## 复现与交付状态

- 12 份正式 JSON：`results/descriptor-readback/abba{1,2,3}-{A1,B1,B2,A2}.json`。
- 功能 JSON：同目录的 `smoke2.json`、`typical8.json`、`functional8.json`。
- 双卡边界日志：同目录 `boundaries2.log`，两 rank 都有
  `SNAPSHOT_BOUNDARIES_PASSED`。
- 尾部 JSON：`results/dispatch-tail/snapshot-{A-1,B-2,B-3,A-4}/rank*-measure.json`。
- 本地 `.scratch/descriptor-readback/` 保留这些 JSON 和摘要，
  `summarize.py` 校验输入、计时协议、样本数、正确性及 B 的调用路径，输出
  `abba-summary.json` 与 `tail-summary.json`。
- 所有节点任务已结束；实验目录恢复为匹配的 B wrapper + B extension，
  SHA256 与上表一致。生产代码没有留下 profiling instrumentation。
- 本轮仅合并 descriptor 回读，没有开始 overlap。

在实验目录按队列规则运行新的 ABBA 组号：

```bash
task-submit --device 0,1,2,3,4,5,6,7 --max-time 900 --run \
  'bash .scratch/descriptor-readback/abba.sh 4'
```

设备边界回归在配套 CANN/HCOMM 和 Python 环境下运行：

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  --module tests.ascend.production.run_dispatch_handle_snapshot
```
