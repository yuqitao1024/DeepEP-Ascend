# 官方 SIMT 通信执行器 ABBA 实验

日期：2026-09-26。基线代码：`080912a`（生产优化来自 `b558d70`）。

## 接口核对

此前对比见 [官方 SIMT 数据面对比](asc-comm-official-simt-comparison-zh.md)。
本轮核对实际安装的 CANN 9.3.0，公开 `hcomm_simt.h` 与官方仓
`9280cf6ea12a0cd7d00acc0709fe8834c669c3e1` 对应文件 SHA256 相同：
`85a795b5d8818bbad99b5079417eab7a5e379b5ac04fc9431dfe0c34c4018cba`。
`hcomm_simt_urma.h` 也通过逐字节比较。

| 项目设备接口/语义 | 官方接口或适配方式 |
| --- | --- |
| `put`、`put_on_channel` | `WriteNbi`，保留逻辑地址到远端注册地址的转换 |
| `put_value` | `WriteValueNbi<uint64_t>` |
| `remote_add_release` | `AtomicFAA<uint64_t>`，保留发布顺序 |
| signal set | `WriteValueNbi<uint64_t>` 写原协议的 signal 槽 |
| signal add/increment | `AtomicFAA<uint64_t>` 写原协议的目标 |
| `flush` | 对所有 channel 调用 `Drain` |
| `flush_async` / `wait` | 保留项目 request、consumed count、generation；完成来源改为官方 Drain |
| `device_barrier` | 保留 scale-out → scale-up 协议，使用官方 FAA 和 Drain |
| 本地 load/store/fence、signal 读取等待 | 保留项目 SIMT 内存操作；并非 Hcomm 通信原语 |
| `get` | 官方有 `ReadNbi`；当前 staged facade 的 get 是空实现，生产命令集没有 Get，本轮不扩展功能 |
| Notify / CAS | 官方有 `WriteWithNotifyNbi` / `AtomicCAS`，本轮生产命令集未使用 |

官方覆盖当前使用的底层通信原语，但没有直接等价的项目 request/barrier API。
本轮切换的是设备通信执行器，不是删除整个项目传输抽象。

## 宏及实验边界

`DEEP_EP_ASCEND_OFFICIAL_SIMT` 是编译开关，默认 `OFF` / `0`。

- A：关闭，完整保留现有 AICore 自研 WQE/SQ/CQ 路径。
- B：开启，由 service 调用单 lane SIMT VF，使用官方 Hcomm 执行整批命令。
- producer、GM command queue、route/copy/reduce、host 资源、协议均保留。
- 全部 service 调用统一切换，禁止同一 channel 混用两套 SQ tail 规则。
- 每个 service 调用由单 lane 持有 Hcomm 对象并提交/等待。
- 第一版全部立即 commit，避免没有独立 Commit 时遗留未发布 WQE。
- 独立 barrier 的原子 fetch-result 使用已注册 sync window；有独立 fetch
  buffer 的 context 优先使用该 buffer，与旧执行器规则一致。
- 本轮使用无 profile 端到端计时。官方分支尚未实现旧执行器全部细分 profile
  计数器，不能拿这些计数器直接作两套 backend 的关键路径对比。

CMake：`-DDEEP_EP_ASCEND_OFFICIAL_SIMT=ON/OFF`。
setup.py：`DEEP_EP_ASCEND_OFFICIAL_SIMT=1/0` 后重新编译 extension。
只使用选定 CANN 安装树中的官方头文件及配套库，不复制修改官方实现。

结果只回答替换通信执行层的效果，不能外推为 producer 直接提交、移除
command/service 层后的性能。是否继续做直接提交需要独立实验。

## 环境与验证计划

- NPU8P，经 NPU8P-ALT 登录。
- CANN：`/data/disk2/cann_version/0916/use_cann/cann-9.3.0`。
- HCOMM：同树 `aarch64-linux`。
- Python：`/home/miniconda3/envs/py310/bin/python`。
- 独立实验目录：`/home/pyptouser/yuqitao/deepep-official-simt.AaR0IW`。
- 全部编译经 `task-submit --no-device`；设备用例经显式卡号分配。
- 顺序：双卡小规模 correctness → 双卡典型 → 8-rank correctness → 三组 ABBA。
- 典型：8192 tokens/rank、hidden 7168、top-k 8、256 experts，FP8、alignment 128。
- 沿用之前的 `num_sms=64` 和性能 selector；两边相同。
- 无 profile，2 warmup、30 iterations；取最慢 rank 延迟，逻辑字节跨 rank 求和。
- 每组按 A1、B1、B2、A2 顺序；保存二进制 SHA256、原始 JSON、日志和任务 ID。
- 所有功能检查开启；已知 teardown SIGSEGV 仅在完整成功结果输出后单独识别。

## 当前验证记录

- 主机侧传输测试：23 passed、3 skipped；平台源码边界测试：7 passed；
  transport contract 测试：3 passed。开关默认值及 opt-in 值验证通过。
- 官方执行器编译通过：`task_20260926_174606_29179072974`。
- 首次双卡小规模：`task_20260926_174814_29286479092`，独立 barrier 在
  官方 FAA 调用前因 adapter 缺少 fetch-result fallback 返回 invalid_address。
  已按原执行器规则补齐。本次不构成功能通过或官方 API 失败证据。
- 修复后编译：`task_20260926_175111_293549913718`，通过。
- 双卡小规模（32 tokens、top-k 2、4 experts）：
  `task_20260926_175203_294358623222`，五个操作正确性通过，1 warmup、2 iterations。
- 双卡典型：`task_20260926_175242_29455478363`，五个操作正确性通过，2 warmup、5 iterations。
- 8-rank 典型：`task_20260926_175331_29479045164`，五个操作正确性通过，2 warmup、5 iterations。
- A 基线重新编译：`task_20260926_175438_295219219260`，通过。
- A SHA256：`7a9b1ba3a2083a18a61af03a9141a73a6970f84a81b5f5a84e5bcd10214e4349`。
- B SHA256：`b2cb11a58c01e51d436811801dded8cd8323ab3d420f8e1945f926c548b5996c`。
- ABBA 第一组：`task_20260926_175557_296723921827`，四次均通过。
- ABBA 第二组：`task_20260926_180038_300710225593`，四次均通过。
- ABBA 第三组：`task_20260926_180443_302386527436`，四次均通过。
- 12 次 ABBA 运行均通过五操作正确性检查，未出现超时或卡死。
  功能小规模、双卡典型、8-rank 典型三次运行也都通过。
- `abba2-B1`、`abba3-B1` 在完整成功结果输出后出现已知 teardown SIGSEGV。
  原有 validator 验证全部 case/操作成功，且异常只发生在结果写出之后；
  按已有约定单独记录，不将其当作 kernel 功能错误或计入性能时间。
- 最终执行器源码只经过格式化；用相同 clang-format 格式化实测源码后，
  与本地待交付源码逐字节相同。

## 无 profile ABBA

正的延迟改善率代表 B 更快，计算为 `(A_ms - B_ms) / A_ms`。
每组每个 variant 合并两次运行的 60 个样本；带宽使用相同逻辑字节除以该均值，
不是挑选最快样本。汇总前检查 workload、计时协议、设备配置一致，
每次五个操作均通过且每操作包含 8 个 rank、30 个计时样本。

| 组 | Dispatch A ms | Dispatch B ms | 延迟改善 | A GB/s | B GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.257775 | 4.877527 | +7.2321% | 1480.872 | 1596.319 |
| 2 | 5.230958 | 5.322092 | -1.7422% | 1488.463 | 1462.976 |
| 3 | 5.258119 | 5.386423 | -2.4401% | 1480.775 | 1445.503 |
| 合并 | 5.248951 | 5.195347 | +1.0212% | 1483.361 | 1498.666 |

合并结果每个 backend、每个操作包含 6 次运行、180 个计时样本：

| 操作 | A ms | B ms | 延迟改善 | A GB/s | B GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dispatch | 5.248951 | 5.195347 | +1.0212% | 1483.361 | 1498.666 |
| Expanded Dispatch | 16.200135 | 16.304024 | -0.6413% | 571.075 | 567.436 |
| Cached Dispatch | 65.285698 | 65.353180 | -0.1034% | 119.262 | 119.139 |
| Combine | 14.509180 | 14.756971 | -1.7078% | 751.331 | 738.715 |
| Reduced Combine | 15.057893 | 15.134654 | -0.5098% | 723.953 | 720.281 |

各组延迟改善率：

| 操作 | 组 1 | 组 2 | 组 3 |
| --- | ---: | ---: | ---: |
| Dispatch | +7.2321% | -1.7422% | -2.4401% |
| Expanded Dispatch | -0.2882% | -0.6517% | -0.9884% |
| Cached Dispatch | -0.0050% | -0.0468% | -0.2589% |
| Combine | -2.3884% | -1.0828% | -1.6466% |
| Reduced Combine | +0.4195% | -0.6173% | -1.3643% |

这里的 GB/s 是 8-rank 聚合逻辑带宽，不是单卡或单条 P2P 链路实测带宽。

## 首轮结论及保留方式

官方 API 能支撑本轮典型用例，但单纯替换通信执行层没有稳定性能收益。
Dispatch 的合并均值快约 1.02%，主要来自第一组；第二、三组都变慢，
不能按“稳定复现即可保留”的标准作为默认性能优化。
Combine 三组均慢，合并延迟增加约 1.71%。其余三个操作也没有稳定正收益。

保留 `DEEP_EP_ASCEND_OFFICIAL_SIMT=OFF` 为默认，自研 API 和 URMA 实现不删除。
官方路径保留为显式 opt-in 实验分支，便于进一步对照。

这不是对官方 API 性能上限的判断：当前 B 仍保留 GM command queue，
并增加 service → SIMT VF 边界，尚未测 producer 直接提交或延迟批量 commit。
若继续，应单独设计 channel 的唯一提交者，并分别验证直接提交和批量发布，
不能把这次数据当作那两种设计的结论。细分 stage profile 和更广的
拓扑/模式矩阵也不在本轮验收范围内。

## 补测：独立运行与 warmup 对照

用户要求增加轮次，并询问是否做了 warmup。每次独立 A1/B1/B2/A2 运行中，
每个操作都先预热，再计时；原协议是 2 warmup + 30 iterations，
warmup 不进入计时均值。补测仍使用上文两份二进制，启动前 SHA256 核对一致，
未修改设备代码或编译配置。

- 组 4：`task_20260926_181241_306873616977`，2 warmup，四次均通过。
- 组 5：`task_20260926_181603_308229718407`，2 warmup，四次均通过。
- 组 6：`task_20260926_181928_309512027384`，改为 30 warmup，独立预热对照，四次均通过。
  其余 workload、计时次数、A/B 顺序保持不变，不与组 1–5 混合汇总。
- 补测 12 次运行均通过五操作正确性检查，未出现超时或卡死。
  `abba4-A1`、`abba4-B1` 在成功结果写出后出现已知 teardown SIGSEGV，
  原有 validator 核验通过，按前述约定单独记录。

| 组 | Dispatch A ms | Dispatch B ms | 延迟改善 | A GB/s | B GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 5.209904 | 5.275234 | -1.2540% | 1494.479 | 1475.970 |
| 5 | 5.318709 | 5.238341 | +1.5110% | 1463.906 | 1486.366 |
| 1–5 合并 | 5.255093 | 5.219923 | +0.6692% | 1481.627 | 1491.610 |

组 1–5 每个 backend、每个操作包含 10 次运行、300 个计时样本：

| 操作 | A ms | B ms | 延迟改善 | A GB/s | B GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dispatch | 5.255093 | 5.219923 | +0.6692% | 1481.627 | 1491.610 |
| Expanded Dispatch | 16.180793 | 16.278103 | -0.6014% | 571.757 | 568.339 |
| Cached Dispatch | 65.294532 | 65.316078 | -0.0330% | 119.246 | 119.206 |
| Combine | 14.527139 | 14.719784 | -1.3261% | 750.403 | 740.582 |
| Reduced Combine | 15.022201 | 15.093192 | -0.4726% | 725.673 | 722.260 |

Dispatch 在相同 2-warmup 协议下仍有正有负，首组 +7.23% 尚未复现。

30-warmup 对照（组 6）四份 JSON 的 `timing_protocol.warmups` 均为 30，
每个 backend、每个操作仍只汇总两次运行的 60 个正式计时样本：

| 操作 | A ms | B ms | 延迟改善 | A GB/s | B GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dispatch | 5.317052 | 5.387165 | -1.3187% | 1464.362 | 1445.304 |
| Expanded Dispatch | 16.299656 | 16.348816 | -0.3016% | 567.588 | 565.881 |
| Cached Dispatch | 65.118884 | 65.373854 | -0.3915% | 119.567 | 119.101 |
| Combine | 14.573521 | 14.756236 | -1.2537% | 748.014 | 738.752 |
| Reduced Combine | 14.946039 | 15.040052 | -0.6290% | 729.371 | 724.812 |

补测结论：2-warmup 五组的 Dispatch 合并均值仅快 0.67%，且方向不一致；
30-warmup 的一组对照中，Dispatch 慢 1.32%，五个操作也都没有正收益。
增加 warmup 后没有观察到收益，但单组对照不足以完全排除所有预热相关影响。
当前证据仍不支持默认切换；保留宏开关及自研路径，默认 `OFF`。
本次只增加测试和记录，没有修改设备代码。

## 原始数据与复现

远端实验目录下：

- `artifacts/A/`、`artifacts/B/`：两份不可混用的编译产物，SHA256 见上文。
- `results/abba{1,2,3,4,5,6}-{A1,B1,B2,A2}.json`：24 份原始计时和各 rank 汇总，
  组 1–5 是 2 warmup，组 6 是 30 warmup。
- 同名 `.log` 及 `abba{1,2,3,4,5,6}-driver.log`：功能结果和每次加载的二进制 SHA256。
- `results/smoke2-fixed.json`、`typical2.json`、`typical8.json`：功能验证记录。
- `.scratch/official-simt/build.sh`、`run.sh`、`abba.sh`：实测构建、环境及运行命令。
- `.scratch/official-simt/abba-warm30.sh`：30 次 warmup 的单独对照脚本。
- `reviewed-source.tar.gz`：整理后的源码及实验脚本快照；官方执行器与实测版仅有格式化差异。

本地 `.scratch/official-simt/` 保存 24 份原始 ABBA JSON 和 `summarize.py`。
汇总命令：

```bash
python3 .scratch/official-simt/summarize.py .scratch/official-simt 1 2 3 4 5
python3 .scratch/official-simt/summarize.py .scratch/official-simt 6
```

在实验目录内按节点队列规则逐组运行（上一任务结束后再提交下一任务）：

```bash
task-submit --device 0,1,2,3,4,5,6,7 --max-time 900 --run \
  'bash .scratch/official-simt/abba.sh 1'
```

脚本拒绝覆盖已存在的结果；复测使用新的组号或独立实验目录。不得在
benchmark 进程运行时替换 extension，也不得把 A/B 混在同一进程或 channel 上。
