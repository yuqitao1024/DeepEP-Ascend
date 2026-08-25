# Ascend EPv2 耗时统计与 Dispatch 阶段归因

**状态：** P3.0 profiling 候选实现与 EP8 representative 数据已完成；
profiling 代码仍位于 `opt/ascend-p3-overlap`，尚未合入 `main`

## 1. 文档目的

本文说明 Ascend EPv2 当前使用的两套计时口径，并回答下面几个容易混淆的
问题：

- 正式性能表中的平均耗时是怎样计算出来的；
- P3.0 如何统计一个 kernel stage 的耗时；
- dispatch 的 `producer` 到底包含哪些操作；
- 按目标 rank 统计 token 和按本地 expert 统计 token 有什么区别；
- token 偏移、FP8 scale 搬运和 SQE/WQE 构造分别属于哪个阶段；
- 为什么不能把 P3.0 的阶段耗时直接相加，当作算子端到端耗时。

本文使用的 P3.0 实现基线是 commit
`17dd7ed9f23a9f5fdadfd3137a356d0ca9319e3f`。最终启用 profiling 的 EP8
结果来自 TaskQueue 任务 `task_20260824_225210_166672421792`。后续 P3
提交调整了 profiling 关闭路径和通信生命周期，但没有改变本文讨论的
`producer = control + group + prefix + record` 归类原则。

## 2. 先看结论

Dispatch 的完整数据流可以概括为：

```text
输入 x、scale、top-k index/weight
                 |
                 v
  +---------------------------------------------+
  | producer                                    |
  |                                             |
  | control -> group -> prefix -> record        |
  |              |         |         |          |
  |              |         |         +-- 搬 hidden、scale、top-k、metadata
  |              |         +------------ 算目标 rank staging slot 偏移
  |              +---------------------- 按目标 rank 分组并统计 record
  +---------------------------------------------+
                 |
                 v
  producer_release / publication
  发布 transport.put、flush、count、generation、release
                 |
                 v
  transport service
  解析命令 -> 构造 URMA WQE/SQE -> post SQ -> 等待 CQ
                 |
                 v
  consumer epilogue
  acquire/validate -> 按本地 expert 计数 -> expert prefix
                   -> metadata -> 输出搬运 -> complete
```

因此，对几个具体问题的回答是：

| 操作 | 是否计入 `producer` | 实际阶段 |
| --- | --- | --- |
| 按目标 rank 统计需要发送的 token record | 是 | `producer_group` |
| 按最终本地 expert 统计 token | 否 | `epilogue_expert_count` |
| 计算目标 rank staging slot 偏移 | 是 | `producer_prefix` |
| 计算最终 expert 输出偏移 | 否 | `epilogue_expert_prefix` |
| 搬运 hidden payload | 是 | `producer_record` |
| 搬运 FP8 scale factor | 是 | `producer_record` |
| 搬运 top-k index、weight 和源 token metadata | 是 | `producer_record` |
| 发布逻辑 `transport.put` 命令 | 否 | `producer_release` / `publication` |
| 构造并提交真实 URMA WQE/SQE | 否 | `service_submit` |
| 等待 CQ completion | 否 | `cq_wait` |

最关键的边界是：`producer` 包含 payload record 的准备和写入，但不包含
真实 SQE/WQE 的构造与提交，也不包含接收侧最终的 expert token 统计。

## 3. 正式性能耗时怎样统计

### 3.1 单个 Rank 的一次样本

正式性能数据由 `tests/ascend/benchmark/timing.py` 中的 `NpuEventTimer`
统计。每次样本的顺序是：

```text
rank 间 barrier                 不在 Event 区间内
torch.npu.synchronize()
记录 start NPU event
调用完整 operation             在 Event 区间内
记录 end NPU event
torch.npu.synchronize()
start.elapsed_time(end)
```

这里同时记录 device event 时间和 host wall time。性能表中的主要耗时与
逻辑带宽使用 device event 时间；wall time 用于诊断 Python、host launch
和同步开销，不作为设备带宽的分母。

预热和正式采样由 benchmark 参数控制。当前 representative 正式口径是
30 次 warmup 加 30 次 measured iteration。每次 warmup 和 measured
iteration 前都有 rank 间 barrier，但 barrier 在 start event 之前，所以不计入
该次 device sample。

### 3.2 八卡样本怎样聚合

一次 EP 操作只有等最慢的 rank 完成后，整个并行操作才算完成。因此第
`i` 次样本取八个 rank 中的最大 device event 时间：

\[
T_i = \max_{0 \le r < 8} T_{i,r}
\]

然后对全部 measured samples 计算：

\[
T_{\text{mean}} = \frac{1}{N}\sum_{i=1}^{N}T_i
\]

P50 和 P95 也从这组 `T_i` 样本计算。也就是说，表中的 mean 不是“先算每卡
平均值，再对八卡求平均”，而是“每轮先取最慢 rank，再对多轮求平均”。

逻辑带宽为：

\[
BW_{\text{logical}}
=
\frac{\sum_r B_{\text{logical},r}}
     {T_{\text{mean,seconds}} \times 10^9}
\quad \text{GB/s}
\]

分子是八个 rank 的逻辑通信字节数之和。它用于跨后端统一比较，不等同于
链路计数器看到的物理线速，也没有扣除协议头、对齐填充或重传。

## 4. P3.0 阶段耗时怎样统计

### 4.1 Profiling launch 与正式样本相互独立

开启 `--profile-stages` 后，benchmark 先完成正常 warmup 和 measured
samples，再额外执行一次带 profiling 的 operation：

```text
rank 间 barrier
清空 stage profile buffer
执行一次完整 operation
torch.npu.synchronize()
读取 stage profile buffer
```

因此，阶段数据不是从某一个 30 次正式样本中切分出来的。正式 Event
耗时和阶段 cycle 数据来自不同 launch，不能要求两者逐 cycle 对齐。

### 4.2 一个 stage 的跨度

每个参与 stage 的 block 在 device 上调用 `AscendC::GetSystemCycle()`，
记录自己的 `start` 和 `end`。多 block stage 的耗时不是各 block 耗时之和，
而是从最早 block 启动到最晚 block 结束的跨度：

\[
T_{\text{stage,cycles}}
=
\max_b(t_{\text{end},b})
-
\min_b(t_{\text{start},b})
\]

这个定义同时覆盖实际工作时间、block 启动偏斜以及同一 stage 内的调度
尾部。它回答的是“这个 stage 占据了多长的设备时间窗口”，不是“所有 block
一共消耗了多少计算资源”。

Ascend 950PR/950DT 的 system counter 按 1 GHz 换算：

\[
T_{\text{ms}} = \frac{T_{\text{cycles}}}{10^6}
\]

即 1,000 cycles 为 1 微秒，1,000,000 cycles 为 1 毫秒。

### 4.3 Rank 间阶段聚合

每个 rank 先在本地计算自己的 phase。八卡报告再对每个 phase 分别取最大
值：

\[
T_{\text{phase}}
=
\max_r T_{\text{phase},r}
\]

这意味着报告里的 `producer` 最大值可能来自 rank 2，`network` 最大值可能
来自 rank 5，`consumer` 最大值又可能来自 rank 7。它是一组“各阶段最坏值”，
不一定对应任意一张卡上的真实完整时间线。

## 5. Dispatch producer 的精确定义

P3.0 把 dispatch stage 编号 1 到 4 归入 `producer`：

\[
T_{\text{producer}}
= T_{\text{control}}
+ T_{\text{group}}
+ T_{\text{prefix}}
+ T_{\text{record}}
\]

`producer_release` 是 stage 5，不计入 `producer`。

### 5.1 `producer_control`

`direct_dispatch_producer_control_vf` 只由 thread 0 执行，主要负责：

- 将 operation status 和 local count 清零；
- 清空 rank/expert error scratch；
- 校验 topology、world size、workspace 布局和 bitmap 容量；
- 校验 receive/staging shard 的数量、大小和 token stride；
- 发现非法布局时发布 producer protocol error。

它不扫描 top-k，不统计 expert token，也不搬运 payload。

### 5.2 `producer_group`

`direct_dispatch_producer_group_vf` 以一个 top-k subgroup 处理一个 token：

1. 每个有效 lane 读取一个 `topk_indices[token, lane]`；
2. 根据 `destination_rank = expert / num_local_experts` 映射目标 rank；
3. 使用 ballot、shuffle 和 owner election 合并相同目标 rank；
4. 每个唯一目标 rank 只选出一个 owner lane；
5. 写入 token/rank owner 表，并累计每个 tile 发往各 rank 的 record 数。

这里统计的是“需要向某个目标 rank 发送几份 token record”，不是“该 rank
上的每个 expert 分别收到几个 token”。一个 token 的 hidden 只需要向同一
目标 rank 发送一次，接收端再根据完整 top-k metadata 分发给本地 experts。

例如 256 个 experts 均匀分布在 8 个 rank 上，每个 rank 有 32 个 experts。
某个 token 的 top-k 为：

```text
expert:  [1, 7, 33, 40, 65, -1, 130, 135]
rank:    [0, 0,  1,  1,  2,  -,   4,   4]
```

有效 expert 有 7 个，但唯一目标 rank 只有 `{0, 1, 2, 4}`。所以 producer
为这个 token 构造 4 份通信 record，而不是 7 份。`-1` 表示 masked top-k
lane，不参与分组和计数。

### 5.3 `producer_prefix`

`direct_dispatch_producer_prefix_vf` 为每个目标 rank 扫描 tile count，执行
exclusive prefix：

\[
O_{t,r} = \sum_{j < t} C_{j,r}
\]

其中 `C[t,r]` 是 tile `t` 发往 rank `r` 的 record 数，`O[t,r]` 是该 tile
在目标 rank staging shard 中的起始 slot。全部 tile 的计数之和为：

\[
C_r = \sum_t C_{t,r}
\]

`C_r` 写入 destination count，并用于检查是否超过 shard capacity。

所以“计算 token 偏移”确实属于 producer，但这里算的是目标 rank staging
record 的 offset。接收之后按 expert 对齐的最终输出 offset 由
`epilogue_expert_prefix` 另行计算，不属于 producer。

### 5.4 `producer_record`

`direct_dispatch_producer_record_vf` 根据 group owner 和 prefix 结果确定最终
destination slot，然后调用 record writer 写入本地 receive shard 或对应远端
rank 的 staging shard。一个 record 包含：

- hidden payload；
- FP8 dispatch 的 scale factors；
- 完整 top-k expert indices；
- top-k routing weights；
- source token index、master lane 和 destination metadata。

Hidden 主体可以走 AI Vector `DataCopy`，SIMT 路径处理剩余尾部以及 scale、
top-k 和 metadata。BF16 dispatch 没有 scale factor，因此相应 copy 长度为
零；FP8 dispatch 才实际搬运 scale。

现有 P3.0 representative dispatch 中，producer 的主要耗时在 record，而
不是 top-k grouping：

| Producer stage | 耗时（ms） | Producer 占比 |
| --- | ---: | ---: |
| `producer_control` | 0.007 | 0.2% |
| `producer_group` | 0.029 | 0.6% |
| `producer_prefix` | 0.575 | 12.9% |
| `producer_record` | 3.860 | 86.4% |
| **合计** | **4.470** | **100.0%** |

各行经过毫秒级四舍五入，显示值相加可能与原始 cycle 合计有微小差异。

## 6. 为什么 SQE/WQE 不属于 producer

`direct_dispatch_producer_release_vf` 在 record 完成后才执行。它遍历目标
rank，调用 `transport.put()` 发布 payload 命令，然后执行 flush 或
`flush_async()`。最终 chunk 还会发布远端 count、generation、release signal，
并执行 device barrier。

这里的 `transport.put()` 发布的是仓库自定义的逻辑 transport command，
还不是网卡能够直接消费的 URMA SQE/WQE。Device transport service 随后：

1. 从 command queue 读取逻辑命令；
2. 解析 channel、local buffer 和 remote address；
3. 调用 `urma::make_write()` 构造 write request，也就是实际 WQE/SQE 内容；
4. `post_request()` 将请求提交到 SQ；
5. 轮询或等待 CQ completion。

所以“拼接 SQE”如果指真实 URMA WQE/SQE 构造，应归入
`service_submit`，而不是 producer。若只是向 device command queue 追加一条
逻辑 `put`，则属于 `producer_release/publication`。

P3.0 对 release 和 transport service 的拆分公式是：

\[
T_{\text{service}}
= t_{\text{service,end}} - t_{\text{service,start}}
\]

\[
T_{\text{publication}}
= \max(T_{\text{release}} - T_{\text{service}}, 0)
\]

\[
T_{\text{service-submit}}
= T_{\text{service}} - T_{\text{cq-wait}}
\]

这是一种阶段归因，不是三段完全独立、互不重叠的 Event 计时。

## 7. 最终 expert token 统计在哪里

数据到达接收 rank，并完成 acquire 和 validation 后，
`direct_dispatch_epilogue_count_experts_vf` 才执行最终 expert token 统计。
它遍历有效的 `(source_rank, source_slot)` records，读取 record 中的完整 top-k
indices，只对当前 rank 的 local experts 累计 tile histogram。

后续 `epilogue_expert_prefix` 将 tile histogram 汇总成每个 expert 的 count，
并计算对齐后的 expert prefix。然后 metadata 和 copy 阶段把 records 放入最终
expert-major 输出布局。

这部分归入 `consumer_compute`：

```text
consumer_compute
  = epilogue_expert_count
  + epilogue_expert_prefix
  + epilogue_metadata
  + epilogue_copy
```

因此 producer 只决定“向哪些 rank 各发送一份 record、放在 rank shard 的
哪个 slot”；consumer 才决定“这个 record 最终进入哪些本地 expert、位于
expert 输出的哪个 slot”。

## 8. Representative Dispatch 阶段数据

P3.0 EP8 representative case 使用 8,192 tokens/rank、hidden 7,168、top-k
8、256 experts、FP8 dispatch、72 data blocks、30 warmups 和 30 measured
iterations。Dispatch 的正式 Event mean 是 36.176 ms，阶段 profiling 为：

| Phase | 耗时（ms） | 主要内容 |
| --- | ---: | --- |
| `producer` | 4.470 | control、rank grouping、rank prefix、record packing |
| `publication` | 0.163 | release 中逻辑命令和控制发布归因 |
| `service_submit` | 5.734 | transport service 解析、构造和提交 WQE/SQE |
| `cq_wait` | 0.851 | 等待 CQ completion |
| `consumer_wait` | 0.605 | 等待远端 release/acquire 条件 |
| `consumer_compute` | 8.179 | expert count/prefix、metadata、output copy |
| `epilogue` | 0.003 | 最终 completion publication |
| **正式 Event mean** | **36.176** | 完整 operation 的 30 次样本平均 |

通信相关的合并诊断值是：

\[
T_{\text{network}}
= 0.163 + 5.734 + 0.851
= 6.748\ \text{ms}
\]

consumer 合并诊断值是：

\[
T_{\text{consumer}}
= 0.605 + 8.179 + 0.003
= 8.787\ \text{ms}
\]

同一次 P3.0 profile 中五个 operation 的汇总如下：

| Operation | Event mean（ms） | Producer（ms） | Network（ms） | Consumer（ms） | 逻辑带宽（GB/s） |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dispatch` | 36.176 | 4.470 | 6.748 | 8.787 | 215.231 |
| `expanded_dispatch` | 38.347 | 4.484 | 6.642 | 10.300 | 241.257 |
| `cached_dispatch` | 85.689 | 50.759 | 10.935 | 8.813 | 90.865 |
| `combine` | 141.098 | 47.147 | 60.302 | 18.424 | 77.260 |
| `reduced_combine` | 168.954 | 74.430 | 62.896 | 18.419 | 64.522 |

这些数据说明普通 dispatch 的 producer 由 record packing 主导，而 combine
和 reduced combine 的主要问题已经转向 producer planning/record 构造及通信
服务。它们不能仅凭这张表就判定已经实现了通信计算 overlap。

## 9. 统计结果的正确使用方式

### 9.1 不要把 phase 简单相加当作端到端耗时

Dispatch 表中列出的 phase 相加约为 20.0 ms，小于 36.176 ms Event mean。
这不是统计错误，原因包括：

- profiling 与正式 Event mean 来自不同 launch；
- stage span 只覆盖显式打点的 kernel 工作窗口；
- Event 区间还包含 stage 之间的设备调度、stream 空隙和未归类控制路径；
- 每个 phase 在八卡间独立取最大值，最大值可能来自不同 rank；
- release 与 transport service 存在并发，publication 是通过差值归因得到的。

阶段数据适合回答“时间主要花在哪一类工作”，不适合作为正式 latency 的
替代值。

### 9.2 不要用 phase 占比直接宣称 overlap 收益

把 producer、network、consumer 三者的和除以最大项，可以得到理想化的
overlap ceiling：

\[
S_{\text{ceiling}}
=
\frac{T_p + T_n + T_c}
     {\max(T_p,T_n,T_c)}
\]

它只表示“假设三个阶段能够完美重叠”的方向性上限。由于三个分量可能来自
不同 rank，且存在协议依赖、buffer 生命周期和 AI Vector 资源竞争，这不是
可兑现的预测，更不是已经测得的 speedup。

### 9.3 跨 H800 比较只使用统一的 Event 口径

Ascend stage cycle 和 H800 CUDA kernel breakdown 的阶段边界不同，不能把
两边名为 producer、network 或 copy 的内部阶段直接相除。跨后端正式比较
只使用相同 case、相同逻辑字节公式和相同 warmup/iteration 协议下的端到端
device Event mean、P50、P95 和 logical GB/s。

## 10. 与其他实现的阶段数据怎样对应

已有的其他实现给出了两组数据：

```text
统计 expert tokens：7 us
计算 token 偏移 + 拼接 SQE + 搬运 scale：小于 30 us
```

只有先确认操作语义、输入 shape、统计范围和同步边界一致，这两个数字才能与
P3.0 数据直接相除。下面先按最接近的语义建立映射。

### 10.1 操作映射

| 其他实现的操作 | 我们最接近的阶段 | 我们的耗时 | 对齐程度 |
| --- | --- | ---: | --- |
| 统计最终本地 expert tokens | `epilogue_expert_count` | 339 us | 基本对应 |
| 发送侧按目标 rank 统计 token records | `producer_group` | 29 us | 不是 expert count |
| 计算发送 staging slot 偏移 | `producer_prefix` | 575 us | 取决于“偏移”的定义 |
| 计算最终 expert 输出偏移 | `epilogue_expert_prefix` | 2,788 us | 取决于“偏移”的定义 |
| 发布逻辑 transport commands | `publication` | 163 us | 比写一条 SQE 的范围更大 |
| 构造并提交 WQE/SQE | `service_submit` | 5,734 us | 包含完整 service submit 窗口 |
| 搬运 scale | `producer_record` 的一部分 | 无独立数据 | 同时搬运 hidden 和 metadata |
| 完整 record packing | `producer_record` | 3,860 us | 明显大于单独搬运 scale |

这里的 `producer_control = 6.964 us` 虽然数值恰好接近对方的 7 us，但它
只执行状态清零和布局校验，不统计 token，二者不能对应。

### 10.2 “统计 expert tokens”的两种解释

如果其他实现生成的是最终 `num_tokens_per_expert`，应当与我们的
`epilogue_expert_count` 对比：

\[
\frac{339\ \mu s}{7\ \mu s} \approx 48.4
\]

在相同 workload 和同步边界成立的前提下，我们的最终 expert count 大约慢
48 倍。当前实现是在 payload 到达后扫描有效
`(source_rank, source_slot, topk_lane)`，再为本地 experts 累计 tile
histogram。其他实现如果复用了路由阶段已经产生的 expert count，或者在发送
控制信息时一并传输了 per-expert counts，实际执行的工作量会小得多。

如果对方所谓的“expert tokens”只是发送侧判断一个 token 要去哪些 rank，
那么更接近我们的 `producer_group = 28.638 us`：

\[
\frac{28.638\ \mu s}{7\ \mu s} \approx 4.1
\]

但这只是 destination-rank record count，不能替代最终 expert count。一个
token 的多个 top-k experts 位于同一个 rank 时，group 只产生一个 record；
consumer 仍需要把这条 record 分配给该 rank 上的多个 local experts。

### 10.3 “计算 token 偏移”也有两种解释

如果偏移指发送侧 staging slot：

\[
O_{t,r}=\sum_{j<t}C_{j,r}
\]

对应 `producer_prefix = 575 us`。它单独就已经是对方整个组合阶段 30 us
上限的约 19 倍：

\[
\frac{575\ \mu s}{30\ \mu s} \approx 19.2
\]

该阶段只有少数 rank-owner threads 活跃，并从 global memory 扫描 tile
counts，是一个明确的串行控制面优化目标。

如果偏移指最终 expert-major 输出位置：

\[
O_e=\operatorname{align}\left(\sum_{j<e}N_j\right)
\]

则对应 `epilogue_expert_prefix = 2,788 us`。这个阶段除了 expert count 的
prefix，还要处理 tile histogram、alignment、capacity 和 cached metadata
约束，不能拿 `producer_prefix` 的 575 us 代替。

### 10.4 为什么“小于 30 us”不能直接对比我们的一个 phase

我们当前将对方组合描述中的工作分散在几个范围更大的阶段里：

```text
producer_prefix       575 us   计算发送 staging slot
producer_record     3,860 us   hidden + scale + top-k + metadata
publication           163 us   发布整批逻辑 transport commands
service_submit      5,734 us   解析、寻址、构造 WQE/SQE、post SQ
```

如果把最宽泛的相关阶段全部相加，可以得到：

\[
575 + 3860 + 163 + 5734 = 10{,}332\ \mu s
\]

这个 10.332 ms 不能与 30 us 直接相除，因为两边包含的工作量明显不同。

第一，`producer_record` 不只是搬 scale。Representative FP8 hidden 每条
record 有 7,168 B，而 scale factor 是每 128 个 hidden 元素一个 FP32，合计：

\[
\frac{7168}{128}\times4=224\ \text{B/token}
\]

Scale 字节数约为 hidden 的 3.1%。`producer_record` 还会搬运完整 hidden、
top-k indices、weights 和 source metadata。因此 3.860 ms 主要反映 hidden
staging copy，不能据此判断 scale copy 自身需要几毫秒。

第二，`service_submit` 不是一条 `make_write()` 的耗时。每个 rank 的一次
representative dispatch 共发布 30 条 transport commands，其中有 7 条
payload puts；payload 总量约为 285～287 MB。`service_submit` 覆盖整批命令
的读取、校验、channel 和 address 解析、WQE/SQE 构造及 request post。它既
不是“单条 SQE 构造耗时”，也不包含已经单列的 CQ wait。

第三，其他实现可能由 DMA/RDMA 直接读取原始 hidden，只统计 descriptor、
offset 和 scale 的准备时间。我们的路径先把完整 record 写入 staging shard，
再由 transport service 提交网络写。如果对方的 30 us 不包含 hidden staging、
payload DMA 和 CQ completion，它测量的是控制面准备时间，而我们的相关
phase 同时包含了大量数据面工作。

### 10.5 当前能够得出的性能判断

在两边 workload 完全一致的前提下，目前可以作出三条有限结论：

1. 如果 7 us 确实生成最终 per-expert counts，我们的 339 us 存在约 48 倍
   差距，expert count 算法或 count 复用方式值得优先检查。
2. 发送侧 `producer_prefix` 单项为 575 us，已经显著超过对方整个组合阶段
   的 30 us 上限；这一差距不依赖 hidden copy 的统计歧义。
3. 现有数据不能证明 scale copy 或单条 SQE 构造分别慢了多少，因为这两项
   尚未从 `producer_record` 和 `service_submit` 中独立计时。

不能用 `10.332 ms / 30 us` 宣称我们的同类操作慢 344 倍。这个比值混合了
hidden staging、整批命令服务和逻辑发布，只能说明当前 profiling 粒度不足以
复现对方的 30 us 口径。

### 10.6 下一次严格对比需要补充的打点

为了把差距落实到可分配的优化任务，下一轮 profiling 至少应拆分：

| 子阶段 | 需要记录的内容 |
| --- | --- |
| Rank grouping/count | 扫描 top-k、rank 去重和 tile count |
| Expert count | 扫描 received top-k 和写 tile histogram |
| Rank prefix | tile-to-rank staging offset |
| Expert prefix | histogram 汇总、alignment 和 output offset |
| Hidden staging copy | vector main body 与 scalar tail |
| Scale copy | scale pack 数、字节数和独立 cycles |
| Top-k/metadata copy | indices、weights、source metadata |
| Command publication | command reserve、写 command、publish producer index |
| WQE/SQE construction | `make_write` 或 inline write 构造时间 |
| SQ post | SQ slot、doorbell 和 post 时间 |
| CQ wait | completion polling，继续保持独立 |

报告还必须同时记录每个 rank 的 command 数、payload put 数、payload bytes、
是否等待 CQ，以及统计的是单条命令、单个 rank 的整批命令还是八卡最大值。

对方数据也需要确认以下条件，才能进入正式横向比较：

- 是否同为 8,192 tokens/rank、hidden 7,168、top-k 8、256 experts 和 EP8；
- 7 us 和 30 us 是单 rank、八卡最大值，还是单条 WQE/SQE 的时间；
- token offset 是 rank staging offset 还是 expert output offset；
- 30 us 是否包含 hidden DMA、doorbell、CQ completion 和跨 rank 同步；
- 一次操作构造了多少条 WQE/SQE、搬运了多少 payload 和 scale bytes。

这些条件对齐后，才能分别计算 expert count、prefix、scale copy 和 SQE
construction 的真实倍数，并据此确定优化优先级。

## 11. 源码索引

正式 benchmark 计时与聚合在 `main` 中已经存在：

| 内容 | 路径 |
| --- | --- |
| NPU Event 计时 | `tests/ascend/benchmark/timing.py` |
| warmup 和 measured iteration | `tests/ascend/benchmark/runtime.py` |
| 八卡样本和逻辑带宽聚合 | `tests/ascend/benchmark/runtime.py` |

下面的 stage profiling 文件和符号以 `opt/ascend-p3-overlap` 候选分支为准：

| 内容 | 路径或符号 |
| --- | --- |
| 独立 profiling launch 和阶段聚合 | `tests/ascend/benchmark/runtime.py` |
| Dispatch stage 枚举 | `csrc/backends/ascend/elastic/kernels.hpp` |
| Producer control/group/prefix/record/release | `csrc/backends/ascend/elastic/dispatch.asc` |
| Expert count/prefix/metadata/copy | `csrc/backends/ascend/elastic/dispatch.asc` |
| Stage profile ABI 与 phase 公式 | `csrc/backends/ascend/transport/stage_profile.hpp` |
| URMA WQE/SQE 构造和请求提交 | `csrc/backends/ascend/transport/aicore_transport_service.hpp` |
| Profile 导出到 Python | `csrc/backends/ascend/elastic_buffer.hpp` |

在 P3 profiling 代码正式合入 `main` 前，阅读或复现实验时必须 checkout
对应候选 commit，不能仅根据 `main` 上的文档假设 `--profile-stages` 已可用。
