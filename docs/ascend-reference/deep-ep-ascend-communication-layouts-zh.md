# DeepEP-Ascend 通信对象与 WQE/SQE/CQE 布局

本文补充《DeepEP-Ascend：SIMT 为什么适合 MoE 通信？》中未展开的底层对象布局，供串讲和代码评审时查阅。内容以官方仓库 commit 3b25377d04b24fc6154698ded78a2bcb2c59afff 为准，并结合 AscendC HCOMM 的 URMA ABI 定义。

## 1. 8 rank / 64 AIV 资源模型

README 的典型性能配置是 32 个 AI core，对应 64 个 AIV。Host 侧资源初始化和单次 kernel launch 使用两个相关但不同的参数：

| 参数 | 来源 | 含义 |
| --- | --- | --- |
| num_jetties | runtime 的设备 AIV 数 | 创建 Jetty endpoint 和 channel 表 |
| num_vec_cores | num_ai_cores 乘每个 AI core 的 AIV 数 | 本次 EP kernel launch 的 AIV block 数 |

典型配置下二者都是 64，但不能把它们说成同一个概念。

8 rank 时，本 rank 有 7 个远端 peer。Host 创建：

    num_peers    = 8 - 1 = 7
    num_jetties  = 64
    num_channels = max(num_jetties, num_peers) = 64

channel 与 Jetty 的绑定规则是：

    jetty_endpoints[channel_idx % num_jetties]

因此 8 rank / 64 AIV 下得到：

    channel[0]  -> Jetty0
    channel[1]  -> Jetty1
    ...
    channel[63] -> Jetty63

前 7 个 channel 覆盖所有非本 rank peer；剩余 channel 连接到与本 rank 编号相邻的 peer，用于把剩余 Jetty 也纳入 channel 表。Device 侧最终看到的是一个 64 项 jetty_ptrs 表。

## 2. Device 侧 channel 表的用法

Host 把 channel_handles 拷贝到 Device，形成 jetty_ptrs。Device 侧有两类索引：

1. AIV 索引：AIV i 用 jetty_ptrs[i] 加载自己的 SQ/CQ 元数据；
2. peer 索引：加载某个远端 peer 的 metadata 时，用非本 rank peer 在前 7 个 channel 中的索引。

peer 索引转换规则是：

    peer_idx < rank_idx ? peer_idx : peer_idx - 1

例如 rank 2 的 peer 表包含 rank 0、1、3、4、5、6、7。

Host 侧要求所有 peer 使用同一个 local UBC_CTP endpoint。因此虽然 peer metadata 来自前 7 个 channel，而 AIV 自己的 SQ 来自 channel[i]，远端 Jetty 信息仍能用于该 AIV 私有队列上的 WQE。

## 3. HCOMM ChannelEntity 到 DeepEP 紧凑结构

HCOMM 的 ChannelEntity 是 256B 的资源描述对象。DeepEP 不把整个对象搬进 UB，而是在 Device 侧解析出两个紧凑结构：

| HCOMM 对象 | DeepEP 提取结果 | 用途 |
| --- | --- | --- |
| remoteBufferAddr[0] | HcommPeerInfo | 远端内存 base、size、token、EID |
| sqContextAddr[0] | HcommJettyInfo 的 SQ 字段 | SQ base、head、tail、doorbell、WQEBB 大小 |
| cqContextAddr[0] | HcommJettyInfo 的 CQ 字段 | CQ base、tail、doorbell、CQE 大小、CQ depth |

EP 路径使用 HCOMM_URMA_DEFAULT_QP_IDX，其值为 0。

## 4. HcommJettyInfo：每个 AIV 的队列元数据

DeepEP-Ascend 中的 HcommJettyInfo 是 88B：

| Offset | Size | 字段 | 含义 |
| ---: | ---: | --- | --- |
| 0x00 | 8B | sq_base_addr | SQ 起始地址 |
| 0x08 | 8B | sq_head_addr | SQ head 地址 |
| 0x10 | 8B | completion_tail_addr | 硬件发布的 SQ completion tail 地址 |
| 0x18 | 8B | sq_doorbell_addr | SQ doorbell 地址 |
| 0x20 | 8B | cq_base_addr | CQ 起始地址 |
| 0x28 | 8B | cq_tail_addr | CQ tail 地址 |
| 0x30 | 8B | cq_doorbell_addr | CQ doorbell 地址 |
| 0x38 | 8B | packed_head | 低 32 位是 SQ head，高 32 位是期望 CQE 数 |
| 0x40 | 4B | reserved_jfs_id | 保留字段 |
| 0x44 | 4B | num_wqebb_bytes | WQEBB 大小，当前为 64B |
| 0x48 | 4B | reserved_jfc_id | 保留字段 |
| 0x4C | 4B | num_cqe_bytes | CQE 大小 |
| 0x50 | 4B | cq_depth | CQ 深度 |
| 0x54 | 4B | reserved | 保留字段 |

packed_head 是提交状态的核心：

    packed_head[31:0]  = 当前 SQ head，以 WQEBB 为单位
    packed_head[63:32] = 期望的 CQE 数量

这样发布 head 时可以用一次 64B store 完成。

## 5. HcommPeerInfo：远端 peer 元数据

每个 peer 的紧凑元数据是 48B：

| Offset | Size | 字段 | 含义 |
| ---: | ---: | --- | --- |
| 0x00 | 8B | tp_id_num_sges_rmt_token_id | 低 24 位 tp_id，8 位 num_sges，20 位远端 token id |
| 0x08 | 8B | rmt_eid[0] | 远端 EID 低 64 位 |
| 0x10 | 8B | rmt_eid[1] | 远端 EID 高 64 位 |
| 0x18 | 8B | rmt_token_value_udf | 远端 token value 与 UDF 字段 |
| 0x20 | 8B | rmt_base_addr | 远端注册内存基地址 |
| 0x28 | 8B | rmt_buffer_size | 远端注册内存大小 |

8 rank 时，每个 AIV 的 UB 中有 7 份该结构，共 336B。

## 6. 接收 buffer 与 slot

EP 接收 buffer 按 rank 分区，再按 token slot 划分：

    symmetric buffer
      ├── rank 0 partition
      │     ├── token slot 0
      │     ├── token slot 1
      │     └── ...
      ├── rank 1 partition
      └── ...

每个 token slot 的大小由 TokenLayout::get_num_bytes() 决定：

    token bytes =
        hidden bytes
      + align(scale factor bytes)
      + align(metadata bytes)

Dispatch 的 metadata 包括 top-k weights、top-k indices 和 source token global index。

Dispatch 中 slot 的计算方式是：

    dst_slot_idx = rank_histogram[dst_rank_idx] + sgep_idx

其中 rank_histogram[dst_rank_idx] 是目标 rank 在接收侧的前缀偏移，sgep_idx 是该 token 在这个目标 rank 内的相对序号。

WQE 中的远端地址由 slot 翻译而来：

    remote_addr =
        目标 rank 接收分区 offset
      + dst_slot_idx * token bytes

因此 slot 不是独立分配的通信内存对象，而是接收 buffer 中的 token 记录索引。

## 7. SQ 与 WQEBB

DeepEP-Ascend 使用：

    kNumWQEBBBytes = 64
    kSQDepth = 32768
    kSQEBBAlignment = 4

因此 SQ 的布局是：

    SQ
    ├── WQEBB[0]      64B
    ├── WQEBB[1]      64B
    ├── ...
    └── WQEBB[32767]  64B

DeepEP 用 kSQDepth 做回绕和容量检查；实际 SQ context 仍由 HCOMM 拥有，队列深度以 HCOMM 配置为准。

WQE 是逻辑提交项，可以跨多个 WQEBB：

| 操作 | WQE 粒度 | WQEBB 数 |
| --- | --- | ---: |
| Dispatch 逻辑 SQE slot | 固定 256B | 4 |
| Combine 单 token WQE | 128B | 2 |
| Combine 单 task，最多 8 token | 最多 1024B | 最多 16 |

## 8. HcommUrmaSqeCtx：48B SQE header

HcommUrmaSqeCtx 共 48B，即 6 个 u64 word：

| Word | 字段 |
| ---: | --- |
| 0 | sqeBbIdx、flag、tokenEn、rmtJettyType、owner、opcode、inlineMsgLen 等 header 位域 |
| 1 | tpId、sgeNum、rmtJettyOrSegId |
| 2 | rmtEidL |
| 3 | rmtEidH |
| 4 | rmtTokenValue、UDF、reduce 类型与 opcode |
| 5 | 远端地址低 32 位和高 32 位 |

word 0 的关键位域：

| Bits | 字段 | 含义 |
| ---: | --- | --- |
| 15:0 | sqeBbIdx | SQ WQEBB index |
| 23:16 | flag | ordering、fence、CQE 等控制位 |
| 28 | tokenEn | 使用 token |
| 30:29 | rmtJettyType | Jetty 模式 |
| 31 | owner | SQ owner bit |
| 47:40 | opcode | WRITE 或 NOP 等 opcode |
| 63:54 | inlineMsgLen | inline message 长度 |

Dispatch 最后一个 payload WRITE 使用 strong order、fence 和 CQE，用于覆盖前面所有 WRITE 的完成语义。

## 9. HcommUrmaSgeCtx：16B SGE

每个 SGE 是 16B：

| Offset | Size | 字段 | 含义 |
| ---: | ---: | --- | --- |
| 0x00 | 4B | len | 本地 payload 长度 |
| 0x04 | 4B | tokenId | 本地 token id |
| 0x08 | 8B | va | 本地 source address |

Dispatch 中一个 SGE pair 是两个 SGE：

    SGE hidden    -> x + token_idx * hidden_bytes
    SGE metadata  -> metadata_send_buffer + token_idx * metadata_bytes

因此一个 SGE pair 占 32B，对应 4 个 lane。

## 10. Dispatch WQE 布局

Dispatch 每个 SQE slot 固定占 4 个 WQEBB，即 256B：

    256B SQE slot
    ├── 48B  SQE header
    ├── 192B 最多 12 个 SGE
    └── 16B  padding / NOP 对齐

最大 SGE 数来自：

    kNumSGEPairsPerSQE = 6
    num_sges = num_sgeps * 2

所以最大有效 WQE 是 48B header 加 12 个 16B SGE，共 240B；向上对齐到 64B WQEBB 后正好是 4 个 WQEBB。

lane 到字段的映射：

| lane | 字段 |
| ---: | --- |
| 0 | SQE header / opcode / owner / index |
| 1-5 | peer info / token id / remote address |
| 6-7 | SGE 0：hidden length / source address |
| 8-9 | SGE 1：metadata length / source address |
| 10-13 | 后续 SGE pair |

如果实际 SGE 数不足 12，剩余 WQEBB 用 NOP header 补齐，保证 SQ head 始终按 4 WQEBB 对齐推进。

## 11. Combine WQE 布局

Combine 每个 token 构造一个 WQE，固定两个 SGE：

    48B SQE header + 2 * 16B SGE = 80B

向上对齐到 64B WQEBB 后为 2 个 WQEBB，即 128B。

每个 task 最多 8 个 token，因此一个 task 最多 8 * 128B = 1024B，即 16 个 WQEBB。

Combine 的 lane 映射：

| lane | 字段 |
| ---: | --- |
| 0 | SQE header |
| 1-5 | peer metadata / remote address |
| 6-7 | hidden SGE |
| 8-9 | top-k weights SGE |

Combine 的 VF launch 是 kNumTokensPerTask * warpSize，即最多 8 * 32 = 256 个 SIMT thread。每个 warp 处理一个 token，并写入该 token 自己的 128B WQE。

## 12. HcommUrmaJfcCqeCtx：64B CQE

当前 AscendC HCOMM ABI 中的 HcommUrmaJfcCqeCtx 是 64B，即 16 个 u32 word：

| Word | 字段 |
| ---: | --- |
| 0 | sR、isJetty、owner、inlineEn、opcode、fd、substatus、status |
| 1 | entryIdx、localNumL |
| 2 | localNumH、rmtIdx |
| 3 | tpn |
| 4 | byteCnt |
| 5 | userDataL |
| 6 | userDataH |
| 7-10 | rmtEid[4] |
| 11 | dataL |
| 12 | dataH |
| 13-15 | inlineData[3] |

DeepEP-Ascend 主要检查 word 0：

    owner     = (cqe_header >> 2) & 1
    substatus = (cqe_header >> 16) & 0xff
    status    = (cqe_header >> 24) & 0xff

解释：

| 字段 | 判断 |
| --- | --- |
| owner | 与当前 tail phase 不一致时，表示硬件发布了新 CQE |
| status | 非 0 表示错误 |
| substatus | 非 0 表示错误子状态 |

实际 CQE 大小从 CqContext.contextInfo.ubJfc.cqeSize 读取，DeepEP 不在 EP 路径中硬编码该值。

## 13. doorbell 与完成路径

### SQ doorbell

ring_doorbell 做两次关键写入：

    1. 写 64B packed_head 到 sq_head_addr
    2. 写低 32 位 head 到 sq_doorbell_addr

两次写入之间使用 DDR DataSyncBarrier，保证 WQE 数据和 head 发布顺序。

### CQ reclaim

CQ 轮询会：

1. 读取 cq_tail_addr；
2. 根据 CQE owner bit 判断完成；
3. 检查 status 和 substatus；
4. 推进本地 tail；
5. 更新 cq_tail_addr、cq_doorbell_addr 和 completion_tail_addr。

Dispatch 在最后一个 SQE 上请求 CQE；Combine 在结尾追加 strong-order NOP 并请求 CQE。

## 14. 串讲速查

8 rank / 64 AIV 下可以说：

> 每个 AIV 拥有一个 Jetty，对应一条 SQ 和一个 CQ。Host 创建 64 个 UBC_CTP channel 并把 channel 表交给 Device。AIV 用自己的 channel 加载 SQ/CQ 元数据，用前 7 个 peer channel 加载远端 metadata。Dispatch 的一个 SQE slot 是 4 个 64B WQEBB，最多容纳 48B header 加 12 个 16B SGE；Combine 每个 token 是 2 个 WQEBB，一个 task 最多 8 个 token。SIMT 写 WQE，scalar 更新 SQ head 并 ring doorbell，硬件执行 URMA，CQE 由 CQ 轮询回收。

## 15. 源码入口

| 内容 | 文件 |
| --- | --- |
| Host Jetty/channel 初始化 | csrc/kernels/comm/hccl.hpp |
| HcommJettyInfo 与 HcommPeerInfo | deep_ep/include/deep_ep/comm/handle.hpp |
| Dispatch WQE 构造 | deep_ep/include/deep_ep/impls/ep/dispatch.hpp |
| Combine WQE 构造 | deep_ep/include/deep_ep/impls/ep/combine.hpp |
| token 与接收 buffer layout | deep_ep/include/deep_ep/layout/ep/token.hpp |
| WQEBB 对齐常量 | deep_ep/include/deep_ep/common/compiled.hpp |

本文对应的官方源码快照：

    3b25377d04b24fc6154698ded78a2bcb2c59afff
