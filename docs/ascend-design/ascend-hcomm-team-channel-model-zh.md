# Ascend HCCL/HCOMM Team、Channel 通信编程模型：CANN 9.2.0 与 9.3.0

## 1. 文档目的和阅读方式

本文专门解释 Ascend 上 DeepEP 使用的一套设备侧通信编程模型：HCCL communicator，HCOMM team、window、channel，endpoint，memory registration，SQ/CQ，URMA，以及这些对象在设备内核中如何被解析和使用。文档先建立概念，再分别描述 CANN 9.2.0 与 CANN 9.3.0 的模型，最后对比两个版本的差异。

本文面向需要理解或适配 DeepEP Ascend transport 的工程师。它不是 HCCL 集合通信教程：`HcclAllReduce`、`HcclBroadcast` 等集合算子不是本文主线。本文关心的是设备发起的一侧通信（one-sided communication），即 AICore transport service 在 kernel 中根据已建立的 HCOMM 资源构造 URMA 请求，完成远端写入、远端原子加和信号同步。

文中区分三类信息：

- **API/ABI 事实**：来自仓库使用的 CANN 头文件、历史提交和 DeepEP 已验证实现；
- **运行时观察**：来自 NPU4Px2、NPU8P/NPU8P-ALT 的实际探测结果；
- **已验证判断**：9.3 的 DeepEP 适配方向已在 NPU8P-ALT 的设备 0/1 上完成 2-rank 资源建立验证；payload 传输仍需接入 DeepEP AICore service 后验证。
- **2026-09-19 更新**：2-rank put、put-value64、faa64、signal、
   signal-set、flush、async-lifecycle、payload-signal-order、
   barrier-repeat 和 teardown 已通过。9.3 的 AIV channel 不能调用
   HcclChannelDestroy；正确 host 生命周期是先停止 service 并释放
   DeepEP 自建 staged 资源，channel 本身交给 HCCL communicator 生命周期
   管理。

除非特别说明，本文中的 9.2 指的是 DeepEP 已在 NPU4Px2 `/data/y00621698/pkg/cann` 验证过的 CANN 9.2.0 环境；9.3 指的是 NPU8P-ALT `/data/disk2/cann_version/0916/use_cann/cann-9.3.0` 组合包环境。

## 2. 总体视图

CANN/HCOMM 的通信资源可以分为控制面和数据面两部分。

```text
进程 / rank
  |
  |  HcclComm
  |    提供 rank/world size、集群通信域、host barrier
  |
  +-- Team：成员集合、设备侧 channel 表、同步内存
  |     |
  |     +-- Channel：到某个 peer 的通信上下文
  |           |
  |           +-- local/remote notify 表
  |           +-- local/remote registered buffer 表
  |           +-- SQ context 表
  |           +-- CQ context 表
  |
  +-- Window：对称内存窗口和远端地址转换
  |
  +-- Endpoint / MR：9.3 显式化的网络端点与内存注册资源
  |
  +-- URMA 数据面：SQE/SGE/CQE、doorbell、完成轮询
```

更直观地说：

- **Communicator** 回答“我是哪个 rank，参与通信的一组进程是谁”；
- **Team** 回答“这组 rank 在设备侧对应哪些 member，我为每个 peer 准备了哪些 channel 和同步内存”；
- **Window** 回答“我的对称内存逻辑地址如何转换成某个远端 rank 的实际内存地址”；
- **Channel** 回答“访问这个 peer 时使用哪套队列、令牌和注册内存描述”；
- **Endpoint** 回答“通信落在哪个网络/总线端点上，协议和地址是什么”；
- **MR** 回答“哪些本端和远端内存允许被这条通信路径访问”；
- **SQ/CQ/URMA** 是实际执行层，负责提交请求、doorbell 和完成通知。

这组对象不是平行替代关系。Team/window/channel 是 HCCL 封装后的资源模型；endpoint/MR/channel 是 HCOMM 更底层的显式资源模型；URMA 是这些资源最终驱动的数据面机制。9.2 和 9.3 的关键差异，正是创建这些资源时的所有权和显式程度发生了变化。

## 3. 基础概念

### 3.1 Communicator：`HcclComm`

`HcclComm` 是 HCCL 的通信域句柄，通常由 `HcclCommInitRootInfo`、`HcclCommInitClusterInfo` 或上层框架（如 Torch-NPU 的 HCCL process group）创建。它包含：

- 本 rank 的 communicator rank ID；
- rank 数量；
- 集群发现、连接和配置状态；
- host 侧通信资源。

DeepEP 不直接创建全局通信域，而是复用进程组暴露的 `HcclComm`。transport 初始化时通过 `HcclGetRankId` 和 `HcclGetRankSize` 校验传入的 rank/world size 是否与 communicator 匹配。

communicator 有两个容易混淆的 ID：

- **rank ID**：communicator 中的全局进程编号；
- **member ID**：某个 team 内部的成员下标。

world team 中二者经常一一对应，但子 team 不必如此。9.3 提供了 `HcclTeamRankToMember` 和 `HcclTeamMemberToRank` 用于二者转换。DeepEP 当前创建的是全 rank team，因此设备侧看到的 member ID 等价于 world peer 下标。

### 3.2 Team

Team 是一组 rank 的设备侧通信资源容器。它不只描述成员关系，还承载：

- **成员表**：`rankIds`、`rankNum`、`selfRankId`，或设备侧 `memberNum`、`selfMemberId`；
- **channel 表**：`channelsBaseAddr` 指向连续的 `ChannelEntity` 数组；
- **每个 member 的 channel 数量**：`channelCntAccumulatePerMember[peer]` 是前缀和，表示 peer 之前所有 member 的 channel 数总和；
- **同步内存**：`remoteMems`、`shadowMem`、`syncMemReq`；
- **网络层信息**：`netLayer`、`worldTeamIds`；
- **引擎/协议**：例如 AIV engine、UBC_CTP 协议。

#### Rank ID 与 member ID

`rankIds` 描述 team 内包含哪些 communicator rank，顺序就是 team 内部 member ID 的来源。例如：

```text
rankIds = {7, 2, 5}
member 0 -> rank 7
member 1 -> rank 2
member 2 -> rank 5
```

设备侧 `HcommTeam.selfMemberId` 是本 rank 在这个顺序中的下标，而不是 communicator rank ID。发送到 peer 时也必须使用 member ID 查 channel 表。

#### channel 表索引

`HcommTeam.channelsBaseAddr` 是连续 `ChannelEntity` 数组，不同 peer 的 channel 连续存放。访问 peer `p` 的第 `c` 条 channel 时：

```text
index = channelCntAccumulatePerMember[p] + c
channel = channelsBaseAddr + index * sizeof(ChannelEntity)
```

如果 `channelCntAccumulatePerMember` 存的是完整前缀和，则第 0 个 peer 的值为 0；如果存的是“peer 之前所有 member 的数量之和”，二者语义相同。DeepEP 的设备侧实现按第二种语义逐项累加并校验 `c < counts[p]`。

#### 同步内存

Team 的同步资源由 `HcommTeamSyncMemRequirement` 声明：

- `signalCount`：信号槽数量；
- `counterCount`：计数器/counter 槽数量；
- `barrierCount`：barrier 会话数量。

总大小为：

```text
syncMemSize =
  (signalCount + counterCount + barrierCount) * sizeof(uint64_t)
  * memberNum
```

`remoteMems` 是从远端交换来的同步内存描述；访问 `selfMemberId` 时表示本地内存，否则表示远端内存。`shadowMem` 是 team 拥有的一块设备内存，可作为 FAA fetch-result 等“本地接收结果”的目标。普通 `aclrtMalloc` 分配没有进入 channel 的 registered buffer 表，也没有有效 local token，因此不能直接替代 `shadowMem` 作为 FAA fetch-result。

#### world team 与子 team

world team 包含 communicator 的全部 rank。子 team 只包含部分 rank，适合表示 scale-up 域、scale-out 对应 rank 或其它逻辑分组。DeepEP 的 backend-neutral `TransportTeam` 有 world、scale-up、scale-out 三种逻辑值，但 CANN 路径当前创建的是一个覆盖全部 EP rank 的 team。

### 3.3 Window

Window 是对称内存窗口。它的作用是把“本 rank 的逻辑对称地址”转换成“目标 rank 的实际远端地址”。

典型的对称内存访问是：

```text
offset = local_address - local_window_base
remote_address = peer_window_base + offset
```

offset 在所有参与方语义相同，但每个 rank 的 `peer_window_base` 可以不同。这使上层算子可以继续使用本地指针表示远端对象，而不必在业务代码里为每个 peer 保存一套实际地址。

Window 不是普通指针表。它还承担：

- 注册内存的范围校验；
- 远端地址转换；
- 网络层/world team 索引；
- LSA 或对称 arena 的 stride 信息；
- 设备侧可见的 `HcommWindow` 描述符。

在 CANN 9.2 的 DeepEP 路径中，`HcclTeamWindowRegister` 返回的 `HcommWindowHandle` 实际指向设备侧可读的 `HcommWindow` 结构。设备代码用它读取每个 member 的 `CommMem`，得到 `address + offset` 后作为 URMA 目标地址。

在 CANN 9.3 中，`HcommWindow` 的 ABI 已经扩展为 `netWin` 和 `lsaWin` 两部分。`netWin` 提供远端地址数组和网络层索引；`lsaWin` 提供本地对称 arena 的 base、stride 和用户大小；`legacySymWindow` 保留旧结构指针。本机 950PR 组合包上，旧的“申请地址后注册/查询得到可用 device-visible `HcommWindow*`”路径未得到运行时支持，这是 9.3 适配的关键问题。

### 3.4 Channel

Channel 是访问一个 peer 的通信上下文。它不是单纯 socket，也不只是一条队列；它是一组可被设备侧解析的资源：

- engine 和 protocol；
- local/remote notify 表；
- local/remote registered buffer 表；
- SQ context 表；
- CQ context 表。

`ChannelEntity` 的核心字段如下：

| 字段组 | 含义 |
| --- | --- |
| `abiHeader` | ABI magic/version/size |
| `engine` | `COMM_ENGINE_AIV`、AICPU、CCU 等 |
| `protocol` | UBC_CTP、RoCE 等 |
| `localNotifyAddr/remoteNotifyAddr` | 本端/远端通知资源表 |
| `localBufferAddr/remoteBufferAddr` | 本端/远端注册内存表 |
| `sqContextAddr` | 发送队列上下文表 |
| `cqContextAddr` | 完成队列上下文表 |
| `sqNum/cqNum` | 队列数量 |

Channel 的粒度是“peer + 一条通信路径”。DeepEP 可以为同一个 peer 创建多条 channel，让独立 SQ/CQ 在硬件中重叠推进。软件可以顺序构造多个 channel 的 WQE，但同一条 SQ 必须有明确的单生产者或并发控制；多个 AICore 不能无协议地并发写同一 SQ。

#### Channel 与 queue 的区别

Channel 是资源容器，SQ/CQ 是其中的队列。一条 channel 至少包含一条 SQ 和一条 CQ，但概念上它们不同：

- channel 负责资源绑定、认证、地址表和协议配置；
- SQ 负责提交请求；
- CQ 负责完成通知。

因此“channel 数量”与“SQ/CQ 深度”、“WQE 数量”都不是一回事。增加 channel 数可以增加独立路径；增加 SQ 深度可以容纳更多未完成请求；一次 payload 传输可能拆成多个 WQE。

#### notify 与 registered buffer

`ChannelEntity` 中的 notify 表和 buffer 表是两类不同资源：

- **notify**：小段用于事件、标志或进度同步的内存。`RegedNotifyEntity` 会区分 IPC 与 RMA，以及 RT 与 MEM 等访问方式；条目中包含地址、大小，部分类型还包含 notifyId 或 protection 信息。
- **registered buffer**：真正承载数据的内存段。`RegedBufferEntity` 分为 IPC 和 RMA：IPC 条目只有地址和大小；RMA 条目还包含 `ProtectionInfo`，UB 协议下是 tokenId/tokenValue，RoCE 下是 lkey/rkey。

`notifyNum` 描述 channel 需要的 notify 数量，`localBufferNum/remoteBufferNum` 描述 channel 绑定的本端/远端 MR 数量。DeepEP 当前 payload 主要依赖 buffer 表，signal/barrier 依赖 team 同步内存；notify 表是 HCOMM 通用模型的一部分，不能简单理解为业务 payload buffer。

#### AIV、AICPU 与 CCU

`CommEngine` 表示请求由哪类引擎消费：

- `COMM_ENGINE_AIV`：AIV/AICore 路径，DeepEP staged transport 使用；
- `COMM_ENGINE_AICPU` / `AICPU_TS`：AICPU 路径；
- `COMM_ENGINE_CCU`：CCU 路径；
- host CPU 相关枚举用于 host 侧或 TS 通信。

不同 engine 对同一 protocol 的队列配置、可用操作和约束可能不同。9.3 的 `ubAttr.scqDepth` 注释明确说明：SCQ 独占模式仅支持 HOST 和 AICPU 场景，AIV 不支持。

#### UBC_CTP 与 RoCE

`CommProtocol` 描述底层通信协议，例如：

- `COMM_PROTOCOL_HCCS`；
- `COMM_PROTOCOL_ROCE`；
- `COMM_PROTOCOL_PCIE`；
- `COMM_PROTOCOL_UB_CTP`，旧名 `COMM_PROTOCOL_UBC_CTP`；
- `COMM_PROTOCOL_UBC_TP`；
- `COMM_PROTOCOL_UB_MEM`；
- `COMM_PROTOCOL_UBOE`；
- `COMM_PROTOCOL_UB_RTP`。

DeepEP 当前单机多卡路径使用 UBC_CTP。9.3 PTO 参考实现展示了 RoCE 的 endpoint/channel 创建方式，资源生命周期可参考，但 RoCE 的 IP、端口、QP、重传和 QoS 参数不能直接照搬到 UBC_CTP。

### 3.5 Endpoint

Endpoint 描述网络/总线端点，核心是三元组：

```text
protocol + commAddr + loc
```

`EndpointDesc` 的字段含义：

| 字段 | 含义 |
| --- | --- |
| `protocol` | `COMM_PROTOCOL_UBC_CTP`、`COMM_PROTOCOL_ROCE` 等 |
| `commAddr.type` | `IP_V4`、`IP_V6`、`ID`、`EID` |
| `commAddr.addr/addr6/id/eid` | 按类型解释的地址 |
| `loc.locType` | `ENDPOINT_LOC_TYPE_DEVICE` 或 `HOST` |
| `loc.device.devPhyId` 等 | 设备物理位置 |
| `loc.host.id` | host 侧位置 |

Endpoint 可以在设备上，也可以在 host 上。RoCE 常见配置是 host 位置加 IPv4；NPU8P-ALT 的 9.3 UBC_CTP 观察结果是 device 位置加 EID。

9.3 提供了显式查询和创建接口：

```text
HcommEndpointGetDescNum(deviceLogicId, &descNum)
HcommEndpointGetDescs(deviceLogicId, &descNum, endpointDescs)
HcommEndpointCreate(&endpointDesc, &endpointHandle)
HcommEndpointDestroy(endpointHandle)
```

典型流程是先按设备查询可用 endpoint 描述，选择匹配协议和位置的描述，再创建 endpoint handle。远端 endpoint 描述需要在 rank 之间交换，供 channel 创建时填写 `HcommChannelDesc.remoteEndpoint`。

9.3 还提供辅助接口：

- `HcommEndpointGetListenPort`：查询 endpoint 的监听端口，主要用于 host/socket 类协议；
- `HcommEndpointCheckFeature`：检查 endpoint 是否支持指定底层特性，例如 NPU Direct RDMA Async；
- `HcommThreadAlloc` / `HcommThreadFree`：按通信引擎分配/释放通信 thread 资源，并指定每个 thread 的 notify 数量；
- `HcommThreadResGetInfo`：读取 thread 关联资源，例如 stream。

这些接口属于显式资源模型的控制面。DeepEP 的 UBC_CTP device endpoint 路径当前不依赖 host listen port，但跨机或 RoCE 路径会用到。

### 3.6 Memory Registration 与 MR

内存注册把一段内存发布给通信路径。`CommMem` 描述内存段：

| 字段 | 含义 |
| --- | --- |
| `type` | `COMM_MEM_TYPE_DEVICE`、`HOST`、`CCU` |
| `addr` | 内存地址 |
| `size` | 字节数 |

9.3 显式 API 为：

```text
HcommMemReg(endpoint, memTag, &commMem, &memHandle)
HcommMemUnreg(endpoint, memHandle)
HcommMemExport(endpoint, memHandle, &memDesc, &memDescLen)
HcommMemImport(endpoint, memDesc, descLen, &commMem)
HcommMemUnimport(endpoint, memDesc, descLen)
```

`HcommMemReg` 建立本端 MR；`HcommMemExport/Import` 用于把 MR 描述传给其它 rank 或导入远端 MR 描述。channel 创建时可以通过两种方式交换 MR：

- `exchangeAllMems = true`：HCOMM 在 channel 建立过程中完成内存描述交换；
- `exchangeAllMems = false`：调用者显式提供 `memHandles` 和 `memHandleNum`。

channel 建立后，`ChannelEntity.localBufferAddr` 指向本端注册内存表，`remoteBufferAddr` 指向远端注册内存表。URMA 请求必须在这些表的地址范围内查找对应的 local/remote buffer，并使用表中的 token。

UB 类协议的 `ProtectionInfo` 使用：

```text
tokenId + tokenValue
```

RoCE 使用：

```text
lkey + rkey
```

这就是为什么“地址在 window 内”还不充分：设备侧还需要找到覆盖该地址的 registered buffer entry，取得 token/key，才能构造合法 URMA 请求。

### 3.7 ABI 头与版本

CANN 9.3 的核心描述符普遍带 `CommAbiHeader`：

| 字段 | 含义 |
| --- | --- |
| `version` | ABI 版本 |
| `magicWord` | 结构体识别码 |
| `size` | 结构体大小 |
| `reserved` | 保留字段 |

当前关键版本包括：

- `HCCL_TEAM_CREATE_DESC_VERSION = 2`；
- `HCOMM_TEAM_VERSION = 1`；
- `HCOMM_WINDOW_VERSION = 2`；
- `HCOMM_CHANNEL_VERSION = 5`。

`EndpointDescInit`、`HcclTeamCreateDescInit` 和 `HcommChannelDescInit` 会先把结构体填成 0xFF，再设置 header 和无效默认值。调用者必须显式覆盖自己要使用的字段，不能假设未设置字段是 0。channel ABI v5 在 v4 基础上增加了 `ubAttr.scqDepth`。
+### 3.8 SQ/CQ 与 URMA

#### SQ：Submission Queue

SQ 是发送队列。生产者把请求描述写入 SQ slot，然后更新 head/doorbell，通知硬件消费。

`SqContext` 的 UB JFS 布局包含：

- `sqVa`：SQ ring 基地址；
- `headAddr` / `tailAddr`：生产者/消费者计数地址；
- `dbVa`：doorbell 地址；
- `jfsID`：队列 ID；
- `wqeSize`：单个 WQE/SQE 大小；
- `sqDepth`：队列深度；
- `tpID`：transport path ID；
- `remoteEID`：远端 EID。

#### CQ：Completion Queue

CQ 是完成队列。硬件写 CQE，消费者检查 CQE 的 owner/status，更新 CQ tail 并敲 CQ doorbell。

`CqContext` 的 UB JFC 布局包含：

- `scqVa` / `cqVa`：CQ ring 基地址；
- `headAddr` / `tailAddr`；
- `dbVa`；
- `jfcID` / `cqn`；
- `cqeSize`；
- `cqDepth`。

#### SQE、WQE、SGE、CQE

这些术语经常混用，但在本文中按以下方式理解：

- **WQE**（Work Queue Entry）：泛指提交给工作队列的请求描述；
- **SQE**（Submission Queue Entry）：SQ 中的 WQE；
- **SGE**（Scatter/Gather Entry）：描述本端数据源或 fetch-result 的地址、长度和 token；
- **CQE**（Completion Queue Entry）：CQ 中的完成描述。

DeepEP 当前使用的 UBC_CTP SQE 为 48 字节，SGE 为 16 字节，CQE 为 64 字节。write 和 inline write 请求占用一个 64 字节 base block；FAA 请求占用两个 base block，并额外携带 fetch-result SGE。

#### doorbell 与 producer/consumer

doorbell 是设备可见的特殊写入，不是普通 global store 的简单替代。CANN 参考实现使用 `st_dev` 写 SQ/CQ doorbell，并用 `ld_dev` 读取设备队列状态。DeepEP 的 staged transport 中，SIMT VF 只记录通信命令，AICore service 在服务边界执行 `st_dev/ld_dev`。

SQ head 是 64 位，低 32 位表示 ring 内位置，高 32 位表示请求计数；tail 是硬件/消费者推进的计数。深度不足时必须先 drain CQ，避免覆盖尚未完成的 SQ slot。

#### URMA 操作

URMA 是底层远程内存访问机制。DeepEP 当前使用三类操作：

- **write**：SGE 描述本端源，SQE 描述远端 token/address；
- **inline write 64**：值直接携带在 WQE 中，用于小值写入；
- **fetch-add atomic（FAA）**：远端地址执行原子加，并把 fetch 结果写回本端注册的 fetch-result 地址。

write/FAA 的目标地址来自 window 或 channel remote buffer 解析；local 源和 fetch-result 地址必须能解析到 local registered buffer；SQE 中还需填写 remote token、remote EID 和 transport path ID。

#### completion、drain 与 flush

提交请求后，软件不能假设立即完成。`flush` 的语义是把本批请求推进到可观察完成：

1. 轮询相关 peer/channel 的 CQE；
2. 校验 owner/status/substatus；
3. 更新 CQ consumer/tail；
4. 敲 CQ doorbell；
5. 确认已消费的 CQE 数量达到已提交数量。

DeepEP 的 transport facade 有 `flush`/`flush_async`/`wait` 语义。当前 staged 实现在 AICore service 边界做 bounded drain，SIMT producer 不直接访问 SQ/CQ doorbell。

## 4. CANN 9.2.0 编程模型

### 4.1 DeepEP 使用的资源创建流程

CANN 9.2 的 DeepEP 路径是 team/window/channel 的封装式模型：

```text
初始化 ACL/设备
  ↓
获取或创建 HcclComm
  ↓
HcclGetRankId / HcclGetRankSize 校验
  ↓
HcclWorldTeamCreate
  ↓
aclrtMalloc 分配对称业务内存
  ↓
HcclTeamWindowRegister
  ↓
HcclTeamChannelsCreate
  ↓
导出 team handle + window handle 到 DeviceTransportContext
  ↓
AICore service 解析 team/window/channel
  ↓
URMA write / inline write / FAA
```

对应的关键 host API：

```text
HcclWorldTeamCreate
HcclTeamWindowRegister
HcclTeamCreateChannelsDescInit
HcclTeamChannelsCreate
HcclTeamWindowDeregister
HcclTeamDestroy
```

DeepEP 在 9.2 中没有直接调用 endpoint/MR 的显式 HCOMM API。endpoint、MR、token 和队列创建由 HCCL team/window/channel 封装完成。

### 4.2 team 创建参数

9.2 的 team 描述中，DeepEP 使用：

```text
rankIds              = 全部 rank ID
rankNum              = world size
selfRankId           = 当前 rank
protocol             = COMM_PROTOCOL_UBC_CTP
requirement.signalCount
requirement.counterCount
requirement.barrierCount
```

创建 world team 时会声明同步内存需求。DeepEP 为 channel 0 的 signal 和 barrier session 预留同步资源，counter 常为 0。

### 4.3 window 注册

DeepEP 先用 `aclrtMalloc` 分配设备内存，然后把这段内存作为 `COMM_MEM_TYPE_DEVICE` 的 `CommMem` 注册到 team：

```text
HcclTeamWindowRegister(comm, team, &memory, &windowHandle, flag)
```

成功后，`windowHandle` 指向设备侧可读的 `HcommWindow`。DeepEP 9.2 的兼容 ABI 将其解释为：

```text
memoryCount
memories          // CommMem 数组，按 member 索引
worldTeam
```

设备侧远端地址解析为：

```text
offset = logical_address - local_window_base
remote_address = memories[peer].addr + offset
```

同时校验本端和远端内存范围：

```text
offset + bytes <= memories[self].size
offset + bytes <= memories[peer].size
```

这个地址是 URMA 目标地址，不是 CPU 或普通 load/store 可以直接解引用的 peer 指针。

### 4.4 channel 创建

window 注册后，DeepEP 调用：

```text
HcclTeamCreateChannelsDescInit
HcclTeamChannelsCreate
```

使用的描述为：

```text
engine      = COMM_ENGINE_AIV
notifyNum   = 0
protocol    = COMM_PROTOCOL_UBC_CTP
channelCnt  = requested_channels
```

channel 建立后，`HcommTeam.channelsBaseAddr` 指向连续 `ChannelEntity` 数组，`channelCntAccumulatePerMember` 提供每个 member 的前缀偏移。设备侧按 peer 和 channel index 解析 `ChannelEntity`，再取得：

- local/remote registered buffer 表；
- SQ/CQ context 表；
- token；
- remote EID；
- queue depth 和 WQE 大小。

### 4.5 9.2 的设备侧上下文

DeepEP 的 `DeviceTransportContext` 中，两个 opaque 字段的含义是：

| 字段 | 9.2 含义 |
| --- | --- |
| `peer_address_table` | device-visible `HcommWindow*` |
| `channel_table` | device-visible `HcommTeam*` |

这并不是说它们只是普通整数表。它们是 CANN 运行时导出的设备侧资源描述句柄，AICore service 将其转换为 `__gm__` 指针后按 ABI 读取。

### 4.6 9.2 的生命周期

创建顺序必须遵守依赖关系：

1. communicator 有效；
2. team 创建成功；
3. 分配并注册 window；
4. 创建 channels；
5. 导出 device context；
6. 启动使用 transport 的 kernel。

销毁顺序相反：

1. 停止通信 kernel；
2. 销毁/释放 DeepEP 自有 service buffer；
3. deregister window；
4. destroy team；
5. 释放业务内存和 communicator。

9.2 中 channel 生命周期由 team 统一管理，DeepEP 没有 channel 级别的显式 destroy 调用。

## 5. CANN 9.3.0 编程模型

### 5.1 9.3 的变化概览

9.3 保留了 team/window/channel 的概念，但公开 API 和资源所有权发生了明显变化：

1. `HcclWorldTeamCreate` 变为更通用的 `HcclTeamCreate`；
2. team 描述新增 engine、notifyNum、channelCnt、共享队列配置；
3. channel 数量可以放进 team 创建描述；
4. symmetric window 走 `HcclCommSymWinGet/Register/Deregister`；
5. HCOMM 公开 endpoint、MR、channel 的显式创建接口；
6. `ChannelEntity` 可以通过 channel handle 对应的 device 指针读取；
7. ABI 使用 header/magic/version/size 做版本约束。

9.3 实际上有三类资源路径：

 - **team/window 路径**：更接近 9.2，但依赖 communicator 的 symmetric window 能返回可用 device-visible `HcommWindow*`；
 - **HCOMM explicit endpoint/MR/channel 路径**：调用者显式选择 endpoint、注册 MR、交换 endpoint/MR、创建 channel、轮询状态并读取 `ChannelEntity`；
 - **HCCL communication-domain 路径**：通过 `HcclCommMemReg` 注册内存，`HcclRankGraphGetLinks` 获取 rank 间真实链路，再用 `HcclChannelAcquire` 创建 AIV channel。

在当前测试的 950PR 组合包上，team API 可用，但旧 symmetric-window 导出路径未返回 DeepEP 9.2 路径所需的可用 `HcommWindow*`。手工枚举 endpoint 后调用 HCOMM explicit path 会停在 `CONNECTING`；**已验证的正确路径是 HCCL communication-domain 路径**：

```text
HcclCommMemReg
  ↓
HcclRankGraphGetLinks
  ↓
HcclChannelAcquire
  ↓
device-visible ChannelEntity*
```

### 5.2 `HcclTeamCreate`

9.3 的 `HcclTeamCreateDesc` 关键字段：

| 字段 | 含义 |
| --- | --- |
| `rankIds` | team 成员的 communicator rank 列表 |
| `rankNum` | 成员数量 |
| `selfRankId` | 本 rank ID |
| `netLayer` | 网络层，0 表示默认 |
| `protocol` | team 使用的协议 |
| `requirement` | signal/counter/barrier 同步内存需求 |
| `engine` | 通信引擎 |
| `notifyNum` | channel 所需 notify 数 |
| `channelCnt` | 请求的 channel 数 |
| `isSharedQueue` | 是否共享队列 |
| `sharedQueueTag` | 共享队列标签 |

DeepEP 当前把 channel count 直接写入 team 描述，并选择：

```text
protocol = COMM_PROTOCOL_UBC_CTP
engine   = COMM_ENGINE_AIV
```

9.3 也提供：

```text
HcclTeamGetLsaTeam
HcclTeamMemberToRank
HcclTeamRankToMember
HcclTeamDestroy
```

其中 LSA team 是 communicator 预建 team。运行时探测显示它可以返回合法 team handle，且 ABI、member 数和 self member ID 正确，但其数据面字段为空：

```text
channelsBaseAddr = null
channelCntAccumulatePerMember = null
remoteMems = null
remoteMemsNum = 0
shadowMem = 空
```

因此 LSA team 是内部/预建资源，不应被当作 DeepEP 可直接使用的 AIV 数据面 team。

### 5.3 symmetric window API

9.3 的 symmetric window API：

```text
HcclCommSymWinRegister(comm, addr, size, &winHandle, flag)
HcclCommSymWinDeregister(winHandle)
HcclCommSymWinGet(comm, ptr, size, &winHandle, &offset)
```

`HcommMemAlloc` 可以从 HCCL 预建对称 arena 中分配地址。DeepEP 9.3 尝试的路径是：

1. `HcommMemAlloc` 分配 window 内存；
2. `HcclCommSymWinGet` 查询 window 和 offset；
3. 将 window handle 交给设备侧地址解析。

探测结果：

- `HcommMemAlloc` 正常，多个 rank 返回相同的对称地址；
- `HcclCommSymWinGet` 返回 `ret=0`，但 `win=nullptr`、`offset=0`；
- `HcclCommSymWinRegister` 对 `HcommMemAlloc` 内存和 `aclrtMalloc` 内存均返回 `ret=9`。

因此不能把“API 存在且 team 创建成功”等价于“9.2 式 device-visible window 可用”。当前组合包中，DeepEP 旧路径缺少运行时提供的 device-visible `HcommWindow*`。

### 5.4 endpoint 与 rank graph API

9.3 提供两种与 endpoint 相关的入口，但适用场景不同。

#### HCOMM explicit endpoint API

```text
HcommEndpointGetDescNum(deviceLogicId, &descNum)
  ↓
分配 EndpointDesc 数组
  ↓
HcommEndpointGetDescs(deviceLogicId, &descNum, descs)
  ↓
选择匹配当前设备、协议和位置的 endpoint
  ↓
HcommEndpointCreate(&selectedDesc, &endpoint)
  ↓
与 peer 交换远端 EndpointDesc
  ↓
HcommChannelCreate(...)
```

NPU8P-ALT 的观察结果是每个逻辑设备 19 个 endpoint 描述，其中可选中当前设备的 UBC_CTP endpoint：

```text
protocol = COMM_PROTOCOL_UBC_CTP (4)
addrType = COMM_ADDR_TYPE_EID (3)
locType  = ENDPOINT_LOC_TYPE_DEVICE (0)
devPhyId = 当前逻辑设备 ID
```

但 2-rank 实测显示，手工选择第一个匹配 UBC_CTP 的 endpoint，再通过
`HcommEndpointCreate + HcommChannelCreate` 建链，两端 channel 会一直停留
在 `CONNECTING(1)`。即使显式设置 server/client role、端口和 channelName，
结果也相同。原因不是端口或角色错误，而是这 19 个 EID 代表不同通信路径，
手工选择的 endpoint 不一定是 rank graph 为当前 src/dst rank 选出的可达
直连链路。

#### HCCL rank graph API

对于复用 `HcclComm` 的自定义 transport，正确入口是 rank graph：

```text
HcclRankGraphGetLayers(comm, &netLayers, &layerNum)
HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &links, &linkNum)
```

`CommLink` 同时给出：

- `srcEndpointDesc`：本端 endpoint；
- `dstEndpointDesc`：远端 endpoint；
- `linkAttr.linkProtocol`：链路协议，例如 UBC_CTP、UBC_TP、UB_MEM。

实测 rank 0 与 rank 1 在 netLayer 0 有 2 条 link：一条 UBC_CTP，一条
UB_MEM。DeepEP 应优先选择 UBC_CTP；若为空，再评估 UBC_TP/UB_MEM，不能
直接按 endpoint 数组下标选择。

### 5.5 MR 与 channel 创建

9.3 PTO 提供了两个不同场景的参考实现：

- RoCE `hns_1825` backend 使用 HCOMM explicit endpoint/MR/channel API；
- URMA workspace manager 使用 HCCL communication-domain API，即本文验证
  成功的路径。

#### RoCE 参考：HCOMM explicit path

```text
EndpointDescInit
  ↓
HcommEndpointCreate
  ↓
HcommMemReg
  ↓
配置 HcommChannelDesc
  ↓
HcommChannelCreate
  ↓
轮询 HcommChannelGetStatus
  ↓
读取 ChannelEntity
```

`HcommChannelDesc` 的公共字段包括：

| 字段 | 含义 |
| --- | --- |
| `remoteEndpoint` | 远端 endpoint 描述 |
| `notifyNum` | notify 数量 |
| `exchangeAllMems` | 是否由 HCOMM 交换全部 MR |
| `memHandles` / `memHandleNum` | 显式传入的 MR |
| `socket` / `role` / `port` | 连接角色和端口 |
| `roceAttr` | QP、重传、QoS、SQ/SCQ 深度 |
| `hccsAttr` | HCCS QoS |
| `ubAttr` | UB SQ/SCQ 深度 |
| `ubMemAttr` | UB_MEM 路径模式 |
| `qos` | 协议无关 QoS |
| `channelName` | 两端业务匹配标识 |

该路径适合 RoCE 场景，但不适合直接照搬到本机 950PR 的 UBC_CTP 自定义
transport。

#### DeepEP 已验证路径：HCCL communication-domain API

```text
HcclCommMemReg(comm, memTag, &commMem, &memHandle)
  ↓
HcclRankGraphGetLinks(comm, layer, selfRank, peerRank, &links, &linkNum)
  ↓
选择 linkProtocol == UBC_CTP/UBC_TP 的 CommLink
  ↓
HcclChannelDescInit(&desc, 1)
  ↓
desc.remoteRank = peerRank
desc.channelProtocol = link.linkAttr.linkProtocol
desc.localEndpoint = link.srcEndpointDesc
desc.remoteEndpoint = link.dstEndpointDesc
desc.memHandles = &memHandle
desc.memHandleNum = 1
  ↓
HcclChannelAcquire(comm, COMM_ENGINE_AIV, &desc, 1, &handle)
```

与 HCOMM explicit path 不同，这条路径不需要手工创建 endpoint，也不需要
调用 `HcommChannelGetStatus` 轮询连接状态。`HcclChannelAcquire` 成功返回
后，AIV 的 `ChannelHandle` 已经是 device-visible `ChannelEntity*`。

2-rank 设备 0/1 实测结果：

| 项目 | 结果 |
| --- | --- |
| `HcclCommMemReg` | `ret=0` |
| `HcclRankGraphGetLinks` | `ret=0`，linkNum=2 |
| link 0 | UBC_CTP，src dev 0/1，dst dev 1/0 |
| link 1 | UB_MEM，src dev 0/1，dst dev 1/0 |
| `HcclChannelAcquire` | `ret=0`，handle=`0x120000040000` |
| `ChannelEntity.engine` | `COMM_ENGINE_AIV` |
| `ChannelEntity.protocol` | `COMM_PROTOCOL_UBC_CTP` |
| local/remote buffer 数量 | 各 2 条 |
| SQ/CQ 数量 | 各 1 条 |
| remote mem tags | `HcclBuffer`、`DeepEPUbcBuffer` |
| SQ WQE 大小/深度 | 64B / 32768 |
| SQ 类型 | UB JFS，包含 base/head/tail/doorbell/tpID/remoteEID |

buffer 数量为 2 的原因是：除 DeepEP 注册的业务内存外，HCCL 还自动加入了
内部 `HcclBuffer`。因此不能假设 buffer 表第 0 项就是业务 MR；必须用
`HcclChannelGetRemoteMems` 按 tag 匹配远端内存，再在
`ChannelEntity.remoteBufferAddr` 中按地址和大小匹配 `RegedBufferEntity`。

2-rank 探针已经证明这条资源建立路径可以走通：

| 阶段 | 结论 |
| --- | --- |
| 内存注册 | `HcclCommMemReg` 成功，业务 tag 可被远端看到 |
| rank link | `HcclRankGraphGetLinks` 返回 UBC_CTP 与 UB_MEM 两条链路 |
| endpoint 选择 | 使用 rank graph 给出的 src/dst endpoint，不做手工枚举选择 |
| channel | `HcclChannelAcquire` 成功返回 device-visible `ChannelEntity*` |
| 远端 MR | `HcclChannelGetRemoteMems` 能按 tag 定位 DeepEP 业务 MR |
| SQ/CQ | WQE/entry 布局、深度、doorbell、remote EID 和 token 均可读 |

这个结论只覆盖资源建立和 ABI 读取。DeepEP AICore service 通过这些
资源提交 write/inline write/FAA、等待 CQE、执行 flush/barrier 的完整数据面
还需要生产路径验证；不能把资源建立成功等价于典型 case 已经成功。

### 5.6 `ChannelEntity` 读取

```text
ChannelHandle 是 device-visible 指针
  ↓
aclrtMemcpy 或设备侧读取
  ↓
解释为 ChannelEntity
  ↓
读取 local/remote buffer、SQ/CQ context
```

对于 AIV + UBC_CTP，`HcclChannelAcquire` 返回的 handle 已经是
`ChannelEntity*`，可以直接从 device 侧读取：

```text
ChannelEntity
  ├─ localBufferAddr  → RegedBufferEntity[]
  ├─ remoteBufferAddr → RegedBufferEntity[]
  ├─ sqContextAddr    → SqContext[]
  └─ cqContextAddr    → CqContext[]
```

DeepEP 9.3 适配需要把这些 handle/device 描述组织成新的设备侧 context，
而不是继续把 `HcommTeam*` 当作唯一 channel 表。

### 5.7 9.3 的生命周期

DeepEP 采用的 HCCL communication-domain 生命周期是：

```text
初始化 ACL/设备/communicator
  ↓
分配业务内存
  ↓
HcclCommMemReg 注册业务 MR
  ↓
HcclRankGraphGetLayers/GetLinks 获取本 rank 与每个 peer 的可用链路
  ↓
按协议策略选择 UBC_CTP/UBC_TP link
  ↓
组装 HcclChannelDesc
  ↓
HcclChannelAcquire 创建 AIV channel
  ↓
读取 ChannelHandle 对应的 ChannelEntity
  ↓
HcclChannelGetRemoteMems 按 memTag 找业务远端 MR
  ↓
导出 DeepEP device context
  ↓
启动 AICore transport service
  ↓
停止 AICore transport service 并确认所有 SQ/CQ 已 drain
  ↓
释放 DeepEP 自建的 command、diagnostic、profile、channel 表等 staged 资源
  ↓
注销 command MR、同步 MR
  ↓
释放 HCCL communicator，由 communicator 释放 AIV channel
```

与 9.2 不同，业务 MR、rank link 和 channel handle 是调用者可见的资源；
但 CANN 9.3.0 的 HcclChannelDestroy 头文件注释写明“暂只支持 CCU
通信引擎创建的 channel”。DeepEP 和 PTO URMA 参考实现使用的都是
COMM_ENGINE_AIV，因此不能把 9.2 的“显式 destroy channel”语义迁移过来。

官方 PTO UrmaWorkspaceManager::Finalize() 也只释放自己的 workspace、
EID 表和 notify pool，然后清空 channelHandles_；它不会调用
HcclChannelDestroy。其注释要求“拥有通信上下文的调用方先停止内核、
drain QP，并销毁 HCCL 通信资源，再调用 Finalize”。这与实测一致：
2-rank put 通过后调用 HcclChannelDestroy 返回 backend=5，说明 AIV
channel 尚在 communicator 拥有的生命周期内，显式 destroy 是非法操作。

DeepEP 因此采用如下所有权规则：

| 资源 | 拥有者 | DeepEP teardown 动作 |
| --- | --- | --- |
| payload MR | DeepEP 注册，communicator 交换 | 释放 staged 资源后注销，再释放业务内存 |
| sync MR | DeepEP 注册，用于 signal/barrier | 在 command MR 后注销，再释放同步内存 |
| command MR | DeepEP 注册，只承载 device command | 先停止 service，最先注销，再释放 command 内存 |
| rank link endpoint 描述 | HCCL rank graph 查询返回 | 只读，不 destroy |
| AIV channel | HCCL communicator | 不调用 HcclChannelDestroy，随 communicator 释放 |
| DeepEP channel handle 表 | DeepEP 自建 | 清空表并释放设备内存 |

失败路径同样要先释放 channel 引用状态，再按 staged 资源、command MR、
sync MR 的顺序清理。不能在 channel 表仍引用 ChannelEntity* 时先释放
channel 表，也不能把业务 MR 先于 command/sync MR 注销。

#### 9.3 生命周期实测结论

1. HcclChannelAcquire(COMM_ENGINE_AIV) 返回的 handle 已经是
   device-visible ChannelEntity*，可作为设备侧 channel 表条目使用。
2. HcclChannelGetRemoteMems 必须按 memTag 匹配业务 MR，不能按
   buffer 表下标取。
3. channel 持有 MR 引用。DeepEP 释放自建 staged 资源后，再注销
   command MR 和 sync MR，最后处理 payload MR。
4. HcclChannelDestroy 对 AIV channel 返回 host backend 错误 5；不能
   作为重试或正常 teardown 的一部分。
5. AIV channel 交给 communicator teardown。业务侧只需停止使用 handle
   并清空自建表。

### 5.8 共享队列

9.3 team 描述和 channel 配置都提供共享队列能力。`HcclChannelConfig` 的
`IS_SHARED_QUEUE` 说明：

- 仅支持 AIV 的 UB 网络语义协议（UB_CTP/UBC_TP/UB_RTP）；
- 使用相同 endpoint 多次创建的 channel 共享一个 Jetty；
- 共享 Jetty 的 channel 不支持并发使用，需要调用者串行；
- destroy endpoint 前必须销毁所有共享 Jetty 的 channel。

这可以减少队列资源，但会牺牲独立 channel 的并发性。DeepEP 的多 channel
设计以硬件重叠为目标，默认应使用独立队列；只有在确认通信提交本身完全
串行时，才考虑共享队列。

### 5.9 DeepEP 9.3 的设备上下文方案

9.2 的 `DeviceTransportContext` 把 `channel_table` 和
`peer_address_table` 分别解释为 CANN 的 `HcommTeam*` 与 `HcommWindow*`。
9.3 不再导出这两个可用对象，DeepEP 需要把这两个字段重新解释为自己的
device-visible 表：

```text
DeviceTransportContext.channel_table
  -> DeepEP DeviceChannelTable*
     channels[peer * channel_count + channel] = ChannelEntity*

DeviceTransportContext.peer_address_table
  -> DeepEP DeviceWindowTable*
     remote_bases[peer] = peer 业务 MR 基地址
     remote_sync_bases[peer] = peer 同步 MR 基地址
```

推荐的最小表布局：

```cpp
struct DeviceChannelTable {
    std::uint32_t abi_version;
    std::uint32_t struct_size;
    std::uint32_t member_count;
    std::uint32_t self_member;
    std::uint32_t channel_count;
    std::uint32_t reserved;
    std::uintptr_t channels;       // ChannelEntity*[]
    std::uintptr_t remote_bases;   // uint64_t[]
    std::uintptr_t remote_sync_bases;
    std::uintptr_t local_sync_base;
    std::uint64_t window_bytes;
};
```

`ChannelEntity*` 本身保持 CANN 返回的 device-visible 指针，DeepEP 只建立
按 peer/channel 索引的连续表，不复制或重排 `ChannelEntity`。远端地址解析
改为：

```text
logical address - local_window_base
  -> window offset
  -> remote_bases[peer] + offset
  -> 在该 peer channel 的 remoteBufferAddr 表中匹配 RegedBufferEntity
  -> 使用该 entity 的 token_id/token_value 构造 URMA WQE
```

同步内存不能再依赖 team 的 `remoteMems`/`shadowMem`。DeepEP 需要单独分配
并注册一个同步 MR，每个 rank 的布局保持 9.2 的逻辑语义：

```text
logical signals[signal_index][source_member]
logical barriers[barrier_index][source_member]
```

设备侧通过 `remote_sync_bases[peer] + sync_layout 偏移` 找到远端同步槽，
通过 `local_sync_base + peer * 8` 作为 FAA fetch-result 的本地注册地址。
fetch-result 的 SGE 必须填写对应业务 MR 的 local token，不能使用未注册地址。

## 6. 两个版本通信模型对比

### 6.1 总表

| 维度 | CANN 9.2.0 | CANN 9.3.0 |
| --- | --- | --- |
| 总体风格 | HCCL 封装 team/window/channel | team 保留，同时提供 explicit endpoint/MR/channel 与 communication-domain channel API |
| world team 创建 | `HcclWorldTeamCreate` | `HcclTeamCreate` |
| channel 创建 | `HcclTeamChannelsCreate` | team `channelCnt`、`HcommChannelCreate`，或已验证的 `HcclChannelAcquire` |
| window 注册 | `HcclTeamWindowRegister` | `HcclCommSymWinRegister/Get`；当前包未返回可用 device window |
| endpoint | 对调用者隐式 | HCOMM 可显式枚举；DeepEP 应用 `HcclRankGraphGetLinks` 获取真实 src/dst endpoint |
| MR | team/window/channel 隐式管理 | `HcclCommMemReg` 显式注册并按 tag 交换 |
| device channel 表 | `HcommTeam.channelsBaseAddr` | `HcclChannelAcquire` 直接返回 device-visible `ChannelEntity*` |
| 远端地址解析 | `HcommWindow.memories[peer] + offset` | remote mem tag + remote `RegedBufferEntity` 地址/大小匹配 |
| 资源所有权 | HCCL/team 拥有大部分资源 | DeepEP 拥有业务 MR 与自建表；AIV channel 由 HCCL communicator 拥有 |
| LSA team | 未作为 DeepEP 主路径 | 可查询，但数据面字段为空 |
| 迁移影响 | team+window 两个设备句柄即可 | 需要 per-peer channel 表和新的业务 MR 选择逻辑 |

### 6.2 team 创建差异

9.2 的调用重心是“先创建 world team，再创建 window 和 channels”。9.3 的
team 描述把 engine、notify 和 channel 数量并入 team 创建参数，减少了
独立 channel desc 的必要性。

职责变化是：

- 9.2：channel 资源特征在 `HcclTeamCreateChannelsDesc` 中描述；
- 9.3：channel 数量和 engine 可以在 `HcclTeamCreateDesc` 中声明；
- 9.3 已验证路径：rank link、业务 MR 和 channel handle 由调用者管理。

### 6.3 window 与地址模型差异

9.2 DeepEP 路径中，window 是设备侧远端地址转换的核心。AICore service
只需要两个 CANN 句柄：`HcommTeam*` 和 `HcommWindow*`。

9.3 的 `HcommWindow` ABI 扩展为：

```text
netWin.baseRemoteMemAddr
netWin.windowSize
netWin.worldTeamAccumulateId
netWin.netLayerNum
lsaWin.baseVa
lsaWin.stride
lsaWin.userSize
legacySymWindow
```

但当前组合包无法从 `HcclCommSymWinGet/Register` 得到可用 device-visible
`HcommWindow*`。已验证的替代方式是通过 `HcclChannelGetRemoteMems` 找
业务 MR，再在 remote buffer 表中匹配 `RegedBufferEntity`。

### 6.4 endpoint 与 MR 的显式化

9.2 中 endpoint 和 MR 对 DeepEP 是黑盒：调用 team/window/channel API 后，
HCCL 内部完成端点选择、内存发布、token 交换和队列建立。

9.3 把这些资源暴露给调用者：

- 可以查询设备 endpoint 描述；
- 可以通过 rank graph 查询 src/dst rank 的真实链路；
- 可以显式注册业务 MR；
- 可以通过 channel desc 传入 MR handle；
- 可以读取 `ChannelEntity` 的 local/remote MR 表和 SQ/CQ context。

显式化的收益是可控性和跨机/跨协议扩展性；代价是生命周期复杂度、
rank 间交换协议和错误处理都转移到调用者。

### 6.5 channel 数据面差异

`ChannelEntity` 在两个版本中都是设备侧数据面核心，但获取方式不同：

```text
9.2:
  HcommTeam*
    -> channelsBaseAddr
    -> ChannelEntity[accumulated peer offset + channel index]

9.3 已验证路径:
  HcclCommMemReg
    -> HcclRankGraphGetLinks
    -> HcclChannelAcquire
    -> device ChannelEntity*
```

DeepEP 不能再把 `channel_table` 解释为 `HcommTeam*`。它需要新的设备侧
布局，例如按 peer/channel 保存 `ChannelEntity` 指针，或复制成自定义连续
数组。

### 6.6 SQ/CQ 与 URMA 的延续与变化

URMA 的基本执行模型在两个版本中是延续的：

```text
构造 SQE/SGE
  ↓
写入 SQ slot
  ↓
发布并 fence
  ↓
更新 SQ head
  ↓
st_dev 写 doorbell
  ↓
硬件访问远端 MR
  ↓
写 CQE
  ↓
软件轮询 CQE
  ↓
更新 CQ tail 并 doorbell
```

变化主要在资源获取和 ABI：

- 9.3 `ChannelEntity` 的 ABI header 更明确；
- 9.3 SQ/CQ context 分为 UB JFS/JFC 和 RoCE 两种布局；
- 9.3 channel desc 允许配置 SQ/SCQ 深度；
- 9.3 支持共享 Jetty，但共享队列不允并发；
- 业务 MR 必须按 tag/地址匹配，不能取 buffer 表第 0 项。

### 6.7 错误处理差异

9.2 的失败点主要是：

```text
create world team failed
register window failed
create channels failed
destroy team failed
```

9.3 的失败点更多：

```text
HcclCommMemReg 失败
HcclRankGraphGetLinks 失败
没有可用 UBC_CTP/UBC_TP link
HcclChannelAcquire 失败
ChannelEntity 读取失败
remote mem tag 匹配失败
remote RegedBufferEntity 匹配失败
SQ/CQ context 校验失败
token 或 remote EID 校验失败
```

因此 9.3 适配需要为每个阶段记录结构化状态，并在部分初始化失败时按反向
顺序清理。不能因为一个 channel 失败就跳过其它已创建资源的 destroy。

## 7. 对 DeepEP 的影响

### 7.1 9.2 路径的依赖

DeepEP 9.2 的设备侧通信实现依赖三个强假设：

1. `HcclTeamWindowRegister` 返回 device-visible `HcommWindow*`；
2. `HcclTeamChannelsCreate` 后 team 的 channel 表非空且可用；
3. `DeviceTransportContext.channel_table` 和 `peer_address_table` 分别
   可以解释为 `HcommTeam*` 和 `HcommWindow*`。

### 7.2 9.3 适配需要重做的部分

在已验证的 communication-domain 路径下，至少需要重做：

- host 资源状态机：业务 MR、rank link、channel handle 的创建/销毁；
- device context：不能继续复用 9.2 的 team/window 双句柄布局；
- 地址解析：从 window 转向 remote mem tag + remote buffer 表；
- channel 解析：从 team 前缀和表转向 per-peer `ChannelEntity*`；
- 生命周期：停止 service → 释放 staged 资源 → 注销 command/sync/payload MR → communicator 释放 AIV channel；
- 诊断：区分 MR、rank link、channel 和 `ChannelEntity` 解析错误。

### 7.3 可以保留的部分

以下部分与 9.2/9.3 的资源创建方式相对解耦，可以保留或小幅调整：

- backend-neutral `TransportTeam` 和 facade 语义；
- SIMT command ABI；
- AICore service 的 SQE/SGE/CQE 构造和 doorbell 流程；
- command buffer、diagnostic、stage profile；
- flush/drain 的完成语义；
- 多 channel 的 payload 切分策略；
- 生产者/队列并发约束。

## 8. 尚待验证的问题

资源建立路径已经验证成功；2-rank 探针已经确认 AIV/UBC_CTP channel、业务
MR、SQ/CQ、remote EID 和 token 都可被读出。以下问题仍需在 DeepEP 生产
路径中验证：

1. AICore service 直接消费 `HcclChannelAcquire` 返回的 `ChannelEntity*`；
2. remote 业务 MR 的地址与对称 offset 的组合方式；
3. signal/barrier 的同步内存来源：显式注册同步 MR，或复用 HCCL 内部资源；
4. 多 peer、多 channel 的资源布局和并发策略；
5. AIV channel 与 communicator teardown 的完整资源释放行为（已确认不能调用 `HcclChannelDestroy`，还需在更多 rank 数下回归）；
6. 完整 write/inline write/FAA payload 传输和 flush/drain 语义；
7. 8-rank 典型 case 的功能与性能。

## 9. 常见误区

1. **“team 创建成功等于数据面可用”**：不成立。LSA team 可以返回合法 handle，但 channel/sync 数据面字段可能为空。
2. **“window API 存在等于 9.2 模型可用”**：不成立。9.3 当前组合包上 `HcclCommSymWinGet` 可返回成功码但 window 为空。
3. **“endpoint 数组匹配协议就可以用”**：不成立。手工选 endpoint 可能不是 rank graph 为该 src/dst rank 选出的链路，实际会停在 `CONNECTING`。
4. **“channel 就是 socket”**：channel 还包含 MR、notify、SQ/CQ 和协议配置。
5. **“channel 数量等于队列深度”**：前者是并发路径数量，后者是单个队列可容纳的 WQE 数量。
6. **“拿到远端地址就能写”**：还必须找到覆盖该地址的 `RegedBufferEntity` 并取得 token，同时使用正确 remote EID。
7. **“buffer 表第 0 项就是业务内存”**：不成立。实测包含 HCCL 内部 `HcclBuffer` 和业务 `DeepEPUbcBuffer`。
8. **“FAA fetch-result 可以随便指向普通内存”**：fetch-result 必须是有效注册内存，否则没有 local token。
9. **“SIMT 可以直接敲 doorbell”**：当前 CANN 参考实现和 DeepEP 验证均指向 AICore `st_dev/ld_dev` 路径；SIMT producer 先记录命令。

## 10. 参考文件

### 本仓库

- `csrc/backends/ascend/transport/cann_transport.cpp`
- `csrc/backends/ascend/transport/cann_transport.hpp`
- `csrc/backends/ascend/transport/cann_compat.hpp`
- `csrc/backends/ascend/transport/aicore_transport_service.hpp`
- `csrc/backends/ascend/transport/urma_wqe.hpp`
- `csrc/backends/ascend/transport/device_transport.hpp`
- `docs/ascend-design/epv2-ascend-simt-urma-transport.md`
- `docs/ascend-design/epv2-ascend-transport-contract.md`
- `docs/ascend-design/epv2-ascend-multi-channel-design-spec-zh.md`
- git 历史：`662a41f^` 中保留了 CANN 9.2 路径实现。

### CANN 9.3 组合包

- `include/hccl/hccl_comm.h`
- `include/hccl/hccl_team.h`
- `include/hccl/hccl_res.h`
- `include/hccl/hccl_channel.h`
- `include/hccl/hccl_rank_graph.h`
- `include/hcomm/hcomm_res.h`
- `include/hcomm/hcomm_res_defs.h`
- `include/hcomm/hcomm_channel.h`
- `pkg_inc/hcomm/hcomm_team_entity_defs.h`
- `pkg_inc/hcomm/hcomm_res_entity_defs.h`
- `include/pto/comm/async/urma/urma_workspace_manager.hpp`
- `include/pto/comm/async/urma/urma_channel_helper.hpp`
- `pkg_inc/pto/comm/async/rdma/backends/hns_1825/*`

### 运行时证据

- NPU4Px2 CANN 9.2.0：DeepEP team/window/channel 路径已用于 4-rank 功能和性能验证。
- NPU8P-ALT CANN 9.3.0：手工 HCOMM endpoint/channel 路径停在 `CONNECTING`。
- NPU8P-ALT CANN 9.3.0：设备 0/1 上已验证 `HcclCommMemReg → HcclRankGraphGetLinks → HcclChannelAcquire` 返回可用 AIV/UBC_CTP `ChannelEntity`，并读取出业务 MR、SQ/CQ 和 token。
- NPU8P-ALT CANN 9.3.0：设备 0/1 上已通过完整 2-rank URMA case 列表；`HcclChannelDestroy` 对 AIV channel 返回 backend=5，PTO `Finalize()` 参考实现同样不销毁 AIV channel。
