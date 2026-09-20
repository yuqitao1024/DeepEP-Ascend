# CANN 9.3.0 通信接口变化清单（供 comm 对齐）

## 1. 结论摘要

CANN 9.3.0 相对 DeepEP 使用的 CANN 9.2.0 路径，核心变化不是简单的
“函数改名”，而是通信资源所有权从 team/window 封装模型变为更显式的
communicator/rank-graph/MR/channel 模型。

对 DeepEP 单机多卡 AIV/UBC_CTP transport 来说，已验证可用的 9.3 路径是：

```text
HcclCommInitRootInfoConfig
  -> HcclCommMemReg
  -> HcclRankGraphGetLayers / HcclRankGraphGetLinks
  -> HcclChannelDescInit
  -> HcclChannelAcquire(COMM_ENGINE_AIV, ...)
  -> HcclChannelGetRemoteMems
  -> device-visible ChannelEntity*
```

9.2 的 `HcclWorldTeamCreate -> HcclTeamWindowRegister ->
HcclTeamChannelsCreate` 路径不能直接迁移到当前 950PR 9.3.0 组合包。

## 2. Team 与 window 接口

### 9.2.0 已使用接口

```text
HcclWorldTeamCreate
HcclTeamCreateDescInit
HcclTeamWindowRegister
HcclTeamWindowDeregister
HcclTeamCreateChannelsDescInit
HcclTeamChannelsCreate
HcclTeamDestroy
```

9.2 的关键假设是：

- `HcclTeamWindowRegister` 返回 device-visible `HcommWindow*`；
- `HcommTeam.channelsBaseAddr` 提供连续 `ChannelEntity` 表；
- 远端地址由 `HcommWindow.memories[peer].addr + offset` 得到。

### 9.3.0 变化

```text
HcclTeamCreate
HcclTeamCreateDescInit
HcclTeamGetLsaTeam
HcclTeamMemberToRank
HcclTeamRankToMember
HcclTeamDestroy
```

`HcclTeamCreateDesc` 相比 9.2 增加了更多显式字段：

| 字段 | 含义 |
| --- | --- |
| `engine` | team/channel 使用的通信引擎 |
| `notifyNum` | channel notify 数量 |
| `channelCnt` | team 创建时请求的 channel 数 |
| `isSharedQueue` | 是否共享队列 |
| `sharedQueueTag` | 共享队列 tag |

9.3 还提供 communicator 级 symmetric window API：

```text
HcclCommSymWinRegister
HcclCommSymWinDeregister
HcclCommSymWinGet
HcommMemAlloc
HcommMemFree
```

并在 `HcclCommConfig` 中提供 symmetric window 相关配置，例如：

```text
hcclSymWinMaxMemSizePerRank
```

### 实测结论

在 NPU8P-ALT 的 CANN 9.3.0 组合包上：

- `HcclTeamCreate` 可以成功；
- `HcclTeamGetLsaTeam` 可以返回合法 team handle；
- LSA team 的数据面字段为空：

```text
channelsBaseAddr = null
channelCntAccumulatePerMember = null
remoteMems = null
shadowMem = null/empty
```

- `HcommMemAlloc` 可以分配对称地址；
- `HcclCommSymWinGet` 返回 ret=0，但 `win=nullptr`、`offset=0`；
- `HcclCommSymWinRegister` 对测试过的内存返回 ret=9。

因此，对当前组合包，不能把 9.3 symmetric window API 当作 9.2
device-visible window 的直接替代。

## 3. Rank graph 与 endpoint 接口

### 新增 rank graph 接口

```text
HcclRankGraphGetLayers
HcclRankGraphGetLinks
```

`HcclRankGraphGetLinks` 返回的 `CommLink` 包含：

- `srcEndpointDesc`：本端 endpoint；
- `dstEndpointDesc`：远端 endpoint；
- `linkAttr.linkProtocol`：协议，例如 UBC_CTP、UBC_TP、UB_MEM。

对复用 `HcclComm` 的自定义 transport，9.3 的正确 endpoint 选择方式是
通过 rank graph 获取当前 src/dst rank 的真实链路，而不是手工遍历 endpoint
数组后自行选择。

### 新增 HCOMM explicit endpoint 接口

```text
HcommEndpointGetDescNum
HcommEndpointGetDescs
HcommEndpointCreate
HcommEndpointDestroy
HcommEndpointGetListenPort
HcommEndpointCheckFeature
```

这些接口适合 RoCE/host socket 等显式建链场景。NPU8P-ALT 上手工选择
UBC_CTP endpoint 后走 `HcommEndpointCreate + HcommChannelCreate`，
channel 会停留在 `CONNECTING`。原因是手工选择的 endpoint 不一定是
rank graph 为该 rank pair 选择的真实直连路径。

## 4. MR 与远端内存接口

### 9.3 新增/显式化的 MR 接口

HCCL communication-domain 路径：

```text
HcclCommMemReg
HcclChannelGetRemoteMems
```

HCOMM explicit endpoint 路径：

```text
HcommMemReg
HcommMemUnreg
HcommMemExport
HcommMemImport
HcommMemUnimport
```

### DeepEP 实测结论

`HcclCommMemReg` 按 tag 注册业务内存，例如：

```text
DeepEPUbcBuffer
DeepEPUbcSync
DeepEPUbcCommands
```

`HcclChannelGetRemoteMems` 返回的远端 MR 表中，除业务 tag 外，还会包含
HCCL 内部 `HcclBuffer`。因此：

- 不能假设 buffer 表第 0 项是业务 MR；
- 必须先按 `memTag` 匹配远端 `CommMem`；
- 再在 `ChannelEntity.remoteBufferAddr` 指向的 `RegedBufferEntity[]`
  中按地址和大小匹配；
- 构造 WQE 时使用该 entity 的 token/protection 信息。

另一个重要差异：当前 9.3.0 包没有导出 DeepEP 可调用的
`HcclCommDeregMem`。DeepEP 目前依赖 `HcclCommDestroy` 释放 tag/MR，
不能在 communicator 仍存活时显式注销业务 MR。

## 5. Channel 接口

### 9.3 新增/显式化的 channel 接口

HCCL communication-domain 路径：

```text
HcclChannelDescInit
HcclChannelAcquire
HcclChannelAcquireWithConfig
HcclChannelQuery
HcclChannelGetRemoteMems
HcclChannelGetHcclBuffer
HcclChannelDestroy
```

Channel 配置接口：

```text
HcclChannelConfigCreate
HcclChannelConfigDestroy
HcclChannelConfigSetInt
HcclChannelConfigSetStr
```

HCOMM explicit endpoint 路径：

```text
HcommChannelDescInit
HcommChannelCreate
HcommChannelCreateWithConfig
HcommChannelGetStatus
HcommChannelDestroy
HcommChannelConfigCreate
HcommChannelConfigDestroy
HcommChannelConfigSetInt
HcommChannelConfigSetStr
```

### AIV channel 的关键语义变化

对于 AIV + UBC_CTP/UBC_TP：

- `HcclChannelAcquire` 成功后返回的 `ChannelHandle` 是 device-visible
  `ChannelEntity*`；
- DeepEP 不再需要通过 `HcommTeam.channelsBaseAddr` 查表；
- `HcclChannelDestroy` 的头文件注释说明当前“暂只支持 CCU channel”；
- 实测对 AIV channel 调用 `HcclChannelDestroy` 返回 backend error 5；
- CANN 自带 PTO `UrmaWorkspaceManager::Finalize()` 也只清空
  `channelHandles_`，不调用 `HcclChannelDestroy`。

因此 9.3 的 AIV channel 所有权是：

| 资源 | 拥有者 | DeepEP teardown 动作 |
| --- | --- | --- |
| AIV channel | HCCL communicator | 不调用 `HcclChannelDestroy`，随 communicator 释放 |
| DeepEP 自建 channel handle 表 | DeepEP | 停止使用后清空并释放 |
| payload/sync/command MR | DeepEP 注册，HCCL 交换 | 停止 service 后按依赖释放/交给 communicator destroy |

### 共享队列

9.3 的 `HcclChannelConfig` 支持 shared queue：

```text
IS_SHARED_QUEUE
SHARED_QUEUE_TAG
```

约束：

- 仅支持 AIV 的 UB 网络语义协议：UB_CTP/UBC_TP/UB_RTP；
- 相同 endpoint 的 channel 共享 Jetty；
- 共享 Jetty 的 channel 不支持并发使用，需要调用者串行；
- destroy endpoint 前必须销毁共享该 Jetty 的 channel。

DeepEP 默认使用独立 channel 以获得硬件重叠能力，暂不使用 shared queue。

## 6. Host/thread 接口

9.3 组合包导出了新的 host/thread API。DeepEP 当前生产路径未使用，
但它们体现了 9.3 host 侧编程模型的变化，需要 comm 明确哪些属于公开稳定
ABI。

### HCCL thread 资源

```text
HcclThreadAcquire
HcclThreadAcquireWithConfig
HcclThreadAcquireWithStream
HcclThreadExportToCommEngine
HcclThreadResGetInfo
HcclDedicatedThreadAcquire
HcclGetNotifyNumInThread
```

### HCOMM thread 资源

```text
HcommThreadAlloc
HcommThreadAllocWithConfig
HcommThreadFree
HcommThreadFreeWithStream
HcommThreadResGetInfo
HcommThreadResAcquireTimeOut
HcommThreadSupplementNotify
HcommThreadGetNotifyNum
HcommThreadExportToCommEngine
HcommThreadExportToCommEngineAiCpu
```

### HCOMM host 通信原语

```text
HcommReadOnThread
HcommReadNbiOnThread
HcommReadReduceOnThread
HcommWriteOnThread
HcommWriteNbiOnThread
HcommWriteReduceOnThread
HcommWriteWithNotifyOnThread
HcommWriteWithNotifyNbiOnThread
HcommWriteReduceWithNotifyOnThread
HcommBatchTransferOnThread
HcommLocalCopyOnThread
HcommLocalReduceOnThread
```

### HCOMM notify/fence/drain

```text
HcommThreadNotifyRecordOnThread
HcommThreadNotifyWaitOnThread
HcommThreadNotifyWaitOnThreadWithDefaultTimeout
HcommAclrtNotifyRecordOnThread
HcommAclrtNotifyWaitOnThread
HcommChannelNotifyRecordOnThread
HcommChannelNotifyWaitOnThread
HcommChannelNotifyWaitOnThreadWithDefaultTimeout
HcommChannelDrainOnThread
HcommChannelFence
HcommChannelFenceOnThread
HcommFenceOnThread
```

这些接口与 DeepEP 当前 AICore 内直接构造 URMA WQE 的路径不同。DeepEP
暂不切换，但需要确认它们是否是 9.3 推荐的 host 发起通信路径。

## 7. ABI 变化

9.3 的核心描述符引入或强化了 `CommAbiHeader`：

```text
version
magicWord
size
reserved
```

当前关键 ABI 版本：

| 结构 | 9.3 版本 |
| --- | --- |
| `HcclTeamCreateDesc` | v2 |
| `HcommTeam` | v1 |
| `HcommWindow` | v2 |
| `HcommChannel` / `ChannelEntity` | v5 |

注意：

- `HcclTeamCreateDescInit`、`HcommChannelDescInit`、`EndpointDescInit`
  会先填充 `0xFF`，调用者必须显式设置需要的字段；
- 不能假设未初始化字段为 0；
- channel ABI v5 增加 `ubAttr.scqDepth`；
- SCQ exclusive mode 仅支持 HOST/AICPU，AIV 不支持；
- 9.3 `HcommWindow` 拆分为 `netWin` 与 `lsaWin`，并保留
  `legacySymWindow`。

## 8. 与 9.2 的生命周期差异

### 9.2 顺序

```text
stop kernels
  -> free DeepEP staged resources
  -> HcclTeamWindowDeregister
  -> HcclTeamDestroy
  -> release communicator
```

### 9.3 已验证顺序

```text
stop AICore transport service
  -> drain SQ/CQ
  -> release DeepEP staged buffers/device tables
  -> release or hand off command/sync/payload MR
  -> HcclCommDestroy
  -> AIV channel released by communicator
```

CANN PTO `UrmaWorkspaceManager::Finalize()` 的注释也要求：

```text
owning communication context must:
  1. stop kernels
  2. drain QPs
  3. destroy HCCL communication resources
then call workspace Finalize()
```

## 9. 当前需要 comm 确认的问题

1. **AIV channel destroy**
   - `HcclChannelDestroy` 是否确定不支持 AIV？
   - AIV channel 是否只能由 `HcclCommDestroy` 释放？
   - 后续版本是否计划补 AIV 显式 destroy？

2. **多 communicator 同进程**
   - 同一进程同时存在 Torch HCCL communicator 与 DeepEP 自建 communicator；
   - 当前 12-case 后偶发退出阶段 SIGSEGV；
   - core 栈在 HCOMM 后台线程 `RaDeinit -> RaPeerDeinit -> RsDeinit -> free()`；
   - 是否存在 communicator 全局 teardown 顺序/同步要求？

3. **HCOMM 全局反初始化**
   - plog 显示 `RaDeinit` 可能早于最终 `HcclCommDestroy` 完成；
   - 是否有 host API 需要等待或显式 finalize？
   - `HcomExecFinalize` 是否只属于 graph executor，不适合本场景？

4. **MR 注销**
   - 当前包未导出 `HcclCommDeregMem`；
   - 业务 tag/MR 是否只能等 `HcclCommDestroy`？
   - 是否计划提供 communicator 存活期间的显式 deregister？

5. **Symmetric window**
   - `HcclCommSymWinGet` 返回成功但 window 为空，是否为预期？
   - 自定义 transport 应使用 SymWin API 还是 rank graph + MR/channel？

6. **rank graph 协议选择**
   - 同一 rank pair 可能返回 UBC_CTP、UBC_TP、UB_MEM；
   - 是否有官方选择策略或能力查询？

7. **host/thread API**
   - 新增 HCOMM host 原语是否公开稳定？
   - 对 AIV one-sided transport，推荐 host thread API 还是设备侧直接构造
     URMA WQE？

8. **ChannelEntity ABI**
   - `HcclChannelAcquire` 返回值直接作为 device-visible `ChannelEntity*`
     是否是稳定 ABI？
   - v5 后续是否保证向前兼容？

## 10. 建议给 comm 的最小描述

可以这样开场：

```text
我们不是在用 9.2 的 team/window API 做 9.3 兼容，而是在 950PR 9.3.0
组合包上验证了 HcclCommMemReg -> HcclRankGraphGetLinks ->
HcclChannelAcquire(COMM_ENGINE_AIV) -> HcclChannelGetRemoteMems 的
communication-domain 路径。AIV channel 不调用 HcclChannelDestroy，交给
HcclCommDestroy 释放。当前需要确认：AIV channel 与多 communicator 场景
下的全局 teardown 顺序，以及业务 MR 是否只能随 communicator destroy 释放。
```
