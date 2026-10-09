# netlayer=0 适配定位进展

日期：2026-10-09。远端诊断仓：NPU8P 的
/data/disk2/pyptouser/yuqitao/deepep-official-20261008-reinstall。
本文只记录当前定位进展和证据，不作为设计方案；本地生产代码未因本诊断修改。

## 背景

官方 DeepEP-Ascend 路径假设使用 netlayer=1。当前环境无法按官方要求配置
netlayer=1，因此在远端官方源码上使用编译宏 DEEP_EP_NETLAYER=0 做适配验证。
目标是先回答 netlayer=0 是否能承载官方 dispatch/combine 的 SIMT/URMA 并发模型，
再考虑功能范围或性能对齐。

环境与命令口径：

- 直连 NPU8P：ssh -p 10135 yuqitao@123.60.114.238，不使用 NPU8P-ALT。
- CANN：/usr/local/Ascend/cann-9.3.0。
- Python：/data/disk2/pyptouser/yuqitao/deepep-venv-py311/bin/python。
- GCC：/data/disk2/pyptouser/yuqitao/toolchain/gcc-15.2.0-install。
- elfutils 前缀：/data/disk2/pyptouser/yuqitao/deepep-deps/elfutils-prefix。
- NPU 任务必须通过 task-submit 申请；本阶段使用 2 卡做最小复现。

## 已确认事实

### 1. layer 选择错误会导致初始化超时

官方 host 代码硬编码使用 layers[1]。在 netlayer=0 场景下继续取
layers[1] 会导致 HcclChannelAcquire 等 120 秒超时。远端已按编译宏改为：

    #if DEEP_EP_NETLAYER == 0
    const auto layer = layers[0];
    #else
    const auto layer = layers[1];
    #endif

这是进入后续用例的必要条件。

### 2. JIT 宏最初没有传播，现已修复并验证

setup.py 给 host extension 定义了 DEEP_EP_NETLAYER=0，但 DeepJIT
生成 device kernel 时不会继承 host 编译宏。最初 JIT 生成的
kernel.asc 中是：

    #if defined(DEEP_EP_NETLAYER_VALUE)
    #undef DEEP_EP_NETLAYER
    #define DEEP_EP_NETLAYER DEEP_EP_NETLAYER_VALUE
    #elif !defined(DEEP_EP_NETLAYER)
    #define DEEP_EP_NETLAYER 1
    #endif

由于 DEEP_EP_NETLAYER_VALUE 在 device 编译中未定义，实际仍落到默认值 1。
因此早期“device 侧已按 netlayer=0 适配”的判断不可靠。

远端修复方式是让 host wrapper 把宏值作为 std::format 参数写入 JIT
源码，生成文件中显式出现：

    #undef DEEP_EP_NETLAYER
    #define DEEP_EP_NETLAYER 0

已验证的生成文件示例：

/tmp/deepep-jit-netlayer0-macro/cache/dispatch.32717f58754f1050ff1ee2f6776d034d/kernel.asc。

同时保留 extra_signature="netlayer=0" 隔离 JIT 缓存，避免 netlayer=0
与 netlayer=1 共用缓存。

### 3. 共享 Jetty 初始化可以成功，但 correctness 仍失败

netlayer=0 下不能沿用官方每 AIV 一个 Jetty 的 64 个资源模型，否则会耗尽
stream/SQ 资源。远端曾尝试：

- host 侧创建 1 个共享 Jetty；
- 每个 peer 建 1 条 channel；
- device 表布局改为 [shared Jetty, peer 0, peer 1, ...]；
- dispatch/combine 中 netlayer=0 固定使用 Jetty index 0。

该配置可完成初始化并进入 kernel，说明 layer=0 的 endpoint、channel 和
Jetty 建链本身是可用的。

但 2 卡 correctness / smoke 用例仍在 dispatch 阶段稳定失败：

    DeepEP HCOMM scalar CQE failed, current tail: 0, status: 6, substatus: 0

失败发生在第一个测试 case 的 dispatch kernel 中，64 个 AIV block 同时报错。
该现象在 --skip-check 下也存在，因此不是正确性比对逻辑触发的失败，
而是 dispatch kernel 自身的数据面通信失败。

### 4. 根因基本收窄到共享 Jetty 的 SQ 并发冲突

官方 netlayer=1 模型是每个 AIV 独占一个 Jetty/SQ：

    handle::HcommJettyInfo::load_jetty_info(..., vec_core_idx, lane_idx);
    handle::HcommJetty jetty(..., vec_core_idx);

每个 AIV 从自己的 SQ head 开始写自己的 SQE，最后各自 ring doorbell。

当前 netlayer=0 适配改成所有 AIV 使用同一个 Jetty index 0。dispatch 中
每个 AIV 的 SIMT worker 仍按本 AIV 的局部 lsqe_counter / psqe_idx 计算
SQE 位置：

    const auto slot_head =
        static_cast<uint32_t>(ub_layout->jetty_info.packed_head) +
        psqe_idx * kNumWQEBBsPerSQESlot;

由于 64 个 AIV 加载的是同一个共享 SQ head，且每个 AIV 的 psqe_idx
从 0 开始，它们会同时写同一个物理 SQE 区间。第一个 SQE 位置被写 64 次，
AIV 0 只按自己的局部 SQE 数推进 head 并 ring doorbell，无法代表其他 AIV
写入的 SQE。因此 hardware CQE 返回 status=6 是符合该写入模式的预期结果。

### 5. CANN 接口契约明确不支持共享 Jetty 并发

CANN 9.3.0 的 hcomm_channel.h 对
HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE 的说明包含：

    共享 Jetty 的不同 Channel 不支持并发使用，需由调用者按业务顺序串行调用。

当前“64 个 AIV 同时写同一个共享 Jetty”不仅存在 SQ 覆盖，还违反了该接口
契约。因此仅调整 SQE 全局偏移或只让 AIV 0 ring doorbell 都不足以证明方案
成立；还需要处理共享队列的所有权和提交顺序。

## 当前远端诊断改动范围

远端仓库存在未提交诊断改动，主要包括：

- setup.py：netlayer=0 时增加 host 编译宏；
- csrc/comm/handle.hpp：host 按 layer 0/1 选择 layers[0] 或 layers[1]；
- csrc/kernels/comm/hccl.hpp：netlayer=0 创建 1 个共享 Jetty、每 peer
  1 条 channel；
- deep_ep/include/deep_ep/comm/handle.hpp：device 表布局注释和 peer
  表偏移；
- deep_ep/include/deep_ep/impls/ep/dispatch.hpp：netlayer=0 固定使用
  Jetty index 0，并尝试让 AIV 0 负责共享 doorbell；
- deep_ep/include/deep_ep/impls/ep/combine.hpp：netlayer=0 固定使用
  Jetty index 0；
- deep_ep/include/deep_ep/comm/barrier.hpp：netlayer=0 的 drain 策略；
- csrc/runtime/jit.hpp 与三个 JIT wrapper：显式生成
  #define DEEP_EP_NETLAYER 0，并隔离缓存签名。

这些改动仍处于诊断阶段，尚未证明 correctness，也不应直接作为生产适配方案。

## 复现命令

构建使用独立 JIT 缓存，避免旧缓存误导：

    export DEEP_EP_PLATFORM=ascend
    export DEEP_EP_DISABLE_TORCH_COMPILE=1
    export DEEP_EP_ASCEND_RELEASE_SIGNAL_ONLY=1
    export DEEP_EP_NETLAYER=0
    export EP_JIT_CACHE_DIR=/tmp/deepep-jit-netlayer0-macro
    export PATH=/data/disk2/pyptouser/yuqitao/toolchain/gcc-15.2.0-install/bin:$PATH
    export CPATH=/data/disk2/pyptouser/yuqitao/deepep-deps/elfutils-prefix/include:$CPATH
    export LIBRARY_PATH=/data/disk2/pyptouser/yuqitao/deepep-deps/elfutils-prefix/lib:$LIBRARY_PATH
    export LD_LIBRARY_PATH=/data/disk2/pyptouser/yuqitao/toolchain/gcc-15.2.0-install/lib64:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/data/disk2/pyptouser/yuqitao/deepep-deps/elfutils-prefix/lib:$LD_LIBRARY_PATH

    python setup.py build_ext --inplace

2 卡最小复现：

    python tests/ep/test_ep.py \
      --num-processes 2 \
      --num-ai-cores 32 \
      --num-tokens 16384 \
      --hidden 7168 \
      --num-topk 6 \
      --num-experts 256 \
      --dispatch-dtype bf16 \
      --test-first-only \
      --skip-check \
      --skip-perf-test

即使带 --skip-check，当前仍会触发 HCOMM scalar CQE failed；
--skip-check 只是排除了正确性比对路径，不排除 kernel 内通信失败。

## 关键判断

netlayer=0 不是把 layers[1] 改成 layers[0] 即可完成的适配。它至少涉及：

1. endpoint layer 选择；
2. Jetty/channel 资源数量；
3. device Jetty 表布局；
4. JIT 宏传播；
5. 共享 Jetty 的 SQ/CQ 所有权；
6. 多 AIV 并发写 SQ 的协调或串行化；
7. dispatch 与 combine 两套提交路径的适配；
8. barrier/drain 路径的语义。

官方 netlayer=1 实现的并发模型依赖“每 AIV 独占 Jetty/SQ”。netlayer=0
如果只有一个共享 Jetty，则与该模型存在结构性冲突。仅做全局 SQE 分配
仍需验证 HCOMM 是否接受多 producer 写同一 SQ，以及 CQ 仲裁和 head/tail
推进是否正确；从当前 CANN 接口注释看，这条路没有被接口契约支持。

## 下一步建议

按优先级继续定位：

1. 资源上限实验：减少物理 Jetty 数量，并让 AIV 数量与之匹配，验证
   netlayer=0 实际能创建多少独立 Jetty。如果资源上限远低于 64，需要确认
   官方 netlayer=1 与当前环境的资源差异是否来自 layer、驱动或 CANN 版本。
2. 最小共享队列验证：写一个只使用 1 个 AIV 的 dispatch 变体，确认
   layer 0 的单个 Jetty 数据面可以完成 URMA 写。如果单 producer 成功，
   再逐步增加 producer，用于区分“资源/队列配置问题”和“并发问题”。
3. 串行 producer 方案评估：如果确认共享 Jetty 必须串行，需要评估将
   dispatch/combine 的多 AIV 写 SQ 改成单 producer 聚合或全局调度的工作量
   和性能代价。
4. 不要继续把宏传播当作当前失败原因：该问题已修复并从生成源码验证；
   后续失败应优先从共享 Jetty 并发模型分析。
5. 完整 correctness 之前不跑性能结论：当前 --skip-check 用例失败，
   不能用该状态测试或解释性能数据。
