# DeepEP-Ascend SIMT通信技术文章大纲（v0.3）

日期：2026-10-13
状态：正文已按大纲完成，当前为文案润色版本

## 1. 文章定位

这篇文章的主线是回答一个问题：MoE通信中为什么适合使用SIMT？

读者无需预先熟悉通信库实现。文章先说明MoE业务：Router为每个token选择expert，expert分布在多个rank上，因此必须执行Dispatch和Combine。随后拆解这两个操作的业务流程，标注哪些阶段由SIMT承担，最后解释这些阶段为什么天然适合SIMT。

写作边界：

1. 正文重点解释“为什么使用SIMT”，而非逐段说明SIMT的实现方式；
2. 通信专业细节只保留能支撑结论的最小集合，其余移到术语表或删除；
3. 不把SIMT描述成更快的数据搬运引擎；它处理的是动态路由、统计、去重、位置计算和描述符构造，payload仍交给URMA/MTE；
4. 准确表述提交边界：SIMT并行构造通信描述，scalar统一维护提交边界，不是每个lane独立提交。

## 2. 核心主张

MoE通信的关键并非数据搬运本身，而在于先回答四个问题：

1. 这个token要发给谁？
2. 同一个目标rank上有哪些token？
3. 对端接收时放在什么位置？
4. 如何将这些不规则答案转换成硬件能执行的通信描述？

这些问题来自每个token的top-k结果，具有动态、稀疏、不均衡三个特点。SIMT适合在Device侧直接处理它们：

1. token天然可并行：多个token的路由可以同时扫描；
2. 分支和归约天然匹配：目标统计、重复rank去重、前缀位置计算是典型的比较、筛选和归约；
3. 字段构造可并行：一条通信描述符里的目标、长度和地址可按lane分工填写；
4. 多AIV可独立推进：每个AIV处理一段token并拥有独立的待提交描述区间，避免所有producer集中在同一提交点。

概括而言：SIMT负责将不规则的路由结果转换为规则的通信描述；URMA/MTE负责按描述搬运payload。

## 3. 目标读者

- 了解基础深度学习，但不熟悉MoE通信实现的工程师；
- 关心Ascend执行模型、SIMT和通信优化的开发者；
- 想判断SIMT是否适合自己业务的技术决策者。

默认读者知道token、expert、rank等基础词；不假设熟悉通信库实现名词。首次出现时只用一句话解释，详细定义放术语表。

## 4. 推荐标题

1. DeepEP-Ascend：SIMT为什么适合MoE通信？
2. 从MoE路由到通信描述：SIMT在做什么？
3. Dispatch和Combine里，哪些工作天然适合SIMT？

## 5. 正文结构

### 0. 导语：MoE通信的动态决策

用三句话引出主题：

1. MoE让每个token走少量expert；
2. expert分布在多个rank上，所以需要Dispatch和Combine；
3. Router的top-k结果每一步都变，通信必须先动态决定“发给谁、放哪里、如何描述”。

结论前置：DeepEP-Ascend把这段动态决策留在Device侧，用SIMT完成，而不是把路由结果送回Host再生成通信计划。

### 1. MoE通信的业务背景

面向通信初学者，先以业务语言介绍：

- MoE：Router为每个token选择top-k expert；
- EP：expert分布在不同rank上；
- Dispatch：把token发送到expert所在rank，并整理成expert可消费的布局；
- Combine：把expert输出送回token原始rank，并聚合top-k结果。

本节不展开通信库结构，不出现队列和描述符实现词。

### 2. Dispatch与Combine的4-rank示例

用一个固定例子建立直觉：4个rank，每个rank本地8个token，top-k简化为2。例子完整覆盖8个token，但只保留业务信息，不展开队列、描述符字段等通信实现。

#### 2.1 Dispatch：从Router结果到expert输入

给出rank 0的8个token完整路由表：token、选中的expert、expert所在rank、Dispatch操作。再从表推导业务结论：哪些token保留在本地、每个远端rank接收哪些token。

说明去重只作用于同一token的hidden数据：如果两个expert在同一rank，hidden只发送一份；接收端再按expert信息展开为多个expert输入。以16个expert选择转换为14份hidden发送的示例，说明发送去重与expert展开的边界。

由此引出Dispatch的通用流程：

1. 读取每个token的top-k结果；
2. 按目标rank统计数量；
3. 对同一token的重复rank去重；
4. 计算对端接收slot；
5. 构造通信描述；
6. 提交并让硬件搬运payload。

#### 2.2 Combine：从expert输出到token输出

再以rank 0为接收方，给出8个token的返回来源表：每个原始token有哪些expert输出、来自本地还是远端rank。说明Combine需将这些输出按top-k权重聚合为一个token输出。

1. 读取返回metadata：source rank、原始token、expert和slot；
2. 按source rank组织返回工作，减少零散提交；
3. 远端输出生成远端写描述；
4. 本地输出直接聚合；
5. 按原始token和slot完成加权reduce。

最后总结共同点：Dispatch把token到expert的路由整理成发送清单；Combine把expert输出到原始token的返回关系整理成聚合计划。通信细节统一抽象为清单、描述、提交和搬运。

本节放一张全流程图，重点标注哪些阶段是SIMT，哪些是scalar/URMA/MTE。图先用正文文字流程图表达，不新增图片文件。

### 3. Dispatch中的SIMT阶段

按业务阶段做小节，而不是按通信实现名词做小节。

#### 3.1 路由扫描：token天然并行

解释SIMT优势：多个token可并行扫描；top-k是每token独立分支；无需把路由结果搬回Host。

#### 3.2 目标统计与去重：分支与归约

解释rank统计、重复rank去重、接收位置计算。强调这些是比较、筛选、前缀和归约，正是SIMT擅长的形态。

#### 3.3 通信描述构造：动态路由的固定表达

用“通信描述”概括底层描述符，不在正文展开字段布局。解释一条描述包含目标、长度和地址，字段可以按lane并行填写；这一步把动态业务结果转换为硬件可消费的固定格式。

#### 3.4 提交与搬运：各执行路径分工

说明提交边界由scalar统一维护，payload由URMA/MTE搬运。SIMT的价值是让多个AIV同时完成前三个阶段，并独立构造自己的通信描述。

### 4. Combine中的SIMT阶段

#### 4.1 返回信息整理：source rank、token与slot

解释Combine的输入比Dispatch更分散：expert输出可能来自不同rank，还需要回到原始token位置。SIMT适合并行整理这些metadata。

#### 4.2 task组织：零散结果的批次化

用业务语言解释task：同一source rank的一组token结果。SIMT可并行读取metadata、组织task并生成远端写描述。

#### 4.3 流水执行：通信准备与本地reduce重叠

解释通信准备和本地聚合可以形成流水线：当前task聚合时，下一个task的描述已在构造。此处说明“重叠的具体阶段”，不展开stage实现细节。

### 5. 为什么SIMT适合MoE通信

先归纳三个优势来源：

1. 结构契合：token天然并行，统计、去重和slot计算是分支加归约；
2. 表达自然：通信描述字段可按lane分工填写，动态路由能直接映射到线程组织；
3. 并发友好：多个AIV同时整理路由并生成描述，避免集中在单一提交点上。

再用总结表回扣全文，列中包含“工作、优势来源、执行路径、原因”。

补充SIMT与SIMD的对比：SIMD适合规则、连续、少分支的向量计算；SIMT允许线程走不同分支，并通过筛选、交换和归约原语协作。MoE通信中的目标rank差异、重复rank去重、动态slot和字段构造更符合SIMT形态。

结论：SIMT的优势不是替代通信硬件，而是把“每个token都不同的业务意图”转换成硬件能执行的规则描述。

### 6. 性能证据与适用边界

保留DeepEP-Ascend官方公开数据，但压缩解释：

- 说明统计口径包含issue和drain、排除最终epilogue；
- 给出EP8-EP128的Dispatch/Combine带宽；
- 只做边界说明，不展开内部任务和日志。

### 7. 结语

回扣四个要点：动态路由、分支归约、字段并行、多AIV独立推进。结尾强调通信不只是payload搬运，还包括将不规则路由结果转换为规则描述。

## 6. 配图与图示策略

优先复用已有图片，不新增图片处理。复杂业务流程必须有图；SIMT并发逻辑用Host/Device架构图或正文流程图表达，避免只靠文字：

1. MoE routing图：解释token到expert/rank的动态映射；
2. 4-rank MoE flow图：完整展示Dispatch和Combine的双向业务流程；
3. Dispatch数据流图：用Router结果、SIMT整理、通信描述和硬件路径表达动态路由到通信描述的转换；
4. Combine数据流图：用expert输出、SIMT返回整理、远端/本地路径和加权reduce表达返回聚合；
5. Why SIMT图：对比SIMD、SIMT线程分支与AIV并发模型；
6. Host/Device架构图：用资源准备、路由读取、SIMT整理、通信描述、提交和payload传输说明边界。

全套图统一使用轻字重、细线条和活泼但克制的现代信息图风格；图3、图4、图6避免出现WQE、SGE、SQ、CQ、doorbell、Jetty、Channel等实现名词，用业务阶段和抽象通信角色表达。

4-rank全流程图先用正文文字流程图表达，重点用`[SIMT]`标签标注阶段。该图必须保留，作为读者定位SIMT阶段的锚点。若后续确需替换为图片，再单独评审，不在本轮处理。

## 7. 写作红线

1. 不假设读者熟悉通信库；
2. 不满篇讲解通信实现细节；
3. 不写“每个lane独立ring doorbell”；
4. 不写“SIMT替代URMA/MTE”；
5. 不宣称所有MoE场景都有同样收益；
6. 不引用内部任务ID、日志路径或私有环境；
7. 不把公开报告和GitHub仓库混成同一实现。

## 8. 术语处理

正文首次出现时只保留一句话解释：

- rank：分布式训练中的一个参与方；
- top-k：Router为每个token选出的k个expert；
- slot：接收端为token预留的位置；
- SIMT：同一指令流驱动大量线程，适合并行处理独立数据；
- AIV：Ascend设备上的执行单元；
- URMA：负责远端写入的通信硬件路径；

术语表只解释正文实际出现的术语。正文没有使用的Jetty、SQ、CQ、doorbell、WQEBB等实现名词不进入术语表。
