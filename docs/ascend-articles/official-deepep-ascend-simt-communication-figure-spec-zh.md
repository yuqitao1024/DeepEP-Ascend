# DeepEP-Ascend SIMT 文章配图文字版规格

## 目标

这版规格用于承接文章配图重构，先确定每张图的业务目标、文案、布局、颜色和放置位置，再生成图片。

当前规划共有7张图：第1章2张、第2章2张、第3章1张、第4章1张、第5章1张。第1章和第2章只讲业务数据流；SIMT 执行方式从第3章开始引入。

## 当前状态

| 事项 | 状态 |
| --- | --- |
| 图1到图7的文字版规格 | 已完成 |
| 图1到图7的生成提示词 | 已完成 |
| 正文图片引用和图注替换 | 已完成，正文引用7张新图 |
| 新图片生成 | 已完成，均使用内置imagegen生成与编辑 |
| 旧图片清理 | 已完成，6张旧图及本地备份已删除 |
| 生成后逐图检视 | 已完成，核对案例、角色与图文衔接 |

处理节奏保持不变：先完成并确认全部文字版设计，再按章节逐张生成、替换和检视图片。

## 全局风格

- 主题：Gruvbox Light
- 画布：16:9
- 背景：浅米色 #fbf1c7
- 卡片：白色 #ffffff
- 主文字：深灰 #3c3836
- 次文字：浅灰 #665c54
- 边框：1px，浅灰 #d5c4a1
- 字体：Noto Sans SC / Inter
- 字重：Light 或 Regular，禁用 Bold
- 风格：扁平、细线、圆角、留白充足
- 不使用 3D、阴影、渐变、拟物质感

### 颜色语义

| 角色 | 颜色 | 色值 |
| --- | --- | --- |
| SIMT | 青蓝色 | #458588 |
| scalar | 紫色 | #b16286 |
| URMA | 橙色 | #d65d0e |
| MTE | 绿色 | #98971a |
| token | 黄色 | #d79921 |
| expert | 红色 | #cc241d |
| rank | 蓝紫色 | #8f3f71 |

### 禁用词

图片中不出现 WQE、SGE、SQ、CQ、queue、doorbell、Jetty、Channel、staged transport、payload、hidden、metadata。

## 参考风格

参考 CANN《AscendC高性能自定义通信编程参考实践》中第3、4、5张图的风格：三列或三段式 rank 泳道、数据流从左到右、token / expert 实例化、通信路径与计算路径分色。图形应保持浅色 Gruvbox Light 基调，卡片和连线有明确业务语义，信息密度可以较高，但阶段边界必须清楚。

## 设计节奏

1. 先按章节推进，一次只设计一张图。
2. 每张图先写清业务目标、放置位置、画面结构、全部文案、颜色语义和禁止项。
3. 文字版确认后，再生成提示词和图片。
4. 高信息密度图优先采用 HTML/CSS 或分块绘制后拼接，不直接一次性生成。
5. 图注只解释业务含义，不描述画面构图。

## 图片重排总原则

1. 先讲业务，再讲实现；图先出现在对应概念首次需要被理解的段落，而不是章节末尾。
2. 第1章先给出 MoE 业务闭环，再解释路由动态性；不把具体 token / rank 案例提前到第1章。
3. 第2章用一个与 4-rank case 完全一致的 Dispatch 数据流图和一个 Combine 数据流图承接文字表格。
4. 图注只解释业务含义，不描述画面构图，例如禁止写“左侧展示”“连线终止在”“节点颜色标明”。
5. 信息密度高的图优先采用分块绘制再拼接，或使用 HTML/CSS 布局后截图，避免直接依赖一次性图片生成。

## 图1：MoE通信的业务闭环

### 目标

用一张图回答“Router、Dispatch、Expert/FFN、Combine 分别在业务里做什么”。读者先看到 MoE 的完整业务闭环，再理解为什么需要通信。

### 放置位置

放在第1章第一段之后、Dispatch / Combine 定义之前。它应是全文第一张图，避免读者先陷入 token 到 expert 的具体映射。

### 画面结构

采用三段式横向流程，左右两个模型阶段卡片夹住中间的 MoE 通信区域：

1. 输入与路由：输入 token、Router、top-k expert 选择；
2. MoE通信与计算：Dispatch、Expert / FFN、Combine；
3. 输出：输出 token。

### 具体文案

主标题：MoE通信的业务闭环

左列卡片：

- 输入 token
- Router
- top-k expert 选择

中间区域标题：MoE Expert Parallel

Dispatch卡片：

- Dispatch
- 发送 token
- 到 expert 所在 rank

Expert卡片：

- Expert / FFN
- 分散在不同 rank
- 并行计算

Combine卡片：

- Combine
- 返回 expert 输出
- 按权重聚合

右列卡片：

- 输出 token

底部注释：Dispatch负责把token送到expert所在rank；Combine负责把expert输出送回原token并聚合成一个结果。

### 布局细节

- 画面分为三列：左侧输入与路由，中间 MoE 通信与计算，右侧输出。
- 中间区域占画面主体宽度的 55% 左右，上下留白收紧。
- Dispatch、Expert / FFN、Combine 从左到右排列，形成明确数据流。
- 连线只表达数据流，不添加装饰性箭头。
- 输入 / 输出 token 使用浅色卡片，避免与中间核心流程争夺注意力。

### 颜色语义

- 输入 token → Router：黄色 #d79921
- Router → Dispatch：蓝紫色 #8f3f71，表示路由结果驱动通信
- Dispatch → Expert / FFN：青蓝色 #458588，表示 token 发送到 expert
- Expert / FFN → Combine：橙色 #d65d0e，表示 expert 输出返回
- Combine → 输出 token：绿色 #98971a，表示聚合完成
- Router卡片：蓝紫色 #8f3f71
- Dispatch卡片：青蓝色 #458588
- Expert / FFN卡片：黄色 #d79921
- Combine卡片：橙色 #d65d0e

### 禁止项

- 不出现 token 0、rank 0、E1 等具体编号，保持业务层抽象。
- 不出现 SIMT、scalar、URMA、MTE 等实现角色。
- 不出现底层通信实现术语。
- 图注只解释业务，不描述画面构成。

### 图注

图1：MoE通信的业务闭环。Router选择expert后，Dispatch把token发送到对应rank，Expert/FFN完成计算，Combine把输出送回原token并按权重聚合。

## 图2：Router动态路由带来的通信变化

### 目标

解释为什么MoE通信计划不能在运行前固定。重点展示两个事实：

1. 同一个token在不同step可能去往不同rank；
2. 同一个step中各rank接收token数量不相等。

### 放置位置

放在第1章“通信的复杂性来自Router的动态选择”这一段之后。

### 画面结构

采用上下两个step面板：

1. 上半部分：step t；
2. 下半部分：step t+1。

每个step面板内部采用三段式：

1. 左侧token列；
2. 中间Router top-k；
3. 右侧4个rank泳道。

### 具体文案

标题：Router动态路由带来的通信变化

上方面板标题：step t

下方面板标题：step t+1

token标签：

- token 0
- token 1
- token 2
- token 3

Router标签：Router top-k

rank泳道标签：

- rank 0
- rank 1
- rank 2
- rank 3

各rank接收数量标签：

- 接收 0
- 接收 1
- 接收 2
- 接收 3

底部注释：同一个token在不同step可能去往不同rank；每个rank接收的数量也随路由变化，无法在运行前固定。

### 数据设计

step t：

| token | 目标rank |
| --- | --- |
| token 0 | rank 1、rank 2 |
| token 1 | rank 0、rank 3 |
| token 2 | rank 1 |
| token 3 | rank 2、rank 3 |

step t 各rank接收：

| rank | 接收token |
| --- | --- |
| rank 0 | token 1 |
| rank 1 | token 0、token 2 |
| rank 2 | token 0、token 3 |
| rank 3 | token 1、token 3 |

step t 接收数量：

| rank | 接收数量 |
| --- | ---: |
| rank 0 | 1 |
| rank 1 | 2 |
| rank 2 | 2 |
| rank 3 | 2 |

step t+1：

| token | 目标rank |
| --- | --- |
| token 0 | rank 0、rank 3 |
| token 1 | rank 2 |
| token 2 | rank 1、rank 3 |
| token 3 | rank 0 |

step t+1 各rank接收：

| rank | 接收token |
| --- | --- |
| rank 0 | token 0、token 3 |
| rank 1 | token 2 |
| rank 2 | token 1 |
| rank 3 | token 0、token 2 |

step t+1 接收数量：

| rank | 接收数量 |
| --- | ---: |
| rank 0 | 2 |
| rank 1 | 1 |
| rank 2 | 1 |
| rank 3 | 2 |

### 布局细节

- step t与step t+1面板之间用细分隔线，保持视觉上是同一个业务场景的两个时刻。
- token卡片使用黄色 #d79921。
- Router卡片使用蓝紫色 #8f3f71。
- rank泳道使用蓝紫色 #8f3f71 的浅色填充。
- 接收数量标签使用橙色 #d65d0e。
- step标题使用次级文字色 #665c54。

### 禁止项

- 不出现expert编号，避免读者误以为这是第2章的完整case。
- 不出现Dispatch、Combine、SIMT、scalar、URMA、MTE。
- 不出现底层通信实现术语。
- 图注只解释业务，不描述画面构成。

### 图注

图2：Router动态路由带来的通信变化。同一个token在不同step可能发送到不同rank，且每个rank接收的token数量不同，因此通信计划必须在运行时生成。

## 图3：4-rank Dispatch数据流

### 目标

用一张图把第2章的 8-token case 讲清楚：Router 结果如何变成去重后的向量发送，再如何在接收端展开为 expert 输入。重点呈现三件事：

1. 每个 token 的目标 rank；
2. 同一 token 命中同一 rank 的多个 expert 时，向量只发送一份；
3. 接收端仍按 expert 展开，最终得到 16 个 expert 输入。

### 放置位置

放在第2章 2.1 小节的路由表之后、发送关系列表之前。读者先看到表格中的精确映射，再通过这张图理解“去重前 — 去重后 — expert 输入”的转换。

### 画面结构

整体分为三个阶段，最终图片采用上下排列以容纳全部案例，依次为：

1. Router结果：8 个 token 卡片，每个卡片列出选中的 expert 和目标 rank；
2. 去重后向量：14 个向量卡片，按目标 rank 分成 4 列；
3. 接收端expert输入：16 个 expert 输入卡片，同样按目标 rank 分成 4 列。

三阶段之间用细箭头连接，表达数据从路由结果到向量发送、再到expert输入的转换。

### 具体文案

主标题：4-rank Dispatch：从Router结果到expert输入

左段标题：Router结果

token卡片文案：

- token 0：E7→rank 1，E10→rank 2
- token 1：E3→rank 0，E12→rank 3
- token 2：E5→rank 1，E8→rank 1
- token 3：E15→rank 2，E2→rank 3
- token 4：E1→rank 0，E6→rank 1
- token 5：E9→rank 2，E11→rank 2
- token 6：E4→rank 0，E14→rank 3
- token 7：E13→rank 1，E16→rank 3

中段标题：去重后向量

分组标题：

- rank 0本地
- rank 1
- rank 2
- rank 3

向量卡片文案：

- rank 0本地：token 1、token 4、token 6
- rank 1：token 0、token 2（E5+E8，一份向量）、token 4、token 7
- rank 2：token 0、token 3、token 5（E9+E11，一份向量）
- rank 3：token 1、token 3、token 6、token 7

右段标题：接收端expert输入

expert输入卡片文案：

- rank 0：token 1 / E3，token 4 / E1，token 6 / E4
- rank 1：token 0 / E7，token 2 / E5，token 2 / E8，token 4 / E6，token 7 / E13
- rank 2：token 0 / E10，token 3 / E15，token 5 / E9，token 5 / E11
- rank 3：token 1 / E12，token 3 / E2，token 6 / E14，token 7 / E16

底部摘要：

- 16个expert选择
- 14份向量：3份本地，11份远端
- 16个expert输入

### 布局细节

- 最终采用上、中、下三阶段；每阶段保留完整案例，减少窄列中的小字。
- 上阶段8个token横排，卡片内expert与目标rank用小标签表达。
- 中阶段按4个rank分列，列内展示去重后的向量。
- token 2 和 token 5 的向量卡片使用橙色徽标“一份向量”，并在卡片内列出两个 expert。
- 下阶段按4个rank分列，每个expert输入单独展示，与16个expert选择对应。
- 三段之间使用细箭头，不绘制 token 到 rank 的交叉连线，避免视觉混乱。
- 底部摘要使用三个小卡片横排，分别对应“16个expert选择”“14份向量”“16个expert输入”。

### 颜色语义

- token卡片：黄色 #d79921
- rank分组：蓝紫色 #8f3f71
- 本地向量：绿色 #98971a
- 远端向量：青蓝色 #458588
- 去重徽标：橙色 #d65d0e
- expert输入卡片：红色 #cc241d
- 主文字：深灰 #3c3836
- 次文字：浅灰 #665c54

### 禁止项

- 不出现 SIMT、scalar、URMA、MTE、slot、通信描述等实现概念。
- 不出现 WQE、SGE、SQ、CQ、queue、doorbell、Jetty、Channel、staged transport、payload、hidden、metadata。
- 不用交叉线表达 token 到 rank 的全部路径，避免画面混乱。
- 图注不描述画面构图。

### 图注

图3：4-rank Dispatch数据流。rank 0的8个token按路由结果发送到对应rank；同一token命中同一rank的多个expert时只发送一份向量，接收端再展开为多个expert输入。

## 图4：4-rank Combine数据流

### 目标

把 Combine 的 8-token case 讲成一条反向数据流：expert 输出先按来源 rank 分组，再回到原始 token，最后按 top-k 权重聚合成一个输出。重点呈现三件事：

1. 16 个 expert 输出分别来自哪个 rank；
2. 本地输出直接参与聚合，远端输出需要送回 rank 0；
3. 每个原始 token 最终只产生一个输出。

### 放置位置

放在第2章 2.2 小节的返回关系表之后、Combine 业务流程列表之前。读者先在表格中看到精确来源，再通过这张图理解“expert 输出 — 返回原始 token — 加权聚合”的完整链路。

### 画面结构

整体分为三段，从左到右依次为：

1. expert输出：16 个输出卡片，按来源 rank 分成 4 列；
2. 返回原始token：8 个 token 卡片，每个卡片列出两个 expert 输出及本地 / 远端路径；
3. token输出：8 个输出卡片，与原始 token 一一对应。

三段之间使用阶段箭头，表达从 expert 输出到返回、再到聚合输出的方向。

### 具体文案

主标题：4-rank Combine：从expert输出到token输出

左段标题：expert输出

来源分组标题：

- rank 0本地
- rank 1
- rank 2
- rank 3

expert输出卡片文案：

- rank 0本地：token 1 / E3，token 4 / E1，token 6 / E4
- rank 1：token 0 / E7，token 2 / E5，token 2 / E8，token 4 / E6，token 7 / E13
- rank 2：token 0 / E10，token 3 / E15，token 5 / E9，token 5 / E11
- rank 3：token 1 / E12，token 3 / E2，token 6 / E14，token 7 / E16

中段标题：返回原始token

token卡片文案：

- token 0：E7来自rank 1，E10来自rank 2
- token 1：E3来自rank 0，E12来自rank 3
- token 2：E5来自rank 1，E8来自rank 1
- token 3：E15来自rank 2，E2来自rank 3
- token 4：E1来自rank 0，E6来自rank 1
- token 5：E9来自rank 2，E11来自rank 2
- token 6：E4来自rank 0，E14来自rank 3
- token 7：E13来自rank 1，E16来自rank 3

路径徽标：

- 本地输出
- 远端输出
- 同一来源rank

右段标题：token输出

输出卡片文案：

- token 0输出
- token 1输出
- token 2输出
- token 3输出
- token 4输出
- token 5输出
- token 6输出
- token 7输出

聚合徽标：按top-k权重聚合

底部摘要：

- 16个expert输出
- 3个本地输出
- 13个远端输出
- 8个token输出

### 布局细节

- 左段占画面宽度约 42%，中段约 34%，右段约 24%。
- 左段最终按4个来源rank纵向分组，保持16个输出全部可见。
- rank 0本地列使用绿色强调，表示本地输出直接参与聚合。
- rank 1、rank 2、rank 3列使用蓝紫色分组，表示输出需要送回 rank 0。
- 中段按 8 个原始 token 纵向排列，每个 token 卡片内列出两个 expert 输出。
- token 2 和 token 5 使用橙色“同一来源rank”徽标，说明两个输出来自同一个 rank，但聚合时仍各自参与。
- 右段 8 个输出卡片与中段 token 一一对应，用橙色“按top-k权重聚合”徽标统一说明聚合规则。
- 三段之间只画阶段箭头，不绘制 16 条输出到 8 个 token 的交叉连线。
- 底部摘要使用四个小卡片横排，展示数量转换。

### 颜色语义

- expert输出卡片：红色 #cc241d
- rank分组：蓝紫色 #8f3f71
- 本地输出：绿色 #98971a
- 远端输出：青蓝色 #458588
- 同一来源rank徽标：橙色 #d65d0e
- token输出卡片：黄色 #d79921
- 主文字：深灰 #3c3836
- 次文字：浅灰 #665c54

### 禁止项

- 不出现 SIMT、scalar、URMA、MTE、slot、通信描述、task、reduce 等实现概念。
- 不出现 WQE、SGE、SQ、CQ、queue、doorbell、Jetty、Channel、staged transport、payload、hidden、metadata。
- 不绘制 16 条交叉连线，避免掩盖分组和聚合关系。
- 图注不描述画面构图。

### 图注

图4：4-rank Combine数据流。expert输出按来源rank返回rank 0；本地输出直接参与聚合，远端输出送回后在原始token处按top-k权重聚合。

## 图5：为什么SIMT适合MoE通信

### 目标

用一张图回答第3章的核心问题：MoE通信前的整理工作为什么是“不规则但可并行”的，以及SIMT为什么契合这种工作形态。重点呈现三件事：

1. 每个token的路由不同，需要不同分支；
2. 多个线程可以执行同一套整理程序，分别处理不同token；
3. 与SIMD相比，SIMT更自然地表达这种动态分支。

### 放置位置

放在第3章第一段之后、三类优势列表之前。读者先通过这张图看到“token不同、线程并行、结果不同”的直观关系，再阅读结构契合、表达自然、并发友好的文字总结。

### 画面结构

整体分为三个区域：

1. 上方主体：4行token的并行整理过程；
2. 中间阶段卡：SIMT程序与4个线程；
3. 底部对比条：SIMD与SIMT的执行形态差异。

上方主体采用四行三列：

1. 左列：路由结果；
2. 中列：SIMT线程与分支；
3. 右列：整理后的通信信息。

### 具体文案

主标题：为什么SIMT适合MoE通信

左列标题：路由结果

左列卡片文案：

- token 0：rank 1、rank 2
- token 1：rank 0、rank 3
- token 2：rank 1、rank 1
- token 3：rank 2、rank 3

中列标题：SIMT：同一程序，多个线程

中列卡片文案：

- 线程0 / token 0：两个目标rank不同，两条分支
- 线程1 / token 1：一个本地rank，一个远端rank
- 线程2 / token 2：两个目标rank相同，去重后只保留一份
- 线程3 / token 3：两个目标rank不同，两条分支

程序步骤标签：

- 读取路由
- 比较rank
- 判断是否重复
- 生成通信信息

右列标题：整理后的通信信息

右列卡片文案：

- token 0：发送到rank 1和rank 2
- token 1：本地保留，另发送到rank 3
- token 2：只发送一份到rank 1
- token 3：发送到rank 2和rank 3

底部对比条标题：执行形态对比

SIMD卡片文案：

- SIMD
- 一条指令处理连续数据
- 适合规则、少分支的计算

SIMT卡片文案：

- SIMT
- 多个线程执行同一程序
- 每个线程可走不同分支
- 适合token路由各不相同的整理工作

### 布局细节

- 上方主体占画面高度约70%，底部对比条占约30%。
- 四行token从上到下排列，左列、中列、右列保持水平对齐，让读者能沿一行看到“路由结果、线程分支、通信信息”的对应关系。
- 中列使用一个浅青蓝色大卡片承载4个线程行，卡片顶部放置4个程序步骤标签，表示所有线程执行同一套程序。
- token 2所在行使用橙色“去重”徽标，突出同一rank重复时只生成一份通信信息。
- 左列token卡片使用黄色，中列SIMT卡片使用青蓝色，右列通信信息卡片使用绿色。
- 底部对比条使用两个并排卡片，SIMD使用浅灰色，SIMT使用浅青蓝色。
- 图中不绘制复杂交叉线，行与行之间保持独立，突出“多个线程并行处理不同token”。

### 颜色语义

- token卡片：黄色 #d79921
- rank标签：蓝紫色 #8f3f71
- SIMT区域：青蓝色 #458588
- 去重徽标：橙色 #d65d0e
- 通信信息卡片：绿色 #98971a
- SIMD卡片：浅灰 #ebdbb2
- 主文字：深灰 #3c3836
- 次文字：浅灰 #665c54

### 禁止项

- 不出现AIV、lane、scalar、URMA、MTE、slot、通信描述等实现细节。
- 不出现WQE、SGE、SQ、CQ、queue、doorbell、Jetty、Channel、staged transport、payload、hidden、metadata。
- 不把SIMT画成单独硬件模块，只表达“多个线程执行同一程序”的执行形态。
- 不用长句填充卡片，所有文案保持短标签化。
- 图注不描述画面构图。

### 图注

图5：为什么SIMT适合MoE通信。不同token的路由结果不同，多个线程可以执行同一套整理程序并分别走不同分支；相比处理连续数据的SIMD，SIMT更贴近这种动态、分支密集的通信整理工作。

## 图6：Dispatch的SIMT并行流程

### 目标

展示第4章 Dispatch 的整体执行方式：8个token由多个线程并行处理，同时完成读取Router结果、目标rank统计、重复rank去重、接收slot计算和通信描述构造。重点呈现四件事：

1. token路由可独立处理，线程组织为示意；统计和位置计算需要协作；
2. token 2和token 5命中重复rank，去重后只生成一份通信信息；
3. slot由同一rank内的前序token数量动态决定；
4. SIMT负责整理通信信息，scalar负责提交，URMA/MTE负责数据搬运。

### 放置位置

放在第4章第一段之后、4.1小节之前。读者先看到8个token并行推进的全貌，再按小节理解每个阶段的并行特点。

### 画面结构

整体分为左侧主矩阵和右侧执行分工两个区域。

左侧主矩阵为“8行token × 5列阶段”：

1. Router结果；
2. 目标rank统计；
3. 重复rank去重；
4. 接收slot计算；
5. 通信描述构造。

右侧执行分工为三个纵向卡片：

1. scalar：提交通信描述；
2. URMA：远端数据搬运；
3. MTE：本地数据搬运。

主矩阵上方放置一个横向标签“SIMT：多个线程同时整理”，覆盖前5列，明确这些阶段由SIMT完成。

### 具体文案

主标题：Dispatch：SIMT并行整理通信信息

左侧主矩阵标题：SIMT：多个线程同时整理

列标题：

- Router结果
- 目标rank统计
- 重复rank去重
- 接收slot计算
- 通信描述构造

行标签：

- token 0
- token 1
- token 2
- token 3
- token 4
- token 5
- token 6
- token 7

各行文案：

- token 0：E7→rank 1，E10→rank 2；目标rank为1、2；无重复；rank 1 slot 0，rank 2 slot 0；生成2条通信信息
- token 1：E3→rank 0，E12→rank 3；目标rank为0、3；无重复；rank 0 slot 0，rank 3 slot 0；生成2条通信信息
- token 2：E5→rank 1，E8→rank 1；目标rank为1；重复rank去重；rank 1 slot 1；生成1条通信信息，接收端展开为2个expert输入
- token 3：E15→rank 2，E2→rank 3；目标rank为2、3；无重复；rank 2 slot 1，rank 3 slot 1；生成2条通信信息
- token 4：E1→rank 0，E6→rank 1；目标rank为0、1；无重复；rank 0 slot 1，rank 1 slot 2；生成2条通信信息
- token 5：E9→rank 2，E11→rank 2；目标rank为2；重复rank去重；rank 2 slot 2；生成1条通信信息，接收端展开为2个expert输入
- token 6：E4→rank 0，E14→rank 3；目标rank为0、3；无重复；rank 0 slot 2，rank 3 slot 2；生成2条通信信息
- token 7：E13→rank 1，E16→rank 3；目标rank为1、3；无重复；rank 1 slot 3，rank 3 slot 3；生成2条通信信息

右侧卡片文案：

- scalar：提交通信描述；维护明确提交边界
- URMA：远端数据搬运；处理11份远端向量
- MTE：本地数据搬运；处理3份本地向量

底部摘要：

- 16个expert选择
- 14条通信信息
- 3份本地向量
- 11份远端向量

底部注释：slot示例按同一rank内的token顺序编号，仅表示这批token的局部位置；实际布局还需结合其他来源rank的计数。14条通信信息是逻辑发送关系，不等于硬件描述数量。

### 布局细节

- 左侧主矩阵占画面宽度约76%，右侧执行分工占约24%。
- 5个阶段列从左到右排列，列间使用细箭头表达推进关系。
- 8个token行从上到下排列，每行左侧保留token标签。
- “Router结果”“目标rank统计”“重复rank去重”“接收slot计算”“通信描述构造”5列使用浅青蓝色背景或顶部色条，表示由SIMT完成。
- token 2和token 5的“重复rank去重”单元格使用橙色“去重”徽标，并在“通信描述构造”单元格标注“1条通信信息”。
- 其余token的去重单元格使用浅灰色“无重复”徽标。
- “接收slot计算”单元格展示具体slot值，帮助读者理解slot来自同一rank内的动态顺序。
- 右侧scalar卡片使用紫色，URMA卡片使用橙色，MTE卡片使用绿色，与左侧SIMT区域形成明确分工。
- 底部摘要使用四个小卡片横排，保持数量关系一眼可见。
- 不绘制token到rank的交叉连线，矩阵行本身表达每个token的独立处理过程。

### 颜色语义

- SIMT阶段：青蓝色 #458588
- token标签：黄色 #d79921
- rank标签：蓝紫色 #8f3f71
- 去重徽标：橙色 #d65d0e
- 通信信息：绿色 #98971a
- scalar：紫色 #b16286
- URMA：橙色 #d65d0e
- MTE：绿色 #98971a
- 主文字：深灰 #3c3836
- 次文字：浅灰 #665c54

### 禁止项

- 不出现AIV、lane、WQE、SGE、SQ、CQ、queue、doorbell、Jetty、Channel、staged transport、payload、hidden、metadata。
- 不把scalar、URMA、MTE画成SIMT的一部分；它们只作为后续执行分工出现。
- 不绘制复杂交叉连线，避免破坏矩阵的可读性。
- 不使用长句填充单元格；每格只保留阶段结果和必要徽标。
- 图注不描述画面构图。

### 图注

图6：Dispatch的SIMT并行流程。8个token并行整理路由，去重后形成14份向量关系。slot只示意局部顺序，硬件描述数量取决于实际实现；scalar维护提交边界，URMA和MTE分别负责远端与本地搬运。

## 图7：Combine的SIMT流水线

### 目标

展示通信准备与数据归约如何协作，明确SIMT、SIMD、scalar和URMA的分工。沿用第2章的返回数量，但将业务来源分组与实际内核task区分开。

### 放置位置

放在第5章第一段之后、5.1小节之前。先展示返回关系和双轨流水线，再解释各阶段。

### 画面结构

保留左侧来源分组、右侧双轨流水线和底部数量摘要的布局。

左侧展示rank 0这8个token的结果来源，标题为“结果来源（业务视角）”：

- rank 1 / 5个expert输出
- rank 2 / 4个expert输出
- rank 3 / 4个expert输出
- rank 0本地 / 3个expert输出

这些是业务分组，不直接标为4个内核task。各发送rank按原始token所在的目标rank和批次组织task；实际task数量还取决于token数量和执行单元划分。

右侧展示“各发送rank内的流水线”：

- 上轨：通信准备 / SIMT。读取返回信息 → 计算token返回位置 → 构造通信描述。
- 下轨：数据处理 / SIMD + MTE。读取expert输出 → 同一token局部归约 → 保存待返回结果。

数据和描述准备就绪后，再进入“scalar提交 → URMA远端写回”。提交通信描述不放在SIMT轨道内。本地结果直接留在本地，不进入URMA路径。

### 流水线文案

- 同一时间窗
- 当前task：数据归约
- 下一task：信息已准备

这里的task是同一发送rank内的相邻任务，不把它们固定对应为来自rank 1或rank 2的分组。状态条说明预先整理下一任务的信息，并不承诺SIMT与SIMD在同一执行资源上同时执行。

### 最终聚合文案

rank 0等待所需结果到齐，再按原始token聚合并写回输出。

本地路径说明：本地结果不经远端写回，直接参与最终聚合。

数量摘要：

- 16个expert输出
- 3个本地输出
- 13个远端输出
- 8个token输出

13个远端expert输出不是13次传输。同一rank内属于同一token的多个输出可以先做局部归约。

### 风格与颜色

沿用全局Gruvbox Light风格，不改变配色、字体、圆角或细线要求。SIMT为青蓝色，来源rank为蓝紫色，本地结果为绿色，数据归约为橙色，scalar提交为紫色。阶段边界清楚，不使用交叉连线。

### 禁止项

- 不把SIMD局部归约标为SIMT。
- 不把scalar提交标为SIMT。
- 不把rank 0已收到的结果重新画成由rank 0远端发送。
- 不把4个业务来源分组当作实际固定task数量。
- 不出现AIV、lane以及全局禁用的底层通信术语。
- 本地结果不得连接远端写回路径。
- 图注只解释业务和执行分工，不描述构图。

### 图注

图7：Combine将SIMT返回信息整理与SIMD局部归约组成流水线，数据和描述准备就绪后由scalar提交远端传输；本地结果直接保留，最终在原始token所在rank聚合。

### 准确性依据

核对官方[Combine实现](https://github.com/deepseek-ai/DeepEP-Ascend/blob/main/deep_ep/include/deep_ep/impls/ep/combine.hpp)及[归约实现](https://github.com/deepseek-ai/DeepEP-Ascend/blob/main/deep_ep/include/deep_ep/impls/ep/combine_utils.hpp)：SIMT构造描述，SIMD处理向量归约，scalar在依赖满足后提交。CANN参考报告仅用作视觉参考，不将其实现细节直接套入本文。

## 图片放置原则

| 图 | 主题 | 建议位置 | 原因 |
| --- | --- | --- | --- |
| 图1 | MoE通信的业务闭环 | 第1章第一段后 | 先建立业务闭环，再进入动态路由 |
| 图2 | Router动态路由带来的通信变化 | 第1章动态选择段后 | 直观说明通信计划必须运行时生成 |
| 图3 | 4-rank Dispatch数据流 | 第2章路由表后 | 将8-token case从路由结果讲到expert输入 |
| 图4 | 4-rank Combine数据流 | 第2章返回关系表后 | 将8-token case从expert输出讲到token输出 |
| 图5 | 为什么SIMT适合MoE通信 | 第3章开头 | 先给出执行形态差异，再解释适配原因 |
| 图6 | Dispatch的SIMT并行流程 | 第4章开头 | 先看并行处理方式，再拆阶段讲 |
| 图7 | Combine的SIMT流水线 | 第5章开头 | 先看流水线，再讲task和重叠 |

## 生成提示词

### 图1提示词

Use case: infographic-diagram
Asset type: first figure of a Chinese technical article for readers new to communication.
Create a production-ready flat 16:9 landscape infographic, title exactly "MoE通信的业务闭环". Gruvbox Light palette: solid cream #fbf1c7 background, white rounded cards, thin #d5c4a1 borders, dark grey #3c3836 text. Noto Sans SC / Inter REGULAR/light typography, including title; no bold, texture, gradient, shadows, 3D, icons or watermark.
Composition: compact top title with 5% top margin, main flow fills middle 60% height, short business explanation at bottom with 5% bottom margin. Avoid large empty bands. Six equal-height cards in a single horizontal connected sequence:
"输入 token" → "Router" with "top-k expert 选择" → "Dispatch" with "发送 token" and "到 expert 所在 rank" → "Expert / FFN" with "分散在不同 rank" and "并行计算" → "Combine" with "返回 expert 输出" and "按权重聚合" → "输出 token".
Group exactly Dispatch, Expert / FFN, Combine in an enclosing rounded area headed "MoE Expert Parallel".
Router accent purple #8f3f71, Dispatch teal #458588, Expert mustard #d79921, Combine orange #d65d0e, output olive #98971a. Thin directional arrows only between immediate neighbours, with sensible colored accents.
Footer exactly "Dispatch把token送到expert所在rank；Combine把expert输出送回并按权重聚合。"
All text must be verbatim readable Chinese and English. No specific token/rank/expert numbers and no implementation details.

### 图2提示词

Create a flat, clean 16:9 technical infographic titled “Router动态路由带来的通信变化”. Use the same Gruvbox Light style, white rounded cards, thin borders, and no bold text.

Create two stacked step panels, “step t” above and “step t+1” below. Each panel contains four token cards on the left, one Router card in the middle, and four rank lanes on the right. Render these exact mappings:

step t:

- token 0 → rank 1、rank 2
- token 1 → rank 0、rank 3
- token 2 → rank 1
- token 3 → rank 2、rank 3

step t rank counts:

- rank 0 接收 1
- rank 1 接收 2
- rank 2 接收 2
- rank 3 接收 2

step t+1:

- token 0 → rank 0、rank 3
- token 1 → rank 2
- token 2 → rank 1、rank 3
- token 3 → rank 0

step t+1 rank counts:

- rank 0 接收 2
- rank 1 接收 1
- rank 2 接收 1
- rank 3 接收 2

Use token #d79921, Router #8f3f71, rank lanes #8f3f71 with light fill, and receiving-count badges #d65d0e. Do not show expert IDs, Dispatch, Combine, SIMT, scalar, URMA, or MTE.

### 图3提示词

Create a high-density but readable flat 16:9 infographic titled “4-rank Dispatch：从Router结果到expert输入”. Use the same Gruvbox Light style, white rounded cards, thin borders, and no bold text.

Use three horizontal sections separated by thin arrows: “Router结果”, “去重后向量”, and “接收端expert输入”.

Render the Router results exactly:

- token 0：E7→rank 1，E10→rank 2
- token 1：E3→rank 0，E12→rank 3
- token 2：E5→rank 1，E8→rank 1
- token 3：E15→rank 2，E2→rank 3
- token 4：E1→rank 0，E6→rank 1
- token 5：E9→rank 2，E11→rank 2
- token 6：E4→rank 0，E14→rank 3
- token 7：E13→rank 1，E16→rank 3

Render the deduplicated vectors by rank:

- rank 0本地：token 1、token 4、token 6
- rank 1：token 0、token 2（E5+E8，一份向量）、token 4、token 7
- rank 2：token 0、token 3、token 5（E9+E11，一份向量）
- rank 3：token 1、token 3、token 6、token 7

Render the receiving expert inputs:

- rank 0：token 1 / E3，token 4 / E1，token 6 / E4
- rank 1：token 0 / E7，token 2 / E5，token 2 / E8，token 4 / E6，token 7 / E13
- rank 2：token 0 / E10，token 3 / E15，token 5 / E9，token 5 / E11
- rank 3：token 1 / E12，token 3 / E2，token 6 / E14，token 7 / E16

Add three summary cards: “16个expert选择”, “14份向量：3份本地，11份远端”, “16个expert输入”. Use token #d79921, rank #8f3f71, local #98971a, remote #458588, deduplication badges #d65d0e, expert inputs #cc241d. Do not draw all token-to-rank crossing lines; organize by rank lanes.

### 图4提示词

Create a high-density but readable flat 16:9 infographic titled “4-rank Combine：从expert输出到token输出”. Use the same Gruvbox Light style, white rounded cards, thin borders, and no bold text.

Use three sections from left to right: “expert输出”, “返回原始token”, and “token输出”.

Render the expert outputs grouped by source:

- rank 0本地：token 1 / E3，token 4 / E1，token 6 / E4
- rank 1：token 0 / E7，token 2 / E5，token 2 / E8，token 4 / E6，token 7 / E13
- rank 2：token 0 / E10，token 3 / E15，token 5 / E9，token 5 / E11
- rank 3：token 1 / E12，token 3 / E2，token 6 / E14，token 7 / E16

Render the return relationships:

- token 0：E7来自rank 1，E10来自rank 2
- token 1：E3来自rank 0，E12来自rank 3
- token 2：E5来自rank 1，E8来自rank 1
- token 3：E15来自rank 2，E2来自rank 3
- token 4：E1来自rank 0，E6来自rank 1
- token 5：E9来自rank 2，E11来自rank 2
- token 6：E4来自rank 0，E14来自rank 3
- token 7：E13来自rank 1，E16来自rank 3

Render eight cards “token 0输出” through “token 7输出”, each labeled “按top-k权重聚合”. Add summary cards: “16个expert输出”, “3个本地输出”, “13个远端输出”, “8个token输出”. Use expert outputs #cc241d, rank #8f3f71, local #98971a, remote #458588, aggregation badges #d65d0e, token outputs #d79921. Do not draw 16 crossing lines.

### 图5提示词

Create a flat, clean 16:9 infographic titled “为什么SIMT适合MoE通信”. Use the same Gruvbox Light style, white rounded cards, thin borders, and no bold text.

Create four aligned rows showing these exact transformations:

- token 0：rank 1、rank 2 → 线程0 / token 0：两个目标rank不同，两条分支 → 发送到rank 1和rank 2
- token 1：rank 0、rank 3 → 线程1 / token 1：一个本地rank，一个远端rank → 本地保留，另发送到rank 3
- token 2：rank 1、rank 1 → 线程2 / token 2：两个目标rank相同，去重后只保留一份 → 只发送一份到rank 1
- token 3：rank 2、rank 3 → 线程3 / token 3：两个目标rank不同，两条分支 → 发送到rank 2和rank 3

Place the shared program steps above the four threads: “读取路由”, “比较rank”, “判断是否重复”, “生成通信信息”. Add an orange “去重” badge on token 2.

At the bottom, show two comparison cards:

- SIMD：一条指令处理连续数据 / 适合规则、少分支的计算
- SIMT：多个线程执行同一程序 / 每个线程可走不同分支 / 适合token路由各不相同的整理工作

Use token #d79921, rank #8f3f71, SIMT #458588, dedup badge #d65d0e, communication information #98971a. Do not show AIV, lane, scalar, URMA, MTE, slot, or communication descriptors.

### 图6提示词

Create a high-density flat 16:9 infographic titled “Dispatch：SIMT并行整理通信信息”. Use the same Gruvbox Light style, white rounded cards, thin borders, and no bold text.

Create an 8-row by 5-stage matrix. The header above the matrix must say “SIMT：多个线程同时整理”. Column headers are “Router结果”, “目标rank统计”, “重复rank去重”, “接收slot计算”, “通信描述构造”.

Render the exact rows:

- token 0：E7→rank 1，E10→rank 2；目标rank 1、2；无重复；rank 1 slot 0，rank 2 slot 0；生成2条通信信息
- token 1：E3→rank 0，E12→rank 3；目标rank 0、3；无重复；rank 0 slot 0，rank 3 slot 0；生成2条通信信息
- token 2：E5→rank 1，E8→rank 1；目标rank 1；去重；rank 1 slot 1；生成1条通信信息，接收端展开为2个expert输入
- token 3：E15→rank 2，E2→rank 3；目标rank 2、3；无重复；rank 2 slot 1，rank 3 slot 1；生成2条通信信息
- token 4：E1→rank 0，E6→rank 1；目标rank 0、1；无重复；rank 0 slot 1，rank 1 slot 2；生成2条通信信息
- token 5：E9→rank 2，E11→rank 2；目标rank 2；去重；rank 2 slot 2；生成1条通信信息，接收端展开为2个expert输入
- token 6：E4→rank 0，E14→rank 3；目标rank 0、3；无重复；rank 0 slot 2，rank 3 slot 2；生成2条通信信息
- token 7：E13→rank 1，E16→rank 3；目标rank 1、3；无重复；rank 1 slot 3，rank 3 slot 3；生成2条通信信息

On the right, add three role cards:

- scalar：提交通信描述 / 维护明确提交边界
- URMA：远端数据搬运 / 处理11份远端向量
- MTE：本地数据搬运 / 处理3份本地向量

Add four summary cards: “16个expert选择”, “14条通信信息”, “3份本地向量”, “11份远端向量”. Use SIMT #458588, scalar #b16286, URMA #d65d0e, MTE #98971a. Do not draw crossing token-to-rank lines and do not show AIV or lane.
Global approved style: 16:9, Gruvbox Light. Solid cream #fbf1c7, white cards #ffffff, dark text #3c3836, secondary #665c54, hairline borders #d5c4a1, Noto Sans SC / Inter Light or Regular ONLY including title. Flat rounded cards, no bold, gradient, shadow, texture or 3D. SIMT #458588, scalar #b16286, URMA #d65d0e, MTE #98971a, token #d79921, expert #cc241d, rank #8f3f71.
Use exactly 8 token rows, without merging, omitting or duplicating any. This is a pedagogical matrix, not a hardware memory layout. Add the small footer note "slot为局部示例；14条为逻辑通信信息，非硬件描述数量". Column 2 shows per-token target ranks. Add one slim aggregate row under the matrix: "去重后数量：rank 0 = 3，rank 1 = 4，rank 2 = 3，rank 3 = 4". Keep scalar OUTSIDE SIMT area, and fork its submission arrow to URMA and MTE, not URMA→MTE. No per-cell miniature arrows. Exact text and numbers matter more than decoration.

### 图7提示词

Use case: infographic-diagram
Asset type: figure 7 of the same Chinese technical article, a precise Combine pipeline.
Global approved style: 16:9, Gruvbox Light. Solid cream #fbf1c7, white cards #ffffff, dark text #3c3836, secondary #665c54, hairline borders #d5c4a1, Noto Sans SC / Inter Light or Regular ONLY including title. Flat rounded cards, no bold, gradient, shadow, texture or 3D. SIMT #458588, scalar #b16286, URMA #d65d0e, MTE #98971a, token #d79921, expert #cc241d, rank #8f3f71.
Title exactly "Combine：SIMT流水线整理返回结果", medium regular font, compact margins. Keep 16:9, readable text, no complex crossing wires.

LEFT 22%: an area headed "结果来源（业务视角）", four stacked cards:
"rank 1" / "5个expert输出"
"rank 2" / "4个expert输出"
"rank 3" / "4个expert输出"
"rank 0本地" / "3个expert输出".
Remote cards use rank-purple outlines; local card olive. Under these cards a short note "业务分组 ≠ 内核task数量". These counts refer to the rank 0 case, NOT four actual scheduled tasks.

RIGHT 78%, two tracks enclosed in an area headed "各发送rank内的流水线":
UPPER teal track label "通信准备 / SIMT", exactly three nodes connected left to right:
"读取返回信息" → "计算token返回位置" → "构造通信描述".
LOWER orange track label "数据处理 / SIMD + MTE", exactly three nodes connected left to right:
"读取expert输出" → "同一token局部归约" → "保存待返回结果".
The lower track is NOT labelled SIMT. This local reduction can happen on a remote sender as well; it means computation local to that sender, not rank 0 final aggregation.

After both tracks have produced prepared data and descriptions, a thin junction leads to a SEPARATE purple card "scalar提交", then separate orange card "URMA远端写回". Do not submit before data is ready. These cards must be outside SIMT area and not absorbed into the SIMT track. The rank 0 local result bypasses URMA.

Below the tracks, a compact aligned two-row pipeline window headed "流水线示意":
"当前task：数据归约"
"下一task：信息已准备"
Make the temporal relationship clear using short parallel colored bars; never imply final token output before required remote results arrive.

One bottom full-width business strip:
"rank 0：所需结果到齐 → 按原始token聚合 → 写回8个token输出".
Beside it a green local-path note exactly "本地结果不经远端写回，直接参与最终聚合". Keep this local path separate from the remote URMA path.

Bottom summary badges exactly:
"16个expert输出" / "3个本地输出" / "13个远端输出" / "8个token输出".

Constraints: readable precise text, no duplicated nodes, no made-up labels. No AIV, lane, WQE, SGE, SQ, CQ, queue, doorbell, Jetty, Channel, payload, hidden, metadata. Do not label SIMD reduction or scalar submission as SIMT. Do not assign task i to a hard-coded source rank. 13 remote expert outputs are logical contributions, not necessarily 13 transmissions.

## 最终资产与验收记录

完成日期：2026-10-14。执行方式：当前会话直接调用内置imagegen，逐张生成、查看，再进行局部编辑。最终图片没有使用Pillow覆盖文字或代码绘制；原历史渲染脚本仅保留为归档工具。

最终检查：7张PNG均为1672×941（约16:9），图像数据完整可解码；正文图号1到7顺序正确，章节分布为1/1/2/2/3/4/5；全部本地图片与文档链接有效。路由表推导的向量数量为3/4/3/4，expert输出数量为3/5/4/4；git diff --check通过。

| 图 | 最终资产 | 重点核对 |
| --- | --- | --- |
| 图1 | [deepep-ascend-simt-moe-business-loop.png](../ascend-assets/deepep-ascend-simt-moe-business-loop.png) | MoE业务闭环、标题和输入输出 |
| 图2 | [deepep-ascend-simt-router-dynamic-flow.png](../ascend-assets/deepep-ascend-simt-router-dynamic-flow.png) | 两步路由、接收token和计数 |
| 图3 | [deepep-ascend-simt-dispatch-case.png](../ascend-assets/deepep-ascend-simt-dispatch-case.png) | 8条路由、14份向量、16个expert输入 |
| 图4 | [deepep-ascend-simt-combine-case.png](../ascend-assets/deepep-ascend-simt-combine-case.png) | 16个expert输出、3本地、13远端、8输出 |
| 图5 | [deepep-ascend-simt-parallel-routing.png](../ascend-assets/deepep-ascend-simt-parallel-routing.png) | 4条示例路由、重复rank去重、SIMD/SIMT对比 |
| 图6 | [deepep-ascend-simt-dispatch-parallel-flow.png](../ascend-assets/deepep-ascend-simt-dispatch-parallel-flow.png) | 8行5阶段、slot、token 2/5去重、提交与搬运分工 |
| 图7 | [deepep-ascend-simt-combine-pipeline.png](../ascend-assets/deepep-ascend-simt-combine-pipeline.png) | 来源分组、SIMT/SIMD/scalar分工、本地绕过远端传输 |

正文已按图1到图7顺序引用全部新图，图前过渡、图注及阶段标题同步调整。高密度Dispatch数据流最终采用上下三阶段，以完整保留案例标签。

6张未再被文档引用的旧图及其本地备份已删除；旧图可从Git历史恢复，其他文章的图片保留。历史脚本默认输出改到 `.scratch/imagegen/legacy-renderer/`，避免再次产生过时正文资产。规格文件已更名为不含版本号的文件名。

### 最终局部修订提示词

以下为生成后按原规格实施的局部修订，配合上面的逐图生成提示词使用。只修正字重、配色、可读性或数据错误，不另设视觉风格。

#### 图1局部修订

Edit the supplied infographic to meet the already approved style specification exactly. Keep all business labels, the six-card horizontal sequence, all grouping and arrows unchanged. Change ONLY visual styling: make background a completely flat solid #fbf1c7 (no gradient, texture, glow); white cards completely flat #ffffff; thin 1px #d5c4a1 borders. Every text including title must be Noto Sans SC / Inter REGULAR or LIGHT, absolutely no bold. Reduce title font size to about 46px on a 1680px wide canvas. Use consistent simple flat geometry with no shadows. Change the final Combine→输出 token arrow to olive #98971a. Keep readable Chinese verbatim. No added text or elements. Preserve 16:9. This is Gruvbox Light technical documentation style, flat, fine-line, rounded cards.

#### 图2局部修订

Targeted correction to the supplied candidate: preserve the two stacked panels and three-zone structure. Use the approved flat style exactly: background #fbf1c7, white cards, thin colored outlines, all title and labels Regular/light, no gradients or shadows. Do NOT leave mappings implicit in wires passing through Router.
In EACH left token card add its exact destination ranks below its token name:
step t: token 0 "→ rank 1、2"; token 1 "→ rank 0、3"; token 2 "→ rank 1"; token 3 "→ rank 2、3".
step t+1: token 0 "→ rank 0、3"; token 1 "→ rank 2"; token 2 "→ rank 1、3"; token 3 "→ rank 0".
In EACH right rank card add the exact received tokens:
step t: rank 0 "token 1"; rank 1 "token 0、2"; rank 2 "token 0、3"; rank 3 "token 1、3".
step t+1: rank 0 "token 0、3"; rank 1 "token 2"; rank 2 "token 1"; rank 3 "token 0、2".
Keep all orange count badges EXACTLY as specified. Center cards say "Router top-k". Arrows can be grouped so they do not obscure text. Header dark gray #3c3836 regular, not purple bold. Exact readable labels, no extra text.

#### 图3局部修订

Use case: precise-object-edit
Change ONLY the color styling of the supplied accurate Dispatch infographic to the already approved Gruvbox Light palette. Preserve ALL text, ALL 8 token mappings, ALL 14 deduplicated vectors, ALL 16 expert inputs, ALL summary counts, all arrows and all positions. Do not redraw, reflow or rearrange the layout.
Replace the pale near-white outer background with flat solid cream #fbf1c7. Main panel interiors and cards white #ffffff. Exterior borders thin #d5c4a1 instead of heavy black. Main title regular weight, dark #3c3836. Preserve mustard token accents #d79921, purple ranks #8f3f71, teal remote accents #458588, olive local #98971a, orange deduplication #d65d0e, red expert #cc241d. All fills should be flat, no gradients/shadows/texture. All letters and numbers are immutable; especially token 2 E5/E8 and token 5 E9/E11, rank counts 3/4/3/4, and total 16→14→16.

#### 图4局部修订

Use case: precise-object-edit
Change ONLY color styling of this accurate Combine infographic to the approved Gruvbox Light palette. Preserve ALL 16 expert cards, ALL 8 return rows, ALL 8 output rows, every label, number, arrow and position. Do not redraw or rearrange anything.
Replace the near-white outer background with flat solid cream #fbf1c7. Main panels and cards white #ffffff. All borders hairline #d5c4a1; title regular/light #3c3836, not bold. Use muted purple #8f3f71 for rank groups, olive #98971a for local, teal #458588 for remote, orange #d65d0e for aggregation, mustard #d79921 for token output, red #cc241d for expert. No gradient/shadow/texture.
Data invariants: rank 0 token 1/E3,4/E1,6/E4; rank 1 token 0/E7,2/E5,2/E8,4/E6,7/E13; rank 2 token 0/E10,3/E15,5/E9,5/E11; rank 3 token 1/E12,3/E2,6/E14,7/E16. Counts 16 expert outputs, 3 local, 13 remote, 8 token outputs. Preserve every return relationship.

#### 图5局部修订

Use case: precise-object-edit
Change ONLY typography and flat visual styling of this figure to match the approved Gruvbox Light article specification. Preserve ALL row texts, all 4 rows, all mappings, bottom SIMD/SIMT comparison, badges and arrows exactly.
Replace the large bold title with a much smaller REGULAR-weight title "为什么SIMT适合MoE通信" in dark gray #3c3836. All fonts including SIMD/SIMT headings must be regular/light, never bold. Make background a uniform cream #fbf1c7, cards white #ffffff, hairline borders #d5c4a1; flat pale accent fills only, no gradients, texture, shadows or 3D.
Add one short regular label above the thread column, without covering existing text: "SIMT：同一程序，多个线程".
Keep all other content and geometry invariant.

#### 图6局部修订

Use case: precise-object-edit
Edit the supplied 8-row Dispatch matrix. Preserve all eight row labels, all expert/rank mappings, ALL slot values, all counts, role cards, layout and arrows exactly.
ONE critical data correction: in row "token 2", final column "通信描述构造", change "生成2条通信信息" to exactly "生成1条通信信息". Keep its second line "接收端展开为2个expert输入". Row "token 5" already correctly has "生成1条通信信息"; preserve it.
Put the word "去重" in an orange #d65d0e small badge in the token 2 and token 5 third-stage cells, without changing any adjacent text.
Make title "Dispatch：SIMT并行整理通信信息" regular/light typography, not bold. Make all fills flat (no gradients); maintain the approved Gruvbox Light #fbf1c7 background, white cards, #d5c4a1 thin borders. Do not add, remove, reorder, duplicate or alter any token row. Exact data accuracy is critical.

#### 图7局部修订

Use case: precise-object-edit
Apply ONLY the approved Gruvbox Light visual style to this accurate Combine diagram. Preserve EVERY label, number, arrow, track, pipeline bar, source card, local bypass and placement exactly. No content edits.
Set title and ALL section headers to REGULAR or LIGHT sans-serif weight. No bold anywhere. Use flat solid #fbf1c7 background, white #ffffff cards, thin #d5c4a1 exterior borders, and the existing muted Gruvbox teal/purple/olive/orange accents. Remove gradients or texture while preserving pale flat track colors. Keep all text readable, no text omissions, no duplicate nodes. This is a typography and flat-color correction only.

### 实现与口径校验

- 同一token的重复目标rank去重，只减少向量发送，expert输入仍需展开。
- 第2章数量仅来自rank 0的8个token；其他来源rank的计数不在图中。
- slot是局部顺序示例，14条通信信息是逻辑发送关系，不保证相同硬件描述数量。
- Combine数值归约使用SIMD，SIMT整理返回信息并构造描述；scalar提交属于单独边界。
- 结果来源分组不是实际固定task数量；发送端按原始token所在的返回目标划分批次。
- 权重聚合是模型业务；权重可以提前应用，Combine归约已加权结果。
- 性能表核对[官方README](https://github.com/deepseek-ai/DeepEP-Ascend#performance)，排除最终后处理，不作为SIMT相对SIMD的独立加速比。
