# DeepEP-Ascend SIMT 文章配图 v6 文字版规格

## 目标

这版规格只解决两个问题：

1. Dispatch 和 Combine 里的 SIMT 到底是怎么并行的；
2. 图片放在哪里最有利于读者理解，避免把所有图都堆在大章节结尾。

本规格先定文案、布局、颜色和位置，再生成图片。图片承担“怎么做”，正文承担“为什么”。

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

## 图片放置原则

| 图 | 主题 | 建议位置 | 原因 |
| --- | --- | --- | --- |
| 图1 | token 到 expert 的动态映射 | 第1章第1段后 | 先建立“每个 token 目标不同”的直观印象 |
| 图2 | MoE 完整业务流程 | 第1章 Dispatch/Combine 定义后 | 先看闭环，再进入细节 |
| 图3 | Dispatch 与 Combine 操作拆解 | 第2章 Combine 流程列表后 | 用一张图承接两条业务链 |
| 图4 | SIMD 与 SIMT 对比 | 第3章开头 | 先给出执行形态差异，再解释适配原因 |
| 图5 | Dispatch 的 SIMT 并行流程 | 第4章开头 | 先看并行处理方式，再拆阶段讲 |
| 图6 | Combine 的 SIMT 流水线 | 第5章开头 | 先看流水线，再讲 task 和重叠 |

## 图5 v6：Dispatch 的 SIMT 并行流程

### 目标

展示 Dispatch 中 8 个 token 如何同时完成读取 Router 结果、统计目标 rank、去重、计算接收 slot 和构造通信描述。重点是“每个 token 独立并行”，以及 token 2、token 5 的去重逻辑。

### 画面结构

整体采用“8 行 token × 5 列阶段”的矩阵式布局。

- 顶部：标题“Dispatch：SIMT 并行整理路由与通信描述”
- 左侧：8 个 token 行，按 AIV 分组着色
  - AIV 0：token 0、token 1
  - AIV 1：token 2、token 3
  - AIV 2：token 4、token 5
  - AIV 3：token 6、token 7
- 中部：5 个阶段列
  1. Router 结果
  2. 目标统计
  3. 重复 rank 去重
  4. 接收 slot 计算
  5. 通信描述构造
- 右侧：提交与搬运
  - scalar：提交描述
  - URMA：远端写入
  - MTE：本地搬运
- 底部：一句注释“每个 token 独立处理；同一 token 的重复 rank 只发送一份，接收端再按 expert 展开”

### 具体文案

标题：Dispatch：SIMT 并行整理路由与通信描述

列标题：Router 结果、目标统计、重复 rank 去重、接收 slot 计算、通信描述构造

行标签：token 0 到 token 7

AIV 分组标签：AIV 0、AIV 1、AIV 2、AIV 3

右侧执行路径标签：scalar 提交、URMA 远端写入、MTE 本地搬运

底部注释：每个 token 独立处理；同一 token 的重复 rank 只发送一份，接收端再按 expert 展开

### 布局细节

- 8 个 token 行横向排列，每行左侧有一个 token 标签。
- 每两行归入同一个 AIV 分组，用浅色背景或左侧色条区分。
- 5 个阶段列从左到右依次排列，列与列之间用细箭头连接。
- 每个单元格内使用小标签或徽标，不写长句。
- token 2 和 token 5 的“重复 rank 去重”单元格用橙色强调，并加上“去重”徽标。
- “通信描述构造”列使用青蓝色强调，表示由 SIMT 完成。
- 右侧 scalar、URMA、MTE 用三个小卡片纵向排列，不与主矩阵混在一起。

### 数据准确性要求

- token 2 的两个 expert 都在 rank 1，去重后只发送一份。
- token 5 的两个 expert 都在 rank 2，去重后只发送一份。
- 其余 token 若两个 expert 分属不同 rank，则分别发送。
- 接收 slot 按“同一 rank 内的去重后顺序”计算，不需要在图中列出全部具体数值。
- 图中不出现底层通信实现名词。

## 图6 v6：Combine 的 SIMT 流水线

### 目标

展示 Combine 中多个 task 如何形成流水线：读取返回信息、按来源 rank 组织 task、构造远端写描述、本地输出直接聚合，并在当前 task 聚合时提前准备下一个 task 的通信。重点是“通信准备”和“本地聚合”可以重叠推进。

### 画面结构

整体采用“左侧 task 分组 + 右侧两条并行轨道”的布局。

- 顶部：标题“Combine：SIMT 并行整理返回结果并流水执行”
- 左侧：4 个 task 卡片
  - task A：来自 rank 1
  - task B：来自 rank 2
  - task C：来自 rank 3
  - task D：本地输出
- 右侧上方轨道：通信准备（SIMT）
  - 读取返回信息
  - 组织 task
  - 构造远端写描述
  - scalar 提交
- 右侧下方轨道：本地聚合（SIMT）
  - 本地 reduce
  - 写回 token 输出位置
- 中间：重叠提示“当前 task 聚合时，下一个 task 已开始准备通信”
- 底部：路径说明“本地输出不需要远端写，直接进入本地聚合”

### 具体文案

标题：Combine：SIMT 并行整理返回结果并流水执行

左侧 task 标签：

- task A，来自 rank 1
- task B，来自 rank 2
- task C，来自 rank 3
- task D，本地输出

上方轨道标题：通信准备（SIMT）

上方轨道节点：读取返回信息、组织 task、构造远端写描述、scalar 提交

下方轨道标题：本地聚合（SIMT）

下方轨道节点：本地 reduce、写回 token 输出位置

中间重叠提示：当前 task 聚合时，下一个 task 已开始准备通信

底部说明：本地输出不需要远端写，直接进入本地聚合

### 布局细节

- 左侧 4 个 task 卡片纵向排列。
- 右侧分为上下两条水平轨道。
- 上方轨道中的节点依次向右推进，表示通信准备。
- 下方轨道中的节点依次向右推进，表示本地聚合。
- 用一条斜向箭头连接“当前 task 的本地 reduce”和“下一个 task 的通信准备”，突出重叠。
- task D（本地输出）只进入下方轨道，不进入上方轨道的远端写描述部分。
- scalar、URMA、MTE 不作为主视觉，只在必要位置用小徽标表示。

### 数据准确性要求

- task 是通信上的分组，不是模型层概念。
- task A、B、C 分别对应来自 rank 1、rank 2、rank 3 的返回结果。
- task D 表示本地输出，不需要远端写。
- 当前 task 执行本地 reduce 时，下一个 task 的返回信息读取和通信描述构造可以并行推进。
- 图中不出现底层通信实现名词。

## 生成提示词

### 图5 v6 提示词

Use case: infographic-diagram
Asset type: technical article figure
Primary request: Create a clean, flat, 16:9 infographic titled “Dispatch：SIMT 并行整理路由与通信描述”.
Scene/backdrop: light warm Gruvbox Light background #fbf1c7, white rounded cards, thin 1px borders #d5c4a1, generous whitespace.
Subject: Show 8 token rows processed in parallel through 5 stages of Dispatch.
Composition/framing:
- Top-left title: “Dispatch：SIMT 并行整理路由与通信描述”
- Left side: 8 rows labeled “token 0” to “token 7”, grouped into 4 AIV groups: AIV 0 handles token 0 and 1, AIV 1 handles token 2 and 3, AIV 2 handles token 4 and 5, AIV 3 handles token 6 and 7.
- Main area: 5 columns with headers “Router 结果”, “目标统计”, “重复 rank 去重”, “接收 slot 计算”, “通信描述构造”.
- Use compact badges inside cells, not long sentences.
- Highlight token 2 and token 5 in the “重复 rank 去重” column with an orange “去重” badge.
- Use light blue accents for SIMT stages, especially “通信描述构造”.
- Right side: three small vertical cards labeled “scalar 提交”, “URMA 远端写入”, “MTE 本地搬运”.
- Bottom caption: “每个 token 独立处理；同一 token 的重复 rank 只发送一份，接收端再按 expert 展开”
Color palette:
- Background #fbf1c7
- Cards #ffffff
- Main text #3c3836
- Secondary text #665c54
- SIMT #458588
- scalar #b16286
- URMA #d65d0e
- MTE #98971a
- token #d79921
- expert #cc241d
- rank #8f3f71
Style/medium: flat modern infographic, thin lines, rounded corners, no 3D, no shadows, no gradients.
Typography: Noto Sans SC or Inter, Light or Regular weight, no bold.
Text (verbatim): render all Chinese labels exactly as listed above.
Constraints:
- Do not add extra text.
- Do not use WQE, SGE, SQ, CQ, queue, doorbell, Jetty, Channel, staged transport, payload, hidden, metadata.
- Keep the diagram accurate to the 8-token Dispatch example.
- Do not show low-level communication implementation details.

### 图6 v6 提示词

Use case: infographic-diagram
Asset type: technical article figure
Primary request: Create a clean, flat, 16:9 infographic titled “Combine：SIMT 并行整理返回结果并流水执行”.
Scene/backdrop: light warm Gruvbox Light background #fbf1c7, white rounded cards, thin 1px borders #d5c4a1, generous whitespace.
Subject: Show Combine tasks forming a pipeline between communication preparation and local aggregation.
Composition/framing:
- Top-left title: “Combine：SIMT 并行整理返回结果并流水执行”
- Left side: 4 task cards labeled “task A / 来自 rank 1”, “task B / 来自 rank 2”, “task C / 来自 rank 3”, “task D / 本地输出”.
- Right side: two horizontal tracks.
- Upper track title: “通信准备（SIMT）”
- Upper track nodes from left to right: “读取返回信息”, “组织 task”, “构造远端写描述”, “scalar 提交”.
- Lower track title: “本地聚合（SIMT）”
- Lower track nodes from left to right: “本地 reduce”, “写回 token 输出位置”.
- Add a diagonal arrow from the lower track's current task to the upper track's next task to show overlap.
- Middle callout: “当前 task 聚合时，下一个 task 已开始准备通信”
- Bottom caption: “本地输出不需要远端写，直接进入本地聚合”
Color palette:
- Background #fbf1c7
- Cards #ffffff
- Main text #3c3836
- Secondary text #665c54
- SIMT #458588
- scalar #b16286
- URMA #d65d0e
- MTE #98971a
- token #d79921
- expert #cc241d
- rank #8f3f71
Style/medium: flat modern infographic, thin lines, rounded corners, no 3D, no shadows, no gradients.
Typography: Noto Sans SC or Inter, Light or Regular weight, no bold.
Text (verbatim): render all Chinese labels exactly as listed above.
Constraints:
- Do not add extra text.
- Do not use WQE, SGE, SQ, CQ, queue, doorbell, Jetty, Channel, staged transport, payload, hidden, metadata.
- task D must not connect to the remote-write path.
- Keep the diagram accurate to the Combine pipeline explanation.
- Do not show low-level communication implementation details.
