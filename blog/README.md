# 技术随笔与机制拆解 / Engineering Essays & Deep Dives

十篇面向推理系统社区的平权独立文章（中英双语），不设上下级阅读门槛，也不预设读者了解本仓库。其中两篇为行业趋势与系统复盘，八篇为单点技术深潜——从反常现象出发，用示意图还原问题建模、算法推导与通用排障思路。

Ten peer-independent articles (in Chinese and English) written for the inference-systems community, assuming zero prior familiarity with this repository. Two pieces examine macro economics and engineering postmortems; eight are focused deep-dives reconstructing problem modeling, derivations, and reusable debugging habits with diagrams.

## 推理引擎解剖路标 / System Anatomy Map

每篇独立文章对应现代连续批处理推理引擎上的一个具体部位：

```text
  [01 行业趋势与引擎试错税] ──────────────────────────────────────────────────────────┐
                                                                                     ▼
  CPU 调度、解包与核验               GPU 编译模型层与分页缓存                     采样尾声与并发状态
 ┌────────────────────┐       ┌──────────────────────────────────┐       ┌────────────────────────┐
 │ • attn_metadata 与 │──────►│ • 编译图改残差与训快权重    [05] │──────►│ • 词表带宽与 IEEE 754  │
 │   三整数解包  [09] │       │ • GQA 头部打包与 MLA 吸收   [08] │       │   保序位变换      [04] │
 │ • 纯数据调度与百万 │       │ • 权重带宽墙与动态跳层配额  [03] │       ├────────────────────────┤
 │   加盐核验    [10] │       └──────────────────────────────────┘       │ • 静态图指针偷渡  [06] │
 │ • 跨 PCIe 零拷贝   │                          ▲                       │ • 无锁 MVCC/RCU   [07] │
 │   (32 CTA)    [03] │                          └───────────────────────│ • 8 刻度最长路调度[02] │
 └────────────────────┘                                                  └────────────────────────┘
```

## 中文目录 (`blog/zh/`)

### 观点思辨与工程复盘
1. **[`[观点思辨]` 《〈苦涩的教训〉还有后半句：当商业推理退回“脏先验”，真正卡脖子的是引擎试错税》](zh/01-after-the-bitter-lesson.md)**  
   *单晶体管造价企稳与 800:1 算存比把训练与推理拉向相反的经济轨道，促成多元语法查表、跨层索引复用与在线快权重回潮；当编码智能体让局部先验变成可规模化搜索时，无状态引擎征收的“三周拉分支税”成了新瓶颈。*
2. **[`[工程复盘]` 《复盘一次过度设计：删掉 45 万行多面体编译器，换成 40 行最长路》](zh/02-forty-lines-over-polyhedral-ir.md)**  
   *复盘引进 45.7 万行 ISL 多面体库却只用来算 `lag >= 1` 的弯路；展示如何将跨步时间线折叠为 8 刻度一致依赖图，以及如何用剖析器定位双卡 All-Reduce 优先级抢占与可分页内存假异步。*

### 硬核机制拆解与排障方法
3. **[`[极客拆解 · 带宽与配额]` 《为什么 32 个线程块跑赢 9,600 个，跳过 23% 的词却省不下 1 毫秒？连续批处理的带宽配额分析》](zh/03-bandwidth-and-sm-quotas.md)**  
   *建立“外部慢速 PCIe vs 片内 466 MB/层权重流”的双瓶颈模型：推导非抢占式 SM 槽位为何会被 9,600 个查表块塞满饿死矩阵乘，以及部分动态跳层为何省不下时间且暗藏概率分布漂移。*
4. **[`[极客拆解 · 尾声与位运算]` 《解码尾声的带宽账，与 5 行 IEEE 754 保序位变换》](zh/04-sampler-tail-and-bit-hack.md)**  
   *从 `top-p` 过滤后平均仅剩 4.83 个幸存词出发降维词表计算；对照 IEEE 754 位图推导单条 `uint64` 整数比较实现“浮点降序 + 同分下标升序”，并提炼“锁种子逐词对账”与“盯批次最大耗时行”两把排障尺子。*
5. **[`[极客拆解 · 编译图与快权重]` 《从 `assert bs == 1` 到编译态多卡并发：在 `torch.compile` 图里改残差与训快权重》](zh/05-compiler-and-host-jujitsu.md)**  
   *一条完整链路打通图内干预与在线快权重训练：借 `fused_add_rms_norm` 接缝与 `data_ptr()` 进图、10 行 `make_fx` 剪掉 61,440 个多余内核、`grad → vmap(chunk_size=1) → compile` 配 `float32` 增量破解 99.99% 精度吞没、以及双假图对比自动证明 `TP > 1` 零通信更新。*
6. **[`[极客拆解 · 静态图寻址]` 《图是死的，表是活的：怎么在冻结的 CUDA Graph 里偷渡动态指针？》](zh/06-cuda-graph-pointer-smuggling.md)**  
   *借用无绑定描述符表（Bindless Descriptor Table）与空位吸收槽建模静态图动态寻址，展示如何把外部搬家张量的裸指针当成 `int64` 填表并在 Triton 内还原，以及从“流事件记录器”视角预判预热污染、跨流 GC 与纯函数编译器克隆冲突。*
7. **[`[极客拆解 · 显存并发状态]` 《当上百个在线请求各自边跑边改权重：在 GPU 显存里手搓无锁 MVCC 与异步 RCU》](zh/07-gpu-mvcc-and-rcu.md)**  
   *深入落地显存多版本状态的五道硬关卡：融合“写时重定向”省掉 `AdamW` 预拷贝并砍掉 30% 动量显存、用无锁代次账本抓出 ABA 悬空槽、靠两级非阻塞 `wait_event` 与 `abort()` 栅栏实现零冗余页回收、跨进程 `CUDA IPC` 远端凭证隔离、以及五行集合交集划定抢占回滚边界。*
8. **[`[极客拆解 · 跨代际注意力]` 《少读 98% 的历史反而慢 5 倍，换上 DeepSeek 又罢工：从显存布局重看自定义注意力》](zh/08-attention-from-gqa-to-mla.md)**  
   *示范如何用一页纸 Roofline 除法锁定 7:1 GQA 头部打包失效（附 `FullAttentionSpec` 吞并 `RSWASpec` 子类的继承排查），再顺着维度复杂度用矩阵乘法结合律推导出零解压的 DeepSeek MLA 潜空间流式注意力。*
9. **[`[极客拆解 · 批次解包与元数据]` 《连续批处理的行坐标还原：探针实测 vLLM `attn_metadata` 与图内无同步反解》](zh/09-anatomy-of-attn-metadata.md)**  
   *示范如何用 5 步最小混批探针与单请求 MSE 对账测清 `attn_metadata` 的真实语义；从 CSR 行偏移出发推导图内零同步 `searchsorted` 坐标还原、边界夹紧抽页与 `slot_mapping = -1` 读写控制，并划分与上游三整数路由的职责边界。*
10. **[`[极客拆解 · 多租户核验]` 《`h - h.mean(dim=0)` 一行错都不报，却会让多租户推理串味：把连续批处理调度变成可测数据》](zh/10-schedule-as-data-verification.md)**  
   *用零字节 `meta` 追踪拦截隐式同步，将连续批处理压缩为静态整数三元组（Schedule-as-Data）；配合分块/抢占不变性、百万加盐蜕变测试与单行 `return None` 抢占注入，两秒内在笔记本 CPU 上抓出跨租户泄密。*

---

## English Articles (`blog/en/`)

### Positions & Engineering Postmortems
1. **[`[Position]` The Second Half of the Bitter Lesson: Why Commercial Inference Is Returning to Dirty Priors, and Why Engine Forks Break the Economics of Search](en/01-after-the-bitter-lesson.md)**
2. **[`[Postmortem]` Deleting 450,000 Lines of Polyhedral C for 40 Lines of Longest-Path: An Over-Engineering Postmortem](en/02-forty-lines-over-polyhedral-ir.md)**

### Technical Deep Dives & Diagnostic Methods
3. **[`[Deep Dive · Bandwidth & Quotas]` Why 32 Thread Blocks Beat 9,600—and Skipping 23% of Tokens Saves Zero Time: Bandwidth and SM Quotas in Continuous Batching](en/03-bandwidth-and-sm-quotas.md)**
4. **[`[Deep Dive · Bit Tricks & Tail Bandwidth]` Bandwidth Accounting at the Decode Tail—and a 5-Line IEEE 754 Order-Preserving Bit Trick](en/04-sampler-tail-and-bit-hack.md)**
5. **[`[Deep Dive · Compiled Fast Weights]` From `assert bs == 1` to Multi-GPU Compiled Serving: Editing Residuals and Training Fast Weights Inside `torch.compile`](en/05-compiler-and-host-jujitsu.md)**
6. **[`[Deep Dive · Frozen Graphs]` Dead Graphs, Live Tables: Smuggling Dynamic Pointers and Autograd Inside Frozen CUDA Graphs](en/06-cuda-graph-pointer-smuggling.md)**
7. **[`[Deep Dive · GPU MVCC & RCU]` When a Hundred Live Requests Mutate Weights Concurrently: Building Lock-Free MVCC and Async RCU in GPU Memory](en/07-gpu-mvcc-and-rcu.md)**
8. **[`[Deep Dive · Attention Layouts]` Why Reading 98% Less History Ran 5x Slower—and Broke on DeepSeek: Rethinking Custom Attention from Memory Layouts](en/08-attention-from-gqa-to-mla.md)**
9. **[`[Deep Dive · Unpacking Jagged Batches]` Reconstructing Token Coordinates in Flattened Batches: Probing vLLM's `attn_metadata` and Sync-Free On-GPU Parsing](en/09-anatomy-of-attn-metadata.md)**
10. **[`[Deep Dive · Multi-Tenant Verification]` `h - h.mean(dim=0)` Throws Zero Errors While Leaking Prompts Across Tenants: Turning Batch Schedules into Testable Data](en/10-schedule-as-data-verification.md)**
