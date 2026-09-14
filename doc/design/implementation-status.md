# pLLM 编译中段：功能与进展收口

日期：2026-09-12。本文是当前实现、验收和剩余工作的统一入口。设计推演与历史实验
不等于当前功能承诺；它们的阶段性描述以本页状态为准。

**整体计划尚未完成。本轮收口范围是架构缩减后的现有功能闭环，不是扩建优化目录。**
最近一次完整收口已完成源事实／调度分离、运行时取值路径缩减，以及 decoder/recirculation 的数据
端口迁移。构建、CPU 回归、Qwen 数值检查和 EngineCore 跨进程检查均已通过，
启动的验收进程均已结束；本轮没有留下待运行的验收或已知失败项。

该收口快照之后，继续完成了只读图导入、vLLM 计算体和条件 replay 的生产构造器迁移，
并删除 `BorrowedExport` 转发容器。下方分别记录后续验证和完整收口基线，避免将基线
模型验收冒充后续每次修改的完整模型复测。

后续还删除了编译契约中的 kernel 调用数和生产 wrapper 源码解析。调用数只在
测试中诊断融合结果；原图重放、代码身份、分配与完成约定继续决定实现是否可用。

旧 Order 源兼容也已退出：源端口、原生数学绑定和参考解释统一只使用数据值，
不再保留旧源转换及双格式分支。旧格式直接拒绝，迁移方式见下文。

并行计划的重复 payload 推导和安装校验已收敛到原 stream 校验；串行候选统一走
已有 payload 推导入口。没有新增产物格式或“可信”标志，具体职责与验证见下文。

函数体签名与内容也已合并为一次源校验，内部内联不再重复遍历已经检查过的
函数体；原源编码与最终展开程序保持不变。

随后清理了 DecoderSource 的冗余层号计算、单层段的零次 Scan，以及没有 tap 时
不会使用的加法图捕获。这一步改变源程序结构，需要重新生成对应计划。

最新冷规划分析定位到 ISL 的整数实现。现已启用上游 `imath-32` 快路径，大整数
仍回退到任意精度；同一纯描述案例从 113.8 秒降到 49.3 秒，全部规划产物保持一致。

共享调度目录进一步改为按需生成窗口与分块候选。原预算、交错顺序与去重规则
保留，目录填满后停止后续候选构造；本轮未证明有明显的额外性能收益。

源程序语义校验现已统一到 Rust。Python 的重复 SSA/签名/资源规则已删除，
共享 codec 保留表示类型检查；原有 execution 编解码入口重导出同一实现。

调用块结构提取进一步统一到现有 native 入口，删除 Python 提取器和额外的完整
结果校验往返。串行策略检查也不再为比较参考顺序而生成整份取值访问器。

Scan 现在区分真正的 carry 与固定输入，DecoderSource、tokenwise 和 TensorScan
已删除固定输入的逐轮返回接线。同一 recirculation 的六个 Scan 减少 30 个无用
输出；除源摘要外，全部分析字段保持一致。

坐标与 tap 分支随后完成静态化：`Coordinate` 直接声明仿射下标，Constant/Scalar
的仿射恢复路径已删除；构造期可判定的 tap If 全部展开或分段，recirculation 源
If 从 10 降为 2。存储归属推导拆为调度无关的 provenance（一次推导、候选共享）
与按调度特化的生存期 envelope，修复了 24 层配置下候选 payload 推导越过 ISL
操作配额的问题；详见下文对应小节。

## 这轮具体收掉了什么

| 原问题 | 当前实现 |
| --- | --- |
| 参考执行和优化执行分别解释同一源程序 | 源事实只走一条提取路径；ISL 选择只消费调用域和依赖关系。 |
| frame 沿源 AST 查常量，并维护另一份常量表 | 删除 AST 回查与 `_literals`；整数使用已有仿射访问器，bool/float 使用已有 primitive。 |
| frame 接收完整 Program 并重复绑定尺寸 | frame 只接收来源绑定、输入值和执行入口已校验的尺寸快照。 |
| 顺序 token 进入数学值表，再由运行时到处过滤 | Order 类型及兼容路径已删除；源端口、数学值表和执行端口统一只含 DataType。 |
| 不变的循环输入被表示为逐次递推 | 直接引用外层定义；保留真正变化的 carry。 |
| decoder/recirculation 手工传递和返回资源 token | 两个生产构造器直接生成数据端口；资源使用已有声明、读写关系和参考顺序。 |

数据端口迁移与早期值表缩减的同一份 recirculation 对照结果：

- 源中的 **108 个 Order 声明已删除**，操作和数学计算保持原样。
- 三类 Scan 的输入数从 **9/12/10 降到 5/9/7**。
- 派生数学值位置从 **193 降到 89**，语句模板仍为 **28** 个。
- 新源码与“先检查旧源码，再去掉 order 端口”的结果逐项一致。
- 除源摘要外，新旧分析事实和候选完全一致；此前值编号缩减通过了 ISL 来源／依赖等价对照。

这些数量是源表示和编译产物的缩减，不能解释为 GPU 流量、显存或吞吐按相同比例改善。
可复核过程见 [中段精简审计](recirculation-middle-end-audit.md)。
后续层号清理又将数学值位置从 89 降为 85，具体对照见下方 DecoderSource 收口；
后续新源不再是只删除 Order 的逐字投影。

## 当前功能边界

| 能力 | 当前状态 | 适用范围和限制 |
| --- | --- | --- |
| 原计算图导入 | 已实现 | 使用 Torch 的原图、输入输出签名、原字段和 guard；当前计算块要求受支持的静态张量布局。 |
| 中段与 modeling 分离 | 核心边界已建立，前端迁移未完成 | contracts/execution 禁止导入 vLLM/Transformers；vLLM 外层接入仍限制 Qwen2，不能宣称任意模型即插即用。 |
| 符号依赖与 ISL 调度 | 已实现并接入区域执行 | 保留参数域、值来源、RAW/WAR/WAW、动态效果边界；支持自动重排、分块、窗口及部分 wavefront。具体机会取决于真实依赖。 |
| 计划执行和跨 stream 完成 | 已实现 | 使用既有执行器、原 owner 和完成协议；未知共享上下文不会被自动当作可并发。运行时不调用 ISL。 |
| 数学与实现候选 | 已实现有界搜索及源重放 | 支持相邻调用 outlining、矩阵重结合、独立 GEMM 合批和受限 Torch 编译；不是任意数学等价证明器。 |
| 模型候选选择 | 默认直接安装参考调度；实测 TokenwiseTuning 为显式 opt-in | 逻辑流量上界调度无关且目录候选无 rewrite flops，静态排名恒为常数函数，默认路径因此不探索目录、不推导非参考 payload；`ExecutionOptions(tuning="measured")` 时才构建目录并由 worker 逐项测量按完成时间选择。 |
| 状态正确性 | 保留原 PTTS 机制 | pin、COW、版本发布、KV 恢复和资源权限仍由原所有者管理；没有新增状态 owner。 |
| 内存依据 | 已有区域范围的上界与拒绝机制 | 覆盖模型化张量、保留对象及已声明临时空间；尚不是包含所有库工作区、allocator、模型权重和 KV 的整设备预算保证。 |
| 成本选择 | 以访问、存储生存期和完成时间为依据 | FLOPs 是辅助事实。逻辑读写量不等于实际 HBM 流量，样本计时不证明全局最优。 |
| 真实模型接入 | 支持已验收的 Qwen2/vLLM 路径 | 当前重点是单设备、eager、FA2、静态单 token 计算块及 recirculation；完整异构模型、其他模型族仍未验收。 |
| 稳定吞吐收益 | **尚未证明** | 短请求和两 token smoke 的小幅差异不作为性能验收。 |
| 新教程、notebook 和示意图 | 尚未交付本轮更新 | 继续排在架构、功能和真实性能验收之后。 |

数值变换默认允许，可用 `ExecutionOptions(allow_numerical_transforms=False)` 关闭。
只有实际选中的数值变换会提示。原 guard 失效会拒绝陈旧绑定；自动重新捕获和规划
尚未实现。数学变换不能改变算法更新次数、RNG 归属或状态发布语义。

## 旧 Order 源的退出与迁移

数据端口形式直接通过 `Function.resources/accesses` 和 Snapshot/Publish 指明资源。
所有生产构造器已迁移，现已删除 Python `OrderType/ValueType`、Rust `Type::Order`、
`without_resource_orders` 和参考解释器的 Order 运行时对象。数据值不再使用可空
编号，也不维护只用于旧顺序 token 的使用次数。没有新增格式开关或另一个执行器。

迁移方式是从原算法构造器重新生成 Program，并重新规划、选择和安装；旧 order
源和对应派生产物不再接受。Python 解码及 Rust 入口会拒绝整个旧格式，包括原本
合法的链、重复／丢弃／过期 token 和 bank 内藏的 order。不能删除字段后沿用旧的
签名或容量证据。历史源快照留在 evidence 中用于审计，不是当前可安装工件。

资源声明、访问秩和类型、SSA 作用域、纯 Map／While 条件限制继续校验。资源
冲突由既有 RAW/WAR/WAW 和参考顺序推导；页权限、版本、发布、COW 和完成仍
由原 PTTS 与后端 owner 负责。退出 Order 并不允许绕过这些边界。

退出检查已收口：release 扩展重建安装，Rust 18 项通过；仓库
`/workspace/pllm/.venv/bin/python` 运行 CPU 回归最终 **369 项通过、61 项跳过**，
CUDA 所有权、原生 kernel、worker 候选安装及原 recirculation 共 **15 项通过**。
61 个跳过不算 GPU 验收，15 项集合中也包含 CPU 参数和 CPU recurrence 检查。
首次 CPU 回归发现三处未迁移干净的测试接线，修正后整组重跑通过；原权限、别名
拒绝和状态断言保留。typing、import layers、region authority、diff 门禁通过。

同一归档 recirculation 数据源的分析与退出前逐字段一致，包含所有候选。详见
[Order 退出证据](evidence/order-retirement-results.json)。这次没有重复加载完整 Qwen
或 EngineCore，先前模型验收仍是历史基线；冷规划和对齐吞吐仍待单独完成。

## 收口后的生产构造器迁移

删除了只读借用中的 `ownerN` 链、vLLM 前后半图中的 context/kv/weights token
接线，以及条件 replay 的 model/trace carry。原权重字段、原图输出和 mutation
签名继续使用已有接口；未增加新的容器、注册中心或原计算的替代实现。

后续检查使用同一仓库虚拟环境：CPU 组合／只读导入 5 项通过；CUDA 只读借用与
worker 命令链 2 项通过；条件 replay 的 CPU/CUDA 检查 12 项通过。条件 replay
还完成新旧源的精确投影对照，独立进程的纯描述规划检查也通过。记录见
[生产构造器迁移证据](evidence/producer-migration-results.json)。这里只记录实际执行过的检查，不宣称重跑了下方
完整 Qwen/EngineCore 基线。

## 编译契约的进一步缩减

生产路径已删除 `_compiled_kernel_calls` 和 Python/Rust 的 `kernel_count` 字段。
该数字不参与依赖、调度或容量计算，却使编译能否接受一个图取决于 wrapper
文本格式，并误拒绝被编译器消除为输入 view 的图。现在实际分配清单可以为空；
原输入只读约定和逐输出复制仍成立，复制的容量继续计入原计划。

原逐元素、outlining、GEMM epilogue 的融合断言保留在测试诊断中，按生成代码的
既有摘要查找对应产物，不向中段传递新的诊断对象。生产路径仍计算代码摘要并在
本地重放编译器，继续拒绝伪造代码身份、分配、隐式工作区和输入修改。

本次 release 已重建安装。使用 `/workspace/pllm/.venv/bin/python` 运行的
CPU/CUDA 编译、原 export 和 worker 候选安装检查 **43 项通过**，独立描述规划及
worker 重建 **2 项通过**；补充旧工件拒绝断言后，相关用例单独复跑通过。
Rust **18 项通过**，typing、import layers、region authority 和 diff 门禁通过。
新增 view 检查同时覆盖 CPU/CUDA：零 kernel、零编译内部分配、正确数值、独立
输出根和包含复制的容量上界。证据见 [编译契约缩减记录](evidence/kernel-contract-results.json)。

迁移策略：带有旧 `kernel_count` 字段的编译 imports/选择工件在 Python 和 Rust
边界均被拒绝，必须从原 Program 与原计算图重新规划、选择、安装；不能删除字段
后沿用旧的签名和容量产物。Program 源格式未因此改变。本次未重跑完整 Qwen 或
EngineCore 模型验收，也没有新的对齐吞吐结论。

## payload 校验职责的收敛

候选评估与计划安装现在沿同一职责分工工作：

- 串行候选由 `region_payload::derive` 组合原容量与保留关系；没有 retained imports
  时直接使用已有空保留事实，源与调度合法性已由前面的 storage/capacity 检查覆盖。
- 并行候选由 `region_streams::validate` 从原源重建完整计划，包含嵌套 payload、
  并发临时空间、等待和保留范围；AER 在其通过后复用 payload，并核对候选的 phase
  schedule。Python 安装保留同样的完整 stream 校验，删除单独 payload 校验往返。
- 候选零的 payload 推导检查原 imports，随后要求其 imports 和参考顺序与请求一致；
  删除此前只为这些检查单独推导的 reference storage。实际函数绑定及调用时 owner
  权限仍分别核验，不使用新的跳过检查开关。

原 wire 产物不变。小网格对照的源、proposal、请求和选择结果逐项一致；并行安装
从两次 native 校验调用减为一次，编码入参从 13,395 字节降为 7,412 字节。这只描述
该样例的安装边界，不是模型吞吐或整体冷启动的改进幅度。

使用 `/workspace/pllm/.venv/bin/python`，主要改动的 CPU 检查 **44 项通过、
19 项跳过**，CUDA 并行选择、编译重放和 worker 安装集合 **4 项通过**（含一个 CPU
拒绝用例）。最后去除单独 reference storage 推导后，补充回归 **10 项通过、1 项
跳过**，并重新确认上述产物完全一致。伪造嵌套容量、保留关系、设备峰值，或基线与
候选零同时漏 imports，均被拒绝。最终 release 重建、Rust 18 项及全部门禁通过。
本次没有重复完整模型或对齐吞吐验收。证据见
[payload 校验收敛记录](evidence/payload-authority-results.json)。

其余重复 source 编码和作者端不变量 carry 仍需继续处理；源语义校验已在后续统一，
不能据此宣称验证职责已全部收敛。

## 函数体校验与内联的收敛

`RegionFunctionBody.validate` 将外层声明交给同一次源校验。声明的数据签名、
资源与访问约定仍按原规则检查，但声明不会变成 body 内部可调用的 import。
Rust 后端接入也使用同一检查方式，删除复制整个 Program 并追加声明再遍历的
路径。自导入、自调用、隐藏捕获、错误结果类型、未声明资源、缺失借用访问和
不允许的 body 参数仍被拒绝。

内部 `_inline_block` 只做变量重命名，复用已有 Value/Block；所有 body 在展开前
检查，最终组合程序再完整检查。公开 `inline_region` 仍检查任意传入的原源。
重复返回和输入别名保留，冲突名称通过既有 fresh-name 规则处理，没有引入另一种
IR 或跳过校验标志。后续又将 Python/Rust 的源语义规则统一到 native，见下文。

三层 DecoderSource 使用原后端 partition bodies 的前后对照中，完整 Python
校验调用从 8 次降为 4 次，展开后的 Program 逐项相同。这是源组合开销的缩减，
不是完整模型冷规划或吞吐验收。

本次使用 `/workspace/pllm/.venv/bin/python`，CPU 组合、源规则、参数、Scan、
只读导入、独立规划和 worker 重建检查 **108 项通过、1 项跳过**；CUDA worker
候选生成、原生后端接入与安装检查 **1 项通过**。Rust **19 项通过**，release 已
重建安装，typing、import layers、region authority 和 diff 门禁通过。原图、资源
声明与展开源对照见 [函数体校验收敛记录](evidence/body-verification-results.json)。
本次没有重复完整 Qwen/EngineCore 或对齐吞吐验收。

## DecoderSource 的层号与空路径清理

连续层段现在直接计算 `index + (first_layer + 1)`，删除中间 ordinal 及其常量。
只在确有后续层时构造 Scan 和后续层声明；单层段直接返回原首层结果。没有 tap
的段不捕获或声明残差相加图。有 tap 时沿用原加法图及 tap 顺序，不连续层号仍
通过原 layer_ids bank 读取。没有新增源操作、控制属性或模型数学替代实现。

归档的三层 recirculation 样例中，Constant 从 23 降到 19，Scalar 从 22 降到 18，
数学值位置从 89 降到 85；语句模板仍为 28。域、数据依赖、资源 RAW/WAR/WAW
及总依赖逐字一致。删除源操作会改变参考时间坐标的数字；ISL 比较确认三个候选
的实例先后关系均保持一致，不用字符串相等代替调度等价。

单层、多层、非连续层段及不同 tap 顺序的执行检查继续核对原层调用次数、结果、
请求/token 坐标和 KV 绑定；接近有符号索引上界的单层及双层源也通过 Python/Rust
校验。这不是一般坐标导入机制的完成；循环不变量接线仍需继续处理。

仓库 `/workspace/pllm/.venv/bin/python` 运行的相关检查 **25 项通过**，包含
层段执行、计算体组合、独立描述规划和 CUDA worker 安装。typing、import layers、
region authority 及 diff 门禁通过；本次只修改 Python 源构造器，未改 native 实现。

这次另外重跑了 Qwen2.5-0.5B/BF16 的完整模型数值 smoke：**8 个正常行、6 个
replay 行、14 次独立数值比较**，KV/RNG/清理检查通过，fallback 为 0，选中编译
候选 12，验收进程已正常退出。共有 16 个测量候选；两 token 请求的计时不能作为
对齐吞吐收益。模型冷准备仍超过十分钟，是明确未解决的工程问题。本次未重复
EngineCore 跨进程完整模型验收，先前记录仍是该项的历史基线。

结果及源对照见 [DecoderSource 收口证据](evidence/decoder-source-results.json)。
改变了源结构的旧计划必须重新生成，不能沿用其来源摘要和派生证据。

## ISL 整数后端与冷规划成本

实际采样表明，纯描述规划的大部分时间落在 native ISL 运算；大量底层样本属于
imath 整数运算和内存分配。现有 `isl-rs 0.3.1` 将 ISL 固定为纯 imath。现在使用
ISL 上游的 `imath-32`：小整数直接处理，超过范围时回退到 imath 任意精度。
这不限制源索引、系数或求解器中间值为 32 位，也没有放松依赖和容量证明。

两个独立进程使用相同的当前源构造器、两请求 × 24 层、最多 64 行、BF16/CUDA
布局描述及原 partition bodies。全程没有初始化 CUDA，也没有加载权重。两次均
启用相同的 cProfile/py-spy 采样；下表是单次诊断记录，不是稳定吞吐测试。

| 纯描述规划阶段 | 纯 imath | imath-32 |
| --- | ---: | ---: |
| 调度目录 | 30.18 秒 | 21.66 秒 |
| 并行计划 | 20.97 秒 | 7.27 秒 |
| payload 分析合计 | 61.76 秒 | 19.78 秒 |
| 完整调用 | 113.82 秒 | 49.30 秒 |

源程序及全部 **8 份 proposal（含调度、容量、保留和并行证据）逐项相同**。
速度改善来自同一精确计算的底层实现，不是减少候选或缩小参数域。

为保证普通 checkout 与离线构建使用相同设置，现有依赖作为固定 path dependency
保存在 [rust/vendor](../../rust/vendor/README.md)。上游 Rust 绑定、ISL 和 imath
算法源码未修改；本地差异仅为整数构建选项与源码分发所需的打包/执行权限处理。
已按原 crate 校验和逐文件核对来源，保留并随发行包附带许可证。源码分发包已在
独立空 target 目录完成离线 release 编译，确认不依赖工作区临时构建配置。

使用 `/workspace/pllm/.venv/bin/python`，CPU 回归 **379 项通过、57 项跳过**；
Rust **20 项通过**，包括 2^80 系数和 2^96 偏移的精确关系检查。Qwen 完整模型 smoke
再次通过 **14 次数值比较**、KV/RNG/清理及零 fallback；本次完整 smoke 为 **355 秒**，
此前同配置约 11 分钟。不同样本的候选计时和最终选择会波动，不能据此宣布生成
吞吐已提高。冷准备仍需数分钟，尚未完成该项工程验收。

最终 release 构建及 typing、import layers、region authority、diff 门禁通过。
证据见 [ISL 冷规划记录](evidence/isl-integer-backend-results.json)。

## 共享调度目录的按需生成

目录不再先生成两份完整窗口/分块候选列表，再从中交错截取。现在共用同一份
源分析，按原顺序取用各组候选；失败尝试不占用成功候选预算或组间轮次。
每个实际生成的候选仍检查完整参数域的覆盖与依赖，组间去重仍保留相同实例的
先后及同时关系。没有缩小候选预算、改变 tile/window 参数或增加运行时模式。

单独的窗口/分块探索入口继续收集原预算内的结果。共享目录只记录实际尝试过
候选的诊断；未计算的尾部候选不会预先产生诊断。这里只提前枚举分块路径和整数
形状，不提前构造其完整 schedule map/tree 或执行证明。

16 组空域、单点、尾块和预算对照中，目录/窗口/分块 **48 份输出逐项一致**。
两请求 × 24 层的纯描述案例也返回完全相同的 8 份 proposal。相同采样方式下，
完整规划从 49.3 秒变为 48.0 秒，不能据此认定明显提速。采样器在子进程完成后
报告了一次回收错误；另一次不带采样器的直接运行已正常结束并核对相同产物，
该次耗时不与带采样器的记录混作性能比较。

使用 `/workspace/pllm/.venv/bin/python`，相关 CPU 检查 **23 项通过、10 项
跳过**，CUDA worker 候选安装 **1 项通过**；Rust **20 项通过**。release 重建及
typing、import layers、region authority、diff 门禁通过。未重复完整 Qwen/
EngineCore 或对齐吞吐验收。证据见 [按需目录记录](evidence/lazy-catalog-results.json)。

## 源语义校验的唯一实现

Python `verify` 从 209 行规则缩减为 18 行入口与结果类型读取。SSA 定义/使用、
操作签名、循环/分支接口、参数关系、资源访问及外层函数声明统一由原 Rust
verifier 检查。`validate_region_program` 使用同一入口，native 规划和 worker
安装仍直接校验其输入；没有新增跳过检查标志。

通用编解码已移到 `contracts/codec.py`，只依赖契约类型和必要的设备类型表示。
`execution/codec.py` 重导出原公开函数，保留原 Torch 图/存储适配代码，转换器
只有一份。Python 表示检查仍需保留：错误枚举或整数子类不能在编码时改变含义，
需要 tuple 的字段不能悄悄接受可变 list，无界 Python 整数常量必须能表示为 i64。
原 native untagged 解码将溢出的 u64 常量重试为浮点数的路径也已删除。

归档的有效 Program、8 份 proposal、选择请求与选择结果，编解码结果及编码字节
均与旧入口一致。错误表示现在会更早被拒绝，验证入口仍统一抛出 ValueError/
InvalidProgram。原常量的 bool/int/float 区分、负零、大数、非法捕获、纯区域效果
限制和外层声明检查继续覆盖。

使用 `/workspace/pllm/.venv/bin/python`，完整 CPU boundary 集合 **492 项通过、
93 项跳过**；CUDA worker、原页发布和 replay 集合 **14 项通过**。Rust **20 项通过**，
release 已重建，typing、import layers、region authority 及 diff 门禁通过。
证据见 [源校验统一记录](evidence/source-verifier-results.json)。本次未重复完整模型
或对齐吞吐验收；非必要的源编码往返及作者端不变量 carry 仍待继续精简。

## outlining 提取与参考顺序的收口

`outline_region_program` 直接读取已有 Rust 入口返回的原调用体、外部输入和
全部逃逸结果，删除 Python `derive_call_outline` 与 `map_outline_region`。
Python 继续发现可尝试的调用范围，并从原 Torch 图重建实际计算体；外来提取
证据仍通过原 native 验证，实际图也继续做本地重放与访问事实比较。

有/无逃逸中间值的两个对照中，原源和提取结果逐项相同。native 边界调用从
3 次降为 1 次，编码入参分别从 7,767/8,199 字节降为 1,568/1,573 字节；新入口
会返回完整提取结果，因此这些数值只表示入参量，不是双向总流量或内存收益。

串行策略直接使用源分析的参考顺序，取值访问器只在执行绑定需要时生成。
两个零长度参数化用例发现旧分析出口尚未规范化参考顺序，现已在 native 出口
复用原规范化规则；参数边界、后续合法循环长度及数值策略检查保持原要求。

使用 `/workspace/pllm/.venv/bin/python`，最终 CPU 回归 **142 项通过、22 项
跳过**；CUDA outlining/worker 集合 **16 项通过**，规范化后的零长度串行策略
补充检查 **3 项通过**。Rust **20 项通过**，release 重建及全部门禁通过。
本次未重复完整模型或对齐吞吐验收。证据见
[调用块提取收敛记录](evidence/outlining-authority-results.json)。

## Scan 的 carry 与固定输入

Scan 增加明确的输入分组：前段是递推 carry，末尾 `invariants` 个输入保持固定。
body 接收索引和全部输入，只返回新 carry 与需要收集的值；固定参数无需返回。
这与区分循环状态和常量参数的计算模型一致，不将只读性作为优化器可随意采信的
注解。所有值仍服从原不可变数据、资源依赖和所有权约定。

DecoderSource 的坐标放在固定输入段；tokenwise 外层行循环没有数据 carry，
只固定元数据并收集输出；TensorScan 的权重参数和序列 bank 也不再生成无用的
final 输出。原公开 TensorScan 调用方式不变。默认 `invariants=0` 保留 all-carry
源含义，新旧源都使用同一规则；新增字段影响来源和产物摘要，旧派生计划需重建。

同一三层 recirculation 的六个 Scan，carry 数从重复的 **5/9/7 降为 0/4/2**，
删除 **30 个无用输出及对应 body 返回项**。数学值位置仍为 85、语句模板仍为 28；
除源摘要外，全部分析字段，包括来源、R/W、依赖及候选，逐项一致。

结构执行器只更新真正的 carry，固定输入保持到循环结束。容量分析按同一分组
计算前轮结果和收集值的保留范围。零次及非零次、仿射及动态 Scan 的 CPU/CUDA
检查同时比较输出、原输入和真实存储峰值；调用方原有存储与区域新增存储分别
计量，不将原输入的 detach view 误算作新增分配。

使用 `/workspace/pllm/.venv/bin/python`，CPU 回归 **419 项通过、78 项跳过**，
GPU/动态 Scan/worker 集合 **29 项通过**，原 recirculation 回归 **1 项通过**，
覆盖输出、更新/重放次数、原状态、取消和清理。Rust **20 项通过**，release 重建
及 typing、import layers、region authority、diff 门禁通过。本次未重复完整模型
或对齐吞吐验收。源与分析对照见 [Scan 固定输入记录](evidence/scan-invariants-results.json)。

## Coordinate 直接声明、tap 静态化与存储推导复用

坐标改为 `Coordinate` 操作直接声明 `AffineIndex`（常数加按位置引用输入的
系数项）：验证器拒绝越界、非仿射操作数和重复项；Rust 依赖分析设直达分支，
原 Constant int→affine 与 Scalar combine 恢复路径已删除，删除后这些位置是
普通语句，不退化为 opaque。decoder、tokenwise 与区域组合构造器已全部迁移，
`Scalar` 只保留真正的动态标量与布尔谓词。

构造期可判定的 tap 分支同时静态化：命中首层的 tap 在构造期展开为加法调用，
其余 tap 的命中迭代是构造期常量，层 Scan 在 tap 层处分段、命中迭代展开为
单段直写。recirculation 源 If 从 10 降为 2（仅保留按请求资格的真动态分支）。
三层对照源语法树为 24 Call、19 Coordinate、6 Scan、2 If，分析语句 28→18、
值 85→61、诊断清零；新旧源参考执行的调用序列、效果事件与数值逐字段相等。

存储归属推导按调度相关性拆分：provenance（源分析、来源传递闭包、存储路由、
消费者、保留集、出生映射）与调度无关，对每个（源, 导入）推导一次并缓存，
候选调度共享；只有生存期 envelope 按调度特化。envelope 的字典序比较改在
时间空间全集上进行、再经 schedule 接回实际时间，最终关系与原表述相同。
真实规模（24 层、64 packed row）配置下，tap 展开结构使调度片从 60 涨到 84、
最胖串行候选调度文本 34.6KB，旧推导在第 4 个串行候选上越过 20M ISL 操作
配额，`make_llm` 即失败；修复后 4 个候选全部在配额内完成，派生 payload 与
旧表述逐字节一致，envelope 中间关系从约 7000 片降到约 100 片。小配置的
语句数下降（28→18）不能外推到真实配置，以真实配置为准。

验收：Rust **21 项通过**，release 扩展已重建安装；CPU 边界回归 **376 项通过、
63 项跳过**（含其余边界文件共 481 项通过、99 项跳过）；原 recirculation
回归 1 项通过；Qwen recirculation smoke 通过：8 个正常行、6 个 replay 行、
14 次独立数值比较、零 fallback，13 个候选完成测量；EngineCore recirculation
检查通过：候选全部完成测量、并行与变换候选存在、重复请求得到 replay 行、
零 fallback。typing、import layers、region authority、diff 门禁通过。执行
对照见 [坐标与 tap 迁移对照](evidence/coordinate-tap-comparison.json)。

## 安装路径收敛（校验三层化落地）

校验按三层重新划界：语义与物理层（Torch guard 重查、`run()` 动态权限、PTTS、
owner 权限）永远在线；传输卫生层只保留 codec 正常解析；digest/签名一致性
比对层整体取消（项目未发布，无历史格式负担）。

同进程重派生校验的四个生产调用点已删除：计划组装的
`validate_region_payload`/`validate_region_streams`（D1）、绑定的
`validate_region_bindings`（C1）与 stream 计划校验（C2）、游标打开的
`validate_lowered_region_schedule`（C3）。组装与绑定保留纯 Python 一致性
（canonical imports、payload 相等、预算、schedule_map 一致、cell 与声明逐项
匹配），`RegionExecutionPlan.retained` 等随之失效的字段一并删除。worker 对过线
proposals 不再逐份触发 Rust 重派生：`with_catalog` 的检查是 codec 解析、参考
proposal 与本地派生参考逐项相等、预算与通道一致性。完整重派生能力保留为
外部/调试入口：`with_schedule`、`bind_selected_region`、`bind_selected_schedule`
及移入 `test/revalidation.py` 的六个校验包装。

调优 finish 不再重放绑定：`TokenwiseTuning.finish` 与 `optimize_region` 直接
安装已测量的执行器（`_visited` 全集检查保证胜者已测量），数值变换警告照发且
取自实际绑定的 cells；候选编译从两次降为一次。VllmSession 构造时的重复物理
核验已删除（`bind_backend` 的唯一核验两条构造路径必经）。digest/签名比对层
删除了 `BackendWorkerState.compilation_signature`、`TraceSnapshot.schedule_signature`
两个过线字段及三处运行期比对；PTTS 页描述符的签名门（语义物理层）、候选去重、
Function 命名与警告标签中的功能性 digest 保留。

验收：Rust **21 项通过**（无功能改动，作兜底）；完整 pytest **614 项通过**，
仅剩 `test_inplace_region` 两个 CUDA 数值失败——该失败在 T1 前基线（eb75810）
上同样出现，是存量问题，与本收敛无关。typing、import layers、region authority、
diff 门禁通过。改动 31 个文件，+105/−210。证据见
[安装路径收敛结果](evidence/install-convergence-results.json)。

## 默认启发式选择与冷调优 opt-in

默认冷启动不再对全部候选做 worker 实测，也不再为默认路径推导候选目录。
`ExecutionOptions.tuning` 默认 `"heuristic"`：attachment 直接安装参考调度，
不探索目录、不推导非参考 payload、不构造 stream 候选、不发
TUNE_BEGIN/CANDIDATE/FINISH，`region_selection` 保持 None。

这比 T3 的静态排名更进一步，依据是一个结构事实而非实测退化：候选逻辑
流量上界是调度无关的（输入 storage 的读取实例数 × 根字节 + 语句域实例数
× 每调用流量事实，推导不引用调度 map），目录候选又不携带 rewrite flops
——任何 source-first 静态排名在此事实上都是常数函数，必然选中参考候选。
排名函数（`heuristic_candidate_cost` / `heuristic_default_proposal`）与
`tuning_flop_byte_ratio` 常量因此退役；边界测试改为直接断言全部候选的
logical bytes 与参考一致。tuning 模式经 `RegionSelectionPolicy.tuning`
进入签名时间线，worker 侧重推导与 driver 一致；`tune_native_schedules`
以签名策略为唯一来源，启发式分支直接返回。

实测调优为显式 opt-in：`ExecutionOptions(tuning="measured")` 时才探索目录、
推导全部候选 payload 与 stream 计划、按候选最大值储备 workspace，TUNE_*
全程、`TokenwiseTuning` 与 `with_measured_selection` 安装路径原样保留。

效果（Qwen2.5-0.5B recirculation，2 请求 × 24 层真实 probe facts）：
`compile_tokenwise_region` 46.6s → 2.1s（剩余为 reference provenance +
payload），`make_llm` 冷启动 82.6s → 36.7s。

验收：CPU 全套通过（含签名时间线新字段的 Rust 镜像与 worker 重推导一致）；
Qwen smoke 默认路径参考候选正确安装、14 次独立数值比较不变；EngineCore
smoke 以 `tuning="measured"` 保留实测路径的跨进程覆盖。证据见
[启发式选择结果](evidence/heuristic-selection-results.json)（T3 原始设计）
与[默认路径消除结果](evidence/t4-followup-elimination-results.json)。

## 跨请求保序提升的调度搜索（T4 定向后续 ①）

③ 把目录推导移出默认路径后，调度搜索只服务实测 opt-in 路径；它的成本是
ISL 系数 ILP，变量数随语句数增长（真实 84 语句配置 636 个变量）。真实负载
的结构是一个请求的语句组在程序顶层复制 R 次：path[0] 分组、尾部路径与
extent 仅差 size 参数后缀（tokens0/tokens1、prefix0/prefix1 完整成族），且
全部跨请求依赖（trace、packing 行）都从小请求号指向大请求号。

据此只对一个请求副本做调度搜索，再按 Θ(S_r(i), p) = (r, φ_r(i, p)) 保序
提升：请求序号作为最外层时间维，所有跨请求依赖被严格满足。半问题 = 请求 0
语句 + 其余请求参数的存在性投影，由未改动的 `schedule_candidates` 求解；
提升是对半调度 map 的文本重写，重新读入、与全域相交后由精确的全域
`legal()` 裁决——任何结构不匹配或检查失败都回退全域搜索并记录诊断，正确性
没有任何放松。该思路先在 docs/lift_prototype.py 上以 2/3 请求验证精确，再落入
正式 explore 路径（analyze 的 optimize 分支）。

效果（同一真实 84 语句配置）：analyze optimize 分支的搜索 8.6-9s → 0.24s，
`explore_region_schedules` 17.37s → 8.31s（残余为全域 window 构造与精确序
比对，不属于本步范围）。目录组成：参考、isl-locality、isl-parallel 与 window
候选保留；tile 候选对复制型程序消失（提升树的最外层请求带不可置换）——实测
安装候选不变（13 个，parallel (2,4,6,9)、transformed (8,9)）。

验收：新增 Rust 钉住测试（共享原子资源上的两个参数化副本：提升生效、无回退、
三个候选均过 legal()）；cargo 22 项通过；CPU 全套 612 项通过，2 项失败为
test_inplace_region 的 CUDA 位级数值比较，在提升前构建上同样失败（本机 GPU
数值问题，且默认路径不进入改动代码）；typing / layers / authority 全绿；
Qwen 默认路径 smoke 与 EngineCore `--recirculation-only` 实测 smoke 均绿
（selected 5，13 候选全部可行，fallback 0）。顺带修复两处既有问题：
test_region_model_proposals 补上 measured opt-in 的资金储备（③ 遗留，CUDA
专属测试），enginecore smoke 移除对 3441c8b 已删除签名字段的断言。证据见
[跨请求提升结果](evidence/t4-followup-lift-results.json)。

## 调度目录去重的单站点收敛（T4 关账）

语义去重原来分散在两处：window 族逐构造对参考和基各跑一次全域
`equivalent_order`，目录 merge 再做全对语义比对。现收敛到 merge 单一站点：
window/tile 族按构造器忠实枚举（含与参考等价的产物），merge 先做文本过滤，
再对幸存候选跑精确序比对；每个族的回合拉取到首个不同候选为止，重复不消耗
轮转回合，因此目录顺序与旧实现逐项一致。去重总次数由各族的发射上限结构性
有界，无需人工预算。tile 恒等构造器规则（size ≥ extent 时等价于基）经评估
拒绝：真实配置层段 extent 21 大于全部 tile size，永不触发。

T4 其余条款按本计划"收益不明显则不合并"关账（用户决策 2026-09-14）：基线
复测显示原验收锚点（explore 20.6s + payload 18.6s，共 48s）已被 T1 结构性
修复兑现，当前同一真实配置共 7.7s；剩余大头是 tile/window 候选的 payload
（1.0-2.3s），恰是串行 (a,m) envelope 摘要不适用的非阶段主序调度，T4.4 够不到；
T4.3 的 compute_schedule 目标只剩 0.35s。验收：真实配置 8 个候选 schedule
map 逐字节一致、streams 签名一致。证据见
[T4 去重收敛结果](evidence/t4-structural-dedup-results.json)。

## 最近完整收口的验收基线

Python/pytest 环境统一为 `/workspace/pllm/.venv/bin/python`。CPU 集合在独立进程中
设置 `CUDA_VISIBLE_DEVICES=''`，CUDA 项目的跳过不计为 GPU 验收。GPU 检查另行运行。

| 检查 | 当前记录 |
| --- | --- |
| Rust 全部检查 | 18 项通过；release 扩展已重建安装。 |
| 区域 CPU 回归及 worker 重建 | 最终整组 368 项通过、57 项跳过；三处陈旧测试契约已修正，原断言保留。 |
| 原 recirculation CPU 执行 | 通过，保留输出、KV、正常／replay 次数、trace 与清理。 |
| 新旧格式、旧后端体组合、原页发布 | 19 项相关检查通过，包括非法旧 token 复用和纯区域状态操作的拒绝。 |
| 完整 Qwen 数值检查 | 收口快照通过：8 个正常行、6 个 replay 行、14 次独立比较，零 fallback；选中编译候选 8。 |
| EngineCore 跨进程检查 | 收口快照通过并退出：父进程 CUDA 未初始化，14 个候选中含 6 个计算变换候选，选中候选 10；重复请求得到 12 个 replay 行，零 fallback。 |
| typing / import layers / region authority / diff | 最终检查全部通过，源码摘要与验收版本核对一致。 |

最终结果见 [收口证据](evidence/closure-results.json)，可重跑命令列于下方。Qwen 与 EngineCore
检查只证明列出的功能和运行条件，不能替代对齐吞吐、其他模型或完整训练迁移。

源接口对照的 [旧源](evidence/recirculation-legacy-source.json) 与
[新源](evidence/recirculation-data-source.json) 已保存在仓库；数学值表缩减的
[顺序值对照](evidence/order-values-comparison.json) 和
[不变量对照](evidence/invariant-values-comparison.json) 也已归档，不只依赖临时日志。

可重跑的主要命令（仓库根目录）：

```bash
cargo test --manifest-path rust/pllm_core/Cargo.toml --offline
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q test/boundary/test_region_*.py test/boundary/test_import_only_planning.py test/boundary/test_worker_reconstruction.py
.venv/bin/python -m pytest -q test/workloads/test_recirculation.py
.venv/bin/python -m test.gpu.vllm_recirculation_smoke
.venv/bin/python -m test.gpu.enginecore_smoke --recirculation-only
.venv/bin/python tools/check_typing.py
.venv/bin/python tools/check_import_layers.py
.venv/bin/python tools/check_region_authority.py
git diff --check
```

完整收口的源码版本由 [收口源码清单](evidence/closure-source-manifest.json) 记录，包含基准 commit、
工作区文件摘要和依赖版本。本轮没有自动提交或推送。检查是在这些工作区改动上运行，
不是把基准 commit 本身当作本轮交付版本。

## 接下来按什么顺序完成整体功能

| 顺序 | 工作 | 完成条件 |
| --- | --- | --- |
| 1. 完成架构缩减 | 减少作者端不变量 carry 和坐标 Constant/Scalar 往返；确定唯一来源关系及校验职责 | 生产 token 构造器及旧 Order 兼容已退出，继续删除重复转换和校验路径；真实 recirculation 与不同依赖结构仍成立。不能只增加 builder。 |
| 2. 收窄编译后端依赖 | 减少对 Inductor 私有配置、模板与源码检查的依赖，明确执行约定、容量依据和性能样本的区别 | 使用稳定接口或有同等保证的后端契约；不以取消容量保证或关掉原功能完成“精简”。 |
| 3. 完成模型与算法迁移 | 通用模型前端；完整异构层；全局 prefill/chunk × layer 在线更新；剩余训练路径 | 原模型计算保留；后续 chunk 读取正确权重版本；KV、更新次数、RNG、取消和 worker 重建全部有端到端证据。 |
| 4. 补齐优化组合 | 结构变换后的并行调度、跨块合批、缓冲复用及完整算法的联合选择 | 每项重新提取来源、访问、容量与完成约束，通过原语义对照；原能力迁移后不永久保留两套核心。 |
| 5. 完成成本与性能验收 | 整设备内存范围、真实库工作区、冷规划成本和对齐吞吐 | 在相同算法与工作量下报告完成时间、吞吐、延迟、内存和冷启动，收益稳定超过噪声；不能用短 smoke 代替。 |
| 6. 最后完成教学 | 更新原 tutorial/notebook；增加调度前后、wavefront、值来源及正确性示意；补充页表、COW、RAW/WAR、custom kernel 和编译栈中段教学 | Notebook 可运行，图来自实际计划，描述与实现一致，措辞沿用现有教程和 Torch 教程风格。 |

对齐性能检查优先使用 Qwen2.5-0.5B/7B、BF16、batch 8、至少 256 个新 token、
ignore EOS、关闭 prefix caching，并固定算法、采样和更新设置。基线必须执行同一
算法，不能通过少做 replay、训练或状态更新得到“加速”。GPU 竞争与样本波动需记录。

## 本轮不会据此宣称的完成项

尚不宣称架构已全部精简、所有模型已解耦、全局 chunk × layer 已优化、整设备
内存已有完整保证、吞吐已达标，或教程已更新。整体完成需要上表逐项验收，而非
将本轮已通过的部分重新定义成整个目标。

冷规划仍可能持续数分钟，尤其涉及完整模型、跨进程重建和多个结构候选时。
这也是未完成的工程问题；本轮没有以调大超时、缩小原问题域或跳过验证来掩盖它。
