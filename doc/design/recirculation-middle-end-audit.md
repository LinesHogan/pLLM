# Recirculation 中段精简审计

本次按 [精简设计 §7.1](post-implementation-simplification.md#71-架构验证必须产生具体结果)
检查已有算法，不扩充优化目录。以下是开发验证材料，不是新增的产品 IR 或协议。

## 输入契约

源来自现有 `compile_tokenwise_source` 构造器，与
`test/workloads/test_recirculation.py` 使用同一算法：两个 request、三个 layer、
最多八个 packed row，hidden 为 CPU float32 `[1, 4]`；`alpha=0.7`、`beta=None`、
`ramp_tokens=2`，从最后一层回送到第零层。它保留正常执行、条件 replay、KV、trace
及恢复边界，没有替换为理想化二维网格。小 decoder 的执行对照不能代替完整 Qwen 验收。

源程序摘要：`9724b32ec593e5338274aa7c5be62267d74799225e932c8a85a23bdd5034a805`。
源码入口是 `pllm/execution/tokenwise_region.py`，端点签名来自
`pllm/execution/decoder_region.py`。审计在现有构造器首次调用分析前保存实际 Program
并停止构造，没有提供假的分析结果或替代计算体。

| 输入事实 | 来源及需要保留的含义 |
| --- | --- |
| 调用体 | 原 layer、normalize、begin、observe、complete、mix 调用及端口；tap 的加法已有 Torch 图。mix 目前仍是原 callable，不能声称全部计算已交给 Torch。 |
| 参数域 | 每个 request 的 `tokens` 和 `prefix`；`tokens0 + tokens1 <= 8`，各自 `prefix + tokens <= 8`、`0 <= prefix <= 7`。 |
| 实例坐标 | request、该 request 内的 token、layer；packed row 是前面各 request 行数之和加当前 token，position 是 prefix 加 token。 |
| 值来源 | 第一层 hidden 来自 begin；随后 hidden/residual 来自前一层。正常输出在 replay 前取得；replay 输入来自原 tap 和 mix。 |
| 条件 | eligibility 是调用输入；只执行实际成立的 replay 分支。不能把两个分支的值都当作已计算。 |
| 资源与所有者 | `context[row]`、`kv[request, layer, position]`、`weights[layer]`、trace；仍由原 backend/PTTS 提供绑定、恢复、pin 和完成。 |

原执行结构是：

```text
begin → 正常 layer 序列 → normalize → 条件 replay → complete
                                      └ mix → observe → 后续 layer 序列
```

资源访问应从原声明提取，不能靠名称猜测：

| 调用 | 逻辑访问 |
| --- | --- |
| begin | 读取当前 token 在各层的 KV 备份；写入该 row 的准备上下文。 |
| layer | 读取原上下文、该层权重及过去位置的 KV；写入当前 KV 位置。 |
| normalize | 读取上下文及末尾权重 cell。 |
| observe | 读取上下文、写 trace。 |
| complete | 写上下文和 trace；原提交／恢复协议留在 backend。 |

## 最小表示验证

先复用现有分析事实，不再新增一套类型：

| 概念 | 当前可复用的来源 |
| --- | --- |
| calls | `Statement` 的原路径、输入输出端口及原计算体引用。 |
| P、D、T0 | Program 的参数约束、`RegionAnalysis.domain`、`reference_schedule`。 |
| U | `value_sources` 与调用的输入端口；消费者、取值和最后使用均从同一来源关系推导。 |
| R、W | `resource_dependencies.reads/writes`。 |
| boundaries | 原动态控制／不透明效果边界；不得用两个资源名字代替物理独立性证明。 |

已完成的第一处分离是 ISL 选择：它只消费调用域 D 和由上述事实导出的完整依赖，
不再接收或重新解释 Program。源解释独立产生事实；参考执行和优化执行复用这一份
事实。候选仍用完整依赖复核，搜索中的边约简不取代正确性依据。

这还不是整个中段的最小表示。坐标已改为 `Coordinate` 直接声明，tap 的静态
位置已在构造期展开；执行器安装和资源接线仍使用 Program；值 frame 已不再
依赖源程序。后续必须实际替换剩余路径，不能把此次函数
分离或缓存复用当作 §7.1 全部完成。

## 基线与执行对照

坐标与 tap 静态化后（2026-09-13），源语法树有 24 个 Call、19 个 Coordinate、
6 个 Scan、2 个 If，不再有 Constant/Scalar；native 分析语句从 28 个降为 18 个、
值位置从 85 个降为 61 个、诊断清零。新旧源经参考解释器执行对照：调用计数与
名称序列、资源效果事件、返回数值逐字段相等，见
[坐标与 tap 迁移对照](evidence/coordinate-tap-comparison.json)。

真实规模（24 层、64 packed row、tap 在第 22/23 层）的账目方向与小配置不同：
语句 83→84、值 125→147、调度片 60→84，最胖的串行候选调度文本从 25.6KB 涨到
34.6KB。展开的 tap 迭代在入口层产生各自的调度片，分段 Scan 也给依赖增加了
析取。生存期推导因此在第 4 个串行候选上越过 20M 操作配额（make_llm 即失败）。
修复按子结构复用做：存储归属的 provenance 与调度无关、一次推导候选共享，
envelope 比较改在时间空间全集上进行；修复后同一程序 4 个候选全部在配额内
完成，payload 与修复前逐字节一致。小配置的语句数下降（28→18）不能外推到
真实配置，此处以真实配置为准。

迁移前源语法树有 26 个 Call、23 个 Constant、22 个 Scalar、6 个 Scan、10 个 If，
以及 108 个 Order 值声明。Scan 的输入数为 9、12、10，并在两个 request 中重复。
这些是语法声明数量，不是实际执行次数，也不是可直接删除的数量。

旧分析得到 28 个语句模板和 193 个值位置，其中有 64 个 Order、65 个 scalar、
22 个 bank、42 个 tensor。原分析已经折叠了部分结构，不能将这两个统计之差算成
本次精简成果。原 `value_sources` 约 46 KB、别名关系约 22 KB，说明仅让调度函数
接收一个“大 Analysis 对象”仍不足以完成表示精简。

改造前的源、分析快照分别保存在本次工作区的
`/tmp/pllm-simplification-recirculation.json` 和 `/tmp/pllm-simplification-before.json`。
改造后的检查比较了来源、资源冲突、依赖和参考顺序，并通过原 recurrence 执行器
检查输出、KV、正常／replay 次数、trace 与清理。第二种依赖结构采用已有资源循环：
不同 cell 可独立执行，同一 cell 的更新必须有序；两者使用同一调度入口。

源解释与调度选择分离时，前后 15 个分析字段逐字相同。数据依赖、总依赖及候选的文本表示发生变化；ISL
集合比较确认两份依赖关系和三个候选调度映射均等价，且全部候选通过原依赖复核。
差异涉及冗余约束的打印，不能将字符串相等代替关系相等。比较结果保存在
`/tmp/pllm-simplification-comparison.json`，新分析在 `/tmp/pllm-simplification-after.json`。

随后从派生产物中剔除了资源顺序值：同一源程序的值位置从 **193 减至 129**，
语句输入／输出端口引用从 **200 减至 136**，语句模板仍为 28 个。64 个 Order 值
不再获得数学值 ID，也不再进入绑定、返回或生存期表。源中的顺序名称仍由原
校验器检查；资源读写和效果边界继续独立保留。该步骤当时尚未删除作者端接线；
下面单独记录源接口迁移的结果，避免混淆两个阶段。

值编号变化后，按原 `(path, name)` 对应数学值，再用 ISL 比较别名、值来源、
标量和消费者关系；这些关系、资源读写、RAW/WAR/WAW、总依赖及参考顺序均等价。
全部候选仍通过合法性检查。结果在 `/tmp/pllm-order-values-comparison.json`，
新产物在 `/tmp/pllm-order-values-after.json`。运行时读取的 `ValueSite.type` 已收窄
为现有 DataType，frame 和执行器中逐项排除 OrderType 的分支已删除。

进一步将不变的 Scan 输入和 Map 输入直接绑定到外层定义，并复用不变 Scan 的
返回值定义：值表从 129 项再减至 **89 项**。相对于最初 193 项，已删除 64 个顺序
值和 40 个重复的不变量位置，语句模板仍为 28 个。别名关系文本由 22,363 字节减至
8,471 字节，来源关系文本由 45,890 字节减至 25,330 字节。这些是编译产物大小，
不是 GPU 流量或峰值内存的测量。

取值按原定义所在作用域的坐标前缀进行；真正变化的 loop carry 仍保留迭代递推。
对照将各语句输入端口映射到原始生产者，再用 ISL 比较整个参数域：每个调用读取
的来源、存续值的来源、原操作及输出、R/W、依赖和参考顺序均保持一致。
别名图本身有意缩减，因此不要求它与旧别名图逐字相同。
证据在 `/tmp/pllm-invariant-values-comparison.json` 和
`/tmp/pllm-invariant-values-after.json`。零次循环、嵌套循环、跨层值、张量、控制流
以及原资源所有者的执行检查继续使用已有测试。

验证记录：原五项 Rust 依赖／调度检查通过；使用
`/workspace/pllm/.venv/bin/python -m pytest` 运行原 recurrence 和参数化 wavefront
执行检查，2 项通过。release 扩展已重建安装，typing、import layers、region authority
和 diff 检查通过。本轮未新增模型加载或模型验收。

此前完整 Qwen 检查已结束，原结果为 8 个正常行、6 个 replay 行、14 次独立数值
比较、零 fallback。它是本次改造前的存量基线，不证明本次架构精简已经通过。

## 作者端源接口的收口

decoder 与 recirculation 构造器现在直接生成数据端口程序，不再构造、传递或
返回资源 Order。此例源语法树中的 **108 个 Order 声明已全部删除**，三类 Scan
的输入数从 **9/12/10 降至 5/9/7**。Call、常量、标量运算和分支操作保持原样；
资源仍通过原声明、访问关系和参考顺序表达。

新源码与 `without_resource_orders` 对旧源码的检查后投影逐项相同。除源摘要外，
分析得到的 89 个数学值、28 个语句、所有来源／读写／依赖关系以及候选均与迁移前
完全一致。新源与分析在 `/tmp/pllm-data-ports-source.json` 和
`/tmp/pllm-data-ports-analysis.json`。

迁移期间，旧格式先检查原线性 order 规则，旧后端体在组合边界检查后迁移。
只读图导入、vLLM 计算体及条件 replay 迁移完成后，生产构造器不再生成 Order；
现已删除这条兼容路径。旧源与对应派生产物直接拒绝，需从算法构造器重新规划，
包括原本合法的 token 链，也包括重复、丢弃、过期或藏入 bank 的 token。
纯区域继续拒绝状态操作，资源权限及发布仍由原 owner 检查。

完整收口快照通过了 Qwen 的 14 次独立数值比较及 KV/RNG/清理检查，零 fallback；
完整收口结果及剩余工作统一见 [功能与进展收口](implementation-status.md)。

## 替换账目

| 旧机制 | 替代来源及删除条件 | 本次状态 |
| --- | --- | --- |
| 按 `(Program, optimize)` 分别解释源、求别名闭包及 RAW/WAR/WAW | 一条源事实提取路径；ISL 选择只使用 D 和依赖，单独缓存选择结果 | 已替换；不再缓存两份分别派生的完整源分析。 |
| 坐标先写成 Constant/Scalar 再恢复为仿射式 | 在导入边界直接形成坐标关系；须保持原索引范围与溢出拒绝行为 | 已替换为 `Coordinate` 直接声明 `AffineIndex`；Rust 的 Constant int→affine 与 Scalar combine 恢复分支已删除，越界或非仿射操作数硬拒绝。 |
| 连续层号先算 ordinal，再加起始层；单层段仍生成零次 Scan | 原构造器已知完整 layers 列表 | 连续层号合并一次加法；单层段直接使用首层结果；无 tap 时不捕获未用加法图。一般索引往返已随 `Coordinate` 一并替换。 |
| 8 个 tap 分支用逐迭代动态谓词表达构造期已知的层位置 | initial tap 在构造期展开（命中首层则直接生成加法调用，否则 carry 直传）；scan tap 命中迭代是构造期常量，层 Scan 在 tap 层处分段，命中迭代展开为单段直写 | 已替换；recirculation 源 If 从 10 个降为 2 个（仅保留按请求资格的真动态分支），参考执行的调用序列、效果事件与数值逐字段相等。 |
| 生产构造器显式传递与返回 Order | 原资源声明、访问及参考顺序 | decoder/recirculation、只读导入、vLLM 计算体和条件 replay 均已删除 token 接线。 |
| Python/Rust 同时接受旧 Order 源与数据端口；参考解释器维护两套值类型 | 生产源已迁移；旧源统一拒绝并从算法重新生成 | 已删除 OrderType、ValueType、Type::Order、转换助手、Order 解释对象及线性 token 使用次数检查。 |
| 原生结构槽位与来源绑定用可空编号表示顺序值 | 每个绑定都是已有数学数据定义 | 已收窄为普通值编号，删除过滤和混合数据／order 别名分支。 |
| BorrowedExport 仅打包并转发原图导入与字段借用 | 直接使用已有导入、字段匹配和借用函数 | 已删除类及对应转发入口；原 Torch 签名继续保留。 |
| 作者端将坐标、权重和序列 bank 列为 carry，并每轮原样返回 | Scan 输入分为 carry 和固定后缀；body 只返回 carry 与收集值 | DecoderSource、tokenwise、TensorScan 已迁移；六个 recirculation Scan 删除 30 个无用输出及对应返回接线。其他循环与坐标表达仍需继续审查。 |
| 编译器为不变量创建循环参数、返回值和循环别名链 | 直接引用同一个外层定义，取值使用其作用域坐标 | 已删除重复值位置和别名链；此例再减少 40 个值位置。 |
| 编译值表与执行端口中保留 Order，然后由各消费者过滤 | 源校验保留顺序约束；数学值表只含 DataType，资源依赖由原 R/W 与边界推导 | 已删除 Order 值 ID、执行端口及 frame/执行器中的过滤分支。 |
| frame 为找常量沿源路径回查 Program | 整数来自已有仿射访问器；bool/float 由既有 Constant primitive 产生，使用相同的来源与生存期机制 | 已删除 `source_operation`、`_literals` 及运行时 AST 遍历。 |
| frame 重复取得 Program 并绑定尺寸 | 执行器在调用入口验证并快照尺寸；frame 消费该快照和既有 RegionBindings | 已删除 frame 的 Program 参数及重复 `bind_sizes`。数据形状仍在复制前检查。 |
| 编译契约携带 kernel 数量，生产路径解析 wrapper 并要求至少一次调用 | 代码身份和本地编译重放绑定实现；实际分配与原复制约定决定容量；调用数只作测试诊断 | 已删除生产解析及 Python/Rust 字段，保留原融合断言；零 kernel 的 view 通过 CPU/CUDA 数值、独立输出及容量检查。 |
| 并行候选先推导 payload，完整 stream 校验又重复推导；安装再单独校验 payload | 完整 stream 校验已从源重建其嵌套 payload | AER 复用已核对的 payload；Python 并行安装删除单独 payload 校验往返，保留 phase、预算及实际绑定检查。 |
| 每个候选调度从 analyze 重算与调度无关的存储归属（来源闭包、路由、消费者、保留集、出生映射） | provenance 按（源, 导入）推导一次并缓存，候选共享；仅生存期 envelope 按调度特化 | 已替换；24 层配置下 4 个串行候选全部在 20M 配额内完成，派生 payload 与旧表述逐字节一致。envelope 的字典序比较改在时间空间全集上进行、经 schedule 接回实际时间，中间关系从约 7000 片降至约 100 片。 |
| 其他安装路径反复编码完整源并重新核验同一派生产物 | 区分本地编译产物和真实传输边界，确定唯一安装检查点 | 仍需继续收敛；未增加“已校验”开关跳过动态权限。 |
| 共享目录先完整生成窗口/分块结果，再交错截取；各组重复取得源分析 | 同一份源分析与按需候选迭代 | 已删除共享目录中的两个提前生成结果列表；实际候选仍逐项完成合法性和完整域等价检查。 |
| 函数体先追加外层声明检查一遍，再按原 imports 检查一遍；每次内联继续重查 | 原源 verifier 同时核对外层签名，内部内联只重命名变量 | 已删除重复 body 遍历与 native 的整 Program 复制；公开入口、原 imports 作用域和组合后检查保留。 |
| Python 与 Rust 分别维护 SSA、操作/区域签名、参数和资源语义规则 | Python 入口与 native 规划共用原 Rust verifier | Python verify 从 209 行缩为 18 行；通用 codec 只有一份，表示类型和动态权限仍在各自边界检查。 |
| Python 重建 outlining 结果，再将原源和完整结果交给 Rust 重做 | 直接使用已有 native 提取结果，Torch 图仍从原调用重建 | 已删除 Python 提取器和递归区域替换器；外来工件的验证入口保留。 |
| 串行策略只为规范化参考顺序而降低完整 bindings | native 源分析出口提供相同规范形式 | 策略检查不再生成取值访问器；执行绑定按原路径生成并核验。 |

旧格式退出后，同一份归档 recirculation 数据源的分析与迁移前逐字段一致，包含
来源、RAW/WAR/WAW、效果依赖、参考顺序和全部候选。退出兼容没有给该算法
增加或删除调用；此前保留的完整模型基线不作为此次模型复测。

本阶段验收依据是以上实际替换项和不变量。候选数量、测试数量及新增接口数量
不作为架构精简成果。

常量路径的执行对照覆盖 bool/int/float 类型、负零、零次及非零次 Map、动态控制流
和原条件 replay。布尔／浮点常量现在是现有计划中的 primitive，调度语句编号可能
相应变化；实际效果调用的顺序和次数仍由原依赖约束保持。未新增常量表或另一个
常量执行器。原模型效果检查也改为读取已生成的 action 计划，不再使用已删除的
源码回查助手。参数测试继续拒绝复制前的错误形状，并确认外部修改原 sizes 字典
不会改变正在执行的调用。
