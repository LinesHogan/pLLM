# 区域程序：语义内核与迁移约定

本文包含初版规格和分阶段记录。当前功能状态以
[功能与进展收口](implementation-status.md) 为准；下文初版的 order 示例已退出支持，
仅保留为设计演进记录，不能作为当前可执行接口使用。

## 当前源接口

所有生产构造器均使用数据端口形式。资源通过 Program 的资源声明、函数的
`resources/accesses` 以及 Snapshot/Publish 的目标资源表达，不作为输入或返回值传递。

| 操作 | 当前数据接口 |
| --- | --- |
| Call | 函数的数据输入 → 数据输出；效果来自原声明。 |
| Snapshot | 资源坐标 → 原版本的 payload。 |
| Publish | 新 payload、资源坐标 → 无数据结果；通过原 owner 发布。 |
| Scan | 输入前段是 carry，末尾 `invariants` 个是固定参数；只返回 carry 和收集结果。 |
| If / While | 显式数据输入输出；谓词约束与实际执行次数保持原规则。 |

旧 Order 源在 Python 解码和 Rust 规划边界均被拒绝，旧选择和派生产物须从算法
构造器重新生成；已删除兼容转换助手。资源声明、访问、纯区域限制和运行时权限
仍须检查。源端口与数学值表统一使用 DataType，没有 OrderType 或 ValueType。

当前源语义规则只有 Rust 一份实现；Python `verify` 通过共享 codec 调用同一
校验器。Python 保留表示类型与编码检查，原运行时保留动态绑定和所有权检查。
下文早期“双 verifier”描述属于演进记录，不代表当前还维护两套 SSA/资源规则。

Scan 的 `invariants` 是输入分组的语义数量，默认为 0。若输入为 `carry + fixed`，
body 的参数为 `(index, *carry, *fixed)`，返回值为 `next_carry + collected`；
Scan 返回最终 carry，以及每个 collected 的 bank。固定参数每轮都读取同一份
输入值，不需要通过 body 返回。它不声明外部资源不会变化，资源 R/W 和原 owner
规则继续独立检查。旧 all-carry 源按默认值读取，派生计划需要重新生成。

## 初版与演进记录

这是 [计算模型设计](computation-model.md) 的实现规格。2026-09-11 已实现
Python 契约与 verifier、闭合编码、Rust 独立 verifier、串行参考解释器及
export 导入。现已接通符号依赖分析、ISL 候选、固定 schedule 的循环指令、
符号取值与最后使用位置、真实 export 张量执行，以及原调用 owner 内的 PTTS
snapshot / publish。In-Place-TTT 的生产绑定已把 chunk 递推导入 scan；外层
任务和模型执行仍待统一迁移。该 chunk 区域已接入冷阶段的预算约束与自动
计划选择，但还不能据此声称真实 prefill 的全局调度已获得自动优化。

## 表示与类型

一个 `Program` 包含入口 `Region`、导入的函数签名与逻辑资源声明。资源声明是
封闭的 `Resource | ExternalResource` 联合，wire 分别标为 `state` / `external`。
首版每个
Region 只有一个 Block；参数定义输入，操作定义结果，`yields` 定义输出。
标识符只在当前块内有效，子区域通过参数接收外部值。区域外不能引用内部结果。
这是一份结构化参考程序；每个循环体只存一次，不按迭代数复制代码。

数据类型包括 host 标量、张量和不可变 `Bank`。标量 `int` / `index` 使用
有检查的有符号 64 位语义，溢出报错；`float` 使用 host 双精度语义；张量
运算保持导入算子的 dtype 语义。常量属性必须可用有限 JSON 数字表示。
循环次数和静态尺寸非负，并须在 index 范围内。bank 长度及 scan/map 次数
支持有界形状参数与入口 guard；张量的具体维度仍按 shape 特化。

`Bank(T, N)` 是 N 个 T 值组成的不可变序列，可以嵌套。`select` 读取一个
元素，`update` 返回新 bank；旧 bank 的每个元素仍可读取。这里不分配页号，
也不隐含原地覆盖。张量实际存储与别名问题留给后续 lowering。

## 封闭操作签名

下表中 `A` 是输入类型序列，`D` 是其中的数据类型序列，`Q` 是资源顺序类型。
操作按独立变体编码，未知操作或多余字段拒绝导入。

| 操作 | 参数 → 结果 | 子区域规则 |
| --- | --- | --- |
| constant | `() → scalar` | 常量与结果类型一致 |
| scalar | `(T,T) → T/bool` | add/sub/mul/less/equal；bool 仅支持 equal |
| call | `(data inputs, resource orders) → (data outputs, new orders)` | 必须匹配已导入的函数签名 |
| select | `(Bank(T,N), index) → T` | 执行时检查索引边界 |
| update | `(Bank(T,N), index, T) → Bank(T,N)` | 更新后的值不改变旧 bank |
| snapshot | `(resource indices, Q) → (payload fields...,Q)` | 读取该逻辑资源单元的当前快照 |
| publish | `(payload fields..., resource indices, Q) → Q` | 发布该单元的新内容 |
| scan | `A → (A, banks of R)` | body `(index,A) → (A,R)`，依次执行 count 步；R 只能是数据 |
| map | `D → banks of results` | body `(index,D) → data results`，各实例独立 |
| if | `(bool,A) → results` | 两个分支均为 `A → results`，只执行选中的分支 |
| while | `A → A` | condition `D → bool`；body `A → A`；最多 limit 步 |

scan 从输入个数推导 carry 的长度，剩余 body 结果按迭代顺序分别收集，不需要
额外声明 carry 个数。收集的是该轮产生的确切值，可以同时返回更新前后的
权重；不能把资源 order 收集进 bank。`scan(count=0)` 原样返回 carry，并为
每个收集列返回空 bank。`map(count=0)` 返回对应类型的空 bank。
有界 while 在每步之前判断条件；条件为假或步数达到上限时返回当前 carry，
不额外执行一次 body。predicate 只接收不可变数据，不能在判断过程中修改 KV
或发布状态。body 可以包含显式效果。

首版 map 要求无资源效果；有状态的并行程序从带完整效果的参考循环分析得到，
不能通过把它声明为 map 跳过独立性证明。无效果 call 也不代表可以推测执行：
未选择分支中的越界访问或可能报错的调用必须保持不执行。

## 原子资源与可索引资源族

`Resource(name, payload, domain)` 的 domain 指定独立原子单元的索引域。
例如 `Resource("layers", (TensorType(...),), (L,))` 表示 L 个可独立发布的单元。
snapshot/publish 的索引是 SSA 操作数，循环变量不会被藏进 callable。

`Resource("bank", (BankType(T,L),), ())` 则表示整组 bank 的原子发布。它和上面
的逐层资源族即使拥有相同数值内容，也是不同语义；编译器不得自行互换。
两个请求是否访问同一路由、两个索引是否别名，由实际绑定事实证明，不能由
不同名称或物理页槽编号推断独立。

`payload` 是非空的数据类型元组，每个单元可以包含不同 shape、dtype 的多个
字段。snapshot 一次返回全部字段和新的 order；publish 一次消费全部字段、
索引和 order。比如 `(参数张量, 动量张量, 标量 int64 step 张量)` 是一个原子
payload，各字段通过普通多结果 SSA 值进入数学程序，不需要把整份状态装进
同质 bank，也不需要按字段拆分发布。Bank 仍只表示同质的逻辑序列。

每个资源族有自己的 OrderType。入口接收、返回每个资源族的一条顺序链；
每个顺序值恰好消费一次，不能丢弃、复制或装入 bank。循环显式携带顺序值，
分支通过相同结果类型合并。普通不可变快照可以有多个消费者。

顺序链记录参考程序的效果次序。后续依赖分析必须查看具体资源索引与访问种类：
若已证明两个单元独立，可以重排操作并重新连接效果链；同一个单元的版本读取
与发布仍受约束。不应把整条资源族 token 链无条件变成 ISL 的串行数据依赖。
没有细化访问声明的 call 保守视为读写整个声明资源族。显式索引区间可以
细化读写关系；外部资源仍需保留原后端的别名与完成边界。

`ExternalResource(name)` 表示由后端管理的效果，没有伪造的 tensor payload 或
页索引。模型 KV、后端输出缓冲和 trace 通过带签名的调用访问；Python 与 Rust
verifier 都拒绝对它做 snapshot/publish。当前尚未导入外部 owner 的别名证明，
所以涉及外部资源的调用作为不可跨越的重排边界；不同名称不等于独立资源。

这些编译语义不授予真实提交权限。PTTS 仍须校验精确版本、写入资格、发布和
完成；OrderType 不等于 PageWrite，也不等于 GPU event。

## 张量图导入边界

`ExportedCell.capture` 保留函数化后的 export 图和明确的输入输出签名。
原地修改必须有可对应的新状态输出，执行纯 graph module，避免使用会恢复
输入修改行为的 `ExportedProgram.module()`。循环结构保留在外层区域中。

当前导入子集要求扁平张量参数和结果、显式张量状态、函数式 ATen 操作；隐式
lifted 状态、未声明效果及 RNG 拒绝导入。捕获示例的 storage 必须彼此独立，
不能通过分别 clone 有别名的示例掩盖原 Python 的修改语义。直接作者调用
仍检查输入别名；Region 使用 `invoke` 执行已经函数化的不可变值，允许相同
旧值或共享 storage 的 view 出现在多个端口。所有入口仍检查 shape、stride、
dtype 和 device。新状态通过显式输出返回，共享旧值不表示可变参数之间的
写穿透。实际调用的别名由原存储来源关系组合，不把独立的导入示例当成实际
独立分配。导入是执行 Python 的冷阶段，不是任意 Python 的纯度证明或沙箱；
外部服务与模型效果通过有资源签名的原语接入。

`compiled` 在选定计算块之后构造编译候选；capture 本身不调用 torch.compile。
数值策略与 warning 归最终候选选择负责。通用 `torch.compile` wrapper 仍没有
相应的成本导入，不能直接参与容量选择。下面的 Inductor 适配器提供一条带
图帐容量依据的受限路径（图上张量 span 加命名盈余，不读代码生成内部分配），
已接入纯 export 区域选择；完整模型生产接入仍待完成。

## 校验与可执行对照

- [Python 契约与 verifier](../../pllm/contracts/region_program.py) 检查定义唯一性、
  作用域、封闭签名、carry、分支和线性效果链。
- [Rust verifier](../../rust/pllm_core/src/planning/region_program.rs) 从原始闭合
  程序独立重做这些检查，不读取 Python 的“已验证”标记。
- [参考解释器](../../pllm/execution/region_reference.py) 用不可变值和串行控制
  流执行算法；复制张量以暴露快照语义。其资源字典只是语义对照，不是第二套
  PTTS，也不进入 worker 的 fallback。参考函数适配器显式接收与返回张量状态
  内容，因此后续 snapshot 可以观察其效果；opaque 外部资源不伪造状态内容，
  由独立 oracle 的回调实现效果，并记录其 order。真实原语使用原 owner 的 ABI。
- [语义测试](../../test/boundary/test_region_program.py) 覆盖 chunk×layer、
  两状态训练与采样、条件重放、路由续接、只读观测、资源族及错误提交链。
- [export 测试](../../test/boundary/test_exported_region.py) 在真实 PyTorch 图
  上检查函数化和多步状态结果；[编解码测试](../../test/boundary/test_region_codec.py)
  覆盖递归结构、标量类型及绕过 Python verifier 的伪造原生输入。

## 从来源关系到结构化 schedule

[原生依赖分析](../../rust/pllm_core/src/planning/region_dependence.rs) 给每个
SSA 定义分配稳定的来源位置。一个 bank 值的坐标包括所在循环的迭代坐标和
逻辑元素索引；张量在本层仍作为完整的数学值，内部算子留在 export 图中。

select、update、scan 的入口/回边/结果以及 map 的收集规则生成值来源关系。
例如 update 的新 bank 在选中的索引取新值，其余索引仍取原 bank 的值。
编译器用 ISL 关系合成与传递闭包追踪这些来源，再连接计算语句的定义与使用。
这保留了旧版本读，不需要把 bank 的每次修改都当成对同一物理缓冲的覆盖。

闭包可能只有保守近似。产物显式记录 `exact_provenance`；精确时进一步检查
每个值元素恰好有一个来源，并覆盖完整值域。非精确来源不能直接拿去生成
张量绑定；保守依赖必须仍与参考顺序相容，否则拒绝当前分析。

资源依赖来自逻辑访问关系与参考顺序中的冲突，包括读写、写读和写写。未知
资源索引保守覆盖整个资源族，并保留不可跨越的边界。动态 if/while 保持为
包含原区域的语句边界；未证明不会溢出的 host 整数计算也不能越过可观察效果。
这些都是符号集合，不生成逐 token 的 DAG。

同一份 chunk-major 源程序目前会得到参考候选、偏局部性候选和偏并行候选。
3×3 网格的实际 ISL 结果包括 `[c,l]` 与 `[c+l,c]`。依赖由源程序推导，
作者没有填写距离向量或 schedule 字符串。百万 chunk × 16 层的测试仍只有
一个数学语句模板，值定义和指令树的数量不随实例数增长。

固定候选的复核重新分析原程序，检查完整域、时间单值性、共同时间空间和
所有依赖向前，不信任候选自带的依赖列表，也不重新做优化搜索。
[循环 lowering](../../rust/pllm_core/src/planning/region_ast.rs) 调用 ISL AST
接口，将结果转成封闭的 loop / guard / sequence / invoke 和整数表达式。
没有生成 C/Python 再 eval 的步骤。其签名绑定源程序、固定 schedule、来源
语句表、循环体及 lowering 版本；复核会重建并比较这些指令。

[机械游标](../../pllm/execution/region_schedule.py) 冷绑定时复核一次，随后
逐次产生语句身份和坐标，内存占用随循环嵌套深度增长。它不决定数学实现或
并行策略，也不持有页、发布资格或完成对象。`preview_region_schedule` 是
另一个有显式实例预算的诊断接口；它的展开上限不是生产指令格式的限制。

[调度测试](../../test/boundary/test_region_schedule.py) 对 30 种网格形状检查
每次 cell 读取的 hidden 和参数、全部中间 hidden 及最终状态；还覆盖旧 bank
读取、丢弃更新、末层反馈、逐单元/整组发布以及不安全重排。
[指令测试](../../test/boundary/test_schedule_instructions.py) 将游标与 ISL
逐点求值对照，覆盖尾 tile、负时间、非单位步长和指令篡改。

最早的 `SemanticTask.iteration` 接线与手工距离实验模块均已移除。现在的
测试直接使用新区域源程序；这些结果尚不替代 CUDA 并行和模型证据。

## 运行时取值与原 PTTS 的接线

[取值 lowering](../../rust/pllm_core/src/planning/region_bindings.rs) 从精确来源
关系生成闭合的分段索引表达式。每个 case 带源值 ID、条件和源坐标；条件在
参数全集下生成，避免把分支条件化简掉。运行时只计算整数表达式，不解析
ISL 文本，不查询 ISL，也不提前展开全部实例。入口参数和返回值的位置同样
来自原程序，产物复核会重新推导来源及生存期，不能只修改签名后放行。

值消费者与选定 schedule 合成后，ISL 的 lexmax 给出每个叶值的最后使用时间。
程序返回值引用到的叶子保留到返回。相同时间的独立语句可能以不同顺序发出，
因此执行帧在进入更晚时间时才释放这些引用，不能在第一个同时间消费者之后
提前释放。[值帧](../../pllm/execution/region_frame.py) 只存活跃的数学叶值，
不会为每个 bank 更新复制一份完整的 bank，也不维护资源的“最新版本”字典。
逻辑死亡释放的是 Torch 引用；它不是 PTTS 页释放或 CUDA 已完成的证明。

[区域执行器](../../pllm/execution/region_execution.py) 已能运行真实导入的
函数式 tensor cell、显式 native 原语、host scalar、动态 bank 选择/更新，以及 snapshot / publish。
常规路径在调用指定的一条 stream 上顺序发出调度后的语句。
`RegionStreamExecution` 另可消费下述已验证的并发阶段计划；生产模型提案与
选择已接入这条路径。native call 通过原 PTTS binding 访问声明的状态，外部
端点绑定到条件 replay 的原模型 owner。数学变换后的 cell 必须通过下述 AER
证据重放绑定，不能直接装入只接受源导入的执行器，也不会转入参考解释器。

[并发 lowering](../../rust/pllm_core/src/planning/region_streams.rs) 生成独立的
`RegionStreamPlan`，不往源操作上增加 stream、wavefront 或算法名称。
`explore_region_streams` 在共享 ISL 目录中寻找阶段划分：把某个调度坐标按可借用
的 lane 数分组，保留其他坐标，再针对完整源依赖重新证明合法性。所有 RAW、
WAR、WAW、值依赖和效果依赖都必须严格跨越阶段边界。完整关系上还必须存在
同阶段、不同 lane 的实例对，只有一条 stream 实际可用的划分不会进入目录。
纯 export、已编译 tensor 导入及带完整执行契约的 native kernel 可按原坐标
的余数放置。控制块递归检查所有分支和循环体；含普通 native 回调、PTTS
状态操作或不透明外部资源的块继续留在 lane 0。native kernel 的可并发资源
必须是具有显式索引访问的借用资源，物理别名证明仍来自原调用。

阶段 schedule 重新进入原来的存储、最后使用和 retention 分析；并发暂存采用
每条 lane 的调用暂存上界之和。snapshot 的独立张量副本现在有自己的 SSA
存储根，snapshot / publish 的显式拷贝有相应的逻辑读写流量。发布仍复制进原
arena 的 write token，其 COW 页、pin、receipt 和 reclaim 没有转移到值帧。
这些数字仍是已建模载荷的上界，不能替代 allocator、库工作区和原 PTTS 预算。

[执行绑定](../../pllm/execution/region_streams.py) 接受调用方提供的 owner 与辅助
streams，不创建 stream 池或另一套任务调度器。lane 0 必须等于原 PTTS pin 的
stream。入口复制通过事件交给辅助 streams。`RegionStreamPlan` version 2
把流水深度、等待关系和宿主屏障列为物理推导结果；深度为一时仍逐阶段等待，
深度大于一时允许多个阶段尚未完成。自动选择会同时考察深度一与二，显式
lowering 可使用一至八的深度，最终仍由原 AER 按内存准入和完成耗时选择。

编译器对每个消费者、每条生产者 lane，从完整 RAW/WAR/WAW 和值/效果依赖
中取最后一个所需阶段，生成带参数的整数访问表。由于同一 lane 按阶段顺序
发出，一个阶段尾事件覆盖它在该阶段及之前的全部工作，不依赖 ISL 如何排列
同时间的独立语句。运行时只解释这些表：不同 lane 之间插入 CUDA event wait，
同一 lane 使用其自身顺序。没有依赖的后续阶段可以先在另一条 lane 上完成。

最多同时保留 `pipeline_depth` 个未完成阶段。进入新阶段前，窗口满时先等
最旧阶段完成；值帧仍按逻辑最后使用释放槽位，但该阶段的 Tensor 引用由
完成窗口继续保留，直到事件完成。这里不复制 Tensor 数据，也不持有第二套页
或版本表。设既有并发阶段载荷上界为 P，未完成阶段的附加预留为 `(depth-1)*P`，
物理总上界为 `depth*P`，包含保守的暂存与后端 retention 费用。它仍不是完整
allocator/库预算；不能省略这些额外引用，再把 CPU 已释放槽位当作 GPU 可回收。

普通 native 回调、PTTS 状态操作和后端 retention 获取/释放端点留在 owner，
调用前排空未完成工作。嵌套控制块递归继承这个约束，宿主观察与页发布不会
只依赖尚未完成的 CUDA wait。已导入 native kernel 的设备读写则必须遵守
发出 stream 上已有的等待关系。返回张量为 owner 登记存储使用，正常返回与
异常退出都先把已发工作汇合到原 owner、等待完成，再关闭值帧。

小型 CPU/CUDA 检查覆盖物理计划重建与篡改拒绝、自动 wavefront 并行、严格
数值一致、异常后的事件完成、原 PTTS 发布次数和 owner 匹配。并行计划已进入
下述 `RegionCandidate` 联合实测选择和数学/编译重放。vLLM tokenwise 适配器
已提供原调用的辅助 stream 借用与复用，任务仅在物理计划需要时请求，关闭
适配器时等待并释放这些句柄。模型候选提案现在携带并行计划；Qwen 层尚未
提供下述完整 native kernel 执行契约，因此其层调用仍在 owner 上执行；
这项接线没有通过 Qwen 吞吐验收。

流水检查还核对二维网格每个实例的生产者 lane/阶段前沿、未完成阶段的附加
预算及篡改拒绝。真实 CUDA 用例在预热后确认后续独立阶段可先于前一阶段完成，
同时核对窗口深度上限、异常排空、wavefront 的精确旧值和原页发布次数。
矩阵重结合加编译 kernel 的参数化重放覆盖深度二以及 0、3、5 步；模型 worker
提案检查也使用深度二，普通模型回调仍经过 owner 屏障。人为延迟只用于观察
跨阶段重叠，不是性能测量；Qwen 的 kernel 并发契约和吞吐验收仍待完成。

动态 if / while 已由[原生控制 lowering](../../rust/pllm_core/src/planning/region_control.rs)
转换为嵌套槽位指令，作为 `LoweredSchedule` version 2 的 action 一起签名。
控制块具有明确的参数、结果、每步输入/输出槽和最后使用后的释放列表。块内
的 scan / map 仍是结构化循环，不按 trip count 展开。循环与条件本身采用
原参考顺序；后续可进一步优化内部仿射区域。

if 的外部可移动性现在由两个分支的源码推导。分析保留调用处已知的仿射值，
沿用普通操作对索引、整数检查和资源效果的规则；两个分支都可证明时，外层
action 可以越过无依赖的语句，内部仍只执行选定分支。可能越界、可能溢出、
未授权外部效果或含动态 while 的分支继续保留不透明边界。资源效果取两个
分支索引读写的保守并集：内部 scan/map 的坐标按其实际域投影到外层 action，
嵌套 if 沿用同一投影。这样可以移动不相交单元上的读写，同时保留实际冲突
和原发布次数；不把某个分支的运行结果作为静态假设。未知访问继续扩大范围，
原外部 owner 的别名授权规则仍然适用。

[同一个 action 执行器](../../pllm/execution/region_control.py) 服务于外层仿射
语句和动态区域。if 只执行选定分支；while 先求值条件、最多运行 limit 次；
limit 为零时不求值条件。局部槽位复用不了资源 order：order 经 verifier 校验
并决定效果顺序后不进入数据槽，snapshot/publish 仍调用同一个 PTTS binding。
每层控制块和原语都检查原调用的取消请求。未使用但可能报错的运算仍执行，
不能通过局部值释放删除它。

[控制流测试](../../test/boundary/test_region_control.py) 包括未选分支中的越界、
未使用结果的整数溢出、嵌套 scan/map、动态边界内的 PTTS 连续发布、取消后
不再提交，以及修改分支/循环上限/释放指令后的独立复核。用真实 tensor 的弱
引用检查多次迭代不持有全部旧 carry；`RegionExecutionResult.peak_live_values`
当前只统计外层调度值帧，不作为嵌套区域或 GPU 实际内存峰值的预算依据。

## Native 导入与张量条件

`BorrowedResource` 表示由原后端管理、且绑定时必须提交别名依据的索引存储。
它与未知外部效果分开：调用必须显式声明读写区间，两个 verifier 都拒绝缺失
声明；可证明的区间参与 RAW/WAR/WAW 推导。运行时只有匹配的原调用证明
齐备时才接受该资源，单独声明资源名不能获得执行资格。

首个绑定实现是 `PagedResourceView`。它投影原 `PagedAccessPlan` 的绝对层号、
请求身份和连续 token 段，允许读取已核验前缀，只允许写本批当前 token 段。
每次外部调用都先核对其访问区间。`PagedResourceBinding` 绑定原调用 owner；
多个资源名的证明合并检查物理冲突，还必须与原 arena 的全部分配隔离，包括
尚未使用的 COW 页和对齐空间。证明只持有不可变访问元数据，不创建 KV 副本、
版本目录或恢复状态，owner 失效后不能继续执行。

`ReadonlyTensorView` 将同一种借用资源绑定到原 owner 的张量 family，允许
不同只读单元共享存储，但拒绝它们与 KV 写区或 arena 分配重叠。调用前校验
访问范围、只读权限和原张量的布局、存储身份及可用的版本计数。inference
tensor 没有版本计数，其内容稳定性仍由原后端的调用生命周期保证；这些检查
不宣称能够检测任意外部 CUDA 写入。绑定保留原张量引用，不复制权重。

tokenwise 的 signed attachment 禁止 overlay 和 activation patch，因此它的
源程序可以单独声明基础权重的只读 family：每层一个单元，最终 norm 一个
单元。decoder ABI `first_layer@6` / `next_layer@6` 接续这个显式导入的顺序值，
每次层调用只读相应层，norm 只读最后一格。vLLM 从原模块参数构建证明；
条件 replay 的 overlay 路径继续保留未知 model 效果，没有套用只读假设。

tokenwise 的行上下文也使用同一借用资源契约。宿主 view 绑定原后端 registry
的身份和本次原始 rows；准备与提交写该行，decoder、norm 和观察读取该行。
读取前检查该行已经准备。trace view 绑定原 `ExecutionTrace` 对象，观察和
提交写同一个原子效果域；两个资源名不能借同一个宿主 namespace 来伪装独立。
宿主适配器必须单独声明所有张量访问；这个 namespace 只证明宿主效果分区，
不代替 KV、权重和 arena 的物理别名检查，也不允许任意宿主代码宣称独立。

这条路径已能在原缓存上执行合法重排。生产 DecoderSource 与 tokenwise
现在使用借用 KV 声明，原模型端点传入同一调用持有的页表证明。tokenwise
复用本次准备的全层证明，conditional replay 复用原层段的证明并保留绝对
层号；两条路径都核对原 owner，清理时释放证明引用。条件层段还在普通前向
首层执行前检查 arena 分配隔离。模型上下文、提交及在途容量的约束仍须保留，
物理别名依据本身不代表完整的模型调度资格。

[NativePrimitive](../../pllm/execution/region_native.py) 是编译器和后端适配器的
显式导入边界，与 ExportedCell 共用 call、控制流和调度指令。它不是给任意
用户 Python 标注“纯函数”的接口；适配器的实现和效果声明属于可信编译边界，
普通作者方法仍须走原有源码检查或明确提供匹配的 lowering。

绑定核验完整 Function 签名。调用前检查数据类型并复制数学输入，原语可在私有
副本上调用原地 kernel；返回值也会复制，避免临时 view 或后端复用缓冲区改变
已经产生的 SSA 值。这是当前正确性基线，复制消除和准确工作区计价仍需后续
bufferization 与生产规划支持，不能直接套用旧 kernel 的零额外复制成本。

带效果的原语接收临时 `PrimitiveResources` 视图，只能 snapshot/publish 其签名
声明的资源。视图委托给同一个 RegionResources，不持有另一份状态、页写入
权限或 completion owner；调用返回、抛错或取消后即失效。原语中的发布仍产生
原 PTTS receipt。提交后的错误保留已经提交的版本，由原调用结算，不承诺跨
多个发布的通用回滚。它也能调用当前 Function 对应的外部端点；端点在原调用
内以完整签名绑定，缺失、重复或不同声明在执行前拒绝。端点借用原后端对象，
不另建模型状态仓库。模型 KV 的恢复与完成协议仍由原调用和后端实现负责。

Function 的 `accesses` 可声明 `ResourceAccess(resource, mode, ranges)`。
每一维是 `[lower, upper)`，端点由常量和 `IndexTerm(argument, coefficient)`
组成的 AffineIndex 表达，只能引用该函数的 INDEX 数据参数。读和写分别
声明，同一资源的多条声明取并集；未细化的资源沿用整族读写语义。Python 与
Rust 独立检查资源归属、维数、参数类型、重复项和整数范围，不解析可执行
表达式字符串。ExternalResource 也能声明逻辑域，但域本身不证明物理独立。

依赖分析把这些区间与调用点的仿射实参组合，检查整个迭代域上的上下界及
整数运算范围，再生成 ISL 读写关系。分析产物的 `resource_dependencies`
分别保留 reads、writes、RAW、WAR、WAW，后续调度仍检查完整依赖集合。
无法证明范围时保留已证明的维度，其余维度覆盖完整范围，并保留不透明边界；外部调用即使区间已知，也仍保留
别名边界，不能凭一条声明就获得可重排资格。

运行时在 native 调用前绑定并核验实际区间；PrimitiveResources 的每次
snapshot/publish 都检查资源、读写模式和索引是否落在声明并集中。越界、
算术溢出或通过只读权限写入会被拒绝，外部端点也不能替换原来的索引访问。
这些检查复用原 RegionResources 和 PTTS，不增加状态目录或页所有者。

[索引访问测试](../../test/boundary/test_region_indexed_access.py) 把独立单元
的写入移过另一个单元的读取，在原 PTTS 上核对旧值、前缀读取、最终版本及
发布次数；反转冲突写读会被拒绝。另检查 native 越过声明范围、只读写入、
调用前越界拒绝，以及两个 verifier 对非法索引参数的拒绝。模型 KV 的物理
区间证明和逐层逻辑声明均已接线；可重排资格仍须同时解决下述恢复和参数化边界。

`tensor_item` 将 rank-zero 张量显式读为相应的 host bool/int/float。条件区域
可以先计算张量 predicate，再调用它驱动 if/while；CUDA 的 host 同步也发生在
这个明确的调用点。测试覆盖零次循环、提前退出、步数上限和实际 CUDA 张量。

[AdamW 导入](../../pllm/execution/region_adamw.py) 接收参数、梯度缓冲、step、
两个动量及本次梯度，返回五个新状态字段。它调用原 apply_adamw，保留 CPU 的
PyTorch 路径及 CUDA 的 capturable foreach 路径，未改写偏差修正公式或算子顺序。
TensorScan 现在同时接受 export 与纯张量 native 导入。跨五轮的 CPU/CUDA
float16、bfloat16、float32、float64 测试逐位核对所有字段；这证明导入路径，
尚不代表旧生产 optimizer 回调已整体迁移。

[原语测试](../../test/boundary/test_region_native.py) 同时覆盖源签名不匹配、
旧数学值不被修改、后端缓冲复用、临时资源视图过期、未声明资源、非法索引、
输出类型错误和提交后取消，真实页测试继续使用原 stream、pin 和 receipts。

## PTTS 资源绑定

[资源绑定](../../pllm/execution/region_resources.py) 把逻辑资源单元对应到原调用
pin 中的 lineage 和字段。完整域、shape、dtype、device 与物理别名在绑定时
检查。当前不同逻辑单元必须对应不同 lineage；同一单元的 payload 可以由不同 shape
和 dtype 的张量、或同质张量 bank 组成。一次 publish 的所有叶字段使用同一个
原始 PageWrite 和 receipt，不会按字段拆成独立提交。一般的异质 tuple 值类型
仍未引入；资源多输入/多结果已经能表达异质原子状态。

[异质 payload 测试](../../test/boundary/test_region_payload.py) 在 CPU/CUDA 上
一次更新矩阵、两个不同 shape 的向量和 int64 step；同时检查缺失/错序字段的
签名拒绝，以及写到第三个字段时报错不会让前两个字段部分可见。

快照先在原 stream 上复制出自有张量，避免 export 返回的 input view 在页 pin
释放后指向已复用的槽位。发布通过原 arena 的 prepare / write / publish 完成，
更新来源必须是原 pin 或本调用原始 publication receipt 证明的精确版本。
若别的写入推进了 lineage，arena 会拒绝以旧来源 prepare，而不会把旧数学
结果套到它选择的最新版本上。当前先采用可验证的复制路径；零复制需要额外
的别名、预算和完成关系证明。

`StateTaskContext` 中单个 write / publication / published pin 的存储已改为
原始 capability 集合；旧的单写接口作为集合上的兼容视图保留。没有新增另一套
lease 或提交 owner。原 `StateExecution` 仍获取 request-directory lease，记录
发布，处理取消，释放全部原始 pin，结算全部 receipt。提交前失败会 abort
私有写入；提交后失败保留原始 receipt，并锁住计算重试，后续只能通过原调用
的取消/结算流程释放。已结算的 receipt 不重复提交或重算。

[张量测试](../../test/boundary/test_region_tensor.py) 检查 serial、ISL 和尾 tile
下每次 cell 的真实输入、最终张量及输入不可变性，也检查长网格的活跃值上界。
[PTTS 测试](../../test/boundary/test_region_pages.py) 把区域执行放在原调用上下文
内，覆盖多次版本发布、COW、物理别名拒绝、重叠 lease、提交前后故障、外部
版本冲突，以及 input view 在页回收复用后的有效性。CUDA 测试使用原 stream
与原 completion receipts。这仍是边界接线证据；旧作者接口与 vLLM 模型层段
尚未统一 lower 到该执行器，不能据此宣布迁移或吞吐验收完成。

## 多资源与原状态目录

`MathBinding` 现在可绑定多个逻辑状态。原 `RequestStateDirectory` 使用
`(resource, projected request key)` 标识 lineage；instance 资源的投影为空。
资源名相同的 route 投影仍共享状态，资源名不同则拥有独立 lineage。
版本继续由 arena 管理，目录没有第二份 latest 表。

`RequestStateSelection` 为每个资源提供独立的 key、ref 和行映射，同时保留供
原 invocation pin 使用的扁平 refs。多资源读取必须指定资源；旧单状态视图
遇到多资源会明确拒绝。初始化函数也按资源注册：一次 admission 中新建的
异构 roots 使用同一张原始 write、一次 publication；加入已有 route 不重新
初始化。实例资源先用空 identities 初始化，后续请求 admission 同时持有请求
资源并取得实例资源的原 inference pin。训练更新不会把该 pin 换成新版本。

lease 在检查所有资源的版本和冲突后统一取得。Region 的多次 publish 仍是
源程序中分开的提交点，并非跨整个调用的一笔事务。后续资源失败时，已提交
前缀保留，原调用持有其 receipts 并回收剩余私有写入；不能从头重放。

[多资源测试](../../test/boundary/test_region_multistate.py) 使用同一签名 arena
中的三种布局，覆盖同键资源隔离、混合 request/instance 生命周期、route 加入
与休眠回收、初始化和 pin 故障、lease 的整体取得、部分提交后的取消，以及
旧推理版本的保持。普通作者接口和 vLLM 的单状态效果适配器仍待逐一迁移；
这些测试不代表多资源模型接线或全局调度已经完成。

## 条件 replay 的生产迁移

[ConditionalReplayRegion](../../pllm/execution/region_replay.py) 显式表达有界 while、
第一轮分支、relative L2 判断、有限值检查、模型 step 和 trace 次序。旧
ReplayStateTask 中的 Python 迭代循环已移除；外层仍按原 request selection 和
已发布 ref 选择行，并在原异常路径调用 `primitive.restore()`。

首轮没有 previous，因此不做收敛比较；后续只在 `relative <= threshold` 为真
时停止。测试专门覆盖有限张量的 FP32 norm 溢出导致 NaN distance 的情形，
保持原比较语义。CUDA 测试核验原 stream，绑定后禁用 ISL 编译入口，确认执行
不重新生成计划。原语输入/输出副本的额外工作区计入后端 effect reservation，
不沿用旧调用的复制成本。此处仍是保守上界，后续需与统一活跃值规划合并。

已用 `/workspace/pllm/.venv/bin/python test/gpu/vllm_conditional_replay_smoke.py`
验证真实 Qwen2.5-0.5B-Instruct、bfloat16、22–23 层的条件 replay：4 次 replay、
4 次独立数值比较、fallback 为 0；检查正常输出、历史 KV、当前槽、RNG、原 pin
与生命周期。沿用现有数值界限。该测试验证有限层段的原模型接线，不证明全局
wavefront、吞吐提升或完整模型迁移。

## 模型层循环进入区域程序

[DecoderSource](../../pllm/execution/decoder_region.py) 表达 vLLM tokenwise
普通前向、tokenwise replay 和 conditional replay 的层循环。第一层单独调用，
明确使用 `residual=None`；后续层通过 scan 携带行索引、请求 ordinal、token 位置、
hidden、residual 和 positions。层调用使用绝对层号，连续层段用仿射偏移计算，
非连续层集合按声明的层号 bank 选择。
需要读取源层、目标层输出时，区域内的条件分支在对应层计算 hidden+residual，
最多携带两份结果，不收集每一层的完整激活。层数不增加循环体模板大小。

每个外部端点只调用一个原 Qwen2DecoderLayer。原 attention metadata、当前
KV 槽、备份和恢复留在 vLLM，overlay 仍逐层取得和清理原 published pin 绑定。
区域借用同一 StateTaskContext 检查取消和完成状态；不创建模型 KV owner 或
新的页状态。原 Python 层循环已经从这三个后端路径移除。

每个模型层调用声明同一个 `kv[request, layer, token]` 资源族上的访问：读取
`[0, token)` 的历史并写入当前 token。当前 K/V 写入后的读取发生在同一个
层 kernel 内，不误记成对旧当前槽的读取。普通前向与 replay 使用相同的
绝对层坐标，因此会对同一个逻辑 KV 单元产生正确的写写约束；token 输入
准备另外声明备份所有层当前槽的读取。

`BackendPlacementProbe.causal_domain` 携带请求、层、序列位置的上限，来自
原 runner 的 `max_num_reqs`、模型层数及 `max_model_len`，进入原 attachment
签名。Python 和 Rust 校验正整数范围，后端绑定再核对原模型；tokenwise
不再依靠 packed row 数猜测历史范围。入口行索引、请求和 token 坐标在
实际层调用前与原 packed row 核对，再由原页表证明定位物理槽。

索引声明已经进入依赖分析。已知的绝对层维度会被保留，即使请求或 token
来自运行时 bank 而不能静态确定；未知维度仍保守覆盖。后端上下文与恢复
数据已按原始行身份绑定，但在途行的容量和完成约束尚未进入可重排证明。因此
目前还不进行跨 token 重排；在途行容量与完成约束继续接入。

[TokenwiseRegion](../../pllm/execution/tokenwise_region.py) 承接了外层行循环：
每个请求有一段有界参数 scan，按原 packed row 顺序完成普通前向，条件分支只对启用的 prefill
行执行混合、观察和 replay，并按原顺序收集普通前向输出。程序在冷绑定时
使用后端模型布局与行数上限构造；执行时只绑定本次行信息与原端点。混合仍
调用原有精确数学，尚未导入 export 图。

普通前向与 replay 的 DecoderSource 直接组合到 token 循环内部。共有的
[inline_region](../../pllm/contracts/region_composition.py) 校验源签名，绑定
入口参数并为局部值换名；它不生成新的调用或执行边界，原有嵌套区域继续使用
自己的显式 block 参数。组合后的 Program 一次性进入原 verifier、依赖分析
和 lowering。每个单层调用都显式携带行索引，vLLM 只绑定输入准备、单层计算、
归一化和本行完成端点，不再创建或执行嵌套的 normal/replay 区域。
conditional replay 的独立 DecoderRegion 也使用同一个 DecoderSource。

独立层段现在把本次 request ordinal 和 token position 绑定为有界 INDEX
参数，再接入同一 DecoderSource。逐层 KV 读写关系因此保留请求和历史位置，
不同位置共用一份冷计划；越过已签名模型域的坐标在调用层端点前拒绝。packed
row 仍传给原后端核对身份，页表、恢复和完成的所有者不变。组合 tokenwise
程序也已移除请求号和位置的入口 bank：请求号来自原 packed plan 的请求段，
位置由该段的 `prefix + token_index` 计算，packed row 来自前段 token 数之和。
每段保留各自的 token 数和历史长度参数，零长度段不调用端点；输出 bank 只在
返回时按原请求顺序拼接。入口校验原请求顺序、段内位置连续性和两个尺寸关系：
各段 `prefix + tokens <= model.positions`，所有段 token 数之和不超过原批次容量。
条件资格仍逐行传入，允许同一段包含 prefill 和 decode 行。

请求段和内部层循环进入同一个 Program，并共用原执行器、端点和 owner。
当前源程序按已签名的最大请求数建立请求段，因此冷规划规模随请求容量增长；
token 数和历史长度的变化不引发重编译。更大请求容量下的冷规划开销仍须优化，
这次接线不代表已放开跨请求或跨 token 重排。

参考执行路径在冷绑定时只推导依赖、参考顺序和取值关系，不搜索随后会被
丢弃的 ISL 候选；TensorScan、AdamW 与条件 replay 的参考绑定采用同一规则。
`analyze_region_program(..., optimize=False)` 仍执行原有来源和合法性分析；
auto 优化入口继续搜索候选，serial 策略保留数学候选而不搜索其他顺序。
组合后的现有 CPU 用例
在取消这次无用搜索后从约 100 秒降到约 8 秒；这是冷绑定费用的变化，不是
模型吞吐结论。

模型绑定现在先通过 [paged access](../../pllm/execution/paged_access.py) 核对
本次调用的物理 KV 访问。每个当前 token 的 slot 必须与该请求页表中的逻辑
位置一致；请求身份与 ordinal 一一对应，同一请求的本批位置连续。历史读取
按页和 head 保留为带 stride 的区间，当前写槽单独表示，备份当前槽的读取也
包含在调用的读取包络中。区间投影使用实际 tensor 地址、stride 和元素字宽，
支持 HND/NHD 以及同一底层存储中的交错层布局。

检查允许请求共享只读前缀，但拒绝当前写槽与其他请求、层、head 或逻辑位置
重叠；同一请求把不同逻辑页映射到同一物理页也会被拒绝。地址包络的扫描后
再检查精确的 strided 区间相交，不把布局中的空隙视为实际访问，也不把历史
展开成逐 token 的计算图。页表与 slot 元数据只复制本次需要的范围，同一视图
只复制一次；KV 载荷仍然借用原缓存。选中单行 attention metadata 时复用这份
已核验槽号，不再逐层从 GPU 读取 slot 标量。

tokenwise 在模型层执行前核验全部层；conditional replay 在选中层段的首层
执行前核验该层段和完整 packed rows。缓存和页表在本次调用期间仍由 vLLM
保持稳定，恢复与 GPU 完成仍由原所有者处理。这是调用级别的物理别名依据，
不能独立替代 ISL 中的索引访问语义。逐层声明现在与原行坐标及页表绑定对应，
这些证明与已拆分的宿主效果一起参与依赖推导；恢复边界与联合调度继续接入。

观察端点同时接续行上下文和 trace 顺序，确保观察先于 replay 层执行；两条
分支都在输出验证和复制成功后调用统一的行完成端点，启用的 prefill 行再
记录 replay。tokenwise 源程序已没有不透明 model 屏障，ISL 可在保持 trace 与
提交顺序的同时移动独立行计算。两请求、三层、最多八行的源程序得到 locality
和 parallel 候选，保留行的保守上界分别为七和八，参考顺序为一。生产仍执行
符合现有单行恢复预算的参考计划；候选合法不等于已获容量许可或获得加速。
下一步需要联合激活、备份、宿主元数据和原 completion owner 的预算选择。

tokenwise 后端用原始 PackedPhaseRow 定位该行的 attention context 与 KV
备份。准备新行不会覆盖其他未提交行的上下文；每次层调用显式使用对应行。
`commit(row, record)` 在原 trace record 成功后释放该行数据；记录失败时仍
保留恢复所需的备份。出错后 restore 恢复全部未提交行，已经提交的前缀不受
影响。区域执行器不再维护单独的 active 标记，恢复数据和提交状态由原后端
一起维护；原 owner 与 stream 继续负责资源和 GPU 完成。

冷绑定现在从 `begin(row)` / `complete(row)` 的原始 INDEX 操数推导保留区间。
`RetainedAllocation` 导入后端每份载荷的字节上界和两个端点；ISL 验证每个身份
恰好准备和释放一次，源顺序与候选顺序都必须先准备后释放。根据候选时间计算
同时存活的身份上界，覆盖整个参数域，包括空 request 和不同长度的 packed rows。
端点藏在动态控制内部或身份无法精确解析时拒绝出具证明。

tokenwise 使用这份分析核对原 attachment 的单行恢复预算，两处共用
`12 + layers*(2*hidden*width + 24)`：共享 int32 sequence、每层当前 KV 槽
备份和最多三个 int64 metadata 元素。后端校验 KV 形状、类型和 metadata 类型，
张量 wrapper 副本另计。当前参考顺序得到一行上界；独立行全部提前准备的合法
顺序则可能需要全部行的备份。释放端点只是载荷保留范围，不能充当 GPU 完成
证据。后续多行重排仍需联合激活预算、原 owner 的完成关系、提交前缀与恢复约束。
外层端点的额外张量复制另计 `12*hidden*width` 字节，保留的普通输出和最终
拼接继续使用原有行数预算。

接线沿用现有 recirculation 用例，验证独立递推、行选择、失败恢复及取消后
保留已完成前缀。Qwen2.5-0.5B 的现有独立 FA2 oracle 检查通过：8 次普通前向、
6 次 replay、14 次数值比较，未放宽容差，fallback 和额外状态根均为零。

NativePrimitive 保留私有输入和独立输出，保护 RMSNorm 等原地修改产生的
旧版本。额外张量副本在 effect reservation 中增加固定的 `16*hidden*width+64`
字节包络，与层数无关；width 至少为 FP32 字宽。它是附加复制预算，原模型和
attention/kernel workspace 继续单独计账。后续需要通过活跃值与 bufferization
证明减少这些副本，不能把这次迁移表述为性能优化。

已有两个真实模型 smoke 使用仓库 `.venv/bin/python` 通过：
`vllm_recirculation_smoke.py` 检查 8 次 normal、6 次 replay 和 14 次独立数值
比较；`vllm_conditional_replay_smoke.py` 检查 4 次更新、4 次 replay 和 4 次
独立数值比较，均为 Qwen2.5-0.5B-Instruct/bfloat16，fallback 为 0。原数值
界限保持不变，覆盖历史 KV、当前槽、普通输出、RNG 和原发布生命周期。

## 第一个生产算法接入

[TensorScan](../../pllm/execution/tensor_scan.py) 将导入 cell 的递推表达成 scan：
cell 接收 carry、不变输入和序列当前行，返回下一轮 carry 及需要收集的结果。
它生成可组合的源 Program，再绑定原生循环与取值指令；执行阶段不调用 ISL。
百万步测试仍只有一个 cell 语句，结果来源和最后使用关系也保持符号表示。

[In-Place-TTT](../../pllm/workloads/inplace_ttt.py) 保留带注解的普通方法作为签名
参考定义。绑定阶段核验该定义，然后通过共有的 overlay 适配器导入 scan 实现。
生产 chunk 循环由新执行器运行，不再调用参考方法里的 Python 数学循环。
tail 的拼接、未满块输出、私有缓冲区写回仍在导入适配器里；请求 lease、页写入、
多 target 的原子发布、条件 replay 与失败回收仍由原执行链负责。
这只迁移了递推数学，尚未把完整 workload 或模型层段交给全局区域规划。

bank 长度现已通过 `chunks` 形状参数绑定到实际 chunk 数。冷绑定对每种张量
布局只生成一份 scan，数量不再随 `max_rows / chunk_size` 增长。参数的上界
仍来自原后端行容量，用于合法性与资源约束。连续和转置权重各有导入签名，
保留 `base + delta` 的布局，避免无意改变 GEMM 舍入路径。

验证覆盖 CPU/CUDA 的 float64、float32、bfloat16，连续/转置权重、跨调用 tail、
prefill/decode、更新计数和页版本，与原方法逐位一致；测试还在绑定后禁用 export
和 ISL 编译入口，确认请求执行不重新编译。原多 target 故障注入与回收测试继续
通过。当前采用参考 schedule 保持原求值次序，尚未做跨层 wavefront 或性能验收。

## Completion 训练的优化器迁移

`CompletionWorkload`（包括 TTRL）默认把 AdamW 数学更新 lower 到
[AdamWRegion](../../pllm/execution/region_adamw.py)。源参数显式包含每组参数、
step、一阶与二阶矩、新梯度；结果是原布局中的五个更新字段。旧 gradient
槽只是待覆写的存储，不作为数学输入。不同形状、dtype 的参数保留原排列，
由一次 native AdamW batch 调用执行，延续 CPU scalar / CUDA foreach 路径。
该 native import 是优化器 kernel 边界；当前没有把内部算术拆开做等价变换。

编译在 host 声明阶段完成，源 Program、容量依据和额外复制预算进入实现身份。写回只借用原
`TensorOperation` 已取得的私有 roots，发布仍发生在原来的 commit task。
无梯度时不执行 region，也不增加 step 或页版本。函数化带来的额外副本由
原 AER 从源存储关系和 native 调用边界推导，再计入 batch adapter 显式分配的
gradient 槽。当前布局下，每组参数的额外载荷上界为
`13 * parameter_bytes + 12`，替代此前的手写倍数。原 kernel 和 loss 的预算
完整保留；这个变化不证明 GPU 总预算完备，也不证明迁移带来加速。

[导入测试](../../test/boundary/test_region_adamw.py) 对 CPU/CUDA、四种浮点
dtype、混合参数组逐位核对原 batch 更新，并验证输入没有被修改。
[生产测试](../../test/workloads/test_completion_region.py) 通过普通 completion
作者接口运行多组参数，与独立 `torch.optim.AdamW` 的参数、梯度、step 和矩
逐位比较；绑定后禁用旧更新函数及 ISL 编译入口，检查跳过更新和预算边界。
TTRL 原有的提交前故障、提交后结算、原 pin 保持与回收测试继续适用。

这一步迁移的是 completion 优化器数学。loss/autograd、完整状态效果和控制
仍待组合进同一源 Program；causal adapter、普通参数绑定以及 paged/captured
优化器仍在迁移范围内。

## 张量数学改写的首条规则

[tensor_rewrites](../../pllm/execution/tensor_rewrites.py) 直接在导入的函数式
ATen 图上生成 `(A @ B) @ C` 与 `A @ (B @ C)` 的单处重结合候选，支持
`aten.mm` 和确定为二维矩阵的 `aten.matmul`。只接受静态 shape、同 dtype /
device 的实数浮点张量。广播、批量矩阵乘法、整数和复数暂不应用这条规则。
中间乘积必须只被最终乘积读取一次；被返回、被其他操作使用或在 `X @ X`
中读取两次时保留原图。

每个候选记录原始声明、前后图摘要、被替换及新建的节点、三个操作数、矩阵
维度和前后计算量。中间字节数只指这两个乘法内部的那一个乘积，不代表整个
图的峰值或 GEMM 后端工作区。例如 `(12×2 @ 2×16) @ 16×3` 改写后，
按一次乘加计两次 FLOP，计算量从 1920 降至 336，中间结果从 192 个元素降至
6 个元素；这些事实本身还不能证明 GPU 加速。

候选使用私有 FX 图，重新计算新节点的 fake tensor 元数据，不运行用户的
数学函数。导入时显式保留 export 的用户结果位置，使 mutation 输出排在
返回元组前面的函数仍有正确 ABI。原图、输入版本和发布机制不变。绑定可以
从原图重放变换记录；重放重新检查匹配条件和成本，不能仅凭摘要信任外来的
callable 或成本数字。每步只改一个匹配位置，搜索可以串联多步，且保留完整证据链。
新节点延续匹配位置的 `aten.mm` / `aten.matmul` 种类，避免逆变换仅因更换算子
名称就被误认为一个新的数学候选。

重结合成立于实数矩阵代数，会改变浮点求值路径，也可能改变溢出行为；它
没有对任意输入承诺统一误差界限。生成候选不会选择它或发 warning。
`RegionExecution` 继续拒绝未经数值选择的改写 cell。纯张量仿射区域现在可
通过下述冷调优路径，将改写与 ISL schedule 一起交给原 AER 选择，再从原图
重放选中的变换。完整生产任务的后端费用和原 owner 预算仍需接入；不能用
局部 FLOP 数在 Python 运行时另选一个数学实现。

[测试](../../test/boundary/test_tensor_rewrites.py) 覆盖双向改写、CPU/CUDA、
独立原算法比较、共享中间值、同一节点的重复读取、多个候选位置、零维长度、
转置布局、函数化写入与其他输出，以及变换记录的篡改拒绝。float32 使用
`rtol=1e-5, atol=1e-6`，float64 使用 `rtol=atol=1e-12`；低精度另外用可精确
表示的二进制分数核验 dtype 和接口。没有因为重结合而放宽原生产算法测试。

## 有界的多步矩阵重结合搜索

`explore_matmul_reassociations` 从原 export 图做广度优先搜索，每条路径保留
按顺序排列的 `MatmulReassociation` 记录。默认搜索两步，可通过
`max_rewrite_depth` 和候选数量限制控制冷调优范围；每步仍重新检查 shape、
dtype、layout、私有中间值和函数式输出 ABI，不能用已改写的 callable 充当
新的来源证明。

搜索不会要求每一步都减少 FLOPs 或中间字节数。例如，维度为
`1×1、1×3、3×4、4×2` 的四矩阵链，两个单步邻居都增加计算量和新增张量峰值，
但其中一条两步路径把 FLOPs 从 46 降到 44，新增张量峰值从 56 字节降到
48 字节。局部贪心会错过这个候选；最终选择仍看整图容量和实测耗时。

去重键忽略自动生成的节点名称，同时保留操作顺序、操作数身份、字面量类型和
输出 ABI。另一个仅用于剪枝的表达式键排除回到源表达式的逆变换路径，避免
仅重新排列独立计算就被当成一次舍入变化。这些键都不作为一般等价或别名
证明，也不代替最终图的存储导入。当前搜索限于有明确前提的重结合规则，
尚未引入任意算术规则。

`explore_matmul_combinations` 将同一搜索扩展到多个原始函数。每个候选的
`max_rewrite_depth` 是整个区域共享的总步数：两步既可以连续改写一个函数，
也可以分别改写两个函数。函数按来源身份排序，各函数内部保留完整证据链；
互换两个独立函数的改写次序不会增加重复候选。搜索不展开完整笛卡尔积，
也不要求单独改写某个函数后就满足容量限制。

例如，两个调用共用输入，各计算 float64 的 `(4×3 @ 3×128) @ 128×2`，
并返回各自结果。只改写其中一个调用时，另一个仍需要 4,096 字节的内部
乘积，区域的张量载荷上界仍为 9,440 字节。联合改写后，两个内部乘积均为
48 字节，区域上界降到 5,392 字节。若限制设为 5,392 字节，原 AER 只会让
联合候选进入计时。这里比较的是源派生张量载荷，不包含后端额外工作区。
这种联合改写保留调用边界；下节的源变换另行探索消除边界后的计算图。

绑定时从本地原图逐步重放整条链，检查前一步输出图与后一步输入图的来源
关系，再把最终图重新导入原 AER。省略后缀、跳过前缀、颠倒顺序或篡改成本
都会在重放或最终导入匹配时被拒绝。只有整条选中路径绑定成功后才提示数值
变化，执行阶段不重放搜索。

[搜索测试](../../test/boundary/test_tensor_rewrite_search.py) 覆盖逆循环去重、
局部计算/存储最小值、CPU/CUDA 实际执行、函数式 mutation 与返回 view、
证据链篡改，以及原 AER 选中两步路径后的绑定。图中间值和输出都沿用原数值
测试要求，没有为更深的搜索放宽容差。
[联合搜索测试](../../test/boundary/test_region_rewrite_combinations.py) 还检查
跨函数去重和共享步数限制、只有联合改写才可容纳的 CPU/CUDA 执行、输入与
RNG 保持不变，以及缺失或篡改任一函数证据后拒绝绑定。

## 从相邻调用组合可检查的张量图

`CallOutline` 指定一个源区域中的连续调用范围。当前由 Rust 从原 Program
推导外部输入，以及还被后续操作或区域结果使用的全部输出；Python 直接解码
该结果，并从原 Torch 调用重建实际计算图，不再维护另一套结构提取器。
例如，`AB` 即使不是最后一个调用的结果，只要区域还返回它，就必须作为
组合调用的结果保留。源变换不跨控制区域，不越过有资源效果或非张量调用，
也不展开循环。Scan、Map、If 和 While 的子区域可以分别提取；带动态控制的
完整成本调优仍受现有分析能力限制。

[张量组合实现](../../pllm/execution/region_outlining.py) 将这些原调用的函数式
ATen 节点按原顺序内联到一张 FX 图，并按每个调用的 `output_positions` 连接
SSA 值。它不执行用户 Python，不包裹一个隐藏调用链，也不删除没有返回的
计算。Fake tensor 传播重新检查原调用的输入、输出布局守卫和最终输出 ABI；
组合图可直接进入既有存储分析与数学变换。

这让两个分别只包含一次乘法的调用 `AB = A @ B`、`out = AB @ C` 暴露出
完整重结合位置。若 `AB` 同时逃逸到区域结果，既有私有中间值前提会禁止
擦掉它。原地作者写法已经函数化：组合后仍返回新值，旧值及其 view 的内容
和存储来源保持原 SSA 语义。

选择候选携带 `CallOutlineEvidence`，而不是一个声称“已融合”的标记。原 AER
依次重建结构变换后的 Program，再从该 Program 重新检查 schedule、来源和
容量。证据里的基准访问图描述精确内联后的数学；本地绑定必须从原图重建
并匹配它，才允许在此基础上重放数值改写。Rust 负责独立的结构证明和成本
验证，ATen 内联由本地张量图重放核验，不能把一个有效的存储描述当成数学
等价证明。返回的执行器使用重建后的 callable，不接受提案中的可执行对象。

精确组合不改变 ATen 求值顺序，关闭数值变换时也能参与选择。随后在组合图
上做重结合则仍受数值策略约束；只在最终选中、绑定成功后提示。参数化区域
的判断覆盖整个声明范围，零步计时样例不能让其他合法长度的数学变换绕过
策略。选定计划仍由原指令执行器运行，不新增调度器或状态存储。

自动探索当前每个候选提取一个连续范围，最多探索八个范围；组合后的候选
和原 Program 轮流占用有限目录，各自再交错探索 schedule 和数学改写。
绑定格式也能重放多次结构提取，自动多范围组合搜索尚待扩展。

组合图仍逐项调用原 ATen kernel，不代表已经做后端 kernel 融合。它还可能
延长外部输入生存期：原调用可释放的输入，在组合调用返回前可能仍被持有。
所以旧 Program 的容量见证不能复用，减少调用次数也不保证减少峰值内存。
`(4×3 @ 3×128) @ 128×2` 的分离调用与精确组合均有 9,376 字节张量载荷
上界；进一步重结合后为 5,328 字节。在 5,328 字节限制下，仅后者进入计时。
这些数字仍不包含后端额外工作区，也不是模型吞吐结果。

[结构测试](../../test/boundary/test_region_outlining.py) 覆盖逃逸结果、内部布局
守卫、共享旧版本、嵌套控制、保留无用计算、效果拒绝与真实存储生存期。
[选择测试](../../test/boundary/test_region_outlining_selection.py) 覆盖 CPU/CUDA
自动组合加重结合、数值关闭、多次结构证据、伪造基准图，以及百万步参数域
在零步样例计时后按其他长度执行。原数值容差保持不变。

## 后端编译及其分配边界

独立矩阵乘法已有一条实际合批变换：在导入的 FX 图中找到 shape、dtype 和
device 相同、全部操作数在第一项乘法前已经可用的一组乘法，将两组操作数
分别 stack 后执行一次 `bmm`，再按原顺序 select 各个结果。原结果的所有
用户都保留；有前后依赖的乘法不能放进同一个 batch。单次规则最多处理
八项，多个规则可在原总步数和候选数预算内组合。相邻调用提取后，原来
分散在不同调用中的独立乘法也能进入这条规则。

合批证据记录原乘法、操作数、shape 与打包字节数，并绑定变换前后的图。
选中后从本地原图逐步重放，不能用一个 `bmm` 标签或较低成本声明替代证明。
这条规则与矩阵重结合共享有界搜索和 AER 选择；关闭数值变换时不探索合批，
只有最终选中才提示可能改变浮点路径。运行阶段执行已绑定的 ATen `bmm`，
不再搜索或改图。它不把循环更新改成一次更新，也不改写 native 因果 KV 调用。

stack 的真实张量分配和复制流量进入原访问图；select 返回的切片共同保留
整块 batched 输出存储，只计一次底层分配。合批因此可能比原方案占用更多
内存，必须通过整个区域的载荷检查，再按完成时间选择。BLAS 内部工作区、
allocator 与完整设备费用仍需后端依据，不能由这些张量字节推出。
[合批检查](../../test/boundary/test_tensor_batching.py) 在 CPU/CUDA 上核验
两个原调用实际成为一个 `bmm`、输出与输入保持、共享用户、来源重放，以及
数值开关和紧预算拒绝；选择链路使用受控计时，不作为加速证据。

[tensor_compilation](../../pllm/execution/tensor_compilation.py) 将已导入的数学图
交给当前环境的 Inductor。当前路径支持 CPU/CUDA 上的静态非负 stride 输入，
以及浮点逐元素算术、sum/mean 归约和它们的组合。例如平方、均值、rsqrt 与
缩放组成的归一化可以作为一个图进入编译；相邻调用提取后也可使用。
CUDA 的 float16/bfloat16/float32 GEMM 也进入这条路径，包括批量乘法和
可融合的后续算术。内部转置、切片、stack/cat 等布局操作交给 Inductor
lowering，返回端口要求规范连续布局；转置权重可保留原 stride 直接借用。其他归约、CPU 库 GEMM 和
未知外部工作区保留原执行候选。运行的是 Inductor 生成的 C++ 或 Triton
kernel，原 FX 图用于来源重放。

适配器用 Inductor 自己的 decomposition table 在私有 fake tensor 上规范化
数学图，原图与输出布局守卫保持不变。GEMM 使用确定的 mm/bmm Triton 模板：
按后端原有顺序至多检查四个配置，采用第一个可生成的模板，不另做模板计时。
这让本地重放取得相同代码，冷阶段也不重复搜索；pLLM 仍测量完整执行来决定
是否采用它。当前不搜索最优 GEMM tile，也不借此宣称比库 GEMM 更快。

编译配置固定开启 `emulate_precision_casts`，融合时保留源算子之间的
float16/bfloat16 舍入。例如 `delta + rate * (y.T @ x)` 中乘法结果必须先
舍入到原 dtype，再参与 carry 更新，否则后续 chunk 会读到不同的状态。
这不保证不同 GEMM 或归约实现普遍逐位相同；编译候选仍受数值变换策略控制。

适配器在融合后的调度节点上检查实际 lowering，并在 codegen 后读取保留下来
的分配布局，包括填充后的 storage 大小。接受逐元素或 sum 归约
ComputedBuffer；mean 由 sum 与缩放组成，分阶段归约的部分和缓冲也计入分配。
还接受无额外修改输出的 TritonTemplateBuffer 和负责打包分配的 ConcatKernel。
填入 concat 的非拥有 view 不算新分配，实际拥有的完整打包缓冲照常计费。
若 lowering 需要不在这个边界中的工作区，则保留原候选。适配器拒绝
外部 kernel、隐式常量、输入修改和未建模工作区。当前关闭 buffer 复用、内存池、
图分区和 CUDA Graph 包装，使导出的分配边界明确；后续可以为这些 lowering
补充独立依据。所有编译 hook 都在 Torch 编译锁及恢复配置的作用域内运行，
并恢复 CPU/CUDA RNG。

生成 kernel 的调用数仅用于测试诊断，不再是 `CompiledTensorImport` 的字段。
测试从最终模块的 `call` 入口统计，不执行生成文本；GEMM 加后续加法仍检查为
一个融合调用。零调用也是合法结果，例如仅返回输入 view 的图：编译器可以没有
新分配，pLLM 仍执行原返回值复制并计入容量。容量依据是实际分配清单、别名及
既有复制约定，不是调用数。生产编译路径不再解析 wrapper 的 Python 源码。

`CompiledTensorImport` 把原数学访问图、编译器配置与代码身份、实际分配清单
分开保存。原访问图不再被当成编译后的内存模型。编译块借用原区域的只读
输入；通用 NativePrimitive 仍复制输入，因为它尚无编译器的无写入证明。
借用不改变原 frame 的存储生存期，也不转移 PTTS owner。原始 kernel 结果在
全部输出复制完成前保留，每个返回端口最终成为独立存储根，包括重复返回
和旧输入别名。输出复制仍会增加流量与峰值；是否值得由容量和完成时间决定。

输入布局守卫不限制实际地址或 storage offset。适配器使用偏移一个元素的
fake tensor 生成允许未对齐读取的 kernel，并核验编译结果没有输入修改、
对齐检查所需的复制或延迟对齐复制。仅关闭 `assume_aligned_inputs` 不够：
当前 Torch 仍会从零偏移样例推断对齐。这个归一化只影响物理代码生成，
原数学图与访问图保持不变；共享旧值或来自前一调用的连续切片可以直接读取。
源输出也允许非零 storage offset，例如合批输出中的第二个切片；原图核验
其合法范围，编译边界仍分别复制每个返回端口，不把切片偏移当成新分配。

原 AER 对这条边界的调用暂存，保守地相加所有编译器分配；借用输入已计入
原存储根，不再重复分配或收取输入复制流量。返回
复制仍是区域存储根。分配即使不同时活跃也先计入上界，避免用某次运行的峰值
冒充全范围证明。融合 kernel 的实际访存尚未导入，因此 `unknown_functions`
仍列出它；已知复制字节数不充当整个 kernel 的流量。CUDA allocator 开销、
驱动资源、冷编译驻留和原 PTTS 所有者预算也仍需单独组合。

开启数值变换时，自动目录会探索可支持函数的编译候选，并在同次调优的不同
schedule 之间复用编译结果。目录先枚举单函数编译，再枚举多个编译块共同
启用的候选；完整组合先于中间子集，搜索仍受原候选总数限制。单函数候选
不必先证明更快，组合才有机会进入选择。每个组合重新导入实际编译分配，
由整个区域的生存期和载荷约束决定是否能测量；绑定时逐个重放编译证据。
这也适用于循环内外无法直接提取为相邻调用的函数。若先做相邻调用提取，
整个组合图还可以进入同一个编译块。编译可能改变融合舍入，因此关闭
数值变换时不编译，只有最终选中的方案绑定成功才提示。归约树也可能改变
累加顺序，沿用同一数值策略，不另设例外。绑定要从本地原图重放
结构和数学变换，再重新取得编译代码与分配清单；仅伪造一个更小的分配数字
不能通过。kernel 在冷阶段编译，已选执行计划不调用编译器或重新调优。

[测试](../../test/boundary/test_tensor_compilation.py) 使用真实 CPU/CUDA 编译，
覆盖 float64、float32、bfloat16、输入和 RNG 保持、重复输出、伪造代码/分配
拒绝、数值关闭、跨调用融合，以及零步计时后的参数化 scan。四个 ATen 运算
可生成一个 kernel；512 字节对齐的 CUDA 例子中，普通返回的完整张量载荷为
2,048 字节，三端口返回为 3,072 字节，与实际 allocator 增量峰值一致；移除
调用内的输入复制前分别为 3,072 和 4,096 字节。另有共享且未对齐的输入切片
检查，确保原版本不变、返回结果独立，且没有隐藏的对齐复制。测试
保持原数值容差；这些局部证据不替代整模型吞吐与完整设备预算验收。
[GEMM 检查](../../test/boundary/test_tensor_gemm_compilation.py) 覆盖未对齐输入、
确定的模板重放、自动合批后继续编译、输出副本和 CUDA 张量分配峰值。

## 存储与访问成本进入原 AER

[tensor_access](../../pllm/execution/tensor_access.py) 从实际 export 图导入
张量 shape、stride、offset、存储别名及操作数关系。导入会在私有图上重放
fake tensor 语义，避免依赖改写后过期的缓存元数据；不运行用户的真实数学。
矩阵改写也据此更新下游 view 的别名，不能只更新两个新建乘法节点的 shape。

原 [AER](../../rust/pllm_core/src/planning/aer.rs) 据此计算每个设备上的张量
生存期、结果保留容量、图内新增张量峰值和已知操作的逻辑读写量。原生边界
检查定义顺序、存储范围、别名一致性和算术溢出；验证成本见证时重新推导，
不只比较签名。这里没有创建第二个运行时或按 FLOPs 排序的 Python 选择器。

转置、切片等 view 不计搬运；返回小切片会保留其整块底层存储；多个返回值
共享同一存储时只保留一次。源图中完全没被使用的局部结果保守保留到函数
返回，与当前 FX 代码生成一致，不假定已经做了 DCE。未知访问模式显式列出，
不能用已知部分的字节数代替整个图的访存成本。后端已融合或 `torch.compile`
实现需要后端自己的访问模型，不能套用原始 ATen 图的费用。

[测试](../../test/boundary/test_tensor_access.py) 给出计算量与访存方向相反的
例子：`(4×1 @ 1×4) @ 4×32` 重结合后，FLOPs 从 1056 降到 512，但 float64
下逻辑读写量从 2368 增到 2624 字节，新增张量峰值从 1152 增到 1280 字节。
这只是成本反例，没有据此宣称实际 kernel 更快或更慢。CPU/CUDA 测试用
实际执行时的张量弱引用和存储身份核对峰值，另覆盖结果别名、空张量、
view 保留大块存储、元数据刷新、未知模式和成本/释放记录篡改。

这些量不是实测 HBM 流量，也不含 allocator rounding、kernel 内部工作区、
外层 frame 副本和 PTTS 驻留容量。输入布局的容量也不能替代真实借用缓冲区
所属 owner 的容量：同 shape/stride 的输入 view 可能来自更大的底层存储。
完整区域组合、后端有效带宽/吞吐、候选选择和真实模型性能验证仍待接入。

## 区域中的符号存储来源

[region_storage](../../pllm/execution/region_storage.py) 将导入的张量存储关系与
源 Program 的精确值来源组合。它区分 entry 的私有张量副本、每次调用产生的
新分配，以及引用旧分配的返回值；多个输出共享一个底层存储时只创建一个
分配根。分配根的大小取完整 storage，不能用返回切片的 shape 替代。

原生分析使用 ISL 关系贯通 call、bank 的 select/update、循环 carry 和结果。
返回视图并不会改变算法值的 SSA 定义，但它的存储来源仍可指向前一次调用或
入口。这两种来源分别推导，不能为了计内存把读取版本关系改成可变 latest。
存储的消费者、出生时间、最后使用和最终保留关系也保持符号形式。百万次
view carry 仍只有一个输入分配根；零次循环和参数边界保留原来的语义。

验证时从原 Program、固定 schedule 和本地重新导入的调用边界重建这些
关系，检查声明、存储大小、来源与保留见证。export 使用实际存储图，
native 调用的标量、张量和 bank 端口使用下述复制规则。
动态控制和 PTTS 效果仍需要相应的存储导入与组合规则，
未支持时明确拒绝该分析，不能把缺失成本计成零。

这一步产出 AER 所需的符号存储依据。下面的容量阶段进一步给出纯张量区域的
载荷上界；后端完成关系、原 owner 驻留容量和自动候选选择仍需继续接入。
不能把最后使用的时间戳当成 GPU 完成证据。

[测试](../../test/boundary/test_region_storage.py) 覆盖返回切片共享大分配、
最后读者、参数化 view carry、零长度、网格在 ISL 候选下的来源和出生时间，
以及篡改后的拒绝。实际执行还修正了普通调用和动态控制步骤的临时作用域：
Python 循环局部变量不再把上一轮输出额外保留到下一次 kernel。CPU/CUDA
弱引用检查核验了这些临时值按原计划释放；状态页和完成对象仍由原 owner
持有，不依赖这些数学结果的 Python 局部变量。

## 参数范围内的张量容量上界

原 AER 的 `region_tensor_memory` 现在组合存储来源、分配时刻、最后使用和
最终保留关系，构造 `分配实例 → 调用实例` 的活跃关系。相同调度时刻采用
保守的共同包络，与执行帧只在时间严格推进后释放旧值的规则一致。没有使用者
的输出也要在产生它的调用期间占容量，不能直接从成本里删掉。

[region_capacity](../../rust/pllm_core/src/planning/region_capacity.rs) 对每个
调用时刻的活跃集合推导有限上界，再覆盖全部声明参数范围。参数先作为同一
调用的条件保留，不能把 `N=1` 和 `N=1000` 的分配混在一起计数；不展开循环，
也不靠测量最大 shape 证明中间 shape 的安全性。

基本方法是坐标范围的乘积，因此是保守上界。对 wavefront 等斜向前沿，也会
尝试用分配时刻的坐标获得更紧的范围；只有该坐标映射覆盖全部根且保持一一
对应时才能采用它。多个独立实例共享一个 schedule time 时不能合并计数。
最终保留结果和暂存实例还会分别求界，取可证明的较小上界。各分配根的最大值
可能出现在不同时刻，合计结果不宣称是精确峰值或最优容量。

最终按设备组合 entry 私有副本、活跃分配和一次调用内部的暂存张量。调用
内部暂存从原始 tensor access 图推导，排除已经计入区域根的输入/返回存储。
逻辑读写量另按源实例域计算，并包含 entry 复制；未知访问模式单独列出，
此时读写数字只是已知部分的上界。零次调用不计该函数的访问和暂存。

[测试](../../test/boundary/test_region_capacity.py) 用真实 CPU/CUDA 张量存储
核验串行、ISL 候选、wavefront 与分块调度。还覆盖百万步递推的常数暂存、
view carry、保留全部结果、输出别名、相同时刻的独立调用、零长度、函数内
暂存和见证篡改。Rust 测试检查参数没有混计，以及部分覆盖或非单射的坐标
映射不能把容量低估。百万行网格仍使用固定数量的存储模板。

这个上界只覆盖当前纯张量仿射区域的张量载荷。allocator rounding、后端
kernel 工作区、异步完成导致的额外保留和原 PTTS/模型驻留容量仍是独立费用；
不能单凭它通过完整 GPU 预算检查。native 内部及动态/效果区域的成本、有效带宽和
延迟模型、完整生产候选选择与模型性能验收仍待完成。

### 多结果切片进入张量访存分析

`chunk` / `split` 常用于 SwiGLU 的 gate/up 和 attention 的 Q/K/V 拆分。
export 原本会保留多结果算子和 Python `getitem`，而区域张量导入要求每个
计算节点产生一个张量，因此此前会拒绝这类普通写法。现在复用 PyTorch 的
`split.Tensor` / `split_with_sizes.default` 分解，将其规范化为实际共享存储的
视图，再进入原有的功能化、访问图与生命周期分析。没有放宽任意 Python
callable 的导入限制，也没有把视图误记成独立的投影副本。

`linear` 和 `silu` 已有明确的逻辑输入读取与输出写入记账；这些仍是未融合
张量图的逻辑载荷，不能当成实测 HBM 流量或 GEMM 库工作区。CPU FP64 与
CUDA BF16 的普通 SwiGLU 图检查保留原输出、输入与共享投影存储；不连续输入、
空分片和分片上的原地修改仍按原功能化规则生成新状态。这项导入能力尚未
把 vLLM 的融合激活或整个 MLP 自动替换成该图，后者还需要原权重绑定、
数值策略、overlay/hook 语义和模型源组合。

## Native 调用边界的复制成本

[region_imports](../../pllm/contracts/region_imports.py) 区分 export 访问图、
普通 native 回调、native kernel 执行契约和编译器生成的 kernel 导入。它们与算法 Program 分开，
没有给每种 kernel 添加一组成本属性，也没有把不透明 kernel 伪装成 ATen 图。

`NativePrimitive.invoke` 在调用前复制每个输入，在返回前独立复制每个输出。
因此，kernel 即使返回同一个输入的两个视图，区域也得到两个独立分配根。
这与 export 返回 view 时继续共享原存储的行为不同。所有私有输入副本在输出
复制结束前仍存活，作为调用暂存计入；返回副本已在区域根中计算，不再重复
计为内部暂存。复制的逻辑读写量按每个端口的 shape 和 dtype 推导。

普通 native 回调的内部算术、暂存和返回值原本持有的分配仍未知。非零次调用会进入
`unknown_functions`，此时容量和流量数字仅覆盖已建模部分。即使返回的切片
很小，它在 kernel 内部也可能保留一个很大的 base；独立输出复制不能消除
复制期间这份 base 的预算要求。native 导入现已接受标量与张量端口，以及
显式资源效果；标量不计设备张量费用，资源本体仍由原 owner 计账。native
bank 端口按不可变容器递归推导每个张量叶子的复制与生命周期。

`NativeKernelImport` 由后端提供本地实现身份、执行设备和完整瞬时分配清单。
这份 ABI 要求 kernel 的工作完成在发出调用的 stream 上，覆盖原始返回值的
backing、内部临时张量和库工作区；包装器的输入/输出复制仍另行推导。借用
权重、KV 或其他驻留资源保留原 owner 和预算。native 检查器验证声明一致性、
缓冲名称唯一性与设备一致性，AER 将清单叠加到包装器暂存，再按并发 lane
计算上界。它是本地后端的执行承诺，不是 ISL 对任意 Python 回调源码的证明。
内部算术和读写流量没有被伪造为已知，`unknown_functions` 仍标记这部分缺失。

外部端点必须绑定完全相同的 kernel ABI。`RegionResources` 在复制输入和产生
效果之前复核所有端点，单次调用时也检查这一绑定；只改变提案中的实现身份
或暂存数字无法授权另一个回调。已有的显式 tensor scalar 读取已使用这份契约，
其设备分配只有包装器的私有标量复制，读取在原发出 stream 上完成。

CPU/CUDA 检查覆盖大 backing 的小返回切片、并发暂存、普通回调不获并发
资格、实际辅助 stream 执行，以及通过原调用借用同一份只读权重。端点 ABI
缺失或实现身份不一致时不会执行回调。vLLM 借用方法的检查使用真实 CUDA
streams 与原 PTTS 调用，不加载模型权重；它不代替原生模型 kernel 的认证。

### 已签后端目录与真实 Qwen final norm

后端目录最初通过 `BackendPlacementProbe.region_kernels` 携带 native kernel，
现已扩展为下文的 `region_imports`。原生 kernel 仍复用上面的执行契约，源
Program 不增加算法优化开关；probe 校验要求原设备上的 eager CUDA kernel、
完整 ABI 和互不重复的函数名。
driver 规划和任务预算使用这份已签目录；worker 从本地模型独立核对实际
提供者，再绑定调用。运行时端点使用实际提供者的契约，不能用提案中的
预期签名替一个不同回调背书。

首个模型端点是 `decoder.final_norm@1`，声明只依赖张量签名和模型层数，
不依赖 recirculation 配置。vLLM 适配器仅在原 `RMSNorm.forward`、加权普通
RMSNorm、连续权重与支持的 dtype/device 均匹配，而且实际派发到原地
`vllm_c` 提供者时公布契约。该 CUDA 操作无额外全局暂存；两个私有输入和
独立输出副本继续由包装器计费，原 norm 权重仍由原模型借用。
epsilon、模块或原权重对象被替换时拒绝执行，即使替换权重的布局相同。

FP32/BF16 的小型检查在两条原调用 stream 上执行真实 vLLM kernel，输出与
原 vLLM 路径逐位一致，输入保持不变；ABI 不匹配在发出调用之前拒绝。
随后 Qwen2.5-0.5B / BF16 的 EngineCore 检查完成八项候选的计时、安装和
两轮双请求生成：目录中实际携带 `vllm.fused_add_rms_norm@1`，三项候选包含
并发费用，记录 12 个 replay 行，原 owner 清理通过，fallback 为零，主进程
未初始化 CUDA。该次已使用有界阶段流水与目录缓存，完整进程约 4 分 34 秒。
选中的第 1 项串行候选约 304.6 ms，参考约 308.9 ms，三项并行候选约
419.1 / 429.2 / 443.1 ms。短请求、共享 GPU 环境下的这次测量不证明稳定
加速，也不替代对齐吞吐验收。Qwen attention/MLP/整层的完整 kernel 契约
和模型数学暴露仍未完成，不能把 final norm 的接入当成全模型并发认证。

### 模型调用的源计算体

后端现在还可以提供 `RegionFunctionBody`：原函数声明与一个闭合的源 Program。
它经 `BackendPlacementProbe.region_bodies` 随原附件签名、传输和重建。
Python 与 Rust 校验数据/效果端口、资源接口和源程序；当前定义为一层展开，
不接受相互调用的定义或隐含的参数域。它是后端对原模型计算体的声明，
不是编译器对任意 Python 回调给出的等价证明。实际提供者必须从本地模型
独立重建相同定义，才能绑定其中的端点。

`inline_function_bodies` 在原源程序中展开这些定义，沿用现有的词法组合，
处理原 scan、分支及 SSA 结果的重命名。后续依赖、调度、载荷和释放点均
从展开后的 Program 推导，不保留一个另外运行的层执行器，也不增加页或
版本所有者。定义没有引入新的算法优化开关；关闭数值变换也不会改变这种
保持原调用顺序的源组合。

首个模型接入把普通 Qwen 层分成两部分：输入 norm 与 attention，以及
post-attention norm 与 MLP。前者声明原 `(request, layer, position)` 的 KV
历史读取和当前槽写入；后者只读取权重与该行的准备上下文，不再带 KV 效果。
两个端点显式传递 hidden/residual，层输出 tap 仍位于完整层之后。
因此 token 0 的 attention 写完 KV 后，token 1 的 attention 可以先于
token 0 的 feed-forward 执行；逆转这两次 attention 的因果顺序仍会被拒绝。
这项合法性检查已通过原 ISL lowering，而不是只打印出另一种调用顺序。

提供者要求原 Qwen 层和未被改写的模块方法。已有 hidden capture、参数 overlay、
激活 patch 或其他 hook 的配置暂不声明该层定义，继续使用其原整层端点；
这些效果需要先显式组合，不能在拆层时绕过。绑定后再次注册 hook、替换
方法或替换子模块都会在源端点绑定时被拒绝。原权重、KV、准备上下文、
提交与恢复仍由原调用提供。

这次拆分暴露了层内的调度边界，尚未把 attention 或 MLP 内部的算术变成
可改写的 export 图。两部分仍是普通 native 回调，内部库工作区与访存未知，
也尚无完整的并发 kernel 契约；它们继续在原 owner stream 上执行。

真实 Qwen2.5-0.5B / BF16 的进程内接入随后完成：八项候选经过计时和安装，
执行 8 个 normal 行与 6 个 replay 行。最后一层共 14 次调用的拆分结果
与原整层调用的 hidden/residual 逐位一致，原整层还通过既有独立 FA2/MLP
oracle；数值阈值未调整。RNG、正常输出保留、原 KV 检查及 owner 清理通过，
fallback 为零。该次仍选中参考顺序，短请求约 395 ms，其余候选约
402–561 ms，没有得到加速。它不证明全模型数值误差范围或对齐吞吐改善；
新增源定义的 EngineCore 跨进程接入仍需单独验收。

数学暴露还需要把原权重借用接入张量计算端口。如果每个层实例先
通过普通 native 返回整个权重矩阵，现有包装器会复制它，掩盖共享权重与
局部性优化的价值。应在原只读资源证明和生命周期下直接使用权重，明确
处理逃逸结果的别名，并从原 export/FX 图接入计算；不能手写另一份模型 forward，
也不能把这些复制从预算中简单删掉。

### 从原 forward 提取层内边界

随后对 modeling 耦合的修订删除了提供者中手写 norm、attention、MLP 顺序的
执行实现。`capture_module_partition` 在不执行叶模块的情况下捕获原 forward，
以最后一个已声明的有状态模块调用为界，提取前后 FX 子图。跨界值按后半图的
首次使用顺序产生，两段共享原模块及其参数，没有第二份模型数学实现。

后端只提供叶调用的外部资源约定：当前 Qwen 适配仍将 attention 绑定到原 KV，
其余已支持叶模块只读权重与准备上下文。无契约的模块、未知函数和未支持的节点
会被拒绝，不能自动认定为纯计算。`None` 输入特化的原 FX 断言仍在前半图执行。
捕获图的身份进入原后端定义，worker 必须从本地原方法重建相同图才能接入。

这一步没有把每个模块变成单独的调度节点，也没有重新实现 ATen。当前切分
仍产生原来的两个调用；数学改写和块内 kernel 编译需要继续接入原张量图。
该捕获器也不为任意 Python 副作用或任意切分的别名 ABI 提供证明；原模型适配器
继续负责支持范围、资源契约、端口和原方法身份检查。

使用 `/workspace/pllm/.venv/bin/python -m pytest` 的定向检查已通过 5 项：
原 Qwen forward 在 CPU 小型叶模块下的两种 residual 分支、原地更新和 KV 写入
时点、另一种模块结构的调用图，以及原有 ISL 因果约束与方法绑定检查。
这组检查不替代真实模型和跨进程接入验收。

原 GPU 检查随后使用同一 `.venv/bin/python` 完成真实 Qwen2.5-0.5B / BF16
接入：8 个 normal 行、6 个 replay 行，最后一层 14 次调用的 hidden/residual
与原整层逐位一致，并通过原独立 FA2/MLP oracle、RNG、KV 和清理检查；fallback
为 0，未修改数值容差。八项候选仍选参考顺序，计时约 401–554 ms，未显示加速。
这是进程内原图切分的证据，尚不覆盖新增源图身份的 EngineCore 跨进程重建。

新增源图身份随后通过真实 EngineCore 重建检查：两份原图定义和原 final-norm
kernel 随附件传输，八项候选完成计时与安装，两轮请求输出一致，共记录 12 个
replay 行，driver 未初始化 CUDA，清理及 fallback 检查通过。该次选择候选 1，
短样例约 393 ms，参考约 400 ms；不构成稳定吞吐改善的证据。这次模型运行验证
的是原图定义接入；下述描述复用改动另由生产命令路径检查覆盖。

### 只读资源直接进入张量计算

这条通用借用边界现已实现为 `BorrowedTensorImport`。它沿用普通函数声明，
输入是原资源坐标，输出是一个有 shape/dtype/device 的连续张量，并指定
原只读 cell 中的字段。Python 与 Rust 都要求它只读取这一个完整 cell，
不接受写入、其他资源效果或任意返回回调。运行时直接从原
`ReadonlyTensorView` 取引用；没有另外创建页池、版本目录或权重快照。
原借用证明继续核对张量身份及其与可写 KV/PTTS 存储的物理隔离，返回前
检查实际字段、类型和连续布局。绑定不匹配在创建执行帧之前拒绝。

存储分析在 `borrowed_storages` 中保留这些引用的来源和生存期，根的逻辑
字节数并不改成零。AER 将其与新分配区分：原权重驻留仍由原 owner 计账，
区域不再预算一份新的完整矩阵。后续 export 运算读取权重的逻辑流量照常
计算；取得引用本身没有设备张量复制。动态控制中的存储上界目前仍偏保守，
不能把它当成已经完成的精确 bufferization。

使用这份借用 ABI 的区域会在返回时逐叶复制完整数学结果，包括 bank 和
分支结果。因此即使 export 返回权重的视图，调用者也拿不到可修改原权重的
别名。`result_copy_bytes_upper_bound` 显式计算该复制，峰值上界还保留复制时
尚未释放的原结果。多 stream 执行先向 owner 排入所有生产 lane 的完成等待，
然后发出返回复制；原完成汇合涵盖这些复制及异常清理。

CPU FP64 与 CUDA BF16 的 SwiGLU 区域已通过原 PTTS 调用执行：实际权重
地址保持一致，Torch dispatch 观察中没有整块 gate/up 或 down 权重的 clone，
输出与原 PyTorch 图逐位一致；返回切片可独立修改。CUDA 检查使用两条借用
stream 和深度二流水。原子分支返回的权重也会物化为独立副本。此处验证的是
借用与复制语义，尚未将 Qwen MLP 的原生投影和融合激活整体迁入张量源图，
也不提供新的 Qwen 吞吐证据。

### 已有 export 图直接进入计算编译路径

`ExportedCell.from_export` 现在接收原 `ExportedProgram`，按其 graph signature
导入全部张量端口。参数、buffer 和提升出的张量常量默认使用原 target 作为端口名，
调用者按原输入顺序绑定版本；cell 不通过 `program.module()` 读取或写回模块状态。
功能化后的 buffer/input 更新保留为显式输出，不能只导入用户返回值而丢弃更新。
需要规范化时复用 Torch 的功能化与分解，原计算不重写成模型专用函数。

原 `capture` 函数入口复用同一导入与检查逻辑，同时保留原有的“修改输入必须
显式返回”要求。输出形状及布局直接读取原图元数据，删除了 export 后为获取
这些信息而额外执行一次真实张量计算的步骤。当前边界支持静态 strided 张量；
符号范围、标量特化输入、未声明效果和 RNG 仍明确拒绝，不静默丢弃约束。

计算编译入口也删除了中段维护的 pointwise、reduction 和 layout 算子白名单。
静态端口检查之后，由 Inductor 适配器判断实现可行性，并从实际 lowering 提供
分配事实。原 CPU library GEMM、未知工作区、输入修改和不支持布局的拒绝
条件仍保留；没有用图层 FLOPs 或未知费用为零来替代后端依据。

使用 `/workspace/pllm/.venv/bin/python -m pytest` 的原导入/数学改写检查
通过 34 项，已有图导入及编译检查通过 32 项，新增 CPU/CUDA 激活图检查通过
2 项。原参数化矩阵链可以直接生成重结合候选；原 buffer 更新返回新版本而不
改变模块状态；原 PyTorch SiLU 图在 CPU/CUDA 均编译为一个 kernel，分配信息
与输出大小一致。沿用既有数值容差，类型检查与两项架构门禁通过。

这是通用张量图与编译接口的接通。vLLM 自定义算子、原模型参数版本到图端口的
生产绑定，以及原模块图与张量图之间的完整替换仍待完成；本节不提供 Qwen
性能结论，也不将外部模型状态视作普通无状态参数。

### 参数借用接入数学候选选择

`borrow_exported_inputs` 将原 export 端口名映射到既有只读资源的字段，生成一份
可内联的源计算体和原函数绑定。普通输入之后是资源坐标；借用调用取得该 cell 的
原参数或 buffer，再将它们传给原图。字段相同且签名相同的端口共享一次读取，
没有新的模型执行器、参数快照或页所有者。当前字段 ABI 要求连续张量。

此前数学优化器只接受完全无资源的程序，导致这类图虽能执行，却无法通过正常
数值策略选择编译实现。现在 `optimize_region` 和 `bind_selected_region` 允许
原 export 图与明确的只读字段访问器共同参与选择。测量前要求原 `RegionResources`
及其只读证明，核对访问器目录与 owner stream；不允许可写资源、宿主资源、普通
外部回调或调用者提供的已变换实现进入这条重复测量路径。

数学规则只改写 export 图，资源读取不参与矩阵重结合或图内融合。调度、容量、
跨 stream 候选仍使用原区域分析；候选测量在原 owner 的 stream scope 中等待实际
完成。只读参数保持原存储，计算结果和功能化状态更新仍是独立的返回版本，发布
继续由调用者的原协议负责。只有选中数值变换时才发出已有提示。

使用 `/workspace/pllm/.venv/bin/python -m pytest` 的 40 项定向检查通过，覆盖
新通路、原图组合选择、stream 选择及计算编译回归。新通路从普通 PyTorch 模块的
export 开始，在原 PTTS 调用下借用矩阵参数和 buffer；CPU/CUDA 结果通过原容差，
没有整块权重 clone，返回切片可独立修改，原参数和 buffer 保持不变。CUDA 的
编译候选实际完成测量，当前样例选中了编译实现；这不是模型吞吐验收。

跨进程后端目录目前仍只传原生 kernel，Qwen 自定义算子及完整参数版本绑定
尚未通过这条通用数学路径。这里打通的是原图、原只读 owner、优化器与执行器，
不能据此宣称模型生产迁移已经完成。

### 冷规划直接消费封闭描述

存储、容量、组合载荷和 stream 规划接口现在既可从本地实现导入，也可直接接收
原 `RegionTensorImport` 元组。描述只携带原端口、存储和实现约定，不包含 Python
可执行体。入口检查名称完整性及唯一性，Rust 继续核验每份描述与原 Program 的
ABI 和存储约束；描述不会因此获得执行权限。

`TokenwiseRegion` 构造时导入一次，容量分析、候选规划、driver 提案检查和计时
结果安装复用这份不可变结果。本地执行仍绑定原函数，并在安装时核对实际实现；
没有增加一种假执行函数来让 driver 冒充持有模型图。

独立进程检查只传递 Program 与封闭描述。在禁止张量图导入和张量分配、且不
初始化 CUDA 的条件下，它产生了与本地图路径相同的载荷和并行候选；返回计划
又通过本地图重建验证。伪造端口 ABI 和重复描述仍被拒绝。与原 worker 命令、
候选测量及安装路径一起，使用 `/workspace/pllm/.venv/bin/python -m pytest`
的两项定向检查通过。
原容量和 stream 回归另通过 27 项；类型检查与两项架构门禁通过。

`RegionExecutionPlan` 进一步把已有 Program、导入描述、调度提案与原预算组合
成无 owner 的本地计划，不增加传输格式。构造时用描述复核依赖、载荷和 stream
约束；`bind` 才接收本地函数，检查它们与描述完全一致并降低为原执行器。
数值变换仍须走候选选择，不能用源计划绑定入口绕过。

tokenwise 源构造和读取 stream 安排都不再创建执行器；首次需要执行时才绑定
并缓存。worker 的候选准备命令显式触发这一步，随后 driver 才开始计时。
每次执行仍使用新的值帧和原调用资源；缓存不持有请求页或复制模型权重。
附件目录检查也复用同一个计划验证入口，不另写预算和 stream 核验逻辑。

对应检查分别禁止冷源构造时创建执行器、禁止候选准备后的执行阶段创建执行器，
独立规划进程仍禁止图导入及张量分配。它们检查的是规划、绑定和计时的边界，
不代表完整模型图已通过附件传输。
使用 `/workspace/pllm/.venv/bin/python -m pytest`，此次最后的 worker 命令与
窗口预算检查共 2 项通过；此前同环境的 tokenwise 工作负载回归也通过。
最终类型检查为 0 errors，导入分层与 Region authority 门禁通过。

### 原图目录进入附件和 worker

`BackendPlacementProbe.region_imports` 复用已有的导入联合类型，替代只容纳
native kernel 的字段，probe schema 升为 2。目录可描述原 export 图、只读字段
借用和 native 端点；编译候选不作为原计算混入目录，仍须经过数值选择。
Rust 在签名前核对名称、源体引用、张量 ABI 和原执行设备，逐个复核源体使用的
导入；纯计算不能因缺少图描述而变成不透明的外部调用。

`DecoderSource` 和 `TokenwiseRegion` 组合源程序时，可以保留尚未绑定的图声明。
已知的算法计算和资源访问器有真实本地实现，远程 export 图只出现在导入描述中。
存储、调度、候选和预算都消费完整描述；不存在为缺失图构造的可调用占位符。
源体展开也覆盖 tokenwise 的最终归一化等端点，不再只限定于层调用。

worker 通过 `TokenwiseRegion.bind_functions` 提供恰好缺失的原 export 图，
计划检查本地 ABI、图身份和存储事实后绑定现有执行器。缺失或不匹配的图不能
进入执行或测量。字段借用直接接入同一 `RegionResources` 和原只读 view，
参数不进入 driver 的计划缓存；结果仍按原借用 ABI 复制，原权重不随候选更新。

生产命令检查使用普通 Torch 模块的原 export 图作为最终归一化，原参数通过
只读字段传入。它经过附件签名、worker 候选准备、原 PTTS/KV 执行和最终安装；
归一化的旧外部回调被禁止，输出与原 Torch 模块逐位比较，原参数保持不变。
旧 probe schema、重复导入、缺少纯图、把编译结果当成源图以及错误本地图均被
拒绝。该检查使用小模型计算体和固定选择分数，不构成 Qwen 性能证据。

同一检查的 driver 在独立进程中仅接收 probe 描述，既没有原模块类，也没有
其图和参数对象；禁止创建 Region 执行器、页 arena 或导入模型 export，CUDA
保持未初始化。返回的已签附件由 worker 原接口重建，并使用前述原图运行。
最终这项跨进程检查通过，附件、窗口预算、源体组合、真实 vLLM norm kernel
及 export 资源借用的 8 项回归通过；pytest 使用
`/workspace/pllm/.venv/bin/python`。Rust 构建、类型检查和两项架构门禁通过。

接口已经允许后端提供原图，但当前 Qwen 提供者仍公布原模块调用的前后分区和
native final norm。把它的内部张量图、vLLM 自定义操作及参数版本接入这一目录，
以及完整模型吞吐验收，仍是后续实现工作。

### 保留原自定义 kernel 的 export 导入

原 Qwen MLP 的 export 含 ATen GEMM、结果缓冲分配，以及原
`_C.silu_and_mul`。Torch 功能化把该写入包装为 `auto_functionalized_v2`，
并显式返回更新后的缓冲。仅接受 ATen 的单计算块入口不能表示这份图；不能
因此手写另一份 SwiGLU，也不能把未知调用默认当成无副作用计算。

`import_exported_region` 现从这份功能化图提取连续的 ATen 与不透明计算块，
保留原 FX 节点、参数端口和块间值来源；tuple 投影留在原调用所在的块中。
ATen 块继续使用 `ExportedCell`，不透明块必须由后端提供真实实现及执行契约。
返回的 `ExportedRegion.signature` 保留 Torch 的功能化签名，调用者可按原
目标与输出位置处理参数、buffer 和输入更新，无需猜测生成名称或隐式发布。

当前 vLLM 提供者覆盖原 `_C.silu_and_mul` 与 `_C.fused_add_rms_norm`：执行复制出来的原 FX 片段，
不重建激活公式，不捕获模型参数。提供者在确认算子后重新传播 fake 元数据，
将 Torch 功能化产生的 backing 纳入分配费用；该 CUDA kernel 没有额外的
全局或库工作区。普通 `NativeKernelImport` 仍采用原输入、输出复制 ABI。
功能化图现使用独立的 `FunctionalKernelImport`：输入只读借用，输出必须各自
拥有完整且互不别名的稠密 backing，直接返回原图结果，取消冗余的外层复制。
后端分配预算包含功能化内部与输出分配；输出已计入外部值存储，调用 scratch
只计剩余部分，未知内部访存仍保持未知。导入只接受原 clone-all 功能化策略，
不复用 Inductor 基于其他上下文消除复制的决定。其他自定义算子需要相应后端
契约，不能凭有 fake 实现就获得执行权限。现有 native 标签语义保持不变，
目录为新 ABI 使用独立的 `functional_kernel` 标签。

目录规划可保留尚未绑定的纯 native kernel，与 export 图一样由 worker 提供
实际计算体。原候选选择器接受这种有完整执行契约、没有资源效果的 kernel，
并把其设备加入完成同步；普通回调和有状态 native 调用仍不进入纯张量调优。
张量 outlining 只枚举有原 ATen 图的相邻调用，遇到不透明 kernel 时保留边界。

小型 `Qwen2MLP(16, 32, "silu")` 使用原类、原 forward 和实际 CUDA kernel，
FP32/BF16 均与原调用逐位一致。两个独立输入共享显式权重端口后进入串行与
并行候选测量；同一份图换入新的参数值也与原模块 `functional_call` 一致。
另检查 buffer/输入更新的原输出次序、输入未被隐式修改，以及未知 native
实现的拒绝。与 outlining、只读借用、worker 接线和模块捕获回归一起，
`/workspace/pllm/.venv/bin/python -m pytest` 共 15 项通过；类型检查为
0 errors，两项架构门禁通过（Region authority 覆盖 58 个生产文件）。

实际 decoder 提供者现已改为导入原只读后半图：Torch Dynamo 先将 `.data`
别名表示为 ATen 节点，再由 export 提取显式参数。vLLM 加载器的参数子类在
后端通过 `functional_call` 换入同存储的 detached 视图，不修改原模型的参数
对象，也不重写 forward。字段绑定同时检查存储身份、设备、dtype、offset、
shape 和 stride；仅形状相同的副本不能成为原权重字段。

混合图通过原 `weights[layer]` 借用参数，并内联到原 decoder 源 Program；
attention 的 KV 与上下文效果仍由原后端端点执行。Torch 的功能化修改结果
仍保留在内部值流中，后半图按原输出签名返回 hidden/residual。driver 接收
源描述，worker 重建实际张量图和 custom kernel 后绑定，不制造空实现。
最初接入要求各层图及字段布局一致，仅支持固定定义；后续 guard 与异构层
接入见下文。该版本真实 Qwen2.5-0.5B/BF16 的进程内验收通过：8 个正常行、6 个重放行、
14 次独立数值比较，导入后半图的两项输出与原调用逐位一致，KV、RNG、状态清理
及 fallback 检查通过。八项候选的短请求完成计时仍选择参考顺序，约 600 ms，
其余候选约 606–832 ms，没有观察到加速。相同的混合源图随后通过 EngineCore
验收：8 项候选约 600–894 ms，选择候选 7，包含 3 项并行候选；父进程 CUDA
未初始化，worker 重建两份源定义及三项 kernel 导入，12 个重放行、重复两请求
prefill/decode、清理与零 fallback 检查通过。这些结果使用保留外层复制的
native ABI；功能化 kernel 去除冗余外层复制后的完整模型验收另行记录。
这项证据不能替代完整模型吞吐验收。

功能化 kernel 的独立 ABI 随后通过同一进程内 decoder 检查：8 个正常行、6 个
重放行、14 次数值比较、KV/RNG、零存留状态根与零 fallback 均通过。8 项候选
完成时间约 579–763 ms，选择参考候选 0；该检查仍不能证明稳定吞吐加速。
新 ABI 的 EngineCore 验收也已完成：8 项候选约 592–776 ms，选择参考候选 0，
并行候选为 2/4/6；worker 重建两份源定义及三项 kernel 导入，12 个重放行、
父进程未初始化 CUDA、重复请求、清理和零 fallback 均通过。

模型捕获随后改为保留 Torch 原有效性条件。`ModuleExport` 将原
`ExportedProgram` 与本地 guard 放在一起；两者来自同一次 `fullgraph_capture`。
原图的输入 source 决定用户输入与借用字段，原输出元数据保留返回结构、重复
结果和直接返回输入；`make_fx` 只转换这份捕获图，不重新追踪模型的另一条路径。
字段仍使用原存储，Torch 的临时参数替换在退出捕获/检查作用域后恢复。
图外 Python 副作用由 Torch 捕获策略拒绝；模型方法、字段顺序、存储绑定和
Torch guard 在端点准备时检查。失败要求重新准备，执行器不自行改变选中计划。
这些本地对象不进入 driver 描述或 wire schema，worker 重建自己的原图与 guard。
实际 vLLM fused-add RMSNorm/MLP 的 FP32、BF16 检查逐位一致，修改 epsilon 被拒绝；
新捕获路径另做完整 decoder 与 EngineCore 检查，结果见下文。
第一次完整检查在运行准备时由 Torch guard 拒绝：冷捕获的 dispatch key 包含
AutogradCUDA，而原 vLLM 推理作用域没有。vLLM 提供者现显式使用原运行器的
`inference_mode` 完成捕获与 guard 检查，并未删除该条件。实际小型 CUDA
后半图在该模式下继续逐位一致，完整路径的重新验收结果如下。
该重新验收随后通过：8 个正常行、6 个重放行、14 次数值比较、零存留状态根、
零 fallback，两份原图源定义均绑定。8 项候选约 587–842 ms，选中候选 1；原输出、
KV/RNG 和清理检查保持原要求。EngineCore 新捕获路径随后也通过：8 项候选约
600–832 ms，选中候选 1，包含并行候选 2/4/6；两份源定义、三项 kernel 导入、
12 个重放行、父进程未初始化 CUDA、零 fallback 与正常退出均核验。单次候选
计时不是对齐吞吐基准。

只读后半图的接入条件随后取消了 MLP 与 post-attention norm 的具体类名要求。
后端只从声明的 Tensor ABI 构造样例，导入原图并验证其字段、更新签名、操作和
kernel 契约。一个自定义残差模块加普通 Sequential MLP 已通过导入及逐位结果
比较；带 buffer 更新的替换模块未能取得只读字段契约。相关 3 项边界检查使用
仓库 `.venv` 通过。

各层定义与字段布局一致的限制随后也已移除。提供者收集每层原计算体，合并
相容的原导入；相同定义直接内联，不同的连续层范围由已有 `If` 区域按层号选择。
没有新增函数注册中心、模型公式或 wire 标签。两个中间维度分别为 8/12、参数
注册顺序不同的 MLP，经原 readonly 字段与区域执行器执行后，结果均与各自原
调用逐位一致；条件区域的容量、资源与依赖仍走现有分析。目前分支内部未进一步
展开为仿射调度实例，且异构完整模型验收尚未完成。Qwen 外层/attention、前半图
资源语义和两张量前沿仍是接入限制，不能据此宣称模型接入已经完全通用。

计算编译入口也已与调度分开：`contracts/tensor_compilation.py` 的本地
`TensorCompiler` 接口传回可执行体、后端/代码身份、分配与 kernel 数量；它不
是另一种序列化描述。原 `CompiledTensorImport` 仍负责候选及重建身份。
Inductor 私有实现移到 `backends/torch_inductor.py`，基础调度只消费导入事实。
`optimize_region` 与 `bind_selected_region` 接受相同的显式编译提供者，默认值
在单一装配处选择；新提供者无需修改候选枚举。35 项原 CPU/CUDA 编译与 GEMM
检查通过，包含真实编译、容量、只读借用、RNG、选中重建和运行阶段不编译；
另核验拒绝编译时仍执行原图。pytest 使用 `/workspace/pllm/.venv/bin/python`。

展开原张量图也暴露了冷调度搜索的成本。保存下来的真实 decoder Program
约 80 KB，仿射分析有 60 个语句模板；原依赖与参考载荷分析分别约 2 秒和
5 秒，而候选搜索在被结束的原实验中运行了约 9 分 38 秒，仍未返回。
这些是编译开销，不是模型执行速度，也不是测试断言的耗时。

求解器现使用 `D - (D ∘ D)` 去除原依赖中已有两跳路径的直接边。
每组有界参数对应有限 DAG，替代路径上的参考区间严格缩短，因此保留原可达
顺序。源依赖、RAW/WAR/WAW 记录以及最终合法性复核仍使用完整的 `D`；
化简失败时保留原关系。百万步参数域和跨语句分支的独立可达性检查通过。
化简后的调试构建在这份真实图上完成两个 ISL 候选及八项组合目录，没有
丢弃候选的诊断，但两阶段分别约 126 秒和 129 秒，冷规划成本仍需改进。

目录去重与窗口过滤随后改为逐语句对比较完整参数域中的严格先后关系。
先检查同一语句的迭代实例，发现不同立即返回；相同的情况仍检查所有语句对
的两个方向，因此不会合并原本同时发生与严格先后的候选。它不预先构造
整个语句对笛卡尔积，也不改变合法候选集合；检查包含跨语句次序、时间平移、
迭代反转和同时执行，并与原完整关系的比较独立核对。
同一真实图生成的八项目录逐项一致，目录组合阶段约 87 秒；这次使用 release
构建，前面的两项 ISL 求解仍约 131 秒，不能把全部耗时变化都归于一种修改。

合法性检查也不再先生成所有语句实例对的执行顺序，而是把每条完整原依赖的
两端映射到调度时间，检查它们严格递增。原覆盖、参数名和单值性要求保留。
保存下来的八项真实候选均通过新检查；首次含原图分析约 1.24 秒，后续每项
约 0.17–0.66 秒，进程未初始化 CUDA。该改动的依赖/参数化回归中有 54 项通过，
另一项发现旧断言仍把无资源效果的动态分支当成全局屏障；按已有的效果投影语义
改为核对单个原子动作后，该项的原运行时结果检查也通过。pytest 均使用
`/workspace/pllm/.venv/bin/python`；五项 Rust 依赖检查通过。

原存储分析也已接入原子 `if/while`。它们返回的张量可能继续别名于输入或
分支内部的 view，因此输出按可能的 backing 上界计费。这是一份保守归属，
不是独立物理分配或可原地复用的证明；源操作决定采用精确调用别名还是控制
结果上界。内部暂存复用原 `ControlBlock` 的 slot、释放点、carry 与收集结果，
嵌套 scan 的普通 carry 不按循环上限累计，收集结果则必须计入完整参数范围。
分支按最大值、顺序调用按峰值组合；有限循环的复制流量按次数计入。

`RegionPayload` 在同一源程序和候选顺序下组合 `RegionCapacity` 与
`RegionRetention`，按设备叠加张量费用和后端保留载荷，并提供独立重建验证。
tokenwise 的 attachment 声明、vLLM 运行时重建和任务绑定现在复用同一份
冷规划结果。原先手写的区域张量暂存公式已由候选的组合上界替代；auto 策略
与纯张量优化使用同一个 `explore_region_schedules` 目录。参考计划排在首位，
全局 ISL 调度、大小为 2、4 的窗口和 ISL band 分块轮流加入，最多保留八个
不同执行顺序。去重比较完整参数域上的先后关系，不能只比较 schedule 字符串
或某次短样例的展开结果。按设备预留可用候选的峰值最大值。
区域载荷现在进入原 `TaskResourceFact.workspace_bytes`，
由同一个 AER 在任务所在锚点约束；不再把这部分重复记入每个锚点的全局预留。
有内存上限时，先扣除已声明的后端预留和驻留费用，只保留剩余容量内的候选。
全局或分块调度放不下而参考顺序可行时，仍保留 auto 策略并安装参考计划；参考载荷
也无法满足这份保守预算时才拒绝。worker 绑定读取原已签 workspace，不能
重新扩大它。serial 策略只生成参考计划，也不进行模型计时。
区域外的 embedding、最终拼接及 native mix 暂存仍进入原 timeline 预留。
外层任务的输出不另分配一份张量，拼接结果的费用已由后端预留承担。
模型/layer/attention 库工作区继续由原后端预留；allocator 和跨 stream
完成费用仍需更完整的建模。当前候选收缩适用于单个无托管状态的模型任务，
按所有锚点上的预留最大值保守计算余量；多任务之间的全局预算优化仍待完成。
存活数量查询复用完全相同的结果，原 ISL 运算额度按独立查询应用，避免一个
较大区域仅因累计多个已完成的小查询而耗尽额度；单个查询仍有相同上限。
源依赖分析、存储来源、生存期、容量推导、指令生成和取值绑定使用有界的本地缓存。键包含
该 pass 的完整输入；调度先做与降低阶段相同的规范化，
仅保存本地成功推导的普通数据；返回副本，外部传入的证据不能填充缓存。

容量查询先识别空关系和每个时间点最多一个存储实例的关系。对其余关系，
先删除仍保留实例唯一性的冗余坐标，再计算最终坐标的边界。删除坐标不会改变
其余坐标的极值，只会从包围盒乘积中去掉不小于一的因子，所以最终结果至少
与每个中间投影一样紧，无需每删除一维就重做整组整数优化。参数仍属于本次
调用的输入域，相同时间上的独立实例也仍分别计数。

Completion AdamW 的内部返回值来自私有参数副本和显式新建的 gradient 槽，
所以适配器可以将后者加到推导结果，再叠加原 kernel/loss 预留。整个过程在
声明阶段完成，预算仍进入原 AER 和原 PTTS 规划，没有新增运行时成本求解器。
[测试](../../test/boundary/test_region_native_capacity.py) 用 CPU/CUDA 实际存活
存储核验复制重叠、返回别名、混合导入和 batch bridge，还检查百万步 carry、
零次调用以及跨 Python/Rust 边界的声明篡改。

native 调用的容量边界也接受已有的 `Bank`，包括嵌套 bank 和有界符号长度。
输入复制按每个张量叶子计费；即使多个输入叶子引用同一底层张量，私有副本
仍分别分配。返回 bank 的每个张量叶子成为独立存储根，使用原符号索引的
来源、出生与最后使用关系；标量叶子不收取张量字节。调用暂存按声明的长度
上界计算，不使用当前样例长度；多个长度有联合约束时，目前相乘各自上界，
因此可能偏保守。运行时仍由原 NativePrimitive 和值帧检查长度并管理副本。
这为批量 kernel 提供同一份参数与容量接口，尚不等于自动合批，也不代表其
内部工作区已知；这些 kernel 仍需后端 lowering 和分配依据。

## 调度与数学候选的实测选择

[region_optimization](../../pllm/execution/region_optimization.py) 提供纯 export
仿射区域的冷调优入口，支持静态张量 shape 与有界的循环/bank 长度参数。
候选零是原图加参考 schedule，后续候选组合调用提取、
ISL schedule、自动分块、可重放的矩阵重结合链，以及原调用可提供的 stream
放置。候选数量和计时轮数有显式上限；
当前支持函数内的有界多步搜索及多个函数的联合改写；分布式调优仍待实现。

原 AER 的 `assess_region_candidates` 对每个候选重新推导依赖合法性、存储
来源和容量依据，检查数值策略与按设备声明的张量载荷限制。不能执行的候选
不计时，也不伪造一个零耗时。源候选超出限制时，只要存在合法、可容纳的
变换候选，仍可继续选择；这不要求先运行会超预算的参考候选。

`RegionCandidate.stream_plan` 是物理选择，源 Program 不包含这些字段。
它绑定阶段调度、lane 访问器和并发容量依据；候选零始终没有并行放置。
AER 对已重放调用提取后的 Program、该候选的数学/kernel 导入和原 retention
重新推导整个 stream 计划，拒绝调度、导入或容量不匹配的提案。Assessment
中的 `payload` 保留独立可验证的阶段存储与 retention 依据，`peak_bytes`
给出用于准入的物理总上界，包括并发调用暂存。只够容纳普通阶段载荷、放不下
多条 stream 暂存的候选不可执行，即使它的计时样本更快也不能被选中。

`optimize_region` 可接收原调用提供的 `RegionStreamBindings`。自动策略下，
每组数学/kernel 导入都能产生串行及并发选择，目录总大小仍受同一上限约束；
同一组导入不会因多个串行调度反复搜索并行计划。serial 策略不搜索并行计划，
本地绑定也拒绝与该策略不一致的并行赢家。选中的执行器不保存 stream 池，
streams 每次由调用方传入；若串行方案更快，使用同一个 owner 并让辅助 streams
空闲。`TensorScan.optimized` 和调用接口已传递这项原调用能力。

并行计划推导使用与其他 native pass 一致的有界缓存，键包含完整 Program、
规范化调度、轴、lane 数、流水深度、张量导入及 retention。缓存仅保存本地成功推导的
结果；传入的计划仍要与本地推导结果逐项比较。

Python 冷路径只负责生成、重放和测量，不决定赢家。每次执行都由原指令
执行器建立私有输入帧；各候选的计时轮次交错排列，在同一个 owner stream 上
开始和完成。测量包含 Python 签发、帧复制、阶段同步与输出完成。上下文记录
设备型号、Torch 版本、线程数、计时轮数和可用 lane 数。它是样例负载的观测，既不是延迟
上界，也不代表长期吞吐或其他形状的表现。参数化区域的样例尺寸还通过
`tuning_sizes` 单独记录；Rust 独立检查它与源参数完全对应、无重复且在范围内。
更换样例尺寸会改变选择请求的身份，但不收窄源 Program 的参数范围。

依赖合法性和载荷限制始终覆盖整个声明范围。例如，保留所有结果的 scan 即使
只用两步计时，也必须为允许的一千步结果保留足够载荷；不能把两步的峰值用于
完整范围的预算判断。同一个选中 schedule 可以接收其他合法尺寸，运行时按
原符号游标绑定，不按样例尺寸重新调优；性能结论仍只对应计时负载。

`select_region_candidate` 在原 AER 内按完成耗时的中位数选择可行候选，
相同耗时保留目录中更早的候选，优先维持源行为。容量是硬约束；FLOPs 和
逻辑字节数都不充当耗时的替代排序键。选中决策绑定完整目录、实测样本、
策略、载荷限制和容量依据，验证时重新求解。

并行选择检查已覆盖普通阶段载荷足够而并发暂存不足的拒绝、快慢样本改变
赢家、物理证据篡改、真实 CUDA 计时与选中计划的无搜索执行。组合用例把
矩阵重结合、编译 kernel 和多 stream 放置一起重放，同一份符号计划执行
0、3、5 步，关闭数值变换时仍选择原实现。测试中的确定性排序用于覆盖
并行赢家的绑定路径，不构成吞吐提升的证据。

绑定选中计划时，还要用本地原图重新导入源事实、重放数学证据，并核对
重放结果与选中候选的访问图。只有这一步完成后才针对实际选中的变换提示；
枚举、私有计时、未选中的候选以及关闭数值变换的路径均不提示。执行阶段
复用同一个指令执行器，不运行 ISL、计时或选择，也不重新解释数学证据。
是否可能改变舍入依据完整源实例域判断：某函数在整个声明范围内都不会执行，
它的改写不触发提示；只是在零步样例中未执行，但其他合法长度会调用它时，
仍须受数值策略和选中提示约束。

### 从 ISL 循环带生成分块

[region_tiling](../../rust/pllm_core/src/planning/region_tiling.rs) 遍历源派生的
ISL schedule tree，寻找至少二维、标记为可置换的循环带，调用 ISL 的
`band_tile` 生成外层 tile 与内层 point 循环。默认尝试 2、4、8、16 的因子，
并在相邻因子之间改变一个轴，得到有限的正方形与长方形候选；不是维度数上的
完整笛卡尔搜索。每个候选只改变一个带，候选数有上限，重复 map 会去重。

可置换标记只是生成前提。分块后仍从原 Program 检查完整实例覆盖、调度函数
的单值性，以及所有数据和资源依赖。尾块保留原域限制；参数化循环保持 ISL
关系，不按最大长度展开。空域和一维带不会为当前串行调用执行器额外生成
只有拆循环、没有重排作用的候选。局部候选失败会作为探索诊断返回。

`optimize_region` 的 auto 路径会纳入这些候选。调度与数学变体按索引对角线
交错枚举，使有限目录同时探索两个方向；serial 策略不生成分块。容量和实测
耗时仍由同一条 AER 选择链处理。分块本身不改变单次调用的数学求值顺序，
因此不属于数值变换提示；执行仍是既有指令游标，还没有把 tile 编译成融合
kernel 或证明它在 GPU 上加速。

[分块测试](../../test/boundary/test_region_tiling.py) 核验自动正方形/长方形
分块下每次读取的旧值、非整除尾块、CPU/CUDA 真实存储上界，以及每个状态
地址上的 snapshot→update→publish 顺序。百万行的参数域仍用短 schedule
表示，并可进入调优选择；绑定后在不同长度下运行，不重新生成分块。

`explore_region_windows` 可以给 ISL 顺序加上原程序的分组边界：例如每两行
组成一个窗口，窗口内允许重排，完成后再进入下一窗口。这与直接限制循环
次数不同，候选仍覆盖完整参数域，并重新核验全部依赖。窗口大小只限制
移动范围；实际同时保留多少恢复数据，要由 acquire/release 生命周期推导。
没有改变执行顺序的窗口会被过滤，避免为相同次序重复计时。

`RegionSelectionRequest.retention_imports` 将原后端保留载荷接入同一条 AER
选择链。`payload_limits` 现在约束张量与保留载荷的合计上界，可行候选仍按
完成耗时排序。`bind_selected_schedule` 校验原函数导入、源程序、选择结果
和本地数值策略后，绑定到既有执行器；资源权限由每次调用的原 owner 提供。
这仍不是完整设备预算：native/library 内部工作区、原 PTTS 驻留与设备完成
费用仍待完成。tokenwise 以候选目录的组合载荷作为已签预留，
`TokenwiseReplayTask.with_schedule` 只接受相同预算
和原恢复端点的实测选择，调用方扩大上限不能获得额外容量权限。绑定返回
独立执行计划，保留可复用源计划；本地策略还必须与原 timeline 签名一致。
vLLM 的无托管状态 tokenwise 路径现在在模型创建后、首个用户请求前做冷选择。
主进程负责搜索候选，任务绑定先重建参考源图。已有签名 workspace 时不再
为验证这份预算重新搜索目录。调优开始时，主进程通过原 worker command
传递 `RegionScheduleProposal`：固定源程序的载荷证明，以及可选的并行物理计划。
worker 从本地数学和源程序重建每项依赖、存储及恢复生存期，并核对原任务预算，
验证通过后才测量。并行候选必须匹配已签名的 `execution/region_streams@1`
后端能力及其两条执行 stream；原生 attachment verifier 限制这项能力只用于
eager CUDA。初次资源规划和 worker 准入都使用含并发暂存的物理峰值，不能
拿普通阶段载荷替代。目录最多八项，第一项始终是没有并行计划的参考执行。
提案不携带 callable，也不能扩大原 workspace。中止调优会恢复开始前的原任务绑定。
测量时绑定的并行计划会随候选进入 AER，并在选中后继续用于原模型任务。

合成文本使用原引擎的请求分配和 KV 回收路径。所有候选先各预热一次，再交错
测量两轮完整 `generate` 的完成耗时；每轮轮换起始候选。worker 缓存本次调优中
不含 owner 的 lowering，避免重复绑定时重新生成指令；每次执行仍建立新的值帧，
借用本次调用的原资源。使用单条短 prefill 和一次 greedy 输出，
不把纯 kernel 计时当成请求完成时间。实际 packed rows 提供计时参数，所有
候选必须使用相同尺寸；合法性与容量仍覆盖原 Program 的整个参数域。

计时阶段复用原 arena、trace 对象和任务执行器；只在没有活动调用时替换
本地已验证 lowering。临时 trace 缓冲区和 PyTorch CPU/CUDA RNG 在结束或
中止时恢复。任务完成后仍按原路径退还请求、等待页回收；提交和恢复回调
没有计时专用版本。先前的 vLLM dummy warmup 使用占位 KV 元数据，不能替代
这里的真实请求计时。原 request/step/invocation 身份继续单调推进，不重置
原所有者的计数器。

所有候选的 greedy 输出先与参考样例比较，再由 AER 按容量约束和实测耗时
选择。`VllmLlm.region_selection` 提供最终决策及各候选费用；短样例的选择
尚不能代表长 prefill、完整吞吐或多请求合批最优，greedy 输出一致也不替代
独立的张量/KV 正确性验证。更广 workload 的自动调优、多任务预算分配
和对齐吞吐验收仍需继续实现。

CPU 的独立递推公式现覆盖目录内每个候选，核对跨行执行下的输出、KV、
多行恢复和后续取消，并检查冷调用结束后公开 trace 与 RNG 不变。Qwen2.5
真实模型检查通过正常/重放层的 14 次独立数值比较；这轮没有据此宣称吞吐
提升。冷规划和 lowering 的后续开销仍需继续衡量。工作负载的按引用传输
已从 GPU worker 模块拆出，CPU 编译与传输校验不再连带导入 vLLM；传输的
冻结字段、模块摘要和 worker 重建规则保持相同。

decoder 源图的 residual addition 现在从 fake tensor 导入，保留目标设备的
shape、stride 和 dtype，不分配模型输入。已经只含无效果、不可变 core ATen 调用
的 export 图直接使用；含 mutation 或 composite ATen 的图仍经过 functionalization。
这样主进程
做源规划时不必因 AOT 的 RNG 保存步骤初始化 CUDA，worker 仍独立重建同一份
源图。fake 输入按存储身份检查别名，不能把其无效数据指针当作真实地址。

EngineCore 的 Qwen2.5-0.5B 检查已完成八个候选的冷计时、选中计划安装、
两轮双请求的 prefill/decode 与原 owner 清理，主进程未初始化 CUDA。
这次短样例选中参考顺序：完整 generate 的中位数约 305 ms，其余候选约
312–345 ms，没有观察到加速。它验证跨进程选择接线和重复请求一致性，
不能替代独立层/KV 数值验收，也不代表对齐吞吐结果。该次进程仍使用缓存
补充前的 native 库，整体冷接入约十三分钟，开销问题尚未解决。

随后用同一份 24 层、64 行、两个请求位置的源图做纯冷规划剖析，仍生成八个
候选，不加载模型或初始化 CUDA。坐标投影改动后，总耗时从约 333 秒降到
100 秒，其中载荷分析从约 305 秒降到 72 秒；目录搜索仍约 26 秒。两个单独
查询的容量推导从约 8.1/21.2 秒降到 1.1/2.2 秒。这些是当前环境的一次编译
诊断，不包含模型加载、worker 重建或 generate，不能作为运行时吞吐改善。
主进程提案、worker 验证的新接线已通过编解码、预算拒绝与 CPU 递推/恢复
检查；仍需在下一次真实模型集成验收中复查完整启动成本。

串行与并行候选现在复用同一份原生 ISL 目录：缓存键包含完整 Program 和候选
上限，最多保留八份由本地求解得到的结果，返回值是独立副本。模型目录也改成
按最终八个名额惰性计算串行载荷，并限制并行候选数，避免算完容量再丢弃。
这些改动不改变候选的交错顺序，也不让外部提案进入编译缓存。
同一份 24 层模型源图的独立目录检查中，首次搜索约 25.95 秒，第二次约
0.013 秒，完整返回内容一致；改用候选上限一仍得到独立的参考目录。这只
衡量目录复用，不包含其余容量分析、模型启动或运行时间。

小型 CUDA 集成已实际经过 `GPUWorker.workload_command`、`VllmSession`、
`TokenwiseTuning` 和原任务执行器：并行提案在选中后保留物理计划，测量与
安装后的输出/KV 均与参考逐位一致，原 owner、公开 trace 和 RNG 得以保留。
这项检查用小型递推替代模型加载，固定分数只为覆盖并行赢家的安装路径，
不证明 Qwen kernel 已能并发或得到加速。

新版提案随后通过 Qwen2.5-0.5B / bf16 的真实 EngineCore 接入：八个候选中
第 2、4 项携带并发暂存费用，全部完成交错冷计时与最终安装；两轮双请求
生成一致，记录 12 个 replay 行，原请求/pin/回收清理通过，fallback 为零，
主进程未初始化 CUDA。参考候选仍胜出，短请求完成耗时约 315 ms，两项并行
候选约 431 / 414 ms，没有观察到加速。此处只并行了符合导入约束的小计算，
不能据此宣称 Qwen 层并发。该次接入尚未使用上段新增的目录缓存与惰性载荷
计算；它也没有完成长 prefill 或对齐吞吐验收。

[测试](../../test/boundary/test_region_selection.py) 覆盖耗时与 FLOPs/流量
排序相反、容量约束、数值开关、非法 schedule、计时及决策篡改、数学证据
重放、只提示选中变换和 CPU/CUDA 实际冷调优。优化执行绑定后禁用规划入口，
核验结果和输入保持；没有放宽现有数值测试的容差。

`TensorScan.optimized(...)` 将这条路径接入已有 scan 作者接口。它接受与普通
调用相同的 carry、invariants 和 sequences 样例，返回新的 scan；原 scan 的
源程序和执行计划保持可用。返回值仍按原顺序给出 carry 和收集列，参数化
长度继续由显式 count 或序列长度提供，并在运行入口检查。调优阶段才会复制
样例和测量候选，普通调用不编译或计时。

`TensorScan` 的循环体也可以直接传入 `Program` 及其函数绑定。作者入口把
该区域的操作放进 scan body，保留原局部名称和输出版本；入口参数依次解释为
carry、invariant 和序列元素。单个函数是只有一条 call 的简写。这里没有再建
一个执行器，也没有把区域封装成不透明调用。当前循环体入口和结果必须都是
静态布局张量，循环次数仍可使用已有的有界参数；带资源效果的组合使用通用
Program API。多调用循环体同样可以进入已有的 outlining、数学和调度选择。

In-Place-TTT 的生产绑定已使用这个入口，将每个 chunk 拆成前向、增量计算、
版本更新三个导入函数。前向读 `delta[c]`，增量只读该 chunk 的输入和目标，
更新产生 `delta[c+1]`；原 ATen 运算次序、布局、尾块和发布协议保持不变。
ISL 可以据此生成先计算各 chunk 增量、再生成权重版本、最后计算前向的合法
候选。最后的前向仍逐个读取其对应的旧版本，不允许改读最新权重。该候选可能
保留更多中间权重，因此生产绑定在冷阶段按容量和完成时间选择，不能只因为
ISL 证明合法就启用。

生产调优读取签名 `SchedulerTimeline.region_policy`。该闭合契约包含已有的
数值变换策略和 schedule 策略，经 `specialize` 写入，随原 attachment 到达
worker，进入原计划签名；worker 不能另用默认值覆盖它。默认允许数值变换，
serial 策略只探索保留原操作次序的候选，关闭数值变换仍允许精确调用组合。

In-Place-TTT 从 transition 的已声明 workspace 扣除输入/目标拼接、剩余行
输出、私有 base 和剩余前向的有效权重，再把余额作为 region 载荷限制。scan
返回后仍保留的 carry 和 columns 计在 region 包络中；最终 cat 的整块存储
计入原输出 value 预算，页和发布仍由原 owner 计账。没有声明外层预算时，
使用同一公式给局部候选设限，但这不等于已经声明了完整设备预算。当前数学
仍保持原 GEMM 的 shape、dtype 与布局；allocator、库 workspace、冷编译驻留
和更广后端变换的完整设备费用还需要继续纳入。

每个布局最多使用三个 chunk 的零值私有样例、十六个候选做冷调优。样例只
决定计时，容量和合法性仍覆盖整个 chunk 参数域。输入使用真实布局，不占用
请求页或初始化 RNG。选定后保留同一个 scan 接口；请求执行不调用 ISL、编译
器或计时器。CPU/CUDA 原算法逐位比较覆盖尾块与发布，另有签名策略传递、
关闭数值编译、剩余预算不足时不测量候选的检查。

[参数化选择测试](../../test/boundary/test_region_parametric_selection.py) 检查
样例尺寸篡改、全范围容量、百万步声明下的常量 carry 暂存、CPU/CUDA 选中
数学改写、零次调用、序列长度与收集顺序，以及双参数网格。绑定后禁用规划
入口，再用未计时的长度核对独立结果公式，避免把重编译当成参数化执行。

## 按新架构继续实现的顺序

`Program.parameters` 可声明有上下界的
`SizeParameter`，`BankType.size` 和 `Scan/Map.count` 可引用同一参数名。
参数在调用入口一次性绑定，bank 的实际长度必须与之相等；相同上下界的两个
符号也不能互换。Python 与 Rust verifier 独立检查名称、边界、引用及结果
类型。参考解释器覆盖零长度、嵌套参数、旧/新值收集和效果次数。

`Parameter` 操作把已声明的参数读成 INDEX 值，因此 `prefix + token_index`
这样的访问可以进入同一份参数化仿射分析。它没有输入，只返回一个索引；
嵌套动态区域也读取本次入口绑定的参数。加减和常数倍运算保留参数系数，
参数乘下标或参数乘参数仍是非仿射运算。范围分析覆盖参数的上下界；可能
溢出的整数运算保留原来的检查和执行顺序，不能因结果未使用而消除。

`SizeConstraint` 表达命名参数的整数线性不等式，例如 `prefix + tokens <= N`。
它是源程序成立的入口条件，随 Program 签名并进入 ISL 参数域；调用和冷调优
样例都必须满足它。Python 与 Rust 分别校验参数引用和系数，运行时使用同一
组已绑定尺寸检查关系。bank、状态单元与函数索引访问的范围证明会使用联合
约束；来源、调度、存储和容量分析继续沿用这一参数域，不能只收窄某次计时的
尺寸而放松其他调用的检查。

这些参数已贯穿 ISL 域、符号来源映射、最后使用关系、AST、动态控制指令和
运行时值帧。内部名称带 `__size_` 前缀，与迭代坐标分开；每次调用先复制并
校验参数绑定，运行时只解释已签名的整数表达式，不调用 ISL。零次 scan 的
carry 直接来自入口，其余情况来自符号末次迭代，不能用上界代替实际长度。
调度进入指令和取值绑定前统一消除冗余约束，使打印/解析不会改变签名语法。

[参数化测试](../../test/boundary/test_region_parametric.py) 用同一份计划运行
不同大小的二维网格，核对 reference、ISL 候选、wavefront 和分块调度下每个
cell 的输入版本、最终值和活跃值上界。还覆盖嵌套动态区域、native bank
签名、调用期间修改外部参数字典、原 PTTS 发布次数与参数绑定不一致的拒绝。
百万步上界不增加语句或来源模板数量。`TensorScan` 和 In-Place-TTT 已使用
这条参数化路径；原 CPU/CUDA 算法比较继续要求逐位一致。

张量维度与资源 family 的符号 extent 尚未开放。旧的原生实例 preview 接口
仍要求静态程序；参数化计划可通过绑定 sizes 的机械游标查看实例，生产执行
不依赖该诊断接口。全局成本选择、跨层模型调度与性能验收也仍待完成。

| 阶段 | 必须完成的交付 | 当前状态 |
| --- | --- | --- |
| 语义内核 | 最小 IR、独立 verifier、场景表示、参考执行与 export 导入 | 已有实现与边界测试；继续随迁移补充 |
| 编译中段 | 从值与逻辑索引推导实例依赖；保留动态区域；ISL 候选与原始依赖复核；可追踪来源映射 | 已接通符号分析、调度和精确来源绑定；纯 export 区域已接入调用提取、数学细化与成本选择；完整生产接入待完成 |
| 统一 lowering 与执行 | 结构化物理计划、实例值与完成关系、执行帧；接回原 PTTS；迁移旧接口并删除旧核心分支 | 循环、值帧、多资源 PTTS 效果、native 导入、张量条件和外部端点已接通；In-Place-TTT 递推、条件 replay 与 completion AdamW 数学已迁移；已增加并发阶段计划、有界阶段流水、借用 streams、完成汇合及模型提案传输；Qwen kernel 并发契约及完整生产路径迁移待接入 |
| 优化链 | 循环交换、tiling、独立合批、计算块编译及有前提的数学改写；成本选择；数值策略默认允许且可关闭 | 已有矩阵重结合、独立 GEMM 合批、ISL 分块、相邻调用提取、逐元素与 sum/mean Inductor kernel；多个编译块可共同参与选择；纯 export 仿射区域按容量和实测耗时选择结构、调度、数学及编译候选，支持参数化 scan；native bank 复制已纳入容量分析；因果模型合批、更广后端 lowering、完整访存/工作区/效果成本及生产接入待完成 |
| 真实模型验收 | Qwen2.5-0.5B / vLLM prefill 层段、因果 KV、尾块、路由、反馈边、取消、预算、正确性与性能 | 条件 replay 的 Qwen2.5-0.5B 原层段接线已通过；完整 prefill 自动优化及吞吐验收未完成 |
| 教学材料 | 迁移相关文档和 notebook；从真实计划生成示意图；页表、COW、kernel、依赖与编译栈教程 | 最后进行 |

严格算法等价贯穿所有阶段：保留更新次数、读取版本、随机数归属和可观察效果；
本轮不自动替换成 Jacobi 或 bounded staleness。不同浮点求值路径按预先固定的
dtype 容差验证，不能因为测试失败而放宽。合法性、GPU 生存期与实际加速分别
提供证据，不能用 CPU 小程序的通过代替模型验收。
