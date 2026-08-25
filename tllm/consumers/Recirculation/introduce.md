基于论文《Recirculation》（arXiv:2608.17981），本文记录算法背景与实现设计。

> **ONGOING / 实验性实现：当前不是 production-ready 功能。** 本文早期版本中的伪代码不是官方实现，也不能视为
> checkpoint/KV 语义的完整规格。当前以逐 token serial oracle 为 exact 定义：当前
> token 的 normal pass 产生 logits，随后重放 destination 以上层并覆盖上层 KV，供
> 后续 token 使用。Jacobi 和 chunked/blockwise 都是显式近似；wavefront 只有在编译器
> 证明 carried dependency 与 KV overwrite-before-read 顺序得到保持时才是 exact。
> 当前仓库已经实现通用 recurrence IR、Rust schedule compiler、Jacobi graph transform、
> serial oracle、backend fail-closed 准入，以及 vLLM 0.27 MRv1/MRv2 的 eager
> model-region/KV materializer 原型。这些 materializer 级工作不是已验证的 native vLLM
> prefill row/model-Region binding，也不能视为已接管 prefill；当前 Region-owned 路径仍只适用于
> 显式、有限的 CPU request。TP=1、进程内、`enforce_eager=True`、关闭 prefix caching 的配置
> 仅是原型准入范围，不构成 native prefill 正确性或性能结论。Recirculation 的 CUDA Graph、
> EngineCore multiprocessing、decode recurrence、speculative decoding、PP/DP 和通用模型结构
> 仍在开发中。超出范围的配置必须明确拒绝，不能静默降级。

---

## 一、工作概述

### 1. 核心动机
Transformer 的前馈（feedforward）本质决定了其状态更新能力受限于模型深度。深层网络虽然能完成语义消歧（如将 "bank" 理解为"河岸"），但这些状态表征被困在深层，浅层无法访问。当模型后续需要基于该状态做推理时（如回答"河边有ATM吗"），浅层只能看到歧义表征，导致**语境化错误（contextualization error）**。

### 2. 方法本质
**Recirculation** 是一种**纯推理时（inference-time）、零训练（training-free）**的架构增强。它在每个输入 token 的处理步骤结束后，将**深层（source layer）的残差流激活**按极小比例 α "泄漏"回**浅层（destination layer）**，与目标层的原有表征混合，从而使模型成为能迭代更新信念状态（belief state）的动态系统。

### 3. 与相关技术的区别
- **vs. Chain-of-Thought**：CoT 用于复杂推理，Recirculation 用于基础状态跟踪，且无需输出到 token 空间。
- **vs. Looped Transformer**：Looping 的循环仅在**深度**维度（同一 token 内反复执行若干层）；Recirculation 的循环同时在**深度和序列步**两个维度展开，深层状态可跨 token 传递。
- **vs. 传统 RNN/SSM**：无需修改或重新训练模型权重。

### 4. 关键实现细节
| 组件 | 说明 |
|------|------|
| **混合公式** | 目标层输入 = α·Renorm(源层输出) + β·目标层原输入 |
| **归一化** | 将源向量按 L2 范数缩放至与目标向量相同，防止模长爆炸 |
| **混合模式** | 1B 模型：凸组合 β=1−α；4B/12B 模型：非凸组合 β=1（即纯加法） |
| **Ramping** | 对序列前 N 个 token 衰减 α，避免初始无状态信息时引入噪声 |
| **计算代价** | exact serial 在每个 token 后重放上层；decode 同样有额外上层计算，exact wavefront 可通过有序合批降低调度开销 |
| **自适应版本** | 冻结原模型权重，仅轻量搜索/调优 (s, d, α) 三个超参数 |

### 5. 论文报告的实验结果（Gemma3 家族）
- 作者在 10 个语言建模数据集上报告 **perplexity 最高降低 35%**（1B/4B 约 16%，12B 达 35%）。
- 作者在 adaptive Recirculation 的 GSM8k 实验中报告约 **21% 的准确率提升**。
- 论文也展示了 Ministral3、Qwen3、Pythia、Phi2 等模型家族上的 layer sweep；这些结果说明存在候选区域，不等于已经得到跨任务、跨实现的稳定超参数。

### 6. 实验可靠性与代码参考边界

截至 2026-08-22，可核验的论文版本是 2026-08-18 发布的 arXiv v1。它的积极证据包括：

- perplexity 主表覆盖 3 个 Gemma3 尺寸和 10 个数据集，评估 token 数较大；
- layer/alpha 的 tuning 数据与主表 evaluation split 分离；
- 论文公开了 context 长度、样本数量、source/destination 选择和主要归一化设置。

但它仍是很新的预印本，尚无独立复现。强结果主要来自 Gemma3；作者也明确指出 12B 的语言建模 baseline 较弱、下游任务结果并非一致获益、adaptive 结果依赖训练数据、最佳超参数可能随模型和任务变化。论文自身还把 blockwise prefill 列为未实验的未来工作。因此这里把论文数字当作值得复现的证据，而不是已建立的普适结论。

论文正文没有给出 wavefront prefill 算法或其正确性证明。外部 GitHub 中声称的 wavefront 实现可以参考其 tile 划分、kernel 和测量方法，但不能仅凭最终 perplexity 或吞吐判断等价。纳入 tLLM 前至少要在相同权重、token、精度和 attention backend 下，与逐 token serial oracle 比较：normal-pass logits、每个受影响上层的 K/V slot、recurrent residual，以及多请求/分块边界。任一项未对齐，就只能标为近似或实现候选。

---

## 二、算法草图（非执行规格）

以下伪代码仅说明归一化、混合和 ramping；KV 覆盖顺序以 serial oracle 和 recurrence IR 为准。

```python
# ============================================================
#  CONFIGURATION & HYPERPARAMETERS
# ============================================================
class RecircConfig:
    # 模型架构
    num_layers: int          # e.g., 26 for Gemma3 1B
    d_model: int             # hidden dimension
    
    # Recirculation 核心三元组 (需通过轻量 grid search 确定)
    source_layer: int        # s, 深层索引, e.g., 11
    dest_layer: int          # d, 浅层索引, e.g., 4  (必须满足 d < s)
    alpha: float             # α, 源信号混合比例, e.g., 0.15
    
    # 混合模式选择 (论文 4.3 节)
    convex_mixture: bool     # True  -> β = 1 - α  (Gemma3 1B)
                             # False -> β = 1.0     (Gemma3 4B/12B 关键需求)
    
    # Ramping 配置 (论文 4.3 节及 Section 9)
    use_ramping: bool        # True for Gemma3 1B, False for larger models
    ramp_steps: int          # e.g., 10
    ramp_initial_alpha: float# e.g., 0.0, 在 ramp_steps 内线性插值到 alpha
    
    # 归一化配置 (论文公式 2)
    normalize_source: bool   # True: 按 L2 范数将源向量缩放至目标向量范数
                             # False: 恒等映射 (论文附录中对比的基线)


# ============================================================
#  UTILITY: RENORMALIZATION FUNCTION  f(z | d, t)
# ============================================================
def renormalize(source_vec: Tensor[batch, seq, d_model],
                dest_vec:   Tensor[batch, seq, d_model],
                config: RecircConfig) -> Tensor:
    """
    论文公式 (2):
    f(z | d, t) = (||dest||_2 / ||source||_2) * source
    若关闭 normalize，则返回恒等映射 source。
    """
    if not config.normalize_source:
        return source_vec
    
    # L2 范数沿最后一维计算，保持维度用于广播
    norm_dest   = torch.norm(dest_vec,   p=2, dim=-1, keepdim=True)
    norm_source = torch.norm(source_vec, p=2, dim=-1, keepdim=True)
    
    # 避免除零
    scale = norm_dest / (norm_source + 1e-12)
    return source_vec * scale


# ============================================================
#  UTILITY: RAMPING SCHEDULE
# ============================================================
def get_effective_alpha(token_idx: int, config: RecircConfig) -> float:
    """对序列前 ramp_steps 个 token 衰减 alpha。"""
    if not config.use_ramping or token_idx >= config.ramp_steps:
        return config.alpha
    
    # 线性 ramp: 从 0 增长到 alpha
    return config.alpha * (token_idx / config.ramp_steps)


# ============================================================
#  CORE: RECIRCULATION FORWARD PASS  (单步 / 单 token)
# ============================================================
def recirculation_step(
    model: Transformer,           # 原始冻结权重的基础模型
    input_token_id: Tensor[batch, 1],   # 当前步输入 token (自回归时为 1 个)
    kv_cache: List[KVCache],      # 标准 KV Cache, 长度 = num_layers
    carry_state: Tensor[batch, 1, d_model],  # 从上一时间步传入的循环状态
    config: RecircConfig,
    token_idx: int                # 当前在序列中的绝对位置，用于 ramping
) -> Tuple[Tensor, Tensor, List[KVCache]]:
    """
    执行一个 recirculation 更新步。
    
    返回:
      logits:       [batch, 1, vocab_size]
      new_carry:    [batch, 1, d_model]  供下一时间步使用
      updated_kv:   更新后的 KV cache
    """
    # --- 1. 准备输入嵌入 ---
    # x: [batch, 1, d_model]
    x = model.embed(input_token_id)
    
    # --- 2. 在 dest_layer 注入上一时刻的循环状态 ---
    # 论文核心: 浅层接收来自深层的"泄漏"激活
    # 注意: 仅在 dest_layer 处进行混合，而非所有层
    if carry_state is not None:
        alpha_eff = get_effective_alpha(token_idx, config)
        beta = (1.0 - alpha_eff) if config.convex_mixture else 1.0
        
        # 混合前对 carry_state 做归一化
        renormed_carry = renormalize(carry_state, x, config)
        x = alpha_eff * renormed_carry + beta * x
    
    # --- 3. 逐层前向传播 (标准 Transformer block) ---
    #    但需在 source_layer 处提取下一时刻的 carry_state
    current = x
    next_carry = None
    
    for layer_idx in range(config.num_layers):
        # 标准自注意力 + FFN，使用 KV cache 实现单步解码
        current, updated_kv_i = model.layers[layer_idx].forward(
            current,
            kv_cache=kv_cache[layer_idx]
        )
        kv_cache[layer_idx] = updated_kv_i
        
        # --- 4. 在 source_layer 捕获循环信号 ---
        if layer_idx == config.source_layer:
            # 提取残差流输出作为下一时间步的 carry
            # 注意: 这是经过该层计算后的完整残差流表征
            next_carry = current.clone()
    
    # --- 5. 输出头 ---
    logits = model.lm_head(current)
    
    return logits, next_carry, kv_cache


# ============================================================
#  FULL INFERENCE: 自回归生成 (Prefill + Decode)
# ============================================================
def generate_with_recirculation(
    model: Transformer,
    prompt_ids: Tensor[batch, prompt_len],
    config: RecircConfig,
    max_new_tokens: int,
    temperature: float = 1.0
) -> Tensor:
    """
    完整的生成流程。Prefill 阶段必须逐 token 串行执行 recirculation，
    无法像标准 Transformer 那样一次性并行前向传播整个 prompt。
    """
    batch_size, prompt_len = prompt_ids.shape
    device = prompt_ids.device
    
    # --- 初始化 KV Cache (全零/空状态) ---
    kv_cache = [KVCache.empty(config.d_model) for _ in range(config.num_layers)]
    
    # --- 初始化循环状态 (第一个 token 无先前状态) ---
    carry_state = None
    
    generated = []
    
    # ========================================================
    # PHASE 1: PREFILL (强制串行，因 recirculation 依赖时序)
    # ========================================================
    for t in range(prompt_len):
        token = prompt_ids[:, t:t+1]  # [batch, 1]
        
        logits, carry_state, kv_cache = recirculation_step(
            model, token, kv_cache, carry_state, config, token_idx=t
        )
        # prefill 阶段通常不采样，仅更新状态
    
    # ========================================================
    # PHASE 2: AUTOREGRESSIVE DECODE (与标准生成同延迟)
    # ========================================================
    # 使用 prompt 最后一个 token 的 logits 或开始生成新 token
    # 为简化，以下以上一步的 logits 做首次采样
    for t in range(max_new_tokens):
        # 采样当前步 token (greedy 或 temperature)
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]
        generated.append(next_token)
        
        # 自回归步进: 输入刚生成的 token，状态自然延续
        logits, carry_state, kv_cache = recirculation_step(
            model, next_token, kv_cache, carry_state, config, token_idx=prompt_len + t
        )
    
    return torch.cat(generated, dim=1)


# ============================================================
#  ADAPTIVE RECIRCULATION: 超参数搜索 (冻结权重)
# ============================================================
def adaptive_recirculation_search(
    model: Transformer,
    tuning_data: DataLoader,       # 轻量级验证集 (~1.5M tokens)
    model_size: Literal["1B", "4B", "12B"]
) -> RecircConfig:
    """
    论文提出的自适应变体。不修改任何模型权重，
    仅通过小规模验证集搜索最优 (s, d, α) 及混合模式。
    """
    # 预定义搜索空间 (基于论文实验观察)
    if model_size == "1B":
        source_candidates = range(8, 14)
        dest_candidates   = range(2, 7)
        alphas            = [0.04, 0.07, 0.10, 0.15, 0.16]
        convex = True
        ramp   = True
    elif model_size == "4B":
        source_candidates = range(15, 22)
        dest_candidates   = range(6, 12)
        alphas            = [0.10, 0.15, 0.20]
        convex = False      # 4B/12B 关键发现: 需要 β=1
        ramp   = False
    else:  # 12B
        source_candidates = range(30, 38)
        dest_candidates   = range(12, 18)
        alphas            = [0.10, 0.15, 0.20]
        convex = False
        ramp   = False
    
    best_ppl = float('inf')
    best_cfg = None
    
    for s, d, a in product(source_candidates, dest_candidates, alphas):
        if d >= s:
            continue
        
        cfg = RecircConfig(
            num_layers=model.num_layers,
            d_model=model.d_model,
            source_layer=s,
            dest_layer=d,
            alpha=a,
            convex_mixture=convex,
            use_ramping=ramp,
            normalize_source=True   # 搜索阶段固定使用归一化
        )
        
        ppl = evaluate_perplexity(model, tuning_data, cfg)
        if ppl < best_ppl:
            best_ppl = ppl
            best_cfg = cfg
    
    return best_cfg


# ============================================================
#  STANDARD TRANSFORMER BLOCK (供参考的对比实现)
# ============================================================
# 标准前向传播无任何跨步循环，供伪代码完整性参考:
def standard_transformer_forward(model, input_ids, kv_cache=None):
    x = model.embed(input_ids)
    for layer in model.layers:
        x, _ = layer(x, kv_cache=kv_cache)
    return model.lm_head(x)
```

---

## 三、伪代码与论文的逐条对应

| 论文章节 | 伪代码对应 |
|----------|-----------|
| **公式 (1)** | `recirculation_step` 中的 `x = alpha * renormed_carry + beta * x` |
| **公式 (2)** | `renormalize(...)` 函数，按 L2 范数比缩放 |
| **第 4.3 节 非凸混合** | `RecircConfig.convex_mixture = False` 时 `beta = 1.0` |
| **第 4.3 节 Ramping** | `get_effective_alpha(...)` 的线性插值逻辑 |
| **图 3c / 图 4b** | `recirculation_step` 的跨步循环：`carry_state` 从上一时间步的 `source_layer` 传递到当前时间步的 `dest_layer` |
| **第 2 节 "不能并行化"** | `generate_with_recirculation` 的 Prefill 阶段显式 for-loop 逐 token 处理 |
| **自适应变体** | `adaptive_recirculation_search` 冻结模型权重，仅搜索三元组 |

以上伪代码只帮助解释论文中的混合、归一化和 unroll 直觉；它不覆盖 vLLM KV slot、packed batch、抢占恢复或精确执行顺序，不能作为实现规格。

让 Recirculation 的 prefill 并行化，本质上是要打破论文中引入的**跨 token 非线性递推**：

$$h_t = f_t(h_{t-1}, x_t)$$

其中 $h_t$ 是 token $t$ 在 `source_layer` 的残差输出，它必须等 token $t-1$ 算完同一层（或更深）才能确定。标准 Transformer prefill 能并行，是因为所有 token 在同一层的计算只共享 KV-cache（可通过 causal mask 一次矩阵乘法完成）；而 Recirculation 让 token $t$ 的 `dest_layer` 输入直接依赖 token $t-1$ 的 `source_layer` 输出，形成了一条**跨位置的串行关键路径**。

下面给出四个由浅入深的并行化方案，从**近似但完全并行**到**精确但降低串行开销**。

---

## 方案一：Jacobi/Richardson 迭代并行化（显式近似候选）

这条路线把串行递推看作一个**不动点迭代问题**，用多轮并行前向传播逼近它。
`alpha` 较小并不能证明整个 Transformer recurrence 是收缩映射；轮数和误差必须实测。

### 核心洞察
第 $r$ 轮迭代时，**所有 token 的 attention 统一使用第 $r-1$ 轮的 KV cache**。这样当前轮中：
- 没有跨 token 的写后读（RAW）依赖；
- 所有 token 在所有层可以一次性矩阵乘法完成；
- 每轮结束后用新的 `source_layer` 输出更新下一轮要注入的 carry。

这相当于对非线性系统做 **Jacobi 迭代**。

### 精确伪代码

```python
def parallel_prefill_jacobi(
    model: Transformer,
    input_ids: Tensor[batch, seq_len],
    config: RecircConfig,
    max_rounds: int = 3,
    tolerance: float = 1e-3
) -> Tensor[batch, seq_len, vocab]:
    """
    用 Jacobi 迭代实现近似并行的 Recirculation prefill。
    每轮读取上一轮 snapshot；返回有限轮近似及其残差。
    """
    batch, seq_len = input_ids.shape
    device = input_ids.device

    # =====================================================================
    # Round 0: 标准并行 prefill（无 recirculation）
    # 保存每一层后的残差流 states[l][:, t, :] 和对应的 KV cache
    # =====================================================================
    states, kv_caches = standard_parallel_prefill_with_kv(model, input_ids)
    # states: List[Tensor[batch, seq, d_model]], 长度 num_layers
    # kv_caches: List[KVCache], 长度 num_layers

    for r in range(max_rounds):
        # -----------------------------------------------------------------
        # 1. 并行构造所有 token 的 carry_state
        #    carry[t] = states[source_layer][:, t-1, :]  (t>0)
        # -----------------------------------------------------------------
        carry = torch.zeros(batch, seq_len, config.d_model, device=device)
        if seq_len > 1:
            carry[:, 1:, :] = states[config.source_layer][:, :-1, :]
        # token 0 的 carry 保持为零（或任意初始化，不影响收敛）

        # -----------------------------------------------------------------
        # 2. 并行计算 dest_layer 的混合输入
        # -----------------------------------------------------------------
        if config.dest_layer == 0:
            dest_input = model.embed(input_ids)
        else:
            dest_input = states[config.dest_layer - 1]

        alpha_eff = config.alpha  # 可替换为 get_effective_alpha(t) 的广播版本
        renormed = renormalize(carry, dest_input, config)
        beta = (1.0 - alpha_eff) if config.convex_mixture else 1.0
        mixed = alpha_eff * renormed + beta * dest_input

        # -----------------------------------------------------------------
        # 3. 从 dest_layer 到末尾，完全并行前向传播
        #    关键：所有层复用 Round 0（或上一轮）的 KV cache，
        #    因此所有 token 之间无任何跨位置依赖，可一次矩阵乘法完成
        # -----------------------------------------------------------------
        new_states = [None] * config.num_layers
        new_kv     = [None] * config.num_layers

        # 0..dest-1 层不受 recirculation 影响，直接继承
        for l in range(config.dest_layer):
            new_states[l] = states[l]
            new_kv[l]     = kv_caches[l]

        current = mixed
        for l in range(config.dest_layer, config.num_layers):
            # 使用上一轮固定的 KV cache，确保并行性
            current, updated_kv = model.layers[l].forward(
                current,
                past_kv=kv_caches[l]   # Jacobi: 固定旧上下文
            )
            new_states[l] = current
            new_kv[l]     = updated_kv

        # -----------------------------------------------------------------
        # 4. 收敛判断：监测 source_layer 输出的最大变化
        # -----------------------------------------------------------------
        delta = torch.norm(
            new_states[config.source_layer] - states[config.source_layer],
            dim=-1
        ).max().item()

        states, kv_caches = new_states, new_kv

        if delta < tolerance:
            print(f"Jacobi converged at round {r+1}, delta={delta:.2e}")
            break

    return model.lm_head(states[-1])
```

### 收敛边界
- LayerNorm、Softmax 和残差连接不足以证明完整 recurrence 是收缩映射。
- runtime 必须报告轮数、残差和 `converged`；未达到 tolerance 的结果不能标为 exact。

### 复杂度
- 延迟：$O(R \cdot (L \cdot seq))$，其中 $R$ 为实际迭代轮数；不能在没有收敛证据时预设 $R=2$ 或 $3$ 足够。
- 每轮可以使用并行矩阵运算，但总成本、KV snapshot 成本和误差都必须在目标模型与序列长度上实测。

---

## 方案二：Two-Pass 局部重算（候选近似）

简单 two-pass 不能保证数学等价：改变 destination 以上状态会改变后续 token 的 K/V、
source state 和下游层输入。只有完整版本依赖得到保持时才能称为 exact。

### 思路
1. **Pass 1（全并行）**：所有 token 标准前向，保存每一层的残差流和 KV cache。
2. **Pass 2（窄带串行）**：只从 `dest_layer` 重算到 `source_layer`。因为 `dest` 之前的层不受 recirculation 影响；`source` 之后的层虽然会被改变，但它们不决定下一个 token 的 carry（carry 只由 `source_layer` 输出决定）。因此，为了得到正确的下一个 carry，每个 token 只需串行重算 $(s-d)$ 层。

```python
def parallel_prefill_two_pass_exact(model, input_ids, config):
    # Pass 1: 全并行，得到基准
    base_states, base_kv = standard_parallel_prefill_with_kv(model, input_ids)
    
    carry = None
    # 只串行重算窄带 [dest_layer, source_layer]
    for t in range(seq_len):
        # 取出 token t 在 dest_layer 的原始输入
        x = base_states[config.dest_layer - 1][:, t:t+1, :] \
            if config.dest_layer > 0 else model.embed(input_ids[:, t:t+1])
        
        if carry is not None:
            alpha_eff = get_effective_alpha(t, config)
            beta = (1 - alpha_eff) if config.convex_mixture else 1.0
            x = alpha_eff * renormalize(carry, x, config) + beta * x
        
        # 串行通过 dest -> source（通常只占全层 20-30%）
        for l in range(config.dest_layer, config.source_layer + 1):
            # 注意：这里为了精确，K,V 应使用前面 token 已经更新后的版本
            # 因此需要逐 token 更新 KV cache，无法并行
            x, _ = model.layers[l].forward(x, kv_cache=...)
        
        carry = x  # 即 token t 在 source_layer 的输出
    
    # source 之后的层可以在此 token 的 carry 确定后并行补完
    # 但 latency 瓶颈仍是这 O(T*(s-d)) 的窄带串行
```

**局限**：虽然计算量减少了，但关键路径仍是 $O(T)$。适合 `source` 与 `dest` 距离很近的模型配置。

---

## 方案三：Chunk-wise Block-Carry（显式 blockwise 近似）

将序列切成大小为 $C$ 的块。块内**完全并行**做标准 prefill（无细粒度 recirculation），只在块边界传递 carry。

```python
def parallel_prefill_chunked(model, input_ids, config, chunk_size=64):
    seq_len = input_ids.shape[1]
    carry = None
    all_logits = []
    
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = input_ids[:, start:end]
        
        # 块内先并行 prefill（无 recirculation）
        chunk_states, _ = standard_parallel_prefill(model, chunk)
        
        # 将上一块的 carry 注入本块第一个 token 的 dest_layer
        # 然后块内其余 token 不再做细粒度 recirculation
        if carry is not None and config.dest_layer == 0:
            # 直接修改 embedding
            chunk_states[0][:, 0, :] = alpha * renormalize(carry, ...) + beta * ...
        elif carry is not None:
            # 修改 dest_layer 输入后，从 dest 到末尾重新并行前向本块
            x = chunk_states[config.dest_layer - 1][:, 0:1, :]
            x = alpha * renormalize(carry, x, config) + beta * x
            # ... 重新计算 token 0 的后续层
            # 块内其余 token 保持 chunk_states 不变（或也统一用相同 carry）
        
        # 提取本块最后一个 token 的 source_layer 输出作为下一块 carry
        carry = chunk_states[config.source_layer][:, -1, :]
        all_logits.append(chunk_states[-1])
    
    return torch.cat(all_logits, dim=1)
```

- **并行度**：块内完全并行，串行度降为 $O(\lceil T/C \rceil)$。
- **精度损失**：块内 token 之间失去了 recirculation；误差是否可接受以及合适的 $C$ 必须通过 serial differential 和任务指标确定，不能预设为可忽略。

---

## 方案四：Wavefront Pipeline（精确，硬件级吞吐优化）

如果既不想要近似，也不接受分块，可以利用 GPU/TPU 的**细粒度流水线**：把计算图展成 `(layer, token)` 二维网格，按**反对角线 wavefront** 调度。

```
时间步 1:  Token0@L0
时间步 2:  Token0@L1,  Token1@L0
时间步 3:  Token0@L2,  Token1@L1,  Token2@L0
...
时间步 k:  所有满足 layer_idx + token_idx = k 的单元并行执行
```

当计算推进到 `layer = dest` 时，token $t$ 需要等待 token $t-1$ 的 `layer = source` 完成（即 wavefront 的斜率由 `source - dest` 决定）。这不会消除跨 token 关键路径；它只暴露依赖图中合法的并行 antichain。实际吞吐还取决于 tile 大小、attention backend 和 kernel launch 开销，不能从 DAG 宽度直接推断为“接近全并行”。

正确性可以先由 eager executor 按编译结果逐 group 落地；要获得有意义的性能，通常还需要 ordered-row batching、融合 kernel 或等价的设备端调度。优化层不得改变同一张依赖图的可见版本。

---

## 总结建议

| 方案 | 是否精确 | 并行度 | 实现复杂度 | 适用场景 |
|------|---------|--------|-----------|---------|
| **Jacobi 迭代** | 近似（收敛未保证） | 每轮可并行 | 中 | 用户接受误差并提供收敛策略 |
| Two-Pass 窄带 | 未证明等价 | $O(T \cdot (s-d))$ 串行 | 中 | 研究候选 |
| Chunk-wise | 近似（块内无 recirc） | $O(T/C)$ 串行 | 低 | 经实测后的质量/吞吐折中 |
| Wavefront | 依赖保持时精确 | 仍有跨 token 关键路径 | 高 | 先做 parity，再评估融合收益 |

Jacobi 是系统预制、用户显式选择的近似变换；wavefront 是编译器在证明依赖保持后
自动生成的 exact schedule candidate。二者不能作为同一种“等价 schedule”处理。

---

## tLLM 实现 TODO

目标不是为 Recirculation 增加一套定制 runtime，而是用它验证 tLLM 对 prefill 时序依赖的通用表达与编译能力。wavefront 是原依赖图的 exact 调度；Jacobi 是生成 snapshot/round 新图的显式近似变换。后续其他带时序依赖的 prefill 算法也应复用这条路径。

- [x] 先实现与框架无关的逐 token 串行 reference oracle，固定 carry、KV 版本、ramping、混合和负 layer index 的精确定义。
- [ ] 为 Jacobi 建立 differential oracle。将其标记为近似 schedule，并用误差/收敛条件验证；在证明前不宣称有限轮 Jacobi 与串行结果等价，也不把收敛性当作框架假设。
- [x] 为 exact wavefront 建立 `(token, layer)` DAG oracle，明确 recirculation 边、普通层内边、causal KV 边和可并行 antichain。
- [x] 在现有 WorkGraph/placement IR 中设计通用迭代域：至少能描述 `token`、`layer`、`round` 轴、跨迭代 carried value、版本读取和依赖距离。不要加入 `if consumer == recirculation` 一类编译特例。
- [x] 将 Jacobi 表达为上一轮版本的 snapshot/read 与双缓冲更新；将 wavefront 表达为同一图上的拓扑 schedule。由编译器选择或拒绝 schedule，而不是由 consumer 直接驱动 runner 私有循环。
- [ ] 优先复用 `ResidualStream`、现有 locator、row metadata、bundle ownership、placement deadline 和 backend capability；只有现有 port 确实无法表达某个稳定的数据契约时才新增通用 port。
- [x] Recirculation consumer 仅拥有算法配置和纯 tensor 数学：`source_layer`、`dest_layer`、mix/renorm、ramping、Jacobi 容差/轮数。vLLM 对象解析、生命周期和执行 materialization 继续归 backend。
- [x] 首期支持 vLLM 0.27 MRv1/MRv2、TP=1、eager；不通过专用 CUDA kernel 绕开 IR。wavefront 首先用可验证的 eager executor 落地，再由通用 region/fusion 机制优化。
- [ ] 将 model-region 迁移为 worker-side、静态 buffer、按 schedule/shape bucket 捕获的 CUDA Graph executor，并验证 EngineCore multiprocessing；eager executor 保留为 correctness oracle，不作为性能终点。
- [x] 编译准入必须 fail closed：无法证明 KV 版本、carry ownership、row/request 隔离或 schedule 依赖正确时，在 model execution 前拒绝。
- [ ] 测试覆盖串行 oracle、Jacobi 误差界、wavefront 拓扑等价、多请求 packed/chunked prefill、零 token、抢占/恢复和 MRv1/MRv2 对称性。
- [x] 用至少第二种非 Recirculation 的时序 prefill 算法验证迭代域抽象，再稳定公开 API；避免只对一个 consumer 有效的“通用”设计。
