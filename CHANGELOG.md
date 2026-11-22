# Changelog

All notable changes to **pLLM (Programmable LLM)** are documented in this file.

## [v0.5.0] - 2026-11-23

`v0.5.0` is a ground-up architectural release transforming pLLM into a declarative programmability layer ("eBPF for vLLM") for stateful and adaptive inference-time algorithms across unmodified **vLLM 0.27–0.34**.

### Highlights

- **Declarative `pllm.Workload` Programming Model**
  - Unified API separating **what to read** (`pllm.hidden`, `pllm.residual`, `pllm.final_hidden`, `pllm.layer_logits`, `pllm.input_tokens`, `pllm.attention`, `pllm.sampled_tokens`), **what state to own** (`pllm.RequestState`, `pllm.InstanceState`, `pllm.PositionState`, `pllm.Field`, `pllm.FromModule`, `pllm.Pretrained`, `pllm.Imported`, `pllm.Fed`), and **where results & updates land** (`@pllm.train`, `@pllm.guide_sampling`, `@pllm.add_logits`, `@pllm.export`, `@pllm.write`).
  - First-class model-site & forward-structure primitives: `pllm.model.ActivationPatch`, `pllm.model.Adapter` (dense, low-rank, per-head overlays on FP16/BF16/GPTQ/AWQ/FP8 projections), `@pllm.train_in_forward`, `@pllm.custom_residual` (including chunked fast-weight state sublayers and multi-stream hyper-connections), `@pllm.custom_attention` (sliding/sink/cross-layer KV layouts and zero-materialization latent MLA), `@pllm.custom_forward`, and parallel associative `pllm.scan`.

- **Rust `pllm_core` Planner, Side-Stream CUDA Graphs & Lock-Free GPU MVCC/RCU**
  - Uniform-dependence longest-path timeline scheduler over 8 step anchors, automatically overlapping token-table lookups (`request.prepare / side`) and per-request autograd + `AdamW` updates (`token.commit / side`) with main-stream forward passes inside captured CUDA graphs.
  - Per-field paged state ledger with Redirect-on-Write (`AdamW` zero-copy updates saving ~30% momentum VRAM), ABA-safe generation counters, two-tier non-blocking CUDA event reclamation, cross-process CUDA IPC tensor borrowing, and exact state rollback on KV preemption.
  - Sustains **97.4%** of native vLLM throughput during per-request online training and **96.3%** with online training + survivor-filtered sampling guidance (`Qwen2.5-7B-Instruct`, 128 concurrent sequences on RTX 4090), and reaches **111.38 tok/s** across 4 concurrent requests on 4B In-Place TTT (2.03× official single-stream loop).

- **Unified Online Serving (`pllm serve`, `make_async_llm`) & vLLM `0.27–0.34` Ecosystem**
  - OpenAI-compatible HTTP SSE server (`pllm serve`) and embedded async Python engine (`pllm.make_async_llm`) with isolated `EngineCore` worker processes, disconnect cancellation, multi-turn conversation state routing (`vllm_xargs.pllm_route`), live `import_state()`, host batch training (`CausalTrainBatch`, `WeightedRowBatch`, `WeightedCEBatch`), and `generate_feedback()`.
  - Verified admission across **15 decoder families** (`Qwen2/2.5`, `Qwen3/MoE`, `Qwen3.5/MoE`, `DeepSeek V2/V3`, `Llama`, `Mistral`, `Mixtral`, `Gemma 2/3`, `Phi-3`, `GLM-4`, `gpt-oss`), quantized checkpoints (`GPTQ`, `AWQ`, `FP8`), and single-node Tensor Parallelism (`tensor_parallel_size > 1`).

- **CPU Pre-Flight Contract Verification & Packaged Coding-Agent Skill (`pllm skill`)**
  - Zero-GPU execution planner (`pllm.explain`) and metamorphic multi-tenant verifier (`pllm.testing.check`) catching hidden host syncs, row-permutation sensitivity, cross-tenant batch leaks, chunked-prefill drift, and preemption bugs on CPU in under two seconds.
  - Built-in CLI skill installer (`pllm skill --install claude|codex|cursor`) shipping generated API signatures, error catalog, and HF/PEFT migration checklists inside the Python package.

- **Complete Bilingual Documentation, Reference Workloads & Illustrated Deep-Dive Series**
  - 13 maintained reference workloads in `pllm.workloads` (`ESampWorkload`, `InPlaceTTTWorkload`, `DoLaWorkload`, `ReFTWorkload`, `OutputLoRAWorkload`, `PrefillAdaptersWorkload`, `RecirculationWorkload`, `StepLoopWorkload`, `RewardHeadWorkload`, `TLMWorkload`, `OverRIDEWorkload`, `FTTTWorkload`, `DummyWorkload`).
  - 46 bilingual (EN & 中文) human documentation pages, tutorials (`00–05`), and algorithm examples (`06–13`) in `doc_en/` and `doc/`, plus 10 bilingual systems deep-dive essays in `blog/` illustrated with 68 vector SVG diagrams (also live at [lineshogan.github.io/pllm](https://lineshogan.github.io/pllm/)).

---

### 中文版本更新摘要（v0.5.0）

- **全新声明式 `pllm.Workload` 编程模型**：将推理期算法统一抽象为“读什么内部表示、持有什么请求/实例级状态、结果与梯度作用于何处”，覆盖激活引导、探针导出、每请求动态 LoRA、解码期在线蒸馏（ESamp）、前向内快权重训练（In-Place TTT）、自定义残差/注意力/KV 布局与宿主强化学习闭环（TTRL / GRPO）。
- **Rust 核心规划器、旁路流静态图重叠与无锁显存 MVCC/RCU**：基于 8 锚点一致依赖最长路调度器与分页状态账本，支持写时重定向 `AdamW`、两级非阻塞流事件回收与 KV 抢占状态回滚；在 7B 模型 128 并发下边训边引维持 **96.3%** 原生 vLLM 吞吐。
- **兼容 vLLM `0.27–0.34`、15 类主流架构与 OpenAI 在线流式服务**：新增 `pllm serve`（支持 SSE 流式输出与 `vllm_xargs` 多轮会话状态路由）与 `pllm.make_async_llm`，覆盖 DeepSeek V2/V3（含零解压潜空间 MLA）、Qwen2/3/3.5、Llama 等 15 类架构及 GPTQ/AWQ/FP8 量化与单机多卡张量并行。
- **CPU 零 GPU 静态契约核验与 `pllm skill` 编码智能体集成**：启动前 2 秒内在 CPU 上静态拦截跨租户串味、隐式同步与抢占失配；内置 `pllm skill --install claude|codex|cursor` 一键导出规范与旧项目迁移清单。
- **完整中英双语文档体系与 68 张矢量图解专栏**：提供全部 46 篇面向人类开发者的中英双语教程、指南与参考手册，以及 10 篇配完整架构/显存矢量插图的推理系统深度拆解文章（同步上线于 [lineshogan.github.io/pllm](https://lineshogan.github.io/pllm/)）。

---

## [v0.1.0] - Initial Prototype Release

- Initial prototype runtime and Exploratory Sampling (ESamp) decoding implementation on vLLM 0.27.
