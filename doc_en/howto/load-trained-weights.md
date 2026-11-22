# Loading Trained Weights

Once you have trained a probe, sparse autoencoder (SAE), hypernetwork, LoRA adapter, steering vector, or n-gram lookup table offline in PyTorch or Hugging Face, the next step is loading those weights into `pllm.Workload` state so vLLM's continuous-batching engine can read or further train them efficiently.

Because modern inference engines rely on CUDA graphs to eliminate CPU launch overhead, the Python interpreter does not step in during model execution. pLLM uses declarative state initializers (`init=`) to resolve tensors, budget VRAM statically, and verify contracts during engine construction. This guide provides six recipes covering the most common loading scenarios along with troubleshooting advice.

---

## Choosing an Initializer

Pick the declaration that matches where your weights live and how they are used:

| What You Have | Best For | Declaration & Parameters | Update & Overhead Behavior |
| --- | --- | --- | --- |
| Offline-exported `.safetensors` file | Linear probes, classification heads, prior lookup tables, offline steering vectors | `pllm.InstanceState(fields, init=pllm.Pretrained(path, placement=...))` | Loaded once at engine startup; file hash is recorded in workload identity; large tables can set `placement="host"` |
| `state_dict` weights of an existing `nn.Module` | Small MLP probes, SAE encoders, hypernetworks | `init=pllm.FromModule(factory, pretrained=path)` | Strict parameter matching (`strict=True`); factory may capture only plain scalar configuration |
| Extra tensors inside the served model's checkpoint unused by vLLM | Fast-weight parameters in checkpoints like In-Place TTT | `pllm.InstanceState(fields, init=pllm.ModelCheckpoint({field: name}))` | Read-only; supports `layers=` and `{layer}` patterns; excludes those tensors from vLLM's loader and routes them to your state |
| Values computed dynamically by host code per generation call | Per-batch experimental steering vectors, LoRAs generated per batch by an external hypernetwork | `init=pllm.Imported()` + `llm.generate(batch=pllm.io.StateImport(...))` | `InstanceState` updates and keeps the latest version; `RequestState` takes one row per prompt and writes before step 0 |
| Continuous stream produced by another independent process | Live video streams, exogenous sensor or RL controller features | `pllm.InstanceState(fields, init=pllm.Fed("ordered" \| "latest"))` | At most one per workload; bounded backpressure ring buffer; async cross-process delivery via `llm.feed` |
| Trained low-rank LoRA weights | PEFT adapter deltas targeting attention/MLP projections | `pllm.model.Adapter(state, targets, operator="low_rank", ...)` | Target fields expand automatically as `a_0, b_0, a_1, b_1`; request-level execution stays inside CUDA graphs |

---

## 1. Named Tensors in a safetensors File → `pllm.Pretrained`

### When you have a named tensor dictionary

If your offline-exported weights are saved in a `.safetensors` file (for example, with keys `"w"` and `"b"`), initialize `pllm.InstanceState` or `pllm.RequestState` with `pllm.Pretrained(path)`:

```python
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)

# Prepare a safetensors probe weights file saved from offline training
tmp_dir = tempfile.TemporaryDirectory()
probe_path = str(Path(tmp_dir.name) / "probe.safetensors")
save_file({"w": torch.randn(896), "b": torch.zeros(1)}, probe_path)


class SentimentMonitor(pllm.Workload):
    """Observer workload: compute per-request final hidden-state sentiment scores using an offline-trained probe."""

    probe = pllm.InstanceState(
        {"w": pllm.Field(lambda width, _: (width,)), "b": (1,)},
        init=pllm.Pretrained(probe_path),
    )

    @pllm.export(name="sentiment", max_rows=4, max_cols=1)
    def score(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        w: torch.Tensor = pllm.parameter("probe", "w"),
        b: torch.Tensor = pllm.parameter("probe", "b"),
    ) -> torch.Tensor:
        return torch.sigmoid(h.float() @ w.float()[:, None] + b.float())


print(testing.check(SentimentMonitor(), CONFIG).passed)
```

### Rules and Mechanics

1. **Field names must match file keys exactly**: Every key in the `fields` dictionary must match a tensor name stored in the safetensors file. Missing fields fail immediately during build.
2. **File hash becomes part of workload identity**: `pllm.Pretrained` reads the file at declaration time and folds its SHA-256 hash into the workload's static fingerprint. This guarantees every worker process loads the exact same weights and immediately catches any tampering with the file on disk after startup.
3. **Automatic cast to storage dtype**: Tensors are automatically cast on load to the state's declared physical storage dtype (matching the base model dtype by default, e.g., `bfloat16`). If the state declares `optimizer=` for online training, storage defaults to `float32`.

### Large Lookup Tables in Host Memory (`placement="host"`)

For very large read-only states accessed only by row index (such as tens-of-millions-entry n-gram prior tables or multi-gigabyte retrieval memories), keeping the entire table in GPU VRAM would squeeze the KV cache. Declare `placement="host"` explicitly:

```python
table_path = str(Path(tmp_dir.name) / "table.safetensors")
save_file({"table": torch.randn(16, 151936)}, table_path)


class HostLookup(pllm.Workload):
    memory = pllm.InstanceState(
        {"table": lambda m: (16, m.vocab_size)},
        init=pllm.Pretrained(table_path, placement="host"),
    )

    def lookup(
        self,
        ids: torch.Tensor = pllm.input_tokens(history=1),
        table: torch.Tensor = pllm.parameter("memory", "table"),
    ) -> torch.Tensor:
        # Look up rows by token ID in CPU pinned memory and return the result as a new tensor
        return table[ids[:, -1].remainder(16)].float()

    @pllm.add_logits()
    def add(self, delta: torch.Tensor = pllm.result("lookup")) -> torch.Tensor:
        return delta


print(pllm.explain(HostLookup(), CONFIG).states[0].placement)
```

**Why is host placement restricted to read-only `InstanceState`?**
Host tables only support row-wise gather: each step transfers only the referenced rows across PCIe to the GPU. Training and writes mutate entire fields in device memory—keeping them on host would force transferring the whole table back and forth every step. Meanwhile, `RequestState` is allocated and freed per request rather than representing a single giant table. Thus `placement="host"` is reserved for read-only `InstanceState` with no optimizer and no `carry=`.

Additionally, model forward sites (such as custom operator slots) cannot read host-placed state directly; a helper method must gather the needed rows first and pass them to the model site via `pllm.result` (see [Example 11: n-gram priors / Engram-style lookup](../examples/11_lookup_memory.ipynb) and [State](../guides/state.md)).

---

## 2. `nn.Module` `state_dict` → `pllm.FromModule`

### When you have a PyTorch module and its `state_dict`

If your algorithm's core component is a standard PyTorch `nn.Module` (such as a multi-layer MLP probe or sparse autoencoder) with a saved `module.state_dict()`, use `pllm.FromModule(factory, pretrained=path)`:

```python
class SmallMLP(nn.Module):
    def __init__(self, hidden_size: int, rank: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, rank),
            nn.ReLU(),
            nn.Linear(rank, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Save standard state_dict weights to safetensors
mlp = SmallMLP(896, 4)
mlp_file = str(Path(tmp_dir.name) / "mlp.safetensors")
save_file(mlp.state_dict(), mlp_file)


class ModuleProbe(pllm.Workload):
    probe = pllm.InstanceState(
        init=pllm.FromModule(lambda m: SmallMLP(m.hidden_size, 4), pretrained=mlp_file)
    )

    @pllm.export(name="score", max_rows=4, max_cols=1)
    def score(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        probe: nn.Module = pllm.module("probe"),
    ) -> torch.Tensor:
        return torch.sigmoid(probe(h.float()))


print(testing.check(ModuleProbe(), CONFIG).passed)
```

### Rules and Mechanics

1. **Strict loading (`strict=True`)**: The runtime loads the weight dictionary with `strict=True`, so saved keys must match the module's parameter names identically. Missing or unexpected keys raise `ValueError` at bind time.
2. **Reading the module**: Receive the callable module instance in your method signature with `probe: nn.Module = pllm.module("probe")`. When declared as `RequestState`, each request row automatically executes with that request's own parameter copy.
3. **Factory functions may capture only plain scalars—never module or tensor objects**:
   `factory` must be a pure function or callable referencing only plain scalars (`int`, `float`, `str`).
   **Why?**
   In multiprocessing serving engines (such as multi-GPU tensor parallelism or `ModelConfig(multiprocessing=True)`), the workload must be rebuilt in each independent worker subprocess and allocate VRAM in that worker's own GPU device context. If a factory closure captures an `nn.Module` instance or `torch.Tensor` created in the parent process, cross-process serialization corrupts device contexts and leaks memory. pLLM's static code-fingerprint analysis catches such closures during analysis and rejects them explicitly.

**Self-check**:
When defining `FromModule`, why is `mlp = MyModule(); init=pllm.FromModule(lambda m: mlp)` rejected?
<details><summary>Answer</summary>
Because the closure directly references an <code>nn.Module</code> object already instantiated in the main process. When the engine spawns worker subprocesses, a worker cannot safely reuse a module object bound to another process's device context. Instead, pass initialization config (such as hidden size or rank) as scalars into the factory so each worker constructs the module skeleton in its own context and loads weights strictly from the safetensors path via <code>pretrained=</code>.
</details>

---

## 3. Extra Tensors in the Base Model Checkpoint → `pllm.ModelCheckpoint` and `pllm.io`

### When your checkpoint contains tensors unused by vLLM's model class

Some checkpoints store extra tensors alongside standard weights—for example, the per-layer fast-weight convolution parameters in community In-Place TTT checkpoints. vLLM's model class does not recognize them and ignores them on load, while your workload needs them. `pllm.ModelCheckpoint` excludes those tensors from vLLM's loader and reads them directly into your declared state:

```python
# check-docs: skip: needs an In-Place TTT checkpoint that stores these tensors
ttt_params = pllm.InstanceState(
    {"conv": pllm.Field(lambda d, _: (d, 1, 5))},
    layers=(0, 1, 2, 3),
    init=pllm.ModelCheckpoint({"conv": "model.layers.{layer}.mlp.ttt_conv.weight"}),
)
```

`layers=` lists the layer indices, and `{layer}` in the tensor name template expands for each layer. For a complete example, see `pllm.workloads.InPlaceTTTWorkload` and [Example 08](../examples/08_inplace_ttt.ipynb).

### Rules and Mechanics

1. **Read-only**: `pllm.ModelCheckpoint` is only for `InstanceState` without an optimizer. These values come from the model checkpoint and belong to the model; if you want to train or adapt starting from them, write updates into a separate state.
2. **Do not use it to read weights that vLLM itself needs**: Any tensor listed in `ModelCheckpoint` is excluded from vLLM's own weight loading. Naming weights that vLLM relies on—such as `model.embed_tokens.weight`—leaves the base model missing those weights. If your workload needs a copy of base weights like embeddings or the LM head, read them on CPU with `pllm.io.checkpoint_tensors` below and pass them into state via `pllm.Pretrained` or `pllm.io.StateImport`.

### Reading Checkpoints on the Host CPU

When host code needs to inspect base-model tensors offline before engine startup (for instance, building vocabulary-embedding lookup indices on CPU), use the two lightweight helpers in `pllm.io`:

```python
# 1. Locate the local snapshot directory for the base model (no extra network download)
checkpoint_dir = pllm.io.checkpoint_directory("Qwen/Qwen2.5-0.5B-Instruct")

# 2. Read the requested tensor dictionary from sharded or single safetensors files
tensors = pllm.io.checkpoint_tensors(checkpoint_dir, ("model.embed_tokens.weight",))
embed_weight = tensors["model.embed_tokens.weight"]
print(f"Loaded embed weight with shape: {tuple(embed_weight.shape)}")
```

---

## 4. Per-Call Dynamic Values from Host Code → `pllm.Imported` and `StateImport`

### When you want to swap weights on every `generate` call

If weights are not fixed in a static file on disk, but are instead computed dynamically by host code for each `generate` call (such as testing different steering vectors per call or generating per-batch LoRA deltas from an external hypernetwork), declare `init=pllm.Imported()`:

```python
class Steering(pllm.Workload):
    steer = pllm.InstanceState(
        {"vector": pllm.Field(lambda width, _: (width,))},
        init=pllm.Imported(),
    )
    patch = pllm.model.ActivationPatch("steer", layers=(12,), operator="add", decode=True)


# Inspect the execution plan: ActivationPatch runs inside CUDA graphs with no eager fallback
print(repr(pllm.explain(Steering(), CONFIG).eager_reason))
```

Pass the tensors into `llm.generate` using `batch=pllm.io.StateImport(...)`:

```python
# check-docs: skip: model inference and generate calls require a GPU engine
from vllm import SamplingParams

vector = torch.randn(896)
greedy = SamplingParams(max_tokens=16, temperature=0.0)

with pllm.make_llm(CONFIG, workload=Steering()) as llm:
    # First call: inject a specific steering vector
    outputs_steered = llm.generate(
        ["Today is a great day"],
        greedy,
        batch=pllm.io.StateImport((vector,)),
    )

    # Subsequent calls: omitting batch= automatically reuses the most recently imported version
    outputs_cached = llm.generate(["Today is a great day"], greedy)
```

### Quick Loading from Weights-Only Checkpoints

If your host already has a weights-only `.pt` file saved with `torch.save`, extract tensors safely using `pllm.io.load_tensor_fields`:

```python
weights_path = Path(tmp_dir.name) / "weights.pt"
torch.save({"vector": torch.randn(896)}, weights_path)

# Read the tensor tuple in field order for direct use in StateImport
(imported_vector,) = pllm.io.load_tensor_fields(weights_path, ("vector",))
print(imported_vector.shape)
```

### Rules and Mechanics

1. **`InstanceState` vs. `RequestState` import behavior**:
   - **`InstanceState`**: Pass tensors without a leading row dimension (matching the declared shape exactly). Each call publishes a **new version** of the instance state read by all requests in that call; subsequent `generate` calls without `batch=` continue using that version.
   - **`RequestState`**: Pass tensors with a leading prompt axis (shape `[len(prompts), *shape]`). Values are written into each request's private state copy **before its first forward pass (`request.prepare`)**; all `n` candidate samples generated from the same prompt share that initial row.
2. **Why pass them through `generate(batch=...)`?**
   Under continuous batching and CUDA graphs, no Python code runs on the GPU decode path. Transferring tensors ad hoc inside the inference loop would force host-device synchronization (`host sync`) and break graph replay. Through `StateImport`, the engine stages device copies and registers state versions up front before entering graph execution, keeping the graph pipeline non-blocking.

---

## 5. Continuous Asynchronous Values from an External Process → `pllm.Fed` and `llm.feed`

### When another independent process is streaming values continuously

If inputs are not generated once offline, but instead stream continuously from another process (such as video decoding, audio feature extraction, or an external controller), use `pllm.Fed`. Data pushed by the external process flows through a `@pllm.write` method into state read by `pllm.model.ActivationPatch`, steering generation. A complete built-in example is `pllm.workloads.video.VideoWorkload`: `features` is a fed state, `encode` writes it into `inject`, and the activation patch reads the latest `inject` at every decode step.

```python
from pllm.workloads.video import VideoWorkload

video_config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=1, max_model_len=64, max_num_batched_tokens=64)
print([state.name for state in pllm.explain(VideoWorkload(), video_config).states])  # ['features', 'inject']
```

After starting the engine, obtain a write handle via `llm.feed("features")`, hand it to the producer process to call `publish` per step, and call `end()` and `close()` on completion (see `examples/video_session.py` in the repository for the full program):

```python
# check-docs: skip: needs a CUDA engine and a producer process
import multiprocessing as mp

with pllm.make_llm(video_config, workload=VideoWorkload()) as llm:
    feed = llm.feed("features")
    producer = mp.get_context("spawn").Process(target=produce, args=(feed,))  # produce() calls feed.publish(...)
    producer.start()
    llm.generate(["Describe the scene in this video frame:"], SamplingParams(max_tokens=8, temperature=0.0))
producer.join()
```

### Rules and Mechanics

1. **At most one `Fed` state per workload**:
   **Why?** Each input stream maintains a bounded publish queue: if the producer outpaces the engine, `publish` applies backpressure and waits instead of growing VRAM unboundedly, and the model consumes one item per step in order. If multiple streams advanced independently, "which frame is read on this step" would no longer be unique, so the current implementation supports a single stream. To pass multiple inputs, pack them as multiple fields inside the same fed state.
2. **`"ordered"` vs. `"latest"` consumption modes**:
   - `"ordered"`: Strictly in-order consumption; the engine consumes one published version per step. Ideal for step-aligned temporal streams such as video frame decoding or continuous audio.
   - `"latest"`: Read-latest; if the producer runs faster than model consumption, queued intermediate versions are skipped in favor of the most recently published value. Ideal for real-time sensor telemetry or external control signals.
3. **No optimizer or `carry`**:
   A `Fed` state is a read-only external input stream whose lifecycle is driven entirely by the external producer; it cannot run backward passes or cross-step self-accumulation inside the engine.

---

## 6. Adapting Trained LoRA Weights to `Adapter`

### When you have LoRA deltas for attention or MLP projections

If you have fine-tuned PEFT / LoRA weights (matrices $A$ and $B$) and want to apply them as weight deltas $\Delta W = \text{scale} \cdot B A$ onto the base model's linear projections, declare `pllm.model.Adapter`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptLoRA(pllm.Workload):
    rank: int = 4

    @property
    def lora(self) -> pllm.RequestState:
        # Declare shape functions for the a and b matrices of each target projection
        return pllm.RequestState(
            {
                "a": pllm.Field(lambda inputs, outputs: (self.rank, inputs)),
                "b": pllm.Field(lambda inputs, outputs: (outputs, self.rank)),
            },
            init=pllm.Imported(),
        )

    @property
    def apply(self) -> pllm.model.Adapter:
        # Specify target layer projections; the low-rank operator automatically computes scale * b @ a
        targets = (
            pllm.model.Projection("q_proj", 12),
            pllm.model.Projection("v_proj", 12),
        )
        return pllm.model.Adapter("lora", targets, operator="low_rank", scale=1.0)


print(pllm.explain(PromptLoRA(), CONFIG).states[0].name)
```

### Rules and Mechanics

1. **Field names expand automatically by target order into `a_0`, `b_0`, `a_1`, `b_1`**:
   When `Adapter` binds multiple target projections (such as `q_proj` and `v_proj` above), the shape function passed to `pllm.Field` receives each target projection's own `(in_features, out_features)`. The underlying physical state allocates separate field copies for each projection, named by target index:
   - Target 0 (`q_proj` at layer 12): fields `a_0` and `b_0`;
   - Target 1 (`v_proj` at layer 12): fields `a_1` and `b_1`.
   When loading a safetensors file with `Pretrained`, the file keys must match these expanded names (`a_0`, `b_0`, `a_1`, `b_1`, etc.).
2. **Zero vs. non-zero initialization**:
   - **Offline-trained LoRA**: Load non-zero trained tensors directly via `init=pllm.Pretrained(...)` or `init=pllm.Imported()`. With no optimizer attached, they enter the forward matrix multiplication directly.
   - **Online-trained LoRA (with `optimizer=`)**:
     **Never use all-zero initialization (`pllm.Zeros()`)!**
     **Why?**
     The low-rank delta is $B A$. If both $A$ and $B$ start at 0, backpropagation yields $\nabla_A \propto B^\top = 0$ and $\nabla_B \propto A^\top = 0$. Both gradients stay identically zero forever and the optimizer can never take a first step. pLLM catches this at build time and raises an error. For online training, use `init=pllm.Computed(...)` to initialize one factor (e.g., $A$) with small random values and the other ($B$) to zero.
3. **Execution behavior of `RequestState` vs. `InstanceState`**:
   - `Adapter` backed by `RequestState`: Different requests in the same batch read their own per-request factor copies and **stay completely inside CUDA graphs** with zero CPU scheduling overhead.
   - `Adapter` backed by `InstanceState`: Addresses deltas inside CUDA graphs via a versioned slot page table, also **staying inside CUDA graphs**; if combined with dynamic writes or concurrent imports during requests, version snapshot pages are reserved according to concurrent request capacity (see [Model Edits](../guides/model-edits.md)).

**Self-check**:
If a LoRA adapter targets 4 projections (`q_proj`, `k_proj`, `v_proj`, `down_proj`) with `operator="low_rank", factors=("a", "b")`, which keys must be stored in the safetensors file?
<details><summary>Answer</summary>
It must store 8 tensor keys, indexed in the order targets were passed: <code>a_0, b_0, a_1, b_1, a_2, b_2, a_3, b_3</code>, where <code>a_i</code> has shape <code>(rank, in_features_i)</code> and <code>b_i</code> has shape <code>(out_features_i, rank)</code>.
</details>

---

## Common Errors and Troubleshooting

For exact error messages, root causes, and fixes when patterns on this page are rejected, see [Loading and placing state](../reference/errors.md#loading-and-placing-state) in the Error Catalog.
