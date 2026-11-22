# Online Training

In standard inference frameworks, LLM inference is strictly designed as a read-only forward pass. Yet in many frontier algorithms, the model needs to **learn while generating**: each request carries a small set of private parameters (such as a shallow-to-deep distillation network, a test-time training (TTT) adapter, or a shared prediction head), computes a loss from forward signals on every generated token, updates its parameters, and immediately uses those updated parameters for subsequent tokens.

Implementing this by hand in an inference engine heavily optimized with continuous batching and CUDA graphs is extremely difficult:
- Continuous batching concatenates the current tokens of different requests along dimension 0, where every row has a distinct lifetime and ownership;
- CUDA graphs require fixed memory addresses and operator execution topologies, so traditional dynamic computation graphs cannot be replayed directly;
- Autograd is disabled by default during inference, and manually calling backward outside the model breaks the inference pipeline;
- If backward passes and optimizer updates block serially before the next forward pass, inference throughput drops sharply.

pLLM lets you define online training purely declaratively: declare an optimizer on the state (such as `optimizer=pllm.AdamW(...)`), attach `@pllm.train(state=...)` to a method, and return a per-row loss vector. pLLM handles padding masking, per-state gradient reduction, optimizer steps, and publishing new versions, while capturing the entire training pipeline into a CUDA graph on a side CUDA stream overlapped asynchronously with the backbone forward pass.

---

## Minimal Example: Online Shallow Distillation

Below is a complete, minimal runnable workload `ShallowDistiller` (from the tutorials and reference implementations): each request owns a private small network `distiller`, trains it during decode to predict final-layer hidden states from layer-0 hidden states, and uses `@pllm.guide_sampling` to turn shallow predictions into sampling guidance.

```python
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


@dataclass(frozen=True)
class ShallowDistiller(pllm.Workload):
    beta: float = 0.5
    rank: int = 4

    # What to store: each request owns a private small network and an independent AdamW optimizer
    @property
    def distiller(self) -> pllm.RequestState:
        return pllm.RequestState(
            init=pllm.FromModule(
                lambda m: nn.Sequential(nn.Linear(m.hidden_size, self.rank), nn.Linear(self.rank, m.hidden_size))
            ),
            optimizer=pllm.AdamW(1e-3),
        )

    # State eligibility check: enable sampling guidance only after a state has received at least one update
    def trained(self, versions: torch.Tensor = pllm.version("distiller")) -> torch.Tensor:
        return versions > 0

    # Training method: return a per-row scalar loss of shape [rows]
    @pllm.train(state="distiller")
    def loss(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        deep: torch.Tensor = pllm.hidden(-1),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return (f(shallow.float()) - deep.float()).square().mean(-1)

    # Dynamically read parameter from the workload instance
    def strength(self) -> float:
        return self.beta

    # Sampling guidance: project shallow predictions to vocabulary space and steer away from predictable features
    @pllm.guide_sampling(beta=strength, rows="trained")
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return f(shallow.float())


check_result = testing.check(ShallowDistiller(), CONFIG, neutral=ShallowDistiller(beta=0.0))
print("publish_lag" in check_result.passed)
```

When you run the code above, pLLM automatically performs the following under the hood:
1. **Automatic masking and isolation**: masks out inactive padding slots so each request updates only its own parameters from its own inputs;
2. **Local differentiation**: builds a first-order computation graph inside the method and backpropagates from the loss to the `distiller` parameters;
3. **Optimizer step**: executes an AdamW update for each request's state;
4. **Side-stream publishing and overlap**: schedules the training work after sampling on a side CUDA stream and captures it as a CUDA graph without stalling the main stream.

---

## Return Value Contract: Per-Row Loss `[rows]`

The core contract of an online training method is: **it must return a 1D tensor of shape `[rows]`, where each entry is the loss for the corresponding row at the current step**.

Do not collapse the row axis inside the Python method body (for example, by calling `.mean()` or `.sum()` to return a scalar).

- **Why**: Under continuous batching, dimension 0 of the input tensor dynamically packs tokens from all active requests in the current batch. The framework needs each row's individual loss value to filter out invalid padding rows according to the current request topology, perform reduction independently per state, and preserve request isolation. If you take a global mean in Python, losses from different requests get mixed together before differentiation, completely breaking mathematical isolation across requests.
- **Consequence of violation**: If the method returns a scalar, `testing.check` and the executor immediately raise an error during shape tracing: `training methods return one loss per row; pLLM reduces the selected rows per state`.

---

## Optimizers on State and Execution Cadence

In pLLM, **the optimizer is declared directly on the state, not on the decorator**:

```python
head = pllm.RequestState(
    {"w": lambda m: (m.hidden_size,)},
    optimizer=pllm.AdamW(1e-3),
)
```

- **Why**: An optimizer's first moments, second moments, and step counter are strictly tied in lifetime to the state itself. For `pllm.RequestState`, when a request finishes and exits, both its parameters and its optimizer momentum slots are reclaimed together from the memory pool; for a cross-request `pllm.InstanceState`, optimizer moments persist globally across the engine's lifetime.
- **Matching rule**: When training on running request rows (without host batch fields), the state must declare `optimizer=pllm.AdamW(lr)`. Omitting the optimizer or declaring an incompatible type raises an error at workload construction time.
- **Which state to train**: Request-row training usually targets a `pllm.RequestState` (one copy per request). You can also train a shared `pllm.InstanceState` across all requests on request rows (like `pllm.workloads.OverRIDEWorkload`): at each step, all requests' rows together apply one update to that single shared state, so one request's results are affected by other requests in the same batch.

### Engine Steps and Chunk Boundaries

For execution cadence, pLLM follows a strict rule: **each state executes one and only one optimizer update per engine step**.

- **Decode phase**: Each active request generates 1 token per step (1 input row), so each request state advances its optimizer once per step.
- **Prefill phase (Prompt rows)**: When chunked prefill is enabled, a long prompt is split by the scheduler into multiple chunks across steps. The chunk processed in each engine step participates as a unit in that step's training method, and the state updates once per chunk. Consequently, the number of training steps and update trajectory during the prompt phase naturally depend on the scheduler's chunk boundaries.

> [!NOTE]
> If your algorithm (such as In-Place TTT) mathematically requires fixed-size token chunks by model sequence position (e.g., updating strictly every 16 tokens) regardless of dynamic scheduler chunking, use `@pllm.train_in_forward(chunk=...)` to inline training into the model's forward projections instead. See [Editing Models](model-edits.md).

---

## Training Reduction: `mean` vs. `sum`

When defining online request training with `@pllm.train(state=..., reduction=...)`, the method returns a per-row loss vector of shape `[rows]` for the current batch's request rows. pLLM's stepping model is: **each state performs one optimizer update per engine step**, first reducing the rows selected for that state in the current step into a scalar loss.

If you are writing the most common case—decode-phase training isolated by default `request_id`—each request has only 1 row per engine step (the number of selected rows for that state is $n_s = 1$). Taking the mean vs. the sum of 1 number is identical, so `reduction="mean"` and `reduction="sum"` behave identically and you don't need to worry about the difference.

Only when **the same state maps to multiple rows within a single engine step** does `reduction` determine the relative weights among rows of that state and the gradient magnitude across steps:
- **Prompt rows** containing multiple tokens (prefill phase);
- Multiple requests mapped to the same state via shared `key_fields`;
- All requests sharing a single `InstanceState`;
- Any other case that produces multiple rows for the same state in one step.

### Mechanics and How to Choose

`@pllm.train`'s `reduction` reduces only the selected rows *within* each state at the current step—**both options keep different states completely independent**:
- `reduction="mean"` (default): averages the $n_s$ rows selected for state $s$ in this step, $\frac{1}{n_s}\sum_{r \in s} \ell_r$.
- `reduction="sum"`: sums the rows selected for state $s$ in this step, $\sum_{r \in s} \ell_r$. Each state's reduced scalar loss is then **summed** across states before backpropagation; rows from different states are never averaged together.

Which one to pick depends on your optimizer and how multiple rows relate to one another:

1. **With `pllm.AdamW`, prefer `"mean"` (the default)**
   AdamW is invariant to global constant scaling, but it is sensitive to **step-to-step jumps in gradient magnitude**. During a prefill step, a request may enter with $P$ prompt rows at once; with `"sum"`, the prefill step's gradient magnitude is roughly $P\times$ larger than a single-row decode step, inflating AdamW's second-moment estimate $v$ by roughly $P^2$ and suppressing the effective step size for many subsequent decode steps. `"mean"` removes the dependence on the row count $n_s$ within a step, keeping prefill and decode gradient scales consistent.
2. **When you want each token's contribution to stay constant, pick `"sum"`**
   `"mean"` divides by the step's row count $n_s$ for that state; if you want each token's contribution to depend solely on its own gradient rather than being diluted by $1/n_s$ based on how many rows happen to arrive in the same batch, choose `"sum"`.
3. **For online group training with shared `key_fields`, pick `"sum"`**
   When a group of sibling sampled requests shares one state, different samples often finish at different steps. Under `"mean"`, as some samples finish and drop out, the denominator $n_s$ shrinks, passively amplifying the per-token weight of the remaining active samples; `"sum"` ensures every surviving token always contributes a constant unit of gradient.

### Common Normalization Recipes

Using `@pllm.train`'s return value and arguments, you can directly compose common training variants:

- **Per-row weighting** (e.g., advantages, confidences, gating weights $w_r$): return `weight * loss` ($w_r \cdot \ell_r$) directly from the training method, paired with `reduction="mean"` or `"sum"`.
- **Dr.GRPO-style constant-length normalization**: when you want to divide by a fixed constant $L_{\max}$ instead of the dynamic per-step row count $n_s$, use `reduction="sum"` and multiply the returned loss by $1 / L_{\max}$.
- **Excluding specific rows: prefer `rows=`, not zero weights with `mean`**: under `reduction="mean"`, rows returning `0.0 * loss` **still count toward the denominator $n_s$**, diluting the gradients of active rows; `rows=` removes the row from both numerator and denominator. Note that `rows=` is only supported when request training is combined with model effects or `@pllm.add_logits`; standalone request training and training alongside `@pllm.guide_sampling` update every row of each selected state. In those two cases, to exclude specific rows, use `reduction="sum"` with zero weights: zero-weight rows contribute nothing to the sum and do not alter other rows' weights.
- **Temporal decay and momentum**: cross-step exponential moving averages (EMA) or decay belong to the optimizer (such as AdamW's $\beta_1$ or `weight_decay`) or an explicit regularization term—do not try to approximate them through intra-step reduction.

Below is a complete example using `reduction="sum"` for constant-length normalization (Dr.GRPO style):

```python
@dataclass(frozen=True)
class LengthNormalizedTrainer(pllm.Workload):
    max_len: float = 256.0

    predictor = pllm.RequestState(
        {"w": lambda m: (m.hidden_size,)},
        optimizer=pllm.AdamW(1e-3),
    )

    @pllm.train(state="predictor", reduction="sum")
    def loss(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        deep: torch.Tensor = pllm.hidden(-1),
        w: torch.Tensor = pllm.parameter("predictor", "w"),
    ) -> torch.Tensor:
        # Constant-length normalization: with reduction="sum", divide each row's loss by fixed max_len
        pred = shallow.float() * w
        return (pred - deep.float()).square().mean(-1) / self.max_len


print("chunking / tokens=" in testing.check(LengthNormalizedTrainer(), CONFIG).passed)
```

---

## Zero Gradient Does Not Mean No Update

In local PyTorch training scripts, developers sometimes assume that if a parameter's gradient is 0, the parameter stays unchanged on that step. **In AdamW, however, a zero gradient still advances the update**:

1. **Step counter advances**: the optimizer's internal step increments $t \leftarrow t + 1$;
2. **Momentum decays**: first moment $m \leftarrow \beta_1 m$ and second moment $v \leftarrow \beta_2 v$ continue decaying;
3. **Weight decay still applies**: even with zero gradient, the weight decay term shrinks the parameters:
   $$w \leftarrow w - \eta \cdot \lambda \cdot w$$

- **Important**: If you want to completely "freeze" or skip updating a state on certain steps, **do not multiply the loss or gradient by 0 inside the loss method**. Exclude that request or state at the scheduler selector level (such as `rows=` or `update_rows=`).

---

## Row Selectors: `rows=` and `update_rows=`

`@pllm.train` provides two optional selector parameters:
- `rows="selector_method"`: names a helper returning a boolean mask `[rows]`; the method still runs on every row, and only rows where the mask is `True` are included in the loss reduction;
- `update_rows="selector_method"`: names a helper returning a boolean mask that explicitly authorizes which rows' corresponding states may be written by the optimizer.

### Why Standalone Request Training Rejects Selectors

If you try to use `rows=` or `update_rows=` on standalone request training (with no other effects) or on a workload that only has `@pllm.guide_sampling`, workload construction fails immediately with:

```text
rows= and update_rows= on request training require a model effect without sampler guidance, or @pllm.add_logits; standalone training and @pllm.guide_sampling update every selected state, so remove rows= and update_rows=
```

- **Why**: Standalone request training and training alongside `@pllm.guide_sampling` follow the path designed for algorithms like ESamp where "every request learns on every step": at each step after sampling, each selected state gets one AdamW update, publishes a new version, and overlaps with the next forward pass. Skipping updates per row requires "conditional publishing," which is not implemented on that path, so pLLM rejects it at build time instead of silently ignoring your mask. When combined with `@pllm.add_logits` or model edits (`Adapter`, `ActivationPatch`, etc.), training is scheduled as a conditional update, and `rows=` / `update_rows=` are supported.
- **Workarounds**:
  1. **Move filtering to the sampler side**: let the training method run unconditionally, and attach the selector to sampling guidance, e.g., `@pllm.guide_sampling(beta=strength, rows="trained")`;
  2. **Use `reduction="sum"` with zero loss**: as noted above, multiply the loss of non-contributing rows by 0. While this does not stop weight decay, it mathematically zeroes those rows' gradient contributions without distorting the reduction denominator of other rows.

---

## Side-Stream Overlap and `publish_lag`

### Why Default `publish_lag=1` Boosts Throughput

If a token generated at step $n$ requires its updated weights to take effect immediately at the start of step $n+1$'s forward pass (i.e. `publish_lag=0`), then every forward step must block until the backward pass and AdamW kernels finish completely, introducing large synchronization bubbles on the GPU timeline.

pLLM defaults to `publish_lag=1`:
1. Once step $n$'s sampling finishes, the training task is dispatched asynchronously to a separate side CUDA stream (`side` stream);
2. Step $n+1$'s model forward **reads the stable state version already published by step $n-1$**, without waiting for step $n$'s training to finish;
3. The new weights computed at step $n$ take effect in step $n+2$'s forward pass.

Inside the training method itself, because it runs on the side stream, it reads the latest current state version; in the forward pass and sampler, by contrast, methods read the published stable version.

How much overlap saves depends on what fraction of a step training occupies. Measured on an RTX 4090 with `ESamp` (rank 4, 128 concurrent sequences = 8 prompts × 16 samples, 256 tokens, vLLM 0.32.0, CUDA graphs, median of 3 rounds after 1 warmup round):
- **Qwen2.5-7B**: with online training only ($\beta=0$), `publish_lag=1` achieves **97.4%** of native vLLM throughput vs. **97.2%** for `publish_lag=0`; with training + guided sampling ($\beta=0.1$), **96.4%** vs. **96.1%**. Training takes only ~0.6 ms of GPU time per ~21.7 ms step, so waiting for it is within measurement noise.
- **Qwen2.5-0.5B**: with online training only, `publish_lag=1` achieves **91.9%** vs. **89.8%** for `publish_lag=0`. When a step is only ~3 ms long, the cost of serially waiting for training becomes clearly visible.

### Setting `publish_lag=0` and Its Cost

If your algorithm strictly requires every generated token's update to be visible immediately on the very next step, disable the lag explicitly in config:

```python
strict_config = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_num_seqs=8,
    execution=pllm.ExecutionOptions(publish_lag=0),
)
```

- **Cost**: In `pllm.explain`, the training task's lane changes from `side` to `main`, the execution note reports `updates serialize before the next forward`, and throughput drops due to serial waiting.
- **Contract check**: Running `pllm.testing.check` on CPU automatically verifies the state version read by every task and enforces compliance with the `publish_lag` publishing contract.

---

## First-Order Gradient Edges: `@pllm.differentiate` and `pllm.result`

In many advanced architectures, a `@pllm.train` method does not consume raw activations directly; instead, it consumes outputs produced by an upstream helper method (via `pllm.result("upstream_name")`).

To allow gradients to backpropagate from the downstream loss to the upstream operation, the upstream method must declare `@pllm.differentiate`, and the downstream training method must also preserve the derivative connection for the corresponding input parameter:

```python
@dataclass(frozen=True)
class ChainedDistiller(pllm.Workload):
    head = pllm.RequestState(
        {"w": lambda m: (m.hidden_size, 64)},
        optimizer=pllm.AdamW(1e-3),
    )

    # Upstream projection: runs in the same phase as training; keeps derivative edge on shallow,
    # and parameters=True lets parameter w receive gradients
    @pllm.on(pllm.Phase.AFTER_SAMPLE)
    @pllm.differentiate(inputs=("shallow",), parameters=True)
    def project(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        w: torch.Tensor = pllm.parameter("head", "w"),
    ) -> torch.Tensor:
        return torch.bmm(shallow.float().unsqueeze(1), w).squeeze(1)

    # Downstream training loss: reads result via pllm.result and declares inputs=("features",) to keep the derivative edge
    @pllm.train(state="head")
    @pllm.differentiate(inputs=("features",))
    def loss(
        self,
        features: torch.Tensor = pllm.result("project"),
        deep: torch.Tensor = pllm.hidden(-1),
    ) -> torch.Tensor:
        return (features - deep[:, :64].float()).square().mean(-1)


print("request isolation" in testing.check(ChainedDistiller(), CONFIG).passed)
```

### Derivative Boundaries and Contract Constraints

- **Same phase as the training method**: gradients propagate only within a single invocation window, so the upstream helper must use `@pllm.on(pllm.Phase.AFTER_SAMPLE)` to run after sampling where training runs; otherwise build fails with `the @pllm.train loss of ... must differentiate only its own state`.
- **Local first-order autograd within one call**: `@pllm.differentiate` is strictly confined to the local task chain of a single engine step. It **never backpropagates through vLLM's base model weights** and never spans across step boundaries.
- **Parameter freeze control**: `parameters=False` (default) differentiates only with respect to the input tensors listed in `inputs`, so parameters read by the helper receive no gradients; `parameters=True` allows parameters to receive gradients and update.
- **Sink contract**: every intermediate operation marked with `@pllm.differentiate` must eventually feed a loss method bound by `@pllm.train`. An orphaned differentiable helper is rejected at build time: `every @pllm.differentiate result must feed a @pllm.train loss`.

---

## Cross-Step State Carry: `pllm.carry`

Some algorithms need to pass plain tensor memory across time steps (for example, the previous step's prediction or a moving statistic of past activations) that **needs neither first-order derivatives nor AdamW momentum slots**.

You can declare these fields via `carry=` on the state and capture them across steps with `pllm.carry` in the training method:

```python
@dataclass(frozen=True)
class CarriedDistiller(pllm.Workload):
    head = pllm.RequestState(
        {"w": lambda m: (m.hidden_size,)},
        optimizer=pllm.AdamW(1e-3),
        # Declare carry field: no momentum slots, untouched by AdamW
        carry={"prev_shallow": lambda m: (m.hidden_size,)},
    )

    @pllm.train(state="head")
    def loss(
        self,
        # Wrap input with pllm.carry: on publish, writes the last row of this request's input to the carry field
        shallow: torch.Tensor = pllm.carry(pllm.hidden(0), state="head", field="prev_shallow"),
        deep: torch.Tensor = pllm.hidden(-1),
        w: torch.Tensor = pllm.parameter("head", "w"),
        # Read the value published at the previous step (zero-initialized on first admission)
        prev: torch.Tensor = pllm.parameter("head", "prev_shallow"),
    ) -> torch.Tensor:
        pred = (shallow.float() + prev.float()) * 0.5 * w
        return (pred - deep.float()).square().mean(-1)


print("publish_lag" in testing.check(CarriedDistiller(), CONFIG).passed)
```

- **Semantics**:
  - Write: when the training update publishes, `pllm.carry` captures the last selected row of that input for the current request and writes it to the carry field;
  - Read: `pllm.parameter(state, field)` reads the value published by the previous step. Because initial state is zero-initialized, the value read on the first step is strictly zero.

---

## Device-Side Counter Returns: `pllm.UpdateOutput`

Ordinary `@pllm.train` methods return a per-row loss `torch.Tensor`. However, in complex or gated training logic, you may need to report to the engine whether state changed and how many rows were actually trained—without ever triggering a GPU-to-CPU synchronization stall.

In that case, have the method return `pllm.UpdateOutput`:

```python
# check-docs: skip: the return statement of a training method body
return pllm.UpdateOutput(
    output=loss,                                          # [rows] loss tensor
    changed=loss.new_ones((), dtype=torch.bool),          # 0-d bool tensor: whether state changed
    update_count=loss.new_ones((), dtype=torch.int64),    # 0-d int64: optimizer steps advanced
    trained_rows=loss.new_full((), loss.shape[0], dtype=torch.int64),  # 0-d int64: rows trained
)
```

`changed`, `update_count`, and `trained_rows` must be **0-dimensional tensors (0-d Tensor) on the same device as `output`**, never native Python scalars (`float` or `int`), ensuring zero host synchronization inside CUDA graphs.

---

## Observing and Verifying That Training Runs

Once the engine is running with your workload, how do you verify that online training actually happened and didn't fall back to slow eager mode?

### 1. Runtime Diagnostics: `llm.diagnostics()`

Calling `llm.diagnostics()` returns a lightweight snapshot of the engine's runtime counters, with four key fields:

```python
# check-docs: skip: requires CUDA GPU engine
from vllm import SamplingParams

sampling = SamplingParams(n=4, max_tokens=32, temperature=0.8, top_p=0.95, seed=0)
with pllm.make_llm(CONFIG, workload=ShallowDistiller()) as llm:
    llm.generate(["Explain gradient descent:"], sampling)
    report = llm.diagnostics()

print(f"loss_count: {report.loss_count}")
print(f"update_count: {report.update_count}")
print(f"trained_rows: {report.trained_rows}")
print(f"eager_reason: {report.eager_reason!r}")
```

- `loss_count`: total number of loss evaluations executed;
- `update_count`: total number of optimizer steps successfully applied;
- `trained_rows`: total number of token rows that participated in gradient updates;
- `eager_reason`: if `""` (empty string), both the model forward pass and online training ran completely inside CUDA graphs; if non-empty, it identifies the exact operator or declaration that forced eager execution.

### 2. Local Static Contract Verification: `testing.check`

On CPU—without any real GPU—`pllm.testing.check` runs comprehensive static contract checks on training workloads:
- Perturbs request row ordering (`row permutation`) to catch illegal cross-row reductions;
- Perturbs neighboring requests (`request isolation`) to verify strict state isolation across requests;
- Checks the `publish_lag` version contract to ensure no operator reads uncommitted versions before side-stream publication;
- Checks input equivalence across prompt chunking and decode steps.

---

## Host Batch Training

If you do not need per-request online learning during decode, and instead want to fine-tune on offline or semi-online batches supplied by the host (such as full rollouts collected for RL), use the host batch training path. See [Host Loops and Batch Training](host-loops.md).

Host training methods that read `pllm.batch(...)` fields (such as `WeightedRowBatch` or `CausalTrainBatch`) **must perform their own reduction and return a scalar loss directly**; declaring a non-default `reduction=` on such a method is rejected at build time.

---

## Common Errors and Troubleshooting

For exact error messages, root causes, and fixes when declarations on this page are rejected, see the [State and training](../reference/errors.md#state-and-training) and [Phases: when a value exists](../reference/errors.md#phases-when-a-value-exists) sections of the Error Catalog.

---

#### Self-Check Questions

1. **Question**: During decode training, if you want certain tokens not to produce gradient updates, is it valid to simply return `0.0 * loss` in the training method? Why or why not?
   <details><summary>Answer</summary>
   No. For two reasons: first, with the default <code>reduction="mean"</code>, zero-loss rows still count toward the denominator $n_s$, diluting the gradients of active rows sharing the same state in that step; second, and more importantly, in AdamW a zero gradient still advances the step counter $t$, decays momentum, and applies weight decay, so parameters still change. To truly skip updates, exclude rows via the <code>rows=</code> selector where conditional updates are supported, or use <code>reduction="sum"</code> with zero weights to avoid diluting the denominator.
   </details>

2. **Question**: Why does the default <code>publish_lag=1</code> mean that step $n$'s training result is not read by the model forward until step $n+2$? What happens if you force <code>publish_lag=0</code>?
   <details><summary>Answer</summary>
   Because online training is scheduled asynchronously on the side CUDA stream (<code>side</code> lane), overlapping step $n$'s parameter update with step $n+1$'s model forward on the GPU timeline. To avoid stalling step $n+1$'s forward while waiting for backward and AdamW to finish, step $n+1$'s forward reads the stable version already published before step $n$ started (i.e., step $n-1$'s update), so step $n$'s new version takes effect at step $n+2$. Setting <code>publish_lag=0</code> forces step $n+1$'s forward to wait serially for step $n$'s training to finish, breaking pipeline overlap and introducing timeline bubbles—most noticeable on shorter steps (e.g., on 0.5B at 128 concurrency with training only, throughput drops from 91.9% to 89.8% of native; on 7B where a step is much longer, the difference is near measurement noise).
   </details>
