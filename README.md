# PLLM

Programmable tensor workloads alongside LLM inference. Algorithms declare math,
dataflow, state and effects; Rust signs one execution plan, and the shared runtime
owns versions, publication, capture and completion.

## Run a Workload

```python
from vllm import SamplingParams
from pllm import ExecutionOptions, ModelConfig, make_llm
from pllm.workloads import ESampRequestConfig, ESampRequestWorkload

config = ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", execution=ExecutionOptions(capture=True))
with make_llm(config, workload=ESampRequestWorkload(ESampRequestConfig(rank=2))) as llm:
    outputs = llm.generate(["Explain gradient descent:"], SamplingParams(max_tokens=32))
    print(outputs[0].outputs[0].text)
    print(llm.diagnostics())
```

Omit `workload` for plain vLLM inference. Host algorithms use the same factory
with `HostRuntimeFacts(device, hidden_size=...)`, without constructing a model.
Existing callers must migrate to `workload=` and `config=`; retired module paths
and `canonical_config` are not compatibility APIs.

The Python import and distribution are now `pllm`; no `tllm` compatibility package
is installed. Existing `canonical.*` persisted schema/layout identities are unchanged.

## Write Tensor Mathematics

Using your own coding agent? Start with [the workload request template and agent
guide](doc/guides/agent-workloads.md): what to provide, which examples to use, and
what the agent should deliver.

```python
import torch
import pllm

class CenteredHidden(pllm.Workload):
    @pllm.on(pllm.Phase.OBSERVE)
    @pllm.hook(0)
    @pllm.export(name="centered_hidden", every=256, max_cols=16)
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden - hidden.mean(dim=-1, keepdim=True)
```

This declares capture and an output effect, not runtime hooks or graph nodes.
See the [authoring guide](doc/guides/authoring.md) for parameter scope, annotated
inputs, selection, first-order gradient edges, activation checkpoints and request
overlays with conditional replay. TLM and TTRL use causal/completion learning
interfaces; FTTT and GTTA use structured transactions. All eleven built-ins now
use their applicable public authoring interface. Resource and performance
acceptance remain tracked in the consolidation plan.

## Algorithms

| Algorithm | Behavior |
| --- | --- |
| Dummy | Selected hidden-row export and bounded trace |
| ESamp | Request-local training and sampler guidance |
| In-Place-TTT | Request-local dense parameter overlays |
| TLM | Instance q/v LoRA, AdamW and online/offline publication |
| Output LoRA | Shared or independent request heads, weighted CE and SGD |
| ReFT | Signed activation edits and checkpoint publication |
| Recirculation | Tokenwise prefill replay with original KV |
| OverRIDE | Iterative reweighting and actual sampled-token feedback |
| FTTT | Edit learning, host feedback and bounded retry |
| GTTA | CPU environment scoring and transactional updates |
| TTRL | Majority-reward learning and retry/barrier output |

All use the same graph, signed plan and execution owners. New algorithms implement
the [authoring contract](doc/guides/authoring.md), not their own executor or arena.
See [architecture and supported boundaries](doc/architecture.md) and
[In-Place-TTT semantics](doc/inplace-ttt.md).

## Installation

Python 3.11+, a Rust toolchain and matching PyTorch/CUDA packages are required.
Install platform-compatible PyTorch and vLLM wheels before installing PLLM:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[vllm027,test]"
```

The current backend targets unquantized Qwen2 models and vLLM 0.27.1, MRv2 by
default, TP1/PP1/DP1, non-speculative execution and sampling `n=1`. MRv1 supports
the shared attachment path but not KV replay. `multiprocessing=True` enables
EngineCore for the closed built-in workload definitions; arbitrary user Python
classes are not serialized or imported dynamically in workers.

`ExecutionOptions(capture=True)` captures supported training kernels, not the
whole model. Model dimensions and bounded resource facts are specialized before
signing. `training_rows` bounds external training rows per state;
`capture_memory_bytes` is the per-task resident limit. Insufficient capacity is
an error, not a silent eager fallback. Overlays cannot use shared prefix caching.
The local environment uses PyTorch 2.13/CUDA 12.9 and vLLM 0.27.1+cu129; select
matching wheels rather than assuming every upstream CUDA variant is compatible.
CPU-only development can install `.[test]`. No global vLLM plugin is installed.

## Benchmarks

Internal scenarios use the same public workload API:

```bash
.venv/bin/python -m tools.benchmark_esamp --batch 8 --tokens 256 --rounds 3
.venv/bin/python -m tools.benchmark_lora --batch 8 --tokens 256 --rounds 3
```

Both compare eager and captured execution with equal effective work. LoRA runs
separate shared and independent-state comparisons. It is no longer a detached
kernel microbenchmark. [Measurement details](doc/guides/benchmarking.md) explain
the workload, limits and historical comparison caveats.

Development uses `/workspace/pllm/.venv/bin/python`.
The [validation guide](doc/guides/validation.md) lists focused checks and the
release gate; [documentation](doc/README.md) links the maintained design sources.
