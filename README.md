# tLLM

tLLM executes auxiliary tensor workloads alongside vLLM inference. Workloads
declare a typed semantic graph; Rust signs scheduling and state representation
decisions, and Python executes the resulting physical plan.

The architecture consolidation is **in progress**. The current implementation
is a development checkpoint, not a completed release of all planned workloads.
See the [remaining work](doc/architecture-consolidation-handoff.md).

## Current Support

| Path | Verified boundary |
| --- | --- |
| Plain inference | `tllm.make_llm` constructs vLLM directly, without global tLLM hooks |
| ESamp | Request training, sampler guidance, exact state pins and cleanup |
| In-Place-TTT | Dense overlays for two requests, training and version cleanup |
| FTTT | Typed host feedback and CUDA receipt/stream boundary |
| Dummy | CPU graph compilation and owned hidden-row export |
| GTTA | CPU environment scores, FIFO abort/requeue and version rollback |
| TLM training core | Shared instance-state q/v AdamW and online/offline objective; real HF/PEFT CPU/CUDA numerical oracle |

The public attached vLLM path currently supports ESamp and In-Place-TTT on
unquantized Qwen2 models, vLLM **0.27.1 MRv2**, eager execution, in process,
TP1/PP1/DP1, non-speculative decoding and sampling `n=1`.
The standalone ESamp CUDA Graph test proves its training core; it does not
establish model-backed vLLM CUDA Graph support.

TLM's public vLLM global overlay and admission pin remain pending; its training
core currently requires dropout=0. Complete TLM, ReFT, Recirculation,
OverRIDE, TTRL, MRv1 attachment, typed multiprocessing and final AER/PTTS
integration remain unfinished. Historical results from deleted implementations
do not establish support for the current code.

## Installation

Python 3.11+, a Rust toolchain and matching PyTorch/CUDA packages are required.
Install a platform-appropriate PyTorch build first. For the public GPU path,
install vLLM 0.27.1 with the matching CUDA variant before installing tLLM.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[vllm027,test]"
```

Build and install from the repository root; the Rust crate is not a separate
Python distribution. The `test` extra includes the test runner, native build
tool and typing checker. CPU-only development can use `.[test]` without vLLM.

The verified local environment uses vLLM runtime 0.27.1 (distribution
0.27.1+cu129). CUDA 12.x installations must select compatible CUDA wheels;
the default upstream wheels may target CUDA 13. On Python 3.11, the verified
environment uses FlashInfer 0.6.17; older FlashInfer annotations may prevent
import. The repository does not install a global vLLM plugin.

## ESamp Example

```python
import tllm
from vllm import SamplingParams
from tllm.canonical.workloads.esamp_request import ESampRequestWorkload
from tllm.canonical.workloads.esamp_request_types import ESampRequestConfig
from tllm.canonical_vllm.runtime import (
    CanonicalVllmConfig, CanonicalVllmDType, CanonicalVllmLlm,
)

llm = tllm.make_llm(
    workloads=(ESampRequestWorkload(ESampRequestConfig(
        rank=2, source_layer=0, target_layer=-1,
        intervention=True, intervention_beta=0.2,
    )),),
    canonical_config=CanonicalVllmConfig(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        dtype=CanonicalVllmDType.BFLOAT16,
        max_model_len=64, max_num_seqs=1,
        max_num_batched_tokens=64, gpu_memory_utilization=0.4, seed=7,
    ),
)
assert isinstance(llm, CanonicalVllmLlm)
try:
    outputs = llm.generate(
        ["The sky is"],
        SamplingParams(max_tokens=3, temperature=0.0, n=1, ignore_eos=True),
    )
    print(outputs[0].outputs[0].text)
finally:
    llm.close()
```

See [In-Place-TTT](doc/inplace-ttt.md) for dense overlays. The old Program,
consumer registration and workload DSL packages have been retired; there is
no compatibility route.

## Validation and Development

[Validation](doc/guides/validation.md) lists the CPU, Rust, wheel and GPU
commands. The [evidence manifest](test/evidence_manifest.toml) distinguishes
executable evidence from missing contracts. Tests require nonzero training
where expected, exact version cleanup and zero fallback.

[Architecture](doc/architecture.md) describes the current owners.
[Documentation](doc/README.md) links the maintained guides and consolidation
plan. Current GPU smokes are correctness evidence, not throughput claims.

License: [Apache-2.0](LICENSE).
