# vLLM Compatibility

This page explains how tLLM handles vLLM version differences.

tLLM installs monkey patches at key `GPUModelRunner` lifecycle boundaries. vLLM changes internal interfaces across versions, especially the return structure of `_prepare_inputs`. tLLM isolates those differences behind versioned adapters, so the runtime can consume a normalized view.

## Supported Version Family

- `0.27.x` (`>=0.27.1,<0.28`)

The current development environment uses `vllm==0.27.1`. Both the legacy
Model Runner V1 and the default Model Runner V2 are covered. Older adapter
classes remain for historical repro code, but they are not the supported
production backend matrix.

## What the Adapter Normalizes

The normalized view contains:

- `attn_metadata`
- `logits_indices`
- `spec_decode_common`
- `num_scheduled_tokens_np`

## Current Parsing Rules

For a 6-tuple `_prepare_inputs` result:

- `attn_metadata = out[0]`
- `logits_indices = out[1]`
- `num_scheduled_tokens_np = out[3]`
- `spec_decode_common = out[4]`

For a 2-tuple result:

- `attn_metadata = out[0]`
- `logits_indices = out[1]`
- `spec_decode_common = None`
- `num_scheduled_tokens_np` is derived from `scheduler_output.num_scheduled_tokens`

## Adapter Family

- `V072PrepareInputsAdapter`: vLLM 0.7.x
- `V010PrepareInputsAdapter`: vLLM 0.10.x
- `V011PlusPrepareInputsAdapter`: vLLM 0.11+
- `V027PrepareInputsAdapter`: vLLM 0.27 MRv1

MRv2 uses its dedicated v0.27 backend path rather than the MRv1 tuple adapter.
The implementation shares parsing helpers but keeps version and runner-family
boundaries explicit.

## Who Uses It?

Developer verification and repro paths:

- `verify_v1_decode_rows_minimal.py`
- `tllm/workflows/repro/prefill_capture_support.py`
- `tllm/runtime/residual_runtime.py`

Runtime paths:

- `tllm/runtime/vllm_patch/port_runtime_hooks.py`
- `tllm/runtime/ports/`

Both groups consume the same normalized adapter view before doing localization or bundle assembly.
