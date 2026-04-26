# Test-Time LoRA Service

The test-time LoRA service defines the runtime contract for updating LoRA
adapters from in-memory tensors. It is intentionally dependency-light:
consumers produce typed LoRA updates, while the runtime owns rank validation,
target-module lookup, request-to-adapter assignment, and commit ordering.

## Contract

`TestTimeLoRAUpdate` is the canonical update object. Its internal identifiers
are deliberately narrow:

- `adapter_key` and request IDs are strings.
- `lora_int_id` is a positive adapter id; `rank` and `step_id` are integers.
- target module names are strings.
- LoRA tensors are rank-2 tensors with matching rank dimensions.

The contract rejects implicit coercion. For example, numeric request IDs or
floating LoRA adapter IDs are invalid at construction time rather than being
silently converted. String identifiers are also preserved exactly; leading or
trailing whitespace is rejected instead of being trimmed.

## Lifecycle

1. A consumer creates a `TestTimeLoRAUpdate` with one or more target modules.
2. The runtime stages the update after checking rank and target-module
   existence. Backend adapters own physical slot and active-capacity policy.
3. At a runtime-owned synchronization point, staged updates are committed into
   the backend LoRA manager.
4. Request-scoped updates bind the listed request IDs to the committed LoRA
   adapter id for later decode routing.

The service does not load PEFT adapters from disk. It expects already-created
LoRA tensors. Backend adapters own physical slot allocation when a backend,
such as vLLM, has separate adapter-id and slot-index concepts.

Request assignments are an explicit runtime binding, not a durable request
identity store. Backends that reuse request IDs after completion should call
`unbind_request()` before admitting the new request.

## Current Scope

This layer is a framework-side contract and unit-tested service shim. It is the
foundation for deeper vLLM integration, where model-load hooks can discover
LoRA modules and pre-execution hooks can route requests to committed in-memory
slots. Exact backend integration still needs backend-specific validation for
packed modules, tensor-parallel layouts, cudagraph compatibility, and stream
synchronization.
