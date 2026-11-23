# Why Custom Op Writes Disappear Under `torch.compile`: Debugging PyTorch AOT Functionalization

> **TL;DR**: When wrapping inter-layer normalization boundaries in a PyTorch custom op, writes to an activation tap pool worked fine in Eager mode yet read back as all zeros under `torch.compile`'s AOT Functionalization pass. The op declared the pool in `mutates_args`, yet its body bypassed the tensor argument and mutated the original buffer through a global dictionary reference—so the compiler's epilogue faithfully copied the untouched zero-filled clone right back over the pool. Splitting tap copying into a standalone op that mutates its tensor arguments directly realigns the data flow with the compiler contract, and fixes stale layer-ID metadata under early-exit custom forwards in the same stroke.

The bug showed up while attaching an activation tap copy to an inter-layer `RMSNorm` boundary.

In a continuous-batching inference engine, to insert custom residual edits and hidden-state taps at inter-layer normalization boundaries without breaking `torch.compile` full-graph capture, we wrapped the boundary logic in a PyTorch custom op, `pllm::model_site`. Because `torch.compile` specializes plain integer arguments as compile-time constants inside the graph, the op cannot accept a raw Python object ID; instead, it uses the data pointer of the 0th read-only tensor, `reads[0].data_ptr()`, as a runtime lookup key into a global `_SITES` dictionary to retrieve the session object and run the hook.

When registering the op, we placed every tensor mutated in place—including `tap_pool`, the GPU memory pool holding tapped hidden states—into the `writes` list, and declared `mutates_args={"outputs", "writes"}` on the decorator:

```python
@torch.library.custom_op("pllm::model_site", mutates_args={"outputs", "writes"})
def model_site(
    inputs: list[Tensor], outputs: list[Tensor], reads: list[Tensor], writes: list[Tensor], hook: int
) -> None:
    # Look up the session by the 0th read-only tensor's address and run the boundary hook
    _SITES[reads[0].data_ptr()].hooks[hook].emit(inputs, outputs)
```

Under Eager execution and raw CUDA Graph capture, everything worked as expected: the residual output held `[7, 7, 7, 7]`, and the corresponding slot in `tap_pool` held `[7, 7, 7, 7]`.

Wrap the same boundary in `torch.compile(..., backend="aot_eager", fullgraph=True)`, however, and the two diverged. The forward pass still returned `[7, 7, 7, 7]` for `residual` without raising an error, yet reading the same slot in `tap_pool` gave back the initial `[0, 0, 0, 0]`.

---

To understand why values written inside the op vanished back to zero on return, look at what the **Functionalization** pass in the PyTorch 2 compiler stack does.

Everyday PyTorch code relies heavily on in-place mutations (`pool.index_copy_()`, `x.add_()`) to avoid extra allocations. Backend graph compilers such as Inductor, however, require **Single Static Assignment (SSA)** form before fusing operators, eliminating dead code, or reordering instructions: every edge in the graph represents an immutable value, and dependencies flow strictly through explicit inputs and outputs, with no hidden in-place memory side effects.

Functionalization inside AOTAutograd translates graphs containing in-place mutations into pure SSA form. When it hits a black-box custom op declaring `mutates_args={"outputs", "writes"}`, it cannot rewrite the op's Python body, so it wraps the call via `auto_functionalized` in three steps:

1. **Clone**: Before invoking the op, clone every tensor in `outputs` and `writes` into temporary buffers (`writes_clone = [w.clone() for w in writes]`), and pass those clones in as arguments.
2. **Execute**: Run the op body assuming all in-place writes land on the cloned arguments, then return the mutated clones as pure functional outputs of the graph node.
3. **Epilogue Copy-back**: At the end of the graph, when updated state must be synchronized back to persistent external buffers, emit `writes_orig.copy_(writes_clone)` to write the clones back into the original tensors.

Put those three steps on a timeline, and the failure mode is immediate:

![Eager in-place write vs. AOT Functionalization clone-and-copy-back overwrite](../../assets/personal-01-fig1-en.svg)

`outputs` came through intact because `emit(inputs, outputs)` actually wrote to its `outputs` parameter, updating the clone to `[7, 7, 7, 7]` so the epilogue copy-back preserved it. `tap_pool` behaved differently: the caller placed it in `writes` to tell the compiler that the buffer would be mutated, yet `emit` ignored the `writes` argument and wrote directly to `self.pool.tensors[0]` via a Python object attribute.

As far as the compiler could see, `writes_clone[0]` entered as all zeros and exited as all zeros, meaning the op's new value for `tap_pool` was zero. The epilogue `copy_` dutifully overwrote the freshly written pool with zeros. Conversely, leaving `tap_pool` out of `writes` is no fix either: the compiler would assume the tensor was untouched, leaving any later read in the same graph open to reordering or dead-code elimination.

---

Once the mechanism is clear, the fix leaves `_SITES` untouched and simply pulls tap copying out of the `model_site` black box.

`model_site` needs the pointer-table Python callback only because custom residual-editing methods can carry arbitrary closure state. An activation tap does one narrow job: take the `(normed, residual)` tensors produced at the normalization boundary and copy selected rows into the memory pool according to an index table.

Moving `self._copy_taps(normed, residual)` out of `emit` and calling it directly after `run_hook` inside the outer `graph_forward` separates the two cleanly:

![Decoupling pure boundary site op from explicit routed_tap_copy mutation](../../assets/personal-01-fig2-en.svg)

After the split, `routed_tap_copy` receives `pool` and its companion layer-ID array `slot_layers` directly as tensor arguments and mutates them in place inside the kernel. AOT Functionalization clones the arguments, the kernel mutates the clones, and the epilogue copies them back—bringing the actual data flow back in line with the compiler's contract.

---

The same split also settled a second bug in dynamic layer routing: **whether exported layer-ID metadata should be written during setup, or only when the layer actually runs.**

Under dynamic-layer export (`@pllm.export`), every exported activation row carries `event.layers[r]` to record which model layer it came from (or `-1` when disabled).

The initial implementation read the target layer ID (say, `2`) from the control state during `bind_routes()` before the forward pass started, and wrote it directly into `_slot_layers[dst]`. Under a standard sequential forward pass across all layers, that holds up. The mismatch appears when paired with a `@pllm.custom_forward` that exits early—for example, running only layer 0 on a given step:

```python
@pllm.custom_forward(tokens="decode")
def early_exit(self, layers: pllm.model.Layers) -> torch.Tensor:
    return layers(0, 1)[0]  # Only runs layer 0 this step; never reaches layer 2
```

Here the control state still points to layer `2`, yet layer `2`'s module hook never executes on this step. Writing `2` into `_slot_layers` up front hands downstream code a row of zeros (the initial buffer contents) tagged as layer `2`. If the user is also using `route.layer_logits()` and the candidate set includes the final layer, that stale tag can even trigger spurious final-layer logits replacement.

Moving the layer-ID write inside the `routed_tap_copy` kernel keeps data and metadata aligned automatically:

```python
@triton.jit
def _routed_tap_copy_kernel(
    output_ptr, src_rows_ptr, dst_slots_ptr, row_layers_ptr,
    pool_ptr, slot_layers_ptr, out_stride0, out_stride1, pool_stride,
    width, layer_id, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    dst = tl.load(dst_slots_ptr + row)
    target_layer = tl.load(row_layers_ptr + row)
    if (dst < 0) | (target_layer != layer_id):
        return
    src = tl.load(src_rows_ptr + row)
    if src < 0:
        return
    if tl.program_id(1) == 0:
        # Write the actual layer ID from column block 0 only when this layer's hook runs and copies activations
        tl.store(slot_layers_ptr + dst.to(tl.int64), tl.full((), layer_id, dtype=tl.int64))
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < width
    values = tl.load(output_ptr + src.to(tl.int64) * out_stride0 + cols * out_stride1, mask=mask, other=0.0)
    tl.store(pool_ptr + dst.to(tl.int64) * pool_stride + cols, values, mask=mask)
```

Setup resets `_slot_layers` to `-1` across the board; `layer_id` is written alongside the activation row only when that layer's hook actually runs and `target_layer == layer_id`. Unvisited layers remain all zeros with a `-1` label—values and provenance are produced at the exact same moment.

---

Looking back, our first round of end-to-end GPU tests missed the overwrite in `model_site` for a simple reason: those workloads also enabled custom residuals or adapters, combinations under which the engine switched to `backend="eager"` with raw CUDA Graph capture, bypassing Inductor's AOT Functionalization altogether. Only when a workload enabled activation patches and taps on their own did it hit the default `backend="inductor"` path and surface the bug.

That gives a cheap way to catch this class of bug early: whenever you write a `torch.library.custom_op` with `mutates_args`, wrap it in `torch.compile(fn, backend="aot_eager", fullgraph=True)` in a pure CPU unit test and pass non-zero inputs to verify the mutated buffers. `aot_eager` needs no GPU and executes the full AOTAutograd clone-and-copy-back path in around a hundred milliseconds, flagging any mutation that bypasses the declared parameters.

The implementation lives in `pllm.model.LayerRoute` inside the open-source project **[pLLM](https://github.com/LinesHogan/pLLM)**. On the trade-off side, placing `routed_tap_copy` outside the boundary as a standalone op keeps one lightweight kernel node inside the CUDA Graph for each layer in the candidate set `over`. A missed layer only reads two `int64` values from L2 cache before returning and never touches the hidden-state pool, yet in-graph kernel launch overhead still scales linearly with `|over|`—making it best suited to declaring only the specific candidate layers you intend to switch between.
