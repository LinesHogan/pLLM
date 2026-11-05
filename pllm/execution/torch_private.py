"""Typed Torch API boundary for execution: tracing, module initialization, storage and memory accounting."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from inspect import getclosurevars, unwrap
from pathlib import Path
from types import FunctionType
from typing import TypedDict
from uuid import UUID

import torch
from torch import _decomp
from torch._dynamo.convert_frame import _bytecode_hooks  # pyright: ignore[reportPrivateUsage]
from torch._dynamo.utils import CleanupManager
from torch._functorch import compile_utils
from torch._library.custom_ops import CustomOpDef as CustomOpDef, _maybe_get_opdef  # pyright: ignore[reportPrivateUsage]
from torch._ops import OpOverload, OpOverloadPacket, _OpNamespace, _Ops  # pyright: ignore[reportPrivateUsage]
from torch._subclasses.fake_tensor import DynamicOutputShapeException, FakeTensor, FakeTensorMode as FakeTensorMode
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils import _python_dispatch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
from torch.utils.flop_counter import flop_registry

fork_rng: Callable[[list[int]], AbstractContextManager[None]] = torch.random.fork_rng  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
manual_seed: Callable[[int], torch.Generator] = torch.manual_seed  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
fx_graph_cse: Callable[[torch.fx.Graph], torch.fx.Graph] = compile_utils.fx_graph_cse  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

CudaPoolHandle = torch.cuda._POOL_HANDLE  # pyright: ignore[reportPrivateUsage]
disable_current_modes = _python_dispatch._disable_current_modes  # pyright: ignore[reportPrivateUsage]


def bytecode_hooks() -> set[int]:
    """Snapshot Dynamo hook handles so a backend can own every decorator's registration."""
    return set(_bytecode_hooks)


def compilation_cleanups() -> set[int]:
    """Snapshot Torch's compiled-code cleanup records without retaining code or tensors."""
    return set(CleanupManager.instance.values)


def release_compilation(hooks: tuple[int, ...], cleanups: tuple[int, ...]) -> None:
    """Release one engine's hooks and module globals using Torch's cleanup records."""
    for key in hooks:
        _bytecode_hooks.pop(key, None)
    for key in cleanups:
        if key in CleanupManager.instance.values:
            CleanupManager.instance._remove_id(key)  # pyright: ignore[reportPrivateUsage]


def release_graph_pool_scope(pool: CudaPoolHandle) -> None:
    """Balance the reference taken when a conditional node reopens its graph's allocation pool."""
    torch._C._cuda_releasePool(torch.cuda.current_device(), pool)  # pyright: ignore[reportPrivateUsage]


def clear_blas_workspaces() -> None:
    """Drop Torch's workspace cache after an engine's streams and graphs have stopped.

    Conditional nodes use child streams; CUDAGraph.reset clears only the root
    stream's scratch. A cached child workspace can keep a released pool alive.
    """
    torch._C._cuda_clearCublasWorkspaces()  # pyright: ignore[reportPrivateUsage]


@dataclass(frozen=True)
class CustomOp:
    """Torch's registered author functions, decoded for signing and diagnostics."""

    name: str
    schema: str
    functions: tuple[tuple[str, object], ...]
    cpu: bool
    autograd: bool
    vmap: bool


def custom_op(value: object) -> CustomOp | None:
    """Read CustomOpDef internals only at this Torch boundary; require pure, shape-known math."""
    if not isinstance(value, CustomOpDef):
        return None
    schema = value._opoverload._schema  # pyright: ignore[reportPrivateUsage]
    name = value._qualname  # pyright: ignore[reportPrivateUsage]
    mutated = tuple(arg.name for arg in schema.arguments if arg.alias_info is not None and arg.alias_info.is_write)
    if mutated:
        raise ValueError(
            f"{name} mutates {', '.join(mutated)}; kernels called from workload math must return new tensors "
            "(mutates_args=()); change state only through pllm effects"
        )
    attributes: dict[str, object] = vars(value)
    if attributes["_abstract_fn"] is None:
        raise ValueError(f"{name} has no fake implementation; add @{name.split('::')[-1]}.register_fake")
    kernels: dict[str | None, Callable[..., object]] = value.__dict__["_backend_fns"]
    registrations = (
        ("implementation", attributes["_init_fn"]),
        ("fake", attributes["_abstract_fn"]),
        ("backward", attributes["_backward_fn"]),
        ("setup_context", attributes["_setup_context_fn"]),
        ("vmap", attributes["_vmap_fn"]),
    )
    functions = [(key, function) for key, function in registrations if function is not None]
    for device, wrapper in kernels.items():
        function = unwrap(wrapper)
        assert isinstance(function, FunctionType)
        functions.append((f"kernel:{device}", getclosurevars(function).nonlocals["fn"]))
    return CustomOp(
        name,
        str(schema),
        tuple(functions),
        "cpu" in kernels or None in kernels,
        any(key == "backward" for key, _ in functions),
        any(key == "vmap" for key, _ in functions),
    )


def custom_op_definition(target: object) -> CustomOpDef | None:
    """Resolve a traced operator through Torch's own definitions, without a pllm registry."""
    return _maybe_get_opdef(target.name()) if isinstance(target, OpOverload) else None


def dispatcher_namespace(value: object) -> str | None:
    """Distinguish Torch dispatcher handles from ordinary imported modules and callables."""
    if isinstance(value, _Ops):
        return ""
    if isinstance(value, _OpNamespace):
        return value.name
    if isinstance(value, OpOverload):
        return value.namespace
    if isinstance(value, OpOverloadPacket):
        return value._qualified_op_name.split("::")[0]  # pyright: ignore[reportPrivateUsage]
    return None


@contextmanager
def fixed_shapes() -> Generator[None]:
    """Explain why data-dependent custom-op outputs cannot be planned or captured."""
    try:
        yield
    except (RuntimeError, DynamicOutputShapeException) as error:
        if isinstance(error, DynamicOutputShapeException) or "dynamic" in str(error).lower() or "get_ctx" in str(error):
            raise ValueError(
                "methods run inside CUDA graphs, so outputs need shapes fixed by the inputs; "
                "return a fixed-size tensor and a tensor mask instead of ctx.new_dynamic_size()"
            ) from error
        raise


class CustomOpTrace(TorchDispatchMode):
    """Collect opaque calls during shape tracing; their fake bodies cannot prove host-sync freedom."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.calls: list[tuple[CustomOpDef, int | None, int]] = []

    def __torch_dispatch__(
        self,
        func: Callable[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        definition = custom_op_definition(func)
        with fixed_shapes():
            result = func(*args, **(kwargs or {}))
        if definition is not None and isinstance(func, OpOverload) and func.namespace != "pllm":
            formula = flop_registry.get(func.overloadpacket)
            flops = None if formula is None else int(formula(*args, **(kwargs or {}), out_val=result))
            size = sum(
                value.numel() * value.element_size()
                for value in tree_leaves((args, kwargs, result))
                if isinstance(value, torch.Tensor)
            )
            self.calls.append((definition, flops, size))
        return result


class GatherTrace(TorchDispatchMode):
    """Validate mapped tables' row reads and count their logical transfer bytes during planning."""

    def __init__(self, tables: tuple[tuple[str, torch.Tensor, bool], ...]) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.tables = {StorageWeakRef(tensor.untyped_storage()): (name, host) for name, tensor, host in tables}
        self.bytes: dict[str, int] = {}

    def check_outputs(self, values: tuple[torch.Tensor, ...]) -> None:
        """A helper must materialize gathered rows before exposing its output to a consumer."""
        for value in values:
            table = self.tables.get(StorageWeakRef(value.untyped_storage()))
            if table is not None and table[1]:
                raise ValueError(f'{table[0]}: gather host rows before returning a result, or use placement="device"')

    def __torch_dispatch__(
        self,
        func: Callable[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        reads = [
            (value, *self.tables[StorageWeakRef(value.untyped_storage())])
            for value in tree_leaves((args, kwargs))
            if isinstance(value, torch.Tensor) and StorageWeakRef(value.untyped_storage()) in self.tables
        ]
        result = func(*args, **(kwargs or {}))
        if not reads:
            return result
        name = str(func)
        views = (
            "aten.view.default",
            "aten.reshape.default",
            "aten.slice.Tensor",
            "aten.select.int",
            "aten.detach.default",
            "aten.alias.default",
            "aten.unsqueeze.default",
        )
        if name in views:
            return result
        gathers = ("aten.index.Tensor", "aten.index_select.default", "aten.embedding.default")
        for tensor, resource, host in reads:
            row_gather = name in gathers and tensor.ndim == 2
            if name == "aten.index.Tensor":
                indices = tree_leaves(args[1])
                row_gather = row_gather and len(indices) == 1 and isinstance(indices[0], torch.Tensor)
            elif name == "aten.index_select.default":
                row_gather = row_gather and args[1] == 0
            if host and not row_gather:
                raise ValueError(
                    f'{resource}: placement="host" supports row gathers only (got {name}); '
                    'gather rows before tensor math, or use placement="device"'
                )
            if row_gather:
                assert isinstance(result, torch.Tensor)
                self.bytes[resource] = self.bytes.get(resource, 0) + result.numel() * result.element_size()
        return result


def register_gradient_hook(tensor: torch.Tensor, hook: Callable[[torch.Tensor], torch.Tensor]) -> None:
    tensor.register_hook(hook)  # pyright: ignore[reportUnknownMemberType]


def tensor_decompositions(operators: tuple[OpOverload, ...]) -> dict[OpOverload, Callable[..., torch.Tensor]]:
    """Torch decompositions for fake tracing without AOT's CUDA RNG initialization."""
    return {  # pyright: ignore[reportReturnType,reportUnknownVariableType]
        **_decomp.core_aten_decompositions(),  # pyright: ignore[reportUnknownMemberType]
        **_decomp.get_decompositions(operators),  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    }


def check_functional_aten(target: object, *, per_token: bool = False) -> None:
    definition = custom_op_definition(target)
    if definition is not None:
        op = custom_op(definition)
        assert op is not None
        if per_token and not op.vmap:
            raise ValueError(
                f"{op.name} runs in a per-token cell under torch.vmap; register a batching rule "
                '(register_vmap) or call it with over="span"'
            )
        return
    if (
        not isinstance(target, OpOverload)
        or target.namespace != "aten"
        or target._schema.is_mutable  # pyright: ignore[reportPrivateUsage]
        or torch.Tag.nondeterministic_seeded in target.tags
    ):
        raise ValueError(f"export requires deterministic functional ATen operators, got {target}")


def tensor_node_metadata(node: torch.fx.Node) -> FakeTensor | None:
    value: object = node.meta.get("val")
    return value if isinstance(value, FakeTensor) else None


def load_tensor_fields(path: Path, fields: tuple[str, ...]) -> tuple[torch.Tensor, ...]:
    """Decode a weights-only checkpoint into the caller's explicit tensor field order."""
    payload: object = torch.load(path, map_location="cpu", weights_only=True)
    tensors = (
        tuple(t for f in fields if isinstance(t := payload.get(f), torch.Tensor))  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        if isinstance(payload, dict)
        else ()
    )
    if len(tensors) != len(fields):
        raise ValueError(f"checkpoint {path.name} must map {fields} to tensors")
    return tensors


def cuda_device_uuid(device: torch.device) -> str:
    return str(UUID(str(torch.cuda.get_device_properties(device).uuid)))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]


CudaStorageHandle = tuple[int, bytes, int, int, bytes, int, bytes, bool]


def share_cuda_storage(tensor: torch.Tensor) -> CudaStorageHandle:
    return tensor.untyped_storage()._share_cuda_()  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType]


def open_cuda_storage(handle: CudaStorageHandle) -> torch.UntypedStorage:
    return torch.UntypedStorage._new_shared_cuda(*handle)  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType]


class _PoolSegment(TypedDict):
    device: int
    total_size: int


def cuda_graph_pool_bytes(pool: tuple[int, int], device: int) -> int:
    """Decode PyTorch allocator facts at the foreign-data boundary, only in cold planning."""
    assert torch.cuda.get_allocator_backend() == "native" and pool != (0, 0)
    segments: list[_PoolSegment] = torch.cuda.memory_snapshot(pool, False)  # pyright: ignore[reportUnknownMemberType,reportAssignmentType]
    return sum(s["total_size"] for s in segments if s["device"] == device)


@dataclass(frozen=True)
class MemoryUsage:
    """Task peak, retained allocations and conservative capture reservation, in bytes."""

    workspace: int
    output: int
    capture: int = 0


class MemoryTrace(TorchDispatchMode):
    """Follow storage lifetimes, including aliases and autograd's saved tensors."""

    active: ContextVar["MemoryTrace | None"] = ContextVar("memory_trace", default=None)

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.sizes: dict[StorageWeakRef, int] = {}
        self.live = 0
        self.peak = 0
        self.small_allocations = 0
        self.large_allocations = 0
        self.oversize_allocations = 0

    def _capture_allocation(self, size: int) -> None:
        # Bound a private native allocator pool without assuming any reuse. Small requests use
        # 2 MiB segments, medium requests 20 MiB; each fills at least half a segment before
        # another is needed. Oversize requests get individually rounded 2 MiB segments.
        if size <= 1 << 20:
            self.small_allocations += size
        elif size < 10 << 20:
            self.large_allocations += size
        else:
            self.oversize_allocations += (size + (2 << 20) - 1) // (2 << 20) * (2 << 20)

    @property
    def capture_bytes(self) -> int:
        """Conservative no-reuse reservation, including each used pool's unfinished segment."""
        return (
            (2 * self.small_allocations + (2 << 20) if self.small_allocations else 0)
            + (2 * self.large_allocations + (20 << 20) if self.large_allocations else 0)
            + self.oversize_allocations
        )

    def _forget(self, ref: StorageWeakRef) -> None:
        self.live -= self.sizes.pop(ref)

    def _remember(self, values: object, owned: bool) -> None:
        for value in tree_leaves(values):
            if isinstance(value, torch.Tensor):
                storage = value.untyped_storage()
                ref = StorageWeakRef(storage)
                if ref in self.sizes:
                    continue
                size = (storage.nbytes() + 511) // 512 * 512
                if owned:
                    self._capture_allocation(size)
                # Native large-pool reuse can keep an unsplit tail of up to 1 MiB.
                size += (1 << 20) if size > (1 << 20) else 0
                self.sizes[ref] = size if owned else 0
                self.live += self.sizes[ref]
                # Torch keeps a storage's Python object as long as its bytes, so this runs exactly at the free;
                # keeping the total incremental makes tracing linear in the number of operations.
                weakref.finalize(storage, self._forget, ref)

    def __torch_dispatch__(
        self,
        func: Callable[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        self._remember((args, kwargs), False)
        result = func(*args, **(kwargs or {}))
        self._remember(result, True)
        self.peak = max(self.peak, self.live)
        return result

    @classmethod
    def reserve(cls, size: int) -> None:
        """Include an executable's opaque workspace at the call's live frontier."""
        trace = cls.active.get()
        if trace is not None:
            trace.peak = max(trace.peak, trace.live + size)
            trace._capture_allocation(size)


def blas_workspace_bytes() -> int:
    """Read the current CUDA device's configured BLAS scratch at GPU planning boundaries."""
    return torch.backends.cuda.cublas_workspace_size() + torch.backends.cuda.cublaslt_workspace_size()


def measure(function: Callable[[], object]) -> MemoryUsage:
    """Trace the actual callable; keep its return alive while measuring retained storage."""
    with torch.inference_mode(False), MemoryTrace() as trace:
        token = MemoryTrace.active.set(trace)
        try:
            _result = function()
            return MemoryUsage(trace.peak, trace.live, trace.capture_bytes)
        finally:
            MemoryTrace.active.reset(token)
