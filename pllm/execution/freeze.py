"""Cold-code signing: fingerprint and freeze authoring dependencies without evaluating them.

Workload mathematics is signed by structural fingerprint — marshaled bytecode plus the
transitive closure of globals and free variables — never by executing user code or
evaluating annotation strings. `freeze` produces a detached deep copy whose function
values are recursively frozen, so later mutation of the authoring environment cannot
change already-signed behavior.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import fields, is_dataclass
from dis import get_instructions
from enum import Enum
from hashlib import sha256
from inspect import getclosurevars
from itertools import pairwise
from marshal import dumps
from types import CellType, CodeType, FunctionType, MethodType, ModuleType
from typing import TypeVar, cast

import torch

from pllm.contracts.codec import boundary_digest

G = TypeVar("G", bound=Callable[..., object])
C = TypeVar("C")


def fingerprint(value: object, seen: frozenset[int] = frozenset()) -> str:
    """Sign cold code/config dependencies, never eval annotations or execute code to trace it."""
    if id(value) in seen:
        raise ValueError("recursive authoring dependencies require an explicit native binding")
    seen = seen | {id(value)}
    if isinstance(value, MethodType):
        return fingerprint(value.__func__, seen)
    if isinstance(value, FunctionType):
        closure = getclosurevars(value)
        globals_used = _global_names(value.__code__)
        if globals_used & {"getattr", "setattr", "hasattr", "globals", "locals", "vars", "eval", "exec", "__import__"}:
            raise ValueError("reflective mathematics requires an explicit native binding")
        dependencies = {
            **{key: value.__globals__[key] for key in globals_used if key in value.__globals__},
            # `__class__` is zero-argument `super()`'s cell: the class being signed, not a dependency.
            **{key: item for key, item in closure.nonlocals.items() if key != "__class__"},
        }
        return boundary_digest(
            sha256(dumps(_code(value.__code__))).hexdigest(),
            tuple((key, fingerprint(item, seen)) for key, item in sorted(dependencies.items())),
            fingerprint(value.__defaults__, seen),
            fingerprint(value.__kwdefaults__, seen),
        )
    if isinstance(value, ModuleType):
        if value.__name__ != "pllm" and not value.__name__.startswith(("torch", "math")):
            raise ValueError(f"{value.__name__}: module dependencies require an explicit native binding")
        return boundary_digest(value.__name__, torch.__version__)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Enum):
        return boundary_digest(type(value).__qualname__, fingerprint(value.value, seen))
    from pllm.execution.tensor_frame import TensorUpdate

    if value is TensorUpdate:
        return TensorUpdate.__qualname__
    if isinstance(value, type):
        if value.__module__.partition(".")[0] == "torch":
            return boundary_digest(value.__module__, value.__qualname__, torch.__version__)
        if issubclass(value, torch.nn.Module):
            # A user module (module state) is signed by the code of its own methods.
            return boundary_digest(
                tuple(
                    (key, fingerprint(item, seen))
                    for cls in value.__mro__
                    if cls.__module__.partition(".")[0] not in ("torch", "builtins")
                    for key, item in sorted(vars(cls).items())
                    if isinstance(item, FunctionType)
                )
            )
        raise ValueError("dynamic type dependencies require an explicit native binding")
    if is_dataclass(value) and not isinstance(value, type):
        return boundary_digest(
            type(value).__qualname__,
            tuple((field.name, fingerprint(getattr(value, field.name), seen)) for field in fields(value)),
        )
    if isinstance(value, Mapping):
        items = cast(Mapping[str, object], value)
        return boundary_digest(tuple((key, fingerprint(item, seen)) for key, item in sorted(items.items())))
    if isinstance(value, (tuple, list)):
        return boundary_digest(tuple(fingerprint(item, seen) for item in cast(tuple[object, ...], value)))
    return boundary_digest(value)


def _code(code: CodeType) -> CodeType:
    return code.replace(
        co_filename="",
        co_firstlineno=0,
        co_linetable=b"",
        co_consts=tuple(_code(value) if isinstance(value, CodeType) else value for value in code.co_consts),
    )


def _global_names(code: CodeType) -> set[str]:
    names = {str(instruction.argval) for instruction in get_instructions(code) if instruction.opname == "LOAD_GLOBAL"}
    for value in code.co_consts:
        if isinstance(value, CodeType):
            names.update(_global_names(value))
    return names


def reachable_functions(candidates: Mapping[str, FunctionType], seeds: tuple[FunctionType, ...]) -> frozenset[str]:
    """Follow owner method loads; an escaping owner conservatively signs the whole class."""
    reachable: set[str] = set()
    pending = [
        (function.__code__, function.__code__.co_varnames[0]) for function in seeds if function.__code__.co_varnames
    ]
    loads = {"LOAD_FAST", "LOAD_DEREF", "LOAD_FAST_CHECK", "LOAD_FAST_AND_CLEAR", "LOAD_CLASSDEREF"}
    while pending:
        code, owner = pending.pop()
        pending.extend((value, owner) for value in code.co_consts if isinstance(value, CodeType))
        for instruction, follower in pairwise(get_instructions(code)):
            if instruction.opname == "LOAD_SUPER_ATTR" or instruction.argval == "super":
                return frozenset(candidates)
            if instruction.argval != owner:
                continue
            if instruction.opname.startswith(("STORE_", "DELETE_")):
                return frozenset(candidates)
            if instruction.opname not in loads:
                continue
            if follower.opname not in ("LOAD_ATTR", "LOAD_METHOD"):
                return frozenset(candidates)
            name = str(follower.argval)
            if name in candidates and name not in reachable:
                reachable.add(name)
                function = candidates[name]
                pending.append((function.__code__, function.__code__.co_varnames[0]))
    return frozenset(reachable)


def freeze(function: G) -> G:
    if isinstance(function, MethodType):
        return cast(G, MethodType(freeze(function.__func__), function.__self__))
    if not isinstance(function, FunctionType):
        raise TypeError("ordinary operations require a Python function; use an explicit native binding")
    globals_used = _global_names(function.__code__)
    global_values = dict(function.__globals__)
    for key in globals_used & global_values.keys():
        value = global_values[key]
        global_values[key] = (
            freeze(value)
            if isinstance(value, FunctionType)
            else (value if isinstance(value, (ModuleType, type)) else deepcopy(value))
        )
    copied = FunctionType(
        function.__code__,
        global_values,
        function.__name__,
        deepcopy(function.__defaults__),
        None
        if function.__closure__ is None
        else tuple(
            CellType(
                freeze(cell.cell_contents)
                if isinstance(cell.cell_contents, FunctionType)
                else deepcopy(cell.cell_contents)
            )
            for cell in function.__closure__
        ),
    )
    copied.__annotations__ = dict(function.__annotations__)
    copied.__kwdefaults__ = deepcopy(function.__kwdefaults__)
    return cast(G, copied)


def members(owner: object) -> dict[str, object]:
    result: dict[str, object] = {}
    for cls in reversed(type(owner).__mro__):
        result.update(vars(cls))
    return result


def owner_values(owner: object, marker_types: tuple[type, ...]) -> dict[str, object]:
    """Authoring values: scalar/class-level markers plus instance and dataclass fields."""
    result: dict[str, object] = {
        name: member
        for name, member in members(owner).items()
        if not name.startswith("__") and isinstance(member, (str, int, float, bool, tuple, *marker_types))
    }
    result.update(vars(owner))
    if is_dataclass(owner) and not isinstance(owner, type):
        result.update((item.name, getattr(owner, item.name)) for item in fields(owner))
    return result


def owner_snapshot(owner: C, marker_types: tuple[type, ...], borrowed: tuple[object, ...] = ()) -> C:
    """Deep-copy the owner, freezing its function values so signed math cannot drift."""
    memo = {id(value): value for value in borrowed}
    copied = deepcopy(owner, memo)
    for key, value in owner_values(copied, marker_types).items():
        object.__setattr__(copied, key, freeze(value) if isinstance(value, FunctionType) else deepcopy(value, memo))
    return copied
