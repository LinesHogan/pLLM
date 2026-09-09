"""Cold-code signing: fingerprint and freeze authoring dependencies without evaluating them.

Workload mathematics is signed by structural fingerprint — marshaled bytecode plus the
transitive closure of globals and free variables — never by executing user code or
evaluating annotation strings. `freeze` produces a detached deep copy whose function
values are recursively frozen, so later mutation of the authoring environment cannot
change already-signed behavior.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import fields, is_dataclass
from dis import get_instructions
from enum import Enum
from hashlib import blake2b, sha256
from inspect import getclosurevars
from marshal import dumps
import sys
from types import CellType, CodeType, FunctionType, MethodType, ModuleType
from typing import TypeVar, cast

import torch

from pllm.execution.codec import boundary_digest

G = TypeVar("G", bound=Callable[..., object])


def fingerprint(value: object, seen: frozenset[int] = frozenset(), *, structured: bool = False) -> str:
    """Sign cold code/config dependencies, never eval annotations or execute code to trace it."""
    if id(value) in seen:
        raise ValueError("recursive authoring dependencies require an explicit native binding")
    seen = seen | {id(value)}
    if isinstance(value, MethodType):
        return fingerprint(value.__func__, seen, structured=structured)
    if isinstance(value, FunctionType):
        closure = getclosurevars(value)
        globals_used = _global_names(value.__code__)
        if globals_used & {"getattr", "setattr", "hasattr", "globals", "locals", "vars", "eval", "exec", "__import__"}:
            raise ValueError("reflective mathematics requires an explicit native binding")
        dependencies = {**{key: value.__globals__[key] for key in globals_used if key in value.__globals__}, **closure.nonlocals}
        return boundary_digest(sha256(dumps(_code(value.__code__))).hexdigest(),
            tuple((key, fingerprint(item, seen, structured=structured)) for key, item in sorted(dependencies.items())),
            fingerprint(value.__defaults__, seen, structured=structured), fingerprint(value.__kwdefaults__, seen, structured=structured))
    if isinstance(value, ModuleType):
        if structured and value.__name__ == "re":
            return boundary_digest(value.__name__, sys.version)
        if not value.__name__.startswith(("torch", "math")):
            raise ValueError(f"{value.__name__}: module dependencies require an explicit native binding")
        return boundary_digest(value.__name__, torch.__version__)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Enum):
        return boundary_digest(type(value).__qualname__, fingerprint(value.value, seen))
    from pllm.execution.tensor_binding import TensorUpdate
    if value is TensorUpdate:
        return TensorUpdate.__qualname__
    if structured and is_dataclass(value) and isinstance(value, type):
        return boundary_digest(value.__module__, value.__qualname__,
            tuple((item.name, str(item.type)) for item in fields(value)),
            fingerprint(value.__init__, seen, structured=True), fingerprint(getattr(value, "__post_init__", None), seen, structured=True))
    if isinstance(value, type):
        if value is torch.Tensor:
            return value.__qualname__
        if structured and value in (object, Counter, blake2b):
            return boundary_digest(value.__module__, value.__qualname__, sys.version)
        if structured and issubclass(value, Enum):
            return boundary_digest(value.__module__, value.__qualname__,
                tuple((name, fingerprint(member.value, seen, structured=True)) for name, member in value.__members__.items()))
        raise ValueError("dynamic type dependencies require an explicit native binding")
    if is_dataclass(value) and not isinstance(value, type):
        return boundary_digest(type(value).__qualname__, tuple((field.name,
            fingerprint(getattr(value, field.name), seen, structured=structured)) for field in fields(value)))
    if isinstance(value, Mapping):
        items = cast(Mapping[str, object], value)
        return boundary_digest(tuple((key, fingerprint(item, seen, structured=structured)) for key, item in sorted(items.items())))
    if isinstance(value, (tuple, list)):
        return boundary_digest(tuple(fingerprint(item, seen, structured=structured) for item in cast(tuple[object, ...], value)))
    return boundary_digest(value)


def _code(code: CodeType) -> CodeType:
    return code.replace(co_filename="", co_firstlineno=0, co_linetable=b"", co_consts=tuple(
        _code(value) if isinstance(value, CodeType) else value for value in code.co_consts))


def _global_names(code: CodeType) -> set[str]:
    names = {str(instruction.argval) for instruction in get_instructions(code) if instruction.opname == "LOAD_GLOBAL"}
    for value in code.co_consts:
        if isinstance(value, CodeType):
            names.update(_global_names(value))
    return names


def freeze(function: G) -> G:
    if isinstance(function, MethodType):
        return cast(G, MethodType(freeze(function.__func__), function.__self__))
    if not isinstance(function, FunctionType):
        raise TypeError("ordinary operations require a Python function; use an explicit native binding")
    globals_used = _global_names(function.__code__)
    global_values = dict(function.__globals__)
    for key in globals_used & global_values.keys():
        value = global_values[key]
        global_values[key] = freeze(value) if isinstance(value, FunctionType) else (
            value if isinstance(value, (ModuleType, type)) else deepcopy(value))
    copied = FunctionType(function.__code__, global_values, function.__name__, deepcopy(function.__defaults__),
        None if function.__closure__ is None else tuple(CellType(freeze(cell.cell_contents)
            if isinstance(cell.cell_contents, FunctionType) else deepcopy(cell.cell_contents)) for cell in function.__closure__))
    copied.__annotations__ = dict(function.__annotations__)
    copied.__kwdefaults__ = deepcopy(function.__kwdefaults__)
    return cast(G, copied)


def members(owner: object) -> dict[str, object]:
    result: dict[str, object] = {}
    for cls in reversed(type(owner).__mro__):
        result.update(vars(cls))
    return result
