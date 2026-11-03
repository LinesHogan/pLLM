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
from dis import Instruction, get_instructions, hasjabs, hasjrel, stack_effect
from enum import Enum
from hashlib import sha256
from inspect import getclosurevars
from itertools import pairwise
from marshal import dumps
from types import CellType, CodeType, FunctionType, MethodType, ModuleType
from typing import TypeGuard, TypeVar, cast

import torch

from pllm.contracts.codec import boundary_digest

G = TypeVar("G", bound=Callable[..., object])
C = TypeVar("C")


def fingerprint(value: object, seen: frozenset[int] = frozenset()) -> str:
    """Sign cold code/config dependencies, never eval annotations or execute code to trace it."""
    if id(value) in seen:
        raise ValueError(f"{type(value).__name__}: workload methods cannot depend on a value that contains itself")
    seen = seen | {id(value)}
    if isinstance(value, MethodType):
        return fingerprint(value.__func__, seen)
    if isinstance(value, FunctionType):
        closure = getclosurevars(value)
        globals_used = _global_names(value.__code__)
        imports = _builtin_dependencies(value)
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
            imports,
        )
    if isinstance(value, ModuleType):
        if value.__name__ != "pllm" and not value.__name__.startswith(("torch", "math")):
            raise ValueError(
                f"{value.__name__}: workload methods may use only the torch, math and pllm modules; "
                "compute other values outside the workload and pass them in its config"
            )
        return boundary_digest(value.__name__, torch.__version__)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Enum):
        return boundary_digest(type(value).__qualname__, fingerprint(value.value, seen))
    if _is_class(value):
        if value.__module__.partition(".")[0] in ("torch", "pllm"):
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
        if issubclass(value, Enum) or is_dataclass(value):
            return _structure(value, seen)
        raise ValueError(
            f"{value.__qualname__}: workload methods may reference torch and pllm classes and the author's "
            "nn.Module subclasses, enums and dataclasses"
        )
    if isinstance(value, (torch.nn.Module, torch.Tensor)):
        raise ValueError(
            f"{type(value).__name__}: a workload cannot hold a module or tensor; save its weights and declare "
            "pllm.InstanceState(module=<factory>, pretrained=path) or field state with pretrained=path"
        )
    if is_dataclass(value):
        return boundary_digest(
            type(value).__qualname__,
            tuple((field.name, fingerprint(getattr(value, field.name), seen)) for field in fields(value)),
        )
    if isinstance(value, Mapping):
        items = cast(Mapping[str, object], value)
        return boundary_digest(tuple((key, fingerprint(item, seen)) for key, item in sorted(items.items())))
    if isinstance(value, (tuple, list)):
        return boundary_digest(tuple(fingerprint(item, seen) for item in cast(tuple[object, ...], value)))
    try:
        return boundary_digest(value)
    except ValueError as error:
        raise ValueError(
            f"{value!r}: workload methods may depend only on torch, math, pllm, plain values, enums and dataclasses"
        ) from error


def _is_class(value: object) -> TypeGuard[type[object]]:
    """Narrow a dependency to a class with known members (isinstance alone gives type[Unknown])."""
    return isinstance(value, type)


def _structure(cls: type[object], seen: frozenset[int]) -> str:
    """Sign an author's enum or dataclass by its members or field names and the code of its own methods."""
    members = tuple((member.name, fingerprint(member.value, seen)) for member in cls) if issubclass(cls, Enum) else ()
    names = tuple(field.name for field in fields(cls)) if is_dataclass(cls) else ()
    methods = tuple(
        (key, fingerprint(item, seen))
        for key, item in sorted(vars(cls).items())
        if isinstance(item, FunctionType) and not key.startswith("__")
    )
    return boundary_digest(cls.__module__, cls.__qualname__, members, names, methods)


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


_REJECTED_BUILTINS = {
    "setattr": "setattr() creates hidden mutable state; return the value instead, or declare "
    "pllm.RequestState/pllm.InstanceState and return a state update",
    "globals": "globals() exposes unsigned globals; pass the needed values as workload config fields",
    "locals": "locals() exposes the runtime namespace; name the needed local variables directly",
    "eval": "eval() hides code dependencies; write the expression in a Python helper function",
    "exec": "exec() hides code dependencies; write the statements in a Python helper function",
}


def _literal_call(code: CodeType, start: int, argument: int) -> tuple[int, str | None]:
    """Read a positional literal from a builtin call without executing its argument expressions."""
    instructions = tuple(get_instructions(code))
    offsets = {item.offset: index for index, item in enumerate(instructions)}
    # LOAD_GLOBAL's low bit requests NULL + callable for a direct call (Python 3.11+).
    if not (instructions[start].arg or 0) & 1:
        return 0, None
    pending: list[tuple[int, int, str | None]] = [(start + 1, 2, None)]
    visited: set[tuple[int, int, str | None]] = set()
    results: set[tuple[int, str | None]] = set()
    slot = 3 + argument
    call = "PRECALL" if any(item.opname == "PRECALL" for item in instructions) else "CALL"
    while pending:
        index, depth, literal = pending.pop()
        if (index, depth, literal) in visited:
            continue
        visited.add((index, depth, literal))
        item = instructions[index]
        if item.opname == call and depth == 2 + (item.arg or 0):
            results.add((item.arg or 0, literal))
            continue
        if item.opname in ("RETURN_VALUE", "RETURN_CONST", "RAISE_VARARGS", "CALL_FUNCTION_EX") or depth < 2:
            return 0, None
        if item.opname == "LOAD_CONST" and depth == slot - 1:
            literal = item.argval if isinstance(item.argval, str) else None
        for successor, jump in _successors(instructions, offsets, index):
            next_depth = depth + stack_effect(item.opcode, item.arg, jump=jump)
            retained = literal
            if next_depth < slot or (
                next_depth == slot
                and item.opname != "LOAD_CONST"
                and not item.opname.startswith(("STORE_", "POP_", "JUMP", "NOP", "EXTENDED_ARG"))
            ):
                retained = None
            pending.append((successor, next_depth, retained))
    return results.pop() if len(results) == 1 else (0, None)


def _successors(
    instructions: tuple[Instruction, ...], offsets: dict[int, int], index: int
) -> tuple[tuple[int, bool], ...]:
    item = instructions[index]
    if item.opcode in hasjabs or item.opcode in hasjrel:
        target = offsets[int(item.argval)]
        if item.opname.startswith("JUMP") and "IF" not in item.opname:
            return ((target, True),)
        return ((target, True), (index + 1, False))
    return ((index + 1, False),) if index + 1 < len(instructions) else ()


def _builtin_dependencies(function: FunctionType) -> tuple[tuple[str, str], ...]:
    """Admit explicit attribute/object reads; reject dependencies hidden in runtime namespaces."""
    codes = [function.__code__]
    imports: set[tuple[str, str]] = set()
    while codes:
        code = codes.pop()
        codes.extend(value for value in code.co_consts if isinstance(value, CodeType))
        for index, item in enumerate(get_instructions(code)):
            name = str(item.argval)
            if item.opname != "LOAD_GLOBAL" or name in function.__globals__:
                continue
            if name in _REJECTED_BUILTINS:
                raise ValueError(f"{function.__qualname__}: {_REJECTED_BUILTINS[name]}")
            if name not in ("getattr", "hasattr", "vars", "__import__"):
                continue
            count, literal = _literal_call(code, index, 0 if name == "__import__" else 1)
            if name == "vars" and count == 1:
                continue
            if name in ("getattr", "hasattr") and literal is not None:
                continue
            if name == "__import__" and literal is not None and count == 1:
                if literal == "pllm" or literal.startswith(("torch", "math")):
                    imports.add((literal, torch.__version__))
                    continue
            messages = {
                "getattr": 'getattr needs a literal attribute name; write getattr(obj, "field") or '
                'getattr(obj, "field", default) so the attribute dependency is recorded',
                "hasattr": 'hasattr needs a literal attribute name; write hasattr(obj, "field") '
                "so the attribute dependency is recorded",
                "vars": "vars() exposes runtime locals; write vars(obj) or explicit obj.field reads",
                "__import__": "__import__ needs a literal torch, math or pllm module name to sign its version; "
                'write import torch, math or pllm at module scope, or __import__("torch"); '
                "compute values from other modules outside the workload and pass them as config fields",
            }
            raise ValueError(f"{function.__qualname__}: {messages[name]}")
    return tuple(sorted(imports))


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
        raise TypeError(f"{function!r}: workload operations must be Python functions")
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
    """Every class attribute of the owner's MRO, base classes first and subclasses overriding."""
    return {name: item for cls in reversed(type(owner).__mro__) for name, item in vars(cls).items()}


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
