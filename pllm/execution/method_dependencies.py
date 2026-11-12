"""Reject what a workload method may not depend on; workload reading checks each method and config value once.

Three rules remain: call a custom op through the object torch.library.custom_op returned (that object, not a
`torch.ops` name, is what carries the op to worker processes), keep state in declared pLLM states instead of
setattr, and load modules and tensors through state initializers instead of capturing them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from dis import Instruction, get_instructions
from inspect import getclosurevars
from types import CodeType, FunctionType, MethodType
from typing import cast

import torch

from pllm.execution.torch_private import dispatcher_namespace


def check_dependencies(value: object) -> None:
    """Walk a method or config value through function globals, closures, defaults and containers."""
    _check(value, set())


def _check(value: object, seen: set[int]) -> None:
    if id(value) in seen:
        return
    seen.add(id(value))
    if dispatcher_namespace(value) not in (None, "", "aten", "prims"):
        raise ValueError(OP_ACCESS)
    if isinstance(value, MethodType):
        _check(value.__func__, seen)
    elif isinstance(value, FunctionType):
        closure = getclosurevars(value)
        check_op_access(value.__code__, {**value.__globals__, **closure.nonlocals})
        _reject_setattr(value)
        for item in (
            *(value.__globals__[key] for key in global_names(value.__code__) if key in value.__globals__),
            # `__class__` is zero-argument `super()`'s cell: the class itself, not a dependency.
            *(item for key, item in closure.nonlocals.items() if key != "__class__"),
            *(value.__defaults__ or ()),
            *(value.__kwdefaults__ or {}).values(),
        ):
            _check(item, seen)
    elif isinstance(value, (torch.nn.Module, torch.Tensor)):
        raise ValueError(
            f"{type(value).__name__}: a workload cannot hold a module or tensor; save its weights and declare "
            "pllm.InstanceState(init=pllm.FromModule(<factory>, pretrained=path)), or field state with "
            "init=pllm.Pretrained(path)"
        )
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _check(getattr(value, field.name), seen)
    elif isinstance(value, Mapping):
        for item in cast(Mapping[object, object], value).values():
            _check(item, seen)
    elif isinstance(value, (tuple, list)):
        for item in cast(tuple[object, ...], value):
            _check(item, seen)


def global_names(code: CodeType) -> set[str]:
    """The module globals a function's code, including nested functions, loads by name."""
    names = {str(instruction.argval) for instruction in get_instructions(code) if instruction.opname == "LOAD_GLOBAL"}
    for value in code.co_consts:
        if isinstance(value, CodeType):
            names.update(global_names(value))
    return names


OP_ACCESS = (
    "call the op object you defined with torch.library.custom_op (from mykernels import gated_add), "
    "not torch.ops.<namespace>.<op>, so worker processes receive the op with its registration; use torch.ops.aten or torch.ops.prims "
    "only for built-in operators"
)


def _builtin_op_access(instructions: tuple[Instruction, ...]) -> bool:
    return bool(
        instructions
        and instructions[0].opname in ("LOAD_ATTR", "LOAD_METHOD")
        and instructions[0].argval in ("aten", "prims")
    )


def check_op_access(code: CodeType, globals_used: dict[str, object]) -> None:
    """Reject dispatcher lookups by name; only the op object reaches another process with its registration."""
    instructions = tuple(get_instructions(code))
    aliases: dict[str, object] = {}
    imported = ""
    for index, item in enumerate(instructions):
        tail = instructions[index + 1 :]
        if item.opname == "IMPORT_NAME":
            imported = str(item.argval)
            if imported == "torch" and tail and tail[0].opname == "STORE_FAST":
                aliases[str(tail[0].argval)] = torch
        if item.opname == "IMPORT_FROM" and item.argval == "ops" and imported == "torch":
            raise ValueError(OP_ACCESS)
        dependency = (
            globals_used.get(str(item.argval))
            if item.opname in ("LOAD_GLOBAL", "LOAD_DEREF")
            else aliases.get(str(item.argval))
            if item.opname == "LOAD_FAST"
            else None
        )
        if dependency is torch and tail:
            if tail[0].opname == "STORE_FAST":
                aliases[str(tail[0].argval)] = torch
            if tail[0].argval == "ops" and not _builtin_op_access(tail[1:]):
                raise ValueError(OP_ACCESS)
            if tail[0].argval in ("__dict__", "_ops"):
                raise ValueError(OP_ACCESS)
        if dependency is torch.ops and not _builtin_op_access(tail):
            raise ValueError(OP_ACCESS)
    for value in code.co_consts:
        if isinstance(value, CodeType):
            check_op_access(value, globals_used)


def _reject_setattr(function: FunctionType) -> None:
    """setattr hides mutable state from pLLM's state planning, also in nested functions."""
    if "setattr" in function.__globals__:
        return
    codes = [function.__code__]
    while codes:
        code = codes.pop()
        codes.extend(value for value in code.co_consts if isinstance(value, CodeType))
        if any(item.opname == "LOAD_GLOBAL" and item.argval == "setattr" for item in get_instructions(code)):
            raise ValueError(
                f"{function.__qualname__}: setattr() creates hidden mutable state; return the value instead, or "
                "declare pllm.RequestState/pllm.InstanceState and return a state update"
            )
