#!/usr/bin/env python3
"""Dependency-light BFCL-shaped toy fixtures for GTTA."""

from __future__ import annotations

from dataclasses import dataclass
import ast
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tllm.consumers.gtta.consumer import TrajectoryFeedback, parse_reward


@dataclass(frozen=True)
class BFCLToyTask:
    id: str
    task_id: str
    environment: str
    question: tuple[object, ...]
    functions: tuple[Mapping[str, Any], ...]
    initial_state: Mapping[str, Any]
    initial_observation: str
    candidate_actions: tuple[str, ...]
    gold_action: str
    tool_result: str
    next_observation: str
    reward: float
    environmental_dynamics: str

    def __post_init__(self) -> None:
        for field_name in ("id", "task_id", "environment", "initial_observation", "gold_action"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"BFCLToyTask requires {field_name}")
        if not self.candidate_actions:
            raise ValueError("BFCLToyTask requires candidate_actions")


@dataclass(frozen=True)
class BFCLToyStepResult:
    task: BFCLToyTask
    action: str
    tool_result: str
    next_observation: str
    reward: float
    feedback: TrajectoryFeedback


def load_bfcl_toy_tasks(path: str | Path) -> tuple[BFCLToyTask, ...]:
    tasks: list[BFCLToyTask] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if not isinstance(payload, Mapping):
                raise ValueError(f"BFCL toy line {line_number} must be a JSON object")
            tasks.append(_task_from_payload(payload))
    return tuple(tasks)


def execute_bfcl_toy_action(task: BFCLToyTask, action: str) -> BFCLToyStepResult:
    if not isinstance(task, BFCLToyTask):
        raise TypeError("task must be BFCLToyTask")
    normalized_action = str(action).strip()
    if normalized_action not in task.candidate_actions:
        raise ValueError("action must be one of the task candidate actions")

    state = _copy_toy_state(task.initial_state)
    if normalized_action.startswith("touch("):
        path = _parse_single_string_kwarg(normalized_action, function_name="touch", kwarg_name="path")
        files = list(state.get("files", ()))
        if path not in files:
            files.append(path)
        state["files"] = files
        contents = dict(state.get("contents", {}))
        contents.setdefault(path, "")
        state["contents"] = contents
        tool_result = f"created {path}"
    elif normalized_action == "ls()":
        tool_result = _format_files_observation(state)
    elif normalized_action.startswith("read("):
        path = _parse_single_string_kwarg(normalized_action, function_name="read", kwarg_name="path")
        files = list(state.get("files", ()))
        if path not in files:
            tool_result = f"missing {path}"
        else:
            contents = dict(state.get("contents", {}))
            tool_result = f"content {path}: {contents.get(path, '')}"
    else:
        raise ValueError(f"unsupported BFCL toy action `{normalized_action}`")

    next_observation = _format_files_observation(state)
    reward = float(task.reward if normalized_action == task.gold_action else -abs(float(task.reward or 1.0)))
    if normalized_action == task.gold_action:
        environmental_dynamics = task.environmental_dynamics
    else:
        environmental_dynamics = f"{normalized_action} did not complete the requested transition."
        if task.reward < 0:
            environmental_dynamics = task.environmental_dynamics

    feedback = TrajectoryFeedback(
        task_id=task.task_id,
        action=normalized_action,
        previous_observation=task.initial_observation,
        next_observation=next_observation,
        reward=reward,
        environmental_dynamics=environmental_dynamics,
    )
    return BFCLToyStepResult(
        task=task,
        action=normalized_action,
        tool_result=tool_result,
        next_observation=next_observation,
        reward=reward,
        feedback=feedback,
    )


def _task_from_payload(payload: Mapping[str, Any]) -> BFCLToyTask:
    return BFCLToyTask(
        id=str(_required(payload, "id")),
        task_id=str(_required(payload, "task_id")),
        environment=str(_required(payload, "environment")),
        question=tuple(_sequence(payload.get("question", ()))),
        functions=tuple(_mapping_sequence(payload.get("function", ()))),
        initial_state=dict(_mapping(_required(payload, "initial_state"), field_name="initial_state")),
        initial_observation=str(_required(payload, "initial_observation")),
        candidate_actions=tuple(str(item).strip() for item in _sequence(_required(payload, "candidate_actions"))),
        gold_action=str(_required(payload, "gold_action")).strip(),
        tool_result=str(payload.get("tool_result", "")),
        next_observation=str(_required(payload, "next_observation")),
        reward=parse_reward(_required(payload, "reward")),
        environmental_dynamics=str(_required(payload, "environmental_dynamics")),
    )


def _required(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"BFCL toy task missing `{key}`")
    return payload[key]


def _sequence(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("BFCL toy field must be a sequence")
    return value


def _mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"BFCL toy field `{field_name}` must be an object")
    return value


def _mapping_sequence(value: object) -> tuple[Mapping[str, Any], ...]:
    return tuple(_mapping(item, field_name="function item") for item in _sequence(value))


def _copy_toy_state(state: Mapping[str, Any]) -> dict[str, Any]:
    files = state.get("files", ())
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes, bytearray)):
        raise ValueError("ToyFileSystem state files must be a sequence")
    contents = state.get("contents", {})
    if not isinstance(contents, Mapping):
        raise ValueError("ToyFileSystem state contents must be an object")
    return {
        "files": [str(item) for item in files],
        "contents": {str(path): str(content) for path, content in contents.items()},
    }


def _format_files_observation(state: Mapping[str, Any]) -> str:
    files = [str(item) for item in state.get("files", ())]
    return f"files={files!r}"


def _parse_single_string_kwarg(action: str, *, function_name: str, kwarg_name: str) -> str:
    try:
        expr = ast.parse(action, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"invalid BFCL toy action `{action}`") from exc
    if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name) or expr.func.id != function_name:
        raise ValueError(f"expected {function_name}(...) action")
    values = {kw.arg: kw.value for kw in expr.keywords if kw.arg is not None}
    if set(values) != {kwarg_name}:
        raise ValueError(f"expected only `{kwarg_name}` keyword")
    value = values[kwarg_name]
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str) or not value.value:
        raise ValueError(f"`{kwarg_name}` must be a non-empty string")
    return str(value.value)
