"""Planning keeps older versions only for the fields their readers pin; tracing checks the declared writer fields."""

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

import pllm
from pllm.config import ExecutionOptions, ModelConfig
from pllm.contracts.semantic_graph import StateEffectKind
from pllm.execution.task_memory import example, task_example
from pllm.testing_contracts import contract_facts
from pllm.workloads import ESampConfig, ESampWorkload, OverRIDEConfig, OverRIDEWorkload, TLMConfig, TLMWorkload
from pllm.workloads.state_producers import DecodeFactorsWorkload, seed_factors


class InstanceDecodeFactors(DecodeFactorsWorkload):
    """DecodeFactors with one factor state shared by every request's Adapter."""

    @property
    def factors(self) -> pllm.InstanceState:
        return pllm.InstanceState(
            {"a": pllm.Field(lambda i, o: (self.rank, i)), "b": pllm.Field(lambda i, o: (o, self.rank))},
            optimizer=self.optimizer,
            init=pllm.Computed(seed_factors),
        )


class FieldReads(TorchDispatchMode):
    """Record the state fields an operation reads as data: arguments neither aliased nor written by the aten
    schema, excluding `new_*`/`*_like` factories that take only a dtype and device."""

    def __init__(self, fields: dict[StorageWeakRef, tuple[str, int]]) -> None:
        super().__init__()
        self.fields, self.reads = fields, set[tuple[str, int]]()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        schema = func._schema
        name = schema.name.split("::")[-1]
        if not (name.startswith("new_") or name.endswith("_like")):
            values = (*args, *(kwargs.get(argument.name) for argument in schema.arguments[len(args) :]))
            for argument, value in zip(schema.arguments, values, strict=False):
                for tensor in tree_leaves(value):
                    if isinstance(tensor, torch.Tensor) and argument.alias_info is None:
                        key = self.fields.get(StorageWeakRef(tensor.untyped_storage()))
                        if key is not None:
                            self.reads.add(key)
        return func(*args, **kwargs)


@pytest.mark.parametrize(
    "workload",
    [
        ESampWorkload(ESampConfig(rank=2, intervention_beta=0.1)),
        OverRIDEWorkload(OverRIDEConfig(rank=2, num_iterations=3)),
        TLMWorkload(TLMConfig(), rank=2),
        InstanceDecodeFactors(),
    ],
    ids=lambda workload: type(workload).__name__,
)
def test_only_a_state_writer_reads_its_writer_fields(workload):
    """No reader task or model hook reads an AdamW step or moment, so older versions need not keep them."""
    _, facts = contract_facts(workload, ModelConfig("Qwen/Qwen2.5-0.5B-Instruct"))
    facts, binding = workload.specialize(facts, ExecutionOptions())
    writer_fields = {(str(state.resource), field) for state in binding.graph.states for field in state.writer_fields}
    assert writer_fields
    data, batch = example(binding, facts, ExecutionOptions())
    fields = {
        StorageWeakRef(tensor.untyped_storage()): (str(resource), index)
        for resource, scoped in batch.resources.items()
        for index, tensor in enumerate(scoped.roots[0])
    }
    writes = {
        str(task.symbol): {str(effect.resource) for effect in task.effects if effect.kind is not StateEffectKind.READ}
        for task in binding.graph.tasks
    }
    for symbol, operation in binding.executables(facts, ()).items():
        task = binding.tensor_tasks.get(symbol)
        with FieldReads(fields) as trace:
            operation.trace_body(task_example(data, None if task is None else task.phase), operation.scoped(batch))
        others = {(resource, field) for resource, field in trace.reads if resource not in writes[symbol]}
        assert not others & writer_fields
    bound = {str(item.resource) for item in (*binding.effects.overlay_bindings, *binding.effects.activation_bindings)}
    hooks = {(resource, int(view.field)) for (resource, _), view in binding.field_views.items() if resource in bound}
    assert not hooks & writer_fields
