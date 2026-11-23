"""CPU and GPU verification for pllm.model.LayerRoute dynamic layer routing."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass

import pytest
import torch

import pllm
from pllm.io import StateImport
from pllm.testing import RequestIdentity, check, check_on_gpu

CPU_CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=4, max_num_batched_tokens=32)


def test_layer_route_declaration_and_validation() -> None:
    with pytest.raises(TypeError, match="non-empty tuple"):
        pllm.model.LayerRoute("ctrl", over=())

    with pytest.raises(ValueError, match="duplicate"):
        pllm.model.LayerRoute("ctrl", over=(4, 4))

    class DuplicateNegative(pllm.Workload):
        ctrl = pllm.InstanceState({"layer": pllm.Field((), torch.int64)})
        route = pllm.model.LayerRoute("ctrl", over=(20, -4))  # 24-layer Qwen2.5-0.5B -> both resolve to 20

        @pllm.export(name="probe", max_rows=4, max_cols=1)
        def read(self, r: torch.Tensor = route.residual()) -> torch.Tensor:
            return r.float().norm(dim=-1, keepdim=True)

    with pytest.raises(ValueError, match="duplicate layers"):
        pllm.explain(DuplicateNegative(), CPU_CONFIG)

    class OutOfBounds(pllm.Workload):
        ctrl = pllm.InstanceState({"layer": pllm.Field((), torch.int64)})
        route = pllm.model.LayerRoute("ctrl", over=(4, 99))

        @pllm.export(name="probe", max_rows=4, max_cols=1)
        def read(self, r: torch.Tensor = route.residual()) -> torch.Tensor:
            return r.float().norm(dim=-1, keepdim=True)

    with pytest.raises(ValueError, match="must name layers within"):
        pllm.explain(OutOfBounds(), CPU_CONFIG)

    class BadDtype(pllm.Workload):
        ctrl = pllm.InstanceState({"layer": pllm.Field((), torch.float32)})
        route = pllm.model.LayerRoute("ctrl", over=(4, 8))

        @pllm.export(name="probe", max_rows=4, max_cols=1)
        def read(self, r: torch.Tensor = route.residual()) -> torch.Tensor:
            return r.float().norm(dim=-1, keepdim=True)

    with pytest.raises(ValueError, match="0-d int64 scalar"):
        pllm.explain(BadDtype(), CPU_CONFIG)

    class BadColocatedAdapterShape(pllm.Workload):
        ctrl = pllm.InstanceState(
            {
                "layer": pllm.Field((3,), torch.int64),
                "a": pllm.Field(lambda ins, _: (4, ins)),
                "b": pllm.Field(lambda _, outs: (outs, 4)),
            },
            init=pllm.Imported(),
        )
        route = pllm.model.LayerRoute("ctrl", over=(2, 6))
        lora = route.adapter(targets=("q_proj",), operator="low_rank")

    with pytest.raises(ValueError, match="0-d int64 scalar"):
        pllm.explain(BadColocatedAdapterShape(), CPU_CONFIG)

    class MiddleFieldOrderSingleTarget(pllm.Workload):
        ctrl = pllm.InstanceState(
            {
                "a": pllm.Field(lambda ins, _: (4, ins)),
                "layer": pllm.Field((), torch.int64),
                "b": pllm.Field(lambda _, outs: (outs, 4)),
            },
            init=pllm.Imported(),
        )
        route = pllm.model.LayerRoute("ctrl", over=(-1, 2))
        lora = route.adapter(targets=("q_proj",), operator="low_rank")

    from pllm.testing_contracts import contract_facts

    _, facts = contract_facts(MiddleFieldOrderSingleTarget(), CPU_CONFIG)
    _, decl = MiddleFieldOrderSingleTarget().specialize(facts, CPU_CONFIG.execution)
    assert [name for name, _ in decl.effects.import_fields] == ["a_0", "layer", "b_0"]

    class BadBankShape(pllm.Workload):
        ctrl = pllm.InstanceState({"layer": pllm.Field((), torch.int64)}, init=pllm.Imported())
        bank = pllm.InstanceState(
            {"w": pllm.Field((7, 7))},
            layers=(2, 6),
            init=pllm.Computed(lambda fields, g: None),
        )
        route = pllm.model.LayerRoute("ctrl", over=(2, 6))
        overlay = route.adapter("bank", targets=("q_proj",), weight="w", operator="dense")

    with pytest.raises(ValueError, match="must be shaped"):
        pllm.explain(BadBankShape(), CPU_CONFIG)

    route_val = pllm.model.LayerRoute("ctrl", over=(2, 6))
    with pytest.raises(ValueError, match="decode=True"):
        route_val.patch(decode=False, refresh="step")
    with pytest.raises(ValueError, match="decode=True"):
        route_val.patch(decode=False, on_missing="hold")

    class ConflictingOver(pllm.Workload):
        ctrl = pllm.InstanceState(
            {"layer": pllm.Field((), torch.int64), "vec": lambda m: (m.hidden_size,)},
            init=pllm.Imported(),
        )
        r1 = pllm.model.LayerRoute("ctrl", over=(4, 8))
        r2 = pllm.model.LayerRoute("ctrl", over=(4, 12))
        patch = r1.patch(operator="add", decode=True)

        @pllm.export(name="probe", max_rows=4, max_cols=1)
        def read(self, r: torch.Tensor = r2.residual()) -> torch.Tensor:
            return r.float().norm(dim=-1, keepdim=True)

    with pytest.raises(ValueError, match="conflicting candidate sets"):
        pllm.explain(ConflictingOver(), CPU_CONFIG)

    class ColocatedOptimizerRejected(pllm.Workload):
        ctrl = pllm.InstanceState(
            {
                "layer": pllm.Field((), torch.int64),
                "a": pllm.Field(lambda ins, _: (4, ins)),
                "b": pllm.Field(lambda _, outs: (outs, 4)),
            },
            optimizer=pllm.AdamW(lr=1e-3),
            init=pllm.Computed(lambda f, g: f["a_0"].fill_(0.1)),
        )
        route = pllm.model.LayerRoute("ctrl", over=(2, 6))
        lora = route.adapter(targets=("q_proj",), operator="low_rank")

    with pytest.raises(ValueError, match="cannot declare optimizer="):
        pllm.explain(ColocatedOptimizerRejected(), CPU_CONFIG)

    class RoutedCausalModelRejected(pllm.Workload):
        ctrl = pllm.InstanceState({"layer": pllm.Field((), torch.int64)})
        weights = pllm.InstanceState(
            {
                "a": pllm.Field(lambda ins, _: (4, ins)),
                "b": pllm.Field(lambda _, outs: (outs, 4)),
            },
            optimizer=pllm.AdamW(lr=1e-3),
            init=pllm.Computed(lambda f, g: f["a_0"].fill_(0.1)),
        )
        route = pllm.model.LayerRoute("ctrl", over=(2, 6))
        lora = route.adapter("weights", targets=("q_proj",), operator="low_rank")

        @pllm.train(state="weights")
        def fit(
            self,
            tokens: torch.Tensor = pllm.batch("input_ids"),
            model: pllm.CausalModel = pllm.causal_model("lora"),
        ) -> torch.Tensor:
            return model(tokens).sum(dim=-1)

    with pytest.raises(ValueError, match="does not support dynamically routed LayerRoute.adapter"):
        pllm.explain(RoutedCausalModelRejected(), CPU_CONFIG)

    class RoutedAdapterWithTrainInForwardRejected(pllm.Workload):
        ctrl = pllm.InstanceState(
            {
                "layer": pllm.Field((), torch.int64),
                "a": pllm.Field(lambda ins, _: (4, ins)),
                "b": pllm.Field(lambda _, outs: (outs, 4)),
            },
            init=pllm.Imported(),
        )
        fast = pllm.RequestState({"w": pllm.Field(lambda ins, outs: (outs, ins))})
        route = pllm.model.LayerRoute("ctrl", over=(2, 6))
        lora = route.adapter(targets=("q_proj",), operator="low_rank")

        @pllm.train_in_forward(state="fast", targets=(pllm.model.Projection("v_proj", 0),), weight="w")
        def adapt_fast(
            self,
            x: torch.Tensor = pllm.operand("input"),
            w: torch.Tensor = pllm.parameter("fast", "w"),
        ) -> pllm.UpdateOutput:
            return pllm.UpdateOutput(
                x @ w.to(x.dtype).T,
                x.new_ones((), dtype=torch.bool),
                x.new_ones((), dtype=torch.long),
                x.new_tensor(len(x), dtype=torch.long),
            )

    with pytest.raises(ValueError, match="does not yet combine with a pllm.model.Adapter"):
        pllm.explain(RoutedAdapterWithTrainInForwardRejected(), CPU_CONFIG)

    class LiveRouteKeyedAdmissionRejected(pllm.Workload):
        req = pllm.RequestState(
            {"layer": pllm.Field((), torch.int64), "vec": lambda m: (m.hidden_size,)},
            key_fields=("route",),
            init=pllm.Imported(),
        )
        route = pllm.model.LayerRoute("req", over=(2, 6))
        patch = route.patch(operator="add", decode=True, refresh="admission")

    with pytest.raises(ValueError, match='refresh="admission" on RequestState requires a frozen per-request state'):
        pllm.explain(LiveRouteKeyedAdmissionRejected(), CPU_CONFIG)

    class WritableRequestAdmissionRejected(pllm.Workload):
        req = pllm.RequestState({"layer": pllm.Field((), torch.int64), "vec": lambda m: (m.hidden_size,)})
        route = pllm.model.LayerRoute("req", over=(2, 6))
        patch = route.patch(operator="add", decode=True, refresh="admission")

        @pllm.on(pllm.Phase.BEFORE_FORWARD)
        @pllm.write(state="req", fields=("layer",))
        def flip(self, cur: torch.Tensor = pllm.parameter("req", "layer", per_row=True)) -> tuple[torch.Tensor]:
            return (cur + 2,)

    with pytest.raises(ValueError, match='refresh="admission" on RequestState requires a frozen per-request state'):
        pllm.explain(WritableRequestAdmissionRejected(), CPU_CONFIG)


def test_layer_route_prefix_caching_and_preemption_policies() -> None:
    from pllm.execution.step_driver import StepDriver
    from pllm.testing_contracts import contract_facts

    class ObservationOnlyRoute(pllm.Workload):
        ctrl = pllm.InstanceState({"layer": pllm.Field((), torch.int64)}, init=pllm.Imported())
        route = pllm.model.LayerRoute("ctrl", over=(2, 8, -1))

        @pllm.export(name="probe", max_rows=4, max_cols=2)
        def read(self, r: torch.Tensor = route.residual()) -> torch.Tensor:
            return r[:, :2].float()

    obs = ObservationOnlyRoute()
    cached_cfg = pllm.ModelConfig(
        "Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=4, max_num_batched_tokens=32, enable_prefix_caching=True
    )
    report = pllm.explain(obs, cached_cfg)
    assert report.eager_reason == ""
    _, facts = contract_facts(obs, CPU_CONFIG)
    _, decl = obs.specialize(facts, CPU_CONFIG.execution)
    dep = obs.bind_backend(facts, obs.plan_backend(facts, decl), decl)
    driver = StepDriver(dep.execution, dep.attachment)
    try:
        assert driver.preemption == "keep"
    finally:
        driver.close()

    class MutableRoutedEdit(pllm.Workload):
        ctrl = pllm.InstanceState(
            {"layer": pllm.Field((), torch.int64), "vec": lambda m: (m.hidden_size,)},
            init=pllm.Imported(),
        )
        route = pllm.model.LayerRoute("ctrl", over=(2, 8))
        patch = route.patch(operator="add", decode=True)

    edit = MutableRoutedEdit()
    with pytest.raises(ValueError, match="enable_prefix_caching=False"):
        pllm.explain(edit, cached_cfg)

    _, edit_facts = contract_facts(edit, CPU_CONFIG)
    _, edit_decl = edit.specialize(edit_facts, CPU_CONFIG.execution)
    edit_dep = edit.bind_backend(edit_facts, edit.plan_backend(edit_facts, edit_decl), edit_decl)
    edit_driver = StepDriver(edit_dep.execution, edit_dep.attachment)
    edit_dep.execution.admit_instances()
    try:
        assert edit_driver.preemption == "reset"
        identities = (RequestIdentity("r0", 0, 0),)
        edit_driver.stage(identities, None)
        edit_driver.schedule((), (), ("r0",))
        with pytest.raises(RuntimeError, match="preempted a request"):
            edit_driver.schedule((), ("r0",), ())
    finally:
        edit_driver.close()

    class SharedSampleIndexOnlineImport(pllm.Workload):
        req = pllm.RequestState(
            {"layer": pllm.Field((), torch.int64), "vec": lambda m: (m.hidden_size,)},
            key_fields=("sample_index",),
            init=pllm.Imported(),
        )
        route = pllm.model.LayerRoute("req", over=(2, 8))
        patch = route.patch(operator="add", decode=True)

    shared = SharedSampleIndexOnlineImport()
    _, sh_facts = contract_facts(shared, CPU_CONFIG)
    _, sh_decl = shared.specialize(sh_facts, CPU_CONFIG.execution)
    sh_dep = shared.bind_backend(sh_facts, shared.plan_backend(sh_facts, sh_decl), sh_decl)
    sh_driver = StepDriver(sh_dep.execution, sh_dep.attachment)
    try:
        assert sh_driver.preemption == "reset"
    finally:
        sh_driver.close()


def test_layer_route_explain_and_step_refresh_page_budget() -> None:
    class RoutedSteerAndProbe(pllm.Workload):
        ctrl = pllm.InstanceState(
            {"layer": pllm.Field((), torch.int64), "vector": lambda m: (m.hidden_size,)},
            init=pllm.Imported(),
        )
        route = pllm.model.LayerRoute("ctrl", over=(4, 12, -1))
        patch = route.patch(operator="add", decode=True)

        @pllm.export(name="probe", max_rows=4, max_cols=2)
        def read(self, r: torch.Tensor = route.residual()) -> torch.Tensor:
            return r[:, :2].float()

    report = pllm.explain(RoutedSteerAndProbe(), CPU_CONFIG)
    assert report.eager_reason == ""
    (state_cost,) = report.states
    # refresh="step" avoids pinning one snapshot per admitted sequence (2 pages instead of 2 + max_num_seqs).
    assert max(state_cost.pages) <= 3
    assert any(item.setting == "LayerRoute" and item.value == "over=(4, 12, 23)" for item in report.escape_hatches)
    result = check(RoutedSteerAndProbe(), CPU_CONFIG)
    assert "request isolation" in result.passed
    assert "row permutation" in result.passed


def test_layer_route_admission_pin_in_forward_and_custom_forward_cpu() -> None:
    from pllm.backends.vllm.activation import VllmActivationPatch
    from pllm.contracts.sampler import SamplerFeedback
    from pllm.execution.step_driver import StepDriver, StepOpen
    from pllm.testing_contracts import _resolve_routed_hidden, contract_facts, contract_step_input

    # 1) Admission pin captures batch StateImport published at request.prepare (version 1, not version 0).
    class AdmissionPinWithRequestState(pllm.Workload):
        inst = pllm.InstanceState(
            {"layer": pllm.Field((), torch.int64), "vec": lambda m: (m.hidden_size,)},
            init=pllm.Imported(),
        )
        req = pllm.RequestState({"bias": (1,)})
        route = pllm.model.LayerRoute("inst", over=(2, 6))
        patch = route.patch(operator="add", decode=True, refresh="admission")

        @pllm.write(state="req", fields=("bias",))
        def touch(self, b: torch.Tensor = pllm.parameter("req", "bias", per_row=True)) -> tuple[torch.Tensor]:
            return (b + 1.0,)

    adm_wl = AdmissionPinWithRequestState()
    shape, facts = contract_facts(adm_wl, CPU_CONFIG)
    _, decl = adm_wl.specialize(facts, CPU_CONFIG.execution)
    dep = adm_wl.bind_backend(facts, adm_wl.plan_backend(facts, decl), decl)
    driver = StepDriver(dep.execution, dep.attachment)
    dep.execution.admit_instances()
    try:
        identities = (RequestIdentity("r0", 0, 0),)
        staged = StateImport((torch.tensor(6, dtype=torch.int64), torch.ones(shape.hidden_size)))
        driver.stage(identities, staged)
        driver.schedule((), (), ("r0",))
        step0 = contract_step_input(shape, facts, ((0, 5, 0),), 0, None, identities)
        ctx = driver.begin(step0, None)
        assert ctx is not None
        res_inst = pllm.contracts.semantic_graph.StateResource("inst")
        ((_, ref_adm),) = ctx.read_versions(identities, res_inst, dep.execution.directory, latest=False)
        ((_, ref_step),) = ctx.read_versions(identities, res_inst, dep.execution.directory, latest=True)
        assert ref_adm.version_id == 1
        assert ref_step.version_id == 1
        driver.after_forward(step0, None)
        driver.sample(
            SamplerFeedback(step0.sampler_rows, torch.zeros((1, shape.vocab_size)), torch.zeros(1, dtype=torch.long)),
            None,
        )
        driver.commit()
    finally:
        driver.finish_call(wait=True)
        driver.release()
        driver.close()

    # 2) One-time StateImport + step-refreshed patch survives VllmActivationPatch.prepare on decode step 1.
    class ImportedStepPatch(pllm.Workload):
        ctrl = pllm.InstanceState(
            {"layer": pllm.Field((), torch.int64), "vec": lambda m: (m.hidden_size,)},
            init=pllm.Imported(),
        )
        route = pllm.model.LayerRoute("ctrl", over=(2, 6))
        patch = route.patch(operator="add", decode=True)

    imp_wl = ImportedStepPatch()
    imp_shape, imp_facts = contract_facts(imp_wl, CPU_CONFIG)
    _, imp_decl = imp_wl.specialize(imp_facts, CPU_CONFIG.execution)
    imp_dep = imp_wl.bind_backend(imp_facts, imp_wl.plan_backend(imp_facts, imp_decl), imp_decl)
    imp_driver = StepDriver(imp_dep.execution, imp_dep.attachment)
    vllm_patch = VllmActivationPatch(
        imp_dep.attachment.spec.activation_bindings, imp_dep.attachment.probe.max_packed_rows, imp_dep.execution.arena
    )
    imp_dep.execution.admit_instances()
    try:
        identities = (RequestIdentity("r0", 0, 0),)
        imp_driver.import_state(StateImport((torch.tensor(6, dtype=torch.int64), torch.ones(imp_shape.hidden_size))))
        imp_driver.stage(identities, None)
        imp_driver.schedule((), (), ("r0",))
        s0 = contract_step_input(imp_shape, imp_facts, ((0, 5, 0),), 0, None, identities)
        c0 = imp_driver.begin(s0, None)
        assert c0 is not None
        vllm_patch.prepare(s0.rows, c0, imp_dep.execution.directory)
        imp_driver.after_forward(s0, None)
        imp_driver.commit()

        s1 = contract_step_input(imp_shape, imp_facts, ((0, 1, 5),), 1, None, identities)
        c1 = imp_driver.begin(s1, None)
        assert c1 is not None
        vllm_patch.prepare(s1.rows, c1, imp_dep.execution.directory)
        assert int(vllm_patch.patches[pllm.contracts.semantic_graph.StateResource("ctrl"), 6].table[0, 0].item()) == 1
        assert int(vllm_patch.patches[pllm.contracts.semantic_graph.StateResource("ctrl"), 2].table[0, 0].item()) == 0
        imp_driver.after_forward(s1, None)
        imp_driver.commit()
    finally:
        imp_driver.finish_call(wait=True)
        imp_driver.release()
        imp_driver.close()

    # 3) StepOpen.IN_FORWARD (@pllm.train_in_forward(visibility="exact") + separate imported LayerRoute + writer) across 2 decode steps.
    in_fwd_events: list[pllm.io.TraceEvent] = []

    class InForwardPlusRoutedTap(pllm.Workload):
        fast = pllm.RequestState({"w": pllm.Field(lambda ins, outs: (outs, ins))})
        ctrl = pllm.InstanceState({"layer": pllm.Field((), torch.int64)}, init=pllm.Imported())
        counter = pllm.InstanceState({"steps": pllm.Field((), torch.int64)})
        route = pllm.model.LayerRoute("ctrl", over=(2, 6))

        @pllm.on(pllm.Phase.BEFORE_FORWARD)
        @pllm.write(state="counter", fields=("steps",))
        def bump(self, s: torch.Tensor = pllm.parameter("counter", "steps")) -> tuple[torch.Tensor]:
            return (s + 1,)

        @pllm.train_in_forward(
            state="fast", targets=(pllm.model.Projection("v_proj", 0),), weight="w", visibility="exact"
        )
        def adapt(
            self,
            x: torch.Tensor = pllm.operand("input"),
            w: torch.Tensor = pllm.parameter("fast", "w"),
        ) -> pllm.UpdateOutput:
            return pllm.UpdateOutput(
                x @ w.to(x.dtype).T,
                x.new_ones((), dtype=torch.bool),
                x.new_ones((), dtype=torch.long),
                x.new_tensor(len(x), dtype=torch.long),
            )

        @pllm.export(name="tap", max_rows=4, max_cols=2)
        def observe(self, r: torch.Tensor = route.residual()) -> torch.Tensor:
            return r[:, :2].float()

        def on_export(self, event: pllm.io.TraceEvent) -> None:
            in_fwd_events.append(event)

    inf_wl = InForwardPlusRoutedTap()
    inf_shape, inf_facts = contract_facts(inf_wl, CPU_CONFIG)
    _, inf_decl = inf_wl.specialize(inf_facts, CPU_CONFIG.execution)
    inf_dep = inf_wl.bind_backend(inf_facts, inf_wl.plan_backend(inf_facts, inf_decl), inf_decl)
    inf_driver = StepDriver(inf_dep.execution, inf_dep.attachment)
    assert inf_driver.open is StepOpen.IN_FORWARD
    inf_dep.execution.admit_instances()
    try:
        identities = (RequestIdentity("r0", 0, 0),)
        inf_driver.stage(identities, StateImport((torch.tensor(6, dtype=torch.int64),)))
        inf_driver.schedule((), (), ("r0",))
        step0 = contract_step_input(inf_shape, inf_facts, ((0, 5, 0),), 0, None, identities)
        ctx = inf_driver.begin(step0, None)
        assert ctx is not None
        (transition_task,) = (
            pllm.contracts.semantic_graph.TaskId(i)
            for i, t in enumerate(inf_dep.attachment.plan.semantic_graph.tasks)
            if t.symbol == inf_dep.attachment.spec.in_forward
        )
        inf_driver.part(ctx, step0, transition_task, complete=True)
        inf_driver.after_forward(step0, None)
        inf_driver.sample(
            SamplerFeedback(
                step0.sampler_rows, torch.zeros((1, inf_shape.vocab_size)), torch.zeros(1, dtype=torch.long)
            ),
            None,
        )
        inf_driver.commit()

        for step_id in (1, 2):
            step_d = contract_step_input(inf_shape, inf_facts, ((0, 1, 4 + step_id),), step_id, None, identities)
            ctx_d = inf_driver.begin(step_d, None)
            assert ctx_d is not None
            resolved_d = _resolve_routed_hidden(inf_shape, inf_facts, inf_decl, inf_dep.execution, None, step_d, ctx_d)
            inf_driver.after_forward(resolved_d, None)
            inf_driver.sample(
                SamplerFeedback(
                    resolved_d.sampler_rows, torch.zeros((1, inf_shape.vocab_size)), torch.zeros(1, dtype=torch.long)
                ),
                None,
            )
            inf_driver.commit()
        inf_dep.execution.trace.drain()
        assert [ev.layers for ev in in_fwd_events[-2:]] == [(6,), (6,)]
    finally:
        inf_driver.finish_call(wait=True)
        inf_driver.release()
        inf_driver.close()

    # 4) @pllm.custom_forward: AFTER_FORWARD writer avoids cycle, BEFORE_FORWARD writer attaches published_0.
    class CustomForwardNextStepRoute(pllm.Workload):
        ctrl = pllm.InstanceState({"layer": pllm.Field((), torch.int64)})
        bank = pllm.InstanceState(
            {"vec": pllm.Field(lambda w, _: (w,))},
            layers=(2, 6),
            init=pllm.Computed(lambda fields, g: None),
        )
        route = pllm.model.LayerRoute("ctrl", over=(2, 6))
        patch = route.patch("bank", operator="add", decode=True)

        @pllm.custom_forward(tokens="decode")
        def prog(self, layers: pllm.model.Layers) -> torch.Tensor:
            return layers(0, 24)[0]

        @pllm.write(state="ctrl", fields=("layer",))
        def pick_next(self, h: torch.Tensor = pllm.residual(-1)) -> tuple[torch.Tensor]:
            choice = torch.where(h.float().sum() >= 0, torch.tensor(2), torch.tensor(6)).to(torch.int64)
            return (choice,)

    cf_cfg = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", cuda_graphs="auto")
    cf_report = pllm.explain(CustomForwardNextStepRoute(), cf_cfg)
    assert "custom_forward" in cf_report.eager_reason

    class CustomForwardSameStepRoute(pllm.Workload):
        ctrl = pllm.InstanceState({"layer": pllm.Field((), torch.int64)})
        bank = pllm.InstanceState(
            {"vec": pllm.Field(lambda w, _: (w,))},
            layers=(2, 6),
            init=pllm.Computed(lambda fields, g: None),
        )
        route = pllm.model.LayerRoute("ctrl", over=(2, 6))
        patch = route.patch("bank", operator="add", decode=True)

        @pllm.on(pllm.Phase.BEFORE_FORWARD)
        @pllm.write(state="ctrl", fields=("layer",))
        def pick_now(self, cur: torch.Tensor = pllm.parameter("ctrl", "layer")) -> tuple[torch.Tensor]:
            return (torch.zeros_like(cur) + 6,)

        @pllm.custom_forward(tokens="decode")
        def prog(self, layers: pllm.model.Layers) -> torch.Tensor:
            return layers(0, 24)[0]

    _, cf_facts = contract_facts(CustomForwardSameStepRoute(), cf_cfg)
    _, cf_decl = CustomForwardSameStepRoute().specialize(cf_facts, cf_cfg.execution)
    fwd_task = next(t for t in cf_decl.graph.tasks if t.symbol == "forward")
    assert any(inp.name == "published_0" for inp in fwd_task.inputs)

    skipped_events: list[pllm.io.TraceEvent] = []

    class TruncatedCustomForwardTap(pllm.Workload):
        ctrl = pllm.InstanceState(
            {"layer": pllm.Field((), torch.int64)},
            init=pllm.Computed(lambda fields, g: fields["layer"].fill_(2)),
        )
        route = pllm.model.LayerRoute("ctrl", over=(2, -1))

        @pllm.custom_forward(tokens="decode")
        def prog(self, layers: pllm.model.Layers) -> torch.Tensor:
            return layers(0, 1)[0]

        @pllm.export(name="skipped", max_rows=8, max_cols=4)
        def observe(self, r: torch.Tensor = route.residual()) -> torch.Tensor:
            return r[:, :4].float()

        def on_export(self, event: pllm.io.TraceEvent) -> None:
            skipped_events.append(event)

    skipped_check = check(TruncatedCustomForwardTap(), CPU_CONFIG)
    assert "request isolation" in skipped_check.passed
    assert skipped_events
    assert all(layer == -1 for ev in skipped_events for layer in ev.layers)
    assert all(torch.count_nonzero(ev.tensor()) == 0 for ev in skipped_events)


def test_layer_route_aot_functionalization_and_unvisited_tap_provenance_cpu() -> None:
    from pllm.backends.vllm.hidden_taps import Tap
    from pllm.backends.vllm.model_sites import _SITES, SublayerBoundary
    from pllm.contracts.tensor_layout import LayoutFieldId, TensorDType, TensorFieldSpec, TensorLayoutSpec
    from pllm.execution.ptts.slot_pools import SlotPools

    pool = SlotPools(
        TensorLayoutSpec("test.hidden@1", (TensorFieldSpec(LayoutFieldId(0), (4,), TensorDType.FLOAT32, 16),)),
        (4,),
        torch.device("cpu"),
    )
    indices = torch.tensor([[0, -1], [1, -1]], dtype=torch.long)
    row_layers = torch.tensor([2, -1], dtype=torch.long)
    slot_layers = torch.full((4,), -1, dtype=torch.long)
    tap_l2 = Tap(routed=[(2, pool, indices, row_layers, slot_layers)])
    tap_l5 = Tap(routed=[(5, pool, indices, row_layers, slot_layers)])

    # Verify an unvisited candidate (only tap_l5 runs when row_layers == 2) leaves pool=0 and slot_layers=-1:
    pool.tensors[0].zero_()
    tap_l5.copy(torch.full((1, 4), 9.0))
    assert int(slot_layers[1].item()) == -1
    assert torch.count_nonzero(pool.tensors[0][1]) == 0

    # Verify PyTorch AOT functionalization (`aot_eager`) on SublayerBoundary.graph_forward preserves tap writes:
    class DummySites:
        def __init__(self) -> None:
            self.streams = 1
            self.hidden = 4
            self.count = 1
            self._requests_of = torch.zeros(2, dtype=torch.long)
            self.wave_inverse = torch.full((2,), -1, dtype=torch.long)
            self.state_values: dict[object, torch.Tensor] = {}
            self.result_values: dict[object, torch.Tensor] = {}
            self.earlier_values: dict[object, torch.Tensor] = {}
            self._activation = None
            self._memories: dict[str, object] = {}
            self.hooks: list[SublayerBoundary] = []

        def run_hook(self, hook: int, inputs: list[torch.Tensor], outputs: list[torch.Tensor]) -> None:
            from pllm.backends.vllm.model_sites import model_site

            model_site(inputs, outputs, [self._requests_of], [], hook)

    dummy = DummySites()
    boundary = SublayerBoundary(
        dummy,  # type: ignore[arg-type]
        lambda b, s=None: b if s is None else (b, b + s),
        ("mlp", 2),
        None,
        ("attention", 3),
        None,
        None,
        (),
        (None, tap_l2),
    )
    dummy.hooks.append(boundary)
    _SITES[dummy._requests_of.data_ptr()] = dummy  # type: ignore[assignment]
    try:
        compiled = torch.compile(boundary.graph_forward, backend="aot_eager", fullgraph=True)
        _, residual = compiled(torch.full((1, 4), 3.0), torch.full((1, 4), 4.0))
        torch.testing.assert_close(residual, torch.full((1, 4), 7.0))
        torch.testing.assert_close(pool.tensors[0][1], torch.full((4,), 7.0))
        assert int(slot_layers[1].item()) == 2
    finally:
        _SITES.pop(dummy._requests_of.data_ptr(), None)


def test_layer_route_import_validation_and_export_provenance_cpu() -> None:
    events: list[pllm.io.TraceEvent] = []

    class LiveProbe(pllm.Workload):
        ctrl = pllm.InstanceState({"layer": pllm.Field((), torch.int64)}, init=pllm.Imported())
        route = pllm.model.LayerRoute("ctrl", over=(2, 6, 10))

        @pllm.export(name="probe", max_rows=4, max_cols=4)
        def capture(self, r: torch.Tensor = route.residual()) -> torch.Tensor:
            return r[:, :4].float()

        def on_export(self, event: pllm.io.TraceEvent) -> None:
            events.append(event)

    workload = LiveProbe()
    from pllm.contracts.sampler import SamplerFeedback
    from pllm.execution.step_driver import StepDriver
    from pllm.testing_contracts import _resolve_routed_hidden, contract_facts, contract_step_input

    shape, facts = contract_facts(workload, CPU_CONFIG)
    _, declaration = workload.specialize(facts, CPU_CONFIG.execution)
    deployment = workload.bind_backend(facts, workload.plan_backend(facts, declaration), declaration)
    driver = StepDriver(deployment.execution, deployment.attachment)
    deployment.execution.admit_instances()
    try:
        with pytest.raises(ValueError, match="not in over=\\(2, 6, 10\\) or -1"):
            driver.import_state(StateImport((torch.tensor(5, dtype=torch.int64),)))

        identities = (RequestIdentity("r0", 0, 0), RequestIdentity("r1", 1, 0))
        driver.stage(identities, None)
        driver.schedule((), (), ("r0", "r1"))
        step0 = contract_step_input(shape, facts, ((0, 5, 0), (1, 1, 0)), 0, None, identities)
        driver.begin(step0, None)
        driver.after_forward(step0, None)
        if driver.step is not None:
            driver.sample(
                SamplerFeedback(
                    step0.sampler_rows, torch.zeros((2, shape.vocab_size)), torch.zeros(2, dtype=torch.long)
                ),
                None,
            )
            driver.commit()

        for step_id, target_layer in ((1, 6), (2, -1), (3, 10)):
            driver.import_state(StateImport((torch.tensor(target_layer, dtype=torch.int64),)))
            driver.schedule((), (), ())
            raw = contract_step_input(shape, facts, ((0, 1, 4 + step_id), (1, 1, step_id)), step_id, None, identities)
            ctx = driver.begin(raw, None)
            resolved = _resolve_routed_hidden(shape, facts, declaration, deployment.execution, None, raw, ctx)
            driver.after_forward(resolved, None)
            driver.sample(
                SamplerFeedback(
                    resolved.sampler_rows, torch.zeros((2, shape.vocab_size)), torch.zeros(2, dtype=torch.long)
                ),
                None,
            )
            driver.commit()

        deployment.execution.trace.drain()
        assert [event.layers for event in events[-3:]] == [(6, 6), (-1, -1), (10, 10)]
        assert torch.count_nonzero(events[-2].tensor()) == 0
        assert torch.count_nonzero(events[-3].tensor()) > 0
        assert not torch.allclose(events[-3].tensor(), events[-1].tensor())
    finally:
        driver.finish_call(wait=True)
        driver.release()
        driver.close()


def test_layer_route_before_forward_write_and_all_tokens_export_cpu() -> None:
    from pllm.contracts.sampler import SamplerFeedback
    from pllm.execution.step_driver import StepDriver
    from pllm.testing_contracts import _resolve_routed_hidden, contract_facts, contract_step_input

    # 1) BEFORE_FORWARD @pllm.write updates a zero-initialized route to layer 2 -> TraceEvent.layers must be (2, 2).
    bf_events: list[pllm.io.TraceEvent] = []

    class BeforeForwardRouteWrite(pllm.Workload):
        ctrl = pllm.InstanceState({"layer": pllm.Field((), torch.int64)})
        route = pllm.model.LayerRoute("ctrl", over=(2, 8))

        @pllm.on(pllm.Phase.BEFORE_FORWARD)
        @pllm.write(state="ctrl", fields=("layer",))
        def set_layer(self, cur: torch.Tensor = pllm.parameter("ctrl", "layer")) -> tuple[torch.Tensor]:
            return (torch.zeros_like(cur) + 2,)

        @pllm.export(name="probe", max_rows=4, max_cols=2)
        def observe(self, r: torch.Tensor = route.residual()) -> torch.Tensor:
            return r[:, :2].float()

        def on_export(self, event: pllm.io.TraceEvent) -> None:
            bf_events.append(event)

    bf_wl = BeforeForwardRouteWrite()
    shape, facts = contract_facts(bf_wl, CPU_CONFIG)
    _, decl = bf_wl.specialize(facts, CPU_CONFIG.execution)
    dep = bf_wl.bind_backend(facts, bf_wl.plan_backend(facts, decl), decl)
    driver = StepDriver(dep.execution, dep.attachment)
    dep.execution.admit_instances()
    try:
        identities = (RequestIdentity("r0", 0, 0), RequestIdentity("r1", 1, 0))
        driver.stage(identities, None)
        driver.schedule((), (), ("r0", "r1"))
        step0 = contract_step_input(shape, facts, ((0, 5, 0), (1, 1, 0)), 0, None, identities)
        driver.begin(step0, None)
        driver.after_forward(step0, None)
        if driver.step is not None:
            driver.sample(
                SamplerFeedback(
                    step0.sampler_rows, torch.zeros((2, shape.vocab_size)), torch.zeros(2, dtype=torch.long)
                ),
                None,
            )
            driver.commit()

        raw1 = contract_step_input(shape, facts, ((0, 1, 5), (1, 1, 1)), 1, None, identities)
        ctx1 = driver.begin(raw1, None)
        resolved1 = _resolve_routed_hidden(shape, facts, decl, dep.execution, None, raw1, ctx1)
        driver.after_forward(resolved1, None)
        driver.sample(
            SamplerFeedback(
                resolved1.sampler_rows, torch.zeros((2, shape.vocab_size)), torch.zeros(2, dtype=torch.long)
            ),
            None,
        )
        driver.commit()
        dep.execution.trace.drain()
        assert bf_events[-1].layers == (2, 2)
        assert torch.count_nonzero(bf_events[-1].tensor()) > 0
    finally:
        driver.finish_call(wait=True)
        driver.release()
        driver.close()

    # 2) tokens="all" export over RequestState (multi-token prefill per request + 32 KiB event splitting).
    all_tok_events: list[pllm.io.TraceEvent] = []

    class AllTokensRequestExport(pllm.Workload):
        req = pllm.RequestState(
            {"layer": pllm.Field((), torch.int64)},
            init=pllm.Computed(
                lambda fields, g: fields["layer"].copy_(
                    torch.tensor(6, dtype=torch.int64)
                    if torch.rand((), generator=g).item() < 0.7
                    else torch.tensor(2, dtype=torch.int64)
                )
            ),
        )
        route = pllm.model.LayerRoute("req", over=(2, 6))

        @pllm.export(name="all_res", max_rows=10, max_cols=2048)
        def capture(self, r: torch.Tensor = route.residual(tokens="all")) -> torch.Tensor:
            # Pad to 2048 float32 cols = 8 KiB/row -> 4 rows per 32 KiB event.
            return torch.nn.functional.pad(r.float(), (0, 2048 - r.shape[-1]))

        def on_export(self, event: pllm.io.TraceEvent) -> None:
            all_tok_events.append(event)

    all_wl = AllTokensRequestExport()
    all_result = check(all_wl, CPU_CONFIG)
    assert "request isolation" in all_result.passed
    assert "mixed prefill/decode" in all_result.passed

    a_shape, a_facts = contract_facts(all_wl, CPU_CONFIG)
    _, a_decl = all_wl.specialize(a_facts, CPU_CONFIG.execution)
    a_dep = all_wl.bind_backend(a_facts, all_wl.plan_backend(a_facts, a_decl), a_decl)
    a_driver = StepDriver(a_dep.execution, a_dep.attachment)
    all_tok_events.clear()
    try:
        ids = (RequestIdentity("r0", 0, 0), RequestIdentity("r1", 1, 0), RequestIdentity("r2", 2, 0))
        a_driver.stage(ids, None)
        a_driver.schedule((), (), ("r0", "r1", "r2"))
        # 5 + 1 + 6 = 12 total prefill tokens -> max_rows=10 keeps 10 (drops 2), split into 4 + 4 + 2 rows.
        raw = contract_step_input(a_shape, a_facts, ((0, 5, 0), (1, 1, 0), (2, 6, 0)), 0, None, ids)
        ctx = a_driver.begin(raw, None)
        resolved = _resolve_routed_hidden(a_shape, a_facts, a_decl, a_dep.execution, None, raw, ctx)
        a_driver.after_forward(resolved, None)
        a_driver.sample(
            SamplerFeedback(
                resolved.sampler_rows, torch.zeros((2, a_shape.vocab_size)), torch.zeros(2, dtype=torch.long)
            ),
            None,
        )
        a_driver.commit()
        a_dep.execution.trace.drain()
        assert [len(ev.identities) for ev in all_tok_events] == [4, 4, 2]
        assert [ev.dropped_rows for ev in all_tok_events] == [2, 0, 0]
        assert all(len(ev.layers) == len(ev.identities) == len(ev.positions) for ev in all_tok_events)
        assert all(layer in (2, 6) for ev in all_tok_events for layer in ev.layers)
    finally:
        a_driver.finish_call(wait=True)
        a_driver.release()
        a_driver.close()


def test_layer_route_request_adapter_and_layer_logits_cpu_contracts() -> None:
    @dataclass(frozen=True)
    class PerRequestRoutedLoRA(pllm.Workload):
        rank: int = 4

        @property
        def req(self) -> pllm.RequestState:
            return pllm.RequestState(
                {
                    "a": pllm.Field(lambda ins, _: (self.rank, ins)),
                    "b": pllm.Field(lambda _, outs: (outs, self.rank)),
                    "layer": pllm.Field((), torch.int64),
                },
                init=pllm.Imported(),
            )

        @property
        def route(self) -> pllm.model.LayerRoute:
            return pllm.model.LayerRoute("req", over=(2, 8, -8))

        @property
        def lora(self) -> pllm.model.Adapter:
            return self.route.adapter(targets=("q_proj", "v_proj"), operator="low_rank")

        @pllm.add_logits()
        def lens_bias(
            self, logits: torch.Tensor = pllm.model.LayerRoute("req", over=(2, 8, -8)).layer_logits()
        ) -> torch.Tensor:
            return 0.01 * logits

        @pllm.export(name="hidden_probe", max_rows=4, max_cols=4)
        def probe(self, h: torch.Tensor = pllm.model.LayerRoute("req", over=(2, 8, -8)).hidden()) -> torch.Tensor:
            return h[:, :4].float()

    report = pllm.explain(PerRequestRoutedLoRA(), CPU_CONFIG)
    assert report.eager_reason == ""
    result = check(PerRequestRoutedLoRA(), CPU_CONFIG)
    assert "request isolation" in result.passed
    assert "mixed prefill/decode" in result.passed
    assert "preempted" in result.passed


def test_layer_route_gpu_self_routing_write_and_layer_bank_cpu_contracts() -> None:
    def init_bank(fields: dict[str, torch.Tensor], generator: torch.Generator) -> None:
        for index, tensor in enumerate(fields.values()):
            tensor.fill_(0.01 * (index + 1))

    class SelfRoutingSteer(pllm.Workload):
        ctrl = pllm.RequestState({"layer": pllm.Field((), torch.int64)})
        bank = pllm.InstanceState(
            {"vec": pllm.Field(lambda width, _: (width,))},
            layers=(2, 8, 14),
            init=pllm.Computed(init_bank),
        )
        route = pllm.model.LayerRoute("ctrl", over=(2, 8, 14))
        patch = route.patch("bank", operator="add", decode=True)

        @pllm.write(state="ctrl", fields=("layer",))
        def choose_next(self, r: torch.Tensor = route.residual()) -> tuple[torch.Tensor]:
            candidates = torch.tensor([2, 8, 14, -1], device=r.device, dtype=torch.int64)
            bucket = (r.float().abs().sum(dim=-1) * 10).to(torch.int64) % 4
            return (candidates.index_select(0, bucket),)

        @pllm.export(name="routed", max_rows=4, max_cols=2)
        def observe(self, r: torch.Tensor = route.residual()) -> torch.Tensor:
            return r[:, :2].float()

    report = pllm.explain(SelfRoutingSteer(), CPU_CONFIG)
    assert report.eager_reason == ""
    result = check(SelfRoutingSteer(), CPU_CONFIG)
    assert "request isolation" in result.passed
    assert "row permutation" in result.passed


@pytest.mark.gpu
def test_layer_route_gpu_contracts_and_live_switching() -> None:
    from vllm import SamplingParams

    from pllm.backends.vllm.activation import activation_delta
    from pllm.kernels.layer_route import gated_dense_add, gated_low_rank_add, masked_patch_affine_into

    # --- 0. Exact CUDA kernel rounding & non-unit stride checks ---
    h_fp16 = torch.tensor([[2048.0, 0.0]], device="cuda", dtype=torch.float16)
    w_fp16 = torch.ones((1, 1, 2), device="cuda", dtype=torch.float16)
    b_fp16 = torch.ones((1, 1), device="cuda", dtype=torch.float16)
    r_fp16 = torch.ones((1, 2, 1), device="cuda", dtype=torch.float16)
    expected_fp16 = h_fp16 + activation_delta(h_fp16, (w_fp16, b_fp16, r_fp16))
    actual_fp16 = h_fp16.clone()
    coeff_fp16 = torch.empty((1, 1), device="cuda", dtype=torch.float32)
    table_fp16 = torch.tensor([[1], [0], [0], [0]], device="cuda", dtype=torch.long)
    masked_patch_affine_into(actual_fp16, table_fp16, w_fp16, b_fp16, r_fp16, coeff_fp16)
    torch.testing.assert_close(actual_fp16, expected_fp16, rtol=0, atol=0)

    # TP down_proj FP32 partial-sum scaling + non-unit column stride guard check:
    h_bf16 = torch.tensor([[27.0, 0.0]], device="cuda", dtype=torch.bfloat16)
    w_bf16 = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], device="cuda", dtype=torch.bfloat16)
    slots_tp = torch.tensor([[0]], device="cuda", dtype=torch.long)
    owners_tp = torch.tensor([0], device="cuda", dtype=torch.long)
    out_strided = torch.zeros((1, 4), device="cuda", dtype=torch.float32)
    out_strided[0, 1::2] = -99.0
    gated_dense_add(h_bf16, slots_tp, owners_tp, w_bf16, out_strided[:, ::2], 0, 2, 1, 0.123)
    ref_scaled = ((h_bf16 @ w_bf16[0].T) * 0.123).to(torch.float32)
    torch.testing.assert_close(out_strided[:, ::2], ref_scaled, rtol=0, atol=0)
    torch.testing.assert_close(out_strided[:, 1::2], torch.full((1, 2), -99.0, device="cuda"))

    a_bf16 = torch.tensor([[[1.0, 0.0]]], device="cuda", dtype=torch.bfloat16)
    b_bf16 = torch.tensor([[[1.0], [0.0]]], device="cuda", dtype=torch.bfloat16)
    mid_tp = torch.empty((1, 1), device="cuda", dtype=torch.float32)
    out_lora_tp = torch.zeros((1, 4), device="cuda", dtype=torch.float32)
    out_lora_tp[0, 1::2] = -77.0
    gated_low_rank_add(
        h_bf16,
        torch.tensor([[0, 0]], device="cuda"),
        owners_tp,
        a_bf16,
        b_bf16,
        mid_tp,
        out_lora_tp[:, ::2],
        0,
        2,
        1,
        0.123,
    )
    torch.testing.assert_close(out_lora_tp[:, ::2], ref_scaled, rtol=0, atol=0)
    torch.testing.assert_close(out_lora_tp[:, 1::2], torch.full((1, 2), -77.0, device="cuda"))

    @dataclass(frozen=True)
    class RoutedGpuWorkload(pllm.Workload):
        scale: float = 1.0

        @property
        def req(self) -> pllm.RequestState:
            return pllm.RequestState(
                {
                    "layer": pllm.Field((), torch.int64),
                    "vec": lambda m: (m.hidden_size,),
                },
                init=pllm.Imported(),
            )

        @property
        def route(self) -> pllm.model.LayerRoute:
            return pllm.model.LayerRoute("req", over=(2, 10, 18))

        @property
        def steer(self) -> pllm.model.ActivationPatch:
            return self.route.patch(operator="add", first_tokens=1, last_tokens=1, decode=True)

        @pllm.export(name="probe", max_rows=4, max_cols=4)
        def probe(self, r: torch.Tensor = pllm.model.LayerRoute("req", over=(2, 10, 18)).residual()) -> torch.Tensor:
            return self.scale * r[:, :4].float()

    gpu_contract_cfg = pllm.ModelConfig(
        "Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=64,
        max_num_seqs=3,
        max_num_batched_tokens=64,
        gpu_memory_utilization=0.25,
        kv_cache_memory_bytes=32 << 20,
        execution=pllm.ExecutionOptions(cuda_graph_pool_budget_bytes=256 << 20),
    )
    contract = check_on_gpu(RoutedGpuWorkload(scale=1.0), gpu_contract_cfg)
    assert "explain/probe" in contract.passed
    assert "host syncs: eight steady decode steps" in contract.passed
    assert "solo vs batched: reversed, chunked and mixed batches match each key group alone" in contract.passed

    live_cfg = pllm.ModelConfig(
        "Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=64,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        gpu_memory_utilization=0.25,
        kv_cache_memory_bytes=32 << 20,
        cuda_graphs=True,
    )
    greedy = SamplingParams(max_tokens=5, temperature=0.0)
    prompts = ["Continue the sequence: 1, 2, 3,", "The capital of France is", "Water boils at"]
    gen = torch.Generator().manual_seed(42)

    # --- A. Static-reference GPU equivalence: LOW_RANK_AFFINE (rank=260 fallback + rank=4 fast path) + shared norm ---
    w_aff = torch.randn((4, 896), generator=gen, dtype=torch.bfloat16) * 2.5
    b_aff = torch.randn((4,), generator=gen, dtype=torch.bfloat16) * 2.5
    q_mat, _ = torch.linalg.qr(torch.randn((896, 4), generator=gen, dtype=torch.float32))
    r_aff = q_mat.to(torch.bfloat16)

    static_aff_res: list[torch.Tensor] = []

    class StaticAffineL10(pllm.Workload):
        inst = pllm.InstanceState(
            {"w": (4, 896), "b": (4,), "r": (896, 4)},
            init=pllm.Imported(),
        )
        patch = pllm.model.ActivationPatch("inst", layers=(10,), first_tokens=1, last_tokens=1, decode=True)

        @pllm.export(name="r10", max_rows=4, max_cols=8)
        def obs(self, r: torch.Tensor = pllm.residual(10)) -> torch.Tensor:
            return r[:, :8].float()

        def on_export(self, event: pllm.io.TraceEvent) -> None:
            static_aff_res.append(event.tensor())

    routed_aff_res: list[torch.Tensor] = []
    routed_aff_shared10: list[torch.Tensor] = []

    class RoutedAffineWithSharedNorm(pllm.Workload):
        inst = pllm.InstanceState(
            {"layer": pllm.Field((), torch.int64), "w": (4, 896), "b": (4,), "r": (896, 4)},
            init=pllm.Imported(),
        )
        route = pllm.model.LayerRoute("inst", over=(2, 10, 18))
        patch = route.patch(first_tokens=1, last_tokens=1, decode=True)

        @pllm.export(name="routed_r", max_rows=4, max_cols=8)
        def obs_routed(self, r: torch.Tensor = route.residual()) -> torch.Tensor:
            return r[:, :8].float()

        @pllm.export(name="static_r10", max_rows=4, max_cols=8)
        def obs_static(self, r: torch.Tensor = pllm.residual(10)) -> torch.Tensor:
            return r[:, :8].float()

        def on_export(self, event: pllm.io.TraceEvent) -> None:
            if event.tensor_name == "routed_r":
                routed_aff_res.append(event.tensor())
            else:
                routed_aff_shared10.append(event.tensor())

    with pllm.make_llm(live_cfg, workload=StaticAffineL10()) as llm_ref:
        llm_ref.import_state(StateImport((w_aff, b_aff, r_aff)))
        ref_aff_tokens = llm_ref.generate([prompts[1]], greedy)[0].outputs[0].token_ids
        llm_ref.trace_snapshot()

    with pllm.make_llm(live_cfg, workload=RoutedAffineWithSharedNorm()) as llm_routed:
        assert llm_routed.diagnostics().eager_reason == ""
        llm_routed.import_state(StateImport((torch.tensor(10, dtype=torch.int64), w_aff, b_aff, r_aff)))
        routed_aff_tokens = llm_routed.generate([prompts[1]], greedy)[0].outputs[0].token_ids
        llm_routed.trace_snapshot()
        routed_aff_active = list(routed_aff_res)
        routed_aff_shared = list(routed_aff_shared10)
        llm_routed.import_state(StateImport((torch.tensor(-1, dtype=torch.int64), w_aff, b_aff, r_aff)))
        disabled_aff_tokens = llm_routed.generate([prompts[1]], greedy)[0].outputs[0].token_ids

    assert routed_aff_tokens == ref_aff_tokens
    assert disabled_aff_tokens != ref_aff_tokens
    assert len(static_aff_res) >= 4
    assert len(routed_aff_active) == len(static_aff_res) == len(routed_aff_shared)
    for r_val, s_val, ref_val in zip(routed_aff_active, routed_aff_shared, static_aff_res, strict=True):
        torch.testing.assert_close(r_val, ref_val, rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(r_val, s_val, rtol=0, atol=0)

    # Also verify rank > 256 affine fallback captures CUDA graphs cleanly with 0 host syncs:
    class LargeRankAffine(pllm.Workload):
        inst = pllm.InstanceState(
            {"layer": pllm.Field((), torch.int64), "w": (260, 896), "b": (260,), "r": (896, 260)},
            init=pllm.Imported(),
        )
        route = pllm.model.LayerRoute("inst", over=(2, 10))
        patch = route.patch(first_tokens=0, last_tokens=1, decode=True)

    with pllm.make_llm(live_cfg, workload=LargeRankAffine()) as llm_hi:
        assert llm_hi.diagnostics().eager_reason == ""
        llm_hi.import_state(
            StateImport(
                (
                    torch.tensor(-1, dtype=torch.int64),
                    torch.zeros((260, 896), dtype=torch.bfloat16),
                    torch.zeros((260,), dtype=torch.bfloat16),
                    torch.zeros((896, 260), dtype=torch.bfloat16),
                )
            )
        )
        assert len(llm_hi.generate([prompts[0]], greedy)[0].outputs[0].token_ids) == 5

    # --- B. Static-reference GPU equivalence: Heterogeneous batch LOW_RANK Adapter (q_proj + down_proj) ---
    lora_q_a = torch.randn((3, 4, 896), generator=gen, dtype=torch.bfloat16) * 0.4
    lora_q_b = torch.randn((3, 896, 4), generator=gen, dtype=torch.bfloat16) * 0.4
    lora_d_a = torch.randn((3, 4, 4864), generator=gen, dtype=torch.bfloat16) * 0.4
    lora_d_b = torch.randn((3, 896, 4), generator=gen, dtype=torch.bfloat16) * 0.4

    def run_static_lora(layer_idx: int, req_idx: int) -> list[int]:
        class StaticLoRA(pllm.Workload):
            inst = pllm.InstanceState(
                {
                    "a": pllm.Field(lambda ins, _: (4, ins)),
                    "b": pllm.Field(lambda _, outs: (outs, 4)),
                },
                init=pllm.Imported(),
            )
            lora = pllm.model.Adapter(
                "inst",
                targets=(pllm.model.Projection("q_proj", layer_idx), pllm.model.Projection("down_proj", layer_idx)),
                operator="low_rank",
                scale=0.75,
            )

        with pllm.make_llm(live_cfg, workload=StaticLoRA()) as ref_llm:
            ref_llm.import_state(
                StateImport((lora_q_a[req_idx], lora_q_b[req_idx], lora_d_a[req_idx], lora_d_b[req_idx]))
            )
            return list(ref_llm.generate(prompts, greedy)[req_idx].outputs[0].token_ids)

    ref_lora_l2 = run_static_lora(2, 0)
    ref_lora_l18 = run_static_lora(18, 1)

    per_req_layers: dict[int, list[int]] = defaultdict(list)
    per_req_all_toks: dict[int, int] = defaultdict(int)

    class HeterogeneousBatchLoRA(pllm.Workload):
        req = pllm.RequestState(
            {
                "layer": pllm.Field((), torch.int64),
                "a": pllm.Field(lambda ins, _: (4, ins)),
                "b": pllm.Field(lambda _, outs: (outs, 4)),
            },
            init=pllm.Imported(),
        )
        route = pllm.model.LayerRoute("req", over=(2, 10, 18))
        lora = route.adapter(targets=("q_proj", "down_proj"), operator="low_rank", scale=0.75)

        @pllm.export(name="het_all", max_rows=32, max_cols=4)
        def capture_all(self, r: torch.Tensor = route.residual(tokens="all")) -> torch.Tensor:
            return r[:, :4].float()

        def on_export(self, event: pllm.io.TraceEvent) -> None:
            for identity, layer in zip(event.identities, event.layers, strict=True):
                per_req_layers[identity.prompt_index].append(layer)
                per_req_all_toks[identity.prompt_index] += 1

    with pllm.make_llm(live_cfg, workload=HeterogeneousBatchLoRA()) as llm_het:
        assert llm_het.diagnostics().eager_reason == ""
        het_layers = torch.tensor([2, 18, -1], dtype=torch.int64)
        het_outs = llm_het.generate(
            prompts,
            greedy,
            batch=StateImport((het_layers, lora_q_a, lora_q_b, lora_d_a, lora_d_b)),
        )
        llm_het.trace_snapshot()
        active_layers = {k: list(v) for k, v in per_req_layers.items()}
        active_toks = dict(per_req_all_toks)
        zero_outs = llm_het.generate(
            prompts,
            greedy,
            batch=StateImport(
                (
                    torch.tensor([2, 18, 2], dtype=torch.int64),
                    torch.zeros_like(lora_q_a),
                    torch.zeros_like(lora_q_b),
                    torch.zeros_like(lora_d_a),
                    torch.zeros_like(lora_d_b),
                )
            ),
        )

    assert list(het_outs[0].outputs[0].token_ids) == ref_lora_l2
    assert list(het_outs[1].outputs[0].token_ids) == ref_lora_l18
    assert list(het_outs[2].outputs[0].token_ids) == list(zero_outs[2].outputs[0].token_ids)
    assert set(active_layers[0]) == {2}
    assert set(active_layers[1]) == {18}
    assert set(active_layers[2]) == {-1}
    assert all(active_toks[idx] >= 6 for idx in (0, 1, 2))

    # --- C. Live InstanceState switching (including mid-generation online switch during active decode + prefill) ---
    captured_events: list[pllm.io.TraceEvent] = []

    class LiveSwitch(pllm.Workload):
        ctrl = pllm.InstanceState(
            {
                "layer": pllm.Field((), torch.int64),
                "vec": lambda m: (m.hidden_size,),
            },
            init=pllm.Imported(),
        )
        route = pllm.model.LayerRoute("ctrl", over=(4, 16, 22))
        steer = route.patch(operator="add", first_tokens=0, last_tokens=1, decode=True)

        @pllm.export(name="res", max_rows=16, max_cols=8)
        def read_res(self, r: torch.Tensor = route.residual(tokens="all")) -> torch.Tensor:
            return r[:, :8].float()

        def on_export(self, event: pllm.io.TraceEvent) -> None:
            captured_events.append(event)

    zero_vec = torch.zeros(896, dtype=torch.bfloat16)
    strong_vec = torch.full((896,), 2.5, dtype=torch.bfloat16)

    with pllm.make_llm(live_cfg, workload=LiveSwitch()) as llm_live:
        assert llm_live.diagnostics().eager_reason == ""
        captured_events.clear()
        llm_live.import_state(StateImport((torch.tensor(-1, dtype=torch.int64), strong_vec)))
        out_disabled = llm_live.generate([prompts[1]], greedy)[0].outputs[0].token_ids
        llm_live.trace_snapshot()
        events_disabled = list(captured_events)

        captured_events.clear()
        llm_live.import_state(StateImport((torch.tensor(4, dtype=torch.int64), zero_vec)))
        out_zero_l4 = llm_live.generate([prompts[1]], greedy)[0].outputs[0].token_ids

        captured_events.clear()
        llm_live.import_state(StateImport((torch.tensor(16, dtype=torch.int64), strong_vec)))
        out_steered_l16 = llm_live.generate([prompts[1]], greedy)[0].outputs[0].token_ids

    assert out_disabled == out_zero_l4
    assert out_steered_l16 != out_disabled
    assert events_disabled
    assert all(layer == -1 for ev in events_disabled for layer in ev.layers)
    assert all(torch.count_nonzero(ev.tensor()) == 0 for ev in events_disabled)

    # Mid-generation online live switch while online-a is decoding and online-b enters chunked prefill.
    async def run_mid_flight_switch() -> tuple[list[int], list[int], set[int], set[int]]:
        online_cfg = pllm.ModelConfig(
            "Qwen/Qwen2.5-0.5B-Instruct",
            max_model_len=64,
            max_num_seqs=4,
            max_num_batched_tokens=8,
            enable_chunked_prefill=True,
            gpu_memory_utilization=0.25,
            kv_cache_memory_bytes=32 << 20,
            cuda_graphs=True,
        )
        b_prompt = " ".join(f"word{i}" for i in range(20))
        async with pllm.make_async_llm(online_cfg, workload=LiveSwitch()) as engine:
            await engine.import_state(StateImport((torch.tensor(4, dtype=torch.int64), zero_vec)))
            switched_to_22 = asyncio.Event()

            async def consume_a() -> None:
                switched = False
                async for out in engine.stream(
                    prompts[0], SamplingParams(max_tokens=28, temperature=0.0), request_id="online-a"
                ):
                    n = len(out.outputs[0].token_ids)
                    if n >= 2 and not switched:
                        switched = True
                        await engine.import_state(StateImport((torch.tensor(22, dtype=torch.int64), strong_vec)))
                        switched_to_22.set()

            async def consume_b() -> None:
                await switched_to_22.wait()
                async for _ in engine.stream(
                    b_prompt,
                    SamplingParams(max_tokens=4, temperature=0.0),
                    request_id="online-b",
                ):
                    pass

            await asyncio.gather(consume_a(), consume_b())
            snap = await engine.trace_snapshot()

        rows_a: list[tuple[int, int, int]] = []
        rows_b: list[tuple[int, int, int]] = []
        for ev in snap.events:
            if not ev.layers:
                continue
            for ident, pos, layer in zip(ev.identities, ev.positions, ev.layers, strict=True):
                if ident.request_id.startswith("online-a"):
                    rows_a.append((ev.step_id, pos, layer))
                elif ident.request_id.startswith("online-b"):
                    rows_b.append((ev.step_id, pos, layer))
        assert rows_b
        max_pos_b = max(pos for _, pos, _ in rows_b)
        layers_a = [layer for _, _, layer in rows_a]
        layers_b = [layer for _, _, layer in rows_b]
        steps_a_decode_22 = {step for step, _, layer in rows_a if layer == 22}
        steps_b_prefill = {step for step, pos, _ in rows_b if pos <= max_pos_b - 4}
        return layers_a, layers_b, steps_a_decode_22, steps_b_prefill

    layers_a, layers_b, steps_a_decode_22, steps_b_prefill = asyncio.run(run_mid_flight_switch())
    assert 4 in layers_a and 22 in layers_a
    assert set(layers_b) == {22}
    assert len(steps_b_prefill) >= 2
    assert bool(steps_a_decode_22 & steps_b_prefill)
