"""CPU binding evidence only: received-page fixtures do not emulate CUDA IPC."""

from contextlib import ExitStack, contextmanager
from dataclasses import replace
from importlib.util import find_spec
from multiprocessing import Pipe
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from pllm.authoring import BackendDeployment
from pllm.contracts.backend_attachment import (
    ActivationPatchBinding,
    ActivationPatchOperator,
    DecodeRefresh,
    DecodeWindow,
    MissingPolicy,
)
from pllm.contracts.codec import encode_boundary
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch, PackedSamplerRowLink
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import AnchorId, StateEffectKind
from pllm.contracts.state_feed import FeedBinding, FeedOrder
from pllm.contracts.state_ref import LineageId, StateRef, VersionId
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.backend_attachment import attachment_buffer_bytes, bind_backend_attachment
from pllm.execution.declaration import Dimensions, Field, When
from pllm.execution.ptts.page_borrow import BorrowedPage, BorrowedStateRoot, PageFieldTransfer
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts, compile_page_bundle_plan
from pllm.execution.ptts.slot_pools import PageSlot
from pllm.execution.row_task_binding import (
    declare_parameters,
    declare_tensor,
)
from pllm.execution.runtime_facts import model_timeline
from pllm.execution.state_execution import MathBinding, StateSettlementError
from pllm.execution.tensor_binding import TensorBinding
from pllm.execution.tensor_frame import TensorFrame
from pllm.execution.tensor_program import FunctionMath, TaskOutput
from pllm.testing import backend_facts, step_input


@contextmanager
def binding_case(*, alias=False, write_failure=False, runtime=False, missing=None):
    b = TensorBinding("video.binding", "a" * 64, Dimensions(2, 0))
    features, ports = declare_parameters(
        b, "features", tuple(Field(name, (2,)) for name in ("qp", "residual", "mv", "segmentation")), optimizer=None
    )
    encoded = declare_tensor(b, "encode", inputs=ports, when=When.BEFORE_MODEL)(
        lambda **values: values["qp"] if alias else sum(values.values())
    )
    inject = b.state("inject", (Field("delta", (2,)),))

    def write(data, batch):
        if write_failure:
            raise ArithmeticError("writer failed after private allocation")
        view = inject.view("delta", b.dimensions)
        view.read(batch.roots[0][view.field]).copy_(batch.results[encoded.symbol, "value"])
        return TaskOutput(changed=(0,), update_count=1)

    transition = b.compute(
        "write",
        inputs={"hidden": encoded["value"]},
        outputs=("updated",),
        changes=inject,
        effect=StateEffectKind.TRANSITION,
        when=When.BEFORE_MODEL,
        function=write,
    )
    publication = b.publish(transition, inject, when=When.BEFORE_MODEL)
    timeline = model_timeline(torch.device("cpu"), ("transition/full_root@1",))
    if runtime:
        b.overlay(
            inject,
            (transition, publication),
            activations=tuple(
                ActivationPatchBinding(
                    i,
                    (inject.view("delta", b.dimensions),),
                    1,
                    1,
                    ActivationPatchOperator.ADD,
                    DecodeWindow.NONE if i == 2 and missing is None else DecodeWindow.ALL,
                    DecodeRefresh.STEP if i == 0 else DecodeRefresh.INHERIT,
                    MissingPolicy.STOP if missing is None else missing,
                )
                for i in range(3)
            ),
        )
        if missing is not None:
            b.conditional_input(("features",))
        facts = backend_facts(2, (), timeline, PageBundleProviderFacts("video.binding@1", 3))
        buffers = attachment_buffer_bytes(facts.probe, b.graph, b.layouts, b.attachment)
        timeline = replace(timeline, reserved_bytes=(buffers,) * len(timeline.anchors))
    plan = compile_page_bundle_plan(
        (b.graph, b.layouts), timeline, (PageBundleProviderFacts("video.binding@1", 3),), 1, 65536, torch.device("cpu")
    )
    math = FunctionMath(
        {"inject": lambda roots, keys: None},
        lambda value: value.rows.request_rows,
        lambda value, batch: TensorFrame(value, value.rows.rows),
        lambda data, result: result.publication,
    )
    binding = MathBinding(
        math,
        b.bind_all(plan, operations=b.executables(None, ())),
        feeds=(FeedBinding("features", FeedOrder.ORDERED, ("qp", "residual", "mv", "segmentation")),),
    )
    arena = PageBundleArena(plan, _owned_resources=("inject",))
    execution = binding.bind(arena, AnchorId(4))
    if not runtime:
        execution.admit(())
    with received_features(arena) as (root, peers):
        yield b, features, ports, binding, execution, root, peers


@contextmanager
def received_features(arena, *, version=7, scale=1):
    (layout,) = (item for item in arena.plan.tensor_layout_decision.bindings if item.resource == "features")
    ref = StateRef(layout.layout_id, LineageId(1), VersionId(version))
    with ExitStack() as lifetime:
        pages, peers = [], []
        for i in range(4):
            peer, receiver = Pipe()
            lifetime.callback(peer.close)
            lifetime.callback(receiver.close)
            peers.append(peer)
            # Deliberately bypass receive_field: CPU fixtures test roots/ACK ordering, not grant validation.
            descriptor = PageFieldTransfer(
                arena.plan_digest, version, ref, "features", LayoutFieldId(i), PageSlot(0, 0), 0
            )
            pages.append(BorrowedPage(descriptor, torch.full((2,), float(scale * (i + 1))), receiver))
        root = BorrowedStateRoot(arena, "features", ref, tuple(pages))
        yield root, peers


def test_video_binding_keeps_remote_roots_out_of_local_ownership():
    with binding_case() as (_, _, _, _, execution, root, peers):
        execution.stage_readonly((root,))
        invocation = execution.start_step(step_input(()))
        for anchor in execution.arena.plan.timeline.anchors:
            execution.execute_anchor(invocation, AnchorId(anchor.ordinal))
            if anchor.ordinal == execution.sampler_anchor_id:
                execution.complete_sampler(invocation)
        assert not any(peer.poll() for peer in peers)  # Publication is not remote completion.
        receipt = execution.finish_step(invocation)
        # The injected delta is the sum of the four received feature pages: 1 + 2 + 3 + 4.
        torch.testing.assert_close(execution.snapshot(receipt.after_refs[0])[0], torch.full((2,), 10.0))
        assert all(
            peer.poll() and peer.recv_bytes() == encode_boundary(page.descriptor).encode()
            for peer, page in zip(peers, root.pages, strict=True)
        )
    if find_spec("vllm") is not None:
        _decode_steps_keep_exact_publications_and_ack_order()
        _hold_steps_reuse_the_settled_injection()


def test_video_binding_failure_retains_grants_and_stops_execution():
    for failure in ("alias", "write", "close"):
        with binding_case(alias=failure == "alias", write_failure=failure == "write") as (*_, execution, root, peers):
            before = execution.latest(())
            execution.stage_readonly((root,))
            if failure == "close":
                peers[0].close()  # The producer vanished: the read ACK cannot be delivered.
            with pytest.raises(StateSettlementError if failure == "close" else (ValueError, ArithmeticError)):
                execution.step(step_input(()))
            if failure != "close":
                assert execution.latest(()) == before and not any(peer.poll() for peer in peers)
            with pytest.raises(RuntimeError):
                execution.admit(())


class _Layer(torch.nn.Module):
    def forward(self, positions, hidden_states, residual=None):
        return hidden_states * 2, residual


def _session(b, execution):
    from pllm.backends.vllm.session import VllmSession

    plan = execution.arena.plan
    facts = backend_facts(2, (), plan.timeline, PageBundleProviderFacts("video.binding@1", 3))
    deployment = BackendDeployment(execution, bind_backend_attachment(plan, facts.probe, b.attachment))
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[_Layer() for _ in range(3)]), lm_head=SimpleNamespace(weight=torch.zeros(2, 2))
    )
    from pllm.backends.vllm.prefix_cache import CachePolicy

    return (
        VllmSession(
            deployment,
            SimpleNamespace(compute_logits=lambda hidden: hidden, model=model),
            None,
            2,
            cache_policy=CachePolicy.for_attachment(deployment.attachment, "isolate"),
        ),
        model,
        deployment,
    )


def _scheduler(session, *identities):
    from vllm import SamplingParams

    return SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        scheduled_new_reqs=tuple(
            SimpleNamespace(
                req_id=identity.request_id,
                sampling_params=SamplingParams(n=1, extra_args={"pllm.prefix_cache": session._cache_envelope}),
            )
            for identity in identities
        ),
    )


def _step_rows(step, identities):
    a, b = identities
    cells = (
        (
            (a, 0, HiddenPhase.PREFILL),
            (a, 1, HiddenPhase.PREFILL),
            (a, 2, HiddenPhase.PREFILL),
            (b, 0, HiddenPhase.PREFILL),
        )
        if step == 0
        else (
            (a, 3, HiddenPhase.DECODE),
            (b, 1, HiddenPhase.PREFILL),
            (b, 2, HiddenPhase.PREFILL),
            (b, 3, HiddenPhase.PREFILL),
        )
    )
    rows = PackedPhaseRowBatch(
        step,
        4,
        tuple(
            PackedPhaseRow(i, int(identity == b), offset, phase, identity)
            for i, (identity, offset, phase) in enumerate(cells)
        ),
    )
    return PackedSamplerRowBatch(rows, (PackedSamplerRowLink(0, 2 if step == 0 else 0), PackedSamplerRowLink(1, 3)))


def _decode_steps_keep_exact_publications_and_ack_order():
    from pllm.execution.state_feed import FeedReceiver, ReadonlyStepInput

    identities = (RequestIdentity("a", 0, 0), RequestIdentity("b", 1, 0))
    # Only the stream adapter is replaced: all state tasks, pins and settlement are real CPU code.
    with (
        binding_case(runtime=True) as (b, _, _, _, execution, first, first_peers),
        received_features(execution.arena, version=8, scale=2) as (second, second_peers),
        patch("torch.cuda.current_stream", return_value=None),
    ):
        session, model, _ = _session(b, execution)
        received = []

        def receive(rows, stream):
            assert execution.diagnostics().update_count == rows.step_id
            received.append(rows.step_id)
            return ReadonlyStepInput(rows.step_id, ((first, second)[rows.step_id],))

        session._feed = Mock(spec=FeedReceiver, side_effect=receive)
        session.stage_input(None, identities)
        for step, (root, peers) in enumerate(((first, first_peers), (second, second_peers))):
            session.before_execute(_scheduler(session, *(identities if step == 0 else ())))
            rows = _step_rows(step, identities)
            rng = torch.random.get_rng_state().clone()
            session.prepare_input(rows)
            assert execution.latest(())[0].version_id == step + 1
            assert not any(peer.poll() for peer in peers)
            expected_deltas = (
                ((10, 0, 10, 10),) * 3 if step == 0 else ((20, 10, 0, 10), (10, 10, 0, 10), (0, 10, 0, 10))
            )
            hidden, residual = torch.arange(8).reshape(4, 2).float(), torch.ones(4, 2)
            for layer, values in zip(model.model.layers, expected_deltas, strict=True):
                actual, carry = layer(torch.arange(4), hidden, residual)
                torch.testing.assert_close(actual, hidden * 2 + torch.tensor(values)[:, None], rtol=0, atol=0)
                assert carry is residual
            session.complete_forward()
            session.complete_sampler()
            assert not any(peer.poll() for peer in peers), "sampler completion alone is not settlement"
            session.commit_token()
            assert torch.equal(rng, torch.random.get_rng_state())
            assert all(
                peer.poll() and peer.recv_bytes() == encode_boundary(page.descriptor).encode()
                for peer, page in zip(peers, root.pages, strict=True)
            )
        assert received == [0, 1] and execution.diagnostics().update_count == 2
        session.close()
        assert execution.diagnostics().active_lineages == execution.diagnostics().inference_pins == 0


def _hold_steps_reuse_the_settled_injection():
    from pllm.execution.state_feed import FeedReceiver, ReadonlyStepAbsent, ReadonlyStepEnd, ReadonlyStepInput

    identity = RequestIdentity("hold", 0, 0)

    def rows(step):
        batch = PackedPhaseRowBatch(
            step,
            1,
            (PackedPhaseRow(0, 0, step, HiddenPhase.PREFILL if step == 0 else HiddenPhase.DECODE, identity),),
        )
        return PackedSamplerRowBatch(batch, (PackedSamplerRowLink(0, 0),))

    with (
        binding_case(runtime=True, missing=MissingPolicy.HOLD) as (b, _, _, binding, execution, first, peers),
        received_features(execution.arena, version=8, scale=2) as (second, second_peers),
        patch("torch.cuda.current_stream", return_value=None),
    ):
        session, model, _ = _session(b, execution)
        outcomes = (
            ReadonlyStepInput(0, (first,)),
            ReadonlyStepAbsent(1),
            ReadonlyStepInput(2, (second,)),
            ReadonlyStepEnd(3),
        )
        session._feed = Mock(spec=FeedReceiver, side_effect=lambda batch, stream: outcomes[batch.step_id])
        session.stage_input(None, (identity,))
        refs = []
        for step in range(4):
            session.before_execute(_scheduler(session, *((identity,) if step == 0 else ())))
            rng = torch.random.get_rng_state().clone()
            session.prepare_input(rows(step))
            hold = step in (1, 3)
            current = execution.latest(())[0]
            refs.append(current)
            assert current.version_id == (1 if step < 2 else 2)
            for index, layer in enumerate(model.model.layers):
                delta = 20 if step >= 2 and index == 0 else 10
                actual, _ = layer(torch.zeros(1), torch.ones(1, 2))
                torch.testing.assert_close(actual, torch.full((1, 2), float(2 + delta)))
            session.complete_forward()
            session.complete_sampler()
            session.commit_token()
            assert torch.equal(rng, torch.random.get_rng_state())
            assert execution.diagnostics().update_count == (step // 2 + 1)
            if not hold:
                root, incoming = (first, peers) if step == 0 else (second, second_peers)
                assert all(
                    peer.recv_bytes() == encode_boundary(page.descriptor).encode()
                    for peer, page in zip(incoming, root.pages, strict=True)
                )
            else:
                assert not any(peer.poll() for peer in (*peers, *second_peers))
        assert refs[0] == refs[1] and refs[2] == refs[3] and refs[0] != refs[2]
