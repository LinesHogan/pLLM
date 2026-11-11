"""CPU binding evidence only: received-page fixtures do not emulate CUDA IPC."""

import os
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from importlib.util import find_spec
from math import cos, pi, tanh
from multiprocessing import Pipe
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from examples.video_session import video_frames
from pllm.config import ExecutionOptions
from pllm.contracts.backend_attachment import (
    ActivationPatchBinding,
    ActivationPatchOperator,
    DecodeRefresh,
    DecodeWindow,
    MissingPolicy,
)
from pllm.contracts.codec import encode_boundary
from pllm.contracts.packed_rows import label_rows
from pllm.contracts.semantic_graph import AnchorId, StateEffectKind, StateResource
from pllm.contracts.state_feed import FeedBinding, FeedOrder
from pllm.contracts.state_ref import LineageId, StateRef, VersionId
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.backend_attachment import attachment_buffer_bytes, bind_backend_attachment
from pllm.execution.declaration import Dimensions, LayoutField, When
from pllm.execution.ptts.page_borrow import BorrowedPage, BorrowedStateRoot, PageFieldTransfer
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts, compile_page_bundle_plan
from pllm.execution.ptts.slot_pools import PageSlot
from pllm.execution.row_task_binding import (
    declare_parameters,
    declare_tensor,
)
from pllm.execution.runtime_facts import model_timeline
from pllm.execution.state_execution import MathBinding, PartialCommitError
from pllm.execution.tensor_declaration import TensorDeclaration
from pllm.execution.tensor_frame import TensorFrame
from pllm.execution.tensor_program import FunctionMath, TaskOutput
from pllm.testing import RequestIdentity, backend_facts, step_input
from pllm.workload import BackendDeployment
from pllm.workloads.video import VideoWorkload, frame_features


@contextmanager
def binding_case(*, alias=False, write_failure=False, runtime=False, missing=None):
    b = TensorDeclaration("video.binding", "a" * 64, Dimensions(2, 0))
    features, ports = declare_parameters(
        b,
        "features",
        tuple(LayoutField(name, (2,)) for name in ("qp", "residual", "mv", "segmentation")),
        optimizer=None,
    )
    encoded = declare_tensor(
        b,
        "encode",
        lambda **values: values["qp"] if alias else sum(values.values()),
        inputs=ports,
        when=When.BEFORE_FORWARD,
    )
    inject = b.state("inject", (LayoutField("delta", (2,)),))

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
        when=When.BEFORE_FORWARD,
        function=write,
    )
    publication = b.publish(transition, inject, when=When.BEFORE_FORWARD)
    timeline = model_timeline(torch.device("cpu"), ("transition/full_root@1",))
    if runtime:
        b.activation_patch(
            tuple(
                ActivationPatchBinding(
                    inject.declaration.resource,
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
            )
        )
        if missing is not None:
            b.conditional_input(("features",), publication)
        facts = backend_facts(2, pages=3)
        buffers = attachment_buffer_bytes(facts.probe, b.graph, b.layouts, b.effects)
        timeline = replace(timeline, reserved_bytes=(buffers,) * len(timeline.anchors))
    plan = compile_page_bundle_plan(
        (b.graph, b.layouts), timeline, (PageBundleProviderFacts("video.binding@1"),), 1, 65536
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
    arena = PageBundleArena(plan, owned_resources=("inject",))
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
            with pytest.raises(PartialCommitError if failure == "close" else (ValueError, ArithmeticError)):
                execution.step(step_input(()))
            if failure != "close":
                assert execution.latest(()) == before and not any(peer.poll() for peer in peers)
            with pytest.raises(RuntimeError):
                execution.admit(())


def _session(b, execution):
    from pllm.backends.vllm.session import VllmSession

    plan = execution.arena.plan
    facts = backend_facts(2, pages=3)
    deployment = BackendDeployment(execution, bind_backend_attachment(plan, facts.probe, b.effects))
    model = SimpleNamespace(model=SimpleNamespace(layers=()), lm_head=SimpleNamespace(weight=torch.zeros(2, 2)))
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
                sampling_params=SamplingParams(n=1, extra_args={"pllm.prefix_cache": session.cache_envelope}),
            )
            for identity in identities
        ),
    )


def _step_rows(step, identities):
    """Prompts of 3 (a) and 4 (b) tokens: a prefills 0-2 and b 0, then a decodes 3 while b prefills 1-3."""
    return label_rows(identities, [3, 1] if step == 0 else [1, 3], [3, 4], [0, 0] if step == 0 else [3, 1], step)


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

        session.state_feed = Mock(spec=FeedReceiver, side_effect=receive)
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
            hidden = torch.arange(8).reshape(4, 2).float()
            for layer, values in enumerate(expected_deltas):
                actual = session.activation_patch.patch(StateResource("inject"), layer, None)(hidden.clone())
                torch.testing.assert_close(actual, hidden + torch.tensor(values)[:, None], rtol=0, atol=0)
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
        assert execution.diagnostics().active_states == execution.diagnostics().inference_pins == 0


def _hold_steps_reuse_the_settled_injection():
    from pllm.execution.state_feed import FeedReceiver, ReadonlyStepAbsent, ReadonlyStepEnd, ReadonlyStepInput

    identity = RequestIdentity("hold", 0, 0)

    def rows(step):
        return label_rows((identity,), [1], [1], [step], step)

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
        session.state_feed = Mock(spec=FeedReceiver, side_effect=lambda batch, stream: outcomes[batch.step_id])
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
            for layer in range(3):
                delta = 20 if step >= 2 and layer == 0 else 10
                actual = session.activation_patch.patch(StateResource("inject"), layer, None)(torch.full((1, 2), 2.0))
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


def frames(count=4):
    path = Path(os.environ.get("PLLM_VIDEO_FIXTURE", "/home/user/TLM/data/mllm_demo_data/1.mp4"))
    if not path.is_file():
        pytest.skip("set PLLM_VIDEO_FIXTURE to a real video")
    result = list(video_frames(path, 2, count))
    assert len(result) == count
    return result


def test_video_real_frame_math_and_signed_composition():
    first, second = frames(2)
    values = frame_features(second, first, 2)
    stationary = frame_features(first, first, 2)
    assert torch.count_nonzero(stationary[1]) == torch.count_nonzero(stationary[2]) == 0
    facts = backend_facts(4, pages=3, layer_count=1)
    workload = VideoWorkload(grid_size=2)
    deployment = workload.compile_backend(facts, ExecutionOptions())
    execution, plan = deployment.execution, deployment.attachment.plan
    (layout,) = (item for item in plan.tensor_layout_decision.bindings if item.resource == "features")
    ref = StateRef(layout.layout_id, LineageId(1), VersionId(1))
    with ExitStack() as lifetime:
        pages = []
        for index, value in enumerate(values):
            peer, receiver = Pipe()
            lifetime.callback(peer.close)
            lifetime.callback(receiver.close)
            # Received-page fixture only; actual receive_field is CUDA-only and tested below.
            descriptor = PageFieldTransfer(
                execution.arena.plan_digest, 1, ref, "features", LayoutFieldId(index), PageSlot(0, 0), 0
            )
            pages.append(BorrowedPage(descriptor, value, receiver))
        execution.admit(())
        execution.stage_readonly((BorrowedStateRoot(execution.arena, "features", ref, tuple(pages)),))
        result = execution.step(step_input(()))
        signal = torch.cat(
            (
                values[0].flatten() / 63,
                values[1].flatten().sqrt(),
                values[2].flatten() / 2**0.5,
                values[3].flatten() / 3,
            )
        ).tolist()
        expected = torch.tensor(
            [
                workload.gain
                * tanh(
                    sum(cos(pi * (k + 1) * (j + 0.5) / len(signal)) * x for j, x in enumerate(signal))
                    / len(signal) ** 0.5
                )
                for k in range(4)
            ]
        )
        torch.testing.assert_close(
            execution.snapshot(result.publication.after_refs[0])[0], expected, atol=1e-6, rtol=1e-6
        )
        execution.retire(())
