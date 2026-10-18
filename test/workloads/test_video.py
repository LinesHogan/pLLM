"""CPU binding evidence only: received-page fixtures do not emulate CUDA IPC."""

from contextlib import ExitStack, contextmanager
from dataclasses import replace
from importlib.util import find_spec
from multiprocessing import Pipe
from types import SimpleNamespace
from unittest.mock import patch

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
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch, PackedSamplerRowLink
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import AnchorId, StateEffectKind
from pllm.contracts.state_ref import LineageId, StateRef, VersionId
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.backend_attachment import attachment_buffer_bytes, bind_backend_attachment
from pllm.execution.codec import encode_boundary
from pllm.execution.declaration import Dimensions, Field, When
from pllm.execution.ptts.page_borrow import BorrowedStateRoot
from pllm.execution.ptts.page_bundle import BorrowedPage, PageBundleArena, PageFieldTransfer, PageSlot
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts, compile_page_bundle_plan
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.execution.state_execution import MathBinding, StateSettlementError
from pllm.execution.tensor_binding import TensorBinding, TensorFrame
from pllm.execution.tensor_program import FunctionMath, TaskOutput, _read
from test.backend_facts import backend_facts


@contextmanager
def binding_case(*, alias=False, write_failure=False, runtime=False, missing=None):
    b = TensorBinding("video.binding", "a" * 64, Dimensions(2, 0))
    features, ports = b.parameters(
        "features", {name: (2,) for name in ("qp", "residual", "mv", "segmentation")}, optimizer=None
    )
    encoded = b.tensor("encode", inputs=ports, when=When.BEFORE_MODEL)(
        lambda values: values["qp"] if alias else sum(values.values())
    )
    inject = b.state("inject", (Field("delta", (2,)),))

    def write(data, batch):
        if write_failure:
            raise ArithmeticError("writer failed after private allocation")
        return TaskOutput(
            field_writes=((LayoutFieldId(0), batch.results[encoded.symbol, "value"].unsqueeze(0)),),
            changed=(0,),
            update_count=1,
        )

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
    timeline = RuntimeFacts(torch.device("cpu")).timeline(("transition/full_root@1",))
    if runtime:
        b.overlay(
            inject,
            transition,
            publication,
            activations=tuple(
                ActivationPatchBinding(
                    i,
                    (LayoutFieldId(0),),
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
        {"inject": lambda roots: None},
        lambda value: () if value is None else value.rows.request_rows,
        lambda value, batch: TensorFrame(value, () if value is None else value.rows.rows),
        lambda data, result: result.publication,
    )
    binding = MathBinding(math, b.bind_all(plan), borrowed_resources=("features",))
    arena = PageBundleArena(plan, _owned_resources=("inject",))
    execution = binding.bind(arena, AnchorId(4))
    if not runtime:
        execution.admit(())
    with received_features(plan) as (root, peers):
        yield b, features, ports, binding, execution, root, peers


@contextmanager
def received_features(plan, *, version=7, scale=1):
    (layout,) = (item for item in plan.tensor_layout_decision.bindings if item.resource == "features")
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
                plan.schedule_decision.signature, version, ref, "features", LayoutFieldId(i), PageSlot(0, 0), 0
            )
            pages.append(BorrowedPage(descriptor, torch.full((2,), float(scale * (i + 1))), receiver))
        root = BorrowedStateRoot(plan, "features", ref, tuple(pages))
        yield root, peers


def test_video_binding_keeps_remote_roots_out_of_local_ownership():
    with binding_case() as (b, features, ports, binding, execution, root, peers):
        arena, plan = execution.arena, execution.arena.plan
        producer = PageBundleArena(plan, _owned_resources=("features",))
        assert len(arena.allocation_ranges) == 1 and len(producer.allocation_ranges) == 4
        assert tuple(state.resource for state in execution.directory.states) == ("inject",)
        with pytest.raises(ValueError, match="signed resources"):
            arena.prepare_admission(("features",))
        with pytest.raises(ValueError, match="partition"):
            binding.bind(PageBundleArena(plan), AnchorId(4))
        with pytest.raises(ValueError, match="read-only"):
            replace(binding, borrowed_resources=("inject",)).bind(arena, AnchorId(4))
        execution.stage_readonly((root,))
        invocation = execution.start_step(None)
        batch = _read(invocation.context)
        frame = invocation.context.data()
        assert batch.after_refs[0].lineage_id == root.ref.lineage_id  # Same integers, different owners.
        assert ports["qp"].read(frame, batch).data_ptr() == root.pages[0].tensor.data_ptr()
        assert batch.parameter_root("inject")[0].data_ptr() != root.pages[0].tensor.data_ptr()
        row = PackedPhaseRow(0, 0, 0, HiddenPhase.PREFILL, RequestIdentity("video", 0, 0))
        metadata = b.state_metadata(features, version=True).read(replace(frame, rows=(row,)), batch)
        assert metadata.item() == 7
        for anchor in plan.timeline.anchors:
            execution.execute_anchor(invocation, AnchorId(anchor.ordinal))
            if anchor.ordinal == execution.sampler_anchor_id:
                execution.complete_sampler(invocation)
        assert not any(peer.poll() for peer in peers)  # Publication is not remote completion.
        _owned_readonly_requires_exact_pin(invocation.context)
        del batch, frame
        receipt = execution.finish_step(invocation)
        torch.testing.assert_close(execution.snapshot(receipt.after_refs[0])[0], torch.full((2,), 10.0))
        assert all(
            peer.poll() and peer.recv_bytes() == encode_boundary(page.descriptor).encode()
            for peer, page in zip(peers, root.pages, strict=True)
        )
        with pytest.raises(RuntimeError, match="closed"):
            root.tensors(invocation.context.pin)
        with pytest.raises(RuntimeError, match="staged roots"):
            execution.begin(())  # Never silently read local inject as the next features chunk.
    if find_spec("vllm") is not None:
        _decode_steps_keep_exact_publications_and_ack_order()
        _hold_steps_require_settled_injection_and_skip_the_producer()


def _owned_readonly_requires_exact_pin(owner):
    from pllm.contracts.region_imports import BorrowedTensorImport
    from pllm.contracts.region_program import (
        AccessMode,
        Block,
        BorrowedResource,
        Call,
        Function,
        Program,
        Region,
        ResourceAccess,
        ScalarType,
        TensorType,
        Value,
    )
    from pllm.execution.decoder_region import index_cell
    from pllm.execution.paged_access import CausalAccess, PagedAccessPlan, PagedResourceView
    from pllm.execution.region_resources import (
        PinnedResourceBinding,
        PinnedResourceCell,
        ReadonlyResourceBinding,
        RegionResources,
    )
    from pllm.execution.tensor_borrow import ReadonlyTensorCell

    (ref,) = owner.publication.after_refs
    pin = owner.borrow_published((ref,))
    resource = BorrowedResource("inject.read", (1,))
    tensor = TensorType((2,), "torch.float32", "cpu")
    getter = Function(
        "read.inject",
        (ScalarType.INDEX,),
        (tensor,),
        (resource.name,),
        (ResourceAccess(resource.name, AccessMode.READ, (index_cell(0),)),),
    )
    program = Program(
        Region(
            Block(
                (Value("row", ScalarType.INDEX),), (Call(("row",), (Value("value", tensor),), getter.name),), ("value",)
            )
        ),
        (getter,),
        (resource,),
    )
    imported = BorrowedTensorImport(getter, 0)
    proof = PinnedResourceCell((0,), pin, ref, (LayoutFieldId(0),))
    bound = PinnedResourceBinding.bind(resource, owner, (proof,))
    pages = RegionResources(program, owner, (), borrows=(bound,), tensor_borrows=(imported,))
    actual = pages.borrow_tensor(imported, (0,))
    expected = owner.arena.borrow_field(pin, ref, LayoutFieldId(0))
    assert (
        actual.data_ptr() == expected.data_ptr()
        and actual.untyped_storage().nbytes() == expected.untyped_storage().nbytes()
    )
    with pytest.raises(ValueError, match="overlaps original arena"):
        RegionResources(
            program, owner, (), borrows=(ReadonlyResourceBinding(bound.view, owner),), tensor_borrows=(imported,)
        )
    forged = replace(bound, view=replace(bound.view, cells=(ReadonlyTensorCell.bind((0,), (actual.clone(),)),)))
    with pytest.raises(ValueError, match="storage differs"):
        RegionResources(program, owner, (), borrows=(forged,), tensor_borrows=(imported,))
    for wrong in (
        replace(proof, pin=replace(pin)),
        replace(proof, pin=owner.pin),
        replace(proof, fields=(LayoutFieldId(99),)),
    ):
        with pytest.raises((ValueError, RuntimeError)):
            PinnedResourceBinding.bind(resource, owner, (wrong,))
    writable = replace(getter, accesses=(replace(getter.accesses[0], mode=AccessMode.WRITE),))
    with pytest.raises(ValueError, match="READ effects only"):
        RegionResources(replace(program, functions=(writable,)), owner, (), borrows=(bound,))
    # Exercise the new binding's writable-alias check directly, independently
    # of the ordinary paged-vs-arena disjointness check in RegionResources.
    write = CausalAccess(
        "kv",
        RequestIdentity("reader", 0, 0),
        0,
        0,
        1,
        str(actual.device),
        actual.data_ptr(),
        actual.numel() * actual.element_size(),
        actual.numel() * actual.element_size(),
    )
    access = PagedAccessPlan((), (write,), {}, ("kv",), {("kv", 0): (0, 1)})
    paged = PagedResourceView(BorrowedResource("kv", (1, 1, 1)), access, ((0, "kv"),))
    with pytest.raises(ValueError, match="aliases writable paged"):
        bound.validate(program, (paged,))
    owner.release_owned_pin(pin)
    with pytest.raises(ValueError, match="live exact pin"):
        pages.borrow_tensor(imported, (0,))


@pytest.mark.parametrize("failure", ("alias", "close", "cancel", "missing", "mixed", "staging", "write"))
def test_video_binding_failure_retains_grants_and_stops_execution(failure):
    with binding_case(alias=failure == "alias", write_failure=failure == "write") as (
        _,
        _,
        _,
        _,
        execution,
        root,
        peers,
    ):
        before = execution.directory.latest_refs(())
        if failure == "mixed":
            with pytest.raises(ValueError, match="exact grant"):
                BorrowedStateRoot(
                    execution.arena.plan, "features", replace(root.ref, version_id=VersionId(8)), root.pages
                )
            return
        if failure != "missing":
            if failure == "staging":
                with pytest.raises(ValueError, match="required signed"):
                    execution.stage_readonly(())
            else:
                execution.stage_readonly((root,))
        if failure == "close":
            root.pages[0]._connection.close()
        if failure == "cancel":
            invocation = execution.start_step(None)
            with pytest.raises(RuntimeError, match="cancelled"):
                execution.cancel_step(invocation)
        else:
            error = StateSettlementError if failure == "close" else ValueError if failure == "alias" else RuntimeError
            error = ArithmeticError if failure == "write" else error
            reason = {
                "close": "publication committed",
                "alias": "overlaps",
                "missing": "staged roots",
                "staging": "fail-stopped",
                "write": "writer failed",
            }[failure]
            with pytest.raises(error, match=reason):
                execution.step(None)
        if failure in ("alias", "cancel", "missing", "staging", "write"):
            assert execution.directory.latest_refs(()) == before and not any(peer.poll() for peer in peers)
            assert execution.arena.usage().private_versions == 0
        else:
            assert execution.directory.latest_refs(()) != before and execution._active
        with pytest.raises(RuntimeError, match="fail-stopped"):
            execution.stage_readonly((root,))
        with pytest.raises(RuntimeError, match="fail-stopped"):
            execution.admit(())
    if failure == "staging" and find_spec("vllm") is not None:
        _step_input_errors_retain_grants()


class _Layer(torch.nn.Module):
    def forward(self, positions, hidden_states, residual=None):
        return hidden_states * 2, residual


def _session(b, execution):
    from pllm.backends.vllm.runtime import VllmSession

    plan = execution.arena.plan
    facts = backend_facts(2, (), plan.timeline, PageBundleProviderFacts("video.binding@1", 3))
    deployment = BackendDeployment(execution, bind_backend_attachment(plan, facts.probe, b.attachment))
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[_Layer() for _ in range(3)]), lm_head=SimpleNamespace(weight=torch.zeros(2, 2))
    )
    return VllmSession(deployment, model, None, 2), model, deployment


def _scheduler(*identities, preempted=(), resumed=()):
    from vllm import SamplingParams

    return SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(preempted),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set(resumed)),
        scheduled_new_reqs=tuple(
            SimpleNamespace(req_id=identity.request_id, sampling_params=SamplingParams(n=1)) for identity in identities
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
        1,
        step,
        4,
        tuple(
            PackedPhaseRow(i, int(identity == b), offset, phase, identity)
            for i, (identity, offset, phase) in enumerate(cells)
        ),
    )
    return PackedSamplerRowBatch(rows, (PackedSamplerRowLink(0, 2 if step == 0 else 0), PackedSamplerRowLink(1, 3)))


def _decode_steps_keep_exact_publications_and_ack_order():
    from pllm.backends.vllm.runtime import ReadonlyStepInput, Runner

    identities = (RequestIdentity("a", 0, 0), RequestIdentity("b", 1, 0))
    # Only the stream adapter is replaced: all state tasks, pins and settlement are real CPU code.
    with (
        binding_case(runtime=True) as (b, _, _, _, execution, first, first_peers),
        received_features(execution.arena.plan, version=8, scale=2) as (second, second_peers),
        patch("torch.cuda.current_stream", return_value=None),
    ):
        session, model, deployment = _session(b, execution)
        assert first.pages[0].descriptor.owner_id != second.pages[0].descriptor.owner_id
        runner = SimpleNamespace(_workload_session=None, cache_config=SimpleNamespace(enable_prefix_caching=True))
        with pytest.raises(ValueError, match="signed activation.*prefix caching"):
            Runner.attach_workload(runner, deployment)
        received = []

        def receive(rows, stream):
            assert stream is None and not execution._active
            assert execution.diagnostics().update_count == rows.step_id
            received.append(rows.step_id)
            return ReadonlyStepInput(rows.plan_epoch, rows.step_id, ((first, second)[rows.step_id],))

        session.bind_readonly_input(receive)
        with pytest.raises(RuntimeError, match="idle borrowed"):
            session.bind_readonly_input(receive)
        session.stage_input(None, identities)
        admission = None
        for step, (root, peers) in enumerate(((first, first_peers), (second, second_peers))):
            session.before_execute(_scheduler(*(identities if step == 0 else ())))
            rows = _step_rows(step, identities)
            rng = torch.random.get_rng_state().clone()
            session.prepare_input(rows)
            invocation = session._step.invocation
            context = invocation.context
            (ref,) = context.publication.after_refs
            assert context.borrowed_roots == (root,) and ref.version_id == step + 1
            if step == 0:
                admission = execution.directory.inference_pin(identities[0])
                assert admission.refs == (ref,)
            else:
                assert execution.directory.inference_pin(identities[0]) is admission
                assert context.published_pin.refs == (ref,) and ref != admission.refs[0]
                # Prepare is a pin borrow, not a second update or a new selection from latest.
                pin = context.published_pin
                session._activation.prepare(rows.row_batch, context, execution.directory)
                assert context.published_pin is pin and len(context.page_pins) == 2
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
            assert all(not layer.prepared for layer in session._activation.layers)
            assert not any(peer.poll() for peer in peers)
            session.complete_sampler(False)
            assert not any(peer.poll() for peer in peers), "sampler completion alone is not settlement"
            session.commit_token()
            assert torch.equal(rng, torch.random.get_rng_state())
            assert all(
                peer.poll() and peer.recv_bytes() == encode_boundary(page.descriptor).encode()
                for peer, page in zip(peers, root.pages, strict=True)
            )
            assert all(release is not None for release in context.page_pins.values())
            with pytest.raises(RuntimeError, match="closed"):
                root.tensors(context.pin)
        assert received == [0, 1]
        diagnostics = execution.diagnostics()
        assert diagnostics.update_count == diagnostics.completed_sampler_pins == 2 and diagnostics.fallback_count == 0
        # Replay of already computed input must stop before receiving another root or publishing.
        with pytest.raises(RuntimeError, match="injection and vLLM preemption/history recomputation"):
            session.prepare_input(
                replace(_step_rows(1, identities), row_batch=replace(_step_rows(1, identities).row_batch, step_id=2))
            )
        with pytest.raises(RuntimeError, match="fail-stopped"):
            session.before_execute(_scheduler())
        assert received == [0, 1] and execution.diagnostics().update_count == 2
        session.close()
        assert execution.diagnostics().active_lineages == execution.diagnostics().inference_pins == 0


def _step_input_errors_retain_grants():
    from pllm.backends.vllm.runtime import ReadonlyStepInput

    identities = (RequestIdentity("a", 0, 0), RequestIdentity("b", 1, 0))
    for failure in ("epoch", "step", "stream", "empty", "receive", "preempted", "resumed"):
        with (
            binding_case(runtime=True) as (b, _, _, _, execution, root, peers),
            patch("torch.cuda.current_stream", return_value=None),
        ):
            session, _, _ = _session(b, execution)
            if failure == "stream":
                root.stream = object()  # Wrong receiving-stream identity; no CUDA transport is emulated.
            incoming = ReadonlyStepInput(
                2 if failure == "epoch" else 1, 1 if failure == "step" else 0, () if failure == "empty" else (root,)
            )
            calls = []

            def receive(rows, stream):
                calls.append(rows.step_id)
                if failure == "receive":
                    raise EOFError("partial receive remains owned by the receiver")
                return incoming

            session.bind_readonly_input(receive)
            session.stage_input(None, identities)
            if failure in ("preempted", "resumed"):
                marker = {failure: ("a",)}
                with pytest.raises(RuntimeError, match="injection and vLLM preemption"):
                    session.before_execute(_scheduler(*identities, **marker))
                assert not calls
            else:
                session.before_execute(_scheduler(*identities))
                with pytest.raises(EOFError if failure == "receive" else ValueError):
                    session.prepare_input(_step_rows(0, identities))
                if failure != "receive":
                    assert session._received_readonly is incoming
            assert not execution._active and execution.diagnostics().update_count == 0
            assert not any(peer.poll() for peer in peers)
            assert all(page.tensor is not None for page in root.pages)
            with pytest.raises(RuntimeError, match="fail-stopped"):
                session.prepare_input(_step_rows(0, identities))
            with pytest.raises(RuntimeError, match="fail-stopped"):
                session.stage_input(None, identities)
            session.close()
            assert not any(peer.poll() for peer in peers), "closing the session cannot forge a read ACK"


def _hold_steps_require_settled_injection_and_skip_the_producer():
    from pllm.backends.vllm.runtime import ReadonlyStepAbsent, ReadonlyStepEnd, ReadonlyStepInput

    identity = RequestIdentity("hold", 0, 0)

    def rows(step):
        batch = PackedPhaseRowBatch(
            1,
            step,
            1,
            (PackedPhaseRow(0, 0, step, HiddenPhase.PREFILL if step == 0 else HiddenPhase.DECODE, identity),),
        )
        return PackedSamplerRowBatch(batch, (PackedSamplerRowLink(0, 0),))

    with (
        binding_case(runtime=True, missing=MissingPolicy.HOLD) as (b, _, _, binding, execution, first, peers),
        received_features(execution.arena.plan, version=8, scale=2) as (second, second_peers),
        patch("torch.cuda.current_stream", return_value=None),
    ):
        session, model, _ = _session(b, execution)
        with pytest.raises(ValueError, match="signed task closure"):
            replace(binding, tasks=tuple(task.original for task in binding.tasks)).bind(
                execution.arena, execution.sampler_anchor_id
            )
        with pytest.raises(ValueError, match="signed conditional producer guard"):
            execution.rebind_implementation(binding.tasks[0].original)
        outcomes = (
            ReadonlyStepInput(1, 0, (first,)),
            ReadonlyStepAbsent(1, 1),
            ReadonlyStepInput(1, 2, (second,)),
            ReadonlyStepEnd(1, 3),
        )
        session.bind_readonly_input(lambda batch, stream: outcomes[batch.step_id])
        session.stage_input(None, (identity,))
        refs = []
        for step in range(4):
            session.before_execute(_scheduler(*((identity,) if step == 0 else ())))
            rng = torch.random.get_rng_state().clone()
            session.prepare_input(rows(step))
            context = session._step.invocation.context
            hold = step in (1, 3)
            assert (context.held_input is not None) == hold
            assert bool(context.borrowed_roots) != hold
            assert (context.publication is None) == hold
            if hold:
                assert not context.results and not context.page_writes and not context.page_publications
                assert context.held_input.successful[0][2] == step - 1
            current = execution.directory.latest_refs(())[0]
            refs.append(current)
            assert current.version_id == (1 if step < 2 else 2)
            for index, layer in enumerate(model.model.layers):
                delta = 20 if step >= 2 and index == 0 else 10
                actual, _ = layer(torch.zeros(1), torch.ones(1, 2))
                torch.testing.assert_close(actual, torch.full((1, 2), float(2 + delta)))
            before = execution.diagnostics()
            session.complete_forward()
            session.complete_sampler(False)
            assert execution.diagnostics().hold_count == before.hold_count
            session.commit_token()
            assert torch.equal(rng, torch.random.get_rng_state())
            assert execution.diagnostics().hold_count == (step + 1) // 2
            assert execution.diagnostics().update_count == (step // 2 + 1)
            assert all(release is not None for release in context.page_pins.values())
            if not hold:
                root, incoming = (first, peers) if step == 0 else (second, second_peers)
                assert all(
                    peer.recv_bytes() == encode_boundary(page.descriptor).encode()
                    for peer, page in zip(incoming, root.pages, strict=True)
                )
            else:
                assert not any(peer.poll() for peer in (*peers, *second_peers))
        assert refs[0] == refs[1] and refs[2] == refs[3] and refs[0] != refs[2]
        assert execution.diagnostics().fallback_count == 0
        assert execution.diagnostics().last_hold.reason == "end"
        # Cancelling a held step cannot register success or revive a poisoned session.
        execution.stage_hold((identity,), "absent", step_id=4)
        invocation = execution.begin(())
        with pytest.raises(RuntimeError, match="cancelled"):
            execution.cancel_step(invocation)
        assert execution.diagnostics().hold_count == 2
        with pytest.raises(RuntimeError, match="fail-stopped"):
            execution.stage_hold((identity,), "absent", step_id=4)
    for policy, seeded in ((MissingPolicy.HOLD, False), (MissingPolicy.STOP, False), (MissingPolicy.STOP, True)):
        with (
            binding_case(runtime=True, missing=policy) as (b, _, _, _, execution, root, peers),
            patch("torch.cuda.current_stream", return_value=None),
        ):
            session, model, _ = _session(b, execution)
            session.bind_readonly_input(
                lambda batch, stream: (
                    ReadonlyStepInput(1, 0, (root,))
                    if seeded and batch.step_id == 0
                    else ReadonlyStepAbsent(1, batch.step_id)
                )
            )
            session.stage_input(None, (identity,))
            session.before_execute(_scheduler(identity))
            if seeded:
                session.prepare_input(rows(0))
                model.model.layers[0](torch.zeros(1), torch.ones(1, 2))
                session.complete_forward()
                session.complete_sampler(False)
                session.commit_token()
                assert execution.diagnostics().update_count == 1
                for peer in peers:
                    peer.recv_bytes()
            with pytest.raises(
                RuntimeError, match="first successful injection" if policy is MissingPolicy.HOLD else "STOP"
            ):
                session.prepare_input(rows(int(seeded)))
            assert not execution._active and not any(peer.poll() for peer in peers)
            with pytest.raises(RuntimeError, match="fail-stopped"):
                session.prepare_input(rows(0))
    with binding_case(runtime=True, missing=MissingPolicy.HOLD) as (_, _, _, _, execution, root, peers):
        execution.stage_readonly((root,))
        with pytest.raises(RuntimeError, match="first successful injection"):
            execution.stage_hold((identity,), "absent", step_id=0)
        assert execution._pending_borrows == (root,) and not any(peer.poll() for peer in peers)
