"""CPU binding evidence only: received-page fixtures do not emulate CUDA IPC."""

from contextlib import ExitStack, contextmanager
from dataclasses import replace
from multiprocessing import Pipe

import pytest
import torch

from pllm.contracts.semantic_graph import AnchorId, StateEffectKind
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.state_ref import LineageId, StateRef, VersionId
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.declaration import Dimensions, Field, When
from pllm.execution.codec import encode_boundary
from pllm.execution.ptts.page_bundle import BorrowedPage, PageBundleArena, PageFieldTransfer, PageSlot
from pllm.execution.ptts.page_borrow import BorrowedStateRoot
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts, compile_page_bundle_plan
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.execution.state_execution import MathBinding, StateSettlementError
from pllm.execution.tensor_binding import TensorBinding, TensorFrame
from pllm.execution.tensor_program import FunctionMath, TaskOutput, _read


@contextmanager
def binding_case(*, alias=False, write_failure=False):
    b = TensorBinding("video.binding", "a" * 64, Dimensions(2, 0))
    features, ports = b.parameters("features", {name: (2,) for name in ("qp", "residual", "mv", "segmentation")}, optimizer=None)
    encoded = b.tensor("encode", inputs=ports, when=When.BEFORE_MODEL)(
        lambda values: values["qp"] if alias else sum(values.values()))
    inject = b.state("inject", (Field("delta", (2,)),))
    def write(data, batch):
        if write_failure:
            raise ArithmeticError("writer failed after private allocation")
        return TaskOutput(field_writes=((LayoutFieldId(0), batch.results[encoded.symbol, "value"].unsqueeze(0)),), changed=(0,))
    transition = b.compute("write", inputs={"hidden": encoded["value"]}, outputs=("updated",),
        changes=inject, effect=StateEffectKind.TRANSITION, when=When.BEFORE_MODEL, function=write)
    b.publish(transition, inject, when=When.BEFORE_MODEL)
    plan = compile_page_bundle_plan((b.graph, b.layouts), RuntimeFacts(torch.device("cpu")).timeline(("transition/full_root@1",)),
        (PageBundleProviderFacts("video.binding@1", 3),), 1, 65536, torch.device("cpu"))
    math = FunctionMath({"inject": lambda roots: None}, lambda value: (),
        lambda value, batch: TensorFrame(value, ()), lambda data, result: result.publication)
    binding = MathBinding(math, b.bind_all(plan), borrowed_resources=("features",))
    arena = PageBundleArena(plan, _owned_resources=("inject",))
    execution = binding.bind(arena, AnchorId(4))
    execution.admit(())
    layout, = (item for item in plan.tensor_layout_decision.bindings if item.resource == "features")
    ref = StateRef(layout.layout_id, LineageId(1), VersionId(7))
    with ExitStack() as lifetime:
        pages, peers = [], []
        for i in range(4):
            peer, receiver = Pipe()
            lifetime.callback(peer.close)
            lifetime.callback(receiver.close)
            peers.append(peer)
            # Deliberately bypass receive_field: CPU fixtures test roots/ACK ordering, not grant validation.
            descriptor = PageFieldTransfer(plan.schedule_decision.signature, 1, ref, "features", LayoutFieldId(i), PageSlot(0, 0), 0)
            pages.append(BorrowedPage(descriptor, torch.full((2,), float(i + 1)), receiver))
        root = BorrowedStateRoot(plan, "features", ref, tuple(pages))
        yield b, features, ports, binding, execution, root, peers


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
        del batch, frame
        receipt = execution.finish_step(invocation)
        torch.testing.assert_close(execution.snapshot(receipt.after_refs[0])[0], torch.full((2,), 10.))
        assert all(peer.poll() and peer.recv_bytes() == encode_boundary(page.descriptor).encode()
            for peer, page in zip(peers, root.pages, strict=True))
        with pytest.raises(RuntimeError, match="closed"):
            root.tensors(invocation.context.pin)
        with pytest.raises(RuntimeError, match="staged roots"):
            execution.begin(())  # Never silently read local inject as the next features chunk.


@pytest.mark.parametrize("failure", ("alias", "close", "cancel", "missing", "mixed", "staging", "write"))
def test_video_binding_failure_retains_grants_and_stops_execution(failure):
    with binding_case(alias=failure == "alias", write_failure=failure == "write") as (_, _, _, _, execution, root, peers):
        before = execution.directory.latest_refs(())
        if failure == "mixed":
            with pytest.raises(ValueError, match="exact grant"):
                BorrowedStateRoot(execution.arena.plan, "features", replace(root.ref, version_id=VersionId(8)), root.pages)
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
            reason = {"close": "publication committed", "alias": "overlaps", "missing": "staged roots",
                "staging": "fail-stopped", "write": "writer failed"}[failure]
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
