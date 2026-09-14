"""Test-only full re-derivation validators, removed from production install paths.

Production binding derives each artifact once and treats the local derivation
as authoritative; these wrappers re-derive from source and compare, which is
useful only when a test wants to check tampered or transported artifacts.
"""

from pllm import _core
from pllm.contracts.backend_attachment import (
    BackendAttachmentCompilerInputs,
    BackendAttachmentDecision,
    BackendAttachmentPhysicalPlan,
)
from pllm.contracts.region_bindings import RegionBindings
from pllm.contracts.region_payload import RegionPayload
from pllm.contracts.region_program import Program
from pllm.contracts.region_retention import RetainedAllocation
from pllm.contracts.region_streams import RegionStreamPlan
from pllm.contracts.schedule_instructions import LoweredSchedule
from pllm.execution.codec import encode_boundary
from pllm.execution.region_storage import RegionImportSource, import_region_functions


def validate_backend_attachment(
    inputs: BackendAttachmentCompilerInputs,
    decision: BackendAttachmentDecision,
) -> None:
    _core.validate_backend_attachment(encode_boundary(inputs), encode_boundary(decision))


def validate_backend_attachment_reconstruction(
    inputs: BackendAttachmentCompilerInputs,
    decision: BackendAttachmentDecision,
    physical: BackendAttachmentPhysicalPlan,
    local_inputs: BackendAttachmentCompilerInputs,
    local_physical: BackendAttachmentPhysicalPlan,
) -> None:
    _core.validate_backend_attachment_reconstruction(
        encode_boundary(inputs),
        encode_boundary(decision),
        encode_boundary(physical),
        encode_boundary(local_inputs),
        encode_boundary(local_physical),
    )


def validate_lowered_region_schedule(program: Program, lowered: LoweredSchedule) -> None:
    _core.validate_lowered_region_schedule(encode_boundary(program), encode_boundary(lowered))


def validate_region_bindings(program: Program, bindings: RegionBindings) -> None:
    _core.validate_region_bindings(encode_boundary(program), encode_boundary(bindings))


def validate_region_payload(program: Program, cells: RegionImportSource, retained: tuple[RetainedAllocation, ...],
                            payload: RegionPayload) -> None:
    _core.validate_region_payload(encode_boundary(program), encode_boundary(import_region_functions(program, cells)),
        encode_boundary(retained), encode_boundary(payload))


def validate_region_streams(program: Program, cells: RegionImportSource, plan: RegionStreamPlan,
                            retained: tuple[RetainedAllocation, ...] = ()) -> None:
    _core.validate_region_streams(encode_boundary(program), encode_boundary(import_region_functions(program, cells)),
                                 encode_boundary(retained), encode_boundary(plan))
