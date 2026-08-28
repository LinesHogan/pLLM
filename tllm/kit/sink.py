"""Legacy import path; production ownership is in the runtime."""

from tllm.runtime.workload_sinks import (
    MissingSinkRecord,
    SinkRecord,
    SinkStore as RuntimeSinkStore,
    StepRowIdentity,
)


class SinkStore(RuntimeSinkStore):
    """Compatibility adapter deleted with RegionAlgorithm."""

    def for_step(self, sink: str, step_id: int) -> SinkRecord | None:
        match self.lookup(sink, step_id):
            case MissingSinkRecord():
                return None
            case SinkRecord() as record:
                return record

__all__ = ["SinkRecord", "SinkStore", "StepRowIdentity"]
