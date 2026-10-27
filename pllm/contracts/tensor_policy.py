"""Numerical permission and preparation choices shared by tensor execution and workers."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


@dataclass(frozen=True, slots=True)
class TensorPolicy:
    """Preparation policy serialized with the signed worker timeline."""

    allow_numerical_transforms: bool = True
    schedule_policy: Literal["auto", "serial"] = "auto"
    tuning: Literal["heuristic", "measured"] = "heuristic"


class ScanForm(StrEnum):
    """Numerical algorithms admitted for an additive carry."""

    ORDERED = "ordered"
    HILLIS_STEELE = "hillis_steele"
