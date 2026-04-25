"""Time horizon model."""

from __future__ import annotations

from dataclasses import dataclass


VALID_STEP_UNITS: frozenset[str] = frozenset({"day", "week", "month", "quarter"})


@dataclass(frozen=True)
class TimeHorizon:
    steps: int
    step_unit: str

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError(f"TimeHorizon.steps must be >= 0, got {self.steps}")
        if self.step_unit not in VALID_STEP_UNITS:
            raise ValueError(
                f"TimeHorizon.step_unit must be one of {sorted(VALID_STEP_UNITS)}, got {self.step_unit!r}"
            )

    def to_dict(self) -> dict:
        return {"steps": self.steps, "step_unit": self.step_unit}

    @classmethod
    def from_dict(cls, data: dict) -> "TimeHorizon":
        return cls(steps=int(data["steps"]), step_unit=str(data["step_unit"]))
