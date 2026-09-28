from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    failure_type: str | None = None
    reason: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    recoverable: bool = False
    scope: str = "execution_validity"

    def to_dict(self):
        return asdict(self)


class VerifiedExecutionError(RuntimeError):
    def __init__(self, result: VerificationResult, trajectory: str):
        self.verification = result
        self.trajectory = trajectory
        super().__init__(f"{result.failure_type}: {result.reason}; trajectory={trajectory}")
