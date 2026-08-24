from __future__ import annotations

from deepfix.investigation.models import InvestigationRecoveryMetadata


class InvestigationCoordinationError(RuntimeError):
    def __init__(self, recovery: InvestigationRecoveryMetadata) -> None:
        self.recovery = recovery
        super().__init__(recovery.error_code)


class InvestigationStateError(InvestigationCoordinationError):
    pass


class InvestigationStagnationError(InvestigationCoordinationError):
    pass

