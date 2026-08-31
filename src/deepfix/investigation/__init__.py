from importlib import import_module

from deepfix.investigation.errors import (
    InvestigationCoordinationError,
    InvestigationStagnationError,
    InvestigationStateError,
)
from deepfix.investigation.identity import stable_investigation_id
from deepfix.investigation.models import (
    InvestigationCapability,
    InvestigationEvent,
    InvestigationEventType,
    InvestigationRecoveryMetadata,
    InvestigationState,
    NewInvestigationEvent,
    ProgressKind,
    RecordHypothesisInput,
    ScopeKind,
)

_LAZY_EXPORTS = {
    "InvestigationMiddleware": (
        "deepfix.investigation.middleware",
        "InvestigationMiddleware",
    ),
    "InvestigationMigrationMiddleware": (
        "deepfix.investigation.migration",
        "InvestigationMigrationMiddleware",
    ),
    "InvestigationMigrator": (
        "deepfix.investigation.migration",
        "InvestigationMigrator",
    ),
    "InvestigationStore": (
        "deepfix.investigation.store",
        "InvestigationStore",
    ),
    "build_record_hypothesis_tool": (
        "deepfix.investigation.tools",
        "build_record_hypothesis_tool",
    ),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value

__all__ = [
    "InvestigationCapability",
    "InvestigationCoordinationError",
    "InvestigationEvent",
    "InvestigationEventType",
    "InvestigationMiddleware",
    "InvestigationMigrationMiddleware",
    "InvestigationMigrator",
    "InvestigationRecoveryMetadata",
    "InvestigationStagnationError",
    "InvestigationState",
    "InvestigationStateError",
    "InvestigationStore",
    "NewInvestigationEvent",
    "ProgressKind",
    "RecordHypothesisInput",
    "ScopeKind",
    "build_record_hypothesis_tool",
    "stable_investigation_id",
]
