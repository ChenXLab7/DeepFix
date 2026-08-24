from __future__ import annotations

from dataclasses import dataclass

from deepfix.investigation.identity import stable_investigation_id
from deepfix.investigation.models import (
    AgentPhase,
    InvestigationEventType,
    NewInvestigationEvent,
    ToolObservation,
)


@dataclass(frozen=True)
class PhaseResolution:
    phase: AgentPhase
    phase_event: NewInvestigationEvent | None


class PhaseResolver:
    def resolve(
        self,
        phase: AgentPhase,
        event: ToolObservation,
    ) -> AgentPhase:
        if event.event_type is InvestigationEventType.NEEDS_INPUT:
            return AgentPhase.CLARIFYING
        if (
            event.event_type is InvestigationEventType.TEST_OBSERVED
            and event.exit_code != 0
        ):
            return AgentPhase.DIAGNOSING
        if (
            phase in {AgentPhase.INVESTIGATING, AgentPhase.DIAGNOSING}
            and event.event_type is InvestigationEventType.HYPOTHESIS_SUPPORTED
        ):
            return AgentPhase.PLANNING
        if (
            phase is AgentPhase.PLANNING
            and event.event_type is InvestigationEventType.FILE_CHANGED
        ):
            return AgentPhase.EDITING
        if (
            phase is AgentPhase.EDITING
            and event.event_type
            is InvestigationEventType.VERIFICATION_EXECUTION_OBSERVED
        ):
            return AgentPhase.TESTING
        if (
            phase is AgentPhase.TESTING
            and event.event_type is InvestigationEventType.POST_EDIT_TEST_OBSERVED
        ):
            return (
                AgentPhase.REVIEWING
                if event.exit_code == 0
                else AgentPhase.DIAGNOSING
            )
        if (
            phase is AgentPhase.REVIEWING
            and event.event_type is InvestigationEventType.USER_INFORMATION_RECEIVED
        ):
            return AgentPhase.INVESTIGATING
        return phase

    def transition(
        self,
        task_id: str,
        phase: AgentPhase,
        event: ToolObservation,
    ) -> PhaseResolution:
        resolved = self.resolve(phase, event)
        if resolved is phase:
            return PhaseResolution(phase, None)
        trigger_identity = (
            event.result_fingerprint
            or event.hypothesis_id
            or event.tool_call_id
            or event.source_message_id
            or event.event_type.value
        )
        phase_event = NewInvestigationEvent(
            event_id=stable_investigation_id(
                "event",
                task_id,
                "phase",
                phase.value,
                resolved.value,
                trigger_identity,
            ),
            task_id=task_id,
            event_type=InvestigationEventType.PHASE_CHANGED,
            source_message_id=event.source_message_id,
            tool_call_id=event.tool_call_id,
            phase_before=phase,
            phase_after=resolved,
            progress_kind=None,
            payload={"trigger": event.event_type.value},
        )
        return PhaseResolution(resolved, phase_event)

