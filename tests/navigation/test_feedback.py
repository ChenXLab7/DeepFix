from __future__ import annotations

from dataclasses import dataclass

from deepfix.compaction.models import FileChangeEvidence, SystemTestEvidence
from deepfix.compaction.store import CompactionStore
from deepfix.investigation.models import (
    AgentPhase,
    InvestigationEventType,
    InvestigationState,
    NewInvestigationEvent,
)
from deepfix.investigation.store import InvestigationStore
from deepfix.navigation.feedback import (
    LegacyNavigationFeedbackSource,
    NavigationFeedback,
)
from deepfix.verification import (
    OracleConflictRule,
    VerificationOracle,
    VerificationPolicy,
    VerificationPolicyStore,
)


@dataclass
class Stores:
    investigation: InvestigationStore
    evidence: CompactionStore
    verification: VerificationPolicyStore

    @property
    def source(self) -> LegacyNavigationFeedbackSource:
        return LegacyNavigationFeedbackSource(
            self.investigation,
            self.evidence,
            self.verification,
        )


def _stores(tmp_path) -> Stores:
    database = tmp_path / "deepfix.sqlite3"
    return Stores(
        investigation=InvestigationStore(database),
        evidence=CompactionStore(database),
        verification=VerificationPolicyStore(database),
    )


def _policy(*oracles: VerificationOracle) -> VerificationPolicy:
    return VerificationPolicy(
        policy_id="policy-a",
        task_id="task-a",
        version=1,
        required_oracles=list(oracles),
        supplemental_oracles=[],
        conflict_rules=[],
    )


def _oracle(
    oracle_id: str = "required-a",
    *,
    origin: str = "user_specified",
    timing: str = "post_change",
) -> VerificationOracle:
    return VerificationOracle(
        oracle_id=oracle_id,
        origin=origin,
        command="python -m pytest tests/test_value.py -q",
        scope="targeted",
        role="required",
        required_timing=timing,
    )


def _test_evidence(
    evidence_id: str,
    *,
    command: str = "python -m pytest tests/test_value.py -q",
    exit_code: int = 0,
    origin: str = "user_specified",
    timing: str = "post_change",
    scope: str = "targeted",
) -> SystemTestEvidence:
    return SystemTestEvidence(
        evidence_id=evidence_id,
        command=command,
        exit_code=exit_code,
        summary="test result",
        tool_call_id=f"call-{evidence_id}",
        source_message_id=f"message-{evidence_id}",
        origin=origin,
        scope=scope,
        timing=timing,
        workspace_baseline_id="baseline-a",
        code_state_hash="code-a",
    )


def _successful_change(path: str = "src/value.py") -> FileChangeEvidence:
    return FileChangeEvidence(
        evidence_id=f"change-{path}",
        path=path,
        operation="edit",
        status="succeeded",
    )


def _save_state(store: InvestigationStore, **updates: object) -> InvestigationState:
    current = store.ensure_started("task-a")
    event = NewInvestigationEvent(
        event_id=f"feedback-state-{current.version}",
        task_id="task-a",
        event_type=InvestigationEventType.TOOL_COMPLETED,
        phase_before=AgentPhase.INVESTIGATING,
        phase_after=AgentPhase.INVESTIGATING,
    )
    return store.commit(
        current.version,
        [event],
        current.model_copy(update=updates),
    )


def test_empty_stores_produce_empty_deterministic_feedback(tmp_path):
    feedback = _stores(tmp_path).source.build("task-a")

    assert feedback.milestone_ids == ()
    assert feedback.lines == ()
    assert feedback.fingerprint == (
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
    )


def test_supported_hypothesis_is_a_navigation_hint(tmp_path):
    stores = _stores(tmp_path)
    _save_state(stores.investigation, supported_hypothesis_ids=["h2", "h1", "h1"])

    feedback = stores.source.build("task-a")

    assert feedback.milestone_ids == (
        "supported-hypothesis:h1",
        "supported-hypothesis:h2",
    )
    assert feedback.lines == ("Supported hypothesis: h1.", "Supported hypothesis: h2.")


def test_closed_evidence_gap_is_not_called_a_question(tmp_path):
    stores = _stores(tmp_path)
    _save_state(stores.investigation, closed_evidence_gap_ids=["gap-2", "gap-1"])

    feedback = stores.source.build("task-a")

    assert feedback.milestone_ids == (
        "closed-evidence-gap:gap-1",
        "closed-evidence-gap:gap-2",
    )
    assert all("question" not in line.lower() for line in feedback.lines)
    assert feedback.lines == ("Closed evidence gap: gap-1.", "Closed evidence gap: gap-2.")


def test_successful_change_without_required_post_change_verification_is_pending(tmp_path):
    stores = _stores(tmp_path)
    stores.verification.save(_policy(_oracle()))
    stores.evidence.save_evidence("task-a", _successful_change("z.py"))
    stores.evidence.save_evidence("task-a", _successful_change("a.py"))

    feedback = stores.source.build("task-a")

    assert "verification-pending" in feedback.milestone_ids
    assert "Changed files: a.py, z.py; required verification is pending." in feedback.lines


def test_satisfied_required_post_change_oracles_are_reported_without_pending(tmp_path):
    stores = _stores(tmp_path)
    stores.verification.save(_policy(_oracle("oracle-b"), _oracle("oracle-a")))
    stores.evidence.save_evidence("task-a", _successful_change())
    stores.evidence.save_evidence("task-a", _test_evidence("post-change-pass"))

    feedback = stores.source.build("task-a")

    assert "required-oracles-satisfied" in feedback.milestone_ids
    assert "verification-pending" not in feedback.milestone_ids
    assert "Required oracles satisfied: oracle-a, oracle-b." in feedback.lines


def test_later_successful_change_makes_historical_oracle_pass_stale(tmp_path):
    stores = _stores(tmp_path)
    stores.verification.save(_policy(_oracle()))
    stores.evidence.save_evidence("task-a", _successful_change("first.py"))
    stores.evidence.save_evidence("task-a", _test_evidence("first-pass"))
    stores.evidence.save_evidence("task-a", _successful_change("later.py"))

    feedback = stores.source.build("task-a")

    assert "verification-pending" in feedback.milestone_ids
    assert "required-oracles-satisfied" not in feedback.milestone_ids
    assert "Changed files: first.py, later.py; required verification is pending." in feedback.lines


def test_required_oracle_pass_after_latest_change_is_current(tmp_path):
    stores = _stores(tmp_path)
    stores.verification.save(_policy(_oracle()))
    stores.evidence.save_evidence("task-a", _successful_change("first.py"))
    stores.evidence.save_evidence("task-a", _test_evidence("first-pass"))
    stores.evidence.save_evidence("task-a", _successful_change("later.py"))
    stores.evidence.save_evidence("task-a", _test_evidence("current-pass"))

    feedback = stores.source.build("task-a")

    assert "required-oracles-satisfied" in feedback.milestone_ids
    assert "verification-pending" not in feedback.milestone_ids


def test_unavailable_failed_or_conflicting_oracles_do_not_claim_satisfaction(tmp_path):
    stores = _stores(tmp_path)
    stores.verification.save(
        VerificationPolicy(
            policy_id="policy-a",
            task_id="task-a",
            version=1,
            required_oracles=[_oracle()],
            supplemental_oracles=[],
            conflict_rules=[],
        )
    )
    stores.evidence.save_evidence("task-a", _successful_change())
    assert "required-oracles-satisfied" not in stores.source.build("task-a").milestone_ids

    stores.evidence.save_evidence("task-a", _test_evidence("failed", exit_code=1))
    assert "required-oracles-satisfied" not in stores.source.build("task-a").milestone_ids

    conflict_stores = _stores(tmp_path / "conflict")
    conflict_stores.verification.save(
        VerificationPolicy(
            policy_id="policy-a",
            task_id="task-a",
            version=1,
            required_oracles=[_oracle()],
            supplemental_oracles=[],
            conflict_rules=[
                OracleConflictRule(
                    rule_id="conflict-rule",
                    description="Related suite failures block completion.",
                    blocking_scopes=["full_suite"],
                )
            ],
        )
    )
    conflict_stores.evidence.save_evidence("task-a", _test_evidence("passed"))
    conflict_stores.evidence.save_evidence(
        "task-a",
        _test_evidence(
            "suite-failed",
            command="python -m pytest -q",
            exit_code=1,
            origin="repository_existing",
            scope="full_suite",
        ),
    )
    assert (
        "required-oracles-satisfied"
        not in conflict_stores.source.build("task-a").milestone_ids
    )


def test_matching_user_baseline_pass_before_change_is_not_reproduced(tmp_path):
    stores = _stores(tmp_path)
    stores.verification.save(_policy(_oracle(timing="baseline")))
    stores.evidence.save_evidence(
        "task-a",
        _test_evidence(
            "baseline-pass",
            command=" PYTHON -m pytest tests\\test_value.py   -q ",
            timing="baseline",
        ),
    )

    feedback = stores.source.build("task-a")

    assert "baseline-not-reproduced" in feedback.milestone_ids
    assert "User-specified baseline verification passed before any code change." in feedback.lines


def test_non_user_baseline_pass_does_not_claim_non_reproduction(tmp_path):
    stores = _stores(tmp_path)
    stores.verification.save(_policy(_oracle(timing="baseline")))
    stores.evidence.save_evidence(
        "task-a",
        _test_evidence(
            "repository-pass", origin="repository_existing", timing="baseline"
        ),
    )
    stores.evidence.save_evidence(
        "task-a",
        _test_evidence("agent-pass", origin="agent_generated", timing="baseline"),
    )

    assert "baseline-not-reproduced" not in stores.source.build("task-a").milestone_ids


def test_successful_change_or_failed_matching_baseline_suppresses_non_reproduction(tmp_path):
    stores = _stores(tmp_path)
    stores.verification.save(_policy(_oracle(timing="baseline")))
    stores.evidence.save_evidence(
        "task-a", _test_evidence("baseline-pass", timing="baseline")
    )
    stores.evidence.save_evidence(
        "task-a", _test_evidence("baseline-failed", timing="baseline", exit_code=1)
    )
    assert "baseline-not-reproduced" not in stores.source.build("task-a").milestone_ids

    change_stores = _stores(tmp_path / "change")
    change_stores.verification.save(_policy(_oracle(timing="baseline")))
    change_stores.evidence.save_evidence(
        "task-a", _test_evidence("baseline-pass", timing="baseline")
    )
    change_stores.evidence.save_evidence("task-a", _successful_change())
    assert "baseline-not-reproduced" not in change_stores.source.build("task-a").milestone_ids


def test_repeated_builds_are_equal_and_distinct_milestones_change_fingerprint(tmp_path):
    stores = _stores(tmp_path)
    first = stores.source.build("task-a")
    second = stores.source.build("task-a")
    _save_state(stores.investigation, closed_evidence_gap_ids=["gap-1"])
    changed = stores.source.build("task-a")

    assert first == second
    assert first.fingerprint != changed.fingerprint


class _ReadOnlyInvestigationStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def load(self, task_id: str) -> None:
        self.calls.append(("load", task_id))


class _ReadOnlyEvidenceStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def list_evidence(self, task_id: str) -> list[object]:
        self.calls.append(("list_evidence", task_id))
        return []


class _ReadOnlyVerificationStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def load(self, task_id: str) -> None:
        self.calls.append(("load", task_id))


def test_build_only_invokes_store_read_methods():
    investigation = _ReadOnlyInvestigationStore()
    evidence = _ReadOnlyEvidenceStore()
    verification = _ReadOnlyVerificationStore()

    feedback = LegacyNavigationFeedbackSource(
        investigation,  # type: ignore[arg-type]
        evidence,  # type: ignore[arg-type]
        verification,  # type: ignore[arg-type]
    ).build("task-a")

    assert feedback == NavigationFeedback(milestone_ids=(), lines=(), fingerprint=feedback.fingerprint)
    assert investigation.calls == [("load", "task-a")]
    assert evidence.calls == [("list_evidence", "task-a")]
    assert verification.calls == [("load", "task-a")]
