from deepfix.compaction.models import FileChangeEvidence, SystemTestEvidence
from deepfix.domain_repositories import DomainRepositories
from deepfix.investigation.models import InvestigationHypothesis
from deepfix.navigation.feedback import RepositoryNavigationFeedbackSource
from deepfix.task_domain.models import TaskDefinition
from deepfix.verification import VerificationOracle, VerificationPolicy
from deepfix.workspace import WorkspaceFactory, compute_code_state_hash


def _stores(tmp_path):
    repositories = DomainRepositories.create(tmp_path / "deepfix.sqlite3")
    return repositories, RepositoryNavigationFeedbackSource(repositories)


def _policy():
    return VerificationPolicy(
        policy_id="policy-a",
        task_id="task-a",
        version=1,
        required_oracles=[
            VerificationOracle(
                oracle_id="required-a",
                origin="user_specified",
                command="python -m pytest tests/test_value.py -q",
                scope="targeted",
                role="required",
            )
        ],
        supplemental_oracles=[],
        conflict_rules=[],
    )


def _test_evidence(timing="post_change"):
    return SystemTestEvidence(
        evidence_id=f"test-{timing}",
        command="python -m pytest tests/test_value.py -q",
        exit_code=0,
        summary="passed",
        tool_call_id=f"call-{timing}",
        source_message_id=f"message-{timing}",
        origin="user_specified",
        timing=timing,
        workspace_baseline_id="baseline-a",
        code_state_hash="code-a",
    )


def test_empty_repositories_produce_stable_empty_feedback(tmp_path):
    _, source = _stores(tmp_path)
    first = source.build("task-a")
    assert first == source.build("task-a")
    assert first.milestone_ids == ()


def test_supported_hypothesis_is_advisory_milestone(tmp_path):
    repositories, source = _stores(tmp_path)
    repositories.evidence.record_deterministic(
        "task-a",
        _test_evidence().model_copy(update={"evidence_id": "e-1"}),
        provenance_root_ids=["call-hypothesis"],
    )
    repositories.investigation.record_hypothesis(
        "task-a",
        InvestigationHypothesis(
            hypothesis_id="h-1",
            statement="parser branch is wrong",
            state="supported",
            evidence_ids=["e-1"],
            checked_locations=[],
            reason="traceback and test agree",
        ),
    )

    feedback = source.build("task-a")

    assert feedback.milestone_ids == ("supported-hypothesis:h-1",)
    assert feedback.lines == ("Supported hypothesis: h-1.",)


def test_change_requires_current_required_oracle(tmp_path):
    repositories, source = _stores(tmp_path)
    repositories.tasks.save_verification_policy(_policy())
    repositories.evidence.record_deterministic(
        "task-a",
        FileChangeEvidence(
            evidence_id="change-1",
            path="src/value.py",
            operation="edit",
            status="succeeded",
        ),
        provenance_root_ids=["call-edit"],
    )

    assert "verification-pending" in source.build("task-a").milestone_ids

    repositories.evidence.record_deterministic(
        "task-a", _test_evidence(), provenance_root_ids=["call-post-change"]
    )
    feedback = source.build("task-a")
    assert "required-oracles-satisfied" in feedback.milestone_ids
    assert "verification-pending" not in feedback.milestone_ids


def test_baseline_pass_without_change_reports_not_reproduced(tmp_path):
    repositories, source = _stores(tmp_path)
    policy = _policy().model_copy(
        update={
            "required_oracles": [
                _policy().required_oracles[0].model_copy(update={"required_timing": "baseline"})
            ]
        }
    )
    repositories.tasks.save_verification_policy(policy)
    repositories.evidence.record_deterministic(
        "task-a", _test_evidence("baseline"), provenance_root_ids=["call-baseline"]
    )
    assert "baseline-not-reproduced" in source.build("task-a").milestone_ids


def test_navigation_rechecks_live_workspace_after_passing_test(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "value.py").write_text("VALUE = 1\n")
    workspace = WorkspaceFactory(tmp_path / "workspaces").create("task-a", project)
    repositories, _ = _stores(tmp_path)
    source = RepositoryNavigationFeedbackSource(repositories, workspace=workspace)
    repositories.tasks.save_verification_policy(_policy())
    passing = _test_evidence().model_copy(
        update={
            "code_state_hash": compute_code_state_hash(workspace.root),
            "workspace_baseline_id": workspace.baseline.baseline_id,
        }
    )
    repositories.evidence.record_deterministic("task-a", passing, provenance_root_ids=["call-pass"])
    assert "required-oracles-satisfied" in source.build("task-a").milestone_ids
    (workspace.root / "value.py").write_text("VALUE = 2\n")
    assert "required-oracles-satisfied" not in source.build("task-a").milestone_ids


def test_navigation_resolves_live_workspace_from_task_definition(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "value.py").write_text("VALUE = 1\n")
    workspace = WorkspaceFactory(tmp_path / "workspaces").create("task-a", project)
    repositories, source = _stores(tmp_path)
    repositories.tasks.create_definition(
        TaskDefinition(
            task_id="task-a",
            original_message_id="message-a",
            original_problem="fix value",
            approval_mode="review",
            source_project_root=str(project),
            workspace_root=str(workspace.root),
            workspace_baseline_id=workspace.baseline.baseline_id,
            project_python="python",
            confinement_level="guarded_local",
            created_at="2026-09-06T00:00:00+00:00",
        )
    )
    repositories.tasks.save_verification_policy(_policy())
    repositories.evidence.record_deterministic(
        "task-a",
        _test_evidence().model_copy(
            update={
                "code_state_hash": compute_code_state_hash(workspace.root),
                "workspace_baseline_id": workspace.baseline.baseline_id,
            }
        ),
        provenance_root_ids=["call-pass"],
    )

    assert "required-oracles-satisfied" in source.build("task-a").milestone_ids
    (workspace.root / "value.py").write_text("VALUE = 2\n")
    assert "required-oracles-satisfied" not in source.build("task-a").milestone_ids


def test_offline_navigation_does_not_use_generated_test_to_refresh_old_pass(tmp_path):
    repositories, source = _stores(tmp_path)
    repositories.tasks.save_verification_policy(_policy())
    repositories.evidence.record_deterministic(
        "task-a", _test_evidence(), provenance_root_ids=["old-pass"]
    )
    repositories.evidence.record_deterministic(
        "task-a",
        FileChangeEvidence(
            evidence_id="change", path="value.py", operation="edit", status="succeeded"
        ),
        provenance_root_ids=["change"],
    )
    repositories.evidence.record_deterministic(
        "task-a",
        _test_evidence().model_copy(
            update={"evidence_id": "generated-pass", "origin": "agent_generated"}
        ),
        provenance_root_ids=["generated-pass"],
    )
    feedback = source.build("task-a")
    assert "required-oracles-satisfied" not in feedback.milestone_ids
    assert "verification-pending" in feedback.milestone_ids
