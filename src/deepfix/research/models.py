from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EvidenceLevel = Literal["E1", "E2", "E3"]
VerificationStatus = Literal["unverified", "verified", "contradicted"]
SourceType = Literal[
    "official_docs",
    "official_source",
    "release_note",
    "pypi_metadata",
    "github_issue",
    "github_pr",
    "github_discussion",
]


class LocalEvidenceReference(BaseModel):
    source: str
    observation: str


class DependencyFinding(BaseModel):
    package_name: str
    declared_constraints: list[str] = Field(default_factory=list)
    installed_version: str | None = None
    python_executable: str
    source_files: list[str] = Field(default_factory=list)
    diagnostic: str | None = None


class DependencyContext(BaseModel):
    package: DependencyFinding
    official_repository: str | None = None
    official_domains: list[str] = Field(default_factory=list)


class SearchCandidate(BaseModel):
    candidate_id: str
    task_id: str
    source_type: SourceType
    evidence_level: EvidenceLevel
    title: str
    url: str
    query: str
    repository: str | None
    created_at: str


class ExternalEvidence(BaseModel):
    evidence_id: str
    task_id: str
    candidate_id: str
    source_type: str
    evidence_level: EvidenceLevel
    title: str
    url: str
    query: str
    relevant_excerpt: str
    retrieved_at: str
    dependency_name: str | None
    documented_version: str | None
    project_version: str | None
    local_verification: VerificationStatus
    local_evidence: list[LocalEvidenceReference] = Field(default_factory=list)
    linked_test_tool_call_ids: list[str] = Field(default_factory=list)
    verification_explanation: str | None
    artifact_path: str


class ResearchQuery(BaseModel):
    query_id: str
    task_id: str
    sanitized_query: str
    providers: list[str] = Field(default_factory=list)
    provider_errors: list[str] = Field(default_factory=list)
    created_at: str
