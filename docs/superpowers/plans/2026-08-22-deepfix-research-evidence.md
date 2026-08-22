# DeepFix Technical Research Evidence Implementation Plan

> **For implementation:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan inline, task-by-task, with a review checkpoint after each task. Do not create subagents.

**Goal:** Give the single DeepFix Repair Agent safe, auditable Python technical research tools whose external findings must be tied to real local evidence before they can support a repair conclusion.

**Architecture:** Add a task-isolated research subsystem around four Agent tools: dependency inspection, constrained source search, safe evidence fetch, and local-evidence linking. Persist metadata in SQLite, store cleaned bodies in the existing artifact backend, inject bounded untrusted excerpts through middleware, and register the feature through a small extension layer.

**Tech Stack:** Python 3.11, Deep Agents 0.7, LangChain/LangGraph ToolRuntime and middleware, Pydantic 2, SQLite, httpx, stdlib tomllib/html.parser/socket/ipaddress, pytest, Ruff.

**Design reference:** docs/superpowers/specs/2026-08-22-deepfix-research-evidence-design.md

---

## Implementation rules

- Work only in C:\Users\17823\Documents\AI Agent\deepfix-agent on the current codex/deepfix-single-agent branch.
- Follow red-green-refactor for every behavior. Run the focused test first and confirm the expected failure before production edits.
- Default tests never access the network. Inject httpx.MockTransport, fake DNS, fake clock, and fake artifact backends.
- Derive task_id from ToolRuntime. Public tool schemas never accept task IDs, arbitrary URLs, API tokens, or artifact paths.
- Never persist or print GITHUB_TOKEN or TAVILY_API_KEY.
- Commit after each task.

## Task 1: Add research configuration and core models

**Files:**

- Modify: pyproject.toml
- Modify: src/deepfix/config.py
- Modify: src/deepfix/models.py
- Create: src/deepfix/research/__init__.py
- Create: src/deepfix/research/models.py
- Modify: tests/test_config.py
- Modify: tests/test_models.py
- Create: tests/research/test_models.py

**Step 1: Write failing tests**

Cover:

- load_config(..., project_python=...) resolves an existing executable; omission uses Path(sys.executable).resolve().
- DEEPFIX_SEARCH_PROVIDER=tavily enables the optional provider only with TAVILY_API_KEY; unknown values raise ValueError.
- TaskState round-trips external_evidence_ids, research_query_count, research_provider_errors, and remains backward-compatible.
- SearchCandidate and ExternalEvidence implement the approved fields/literals.
- DependencyFinding records package_name, constraints, installed_version, interpreter, sources, and a diagnostic.
- DependencyContext records the finding, confirmed official repository, and official domains.
- ResearchQuery records query ID, task ID, sanitized query, providers, provider errors, and timestamp.

Run and confirm import/attribute failures:

    pytest tests/test_config.py tests/test_models.py tests/research/test_models.py -q

**Step 2: Implement the minimum**

- Add direct dependency httpx>=0.27,<1.
- Add AppConfig.project_python and search_provider. Read secret values only during Provider construction, never store them in AppConfig.
- Make project_python a keyword-only load_config argument and validate that it is a file.
- Add the three research summary fields to TaskState with defaults.
- Define EvidenceLevel = Literal["E1", "E2", "E3"] and VerificationStatus = Literal["unverified", "verified", "contradicted"].
- Match the spec's SearchCandidate and ExternalEvidence fields exactly.

**Step 3: Verify and commit**

    pytest tests/test_config.py tests/test_models.py tests/research/test_models.py -q
    ruff check src/deepfix/config.py src/deepfix/models.py src/deepfix/research tests/research/test_models.py
    git add pyproject.toml src/deepfix/config.py src/deepfix/models.py src/deepfix/research tests/test_config.py tests/test_models.py tests/research/test_models.py
    git commit -m "feat: define research evidence models"

## Task 2: Persist task-isolated research metadata

**Files:**

- Create: src/deepfix/research/store.py
- Create: tests/research/test_store.py

**Step 1: Write failing tests**

Prove:

- Initialization creates research_queries, search_candidates, and external_evidence without disturbing existing tables.
- Unicode and nested local Evidence values round-trip.
- Candidate/evidence reads and updates require both task_id and record ID.
- IDs are random uuid4 hex values, not URL hashes.
- list_evidence and query_summary never leak another task's data.
- update_verification updates one current-task row.

Run:

    pytest tests/research/test_store.py -q

Expected: module import failure.

**Step 2: Implement ResearchEvidenceStore**

Reuse open_sqlite_connection, WAL, and busy timeout. Use composite primary keys and JSON payloads. Required methods:

    save_query(task_id, sanitized_query, providers, errors) -> ResearchQuery
    save_candidates(task_id, query, candidates) -> list[SearchCandidate]
    get_candidate(task_id, candidate_id) -> SearchCandidate
    save_evidence(evidence) -> None
    get_evidence(task_id, evidence_id) -> ExternalEvidence
    list_evidence(task_id) -> list[ExternalEvidence]
    update_verification(task_id, evidence_id, ...) -> ExternalEvidence
    query_summary(task_id) -> tuple[int, list[str]]

Generate UTC ISO timestamps and UUIDs before inserts.

**Step 3: Verify and commit**

    pytest tests/research/test_store.py tests/test_persistence.py -q
    ruff check src/deepfix/research/store.py tests/research/test_store.py
    git add src/deepfix/research/store.py tests/research/test_store.py
    git commit -m "feat: persist task research evidence"

## Task 3: Reject sensitive queries and unsafe URLs

**Files:**

- Create: src/deepfix/research/sanitizer.py
- Create: tests/research/test_sanitizer.py
- Create: tests/research/test_url_safety.py

**Step 1: Write failing query tests**

Reject API keys/tokens/private keys/high-entropy credentials, Windows/UNC/user-home paths, more than 20 lines, more than 2,000 characters, continuous code/log blocks over 400 characters, emails, internal hosts, database strings, and business identifiers. Accept package/version/exception/API queries. Rejection exposes a stable rule code and never returns a partially edited query.

**Step 2: Write failing URL tests**

With an injected resolver, reject HTTP, URL credentials, non-443 ports, IP literals, localhost, and every non-public resolved address. Reject a hostname if any answer is non-public. Test exact/subdomain allowlists and suffix tricks.

Run:

    pytest tests/research/test_sanitizer.py tests/research/test_url_safety.py -q

**Step 3: Implement pure validators**

Add SanitizedQuery, QueryRejected(rule), UnsafeUrl, ValidatedUrl, QuerySanitizer, and UrlSafetyPolicy. Use urllib.parse, socket.getaddrinfo, and ipaddress. Fail closed on DNS errors.

**Step 4: Verify and commit**

    pytest tests/research/test_sanitizer.py tests/research/test_url_safety.py -q
    ruff check src/deepfix/research/sanitizer.py tests/research
    git add src/deepfix/research/sanitizer.py tests/research/test_sanitizer.py tests/research/test_url_safety.py
    git commit -m "feat: guard research queries and urls"

## Task 4: Inspect declared and installed dependency versions

**Files:**

- Create: src/deepfix/research/dependency.py
- Create: tests/research/test_dependency.py

**Step 1: Write failing declaration parser tests**

Cover PEP 621 project.dependencies, requirements.txt, poetry.lock, uv.lock, normalized package names, duplicate declarations, and missing packages.

**Step 2: Write failing interpreter probe tests**

Inject a runner. Assert execution is an argument list:

    <project_python> -c <fixed importlib.metadata script> <normalized-package-name>

The package name is never interpolated into code. Cover success, not installed, timeout, malformed JSON, and nonzero exit. Failures return installed_version=None plus diagnostics.

Run:

    pytest tests/research/test_dependency.py -q

**Step 3: Implement DependencyInspector**

Use tomllib and packaging.requirements.Requirement; add packaging>=24,<27 as a direct dependency if necessary. Read only the four approved root files. Probe with subprocess.run(shell=False), captured output, a sanitized environment, and a short timeout. Do not expose this as a tool yet.

**Step 4: Verify and commit**

    pytest tests/research/test_dependency.py -q
    ruff check src/deepfix/research/dependency.py tests/research/test_dependency.py
    git add pyproject.toml src/deepfix/research/dependency.py tests/research/test_dependency.py
    git commit -m "feat: inspect project dependency versions"

## Task 5: Search official technical sources

**Files:**

- Create: src/deepfix/research/providers.py
- Create: tests/research/test_providers.py
- Create: tests/research/fixtures/pypi_project.json
- Create: tests/research/fixtures/github_search.json
- Create: tests/research/fixtures/github_release.json
- Create: tests/research/fixtures/tavily_search.json

**Step 1: Write failing Provider tests**

Using httpx.MockTransport, prove:

- PyPI normalizes names and discovers release/version, official repository, docs domains, and E1 candidates.
- GitHub refuses to search without a confirmed owner/repository and every query includes repo:owner/repository.
- GitHub code search is never called.
- Official releases/source are E1; merged PRs and API-confirmed maintainer/closed items are E2; unconfirmed items stay E3.
- GITHUB_TOKEN is only an HTTP header and never candidate/error content.
- Tavily skips cleanly without config and enforces official domains when enabled.
- Composite search retains successes when one provider times out or is rate-limited and returns structured errors.

Run:

    pytest tests/research/test_providers.py -q

**Step 2: Implement Providers**

Implement TechnicalSearchProvider, PyPIProvider, GitHubProvider, TavilyProvider, CompositeTechnicalSearchProvider. Constructors accept httpx.Client. PyPI establishes trusted URLs before GitHub/Tavily. Provider output is unsaved candidate data; the Store assigns task-local IDs. Do not add a Tavily SDK. Do not indefinitely retry 403/429.

Use GitHub REST for Issue/PR/Release discovery and metadata. Use GitHub GraphQL only for repository Discussions because GitHub exposes Discussions through its GraphQL API; keep the same confirmed-repository restriction, injected HTTP client, token handling, and offline fixtures. If Discussions are disabled or GraphQL is unavailable, record a Provider error and retain successful REST results.

**Step 3: Verify and commit**

    pytest tests/research/test_providers.py -q
    ruff check src/deepfix/research/providers.py tests/research/test_providers.py
    git add src/deepfix/research/providers.py tests/research
    git commit -m "feat: search official technical sources"

## Task 6: Fetch and clean selected evidence safely

**Files:**

- Create: src/deepfix/research/fetcher.py
- Create: tests/research/test_fetcher.py

**Step 1: Write failing fetch tests**

Cover manual redirect revalidation, maximum three redirects, public-to-private redirect rejection before request, approved content types only, missing/invalid type, binary content, declared/streamed size over 2 MiB, and timeout errors without retry loops.

**Step 2: Write failing cleaning tests**

Remove scripts, styles, forms, nav, hidden nodes, comments, and cookie boilerplate. Preserve headings, prose, lists, code, and useful links. Format JSON. Bound cleaned text to 100,000 characters. Wrap content in external_untrusted_source with the approved warning.

Run:

    pytest tests/research/test_fetcher.py -q

**Step 3: Implement SafeEvidenceFetcher**

Use follow_redirects=False and validate every target. Stream and stop after 2 MiB. Use stdlib HTMLParser. Return final URL, media type, cleaned body, and bounded excerpt; never raw HTML or response headers.

**Step 4: Verify and commit**

    pytest tests/research/test_fetcher.py tests/research/test_url_safety.py -q
    ruff check src/deepfix/research/fetcher.py tests/research/test_fetcher.py
    git add src/deepfix/research/fetcher.py tests/research/test_fetcher.py
    git commit -m "feat: safely fetch external evidence"

## Task 7: Expose inspect, search, and fetch tools

**Files:**

- Create: src/deepfix/research/tools.py
- Create: tests/research/test_tools.py

**Step 1: Write failing tool tests**

Prove:

- Missing runtime thread_id returns an error ToolMessage and does no work.
- Tool schemas do not expose task_id, URL, secrets, or artifact path.
- inspect_dependency returns declared and installed versions.
- Rejected search sends no request and saves nothing.
- Valid search saves query/errors/current-task candidates and returns bounded metadata.
- Fetch accepts only candidate_id; missing/cross-task candidates send no request.
- Success writes exactly /.deepfix-artifacts/research/{task_id}/{evidence_id}.md before persisting.
- Artifact write failure creates no evidence row.

Run:

    pytest tests/research/test_tools.py -q

**Step 2: Implement tool factories**

Add:

    build_inspect_dependency_tool(inspector)
    build_search_technical_sources_tool(sanitizer, inspector, provider, store)
    build_fetch_external_evidence_tool(fetcher, store, artifact_backend)

Use a shared runtime task-ID helper. Validation/provider/fetch failures return status="error" ToolMessages so local investigation can continue.

**Step 3: Verify and commit**

    pytest tests/research/test_tools.py tests/research/test_store.py -q
    ruff check src/deepfix/research/tools.py tests/research/test_tools.py
    git add src/deepfix/research/tools.py tests/research/test_tools.py
    git commit -m "feat: add technical research tools"

## Task 8: Link external findings to real local evidence

**Files:**

- Modify: src/deepfix/research/tools.py
- Modify: tests/research/test_tools.py

**Step 1: Add failing link tests**

Cover current-task ownership, fabricated IDs, missing exit_code, failing pytest cannot verify, real passing pytest can verify, contradicted accepts a real failed test or explicit source Evidence, empty contradiction fails, and relinking updates rather than duplicates. Build runtime state with paired AIMessage.tool_calls and ToolMessages, matching BugfixService behavior.

Run:

    pytest tests/research/test_tools.py -q -k link

**Step 2: Implement build_link_external_evidence_tool(store)**

Inspect runtime.state["messages"]. Map real calls to real ToolMessages. For verified, accept only a referenced execute call whose command is pytest and artifact exit_code is integer zero. Validate everything before updating the Store. Return only evidence ID and verification state in the Tool artifact.

**Step 3: Verify and commit**

    pytest tests/research/test_tools.py -q
    ruff check src/deepfix/research/tools.py tests/research/test_tools.py
    git add src/deepfix/research/tools.py tests/research/test_tools.py
    git commit -m "feat: bind research to local verification"

## Task 9: Add extension registration without weakening HITL

**Files:**

- Create: src/deepfix/extensions.py
- Modify: src/deepfix/agent.py
- Modify: src/deepfix/approval.py
- Modify: tests/test_agent.py
- Modify: tests/test_approval.py
- Create: tests/test_extensions.py

**Step 1: Write failing extension tests**

Prove duplicate tool names fail; each extension has RiskLevel, PolicyAction, and network flag; ASK/DENY enter interrupt_on while ALLOW does not; protected middleware cannot be replaced; skills must be beneath an explicit allowlist and default to none; research tools have approved L0/L1 ALLOW metadata.

**Step 2: Write Agent assembly regressions**

Patch create_deep_agent and assert save_progress/compaction remain, the four research tools appear once, subagents stays empty, and write/edit/delete/execute interrupts are unchanged.

Run:

    pytest tests/test_extensions.py tests/test_agent.py tests/test_approval.py -q

**Step 3: Implement the registry**

Create immutable ToolRegistration and AgentExtensions plus a validated merge function. build_agent accepts extensions with an empty default. Build research extensions in a separate factory so agent.py does not know Provider internals. Preserve unknown-tool L2 behavior and never route ALLOW research tools through HITL.

**Step 4: Verify and commit**

    pytest tests/test_extensions.py tests/test_agent.py tests/test_approval.py -q
    ruff check src/deepfix/extensions.py src/deepfix/agent.py src/deepfix/approval.py tests
    git add src/deepfix/extensions.py src/deepfix/agent.py src/deepfix/approval.py tests/test_extensions.py tests/test_agent.py tests/test_approval.py
    git commit -m "feat: register agent capability extensions"

## Task 10: Inject phase policy and bounded evidence

**Files:**

- Create: src/deepfix/prompting.py
- Create: src/deepfix/research/middleware.py
- Modify: src/deepfix/prompts.py
- Modify: src/deepfix/context.py
- Modify: src/deepfix/agent.py
- Create: tests/test_prompting.py
- Create: tests/research/test_middleware.py
- Modify: tests/test_context.py

**Step 1: Write failing PromptPolicyMiddleware tests**

Assert default investigating phase, current-task latest snapshot selection, CORE_REPAIR_PROMPT plus exactly one phase prompt plus RESEARCH_POLICY_PROMPT, no dynamic prompt in message history, and local evidence priority.

**Step 2: Write failing ResearchEvidenceMiddleware tests**

Assert current-task isolation; ordering of verified E1/E2, contradicted, unverified E1/E2, unverified E3; maximum five records; maximum 800 excerpt characters; maximum 8,000 total characters; version/E3 warnings; escaped untrusted XML; artifact references but no full bodies.

Run:

    pytest tests/test_prompting.py tests/research/test_middleware.py tests/test_context.py -q

**Step 3: Implement prompt composition**

Split the current static prompt into CORE_REPAIR_PROMPT, all six PHASE_PROMPTS, and RESEARCH_POLICY_PROMPT. Use the same runtime thread-ID technique as ContextMemoryMiddleware. Add explicit middleware ordering tests. Keep a temporary REPAIR_SYSTEM_PROMPT alias only if needed during migration.

**Step 4: Verify and commit**

    pytest tests/test_prompting.py tests/research/test_middleware.py tests/test_context.py tests/test_agent.py -q
    ruff check src/deepfix/prompting.py src/deepfix/prompts.py src/deepfix/research/middleware.py tests
    git add src/deepfix/prompting.py src/deepfix/prompts.py src/deepfix/context.py src/deepfix/agent.py src/deepfix/research/middleware.py tests
    git commit -m "feat: inject phase and research context"

## Task 11: Synchronize state, report evidence, and wire CLI

**Files:**

- Create: src/deepfix/research/reporting.py
- Modify: src/deepfix/service.py
- Modify: src/deepfix/reporting.py
- Modify: src/deepfix/cli.py
- Modify: src/deepfix/models.py
- Modify: tests/test_service.py
- Modify: tests/test_reporting.py
- Modify: tests/test_cli.py

**Step 1: Write failing service tests**

Every save synchronizes current-task evidence IDs, query count, and bounded Provider errors. External excerpts never enter TaskState.evidence. Provider failure never makes the task FAILED.

**Step 2: Write failing report tests**

Change the boundary to:

    render_report(task, external_evidence: Sequence[ExternalEvidence] = ()) -> str

Verified E1/E2 shows source, versions, external conclusion, true local linkage, URL, artifact. E3/unverified says “仅为外部线索”. Contradictions and version mismatches remain visible. Report data comes from Store records, not model prose.

**Step 3: Write failing CLI tests**

deepfix new --python PATH passes/persists the interpreter; resume reuses it. CLI creates one shared store/client/fetcher/extension/service. No-key startup works. Reports query only the current task.

Run:

    pytest tests/test_service.py tests/test_reporting.py tests/test_cli.py -q

**Step 4: Implement integration**

Add ResearchEvidenceStore to BugfixService and sync it before persistence. Persist project_python in TaskState for reliable resume. Add --python only to new. Construct clients with 5-second connect and 15-second overall timeout, close them at CLI boundary, and read tokens only during Provider construction.

**Step 5: Verify and commit**

    pytest tests/test_service.py tests/test_reporting.py tests/test_cli.py -q
    ruff check src/deepfix/service.py src/deepfix/reporting.py src/deepfix/research/reporting.py src/deepfix/cli.py tests
    git add src/deepfix/service.py src/deepfix/reporting.py src/deepfix/research/reporting.py src/deepfix/cli.py src/deepfix/models.py tests
    git commit -m "feat: integrate research evidence workflow"

## Task 12: Prove the workflow and document it

**Files:**

- Create: tests/research/test_workflow.py
- Create: tests/research/test_online.py
- Modify: pyproject.toml
- Modify: README.md

**Step 1: Write a failing offline workflow test**

Drive real components with mocked HTTP through dependency inspection, official search, E1/E2/E3 fetch, real passing pytest ToolMessage linkage, TaskState sync, and report rendering.

In the same test prove: sensitive queries cause zero requests; arbitrary URLs and cross-task candidates fail; fabricated/failed tests cannot verify; bodies exist only in artifacts; the target project remains unchanged; external evidence cannot satisfy BugfixService's local passing-test completion gate.

Run:

    pytest tests/research/test_workflow.py -q

**Step 2: Add optional online smoke tests**

Register online marker and default exclusion. Online tests may call only public PyPI/GitHub endpoints and must skip cleanly if their explicit prerequisites are absent.

    pytest --collect-only -q
    pytest -m online tests/research/test_online.py -q

The second command may be skipped in an offline environment; a skip is not proof of online success.

**Step 3: Update README**

Document evidence flow, four tools, E1/E2/E3, --python, optional tokens/Tavily and no-key fallback, security boundaries, artifacts, approval behavior, test commands, and an interview demonstration using a disposable Python fixture.

**Step 4: Full verification**

    pytest -q
    ruff check .
    git diff --check
    git status --short

**Step 5: Commit**

    git add pyproject.toml README.md tests/research/test_workflow.py tests/research/test_online.py
    git commit -m "test: prove research evidence workflow"

## Final acceptance checkpoint

Run and record exact outputs:

    pytest -q
    ruff check .
    git diff --check
    git status --short
    git log --oneline -12

Then compare the final diff against every design-spec section. Completion requires:

- Default tests make zero real network requests.
- Four research tools are visible to the single Agent; no subagent is enabled.
- Sensitive queries cannot leave the process.
- Fetch cannot accept arbitrary URLs or cross task boundaries.
- SSRF and redirect checks fail closed.
- Bodies stay under DEEPFIX_HOME artifacts.
- Fabricated or failing ToolMessages cannot produce verified evidence.
- Prompt/evidence injection is task-local and bounded.
- Existing HITL and local-test completion gates remain unchanged.
- Missing Tavily/GitHub tokens do not prevent local-only repair.
- Reports distinguish external leads, verified support, and contradictions.

Do not claim completion from focused tests alone; the final full-suite output is the completion evidence.
