# DeepFix Role-Specific Model Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split DeepFix into independently configurable main-repair and compaction-extraction DeepSeek model roles while allowing one shared API Key and loading developer secrets safely from `src/deepfix/.env`.

**Architecture:** `AppConfig` owns two immutable `ModelRoleConfig` values containing a model name, `SecretStr` API Key, and shared HTTPS Base URL. `build_agent()` creates one `ChatDeepSeek` per role: the main instance is passed only to `create_deep_agent`, and the compaction instance is passed only to `CompactionCoordinator`/`CompactionDeltaGenerator`; `save_progress` remains a main-Agent tool with no model dependency. `load_config()` loads the package-local `.env` without overriding process variables and applies deterministic Key fallback before any runtime component is built.

**Tech Stack:** Python 3.11+, `python-dotenv`, Pydantic `SecretStr`, LangChain `ChatDeepSeek`, Deep Agents, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-22-deepfix-model-role-configuration-design.md`

## Global Constraints

- Default environment file is exactly `src/deepfix/.env`; do not search the current directory or target project.
- Process environment variables override `.env` values (`override=False`).
- Main Key fallback is `DEEPFIX_MAIN_API_KEY -> DEEPSEEK_API_KEY`.
- Compaction Key fallback is `DEEPFIX_COMPACTION_API_KEY -> DEEPFIX_MAIN_API_KEY -> DEEPSEEK_API_KEY`.
- Blank values count as missing; missing final credentials fail before model, Backend, task, or database initialization.
- Main model defaults to `deepseek-v4-pro`; compaction model defaults to `deepseek-v4-flash`.
- Both models use `temperature=0` and the shared `DEEPSEEK_BASE_URL`, defaulting to `https://api.deepseek.com`.
- Base URL must be a non-empty HTTPS URL; errors must not echo credentials or sensitive URL components.
- Working Memory remains main-Agent-driven through `save_progress`; do not add a third model or automatic memory extraction call.
- The compaction model has no tools and may only produce untrusted structured `CompactionDelta` candidates.
- Secrets must not enter `TaskState`, Graph state/config, SQLite payloads, Artifacts, reports, CLI output, exceptions, or `LocalShellBackend.env`.
- Do not fall back from the compaction model to the main model after a runtime failure; retain the current normal-zone passthrough and emergency pause behavior.
- Offline tests must never send a real DeepSeek request or read the developer's real `src/deepfix/.env`.

---

## File Structure

- Modify `pyproject.toml`: add the bounded `python-dotenv` runtime dependency.
- Modify `src/deepfix/config.py`: own `.env` loading, role configuration, Key fallback, HTTPS validation, and Secret redaction.
- Modify `src/deepfix/agent.py`: construct and inject two role-specific `ChatDeepSeek` instances.
- Modify `src/deepfix/compaction/budget.py`: provide explicit DeepSeek V4 context limits when the SDK profile is absent.
- Modify `src/deepfix/service.py`: sanitize model-boundary exception text before persisting it.
- Modify `README.md`: document one-Key/two-Key setup and the true Working Memory boundary.
- Modify `tests/test_config.py`: verify `.env`, precedence, fallback, defaults, validation, and Secret representation.
- Modify `tests/test_agent.py`: verify model construction and strict role wiring.
- Modify `tests/test_context.py`: migrate callers from `build_model` to `build_main_model`.
- Modify `tests/compaction/test_budget.py`: verify V4 fallback limits and unknown-model rejection.
- Modify `tests/test_backend.py`: prove no model-role Key reaches the target Shell.
- Modify `tests/test_service.py`: prove persisted failures redact both role Keys.
- Modify `tests/test_cli.py`: prove CLI configuration uses package-local `.env` and does not require the caller's current directory.
- Modify `tests/compaction/test_long_context_workflow.py`: construct the new `AppConfig` shape in the offline end-to-end fixture.

---

### Task 1: Add immutable role configuration and package-local `.env` loading

**Files:**

- Modify: `pyproject.toml:9-18`
- Modify: `src/deepfix/config.py:1-65`
- Modify: `tests/test_config.py`
- Modify: `tests/compaction/test_long_context_workflow.py:468-482`

**Interfaces:**

- Produces: `ModelRoleConfig(model_name: str, api_key: SecretStr, base_url: str)`.
- Produces: `AppConfig.main_model: ModelRoleConfig` and `AppConfig.compaction_model: ModelRoleConfig`.
- Produces: `load_config(project_root, approval_mode, *, project_python=None, env_file=_USE_DEFAULT_ENV_FILE) -> AppConfig`; the private sentinel resolves `_DEFAULT_ENV_FILE` at call time so tests can safely replace the path.
- Produces: `redact_config_secrets(value: str, config: AppConfig) -> str` for Task 4.
- Removes: `AppConfig.model_name`.

- [ ] **Step 1: Add the failing role default and Key fallback tests**

Extend `tests/test_config.py` with helpers that always pass an isolated environment-file path so tests never read the developer's real file:

```python
from pydantic import SecretStr


def load_isolated(project, mode, tmp_path, **kwargs):
    return load_config(
        project,
        mode,
        env_file=tmp_path / "missing.env",
        **kwargs,
    )


def test_model_roles_use_v4_defaults_and_common_key(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "common-secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))

    config = load_isolated(tmp_path, ApprovalMode.MANUAL, tmp_path)

    assert config.main_model.model_name == "deepseek-v4-pro"
    assert config.compaction_model.model_name == "deepseek-v4-flash"
    assert config.main_model.api_key.get_secret_value() == "common-secret"
    assert config.compaction_model.api_key.get_secret_value() == "common-secret"
    assert isinstance(config.main_model.api_key, SecretStr)


def test_compaction_key_falls_back_to_main_role_key(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPFIX_MAIN_API_KEY", "main-secret")
    monkeypatch.delenv("DEEPFIX_COMPACTION_API_KEY", raising=False)

    config = load_isolated(tmp_path, ApprovalMode.MANUAL, tmp_path)

    assert config.main_model.api_key.get_secret_value() == "main-secret"
    assert config.compaction_model.api_key.get_secret_value() == "main-secret"


def test_role_specific_keys_and_models_remain_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "common-secret")
    monkeypatch.setenv("DEEPFIX_MAIN_API_KEY", "main-secret")
    monkeypatch.setenv("DEEPFIX_COMPACTION_API_KEY", "compact-secret")
    monkeypatch.setenv("DEEPFIX_MAIN_MODEL", "main-model")
    monkeypatch.setenv("DEEPFIX_COMPACTION_MODEL", "compact-model")

    config = load_isolated(tmp_path, ApprovalMode.MANUAL, tmp_path)

    assert config.main_model.model_name == "main-model"
    assert config.compaction_model.model_name == "compact-model"
    assert config.main_model.api_key.get_secret_value() == "main-secret"
    assert config.compaction_model.api_key.get_secret_value() == "compact-secret"
```

- [ ] **Step 2: Add failing `.env`, precedence, validation, and redaction tests**

```python
def test_package_env_file_loads_without_overriding_process_environment(tmp_path, monkeypatch):
    for name in (
        "DEEPSEEK_API_KEY",
        "DEEPFIX_MAIN_API_KEY",
        "DEEPFIX_COMPACTION_API_KEY",
        "DEEPFIX_MAIN_MODEL",
        "DEEPFIX_COMPACTION_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DEEPSEEK_API_KEY=file-common\n"
        "DEEPFIX_MAIN_API_KEY=file-main\n"
        "DEEPFIX_MAIN_MODEL=file-model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPFIX_MAIN_API_KEY", "process-main")

    config = load_config(tmp_path, ApprovalMode.MANUAL, env_file=env_file)

    assert config.main_model.api_key.get_secret_value() == "process-main"
    assert config.compaction_model.api_key.get_secret_value() == "process-main"
    assert config.main_model.model_name == "file-model"


@pytest.mark.parametrize("name", ["DEEPFIX_MAIN_MODEL", "DEEPFIX_COMPACTION_MODEL"])
def test_blank_model_name_is_rejected(tmp_path, monkeypatch, name):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv(name, "   ")

    with pytest.raises(ValueError, match=name):
        load_isolated(tmp_path, ApprovalMode.MANUAL, tmp_path)


@pytest.mark.parametrize("url", ["", "http://api.deepseek.com", "not-a-url"])
def test_base_url_requires_https(tmp_path, monkeypatch, url):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", url)

    with pytest.raises(ValueError, match="DEEPSEEK_BASE_URL"):
        load_isolated(tmp_path, ApprovalMode.MANUAL, tmp_path)


def test_config_repr_and_redaction_never_expose_role_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPFIX_MAIN_API_KEY", "main-secret-value")
    monkeypatch.setenv("DEEPFIX_COMPACTION_API_KEY", "compact-secret-value")

    config = load_isolated(tmp_path, ApprovalMode.MANUAL, tmp_path)
    rendered = repr(config)
    redacted = redact_config_secrets(
        "main-secret-value / compact-secret-value",
        config,
    )

    assert "main-secret-value" not in rendered
    assert "compact-secret-value" not in rendered
    assert redacted == "[REDACTED] / [REDACTED]"
```

Update the existing missing-Key test to delete `DEEPSEEK_API_KEY`, `DEEPFIX_MAIN_API_KEY`, and `DEEPFIX_COMPACTION_API_KEY`, then call:

```python
load_config(
    tmp_path,
    ApprovalMode.MANUAL,
    env_file=tmp_path / "missing.env",
)
```

Update every other `load_config` call in this file to use `load_isolated(project, mode, tmp_path)` or an explicit temporary env file.

- [ ] **Step 3: Run the configuration tests and confirm the expected failures**

Run:

```powershell
pytest tests/test_config.py -q
```

Expected: FAIL because `ModelRoleConfig`, the two `AppConfig` fields, `env_file`, and `redact_config_secrets` do not exist and the current default is `deepseek-chat`.

- [ ] **Step 4: Add `python-dotenv` and implement the minimal configuration boundary**

Add to `pyproject.toml` runtime dependencies:

```toml
"python-dotenv>=1,<2",
```

Implement the following shape in `src/deepfix/config.py`:

```python
from urllib.parse import urlsplit

from dotenv import load_dotenv
from pydantic import SecretStr

_DEFAULT_ENV_FILE = Path(__file__).with_name(".env")
_USE_DEFAULT_ENV_FILE = object()


@dataclass(frozen=True)
class ModelRoleConfig:
    model_name: str
    api_key: SecretStr
    base_url: str


def _configured(name: str) -> str | None:
    value = os.environ.get(name)
    normalized = value.strip() if value is not None else ""
    return normalized or None


def _model_name(name: str, default: str) -> str:
    if name in os.environ and not _configured(name):
        raise ValueError(f"{name} 不能为空")
    return _configured(name) or default


def _base_url() -> str:
    if "DEEPSEEK_BASE_URL" in os.environ and not _configured("DEEPSEEK_BASE_URL"):
        raise ValueError("DEEPSEEK_BASE_URL 不能为空")
    value = _configured("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("DEEPSEEK_BASE_URL 必须是有效的 HTTPS URL")
    return value.rstrip("/")
```

At the start of `load_config`, resolve `env_file` to `_DEFAULT_ENV_FILE` when it is `_USE_DEFAULT_ENV_FILE`, then execute `load_dotenv(resolved_env_file, override=False, encoding="utf-8")` when the resolved value is not `None`. Resolve the two role Keys with the exact fallback order from Global Constraints, wrap them in `SecretStr`, and return them as `AppConfig.main_model` and `AppConfig.compaction_model`.

Implement redaction without logging or serializing the secret list:

```python
def redact_config_secrets(value: str, config: AppConfig) -> str:
    result = value
    for role in (config.main_model, config.compaction_model):
        secret = role.api_key.get_secret_value()
        if secret:
            result = result.replace(secret, "[REDACTED]")
    return result
```

- [ ] **Step 5: Migrate the direct `AppConfig` construction in the long-context fixture**

Import `ModelRoleConfig` and `SecretStr` in `tests/compaction/test_long_context_workflow.py`, then replace `model_name="deepseek-chat"` with:

```python
main_model=ModelRoleConfig(
    model_name="deepseek-v4-pro",
    api_key=SecretStr("offline-main"),
    base_url="https://api.deepseek.com",
),
compaction_model=ModelRoleConfig(
    model_name="deepseek-v4-flash",
    api_key=SecretStr("offline-compaction"),
    base_url="https://api.deepseek.com",
),
```

- [ ] **Step 6: Run focused tests and lint**

Run:

```powershell
pytest tests/test_config.py tests/compaction/test_long_context_workflow.py -q
ruff check src/deepfix/config.py tests/test_config.py tests/compaction/test_long_context_workflow.py
```

Expected: all tests PASS and Ruff exits 0.

- [ ] **Step 7: Commit the configuration boundary**

```powershell
git add pyproject.toml src/deepfix/config.py tests/test_config.py tests/compaction/test_long_context_workflow.py
git commit -m "feat: configure role-specific DeepSeek models"
```

### Task 2: Construct and wire distinct main and compaction model instances

**Files:**

- Modify: `src/deepfix/agent.py:48-146`
- Modify: `tests/test_agent.py`
- Modify: `tests/test_context.py:1-245`

**Interfaces:**

- Consumes: `AppConfig.main_model` and `AppConfig.compaction_model` from Task 1.
- Produces: `build_main_model(config: AppConfig) -> ChatDeepSeek`.
- Produces: `build_compaction_model(config: AppConfig) -> ChatDeepSeek`.
- Guarantees: `create_deep_agent(model=main_model)` and `CompactionCoordinator(model=compaction_model)`.
- Removes: `build_model(config)`.

- [ ] **Step 1: Replace the single-model test with failing role factory tests**

Update imports in `tests/test_agent.py` and add:

```python
from deepfix.agent import build_compaction_model, build_main_model


def test_role_models_use_independent_names_keys_and_shared_base_url(config):
    main = build_main_model(config)
    compaction = build_compaction_model(config)

    assert isinstance(main, ChatDeepSeek)
    assert isinstance(compaction, ChatDeepSeek)
    assert main is not compaction
    assert main.model_name == "deepseek-v4-pro"
    assert compaction.model_name == "deepseek-v4-flash"
    assert main.temperature == compaction.temperature == 0
    assert main.openai_api_base == compaction.openai_api_base == "https://api.deepseek.com"
```

Set distinct role Keys in the test fixture and assert the installed `ChatDeepSeek` fields explicitly:

```python
assert main.openai_api_key.get_secret_value() == "main-secret"
assert compaction.openai_api_key.get_secret_value() == "compact-secret"
```

Do not include either model object in assertion failure messages.

- [ ] **Step 2: Strengthen the Agent assembly test so role crossover fails**

Monkeypatch both factories with sentinel model objects and assert exact identity:

```python
main_model = object()
compaction_model = object()
monkeypatch.setattr("deepfix.agent.build_main_model", lambda config: main_model)
monkeypatch.setattr(
    "deepfix.agent.build_compaction_model",
    lambda config: compaction_model,
)

result = build_agent(
    config,
    checkpointer=InMemorySaver(),
    working_memory_store=WorkingMemoryStore(config.database_path),
    extensions=extensions,
)

assert captured["model"] is main_model
deepfix_middleware = next(
    item
    for item in captured["middleware"]
    if isinstance(item, DeepFixCompactionMiddleware)
)
assert deepfix_middleware.coordinator.model is compaction_model
assert deepfix_middleware.coordinator.model is not captured["model"]
assert registered["key"] == f"deepseek:{config.main_model.model_name}"
```

Keep the existing assertion that `save_progress` is present exactly once, and additionally assert the tool has no `model`, `main_model`, or `compaction_model` attribute.

- [ ] **Step 3: Run the Agent tests and confirm the import/identity failures**

Run:

```powershell
pytest tests/test_agent.py -q
```

Expected: FAIL because the role factories do not exist and the coordinator still receives the main model.

- [ ] **Step 4: Implement the two factories and strict dependency wiring**

Replace `build_model` in `src/deepfix/agent.py` with:

```python
def _build_deepseek_model(role: ModelRoleConfig) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=role.model_name,
        api_key=role.api_key,
        base_url=role.base_url,
        temperature=0,
    )


def build_main_model(config: AppConfig) -> ChatDeepSeek:
    return _build_deepseek_model(config.main_model)


def build_compaction_model(config: AppConfig) -> ChatDeepSeek:
    return _build_deepseek_model(config.compaction_model)
```

Inside `build_agent`, bind both once:

```python
main_model = build_main_model(config)
compaction_model = build_compaction_model(config)
```

Use `config.main_model.model_name` for `register_harness_profile`, pass `compaction_model` to `CompactionCoordinator`, and pass `main_model` to `create_deep_agent`. Do not pass either model into `build_save_progress_tool`.

- [ ] **Step 5: Migrate context tests to the explicit main model factory**

In `tests/test_context.py`, replace imports and both `build_model(config)` calls with `build_main_model(config)`. Do not introduce a compaction model into context rendering tests because `render_working_memory` and its Tool are deterministic.

- [ ] **Step 6: Run focused tests and lint**

Run:

```powershell
pytest tests/test_agent.py tests/test_context.py -q
ruff check src/deepfix/agent.py tests/test_agent.py tests/test_context.py
```

Expected: all tests PASS and Ruff exits 0.

- [ ] **Step 7: Commit the role wiring**

```powershell
git add src/deepfix/agent.py tests/test_agent.py tests/test_context.py
git commit -m "feat: isolate repair and compaction models"
```

### Task 3: Add DeepSeek V4 budget fallbacks and preserve compaction failure semantics

**Files:**

- Modify: `src/deepfix/compaction/budget.py:12-17`
- Modify: `tests/compaction/test_budget.py`
- Modify: `tests/compaction/test_coordinator.py`

**Interfaces:**

- Consumes: role-specific `ChatDeepSeek` instances from Task 2.
- Produces: explicit `1_000_000` input limits for `deepseek-v4-flash` and `deepseek-v4-pro` when no SDK profile exists.
- Preserves: runtime compaction-model failures become `delta_generation_failed`; no main-model fallback occurs.

- [ ] **Step 1: Add failing V4 fallback budget tests**

In `tests/compaction/test_budget.py`, use a model stub without a profile:

```python
@pytest.mark.parametrize("model_name", ["deepseek-v4-flash", "deepseek-v4-pro"])
def test_v4_models_have_explicit_one_million_token_fallback(model_name):
    monitor = ContextBudgetMonitor(
        token_counter=lambda value: 820_000,
        output_reserve_tokens=0,
    )
    request = SimpleNamespace(
        model=SimpleNamespace(model_name=model_name, profile=None),
        system_message=None,
        messages=[HumanMessage(content="large")],
        tools=[],
    )

    report = monitor.measure(request, [])

    assert report.max_input_tokens == 1_000_000
    assert report.usage_ratio == pytest.approx(0.82)
    assert report.zone == "observe"
```

Keep the existing test proving an unknown model without a profile raises `ContextBudgetConfigurationError`.

- [ ] **Step 2: Add a coordinator regression proving no runtime model fallback**

Add a recording main-model sentinel and a failing compaction Delta generator in `tests/compaction/test_coordinator.py`:

```python
class _FailingCompactionDelta:
    def __init__(self):
        self.models = []

    def generate(self, model, units):
        self.models.append(model)
        raise RuntimeError("compaction model unavailable")


def test_delta_failure_never_falls_back_to_main_model(tmp_path):
    compaction_model = object()
    main_model = object()
    delta = _FailingCompactionDelta()
    coordinator = CompactionCoordinator(
        adapter=DeepAgentsArtifactAdapter(
            FilesystemBackend(root_dir=tmp_path / "artifacts", virtual_mode=True)
        ),
        delta_generator=delta,
        snapshot_builder=CompactionSnapshotBuilder(),
        snapshot_store=CompactionStore(tmp_path / "deepfix.sqlite3"),
        memory_store=WorkingMemoryStore(tmp_path / "deepfix.sqlite3"),
        model=compaction_model,
    )
    request = replace(_request(ratio=0.85), model=compaction_model)
    handler_calls = []

    coordinator.invoke_automatic(
        request,
        lambda messages: handler_calls.append(messages)
        or ModelResponse(result=[AIMessage(content="continued")]),
    )

    assert delta.models == [compaction_model]
    assert main_model not in delta.models
    assert len(handler_calls) == 1
    assert coordinator.snapshot_store.list_failures("task-a")[0].error_code == "delta_generation_failed"
```

- [ ] **Step 3: Run focused tests and confirm the V4 failures**

Run:

```powershell
pytest tests/compaction/test_budget.py tests/compaction/test_coordinator.py -q
```

Expected: the V4 cases FAIL with `ContextBudgetConfigurationError`; the existing coordinator behavior may already pass and remains a regression guard.

- [ ] **Step 4: Add the V4 limits without weakening unknown-model validation**

Update `_MODEL_INPUT_LIMITS` in `src/deepfix/compaction/budget.py`:

```python
_MODEL_INPUT_LIMITS = {
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4-pro": 1_000_000,
}
```

Remove the retired `deepseek-chat` and `deepseek-reasoner` fallback entries. Models with a valid SDK profile remain supported because `_max_input_tokens` checks the profile before this table.

- [ ] **Step 5: Run focused tests and lint**

Run:

```powershell
pytest tests/compaction/test_budget.py tests/compaction/test_coordinator.py -q
ruff check src/deepfix/compaction/budget.py tests/compaction/test_budget.py tests/compaction/test_coordinator.py
```

Expected: all tests PASS and Ruff exits 0.

- [ ] **Step 6: Commit the budget compatibility change**

```powershell
git add src/deepfix/compaction/budget.py tests/compaction/test_budget.py tests/compaction/test_coordinator.py
git commit -m "feat: budget DeepSeek V4 model roles"
```

### Task 4: Enforce secret non-leakage, document setup, and run the full gate

**Files:**

- Modify: `src/deepfix/service.py:64-104`
- Modify: `tests/test_service.py`
- Modify: `tests/test_backend.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md:5-30,66-110,119-148`

**Interfaces:**

- Consumes: `redact_config_secrets(value, config)` from Task 1.
- Guarantees: persisted Agent-boundary errors cannot contain either configured role Key.
- Documents: package-local `.env`, one-Key/two-Key modes, role responsibilities, and cost/quality model combinations.

- [ ] **Step 1: Add failing Service redaction and Shell isolation tests**

In `tests/test_service.py`, configure distinct role secrets and make the Agent raise an exception containing both:

```python
def test_agent_failure_redacts_both_model_role_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPFIX_MAIN_API_KEY", "main-secret-value")
    monkeypatch.setenv("DEEPFIX_COMPACTION_API_KEY", "compact-secret-value")
    config = load_config(
        tmp_path,
        ApprovalMode.MANUAL,
        env_file=tmp_path / "missing.env",
    )
    main_secret = config.main_model.api_key.get_secret_value()
    compaction_secret = config.compaction_model.api_key.get_secret_value()
    service, repository = make_service(
        config,
        FakeAgent(
            RuntimeError(f"auth failed {main_secret} {compaction_secret}")
        ),
    )

    task = service.start("修复错误")
    persisted = repository.get(task.task_id)

    assert task.status is TaskStatus.FAILED
    assert main_secret not in persisted.final_summary
    assert compaction_secret not in persisted.final_summary
    assert persisted.final_summary == "Agent 执行失败: auth failed [REDACTED] [REDACTED]"
```

Adapt the helper to use a config fixture with distinct role Keys.

In `tests/test_backend.py`, extend the current environment leak test:

```python
for name, secret in {
    "DEEPSEEK_API_KEY": "common-secret",
    "DEEPFIX_MAIN_API_KEY": "main-secret",
    "DEEPFIX_COMPACTION_API_KEY": "compact-secret",
}.items():
    monkeypatch.setenv(name, secret)

result = backend.execute(
    "python -c \"import os; "
    "print(os.getenv('DEEPSEEK_API_KEY')); "
    "print(os.getenv('DEEPFIX_MAIN_API_KEY')); "
    "print(os.getenv('DEEPFIX_COMPACTION_API_KEY'))\""
)

assert result.exit_code == 0
assert "common-secret" not in result.output
assert "main-secret" not in result.output
assert "compact-secret" not in result.output
assert result.output.count("None") == 3
```

- [ ] **Step 2: Add a failing CLI test for package-relative `.env` loading**

Patch the config module's `_DEFAULT_ENV_FILE` to a temporary file, change the current directory to an unrelated target project, and invoke `deepfix new` with model construction patched out:

```python
def test_cli_loads_package_env_independent_of_current_directory(tmp_path, monkeypatch):
    env_file = tmp_path / "package" / ".env"
    env_file.parent.mkdir()
    env_file.write_text("DEEPSEEK_API_KEY=file-secret\n", encoding="utf-8")
    project = tmp_path / "target"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setattr("deepfix.config._DEFAULT_ENV_FILE", env_file)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    config = load_config(project, ApprovalMode.MANUAL)

    assert config.main_model.api_key.get_secret_value() == "file-secret"
    assert config.compaction_model.api_key.get_secret_value() == "file-secret"
```

The assertion must inspect `SecretStr` only in the test and must not print the config.

- [ ] **Step 3: Run the security tests and confirm Service redaction fails**

Run:

```powershell
pytest tests/test_service.py tests/test_backend.py tests/test_cli.py -q
```

Expected: FAIL because `BugfixService` currently persists `str(exc)` without role-Secret redaction. The Shell isolation test may already pass because the Backend uses an allowlist; retain it as a regression guard.

- [ ] **Step 4: Sanitize Agent-boundary exceptions before persistence**

Import `redact_config_secrets` in `src/deepfix/service.py` and change only the general Agent failure boundary:

```python
except Exception as exc:  # noqa: BLE001 - persist every Agent boundary failure
    safe_error = redact_config_secrets(str(exc), self.config)
    task.final_summary = f"Agent 执行失败: {safe_error}"
    task.transition_to(TaskStatus.FAILED)
    self._save(task)
    return task
```

Do not alter `ContextCoordinationError` handling: its typed `error_code` and recovery metadata already contain no credentials.

- [ ] **Step 5: Update README with exact configuration and responsibility examples**

Replace the single-Key startup block with:

```dotenv
# src/deepfix/.env
DEEPSEEK_API_KEY=sk-your-key
DEEPFIX_MAIN_MODEL=deepseek-v4-pro
DEEPFIX_MAIN_API_KEY=
DEEPFIX_COMPACTION_MODEL=deepseek-v4-flash
DEEPFIX_COMPACTION_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

Document these exact statements:

- one common Key is sufficient; role Keys override it using the specified fallback chains;
- `deepseek-v4-flash` for both roles is the cost-first setup;
- Pro for main plus Flash for compaction is the quality/cost balanced setup;
- `save_progress` is called by the main Agent and does not invoke another model;
- the compaction model only produces `CompactionDelta`, has no tools, and cannot overwrite deterministic evidence;
- process environment variables override `src/deepfix/.env`;
- `.env` is ignored by Git and model Keys are excluded from the target Shell environment.

- [ ] **Step 6: Run all focused tests and dependency-boundary scans**

Run:

```powershell
pytest tests/test_config.py tests/test_agent.py tests/test_context.py tests/compaction/test_budget.py tests/compaction/test_coordinator.py tests/test_backend.py tests/test_service.py tests/test_cli.py tests/compaction/test_long_context_workflow.py -q
ruff check src tests
rg -n "build_model|config\.model_name|deepseek-chat|deepseek-reasoner" src tests README.md
rg -n "DEEPFIX_MAIN_API_KEY|DEEPFIX_COMPACTION_API_KEY|DEEPSEEK_API_KEY" src/deepfix
```

Expected:

- focused tests PASS;
- Ruff exits 0;
- the retired factory/property/model-name scan returns no runtime references (historical migration fixtures may be assessed individually rather than mechanically changed);
- Key-name references occur only in configuration parsing/redaction and intentional Backend security boundaries, never in prompts, Graph state, persistence models, reports, or Artifacts.

- [ ] **Step 7: Run the final verification gate**

Run as separate commands so a later command cannot mask a pytest failure:

```powershell
pytest -q
ruff check src tests
git diff --check
git status --short
```

Expected: all offline tests PASS, online-marked tests remain deselected, Ruff and `git diff --check` exit 0, and status lists only the intended Task 4 source/test/doc changes.

- [ ] **Step 8: Commit the security and documentation boundary**

```powershell
git add src/deepfix/service.py tests/test_service.py tests/test_backend.py tests/test_cli.py README.md
git commit -m "docs: secure role-specific model setup"
```

---

## Final Review Checklist

- `src/deepfix/.env` is loaded by absolute package-relative path with `override=False`.
- Tests use missing or temporary env files and never read the developer's real `.env`.
- `AppConfig` has exactly two `ModelRoleConfig` fields and no ambiguous `model_name` property.
- Main and compaction role Keys follow the approved fallback chains, with blank values treated as missing.
- Both credentials are `SecretStr` and disappear from repr, persisted failures, reports, Artifacts, and Shell output.
- `create_deep_agent` receives only the main model instance.
- `CompactionCoordinator` and `CompactionDeltaGenerator` receive only the compaction model instance.
- `save_progress` remains main-Agent-driven and has no model dependency.
- DeepSeek V4 model names work with explicit 1M-token budget fallback when SDK profile data is absent.
- A compaction-model runtime failure never calls the main model as a fallback.
- Existing evidence authority, transactional compaction, PAUSED recovery, approval, research, and real pytest completion invariants remain green.
