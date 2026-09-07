# Verification matching and controlled execution — 2026-09-07

Base revision: `356e50d`. Scope: command matching, test scope, and execution approval.
No investigation controls or lifecycle changes were added.

## Changes

- Required Oracle extraction retains the full configured interpreter/wrapper invocation.
  Policy and Evidence use the same argument-aware, case-preserving command identity.
  Windows uses native argv parsing; Linux uses POSIX shell argument parsing. Different
  interpreters, targets, selectors, or argument grouping cannot silently match.
- A whole test file is module scope, not targeted scope. A same-code-state module
  failure continues to block FIXED even when the required node passes.
- Workspace command preflight is authoritative before approval. Configured absolute
  Python/wrapper commands follow the same routine approval rules as `python`:
  guarded mode permits routine tests; manual mode still asks. Hard denials cannot
  be approved through the service.
- A small subset of literal Python diagnostic expressions can run without approval.
  It uses direct argv with `-I -S`, excludes imports, attributes, file/process/network
  access and resource-amplifying operators, and limits expression/AST size.
  This is not a sandbox for arbitrary Python or project-import reproduction scripts.
  Those remain outside this approval-free subset; approved pytest execution remains
  available. Existing timeout, workspace, environment and dependency restrictions remain.

## Actual PySnooper retest

Container: `deepfix-bugsinpy`. Task: `271fa4d0cce14a17a133a945efcbd0db`.
The relevant container source files matched base revision `356e50d` after newline
normalization, including verification, collector, execution, backend and service.

Original Required Oracle omitted `/home/python-venv-wrapper -m`, while actual passing
Evidence retained it and was classified `repository_existing`. Both command matching
and user-specified origin were therefore wrong.

Tests were rerun against the existing isolated repaired Task Workspace with candidate
code, a reconstructed policy and a separate diagnostic Evidence database. The original
Task database was opened read-only. No previous receipts or facts were rewritten.

| Actual command scope | Result |
| --- | --- |
| `test_custom_repr_single` | 1 passed; `user_specified`, post-change; required oracle accepted |
| Previous five-node related command | 4 passed, 1 failed |
| Entire `tests/test_pysnooper.py` | 30 passed, 1 failed |

The remaining failure is `test_disable`: `pysnooper.tracer` lacks `DISABLED`.
The corrected evaluation reports `all_required_satisfied=true`, `fixed_allowed=false`,
with the module failure as conflicting evidence. This is a verification diagnostic,
not a new autonomous Agent run or a persisted Task completion.

Before/after code hash:
`1cf67a0222f657f0d6a1539a83a963a474ac041908fa4a9b147587aed3c0c3ef`.

Container artifacts: `/home/deepfix-state/verification-retest-20260907-114956/`.
`report.json` contains the old policy, old tests, reconstructed policy, fresh Evidence
and evaluation; `retest-*-toolmessage.json` and `retest-*.txt` contain actual results.

## Regression and remaining boundaries

- Windows: 138 passed, 1 skipped across verification, execution, backend, approval
  and service integration tests. Changed files pass Ruff.
- Linux: 73 passed, 1 skipped, 1 existing platform-specific failure: the test expects
  `rg secret C:/outside` to be outside the Linux workspace. The same failure was
  reproduced against the original container source. It is not changed here.
- Review covered startup-code bypass, resource amplification and Windows path
  collisions; the identified issues have regression coverage.
- Existing Tasks retain their persisted policies and Evidence. This change fixes
  newly constructed policies and newly collected evidence; it does not migrate the
  malformed policy of the old Task. Use a fresh Task to exercise the complete new
  flow. Its unrelated module failure still needs investigation before claiming FIXED.
- The repair remains in the Task Workspace. The source PySnooper project is not
  automatically updated, and these results do not establish full repository regression.
