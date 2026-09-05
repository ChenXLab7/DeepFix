# Repository workflow

- The primary remote is `origin`: https://github.com/ChenXLab7/DeepFix.git.
- After completing code updates and appropriate verification, commit the task's changes, integrate them into `main`, verify the integrated result, and push `main` to `origin` automatically. The user has authorized this workflow; no repeated confirmation is required.
- Preserve commit history and report the published commit hash for each update. Use commits and optional version tags to retrieve previous versions; prefer a new revert commit when restoring an older version on shared `main`.
- If verification or pushing fails, report the failure accurately. Do not force-push or overwrite remote history.
- Keep unrelated user changes out of task commits. In particular, preserve the existing uncommitted changes in `src/deepfix/debug.py` unless the user explicitly includes them in the task.
- Keep `gitee` as a secondary remote; routine pushes target GitHub.
