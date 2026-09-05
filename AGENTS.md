# Repository workflow

- The primary remote is `origin`: https://github.com/ChenXLab7/DeepFix.git.
- After completing code updates and appropriate verification, commit the task's changes and push the current working branch to `origin` automatically. The user has authorized this workflow; no repeated confirmation is required.
- If verification or pushing fails, report the failure accurately. Do not force-push or overwrite remote history.
- Keep unrelated user changes out of task commits. In particular, preserve the existing uncommitted changes in `src/deepfix/debug.py` unless the user explicitly includes them in the task.
- Keep `gitee` as a secondary remote; routine pushes target GitHub.
