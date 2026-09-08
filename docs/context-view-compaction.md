# Context view optimization

Base: `3f986df`. This change reuses the existing Compaction middleware, Artifact
adapter, operation-result storage and diagnostic retrieval. It does not introduce
a Context Runtime, Message Store, HistoryRepository schema or debugging controller.

## Production behavior

1. Oversized completed Shell output is saved through existing operation-result
   Artifact storage before producing a head/tail preview and full-output reference.
   The preview retains a nonzero exit status. If saving fails, the real full result
   is returned rather than silently discarded. Native DeepAgents per-tool controls
   continue to apply; they are not duplicated.
2. Above 50 messages, older complete WorkUnits outside the final approximately
   40 messages are omitted from the model view after archival. Human/System units,
   incomplete/ambiguous units, the last three consumed tool rounds and unseen tool
   results remain intact. These are soft view limits, not limits on user content.
3. After snip, if serialized message-view size exceeds 50,000 characters, Micro
   shortens eligible old ToolMessage bodies above 2,000 characters to a 500-character
   head and 500-character tail plus archive reference, stopping at 40,000 characters
   or when no eligible results remain. Size excludes artifact/response_metadata;
   it includes message content, tool calls and reasoning. It is independent of the
   model-window ratio. The same recent/unseen protections apply. All operations
   affect the request view only; no persistent consumed-result state is added.
4. The request is measured again. Existing semantic compaction is used only if
   pressure still requires it. Empty selections do not invoke the summary model.

**Snip is request-local. It never deletes, replaces or annotates original Checkpoint
messages, and does not rewrite Receipt or Evidence.** A resumed process reconstructs
the view from the original history; no new persistent snip state is needed.

## Reused mechanisms and corrections

- Existing conversation artifacts now include hash-checked serialized message
  objects alongside readable XML. `restore_history(path, attempt_id)` restores
  tool arguments/results, IDs, artifact and additional fields. Existing old XML-only
  artifacts are still readable; they are not silently described as lossless objects.
- Snip archive attempts are content-addressed per WorkUnit, so reapplying the same
  view does not append the same history again. Persistence failure preserves the
  original view. The diagnostic collector can find the canonical task history even
  before a semantic Snapshot exists.
- Semantic compaction receives the actual selected messages, not merely WorkUnit
  IDs and categories. Existing snapshot/provenance validation remains in place.
- Approximate budgeting now includes tool arguments and reasoning text. Retention
  uses per-message estimated sizes, with allowance for the rest of the request;
  no tokenizer service or budget subsystem was added. Human/System messages are
  protected in semantic retention as well as snip.

## Verification and scope

Focused regressions cover message-object roundtrip, actual semantic input, original
checkpoint preservation, whole tool pairs, archive failure, idempotent reconstruction,
no-summary paths, and full large Shell output including failure status. A real
SQLite Checkpoint is closed and reopened: all 222 source messages survive and a new
user input extends the same history to 223, while both model views remain reduced.

This controls model input, not database/disk growth. It does not prune Checkpoints,
change process output buffering/timeouts, or expand diagnostic search's existing
10 MiB artifact limit. Larger history files remain accessible through paginated
`read_file`; the view points to that fallback. No paid model run or benchmark is
required to establish these deterministic boundaries, and none was performed here.
