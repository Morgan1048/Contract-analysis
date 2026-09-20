# Connector and MCP design

## Implementation boundary

MCP stdio is executable. The REST paths and HTTP error codes below are an OpenAPI
design, not an implemented HTTP server. Hosted tenant authorization is proposed,
not enabled in this local prototype. Remote processing requires explicit
`allow_remote` authorization for non-confidential material.

Contract v2 aligns the MCP response with the UI download. This deliberately
replaces the old simplified `questions[].state/findings` response; clients should
read the assignment fields under `questions`, atomic findings under `check_results`,
and unresolved questions under `review_required`.

The graph and chat share `search_evidence`, `read_context`, `read_table`,
`resolve_reference` and `find_modifiers`. `get_audit_trail` returns locally stored
node/tool events. Analysis accepts an optional ISO `assessment_date`; incomplete
results carry review status rather than a fourth public compliance classification.

## Decisions

### Authentication

The local MCP deployment trusts the OS user launching the stdio process; it is not
a network listener and offers no multi-tenant authorization. The UI listens on
localhost. Contracts stay in the local process/SQLite store. A hosted
deployment changes this boundary. The connector terminates TLS and requires a per-tenant API
key (or an equivalent service identity) with scopes such as `runs:write`, `runs:read`,
`contracts:search`, and `chat:write`. Tenant identity is attached to every `run_id` and
`session_id`, and authorization is checked before either identifier is read. This is necessary
because contract content is confidential and a hosted connector runs on infrastructure outside
the user's process.

The proposed OpenAPI contract carries the key in `X-API-Key` and requires it
globally. Each operation declares its permission in `x-required-scopes`; the
hosted connector must enforce these tenant-key permissions. OpenAPI API-key
security requirements use an empty scope array (native scopes are OAuth2/OpenID
Connect only), so these permissions are specified with an extension instead.

### Request and response contracts

In the proposed REST design, `POST /analyze` accepts a PDF and returns
`202 Accepted` with a `run_id` and `status: running`. `GET /runs/{run_id}` returns the assignment
output schema after completion. The result is fetched by identifier rather than returned inline
because parsing, retrieval, judging, and evaluation are slow and the evidence payload can be
large; holding a chat connector request open for minutes is unreliable. The same contract is
exposed through MCP as `analyze_contract` followed by `get_compliance_result`.

MCP `analyze_contract` accepts exactly one of `pdf_path` or base64 `pdf_bytes`,
plus optional `allow_remote` (default false) and `assessment_date`. The submitted
ID is passed into the shared pipeline; results, graph audit, history and traces
all use that same `run_id`. Polling uses that ID even after a completed run's
server restarts. No in-memory alias is needed.
UI and MCP both honor `CA_RUN_DB` (default `data/runs.db`). Keep the stdio process
open until jobs finish; a one-shot `echo` call closes stdin and cannot host a
long-running asynchronous analysis.

Completed and incomplete results use `core.output.to_output_document`, exactly
as the UI download does. Public compliance states remain the three assignment
values; incomplete questions are separate review items, not `Indeterminate`
compliance classifications. Verified citation labels use saved source metadata
when the live index is unavailable. The optional MCP `question_id` filter narrows
questions/checks/review items, but `status` still describes the whole run.

This is an MCP design constraint, not merely an implementation convenience: an MCP tool call
must return promptly enough for a client to remain responsive, so long-running analysis returns
a `run_id` immediately and the client polls the result tool.

During a provider wait, MCP `get_compliance_result` retains `status: "running"`
and adds `wait`: `provider`, `model`, `reason`, `remaining_s`, `retry`,
`max_retries`, `http_status`, and `phase` (`waiting` or `blocked`). The countdown
updates approximately every five seconds; the object clears on resume or terminal
status. `retry: 0` denotes pacing or a shared cooldown before a new request.
Active jobs/cooldowns are in-memory; completed results remain in SQLite.

### State management

`run_id` scopes one uploaded document and analysis result. `session_id` scopes one
UI conversation. Saved results, evidence snapshots and audit events remain in SQLite.
Active MCP jobs, wait countdowns, failed-job statuses and source indexes remain
in memory; a server restart does NOT resume jobs or restore unsaved failures.
Completed/incomplete saved results remain queryable by their original ID.

Every MCP source tool requires an explicit `run_id`: `search_evidence`,
`read_context`, `read_table`, `resolve_reference`, `find_modifiers`,
`search_contract`, `get_section`, `get_requirement`, and `list_deviations`.
Each resolves that run's index, never the latest completed document. After a
restart, or for a run created by the UI, MCP still serves saved results/audit,
but source retrieval returns `SOURCE_UNAVAILABLE` with `source_required: true`.
Re-uploading starts a new run; no silent reparse, embedding call or partial-evidence
snapshot masquerades as a complete searchable source. UI chat/graph tools remain
bound to their existing single-document index and do not gain an extra argument.

### Multi-turn interaction

The UI chat agent persists and reloads history from `chat_messages`. A proposed REST connector sends
`session_id`, `run_id`, and the new message; the backend supplies prior history to the agent.
The client therefore does not replay the transcript on every call, reducing payload size and
avoiding divergent client-side histories. Open chat uses model-driven selection of the shared
tool registry, while fixed compliance analysis uses a bounded LangGraph Router/Judge/Verifier/Critic
loop over the five questions and 25 fixed checks.
MCP clients maintain their own conversation and call these tools with explicit
run identifiers; the MCP server does not currently expose a separate `chat` tool.

### Failure modes

Failures are typed and distinguishable:

| Condition | Client-visible response |
|---|---|
| Invalid MCP intake (neither/both inputs) | `INVALID_ARGUMENTS`, `isError: true`; no job created. |
| Unreadable path, malformed PDF/base64 or unexpected worker failure | Poll returns `status: failed`, `ANALYSIS_FAILED`, `isError: true`; failure logged with run ID. |
| Backend failure escaping the pipeline | Poll returns `status: failed`, `BACKEND_UNAVAILABLE`, `isError: true`. |
| Pipeline completes with unresolved checks | Public `status: incomplete` plus `review_required`; not a tool execution error or compliance classification. |
| Run still processing or waiting | `{run_id, status: "running", wait?: ...}`; `isError: false`. |
| Source queried before analysis finishes | `RUN_NOT_READY`; no fallback to another document. |
| Unknown run/question | `RUN_NOT_FOUND` / `QUESTION_NOT_FOUND`, `isError: true`. |
| Saved run without live source | `SOURCE_UNAVAILABLE`, `source_required: true`, `isError: true`; saved result/audit remain readable. |

These rows describe executable MCP behavior. REST HTTP status codes in
`openapi.yaml` remain a proposed mapping, not evidence of a running API at port 8000.

No failure is converted into an empty successful answer. JSON logs retain the trace and span
identifiers across submission, parsing, routing, and evaluation without writing protocol data
to stdout.
