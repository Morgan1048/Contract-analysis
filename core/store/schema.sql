-- The `demo` flag marks rows written by `eval/seed_demo.py` (never by the real
-- pipeline, which always inserts 0). It exists as a second, row-level signal in
-- addition to demo data living in a physically separate database file
-- (`data/demo.db`), so a demo row can never be mistaken for a real measurement
-- even if a database file were ever inspected or merged out of context.
CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, source_path TEXT, started_at TEXT, finished_at TEXT, model TEXT, backend TEXT, total_prompt_tokens INTEGER, total_completion_tokens INTEGER, cost_usd REAL, duration_ms REAL, status TEXT, unpriced_model INTEGER DEFAULT 0, chunk_count INTEGER DEFAULT 0, table_row_chunks INTEGER DEFAULT 0, demo INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS question_results (run_id TEXT, question_id TEXT, state TEXT, confidence REAL, rationale TEXT, applied_deviations TEXT, demo INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (run_id, question_id));
CREATE TABLE IF NOT EXISTS findings (run_id TEXT, question_id TEXT, sub_requirement_id TEXT, verdict TEXT, confidence REAL, quote TEXT, source_chunk_id TEXT, quote_verified INTEGER, retry_count INTEGER, rationale TEXT, prompt_tokens INTEGER NOT NULL DEFAULT 0, completion_tokens INTEGER NOT NULL DEFAULT 0, cost_usd REAL, demo INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (run_id, sub_requirement_id));
CREATE TABLE IF NOT EXISTS checks (run_id TEXT, sub_requirement_id TEXT, name TEXT, passed INTEGER, detail TEXT, demo INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS stage_timings (run_id TEXT, stage TEXT, duration_ms REAL, tokens INTEGER, cost_usd REAL, demo INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS eval_results (eval_id TEXT PRIMARY KEY, run_id TEXT, question_id TEXT, expected_state TEXT, actual_state TEXT, correct INTEGER, created_at TEXT, demo INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS chat_messages (session_id TEXT, run_id TEXT, seq INTEGER, role TEXT, content TEXT, tool_calls TEXT, citations TEXT, created_at TEXT);
-- Presence of a row here is a belt-and-suspenders demo marker independent of any
-- particular table's `demo` column, so `RunStore.is_demo_store()` still works
-- even against a demo database that (in some future extension) stores rows
-- through a path that forgets to set `demo=1`.
CREATE TABLE IF NOT EXISTS demo_meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS run_snapshots (run_id TEXT PRIMARY KEY, schema_version INTEGER NOT NULL, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit_events (run_id TEXT NOT NULL, sequence INTEGER NOT NULL, timestamp TEXT NOT NULL, node TEXT NOT NULL, question_id TEXT, payload TEXT NOT NULL, PRIMARY KEY (run_id, sequence));
