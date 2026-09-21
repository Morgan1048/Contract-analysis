from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path
from datetime import datetime, timezone

from core.agents.evaluator import Check
from core.pipeline import AnalysisResult
from core.rubric.schema import Finding, QuestionResult
from core.rubric.aggregate import load_rubric


class RunStore:
    def __init__(self, path="data/runs.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connections = threading.local()
        self._initialization_lock = threading.Lock()
        self._schema_initialized = False

    def _connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    @property
    def db(self):
        connection = getattr(self._connections, "connection", None)
        if connection is not None:
            return connection

        if not self._schema_initialized:
            with self._initialization_lock:
                if not self._schema_initialized:
                    connection = self._connect()
                    connection.executescript(
                        Path(__file__).with_name("schema.sql").read_text()
                    )
                    self._migrate(connection)
                    connection.commit()
                    self._schema_initialized = True

        if connection is None:
            connection = self._connect()
        self._connections.connection = connection
        return connection

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        """Apply additive migrations so databases created by older releases open."""
        migrations: dict[str, dict[str, str]] = {
            "findings": {
                "prompt_tokens": "INTEGER NOT NULL DEFAULT 0",
                "completion_tokens": "INTEGER NOT NULL DEFAULT 0",
                "cost_usd": "REAL",
                "demo": "INTEGER NOT NULL DEFAULT 0",
            },
            # `demo` arrived after these tables shipped. Without it, the
            # positional INSERTs in save_run fail against a pre-existing
            # database: "table runs has 14 columns but 15 values were supplied".
            "runs": {"demo": "INTEGER NOT NULL DEFAULT 0"},
            "question_results": {"demo": "INTEGER NOT NULL DEFAULT 0"},
            "checks": {"demo": "INTEGER NOT NULL DEFAULT 0"},
            "stage_timings": {"demo": "INTEGER NOT NULL DEFAULT 0"},
            "eval_results": {"demo": "INTEGER NOT NULL DEFAULT 0"},
        }
        for table, columns in migrations.items():
            existing = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
            if not existing:
                continue
            for name, definition in columns.items():
                if name not in existing:
                    connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    def save_run(self, result: AnalysisResult, *, demo: bool = False):
        """Persist one analysis run.

        ``demo`` must stay ``False`` for every real pipeline call; it exists
        only so ``eval/seed_demo.py`` can stamp its synthetic rows and never
        be confused with real measurements (see ``core/store/schema.sql``).
        """
        demo_flag = int(bool(demo))
        usage = result.usage
        started = datetime.fromisoformat(result.started_at)
        finished = datetime.fromisoformat(result.finished_at)
        duration = (finished - started).total_seconds() * 1000
        try:
            with self.db:
                self.db.execute("INSERT INTO run_snapshots VALUES (?,?,?)",
                                (result.run_id, 2, json.dumps(asdict(result))))
                self.db.executemany(
                    "INSERT INTO audit_events VALUES (?,?,?,?,?,?)",
                    [(result.run_id, seq, event["timestamp"], event["node"], event["question_id"],
                      json.dumps(event)) for seq, event in enumerate(result.audit_events)],
                )
                self.db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (result.run_id, result.source_path, result.started_at, result.finished_at, usage.get("model", "local"), usage.get("backend", "local"), usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), usage.get("cost_usd", 0), duration, "incomplete" if getattr(result, "incomplete", False) else "completed", int(bool(usage.get("unpriced_model", False))), usage.get("chunk_count", 0), usage.get("table_row_chunks", 0), demo_flag))
                for q in result.questions:
                    self.db.execute("INSERT INTO question_results VALUES (?,?,?,?,?,?,?)", (result.run_id, q.question_id, q.state, q.confidence, q.rationale, json.dumps(q.applied_deviations), demo_flag))
                    for f in q.findings:
                        self.db.execute(
                            "INSERT INTO findings "
                            "(run_id,question_id,sub_requirement_id,verdict,confidence,quote,"
                            "source_chunk_id,quote_verified,retry_count,rationale,prompt_tokens,"
                            "completion_tokens,cost_usd,demo) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (
                                result.run_id, q.question_id, f.sub_requirement_id, f.verdict,
                                f.confidence, f.quote, f.source_chunk_id, int(f.quote_verified),
                                f.retry_count, f.rationale, f.prompt_tokens,
                                f.completion_tokens, f.cost_usd, demo_flag,
                            ),
                        )
                for sub_id, checks in result.checks.items():
                    for check in checks:
                        self.db.execute("INSERT INTO checks VALUES (?,?,?,?,?,?)", (result.run_id, sub_id, check.name, int(check.passed), check.detail, demo_flag))
                for stage, duration_ms in usage.get("latency_ms", {}).items():
                    stage_usage = usage.get("stage_usage", {}).get(stage, {})
                    self.db.execute(
                        "INSERT INTO stage_timings VALUES (?,?,?,?,?,?)",
                        (
                            result.run_id, stage, duration_ms,
                            stage_usage.get("tokens", 0),
                            stage_usage.get("cost_usd", 0.0),
                            demo_flag,
                        ),
                    )
        except Exception:
            raise

    def is_demo_store(self) -> bool:
        """True when this store contains data seeded by ``eval/seed_demo.py``.

        Checked by row content (not by file path), so the UI banner stays
        correct even if a demo database is renamed or copied.
        """
        try:
            demo_runs = self.db.execute("SELECT COUNT(*) AS n FROM runs WHERE demo=1").fetchone()["n"]
            if demo_runs:
                return True
            marker = self.db.execute(
                "SELECT COUNT(*) AS n FROM demo_meta WHERE key='seeded_by'"
            ).fetchone()["n"]
            return bool(marker)
        except sqlite3.Error:
            return False

    def get_run(self, run_id):
        row = self.db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            return None
        saved = self.db.execute("SELECT payload FROM run_snapshots WHERE run_id=?", (run_id,)).fetchone()
        if saved is not None:
            payload = json.loads(saved["payload"])
            payload["questions"] = [
                QuestionResult(**dict(q, findings=[Finding(**f) for f in q["findings"]],
                                      relevant_quotes=[Finding(**f) for f in q.get("relevant_quotes", [])]))
                for q in payload["questions"]
            ]
            payload["checks"] = {key: [Check(**c) for c in value] for key, value in payload["checks"].items()}
            return AnalysisResult(**payload)
        qs = self.db.execute("SELECT * FROM question_results WHERE run_id=? ORDER BY question_id", (run_id,)).fetchall()
        rubric = load_rubric()
        checks = {}
        for c in self.db.execute("SELECT * FROM checks WHERE run_id=?", (run_id,)).fetchall():
            checks.setdefault(c["sub_requirement_id"], []).append(
                Check(c["name"], bool(c["passed"]), c["detail"])
            )
        questions = []
        for q in qs:
            fs = self.db.execute("SELECT * FROM findings WHERE run_id=? AND question_id=?", (run_id, q["question_id"])).fetchall()
            questions.append(QuestionResult(q["question_id"], rubric.get(q["question_id"]).text if q["question_id"] in rubric else q["question_id"], q["state"], q["confidence"], [Finding(f["sub_requirement_id"], f["verdict"], f["confidence"], f["quote"], f["source_chunk_id"], f["rationale"], bool(f["quote_verified"]), f["retry_count"], f["prompt_tokens"], f["completion_tokens"], f["cost_usd"]) for f in fs], json.loads(q["applied_deviations"]), q["rationale"]))
        return AnalysisResult(row["run_id"], row["source_path"], questions, checks, {"prompt_tokens": row["total_prompt_tokens"], "completion_tokens": row["total_completion_tokens"], "cost_usd": row["cost_usd"], "unpriced_model": bool(row["unpriced_model"])}, row["started_at"], row["finished_at"], incomplete=(row["status"] == "incomplete"))

    def audit_trail(self, run_id: str) -> list[dict]:
        return [json.loads(row["payload"]) for row in self.db.execute(
            "SELECT payload FROM audit_events WHERE run_id=? ORDER BY sequence", (run_id,)
        )]

    def list_runs(self, limit=50):
        return [dict(row) for row in self.db.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,))]

    def kpi_snapshot(self):
        from core.kpi import snapshot
        metrics = snapshot(self)
        row = self.db.execute("SELECT COUNT(*) runs, COALESCE(AVG(cost_usd),0) cost_per_run, COALESCE(AVG(duration_ms),0) mean_latency FROM runs").fetchone()
        unpriced_runs = self.db.execute("SELECT COUNT(*) FROM runs WHERE unpriced_model=1").fetchone()[0]
        return {
            **metrics,
            "cost_per_run": metrics["cost_per_document"],
            "mean_latency": row["mean_latency"],
            "unpriced_runs": unpriced_runs,
        }

    def kpi_history(self, days=30):
        from core.kpi import history
        return history(self, days)["days"]

    def latest_run_id(self) -> str | None:
        row = self.db.execute(
            "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return row["run_id"] if row else None

    def save_chat_message(self, session_id, run_id, seq, role, content, tool_calls=None, citations=None):
        with self.db:
            self.db.execute(
                "INSERT INTO chat_messages VALUES (?,?,?,?,?,?,?,?)",
                (
                    session_id,
                    run_id,
                    seq,
                    role,
                    content,
                    json.dumps(tool_calls) if tool_calls is not None else None,
                    json.dumps(citations) if citations is not None else None,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_chat_history(self, session_id):
        rows = self.db.execute(
            "SELECT * FROM chat_messages WHERE session_id=? ORDER BY seq", (session_id,)
        ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "seq": row["seq"],
                "role": row["role"],
                "content": row["content"],
                "tool_calls": json.loads(row["tool_calls"]) if row["tool_calls"] else None,
                "citations": json.loads(row["citations"]) if row["citations"] else None,
                "created_at": row["created_at"],
            }
            for row in rows
        ]
