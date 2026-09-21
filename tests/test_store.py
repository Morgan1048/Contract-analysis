import threading
import sqlite3

from core.pipeline import AnalysisResult
from core.rubric.schema import Finding, QuestionResult
from core.store import RunStore


def _result(run_id):
    return AnalysisResult(
        run_id, "contract.pdf",
        [QuestionResult("Q1", "Question", "Fully Compliant", .9,
                        [Finding("P1", "MET", .9, "quote", "sec:6.6", "reason")], [], "reason")],
        {}, {"prompt_tokens": 2, "completion_tokens": 3, "cost_usd": .01},
        "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:01+00:00",
    )


def test_store_round_trip(tmp_path):
    result = _result("run-1")
    store = RunStore(tmp_path / "runs.db")
    store.save_run(result)
    loaded = store.get_run("run-1")
    assert loaded.questions[0].findings[0].quote == "quote"
    assert store.list_runs(1)[0]["run_id"] == "run-1"


def test_completed_run_persists_judge_tokens_cost_and_finding_usage(tmp_path):
    finding = Finding(
        "P1", "MET", .9, "quote", "sec:6.6", "reason",
        prompt_tokens=120, completion_tokens=30, cost_usd=.004,
    )
    result = AnalysisResult(
        "run-usage", "contract.pdf",
        [QuestionResult("Q1", "Question", "Fully Compliant", .9,
                        [finding], [], "reason")],
        {},
        {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "cost_usd": .004,
            "latency_ms": {"parse": 5, "judge": 20},
            "stage_usage": {
                "judge": {
                    "prompt_tokens": 120,
                    "completion_tokens": 30,
                    "tokens": 150,
                    "cost_usd": .004,
                }
            },
        },
        "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:01+00:00",
    )
    store = RunStore(tmp_path / "runs.db")
    store.save_run(result)

    judge = store.db.execute(
        "SELECT tokens, cost_usd FROM stage_timings "
        "WHERE run_id='run-usage' AND stage='judge'"
    ).fetchone()
    persisted = store.db.execute(
        "SELECT prompt_tokens, completion_tokens, cost_usd FROM findings "
        "WHERE run_id='run-usage' AND sub_requirement_id='P1'"
    ).fetchone()
    assert dict(judge) == {"tokens": 150, "cost_usd": .004}
    assert dict(persisted) == {
        "prompt_tokens": 120, "completion_tokens": 30, "cost_usd": .004,
    }


def test_older_database_is_migrated_with_finding_usage_columns(tmp_path):
    path = tmp_path / "old.db"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE findings (run_id TEXT, question_id TEXT, "
        "sub_requirement_id TEXT, verdict TEXT, confidence REAL, quote TEXT, "
        "source_chunk_id TEXT, quote_verified INTEGER, retry_count INTEGER, "
        "rationale TEXT, demo INTEGER NOT NULL DEFAULT 0, "
        "PRIMARY KEY (run_id, sub_requirement_id))"
    )
    connection.commit()
    connection.close()

    columns = {
        row["name"] for row in RunStore(path).db.execute("PRAGMA table_info(findings)")
    }
    assert {"prompt_tokens", "completion_tokens", "cost_usd"} <= columns


def test_store_reads_run_from_different_thread(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    store.save_run(_result("run-1"))
    loaded = []

    thread = threading.Thread(target=lambda: loaded.append(store.get_run("run-1")))
    thread.start()
    thread.join()

    assert loaded[0].questions[0].findings[0].quote == "quote"


def test_store_supports_concurrent_writes(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    barrier = threading.Barrier(2)
    errors = []

    def save(run_id):
        try:
            barrier.wait()
            store.save_run(_result(run_id))
        except Exception as error:
            errors.append(error)

    threads = [
        threading.Thread(target=save, args=("run-1",)),
        threading.Thread(target=save, args=("run-2",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert {run["run_id"] for run in store.list_runs()} == {"run-1", "run-2"}
