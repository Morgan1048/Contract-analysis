"""Demo-data seeder for the KPI dashboard -- ``python -m eval.seed_demo``.

**This is demo-only tooling.** It fabricates a plausible run history so the KPI
dashboard has shape (distributions, a calibration curve, a fired threshold
alert) to look at without running ~40 real analyses. It must never be
confusable with real measurements, so it enforces three separate safeguards:

1. It writes to a database file that is *not* ``data/runs.db`` (default
   ``data/demo.db``) and refuses to run against ``data/runs.db`` at all.
2. Every row it inserts sets the schema's ``demo=1`` flag (see
   ``core/store/schema.sql``); real pipeline runs always write ``demo=0``.
3. It stamps a ``demo_meta`` marker row, and ``RunStore.is_demo_store()``
   checks both signals so the UI can render an unmissable banner whenever it
   is pointed at a demo-seeded store.

None of the numbers here are measurements of anything. They are invented for
UI/dashboard development and manual review only.
"""
from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.rubric.aggregate import load_rubric
from core.store import RunStore

REAL_DB_NAMES = {"runs.db"}

STATE_CHOICES = ["Fully Compliant", "Partially Compliant", "Non-Compliant"]
STATE_WEIGHTS = [0.55, 0.28, 0.17]

MODEL_CHOICES = [
    ("qwen2.5:14b-instruct", "ollama", True),
    ("llama3.1:8b-instruct", "ollama", True),
    ("gpt-4o-mini", "openai", False),
    ("openai/gpt-oss-120b", "openai", False),
]

# Deliberately elevated so at least one KPI threshold alert fires in the seeded
# history -- this is fabricated instability, not an observed regression.
QUOTE_FAILURE_RATE = 0.18
RETRY_RATE = 0.35


def _refuse_real_db(path: Path) -> None:
    if path.name in REAL_DB_NAMES:
        raise SystemExit(
            f"Refusing to seed demo data into {path} -- that is the real run store. "
            "Use --db-path to target a demo-only file (default data/demo.db)."
        )


def _random_timestamp(now: datetime, days: int, rng: random.Random) -> datetime:
    offset = rng.uniform(0, days)
    return now - timedelta(days=offset, hours=rng.uniform(0, 23), minutes=rng.uniform(0, 59))


def _weighted_state(rng: random.Random) -> str:
    return rng.choices(STATE_CHOICES, weights=STATE_WEIGHTS, k=1)[0]


def _verdict_for_state(state: str, rng: random.Random) -> str:
    if state == "Fully Compliant":
        return rng.choices(["MET", "PARTIAL"], weights=[0.9, 0.1], k=1)[0]
    if state == "Partially Compliant":
        return rng.choices(["MET", "PARTIAL", "NOT_FOUND"], weights=[0.35, 0.45, 0.2], k=1)[0]
    return rng.choices(["PARTIAL", "NOT_FOUND"], weights=[0.3, 0.7], k=1)[0]


def seed(store: RunStore, *, num_runs: int = 40, days: int = 30, seed: int = 20260101) -> int:
    """Populate ``store`` with ``num_runs`` fabricated runs spread over ``days``.

    Returns the number of runs inserted. Every row this function writes has
    ``demo=1``; see the module docstring for why that matters.
    """
    rng = random.Random(seed)
    rubric = load_rubric()
    question_ids = sorted(rubric)
    now = datetime.now(timezone.utc)

    with store.db:
        store.db.execute(
            "INSERT OR REPLACE INTO demo_meta VALUES (?,?)",
            ("seeded_by", "eval.seed_demo"),
        )
        store.db.execute(
            "INSERT OR REPLACE INTO demo_meta VALUES (?,?)",
            ("seeded_at", now.isoformat()),
        )

    for _ in range(num_runs):
        run_id = str(uuid.uuid4())
        started = _random_timestamp(now, days, rng)
        model, backend, is_local = rng.choice(MODEL_CHOICES)
        chunk_count = max(1, int(rng.gauss(130, 20)))
        table_row_chunks = max(0, int(rng.gauss(16, 4)))
        duration_ms = max(500.0, rng.gauss(45_000, 22_000))
        finished = started + timedelta(milliseconds=duration_ms)
        cost_usd = 0.0 if is_local else round(max(0.0, rng.gauss(0.12, 0.08)), 6)
        unpriced_model = 0
        status = "completed" if rng.random() > 0.03 else "incomplete"

        with store.db:
            store.db.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    "demo-seeded-contract.pdf",
                    started.isoformat(),
                    finished.isoformat(),
                    model,
                    backend,
                    rng.randint(400, 4000),
                    rng.randint(100, 1200),
                    cost_usd,
                    duration_ms,
                    status,
                    unpriced_model,
                    chunk_count,
                    table_row_chunks,
                    1,
                ),
            )

            for question_id in question_ids:
                question = rubric[question_id]
                state = _weighted_state(rng)
                confidence = round(min(0.99, max(0.35, rng.gauss(0.8, 0.15))), 3)
                applied_deviations = (
                    ["RA-2026-001"] if question_id in {"Q1", "Q5"} and rng.random() < 0.3 else []
                )
                rationale = f"Demo-seeded rationale for {question_id} ({state})."
                store.db.execute(
                    "INSERT INTO question_results VALUES (?,?,?,?,?,?,?)",
                    (run_id, question_id, state, confidence, rationale, json.dumps(applied_deviations), 1),
                )

                # Calibration needs noisy correctness, not a clean confidence/accuracy
                # correlation, so bins stay non-degenerate.
                correct = rng.random() < min(0.97, max(0.05, confidence - rng.uniform(-0.25, 0.25)))
                eval_id = str(uuid.uuid4())
                expected_state = state if correct else rng.choice([s for s in STATE_CHOICES if s != state])
                store.db.execute(
                    "INSERT INTO eval_results VALUES (?,?,?,?,?,?,?,?)",
                    (
                        eval_id,
                        run_id,
                        question_id,
                        expected_state,
                        state,
                        int(correct),
                        started.isoformat(),
                        1,
                    ),
                )

                for sub in question.sub_requirements:
                    verdict = _verdict_for_state(state, rng)
                    quote_verified = 0 if rng.random() < QUOTE_FAILURE_RATE else 1
                    retry_count = rng.choices([0, 1, 2, 3], weights=[0.65, 0.2, 0.1, 0.05], k=1)[0] if rng.random() < RETRY_RATE else 0
                    finding_confidence = round(min(0.99, max(0.1, rng.gauss(0.75, 0.18))), 3)
                    store.db.execute(
                        "INSERT INTO findings "
                        "(run_id,question_id,sub_requirement_id,verdict,confidence,quote,"
                        "source_chunk_id,quote_verified,retry_count,rationale,prompt_tokens,"
                        "completion_tokens,cost_usd,demo) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            run_id,
                            question_id,
                            sub.id,
                            verdict,
                            finding_confidence,
                            f"Demo-seeded quote for {sub.id}.",
                            f"demo-chunk:{sub.id}",
                            quote_verified,
                            retry_count,
                            "Demo-seeded finding; fabricated for dashboard shape, not a real judgment.",
                            prompt_tokens := rng.randint(400, 1800),
                            completion_tokens := rng.randint(50, 300),
                            0.0 if is_local else round(
                                (prompt_tokens * 0.15 + completion_tokens * 0.6) / 1_000_000,
                                8,
                            ),
                            1,
                        ),
                    )

    return num_runs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Seed a demo-only run history for the KPI dashboard. "
            "Writes fabricated rows flagged demo=1 to a separate database file; "
            "never targets data/runs.db."
        )
    )
    parser.add_argument("--db-path", default="data/demo.db", help="Demo database path (default: data/demo.db).")
    parser.add_argument("--runs", type=int, default=40, help="Number of fabricated runs to insert (default: 40).")
    parser.add_argument("--days", type=int, default=30, help="Spread runs over this many past days (default: 30).")
    parser.add_argument("--seed", type=int, default=20260101, help="Random seed for reproducible demo data.")
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete any existing file at --db-path before seeding.",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db_path)
    _refuse_real_db(db_path)

    if args.reset and db_path.exists():
        db_path.unlink()

    store = RunStore(db_path)
    inserted = seed(store, num_runs=args.runs, days=args.days, seed=args.seed)
    print(
        f"Seeded {inserted} DEMO runs (demo=1) into {db_path}. "
        "This is fabricated data for dashboard development only -- it is not a measurement. "
        f"View it with: CA_RUN_DB={db_path} ./.venv/bin/streamlit run app.py"
    )


if __name__ == "__main__":
    main()
