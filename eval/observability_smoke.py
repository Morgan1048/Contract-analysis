"""Verify local Langfuse ingestion with a deterministic HTTP model fixture, not an accuracy test."""
from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from core.ingest.parse import parse_pdf
from core.llm.client import LLMClient
from core.llm.stub import StubLLM
from core.obs.langfuse import config
from core.obs.otel import configure_otel, flush
from core.pipeline import analyze
from core.providers.config import ProviderConfig
from core.providers.progress import backend_wait_updates
from core.retrieval.embeddings import FakeEmbedder
from core.retrieval.index import ChunkIndex
from core.retrieval.vectorstore import VectorStore
from core.store import RunStore
from eval.synthetic import rubric, write_contract


class FixtureHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.server.rate_limit_once:
            self.server.rate_limit_once = False
            self.server.limited_at = time.monotonic()
            self.send_response(429)
            self.send_header("Retry-After", "1")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.server.limited_at is not None and self.server.resumed_after_s is None:
            self.server.resumed_after_s = time.monotonic() - self.server.limited_at
        messages = request["messages"]
        result, usage = StubLLM().complete_json(messages[0]["content"], messages[1]["content"], {})
        body = json.dumps({"choices": [{"message": {"content": json.dumps(result)}}],
                           "usage": {"prompt_tokens": usage.prompt_tokens,
                                     "completion_tokens": usage.completion_tokens}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rate-limit-once", action="store_true", help="Exercise a real 429 wait before graph completion.")
    args = parser.parse_args()
    lf = config()
    if lf is None:
        parser.error("Start local Langfuse first: ./setup.sh --langfuse")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CA_OBS_ENVIRONMENT"] = "test"
    configure_otel()
    server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
    server.rate_limit_once = args.rate_limit_once
    server.limited_at = None
    server.resumed_after_s = None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        pdf = write_contract(args.output_dir / "observability-fixture.pdf")
        index = ChunkIndex(parse_pdf(pdf), VectorStore(FakeEmbedder(), args.output_dir / "index"))
        client = LLMClient(ProviderConfig("openai", f"http://127.0.0.1:{server.server_port}/v1",
                                          "observability-fixture"))
        # Provider classification does not recognize this ephemeral loopback port.
        waits = []
        with backend_wait_updates(waits.append):
            result = analyze(pdf, client, rubric=rubric(), index=index, allow_remote=True)
        assert not result.incomplete
        if args.rate_limit_once:
            assert server.resumed_after_s >= 1, "Retried before the server's cooldown ended"
            assert any(w.http_status == 429 and w.phase == "waiting" for w in waits)
            assert any(w.phase == "resumed" for w in waits)
        RunStore(args.output_dir / "runs.db").save_run(result)
        assert flush(), "OTel exporter did not flush before its deadline"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    auth = base64.b64encode(f"{lf.public_key}:{lf.secret_key}".encode()).decode()
    trace_id = result.usage["trace_id"]
    url = f"{lf.base_url}/api/public/v2/observations?traceId={trace_id}&limit=1000&fields=core,basic,usage,model,io"
    observations = []
    for _ in range(30):
        try:
            with urlopen(Request(url, headers={"Authorization": f"Basic {auth}"}), timeout=5) as response:
                observations = json.load(response)["data"]
        except HTTPError as exc:
            if exc.code != 404:
                raise
        generations = [o for o in observations if o["type"] == "GENERATION"]
        if len(generations) == result.usage["llm_calls"] and any(o["name"] == "analyze" for o in observations):
            break
        time.sleep(2)
    else:
        raise RuntimeError(f"Langfuse did not expose the complete trace: {len(observations)} observations")
    expected = {"agent:router", "agent:judge", "agent:verify", "agent:critic", "tool:search_evidence"}
    assert expected <= {o["name"] for o in observations}
    if args.rate_limit_once:
        assert "backend:wait" in {o["name"] for o in observations}
    assert all(o.get("model") == "observability-fixture" for o in generations)
    assert all(o.get("input") and o.get("output") for o in generations)
    assert sum(o["usageDetails"]["input"] for o in generations) == result.usage["prompt_tokens"]
    assert sum(o["usageDetails"]["output"] for o in generations) == result.usage["completion_tokens"]
    report = {"synthetic": True, "run_id": result.run_id, "trace_id": trace_id,
              "trace_url": lf.trace_url(trace_id), "observations": len(observations),
              "generations": len(generations), "llm_calls": result.usage["llm_calls"],
              "rate_limit_recovery": args.rate_limit_once, "resumed_after_s": server.resumed_after_s}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
