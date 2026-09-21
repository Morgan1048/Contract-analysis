from __future__ import annotations

import re
import json
from dataclasses import dataclass

from core.llm.client import LLMResponse
from core.providers.errors import BackendError


@dataclass
class StubLLM:
    """Deterministic offline backend with deliberately configurable bad outputs."""

    mode: str = "success"
    failures_before_success: int = 0
    model: str = "stub"

    def __post_init__(self) -> None:
        self.calls = 0

    def complete_json(self, system: str, user: str, schema: dict, example: dict | None = None) -> tuple[dict, LLMResponse]:
        del schema, example
        self.calls += 1
        if self.mode == "success" and "ROLE: evidence_router" in system:
            data = json.loads(user)
            actions = [{"check_id": s["id"], "tool": "search_evidence",
                        "arguments": {"query": data["feedback"].get(s["id"]) or s["text"]},
                        "reason": "Offline orchestration fixture; not a semantic assessment."} for s in data["checks"]]
            text = json.dumps({"actions": actions})
            return {"actions": actions}, LLMResponse(text, len(user.split()), len(text.split()), 1, self.model)
        if self.mode == "success" and "ROLE: evidence_critic" in system:
            data = json.loads(user)
            reviews = [{"check_id": f["sub_requirement_id"], "decision": "accept", "gap": "", "confidence": 0.85} for f in data["findings"]]
            text = json.dumps({"reviews": reviews})
            return {"reviews": reviews}, LLMResponse(text, len(user.split()), len(text.split()), 1, self.model)
        first = re.search(r"\[chunk_id=([^\]]+)\]\s*(.*?)(?=\n\[chunk_id=|\Z)", user, re.DOTALL)
        chunk_id = first.group(1) if first else ""
        quote = first.group(2).strip() if first else ""
        if self.calls <= self.failures_before_success:
            mode = "invented_quote"
        else:
            mode = self.mode
        if mode == "rate_limited":
            # Simulates a 429/5xx that exhausted its transport-layer retries
            # (X2): this must never be reinterpreted as "the contract has no
            # controls" (X1).
            raise BackendError("simulated 429 Too Many Requests: retries exhausted")
        if mode == "interface_mismatch":
            # Simulates a client whose signature does not accept a keyword
            # argument the caller supplies -- an infrastructure/integration
            # bug, not malformed model output.
            raise TypeError("complete_json() got an unexpected keyword argument 'example'")
        if mode in {"malformed", "invalid_json"}:
            text = "not json"
        else:
            if mode in {"invented_quote", "invented"}:
                quote = "This sentence is not present in the supplied document."
            elif mode in {"whitespace", "whitespace_quote"}:
                quote = re.sub(r"\s+", "  ", quote, count=1)
            verdict = "MET"
            confidence = 0.9
            if mode in {"not_found", "empty"}:
                verdict, quote, chunk_id = "NOT_FOUND", "", ""
            if mode in {"out_of_range", "bad_confidence"}:
                confidence = 1.5
            text = json.dumps({
                "verdict": verdict,
                "confidence": confidence,
                "quote": quote,
                "source_chunk_id": chunk_id,
                "rationale": "Evidence supports the requirement.",
            })
            spec = re.search(r"Check specification: (\{[^\n]+\})", user)
            if spec:
                facets = json.loads(spec[1])["facets"]
                payload = json.loads(text)
                payload["evidence"] = ([{"quote": quote, "source_chunk_id": chunk_id, "role": "support"}] if quote else [])
                payload["facet_results"] = [
                    {"facet": facet, "verdict": verdict, "evidence_indices": [0] if quote else [],
                     "rationale": "Offline fixture only."} for facet in facets
                ]
                payload["gaps"], payload["applied_modifiers"] = [], []
                text = json.dumps(payload)
        return (
            {"raw": text} if mode in {"malformed", "invalid_json"} else json.loads(text),
            LLMResponse(text=text, prompt_tokens=max(1, len(user.split())), completion_tokens=max(1, len(text.split())), latency_ms=1.0, model=self.model),
        )
