from __future__ import annotations

from dataclasses import dataclass, replace
import re

from core.agents.judge import judge
from core.llm.client import LLMResponse
from core.models import ParsedDocument
from core.providers.errors import BackendError
from core.rubric.schema import Finding, SubRequirement
from core.retrieval.router import EvidencePack
from core.obs.trace import set_span_attributes, span


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _expand_elision(quote: str, text: str) -> str | None:
    """Restore omitted source text only between unique, ordered literal anchors."""
    source = " ".join(text.split())
    if " ".join(quote.split()) in source:
        return None
    parts = [" ".join(part.split()) for part in re.split(r"\.{3,}|\u2026", quote)]
    if len(parts) < 2 or not all(parts):
        return None
    start, end = None, 0
    for part in parts:
        position = source.find(part)
        if position < 0 or position < end or source.find(part, position + 1) != -1:
            return None
        if start is None:
            start = position
        end = position + len(part)
    return source[start:end]


def restore_elided_quotes(finding: Finding, pack: EvidencePack) -> tuple[Finding, list[dict]]:
    sources = {chunk.chunk_id: chunk.text for chunk in pack.chunks}
    evidence = finding.evidence or ([{"quote": finding.quote, "source_chunk_id": finding.source_chunk_id}]
                                   if finding.quote else [])
    restored, changes = [], []
    for index, item in enumerate(evidence):
        source = sources.get(item["source_chunk_id"])
        expanded = _expand_elision(item["quote"], source) if source is not None else None
        if expanded is None:
            restored.append(item)
            continue
        restored.append({**item, "quote": expanded})
        changes.append({
            "check_id": finding.sub_requirement_id, "evidence_index": index,
            "source_chunk_id": item["source_chunk_id"], "original_quote": item["quote"],
            "restored_quote": expanded,
        })
    if not changes:
        return finding, []
    return replace(finding, quote=restored[0]["quote"], source_chunk_id=restored[0]["source_chunk_id"],
                   evidence=restored if finding.evidence else [], quote_verified=False), changes


def verify(finding: Finding, pack: EvidencePack, doc: ParsedDocument) -> list[Check]:
    fields_ok = (
        finding.verdict in {"MET", "PARTIAL", "CONTRADICTED", "NOT_FOUND", "UNRESOLVED"}
        and 0 <= finding.confidence <= 1
        and isinstance(finding.quote, str)
        and isinstance(finding.source_chunk_id, str)
    )
    checks = [Check("schema", fields_ok, "finding fields and enum are valid" if fields_ok else "invalid finding schema")]
    evidence = finding.evidence or ([{"quote": finding.quote, "source_chunk_id": finding.source_chunk_id}] if finding.quote else [])
    absence = finding.verdict in {"NOT_FOUND", "UNRESOLVED"} and not evidence
    sources = {c.chunk_id: c for c in pack.chunks}
    source_ok = absence or bool(evidence) and all(e["source_chunk_id"] in sources for e in evidence)
    checks.append(Check("source_in_pack", source_ok, "sources supplied" if source_ok else "source chunk was not supplied"))
    def text_matches(quote, text):
        return bool(quote.strip()) and " ".join(quote.split()) in " ".join(text.split())
    verbatim = absence or bool(evidence) and all(
        e["source_chunk_id"] in sources and text_matches(e["quote"], sources[e["source_chunk_id"]].text)
        for e in evidence
    )
    checks.append(Check("quote_verbatim", verbatim, "quotes verified (or absent)" if verbatim else "quote is not in the supplied chunk"))
    in_document = absence or bool(evidence) and all(text_matches(e["quote"], doc.text) for e in evidence)
    checks.append(Check("quote_in_document", in_document, "quote occurs in document" if in_document else "quote is not in document"))
    consistent = finding.verdict != "NOT_FOUND" or not evidence
    checks.append(Check("not_found_consistency", consistent, "NOT_FOUND has no quote" if consistent else "NOT_FOUND must not carry a quote"))
    cited = {e["source_chunk_id"] for e in evidence}
    if finding.applied_modifiers:
        checks.append(Check("modifier_citations", set(finding.applied_modifiers) <= cited,
                            "every applied modifier must have source evidence"))
    return checks


def judge_with_retry(sub_req, pack, doc, client, max_retries=2, return_response=False):
    retries = 0
    reason = ""
    last_finding: Finding | None = None
    last_checks: list[Check] = []
    last_response = None
    responses: list[LLMResponse] = []
    while True:
        try:
            with span(f"judge:{sub_req.id}", sub_requirement_id=sub_req.id):
                finding, last_response = judge(sub_req, pack, client, reason)
                responses.append(last_response)
                set_span_attributes(
                    model=last_response.model,
                    prompt_tokens=last_response.prompt_tokens,
                    completion_tokens=last_response.completion_tokens,
                    cost_usd=last_response.cost_usd,
                    retry_count=retries,
                )
            with span(f"verify:{sub_req.id}", sub_requirement_id=sub_req.id):
                checks = verify(finding, pack, doc)
        except BackendError as exc:
            # An infrastructure/interface failure is not a judgement about the
            # evidence and must never become NOT_FOUND/MET/PARTIAL. Transport
            # retries have already been exhausted (see X2); do not retry the
            # judge/verify loop again here -- report an unjudgeable finding
            # immediately so `aggregate` can mark the question Indeterminate.
            detail = f"{type(exc).__name__}: {exc}"
            finding = Finding(
                sub_req.id, "ERROR", 0.0, "", "",
                f"Backend call failed and this sub-requirement could not be judged: {detail}",
                False,
            )
            checks = [Check("backend", False, detail)]
            result = (replace(finding, quote_verified=False, retry_count=retries), checks, retries)
            return (*result, _combined_response(responses)) if return_response else result
        except ValueError as exc:
            detail = f"{type(exc).__name__}: {exc}"
            finding = Finding(sub_req.id, "UNRESOLVED", 0.0, "", "", f"Invalid model output: {detail}", False)
            checks = [Check("schema", False, detail)] + [
                Check(name, False, "not evaluated because judge output was invalid")
                for name in ("source_in_pack", "quote_verbatim", "quote_in_document", "not_found_consistency")
            ]
        last_finding, last_checks = finding, checks
        if all(check.passed for check in checks):
            result = (replace(finding, quote_verified=True, retry_count=retries), checks, retries)
            return (*result, _combined_response(responses)) if return_response else result
        if retries >= max_retries:
            result = (replace(finding, quote_verified=False, retry_count=retries), checks, retries)
            return (*result, _combined_response(responses)) if return_response else result
        reason = "; ".join(check.detail for check in checks if not check.passed)
        retries += 1


def _combined_response(responses: list[LLMResponse]) -> LLMResponse | None:
    """Account for every valid judge attempt, not only the final retry."""
    if not responses:
        return None
    last = responses[-1]
    costs = [response.cost_usd for response in responses]
    return replace(
        last,
        prompt_tokens=sum(response.prompt_tokens for response in responses),
        completion_tokens=sum(response.completion_tokens for response in responses),
        latency_ms=sum(response.latency_ms for response in responses),
        usage_estimated=any(response.usage_estimated for response in responses),
        cost_usd=None if any(cost is None for cost in costs) else sum(costs),
        unpriced_model=any(response.unpriced_model for response in responses),
    )
