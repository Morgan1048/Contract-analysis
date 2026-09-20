# Limitations and interpretation

- One supplied sample is not evidence of broad generalization. Independent synthetic
  layouts, semantic mutations and live-provider fixtures exercise behavior, but a
  representative held-out contract corpus and human labeling are still required.
- Historical `eval/gold*.yaml` labels reflect an earlier rubric/override policy.
  They are retained for diagnostics, not automatically treated as current ground
  truth. Stub evaluations do not write accuracy or calibration rows.
- Exact retrieval-quality measurement needs a reviewed gold set against the
  shipping LangGraph retriever. Context precision/recall are not dashboard KPIs:
  `route()` / `collect_evidence()` and their `eval/` callers remain offline
  diagnostics of a different retriever, not measurements of production retrieval.
- The parser retains layout/table metadata and handles unnumbered prose and
  arbitrary document IDs. Borderless tables, merged cells, multi-column layouts and
  cross-page continuations remain heuristic. There is no universal PDF parser.
- OCR and image interpretation are not implemented. Image-only or potentially
  image-dependent evidence is explicitly marked for review. This is partial
  coverage of the assignment's general image-processing objective.
- Matching a quotation proves its provenance, not whether it entails a conclusion.
  The semantic critic may still be wrong. Optional LLM-as-judge scores also depend
  on the judge and are not substitutes for human reference labels.
- Confidence is a conservative heuristic, not a calibrated probability. The
  calibration UI requires labeled evaluations; a few synthetic cases do not
  establish statistical calibration.
- Contractual promises do not prove operational implementation. Risk acceptance is
  not automatic remediation/waiver, and expiry does not prove a gap was fixed.
  Scope, incorporation and precedence can require human interpretation.
- Agent budgets intentionally limit exploration. Context truncation, unresolved
  material references and exhausted budgets yield incomplete/review outcomes.
  A NOT_FOUND finding means a documented bounded search, not proof that no
  possible wording exists anywhere in the document.
- Indexing may explicitly degrade to lexical-only retrieval if embeddings fail.
  That degradation is recorded; fake embeddings are never a production fallback.
- Costs are estimates from configured price tables, not invoices. Groq zero rates
  assume free-tier use. Hardware/energy and optional offline evaluator costs are
  outside the main analysis API-cost figure.
- Audit snapshots are local SQLite records saved at completion. They preserve
  sources and decisions but are not tamper-proof, independently attested, or
  resumable LangGraph checkpoints.
- MCP is executable, but the REST API is a specification. Local deployment lacks
  hosted authentication, tenant isolation, retention enforcement and production
  access controls. Do not expose it to untrusted clients.
- Confidential source/derived material must remain local. Hosted tracing is
  disabled for graph state; remote model use requires explicit authorization
  for non-confidential material. Keep logs, fixtures and databases private.
