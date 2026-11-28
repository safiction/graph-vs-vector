# Weeks 1–2: BM25/vector and shared generation

Both baseline indexes cover the same frozen **15,000 documents / 45,142 chunks**.
The recorded answer model is **Qwen2.5 3B Q4_K_M**, served by **Ollama 0.18.0**
with its full model digest pinned in `configs/baseline_15000.json`. BGE and the
answer tokenizer also use immutable Hugging Face revisions.

| Recorded work | Result |
| --- | --- |
| BM25 indexing | 45,142 chunks; 17.87 seconds |
| BGE indexing | 45,142 normalized 768-dimensional vectors; 413.58 seconds |
| Development | 24 answers, 0 failed calls; 79.17 seconds for the first execution |
| Preliminary | 80 answers, 0 failed calls; 233.74 seconds for the first execution |
| Preliminary output limits | BM25: 5; vector: 3 answers reached the 384-token cap |
| Tests | 41 passed; 0 failed, 0 errors, 0 skipped |
| Actual cache rerun | All 104 attempts preserved; all four prediction JSONL hashes unchanged |

Every saved answer includes actual context, supplied chunk/document IDs, model
and configuration identity, token usage, latency and finish reason. Contexts use
the real answer tokenizer and a 2,560-token cap. Evaluation questions were not
used for configuration or routing selection. The balanced 40-question sample
contains five questions per type per original evaluation fold, disjoint from
the 12 development questions.

`implementation.json` is the machine-readable artifact/audit report.
`development_trace.json` contains two source tracing examples checked by the
agent, one per baseline. These checks do not replace human correctness grading.
`requirements-lock.txt` records the actual installed environment.

Large artifacts are under `data/baselines/corpus_15000`; their detailed map and
reproduction commands are in `docs/VECTOR_RAG.md`. Both initial execution
summaries are retained separately from the subsequent cached invocation. Earlier
development startup/performance diagnostics are preserved separately and are
excluded from the frozen preliminary results.

## Current handoff — 28 November 2026

Artem's matching 104 graph attempts are delivered and verified, using this
shared generator on RTX 5080. No extra baseline delivery/rerun is required;
see `docs/ARTEM_STATUS.md`. The original 104 baseline outputs stay immutable.

Safina coordinates independent grading and failure/N/A policy; Karina helps
grade and adjudicate. The final 200-question baseline run waits for the team
readiness decision. `docs/FINAL_BASELINES.md` documents the final configuration,
read-only preflight, 400-attempt audit and saved-output demo. Clean-environment
baseline reproduction by another teammate remains to be recorded.

The missing original source is available in the graph delivery; restore it only
if a complete raw-source audit is needed on the original machine. Verified
chunks and indexes already cover every shared document. Historical
`implementation.json` is the original run receipt; its graph handoff blockers
are superseded by the November 27 delivery. The corpus is conditioned on the
questions' gold documents; corpus-global answers may be incomplete on this subset.
