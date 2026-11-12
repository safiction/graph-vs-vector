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

## Handoff work

1. Artem supplies the complete 15k graph index and summaries to execute paired
   graph-local/global answers. Both adapters already use this shared generator;
   sampled or mismatched graph indexes are rejected.
2. Safina performs fact-level correctness grading and human/judge calibration.
   Coverage exports are available; coverage is not answer correctness. Output
   limit flags must remain visible in evaluation.
3. Restore the one missing original source JSON for a complete raw-bundle audit
   or graph rebuild. Its verified chunks are present; both baselines indexed
   every shared chunk. The corpus is conditioned on including the questions'
   gold documents, and corpus-global answers may not be exhaustive on a subset.
