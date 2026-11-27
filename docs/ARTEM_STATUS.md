# Artem — paired preliminary delivery

## Current result
**104/104 graph attempts saved:24 development +80 preliminary;104 successful calls,0 technical failures.** The user explicitly switched execution from Kaggle T4 to **RTX5080**. These are RTX5080 results, not T4 measurements. Correctness remains ungraded.

The original baseline corpus, question IDs, generator, prompt and budgets are unchanged. Frozen experiment SHA256:`3c6340bfa640523840e3962d217711afef428468b73bb4fdde63315836b095a6`. Both query files are byte-identical to the supplied baseline. Its264 payload hashes and104 answers passed verification. No additional baseline delivery or rerun is required from Karina.

## Execution evidence
- Same15,000 documents /45,142 chunks;136,470 graph nodes,43,147 relations and102,679 source-linked summaries, including12 extractive fallbacks.
- Shared `baseline_rag`/`SharedGenerator`, Ollama0.18.0, Qwen2.5-3B Q4_K_M digest357c53fb…c9e4b; k10, context2560, full input4096, output384, seed42.
- Actual RTX5080 hardware and successful GPU-resident shared-generator smoke are saved with the run. The graph model calls used GPU memory, not CPU Torch inference.
- All104 contexts and complete prompts were independently recounted: max context2558, max prompt2683; counts and pinned tokenizer/chat-template identities match. See [recount receipt](../reports/graph/retokenization_rtx5080.json).
- Six preliminary graph answers reached the384-output cap (four local, two global). All are retained, as are the original eight capped baseline answers. No selective reruns.
- Recorded split invocation elapsed:71.61s development and168.35s preliminary. Query components are exported separately; these are not historical indexing/GPU hours.

## Handoff ready
The private/local deliverable is [paired_graph_rtx5080_20261127.zip](../data/exports/paired_graph_rtx5080_20261127.zip),8,542,620 bytes, SHA256 `dfb5b4a965628056e2667afbb287ce421378c8e2848159297237c39e2d267620`. It is outside Git and supplied as a downloadable file, not a public data release.

It contains original104 graph attempts with contexts, evidence/source links, statuses, timings and usage; unchanged baseline ZIP; frozen identities/hardware; and:
- **160 preliminary answers** across all four retrievers, globally shuffled into blinded forms:1296 fact rows per reviewer;
- **24 graph development answers**,68 fact rows per reviewer; existing baseline development grades are not repeated;
- two independent human-review templates and a blank optional judge template; all labels remain null;
- coordinator-only private keys/reference labels, which must not be sent to reviewers;
-208 per-question baseline+graph measurements and separate index/query receipts, with a standalone261-file integrity verifier.

## Separate measurements
[Validated graph report](../reports/graph/completion.json), [paired summary](../reports/graph/paired_summary.csv), [per-question measurements](../reports/graph/paired_per_question.csv) and [measurement qualifications](../reports/graph/paired_measurements.json) keep coverage, correctness, context size, indexing and query cost separate. Baseline Windows and graph WSL/Python stacks differ; baseline physical GPU/warm state is not established, so raw latency differences are not a controlled architecture speedup.

Coverage is not correctness. Two preliminary questions per system have N/A document coverage, not automatically N/A correctness. Full historical extraction elapsed/GPU hours and electricity are unknown; summed overlapping requests are not elapsed time.

## Remaining team work
Safina coordinates independent grading, disagreement adjudication and the failed/N/A aggregation policy. Karina has no graph integration blocker to fix and need not regenerate existing baselines. The team decides readiness after paired correctness; **core200, ablations and learned routing remain deferred**.

## Reproduction
Use a fresh output directory for any genuinely new run; do not overwrite this first-execution evidence or human forms. Shared environment/config: [requirements](../requirements-graph-answers.txt), [local config](../configs/graph_local_15000.json). Analysis commands:

```sh
PYTHONPATH=src data/.venv-graph-answers/bin/python scripts/report_graph_delivery.py --config configs/graph_local_15000.json --extraction data/graph/corpus_15000 --output reports/graph/completion.json
python data/exports/paired_graph_rtx5080_20261127/verify_delivery.py
PYTHONPATH=src data/.venv-graph-answers/bin/python -m unittest discover -s tests -v
```

Earlier private Kaggle attempts stopped before answer generation (input packaging, zstd, then virtualenv/ensurepip). They are preserved as failures, not T4 validation; see [historical T4 notes](T4_VALIDATION.md). The [experiment log](../experiments/001-graph-completion.md) remains append-only.
