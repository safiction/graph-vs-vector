# Artem — paired preliminary delivery

## Current scope
Following the user's 27 November staff feedback: produce exactly **104 graph attempts** — graph-local/global on the same twelve development and forty preliminary questions as BM25/vector. Preserve the shared generator, prompt and context-token budget. **Do not train routing, run the 200-question core comparison or start optional ablations before paired correctness and team readiness review.**

Authority: [accepted scope](specs/2026-11-27-paired-preliminary.md), [execution plan](plans/2026-11-27-paired-preliminary.md), [team handoffs](DETAILED_PROJECT_PLAN.md).

## Verified inputs
- User supplied the baseline result archive; all **264 SHA256 checks, 104 baseline answers and 52 question IDs** passed verification. Eight preliminary baseline answers hit the fixed 384-token output cap; retain these flags, do not silently rerun them.
- The prepared graph frozen experiment equals the delivered baseline experiment exactly; development/preliminary query files are byte-identical. See [parity receipt](../reports/graph/baseline_parity.json).
- Same 15,000 documents /45,142 chunks; no raw source missing here. Existing graph:136,470 nodes,43,147 relations,102,679 summaries, including12 extractive fallbacks. Fresh structural source audit passed. No re-extraction is necessary.
- Index SHA256:`c4d7a3e84f8ebc05aa6e3ed339dd7ac9954f0a2d17d528dcc82968f9f39a2987`.

## Execution state
The graph answers have **not yet been generated**. The exact Ollama0.18.0 release and Qwen2.5:3b digest are verified locally for an offline private Kaggle payload. Kaggle authentication now works for account m1r0tvorxc; quota check showed29.97 GPU hours remaining. No notebook/dataset has been uploaded yet and no T4 result is claimed. The standalone kernel requires actual T4 hardware and preserves final success/error receipts inside its output archive.

The source-audit/cost report is [completion record](../reports/graph/completion.json); its answer-completion flags, not this heading, determine which attempts actually exist. Its routing field is deferred and the original folds are not required for this scope. [Experiment log](../experiments/001-graph-completion.md) is append-only.

## Reproduction / handoff
From the project root, create the minimal graph-answer environment:

```sh
uv venv data/.venv-graph-answers --python 3.12
uv pip install --python data/.venv-graph-answers/bin/python -r requirements-graph-answers.txt
PYTHONPATH=src data/.venv-graph-answers/bin/python -m baseline_rag --config configs/graph_local_15000.json prepare
```

CPU Torch is used only for tokenizer tensors. The exact pinned Ollama server/model owns inference and the shared adapter requires actual GPU residency. This environment is not for rebuilding BGE embeddings. Kaggle hardware receipts must explicitly identify T4; using the existing graph on T4 does not demonstrate graph construction there. See [hardware acceptance](T4_VALIDATION.md).

After the actual frozen T4 run, collect its saved development/preliminary runs, then export the human grading handoff and measured graph costs:

```sh
PYTHONPATH=src data/.venv-graph-answers/bin/python scripts/export_graph_development.py --config configs/graph_local_15000.json --output data/handoffs/graph_development_20261127
PYTHONPATH=src data/.venv-graph-answers/bin/python scripts/report_graph_delivery.py --config configs/graph_local_15000.json --extraction data/graph/corpus_15000 --output reports/graph/completion.json
PYTHONPATH=src data/.venv-graph-answers/bin/python -m unittest discover -s tests -v
```

Use a T4-local configuration/output path for artifacts collected from Kaggle; do not merge them into a differently identified 5080 run. Never remove run_config files to force reuse. The shared runner command for the authorized experiment is `run --retrievers graph-local graph-global --splits development preliminary` only.

## Separate measurements and responsibilities
- **Artem:**104 graph attempts with actual context/provenance/status/tokens/timings; graph supporting-document coverage, context size, measured indexing/query costs; private T4 receipts and handoff.
- **Safina:** agrees failure/N/A policy with the team, prepares full blinded grading batches, coordinates grading of all160 preliminary answers plus24 graph development answers, resolves disagreements and owns combined correctness/paired differences/uncertainty.
- **Karina:** original baseline delivery is received and verified; no new baseline run is required. Support shared-generator integration if a concrete issue appears and independently grade the assigned batch.
- **Whole team:** review failures/readiness before any full200 run. Corpus-global development coverage belongs to the later routing decision; it does not change this frozen preliminary sample.

Coverage is not correctness; keep coverage, factual correctness, actual context size, indexing cost and query cost separate. Human labels remain blank until reviewers fill them. Failed/N/A answers remain explicit until the team agrees an aggregate policy. No accuracy winner, confidence interval or learned router is inferred from execution success.

## Known limitations
Historical extraction session walls cover only a fraction of interrupted work; all retained attempt/retry tokens and request latencies can be counted, but total historical elapsed/GPU hours cannot be reconstructed. Current answer runner retries overwrite failed attempt files, so preserve first-execution summaries and never claim unrecoverable prior costs. Structural provenance is not semantic entailment. Qwen2.5-3B uses its upstream Qwen research license; private model cache packaging is not relicensing it as MIT.
