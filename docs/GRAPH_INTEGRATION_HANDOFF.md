# Graph/shared-generator handoff — completed on RTX5080

## 1. Common settings confirmed
Same12 development +40 preliminary IDs,15,000 documents /45,142 chunks, original prompt and decoding. Frozen experiment SHA256 `3c6340bfa640523840e3962d217711afef428468b73bb4fdde63315836b095a6`; both query files byte-identical to baseline. Shared Qwen2.5-3B Q4_K_M/Ollama0.18.0, model digest357c53fb…c9e4b, immutable tokenizer revisionaa8e7253…, k10, context2560, input4096, output384 and seed42. See [parity](../reports/graph/baseline_parity.json) and [independent graph recount](../reports/graph/retokenization_rtx5080.json).

## 2. Integration closed
Both graph retrievers ran through Karina's existing `baseline_rag`/`SharedGenerator`. Real [GPU smoke](../data/experiments/graph_local_15000/rtx5080_smoke.json), [hardware receipt](../data/experiments/graph_local_15000/rtx5080_hardware.json) and104 saved successful calls verify the path; no missing input or code change is required from Karina. Execution was **RTX5080**, explicitly authorized by the user instead of waiting for T4. T4 attempts failed before generation and are not model-feasibility evidence.

## 3. Attempts and grading handoff ready
**24 graph development +80 graph preliminary =104**,0 technical failures. Six graph outputs reached384; keep those flags and the original eight capped baseline outputs. Saved records include actual context, chunk/final document IDs, source links, errors/statuses, retrieval/packing/generation/query timings, usage, checkpoints and frozen run identities.

[Delivery/status](ARTEM_STATUS.md) identifies the private/local ZIP: raw104 graph attempts, unchanged baseline, full160-answer preliminary blinded forms,24 graph-development forms, coordinator keys and separate measurements. All human/judge labels are blank; do not give private system keys to reviewers. Safina owns independent grading, adjudication and failed/N/A policy. Existing baseline development labels cover the same12 questions (24 answers) and must not be re-labeled.

[Measurements](../reports/graph/paired_measurements.json) are not an accuracy result: coverage and correctness differ, two questions per system have N/A coverage, and hardware/platform/warm-state qualifications prevent a pure architecture speedup claim. No baseline rerun is needed. No core200, learned routing or ablation before paired correctness and team readiness review.
