# Immediate graph/shared-generator handoff

## 1. Shared settings confirmed
The received baseline package passed264 payload hashes and104 saved answers. Prepared graph `experiment.json` is exactly equal to the delivered baseline experiment (SHA256 `3c6340bfa640523840e3962d217711afef428468b73bb4fdde63315836b095a6`). Development/preliminary query files are byte-identical:12 +40 questions, graph-local and graph-global,104 attempts total.

Both graph systems use Karina's existing `baseline_rag` runner and `SharedGenerator`: Qwen2.5-3B Q4_K_M, Ollama0.18.0/digest357c53fb…c9e4b, tokenizer revisionaa8e7253…, exact same prompt/decoding, k=10, context2560, full-input4096, output384 and seed42. All104 baseline contexts/prompts were retokenized successfully with the pinned tokenizer. See [parity receipt](../reports/graph/baseline_parity.json) and [current execution status](../reports/graph/paired_execution_status.json).

## 2. Interface and integration issues
The graph adapters already implement `retrieve_with_trace(question, k)` and feed source-backed evidence to the common packer/generator. Saved attempts follow the existing baseline format: question ID/text, retriever, status, answer, actual context/token count, evidence/source links, chunk/final document IDs, retrieval/packing/generation/query timings, generation token metrics and explicit errors. Each run preserves `run_config.json`, model identity, summaries, per-question checkpoints and prediction JSONL files.

**No missing data or code changes are currently required from Karina.** The actual end-to-end T4 smoke remains pending; software tests do not substitute for it. The first platform issue was Kaggle auto-expanding archives and converting tokenizer symlinks to empty files. This is resolved in private dataset revision2: all six opaque `.bin` archives and manifest are present with original sizes. Smoke notebook version2 is queued for actual NvidiaTeslaT4 execution. The first smoke failed before any model call and is retained as a failure, not relabeled success.

## 3. Next deliverable —104 attempts, not routing/core
After a real T4 smoke passes, run only:

```sh
python -m baseline_rag --config <T4-local-config> run \
  --retrievers graph-local graph-global --splits development preliminary
```

Save every success/failure. Provide Safina the24 graph development and80 graph preliminary attempts with contexts/provenance for independent grading; provide separate coverage/context/index/query measurements for comparison. Success/coverage are not correctness. Baseline preliminary output-cap flags remain visible; no baseline rerun is requested.

Do not start learned routing, optional ablations or the full200-question run before paired correctness and team readiness review. Raw query-time comparisons must disclose differing baseline/T4 hardware. This run consumes the prebuilt graph and cannot establish T4 graph-construction feasibility.
