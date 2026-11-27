# Private Kaggle T4 execution

The user authorized this run. Account: `m1r0tvorxc`. The pre-run quota check returned29.97 GPU hours remaining. Never use `--public`, publish a notebook, submit to a competition, or replace the T4 check with a label change.

## Frozen inputs and preparation
The source is the same15,000-document/45,142-chunk graph and immutable baseline package. Manifest schema2 stores archives with an extra `.bin` suffix: Kaggle automatically expands recognized archives and turns their symlinks into empty files, so raw `.zip`/`.tar.gz` uploads are not byte-preserving. The private payload contains six checksum-verified assets: project source, graph delivery, baseline delivery, official Ollama0.18.0 archive, exact Qwen2.5:3b model cache and pinned answer-tokenizer cache. Model weights retain their upstream Qwen research license; this is not a new MIT distribution. Credentials, unrelated models, `.git`, environment files and local caches are excluded from the source archive.

```sh
PYTHONPATH=src data/.venv-graph-answers/bin/python scripts/prepare_kaggle_t4.py \
  --output data/kaggle/new-private-payload \
  --baseline-archive data/exports/baseline_results_15000.zip
```

The executed payload version is recorded in its `paired-inputs.json`, including source commit/dirty marker, exact code files and all asset hashes. Metadata declares dataset license `other` to preserve mixed upstream terms; the dataset and both kernels remain private. Do not replace this with a public dataset for convenience.

The dataset is `m1r0tvorxc/graph-vs-vector-paired-t4-inputs-20261127`. Server metadata confirms `isPrivate:true`. The first version was automatically expanded and must not be used for execution; upload the schema2 opaque revision with `kaggle datasets version`, preserving the failed version/logs. The corrected local payload is `paired-t4-inputs-20261127-v3`, published as private dataset revision2. Revision creation keeps the existing private visibility; do not recreate the expanded v2 local payload:

```sh
kaggle datasets version -p data/kaggle/paired-t4-inputs-20261127-v3 \
  -m "Preserve schema2 opaque archive bytes" -t -r skip -q
```

Kaggle may require phone verification/internet permission. Report that condition rather than bypassing it. Model/runtime assets are attached for offline availability; internet is enabled for the isolated Python environment and, if absent, the `zstd` system unpacker (`apt-get update/install`, logged separately). `OLLAMA_NO_CLOUD=1` disables remote model inference.

## Smoke before the104 attempts

Both metadata files set `is_private:true`, `machine_shape:NvidiaTeslaT4` and `enable_gpu:true`. The kernel asserts the actual GPU0 name from `nvidia-smi`, sets `CUDA_VISIBLE_DEVICES=0`, verifies every asset hash before unpacking and rejects wrong hardware. A safe archive extractor rejects path escapes and chained symlink pivots. The pinned server must recognize the original model digest and report actual GPU residency.

```sh
kaggle kernels push -p data/kaggle/t4-smoke-kernel --accelerator NvidiaTeslaT4 -t 1200
kaggle kernels status m1r0tvorxc/graph-vs-vector-t4-smoke-20261127
kaggle kernels output m1r0tvorxc/graph-vs-vector-t4-smoke-20261127 -p data/kaggle/t4-smoke-output
```

Inspect actual `hardware.json`, `ollama_server.json`, `smoke.json` and final `status.json`/`error.json`. The smoke uses synthetic evidence and is a diagnostic call, not one of the104 project attempts. Archive publication is atomic; a failed archive cannot overwrite a complete one or leave a contradictory success receipt.

## Paired preliminary run
The preliminary stage itself runs and checks the actual T4 synthetic smoke before any project attempt. After a standalone smoke succeeds, or directly as one gated smoke→104-attempt job to avoid a second allocation queue:

```sh
kaggle kernels push -p data/kaggle/t4-preliminary-kernel --accelerator NvidiaTeslaT4 -t 3600
kaggle kernels status m1r0tvorxc/graph-vs-vector-paired-preliminary-t4-20261127
kaggle kernels output m1r0tvorxc/graph-vs-vector-paired-preliminary-t4-20261127 -p data/kaggle/t4-preliminary-output
```

The code runs `baseline_rag prepare`, asserts frozen experiment equality and byte-identical development/preliminary queries against the delivered baseline, then runs only `--retrievers graph-local graph-global --splits development preliminary`. No core-fold execution, routing training or evaluation-label prompt input occurs.

Expected saved attempts: development24, preliminary80. Each attempt retains context, source links, status/error, tokens, timing and run identity. A nonzero runner exit/failure remains explicit; retrieve the artifacts and inspect saved attempt counts rather than inventing answers or blindly retrying. Do not enable `--retry-errors` on the original run, because that runner overwrites failed attempts and historical costs.

Scratch/model/runtime/source files stay under `/tmp/paired_graph`; only small receipts under `/kaggle/working/paired_graph` and `/kaggle/working/paired_graph_output.tar.gz` are published. Download only the archive with `--file-pattern '^paired_graph_output\\.tar\\.gz$'`; do not bulk-copy the unpacked input cache. The archive contains receipts and the experiment. Local analysis may mount the immutable input artifacts elsewhere; reporting verifies their content hashes without rewriting original saved paths/configuration. Retain the original T4 receipts before packaging the human-grading handoff.

## Claims that remain separate
T4 query-time use of a prebuilt graph is **not** T4 graph construction. Supporting-document coverage is **not** factual correctness. Setup/download/extraction time is separate from retrieval/generation query time. Summed overlapping request latency is not historical elapsed/GPU hours. Human correctness and failure/N/A aggregation remain Safina/team handoffs; no learned-router or full200 result is claimed.
