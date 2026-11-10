# RTX 5080 graph inference with vLLM

The graph extraction/summarization preset uses **Qwen3.5 9B FP8 dynamic** on the
local RTX 5080, with vLLM continuous batching and up to 16 concurrent requests.
The existing checkpoint is `/root/models/Qwen3.5-9B-FP8`, converted by
[axteam](https://huggingface.co/axteam/Qwen3.5-9B-FP8-dynamic) from Qwen's model.
Every local weight, tokenizer and configuration file is SHA-256 pinned in
`reports/graph/vllm_weights.json`. Model loading consumes 10.29 GiB of GPU memory.
The vision encoder is skipped, CPU weight offload is zero, and the KV cache uses
the model's BF16 dtype. This preset does not change the comparison answer model,
baseline settings, shared chunking, question splits or corpus selection.

This is a tested local configuration, not a claim of the most accurate model
among every model that could fit on a 5080. Source-link checks prove traceability;
they do not replace independent semantic grading.

## Measured result on this PC

The final longer run used 144 random chunks (16 from each source family) and 15
additional development-source chunks. All 159 extraction calls completed.

| Measurement | Result |
| --- | --- |
| Extraction elapsed wall time, including client bookkeeping | 121.83 seconds |
| Input / generated extraction tokens | 118,235 / 79,803 |
| Aggregate generation throughput | 655.06 output tokens/second |
| Mean / peak GPU utilization | 98.50% / 100% |
| Peak total device memory | 15,593 MiB, about 15.23 GiB |
| Merged entities / relationships | 1,030 / 191 |
| Rejected extraction items | 679 |
| Leiden communities / completed summaries | 842 / 842 |
| Summary methods | 718 extractive singletons, 124 model syntheses |
| Summary elapsed wall time, including retries | 54.24 seconds |
| Structural source audit | Passed: 159 chunks from 148 documents, 1,184 entity mentions, 191 relation supports, 985 summary claims |
| Development retrieval | Both retrievers returned evidence for 12/12 questions; no generated answers or correctness grades |
| Tests | 25 passed |

The initial concurrency sweep, using a fixed earlier extraction prompt on the
same 18 random chunks, measured:

| Concurrent requests | Wall seconds | Output tokens/second |
| --- | --- | --- |
| 1 | 71.10 | 83.26 |
| 4 | 23.21 | 270.42 |
| 8 | 13.83 | 439.99 |
| 16 | 9.78 | 637.53 |

That sweep selected 16 workers. Its prompt/grammar differ from the final qualified
run, so use the 159-chunk result for current throughput and cost. Output contents
can vary with batch size despite greedy decoding. The final run kept the stricter
source/condition checks rather than selecting a configuration by relation count.

The larger stratified sample projects extraction at **3.05 hours for all 15,139
pilot chunks**, versus the original 35B pilot's rough 25-hour estimate. The full
benchmark projects **312 hours, about 13 days**, before summaries, preparation and
answer generation. The observed min/max sensitivity is 1.41–3.89 hours for the
pilot corpus and 145–399 hours for the full corpus; these are not confidence
intervals. A project-wide compute budget and frozen corpus decision are still
needed. A full-corpus job was not started.

Eight agent-authored engineering probes were inspected after prompt/validator
development. Retained relation quotations covered **16 of 17 reference facts**;
one numeric limit was omitted when its invented quotation was rejected. No
unsupported retained relation was found in that small reviewed set. This is a
tuned development diagnostic, not independent accuracy measurement. The raw model
still lost a conditional qualifier; the source-based condition guard restored it
before indexing. Arbitrary semantic errors remain possible.

One summary repeatedly mistyped a character in its source UUID. Its exact quote
matched only one supplied chunk, so source binding attached that chunk ID and
recorded the original ID and correction. Both failed attempts and their costs
remain in the checkpoint. No claim text or quotation was changed.

Reviewable reports: [GPU profile](../reports/graph/vllm_expanded_profile.json),
[cost projection](../reports/graph/vllm_cost.json),
[source audit](../reports/graph/vllm_source_audit.json),
[quality review](../reports/graph/vllm_quality_review.json), and
[initial concurrency sweep](../reports/graph/vllm_concurrency.json).

## Run the server

The PC already has a compatible inference environment at `/root/.venv-vllm`.
The graph client stays in `data/graph/.venv`. These are separate because the shared
chunker pins Transformers 5.22.1, while the measured vLLM environment has 5.20.2.
To install a separate inference environment elsewhere:

```bash
uv venv data/graph/vllm_env --python 3.12
uv pip install --python data/graph/vllm_env/bin/python --torch-backend cu130 -r requirements-vllm-graph.txt
GRAPH_VLLM_PYTHON="$PWD/data/graph/vllm_env/bin/python" scripts/serve_graph_vllm.sh
```

On this PC, simply run `scripts/serve_graph_vllm.sh`. The endpoint is
`http://127.0.0.1:8001`, served model `graph-qwen35-9b-fp8`. The script uses the
already cached local weights and local-only networking. It leaves the existing
Ollama service configuration untouched. The old loaded 35B instance was unloaded
from memory to free the GPU; its installed weights were retained. An Ollama 9B Q8
fallback was downloaded during initial selection but is not the measured vLLM model.

The current server runs as the transient systemd unit `graph-rag-vllm`:

```bash
systemctl status graph-rag-vllm
curl http://127.0.0.1:8001/health
systemctl stop graph-rag-vllm
```

To recreate that unit as root on this PC:

```bash
systemd-run --unit=graph-rag-vllm --collect --property=WorkingDirectory="$PWD" --property=KillMode=control-group "$PWD/scripts/serve_graph_vllm.sh"
```

It is not enabled to start at boot. CUDA compilation/autotuning takes extra time
on a cold start and is excluded from warmed extraction throughput measurements.
The launcher adds the inference environment's `bin` directory to PATH so FlashInfer
can find Ninja. It uses one CPU Torch thread, avoiding CPU spin-wait contention.

`configs/graph_vllm_5080.json` records 8,192 context tokens, 4,096 batched prefill
tokens, 16 concurrent sequences, prefix caching, CUDA graph capture sizes up to 16,
85% vLLM memory reservation and throughput mode. The client checks the live model
path/context against the pinned configuration. Overrides of model/runtime settings
need their own updated manifests and fresh extraction directory.

## Run graph extraction and summaries

From the project root with `PYTHONPATH=src` and `data/graph/.venv` active:

```bash
python -m graph_rag extract --chunks data/graph/shared_chunks/chunks.jsonl --manifest manifests/pilot_corpus.jsonl --sample data/graph/5080_expanded_sample.json --output data/graph/5080_expanded --workers 16
python -m graph_rag build --chunks data/graph/shared_chunks/chunks.jsonl --manifest manifests/pilot_corpus.jsonl --extraction data/graph/5080_expanded --dataset ../EnterpriseRAG-Bench --output data/graph/5080_expanded/index.json
python -m graph_rag summarize --index data/graph/5080_expanded/index.json --output data/graph/5080_expanded/summaries --workers 16
python -m graph_rag audit --index data/graph/5080_expanded/index.json --manifest manifests/pilot_corpus.jsonl --dataset ../EnterpriseRAG-Bench --summaries data/graph/5080_expanded/summaries --output data/graph/5080_expanded/source_audit.json
```

`--backend vllm` is now the CLI default. It uses the manifest above and local
endpoint 8001. `--backend ollama --model qwen3.6:35b` explicitly selects the old
backend with its original default of one worker. Extraction and summaries both
use a bounded thread pool and atomically checkpoint each item. Re-running an
identical configuration resumes without new calls. Worker count is pinned;
greedy decoding with a fixed seed is not guaranteed to be bitwise identical
across different batch sizes or GPU kernels.

Parallel runs log batch wall time separately from each request's latency. The cost
projection calibrates overlapping request latency against measured batch wall time;
summing concurrent latencies would overstate elapsed cost. Cost estimates remain
small-sample scenarios and do not establish fit against an unspecified budget.

## Structured output and quality controls

Temperature, presence/frequency penalties and thinking are zero/disabled; top-p
and repetition penalty are one, with seed 42. Exact source substrings and endpoint
names remain mandatory. The extraction prompt explicitly preserves complete noun
phrases, negation, proposal/completion status, numbers, units and conditions.
Summaries also preserve those qualifiers. Unsupported items and truncated calls
are retained as rejected items or failures rather than repaired into facts.
An additional narrow guard attaches an exact leading if/unless/when clause to
its consequent and rejects the antecedent as an observed fact. It supports leading
comma/semicolon-delimited clauses, including Russian если/когда, rather than general
semantic condition detection. A mistyped summary source ID can be bound to a source
only when its exact quotation identifies one unique supplied chunk; ambiguous or
invented quotations are rejected, and every binding correction is logged.

The JSON grammars are generated from the extraction/summary schemas with xgrammar,
allowing up to eight whitespace characters between JSON components. This bounds
whitespace loops while keeping ordinary formatted JSON possible. Checksums and
the schema-to-grammar mapping are pinned in the server configuration. To regenerate
after a schema change, use the inference environment:

```bash
/root/.venv-vllm/bin/python scripts/build_graph_grammars.py
```

A trial that forbade all flexible whitespace collapsed to empty graphs and was
rejected. Such results are never reported as speedups. The benchmark script rejects
aggregate empty extraction on the corpus sample and requires valid nonempty output
for every explicit-fact microbenchmark case. The eight agent-authored cases check
ownership scope, direction, negation, rejected proposals, numeric units, distinct
names, conditional statements and Russian text. They are engineering probes,
not benchmark answer grades or evidence of general precision/recall.

## Reproduce throughput measurements

```bash
python scripts/benchmark_graph_5080.py --weight-manifest reports/graph/vllm_weights.json --server-config configs/graph_vllm_5080.json --output data/graph/new_concurrency_run --workers 1 4 8 16
```

This uses the original 18 random pilot chunks for the concurrency sweep, then
runs the microbenchmark and original 33-chunk pilot. Warm-up is saved separately.
The GPU sampler records utilization, memory and power every 0.5 seconds. A fresh
output directory prevents cached extraction from being counted as throughput.

For a longer cost sample on the same shared pilot, generate a private graph sample:

```bash
python -m graph_rag sample --chunks data/graph/shared_chunks/chunks.jsonl --manifest manifests/pilot_corpus.jsonl --questions-manifest manifests/questions.json --dataset ../EnterpriseRAG-Bench --output data/graph/5080_expanded_sample.json --per-source 16
```

This selects 144 random chunks (16 per source family) and 15 additional development
source chunks. It does not freeze or change the comparison corpus. Development
examples are excluded from stratified cost extrapolation. Frozen-corpus indexing,
preliminary answer generation and score-fitted routing still require the shared
inputs listed in [GRAPH_PILOT_REPORT.md](GRAPH_PILOT_REPORT.md).

Implementation references: [vLLM's Qwen3.5 9B recipe](https://recipes.vllm.ai/Qwen/Qwen3.5-9B)
and the [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-9B).
