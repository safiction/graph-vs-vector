# BM25, vector RAG and shared generation on the 15k corpus

## Local delivery integration — 12 November update

The original paths/reproduction commands below describe Karina's recorded run. This checkout keeps immutable deliveries under `data/packages/`. See [Weekly project plan](DETAILED_PROJECT_PLAN.md). Use `configs/graph_delivery_15000.json` for graph-only generation against those packages with the same recorded generator settings and new outputs. Its preparation passed; inference remains pending. Do not run `all` merely to consume existing graph artifacts. The raw source missing on the original machine exists in the graph delivery here; ordinary Windows access still reports 64 overlong source paths as missing, all found through extended paths. These do not remove verified input chunks.

The baselines consume all 45,142 existing chunks in `data/corpus_15000`,
preserving all 15,000 document IDs. No re-chunking, exclusion or deduplication
is applied. Large models, indexes and predictions stay in Git-ignored `data/`.

## Setup

Run from the repository root, using Python 3.12 and a separate environment:

```powershell
py -3.12 -m venv data/.venv-baselines
data/.venv-baselines/Scripts/python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
data/.venv-baselines/Scripts/python.exe -m pip install -r requirements-baselines.txt
data/.venv-baselines/Scripts/python.exe scripts/download_baseline_models.py
data/.venv-baselines/Scripts/python.exe scripts/manage_baseline_ollama.py start
data/.venv-baselines/Scripts/python.exe scripts/manage_baseline_ollama.py pull
```

On Linux use `bin/python` in place of `Scripts/python.exe`. Install Ollama 0.18.0
from its official release if it is not on PATH. The CUDA wheel requires a
compatible NVIDIA driver. The dedicated server uses only 127.0.0.1:11435, with
model storage and logs inside this project. User-wide environment settings and
the default Ollama service are independent of this server.

The configuration pins BGE-base-en-v1.5 and the Qwen2.5-3B tokenizer to immutable
Hugging Face revisions. Answers use Ollama's Qwen2.5 3B Q4_K_M model, pinned by
its full digest and Ollama version. It is a runnable laptop baseline; the answer
model remains configurable for the team. The runner verifies server/model
identity and GPU residency. Downloads stay under `data/models`. The download
helper fetches BGE weights and the answer tokenizer; Ollama pulls separate
quantized answer weights into `data/models/ollama`.

The alternative `transformers` backend supports FP16 inference with
`quantization: null`, or NF4 with FP16 computation/double quantization using
`quantization: "nf4"`. Select that backend before downloading full answer weights
with the helper. Query embeddings finish before Transformers loads the answer
model. Explicit CPU experiments need a separate config/output, float32 dtypes,
CPU devices, no quantization, and no GPU requirement. CUDA errors never silently
turn into CPU results.

The exact observed package versions, including transitive dependencies, are saved
in `reports/baselines/requirements-lock.txt`. Recreate that environment with
`pip install -r reports/baselines/requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128`.

References: [BGE pooling/query instructions](https://huggingface.co/BAAI/bge-base-en-v1.5),
[Qwen model, license and chat template](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct),
[Ollama quantized model](https://ollama.com/library/qwen2.5:3b),
[SQLite FTS5 BM25](https://www.sqlite.org/fts5.html#the_bm25_function).

## Run the two-week deliverables

```powershell
$env:HF_HUB_OFFLINE = '1'
data/.venv-baselines/Scripts/python.exe scripts/run_baselines.py all
data/.venv-baselines/Scripts/python.exe scripts/report_baselines.py
data/.venv-baselines/Scripts/python.exe scripts/manage_baseline_ollama.py stop
```

Individual stages (start the project server before generation or report auditing):

```powershell
data/.venv-baselines/Scripts/python.exe scripts/run_baselines.py prepare
data/.venv-baselines/Scripts/python.exe scripts/run_baselines.py build
data/.venv-baselines/Scripts/python.exe scripts/run_baselines.py query --retriever bm25 --question "What is the deployment policy?"
data/.venv-baselines/Scripts/python.exe scripts/run_baselines.py run --retrievers bm25 vector --splits development preliminary
data/.venv-baselines/Scripts/python.exe scripts/run_baselines.py status
data/.venv-baselines/Scripts/python.exe scripts/report_baselines.py
```

Pass `--config path.json` before the subcommand for another configuration.
The default is `configs/baseline_15000.json`. Configuration/code changes cannot
reuse saved predictions; use a new output directory for another experiment.
Indexes are checked against both corpus identity and artifact checksums.

`all` saves 24 development attempts (12 per baseline) and 80 preliminary attempts
(40 per baseline). The preliminary sample is frozen before inference: five
questions per project type from each evaluation fold, using seed 42 and SHA256
ordering. It is disjoint from development. Query inputs contain only ID and
question; labels are read separately after inference for coverage scoring.
The preliminary set is for evaluation, not tuning.

For a teammate to inspect the exact recorded outputs, copy the ignored
`data/baselines/corpus_15000` directory. Reproduction additionally needs the
portable corpus at `data/corpus_15000` and the pinned model assets (download them
with the setup commands or copy `data/models`). A Git clone contains the code,
small manifests/reports and commands, not those large payloads.

The small tracked `manifests/baseline_corpus_15000.json` and
`manifests/baseline_preliminary.json` preserve the corpus/configuration hashes and
selected IDs for the handoff; the original 5k pilot manifests remain separate.

## Shared comparison settings

All 45,142 chunks are embedded using CLS pooling and normalized 768-dimensional
BGE vectors. Documents have no instruction prefix; queries use BGE's retrieval
instruction. Exact cosine search uses float32 vectors (about 132 MiB).
SQLite BM25 uses Unicode tokenization, k1=1.2 and b=0.75, with OR over query
terms. Both return stable IDs, scores and source links.

Every system retrieves 10 candidates and uses the existing graph-side whole-item
packing function, identical prompt and answer tokenizer. The context cap is
2,560 answer-model tokens, including source labels and separators. Full prompts
have a separate 4,096-token input cap; oversized prompts fail instead of silently
truncating. Greedy decoding produces up to 384 answer tokens. Finish reasons
record answers reaching that limit. Only sources actually packed into context
receive provenance credit.

## Artifact map

Paths are relative to `data/baselines/corpus_15000/`:

| Artifact | Purpose |
| --- | --- |
| `experiment.json` | Frozen corpus, models, IDs, prompt, decoding, k and budget. |
| `comparison.json`, `questions/preliminary.graph.jsonl` | Same frozen settings and label-free typed queries for the older graph CLI. |
| `corpus_audit.json` | Chunk/manifest hashes, counts and raw-file completeness. |
| `questions/*.queries.jsonl` | Label-free development, preliminary and original fold queries. |
| `questions/selection.json`, `metadata.json` | IDs, seed, input hashes and fold/type mapping. |
| `indexes/bm25/index.sqlite` | BM25 index and canonical chunk text/provenance. |
| `indexes/vector/embeddings.npy` | Vectors in frozen shared chunk order. |
| `indexes/*/settings.json`, `metadata.json` | Index settings, hashes, sizes and build times. |
| `indexes/vector/progress.json` | Embedding checkpoint, throughput and active time. |
| `indexing_report.json` | Index build costs and runtime environment. |
| `runs/<split>/run_config.json` | Corpus, code, question, index and generation identities. |
| `runs/<split>/retrieval/<system>/*.json` | Candidates, packed context and retrieval timing. |
| `runs/<split>/attempts/<system>/*.json` | Atomic answer checkpoints, including failures. |
| `runs/<split>/<system>.predictions.jsonl` | Answers, contexts, provenance, statuses and runtime logs. |
| `runs/<split>/<system>.coverage.json` | Coverage from final context, per question and type. |
| `runs/<split>/summary.json` | Success/failure counts, latency, tokens and per-type coverage. |
| `runs/<split>/first_execution_summary.json` | Preserved first-run costs for the recorded experiment, before cache validation. |
| `run_report.json` | Combined execution report. |

`scripts/report_baselines.py` checks all 104 attempts, real tokenizer budgets,
source IDs and exact supplied source text. It writes the small, Git-trackable
`reports/baselines/implementation.json` with input/output hashes, costs, tests and
handoff blockers. It exits unsuccessfully if any required baseline output fails
the audit.

Interrupted embedding runs resume after the last flushed batch. Each answer is
written atomically; rerunning skips existing attempts. An OS lock prevents
concurrent writers and releases automatically after a crash. `run --retry-errors`
retries failures with unchanged settings. Aggregate exports regenerate on resume.

Retrieval timing includes an amortized share of batched query encoding and BGE
loading. Packing is separate. First-answer generation wall time includes model
loading; generation metrics also isolate inference time. Summaries distinguish
invocation time from per-query time. Tokens use the real model tokenizer.

Coverage reuses `src/coverage.py`, including qst_0413 and N/A for no-gold-document
questions. Failures have N/A coverage and explicit counts. Correctness is not
inferred from coverage; fact grading/human calibration remain separate.

## Graph handoff

Set `graph_index` and `graph_summaries` to Artem's complete 15k artifacts:

```powershell
data/.venv-baselines/Scripts/python.exe scripts/run_baselines.py run --retrievers bm25 vector graph-local graph-global --splits preliminary
```

Use a new output directory when changing selected systems after a saved run.
Sampled graphs or different chunk/manifest hashes are rejected. The older graph
CLI can use `--adapter baseline_rag.generation:make_adapter` with identical
`generation_settings`; the adapter exposes `settings`, `count_tokens` and
`generate` as required by its existing contract.

`comparison.json` and `questions/preliminary.graph.jsonl` are the compatible
comparison config and typed, label-free 40-query input for that older CLI. Start
the same pinned answer server and point its adapter at these generation settings.

For local servers, backends `ollama` and `openai-compatible` accept `endpoint`
and `served_model` in `generation_settings`. The pinned tokenizer must match the
served model. Transformers generation needs no server or API key.

`scripts/manage_baseline_ollama.py` verifies its PID's command and project model
directory before stopping anything. `start` is idempotent; `status` reports model
identity/residency; `stop` releases only this project's server and its children.
For the default Ollama backend the runner additionally checks the frozen server
version, full model digest, quantization level and GPU residency.

Saved predictions are the reference artifacts. GPU kernels and backend caches
can vary generated wording even with a fixed seed and greedy decoding; changing
hardware or backend calls for a separate experiment/output directory.

The original baseline-machine bundle lacked one raw source JSON; its verified chunks and rendered
document are present. Baseline indexing uses all chunks. Full raw-source
verification or a graph rebuild there needs the missing original file. It is present in the graph delivery here. Its exact path
is recorded in `corpus_audit.json`.

## Tests

```powershell
data/.venv-baselines/Scripts/python.exe -m unittest discover -s tests -p test_baseline_rag.py
```

To include the existing graph regression tests, install
`requirements-baselines-test.txt` and run `-m unittest discover -s tests`.

Tests cover real BM25 ranking, cosine ranking/provenance, embedding resumption,
checksums, balanced disjoint selection, shared generation for all four retrievers,
label leakage, context limits, saved failures, prediction resumption and rejection
of sampled graphs. Test encoders/generators are fixtures; production uses real
configured models.
