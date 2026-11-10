# Graph extraction, retrieval and development routing

This is Artem's graph component. It reads shared chunks and manifests and does not
select/freeze a comparison corpus, select evaluation questions, implement the
baselines, change the common answer prompt or grade correctness. The actual local
pilot and remaining dependencies are recorded in [GRAPH_PILOT_REPORT.md](GRAPH_PILOT_REPORT.md).

The current graph inference default is **local vLLM with Qwen3.5 9B FP8 and 16
concurrent requests**. Use [GRAPH_5080_VLLM.md](GRAPH_5080_VLLM.md) for its separate
server environment, launch commands and current measurements. The 35B commands
below describe the original pilot; its exact code is commit `5e7aae6`. Checkpoints
from that version must not be mixed with the updated prompt/backend.

## Local setup

Run from the project root, in WSL/Linux on the PC with the RTX 5080. All inference
uses an already installed local Ollama model; there are no paid API calls. Nothing
downloads or replaces model weights automatically. The tokenizer download below
is needed only when reproducing chunks.

```bash
uv venv data/graph/.venv --python 3.12
uv pip install --python data/graph/.venv/bin/python -r requirements-graph.txt
source data/graph/.venv/bin/activate
export PYTHONPATH=src
python -m graph_rag --help
```

The original measured pilot used `qwen3.6:35b` with digest
`a7eb95c53bcf96b4bdd008d0fab4a5dac88047d9c1a7a9ab88ed453423fbd87c`,
Ollama `0.40.14`, Python `3.12.16` and igraph `0.11.9`. Its weight file is larger
than the 5080's VRAM, so measured wall time includes the existing server's CPU
offload. Hardware and the complete installed Python versions are saved in
`data/graph/hardware.txt` and `data/graph/environment.freeze.txt`.

## Original pilot workflow (commit 5e7aae6)

The pilot is an extraction sample of the existing 5,000-document pilot, not a new
frozen comparison corpus. It has two uniformly sampled chunks per source family
(18 total; seed 42), plus the first chunk of each available development gold
document (15 additional chunks). Cost estimates use only the random sample, with
source-stratum weights. Development gold documents are used only for this pilot
demonstration, never for evaluation graph construction or query input.

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-base-en-v1.5', revision='a5beb1e3e68b9ab74eb54cfd186867f64f240e1a', allow_patterns=['tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','config.json'])"
python -m graph_rag prepare --manifest manifests/pilot_corpus.jsonl --dataset ../EnterpriseRAG-Bench --output data/graph --chunk-config manifests/pilot_chunk_config.json
python -m graph_rag sample --chunks data/graph/shared_chunks/chunks.jsonl --manifest manifests/pilot_corpus.jsonl --questions-manifest manifests/questions.json --dataset ../EnterpriseRAG-Bench --output data/graph/pilot_sample.json
python -m graph_rag extract --chunks data/graph/shared_chunks/chunks.jsonl --manifest manifests/pilot_corpus.jsonl --sample data/graph/pilot_sample.json --output data/graph/pilot --max-seconds 1800
python -m graph_rag build --chunks data/graph/shared_chunks/chunks.jsonl --manifest manifests/pilot_corpus.jsonl --extraction data/graph/pilot --dataset ../EnterpriseRAG-Bench --output data/graph/pilot/index.json
python -m graph_rag summarize --index data/graph/pilot/index.json --output data/graph/pilot/summaries --max-seconds 1200
python -m graph_rag audit --index data/graph/pilot/index.json --manifest manifests/pilot_corpus.jsonl --dataset ../EnterpriseRAG-Bench --summaries data/graph/pilot/summaries --output data/graph/pilot/source_audit.json
python -m graph_rag cost --extraction data/graph/pilot --sample data/graph/pilot_sample.json --manifest manifests/pilot_corpus.jsonl --full-index ../EnterpriseRAG-Bench/generated_data/uuid_index.json --summaries data/graph/pilot/summaries --output data/graph/pilot/cost.json
python -m graph_rag demo --index data/graph/pilot/index.json --summaries data/graph/pilot/summaries --questions-manifest manifests/questions.json --dataset ../EnterpriseRAG-Bench --output data/graph/pilot/development
```

The preparation command verifies every original manifest hash. On this PC, LF to
CRLF recovery reproduces those hashes in a private `data/graph/source_snapshot/`;
the original source clone is unchanged. It then invokes the existing chunker and
requires exact equality with the existing chunk configuration and chunk checksum.
An incomplete shared-chunk output is intentionally refused; choose a new graph
output directory before reproducing an interrupted preparation stage.

Extraction and summaries checkpoint each completed call through an atomic rename.
Re-run the same command to resume. `--retry-errors` explicitly retries saved
failures while retaining earlier attempts and their costs. Current commands also
accept `--workers` for bounded parallel requests and record elapsed batch wall time
separately from summed request latency. `--max-seconds` is a
per-invocation wall-time bound checked between calls, not an agreed project budget;
one in-flight request may finish after it. Run only one writer per output directory.
New weights, prompts or input manifests require a separate extraction directory.
Index construction refuses incomplete extraction. An extraction record can succeed
with zero valid entities/relations; retained and rejected item counts are reported
separately. Successful calls do not establish extraction quality.

## Retrieve evidence

```bash
python -m graph_rag query --system graph-local --index data/graph/pilot/index.json --query 'What default grace-window duration was chosen for the circuit-breaker?' --depth 1 --k 10
python -m graph_rag query --system graph-global --index data/graph/pilot/index.json --summaries data/graph/pilot/summaries --query 'What are the responsibilities around dedicated throughput and customer escalations?' --k 10
```

Local retrieval links normalized exact entity names, then strong multiword token
matches, with at most five seeds and 100 expanded nodes. It follows actual extracted
relations for depth 1 or 2 and ranks the mentioned chunks by TF-IDF cosine plus a
small graph-support contribution. Names are merged only across Unicode/case/
punctuation variants. Acronyms and abbreviated people names are not automatically
resolved. Generic terms such as `Support` can produce false links. An unlinked
question returns no evidence rather than silently becoming a lexical baseline.

Leiden detects communities on an undirected projection of extracted relations,
weighted by support count, with seed 42, modularity, resolution 1 and iterations
until stable. Each multi-node community gets up to four local-model claims from
at most 12 exact source snippets. Single-node communities use exact excerpts.
Every claim cites one supplied chunk and includes an exact supporting quotation.
Global retrieval ranks community summaries by fixed TF-IDF cosine and returns
their cited claims; `k` counts communities, so there may be up to `4*k` claim items.
This implementation is summary retrieval; it does not add a separate map/reduce
answer model or hierarchical community reports.

Both Python retrievers expose `retrieve(query, k=10) -> list[dict]` and
`retrieve_with_trace(query, k=10) -> dict`. Evidence has `text`, `score`, `chunk_id`,
`source_document_id`, `source_document_ids`, and `source_links`. Source links contain
the original document path, local file URI and manifest hash. Entity/relation
supports also retain exact quotation offsets inside their chunk. Global evidence
credits only sources of the actual supplied claims, never all documents in a
community. The audit verifies hashes, IDs, quotations, offsets and community links;
it does not establish semantic entailment or unambiguous real-world identity.

## Use the shared answer generator for the preliminary run

The actual frozen corpus, shared runner/settings and agreed 40-question set must
come from the data/pipeline owners. They are absent from both current checkouts.
Do not treat the benchmark's upstream OpenAI/Qdrant/OpenSearch examples as this
project's BGE/shared-chunk pipeline.

Once those inputs exist, extract **all** frozen shared chunks by omitting
`--sample`, using a fresh extraction directory. Build, summarize and audit that
index using its frozen source manifest. Do not rerun the pilot-only `prepare` or
`sample` steps to choose a comparison corpus.

Provide `comparison.json` from the actual frozen comparison configuration:

```json
{
  "shared_chunks_sha256": "sha256 of frozen shared chunks.jsonl",
  "frozen_source_manifest_sha256": "sha256 of frozen source manifest.jsonl",
  "preliminary_question_ids": ["the actual 40 agreed IDs"],
  "retrieval_k": 10,
  "context_token_budget": 4096,
  "generation_settings": {"all": "the team's exact model, tokenizer, prompt and decoding settings"}
}
```

The numbers above are schema examples, not chosen project settings. The external
`module:factory` accepts `generation_settings` and returns an adapter exposing:

```python
adapter.settings  # exactly the supplied generation_settings
adapter.count_tokens(text)  # the answer generator's real token counter
adapter.generate(question, context)  # {"answer": str, "metrics": {...}}
```

The adapter must reuse the team's exact prompt, model and decoding settings. It
receives no gold answer/facts/documents. Evidence packing counts chunk citation
labels and separators with that same tokenizer and omits whole items that do not
fit. The answer runner saves the actual context, final source IDs, retrieval trace,
latency, model usage and explicit generation/retrieval failures. These records use
the existing coverage script's `success`/`failed` statuses and
`final_source_document_ids` field. Failed attempts remain saved on resume; retry
them in a separately identified run rather than silently replacing failures.

```bash
python -m graph_rag run-preliminary --index data/graph/frozen/index.json --summaries data/graph/frozen/summaries --questions path/to/preliminary.jsonl --selection manifests/questions.json --comparison-config path/to/comparison.json --adapter shared_generation:make_adapter --output data/graph/preliminary
```

The command requires 40 distinct evaluation IDs, ten per project type, disjoint
from development. It rejects sampled graphs and mismatched corpus/settings. Both
systems use one shared configuration. Source hashes, summary content, graph code
and retrieval parameters are pinned in the run log. If the real shared runner
already supports retriever plugins, import the two retriever classes directly and
keep that runner's common evidence-packing procedure instead.

## Routing from development scores

```bash
python -m graph_rag train-router --scores path/to/development_scores.jsonl --questions data/graph/pilot/development/development.questions.jsonl --selection manifests/questions.json --output data/graph/router.json
python -m graph_rag route --router data/graph/router.json --question 'What is the tenant timeout?' --known-type single-fact
python -m graph_rag route --router data/graph/router.json --question 'What is the tenant timeout?'
```

Scores are JSONL objects with `question_id`, `retriever` and a finite `score` in
`[0,1]` (the team's externally graded correctness score). Every participating
retriever must have one score for every development ID. The default participants
are `bm25`, `vector`, `graph-local`, `graph-global`; `--systems` can explicitly name
a prototype subset. The trainer rejects evaluation IDs, duplicates, incomplete
pairings and unknown types/retrievers. It selects the highest development mean
for each type, with a fixed tie order. Known-type routing uses that table; predicted
types use token centroids trained only on development question text.

The current split has **zero corpus-global development questions**. That route
falls back to the overall development winner and is explicitly marked unlearned;
the classifier cannot learn corpus-global from this split. No real router is fitted
until actual paired development scores are available. These prototype outputs are
not cross-fold evaluation results; opposite-fold routing remains a later task.

## Validation

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The checks cover resume without model calls, corpus-change rejection, invented
quotations, graph depth, source-specific global evidence, citation-aware packing,
tampered provenance, evaluation-score leakage and missing score pairings. A
synthetic shared-generator fixture checks that 80 preliminary attempts are saved,
including failures, without sending gold answers to the generator. Synthetic
fixtures are tests only and are never reported as benchmark results.

Implementation references: [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md)
and [igraph Leiden API](https://python.igraph.org/en/develop/api/igraph.Graph.html).
