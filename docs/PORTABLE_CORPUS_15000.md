# Portable corpus for comparison on another computer

The standalone export is `data/exports/corpus_15000.tar.gz`, with an adjacent
`.sha256` file. Its unpacked directory is `data/exports/corpus_15000/`.
Large payloads are local, ignored by Git; copying the archive transfers the
actual corpus. A repository clone alone does not contain the archive.

The package contains exactly 15,000 source documents from nine source families,
45,142 BGE chunks, and the existing 212 project questions: 12 development,
100 fold A and 100 fold B. All 423 unique gold document IDs are included.
The selection uses source-proportional largest-remainder quotas, seed 42, and
mandatory inclusion of the question gold documents. Confluence's minimum of 167
gold documents exceeds its proportional quota of 152; 15 places are transferred
from Slack. No question scores or system outputs influence this selection.

This request explicitly requires a corpus with its questions, so evaluation gold
IDs are used to ensure source inclusion. This makes the corpus conditioned on
the question set. Gold labels remain separate from index and query inputs, and
must not be used for routing or tuning on evaluation questions. The original
shared manifests and question splits are unchanged. This export is a separate
comparison candidate.

## Transfer and use

Copy the `.tar.gz` and `.tar.gz.sha256` files. On the destination computer:

```sh
sha256sum -c corpus_15000.tar.gz.sha256
tar -xzf corpus_15000.tar.gz
python3 corpus_15000/verify.py corpus_15000
```

Verification needs only Python 3.10+ standard library. It checks every payload
hash, source ID/hash, document and chunk coverage, disjoint question splits and
presence of all expected document IDs. It fails on modified or extra files.

For direct comparison, feed **only** `chunks/chunks.jsonl` to each indexer and
preserve `chunk_id` and `source_document_id` in retrieval outputs. These frozen
chunks need no tokenizer download. For an indexer that accepts full documents,
use `documents.jsonl` (ID, source type, source path, rendered text); record any
different chunking policy. Raw source JSON is also supplied under
`dataset/generated_data/sources/`. Its manifest uses exact exported source bytes.

Use `questions/development.queries.jsonl` for development, and
`questions/fold_a.queries.jsonl` / `fold_b.queries.jsonl` for evaluation. Each query
contains only ID and question text. The corresponding `*.labels.jsonl` contain
the unchanged upstream gold answers, expected document IDs and answer facts.
Fit routing only from development scores. The package includes no generated
answers or fitted routing policy.

Source presence is not proof of answer completeness. In particular, the meaning
of corpus-global questions can change on a subset. The existing `qst_0413`
duplicate-document-ID ambiguity remains documented in
`questions/coverage_exclusions.json`; its document coverage stays N/A.

## Recreate the export

```sh
data/graph/.venv/bin/python scripts/package_corpus_subset.py \
  --dataset ../EnterpriseRAG-Bench \
  --output data/exports/corpus_15000 --documents 15000 --seed 42
```

The builder refuses to overwrite an existing directory. `bundle.json` records
the upstream revision, population hashes, unconstrained/actual quotas and seed.
The report `reports/graph/portable_15000_bundle.json` records this built export;
`reports/graph/portable_15000_document_ids.txt` freezes its selected IDs in Git.
Use the pinned BGE tokenizer revision, Python 3.12 and transformers 5.22.1 for
byte-identical chunk regeneration. The existing project chunker and renderer
are used unchanged; their code and tokenizer files are bundled for reference.
The archive is the portable source of exact source/newline bytes and chunks.

## Local indexing with memory limits

The 15k run uses `scripts/index_portable_corpus.py`, 8 concurrent requests and
the separate `configs/graph_vllm_5080_bounded.json` profile: GPU utilization 0.82,
8 sequences, 2,048 batched tokens, 8,192 context, no CPU weight offload. This
leaves more VRAM headroom than the previous 16-request profile. Unload Ollama's
35B model before starting the dedicated vLLM server; do not run both on the GPU.

The server's systemd RAM limit is 13 GiB (high watermark 11 GiB); the indexer's
limit is 5 GiB (high watermark 4 GiB). Swap is limited to 1 GiB per service.
These limits constrain the jobs rather than allowing them to exhaust the host.
If a limit is reached, inspect the service journal; do not silently raise it.
The previous crash could not be attributed from the saved kernel journals.

Graph assembly, summary inference, audit and each development retrieval run
execute in separate child processes. Memory is released between stages. Edge
supports are grouped by community in one pass instead of scanning all edges for
every community. Per-chunk extraction checkpoints allow resuming after a stop.
Run settings/code hashes are pinned; do not change graph code mid-run.

Progress: `data/graph/corpus_15000/progress.json`. Logs:
`data/graph/corpus_15000/job.log` and `journalctl -u graph-rag-vllm`.
If both services were stopped, first start the model server:

```sh
systemd-run --unit=graph-rag-vllm --collect \
  --property=WorkingDirectory=/root/graph-vs-vector \
  --property=MemoryHigh=11G --property=MemoryMax=13G --property=MemorySwapMax=1G \
  --property=KillMode=control-group \
  /root/graph-vs-vector/scripts/serve_graph_vllm_bounded.sh
```

Wait for startup (usually a few minutes). `curl -f http://127.0.0.1:8001/health`
must return successfully before starting the indexer. Use
`journalctl -u graph-rag-vllm -n 20 --no-pager` to inspect startup.
If either service is already active, do not launch a duplicate.

Resume the indexer with the same service command and output directory:

```sh
systemd-run --unit=graph-rag-index-15000 --collect \
  --property=WorkingDirectory=/root/graph-vs-vector \
  --property=MemoryHigh=4G --property=MemoryMax=5G --property=MemorySwapMax=1G \
  --property=KillMode=control-group --property=OOMPolicy=stop \
  --property=Requires=graph-rag-vllm.service --property=After=graph-rag-vllm.service \
  --property=StandardOutput=append:/root/graph-vs-vector/data/graph/corpus_15000/job.log \
  --property=StandardError=inherit \
  /root/graph-vs-vector/data/graph/.venv/bin/python -u \
  /root/graph-vs-vector/scripts/index_portable_corpus.py
```

This indexes all 45,142 supplied chunks, builds communities and summaries, audits
source links, and runs retrieval on development queries. It does not generate
evaluation answers or fit routing without the shared generation/scoring inputs.
