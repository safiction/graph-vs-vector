# Bounded graph indexing run: 2,000 documents

The user waived the agreed-budget acceptance check for the first project block
and authorized starting a practical partial-corpus indexing run. The pilot graph,
source links, entity-linking/neighbourhood demonstration and measured cost estimate
already satisfy that block under the revised requirement.

This run selects **2,000 complete documents, 6,061 original shared chunks**, from
the existing 5,000-document pilot. It preserves source-family proportions, includes
the 15 development gold documents, and uses seed 42. It does not inspect evaluation
gold documents or scores. Every original chunk of each selected document is kept;
chunk text, IDs, ordering and tokenization are unchanged.

The source quotas are: Confluence 20, Fireflies 40, GitHub 32, Gmail 474, Google
Drive 98, HubSpot 59, Jira 24, Linear 138 and Slack 1,115. The exact document IDs
are saved in [bounded_2000_document_ids.txt](../reports/graph/bounded_2000_document_ids.txt),
and the input hashes and estimates in [bounded_2000_plan.json](../reports/graph/bounded_2000_plan.json).

The measured vLLM pilot projects **1.22 hours of extraction**, plus a rough
**0.57 hours for community summaries**. A practical completion window is about
**1.5–2.5 hours**. This is an estimate, not a hard time limit: community structure,
model output length and retries can change the duration. The extraction output
cap is 2,048 tokens to give longer supported quotations room; normal shorter
outputs still terminate as usual. The graph server uses 16 concurrent requests.

## Background execution and progress

The job runs independently of the chat/terminal as `graph-rag-index-2000.service`,
using the separate `graph-rag-vllm` GPU server. It automatically performs extraction,
graph assembly, Leiden summaries, source auditing and development retrieval traces.
The tracked plan is not a completion claim; the live state is authoritative:

```bash
systemctl status graph-rag-index-2000
cat data/graph/bounded_2000/progress.json
cat data/graph/bounded_2000/run_status.json
tail -f data/graph/bounded_2000/job.log
```

`progress.json` is updated every ten seconds. `checkpointed_items` counts saved
attempts during extraction/summarization; only the final validated stage status
confirms success. The final `run_status.json` records `complete` only after source
auditing and both development retrieval runs succeed. It records failures explicitly.

The job uses an exclusive output lock and pins its implementation, model and
input manifests. Every model response is atomically checkpointed. Extraction
failures get at most two retry passes; unresolved failures prevent publication
of a complete index. A community summary that fails validation twice falls back
to exact supplied source excerpts, with method `extractive_fallback`; original
failed responses/costs and the fallback count remain visible. This preserves
provenance while giving a less abstract summary.

## Artifacts and resumption

All generated data stays in `data/graph/bounded_2000/`:

- `inputs/`: fixed source manifest, full shared chunks for the selected documents,
  and document IDs.
- `plan.json`, `run_config.json`: selection, timing estimate, hashes, model and code.
- `records/`, `sessions/`: extraction checkpoints, attempts, usage and elapsed times.
- `index.json`: assembled graph, communities and source links.
- `summaries/`: validated model summaries or explicitly tagged extractive excerpts.
- `source_audit.json`: source hash/ID/quotation audit.
- `development/`: local/global retrieval traces for the existing development set.

Prepare or verify the same selection:

```bash
PYTHONPATH=src data/graph/.venv/bin/python scripts/index_graph_subset.py prepare --documents 2000
```

Resume directly with the same pinned code and server:

```bash
PYTHONPATH=src data/graph/.venv/bin/python -u scripts/index_graph_subset.py run --output data/graph/bounded_2000
```

To create the independent background unit as root, with the vLLM server healthy:

```bash
systemd-run --unit=graph-rag-index-2000 --collect --property=WorkingDirectory="$PWD" --property=KillMode=control-group --property=Requires=graph-rag-vllm.service --property=After=graph-rag-vllm.service --property=StandardOutput="append:$PWD/data/graph/bounded_2000/job.log" --property=StandardError=inherit "$PWD/data/graph/.venv/bin/python" -u "$PWD/scripts/index_graph_subset.py" run --output "$PWD/data/graph/bounded_2000"
```

The original corpus, chunk/question manifests, baseline indexes and shared answer
generator are unchanged. This fixed subset is a graph indexing candidate; adopting
it as the shared frozen comparison corpus requires every retriever to use these
same inputs. This job does not choose the preliminary 40 questions, generate
comparison answers or fit routing from missing development scores.
