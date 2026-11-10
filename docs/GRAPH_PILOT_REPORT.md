# Local graph pilot — 10 November 2026

The graph-only pilot ran on the PC's RTX 5080 using its existing local Ollama
model. Local/global retrieval implementations, source audits, cost projection,
the shared-generator adapter and development-only routing code are available.
**Frozen-corpus indexing, preliminary answer generation and fitting routing from
real scores are still waiting for shared inputs.** No other role's code or shared
manifests were changed.

## Measured outputs

| Item | Result |
| --- | --- |
| Existing pilot corpus | 5,000 documents, 15,139 shared chunks |
| Exact reproduced chunks SHA-256 | `e0baecd1cd80a430799ad6f73d224058e49c7ceda4769787fc6064e8eb8baee7` |
| Extraction sample | 18 random cost chunks + 15 development demonstration chunks |
| Extraction calls completed / errors | 33 / 0 |
| Extraction elapsed wall time | 215.80 seconds |
| Extraction input / output tokens | 19,600 / 14,667 |
| Retained entity mentions / merged entities | 214 / 196 |
| Retained relationships | 19 |
| Rejected ungrounded extraction items | 141 |
| Leiden communities | 177 |
| Summary methods | 17 local-model syntheses, 160 extractive singletons |
| Summary elapsed wall time | 39.15 seconds |
| Summary input / output tokens | 3,988 / 2,278 |
| Structural source audit | Passed: 33 source documents/chunks, 214 entity mentions, 19 relation supports, 182 summary claims |
| Development retrieval traces | 12 questions for each graph retriever; 24 saved traces |
| Actual preliminary generated answers | 0; shared inputs unavailable |
| Tests | 17 passed, including the four existing coverage tests |

The graph is sparse because unsupported model names/quotations/endpoints were
rejected. A successful JSON extraction call is not evidence that it extracted a
useful graph. The audit proves traceability, not semantic correctness. The local
model has more weights than available VRAM; measurements include its existing
CPU/GPU offload and server overhead, rather than isolated GPU kernel time.

## Extraction cost projection

Two chunks were sampled uniformly from each of the nine source families, then
weighted by each family's chunk population. Development-targeted chunks were
excluded from extrapolation. Full-corpus document counts come from the sibling
benchmark's 511,958-entry UUID index; chunk counts are estimated using each source
family's pilot chunks/document ratio.

| Target | Estimated extraction wall time | Observed min/max sensitivity |
| --- | --- | --- |
| All 15,139 chunks of the 5,000-document pilot | 25.03 hours | 19.75–30.32 hours |
| Full 511,958-document benchmark, estimated 1,550,253 chunks | 2,563.19 hours, about 107 days | 2,022.19–3,104.19 hours |

This is a small-sample estimate for this exact model/server, not a statistical
confidence interval or a lower bound for every possible model. It excludes full
corpus preparation, graph assembly, summary inference and answer generation.
Full extraction is approximately 839 million input and 661 million output tokens.
The local API incurs no service billing; electricity and hardware are unpriced.
No agreed compute budget was found in either repository, so budget fit remains
undetermined. A smaller model or reduced common corpus needs a new measured pilot
and the team's shared corpus decision; this work did not make that decision.

## Entity linking and neighbourhood demonstration

The development question `qst_0315` asks about the escalation-mode UI indicator
and the default grace-window for a circuit-breaker. The linker matches
`grace-window`; one-hop expansion reaches `retry storms` through the quoted
relationship “implement failure grace-window + customer fast-path to contain
retry storms”. Retrieval returns the original chunk, retaining its stable ID,
source-document ID, source JSON path and hash. See the
[saved trace](../reports/graph/neighbourhood_demo.json).

Both graph retrievers were run on all 12 development questions. Local retrieval
returned evidence for 6/12 and no evidence for 6/12. Global retrieval returned
summary evidence for 12/12. These are retrieval availability counts, not answer
correctness or final-context coverage scores. Generic entity matching can also be
wrong: `support triage` in `qst_0170` links to the `Support` team and expands to
`Customer Success`, yielding unrelated documents. No development query tuning or
evaluation-score claims were made from these traces.

The saved development files explicitly say
`retrieval_only_pending_shared_generator` and have no generated answers. A source
audit does not change that status.

## Dependencies outside the graph component

Both `/root/graph-vs-vector` and `/root/EnterpriseRAG-Bench` were checked. The
sibling benchmark has the source corpus, all 500 benchmark questions, upstream
retrieval examples and evaluation scripts. It has no saved project-specific
frozen manifest, preliminary selection, shared local BGE generation configuration,
paired development scores or agreed graph compute budget.

| Needed input | Owner / reason |
| --- | --- |
| Agreed compute budget and frozen corpus/chunk manifest | Team/data: required before selecting an affordable identical corpus for every system |
| Actual 40-question preliminary selection, ten per type | Data/evaluation: must be identical for all four systems |
| Shared answer generator, prompt, tokenizer, context cap and decoding settings | Pipeline: required for comparable generated answers and final-context coverage |
| Paired development scores for the participating systems | Evaluation: required to fit routes without using preliminary/evaluation answers |

The graph command for the preliminary run is implemented and tested against a
synthetic shared-generator fixture. It refuses to present a sampled pilot graph
as a full frozen-corpus comparison. No real preliminary answers or empirically
fitted routes were fabricated. The development manifest has no corpus-global
examples; routing explicitly marks that type unlearned and uses a documented
fallback rather than borrowing evaluation labels.

## Artifact locations

- [Measured extraction/summary costs](../reports/graph/cost.json)
- [Source audit](../reports/graph/source_audit.json)
- [Neighbourhood trace with source links](../reports/graph/neighbourhood_demo.json)
- [Outstanding dependencies](../reports/graph/dependencies.json)
- Full graph and call checkpoints: `data/graph/pilot/index.json`, `data/graph/pilot/records/`
- Summaries and their checkpoints: `data/graph/pilot/summaries/`
- Development traces: `data/graph/pilot/development/`
- Setup, interfaces and reproduction commands: [graph_rag.md](graph_rag.md)

Large inputs, indexes, snapshots and the private Python environment stay under
the existing Git-ignored `data/` directory. Only graph code, graph tests, graph
documentation, small graph reports and a separate graph requirements file were added.
