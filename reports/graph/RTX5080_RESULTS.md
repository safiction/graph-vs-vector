# Paired graph delivery — RTX5080

## Execution and parity
**104/104 graph attempts saved;104 successful calls,0 technical failures:**12 development +40 preliminary questions for each of graph-local/global. Actual hardware: NVIDIA GeForce RTX5080, not T4. Both retrievers use the original shared generator and identical baseline IDs, corpus, prompt and budgets. Frozen experiment SHA256:`3c6340bfa640523840e3962d217711afef428468b73bb4fdde63315836b095a6`.

Independent recount of all104 saved contexts/prompts passed: maximum2558 context and2683 complete-prompt units, within2560/4096. Output cap384 is unchanged. Six graph preliminary answers hit it; eight original baseline preliminary answers also hit it. All14 remain in the grading data. See [recount receipt](retokenization_rtx5080.json) and [complete artifact audit](completion.json).

## Preliminary measurements — not correctness
All four methods answer the same40 IDs. Document coverage is applicable to38 questions per method; two N/A cases are excluded explicitly from coverage denominators, not assigned zero. Context averages use all40 answers.

| Retriever | Attempts | Mean document coverage | Full document coverage | Mean context tokens | Mean query seconds | Output-cap hits |
|---|---:|---:|---:|---:|---:|---:|
| BM25 |40|0.828947|0.736842|2296.250|2.908436|5|
| Vector |40|0.708333|0.631579|2346.000|2.603261|3|
| Graph-local |40|0.552266|0.394737|2383.650|2.050845|4|
| Graph-global |40|0.741959|0.631579|1807.125|1.607159|2|

**Factual correctness is pending independent human grading; no accuracy winner is claimed.** Raw latency is an observation, not a controlled architecture speedup: baseline Windows/Python3.12.3 and graph WSL/Python3.12.16 stacks differ, and matching baseline physical GPU/warm state is not established. Baseline cached-resumption invocation walls are not historical full generation time.

[Per-question CSV](paired_per_question.csv) contains208 records across development/preliminary:104 original baseline +104 graph. It keeps coverage, context size, prompt/completion usage and retrieval/packing/generation/amortized-embedding/query times distinct. [Summary CSV](paired_summary.csv) and [provenance/limitations](paired_measurements.json) preserve the source run identities.

## Indexing and query costs stay separate
- Original BM25 index elapsed17.865570s; vector/BGE index elapsed413.580754s, per supplied baseline receipts. These are not answer-generation timings.
- Graph-only assembly elapsed15.947479s excludes model extraction/summaries. Recorded extraction+summary usage:36,292,624 prompt +23,965,281 completion units across147,848 retained attempts. The351,462.716s summed request wall overlaps across workers and is **not** historical elapsed/GPU hours.
- Historical total extraction wall, GPU hours and electricity are unknown. All retained retries are counted once; recovery attempts are a subset of retries.
- Graph target-run invocation walls:71.606709s development +168.345535s preliminary. The separate synthetic warm-up39-output-token call/29.602334s is not one of104 target attempts.

## Human handoff
The supplied private/local ZIP contains104 raw graph attempts with contexts/provenance, original baseline ZIP, separate measurement receipts and blank blinded forms:160 preliminary answers/1296 fact rows plus24 graph-development answers/68 fact rows per reviewer. Keep private system keys with the coordinator. Existing24 baseline development grades are not repeated. The legacy sampled-development scorer's hardcoded scope text is not a final full-preliminary report; Safina owns the final paired aggregation/policy.

Archive: `paired_graph_rtx5080_20261127.zip`,8,542,620 bytes, SHA256 `dfb5b4a965628056e2667afbb287ce421378c8e2848159297237c39e2d267620`;261 checksum-listed files verified. No model weights or bulk graph index are included. No core200, learned router or optional ablation was run. Team readiness review follows paired grading, not successful API-call counts.
