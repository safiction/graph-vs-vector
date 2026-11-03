# Project Plan

## 1. Project scope

**Project:** GraphRAG vs Vector RAG, Refereed Properly.

Use EnterpriseRAG-Bench with 200 questions, 50 per type. Compare BM25, vector RAG, GraphRAG-local, and GraphRAG-global on the same frozen corpus. If the full corpus is too expensive, use one fixed sub-corpus for all systems and report this limitation.

## 2. Experiments and evaluation

- Keep chunking, answer generator, prompt, and context budget consistent across systems.
- Measure supporting-fact coverage, answer correctness, indexing cost, and query latency. Mark coverage as N/A when gold documents are unavailable.
- Check the local judge against human grading; use manual grading where it is unreliable.
- Choose routing rules on one question fold and evaluate on the other, then swap. Report results by question type, uncertainty, and representative failures.
- After the core comparison, test graph expansion depth and retrieval size if time permits. A negative result is acceptable.

## 3. Team responsibilities

### Safina — Role 1: Data and evaluation

- **Prepare the corpus:** check the dataset release and license, inspect missing/duplicate documents, preserve source document IDs, and create the shared chunk file. Save the exact document list so every retriever uses the same data.
- **Prepare the questions:** select 200 questions, 50 per type; save question IDs, type labels, reference answers, atomic answer facts, and gold document IDs. Create two balanced routing folds and a separate small development set for debugging.
- **Implement coverage:** compare gold document IDs with the source IDs represented in the final supplied context. Report average coverage and the fraction of questions with all gold documents covered; mark questions without gold documents N/A.
- **Coordinate correctness grading:** write a fact-level grading rubric, export answers for blinded review, compare the local judge with two human graders, and organise manual grading where needed. All members help grade; Safina owns the combined labels and scoring code.
- **Analyse results:** produce per-type comparisons, confidence intervals, and failure examples. Distinguish retrieval failures from cases where the generator had evidence but answered incorrectly.

**Main outputs:** corpus/question manifests, evaluation script, grading file, results tables, and evaluation/report sections.

### Karina — Role 2: Vector RAG and shared pipeline

- **Implement BM25:** build a lexical index over the shared chunks and return ranked chunks with source IDs and scores.
- **Implement vector retrieval:** embed the same chunks with a fixed BGE model, build the vector index, and retrieve chunks by similarity. Record the model version and retrieval settings.
- **Build shared generation:** accept evidence from any of the four retrievers, pack it into the same context budget, and call the same local answer model with the same prompt and decoding settings. Save the actual context and answer.
- **Integrate and log runs:** provide a common command for choosing a retriever and question set; cache outputs and record latency, token use, errors, and model/configuration versions.
- **Maintain reproducibility:** pin dependencies, document setup and commands, and check that another member can regenerate the main results.

**Main outputs:** BM25/vector indexes and retrievers, shared generation runner, baseline predictions, runtime logs, README, and baseline/pipeline report sections.

### Artem — Role 3: GraphRAG and routing

- **Build the graph:** extract entities and relations from shared chunks, merge equivalent entity names, and retain source chunk/document IDs for every extracted item. Save checkpoints so interrupted extraction can resume.
- **Implement local retrieval:** link question entities to graph nodes, expand their neighbourhood, and return relevant supporting evidence through Karina's shared interface.
- **Implement global retrieval:** detect Leiden communities, create summaries linked to their supporting sources, and retrieve summaries for broad questions. Supplied evidence must obey the same context budget as the baselines.
- **Implement routing:** use Safina's scores to choose a retriever for each question type on one fold and evaluate that choice on the other. Compare routes using known types and predicted types; report classification errors and routing overhead.
- **Manage graph compute:** measure extraction/indexing time and token use, estimate full-run cost, and organise GPU jobs and checkpoints.

**Main outputs:** graph and summaries with provenance, local/global retrievers, routing code/results, indexing-cost logs, and graph/routing report sections.

**Compute:** GPU is the main resource for extraction and generation. Each member prepares runnable code for their own component; GPU access does not change role ownership. Agree a compute budget after an early pilot.

**Shared work:** everyone reviews another member's work, helps grade answers, analyses failures, and contributes to the report and presentation.

## 4. Weekly deliverables

All dates are in **2026**.

| Deadline | Safina — data and evaluation | Karina — vector RAG and pipeline | Artem — GraphRAG and routing | Integrated deliverable / acceptance check |
| --- | --- | --- | --- | --- |
| **5 Nov — data and working pilot** | Check dataset schema/license; save the 200-question selection, two folds, and separate development questions. Prepare shared chunks with stable source IDs. Write the grading rubric and verify coverage on a hand-worked example. | Build BM25 and BGE retrieval on pilot chunks. Implement a runner that retrieves evidence, applies the common token limit, generates an answer, and saves context/source IDs. | Extract a pilot graph with source links; demonstrate entity linking and neighbourhood retrieval. Measure extraction cost and estimate whether the full corpus fits the agreed budget. | **End-to-end pilot:** BM25, vector, and graph-local produce saved answers on the same development questions. Manually trace one answer per system to source text. Choose full corpus or a bounded corpus; save its manifest and freeze shared settings before comparison runs. |
| **12 Nov — all four systems and preliminary evaluation** | Score the same preliminary set of 40 questions, ten per type. Arrange independent grading of 40 sampled answers by two members; compare judge/human agreement and identify types needing manual grading. Produce the first per-type results table. | Finish both baseline indexes on the frozen corpus. Integrate both graph retrievers into shared generation; verify identical prompts/budgets and log query latency/token use. Run baseline answers for the preliminary set. | Finish local retrieval and community summaries/global retrieval on the frozen corpus. Run both graph systems on the preliminary set. Audit source links and build a routing prototype using development scores only. | **Official midway check:** runnable repo, paired preliminary results for all four systems, and a prioritised blocker list for the 15-minute staff meeting. All 160 preliminary answer attempts are saved, including failures. No evaluated answer is used to tune its own routing decision. |
| **19 Nov — complete core experiments** | Score all core outputs; complete manual grading where calibration requires it, with teammates' help. Check that missing gold support and failed runs are handled consistently. Export per-question and per-type metrics. | Complete BM25/vector answers for all 200 questions; check logs and resolve pipeline failures. Support execution of the shared generation jobs on Artem's GPU and export comparable query-cost summaries. | Complete local/global graph answers for all 200. Build routing tables on opposite folds; evaluate known-type and predicted-type routes. Save chosen retriever, answer score, and routing cost for every held-out question. | **Core results complete:** 800 answer attempts with explicit failure records, coverage/correctness tables by type, and index/query costs. Compare held-out routing with fixed retrievers. Every reported score traces to a saved prediction and grading record. |
| **26 Nov — analysis, ablations, and draft** | Add 95% bootstrap intervals and paired differences. Select failures covering all four question types and explain their causes. Draft data, evaluation, and results text. | Test retrieval size k=3/5/10 while keeping the context cap fixed; save the setting and result of each run. Have another member run the documented reproduction command. Draft baseline/pipeline/cost text. | Compare local graph expansion depth 1 vs 2 with other settings fixed. Explain routing wins, ties, and failures; finalise graph-cost records. Draft graph/routing methods and limitations. | **Reviewable first report:** complete draft, main tables/figures, failure examples, and documented reproduction commands. Include available one-factor ablations; if core work is delayed, reduce optional ablations first and state what was omitted. |
| **3 Dec — submission and presentation** | Cross-check final tables against evaluation files; ensure conclusions match uncertainty and failures. Finalise evaluation/results slides and text. | Finalise pinned environment, setup/download instructions, and main-result command. Check reproduction from a clean environment; assemble slides and a saved-output demo. | Check graph/routing descriptions against code and configs. Finalise methods slides, source-code attribution, contribution paragraph, and AI-use statement with input from everyone. | **Official submission:** six-page NeurIPS-style report, reproducible code, and eight-minute talk plus questions. Freeze content on 1 Dec; check files/links and rehearse on 2 Dec. Each member presents their part and can explain the full comparison. |

## 5. Working agreements

- Save configurations, corpus/question IDs, model versions, predictions, provenance, and costs. Keep large data and indexes outside Git with documented access instructions.
- Use development questions for debugging; keep evaluation questions separate from tuning. Preserve identical comparison conditions.
- If graph construction exceeds the budget, reduce the corpus consistently for every system. Drop optional experiments before sacrificing the core comparison.