# EnterpriseRAG-Bench: initial data audit

## Snapshot and licensing

- Repository: https://github.com/onyx-dot-app/EnterpriseRAG-Bench
- The repository's `LICENSE` is MIT, copyright 2026 DanswerAI, Inc. It requires inclusion of its copyright and permission notice in copies or substantial portions. Preserve that notice when distributing copied dataset material. The README labels it a *code* license; verify the license for any separately downloaded release or Hugging Face data before redistributing it. No legal conclusion about those separate distributions is implied.
- The corpus is synthetic. The upstream repository warns against including benchmark data in training corpora.

## Question schema and observed data

The records in the core `questions.jsonl` have fields:

| Field | Meaning / project use |
| --- | --- |
| `question_id` | Original `qst_...` identifier; 500 distinct IDs in the core file. |
| `question_type` | One of the ten original dataset categories. Preserve it alongside our 4 project types. |
| `source_types` | Source categories associated with the question; descriptive metadata. |
| `question` | Question text supplied to the systems. |
| `expected_doc_ids` | Gold document identifiers (`dsid_...`). |
| `gold_answer` | Reference answer. |
| `answer_facts` | List of pre-existing atomic facts for fact-level grading. Do not assume that they are independently human-verified. |

All 500 core records have a nonempty gold answer and at least one answer fact. `extra_questions.jsonl` contains 100 metadata questions, also numbered from `qst_0001`: IDs are therefore not globally unique across these two files. We will use the core file only.

| Original category | Actual core count | Subset we will use |
| --- | ---: | --- |
| `basic` | 175 | 50 single-fact |
| `intra_document_reasoning` | 40 | 25 multi-hop |
| `project_related` | 40 | 25 multi-hop |
| `completeness` | 20 | 20 corpus-global |
| `high_level` | 10 | 10 corpus-global |
| `conflicting_info` | 20 | 20 corpus-global |
| `semantic` | 125 | 30 fuzzy |
| `constrained` | 30 | 20 fuzzy |

Upstream `quickstart.md` has different counts for several categories. Use the pinned question file for reproducibility. The ten High Level records have empty gold document lists by design, so document coverage is N/A for them; they still have reference answers and facts.

The revised allocation places Project Related only in multi-hop, rather than splitting it as in the submitted proposal. Intra-document reasoning involves evidence from a single document; the multi-hop project label therefore does not imply every question needs cross-document hops. Conflicting Info is assigned to corpus-global, although reconciling two sources does not necessarily require corpus-wide synthesis. Results should also be reported by original category.

All Completeness, High Level, and Conflicting Info questions are consumed by the revised selection. A disjoint development set cannot include those original categories; 15 Project Related questions remain available for development.

## Corpus and provenance

The source tree contains JSON documents under `generated_data/sources/` and `generated_data/uuid_index.json`, which maps `dsid_...` IDs to source-relative paths. The index has 511,958 entries and 511,958 distinct paths. The filesystem has 511,962 JSON files; every indexed path exists, and the four additional paths each reuse an indexed ID for a document with different text. These ID collisions require an explicit corpus and provenance decision before freezing the full corpus.

One collision affects selected Conflicting Info question `qst_0413`: both entries in `expected_doc_ids` are the same ID, `dsid_6df52fdb96ae4edcb76464738bca3340`, yet the indexed `jira/misc-requests/INT-7832-cleanup-old-test-instances.json` and unindexed `jira/internal-support/INT-7832-cleanup-stale-test-instances-and-archived-volumes-access.json` contain different versions. The gold answer refers to the indexed 72-hour version; the other version mentions updated five-business-day guidance and explicit infra-manager approval. A set-based coverage score cannot distinguish these documents by the shared upstream ID. The question remains in correctness evaluation, with document coverage marked N/A as recorded in `manifests/coverage_exclusions.json`. The pilot uses the index and includes none of the four colliding IDs; the broader full-corpus collision policy remains unresolved.

Across all 500 core questions there are 722 distinct expected document IDs. All 722 are in the UUID index, point to existing files, and match each file's `dataset_doc_uuid`; the collision above shows that an ID match does not guarantee uniqueness across the source tree. The question and pilot selections are recorded in `manifests/`; the final experiment corpus has not yet been selected.

The completed full source scan found 1,300 groups of exactly matching rendered title and content, containing 2,612 document records with distinct IDs (1,288 pairs and 12 triples). It reported four duplicated IDs and four index mismatches, corresponding to the four collisions above, and no invalid documents or missing indexed paths. Identical rendered content does not imply identical metadata or permission to discard a document; no deduplication has been applied. These counts concern exact text matches, not near-duplicate or semantic similarity. Detailed reports are in Git-ignored `data/audit/`.

Each raw document is a JSON object with source-specific fields. It identifies its title field using `title_field_name`, its content fields using `content_field_names`, and its original ID using `dataset_doc_uuid`. Fields can vary by source. The upstream exporter `src/scripts/data_gen_stage_4_data_export/export_data.py`, `convert_to_text`, renders the title followed by the listed content fields, joining list values by newlines. Released `.txt` exports are a different representation; do not assume raw JSON and exported text have identical bytes. Pick, document, and consistently use one representation before chunking.

The proposal specifies 512-token chunks and a 5 × 512-token context budget. The shared pilot uses the pinned BGE tokenizer with no overlap; the answer-generator tokenizer remains to be selected. Chunk IDs must be tied to the corpus and chunking configuration, and all systems must use the same chunks and source IDs.

## Evaluation implications and open work

- Use the provided `answer_facts` as the starting grading rubric and review their correctness; do not rewrite them silently. The proposal defines correctness as the fraction of gold facts stated and full correctness as all facts stated.
- Coverage compares unique gold `expected_doc_ids` with source document IDs represented in the **final supplied context**, including documented provenance for graph entities, relations, and summaries. The proposal also requires mean number of distinct supplied source documents, which reveals when broad graph summaries inflate provenance coverage. An ID on a summary is provenance, not proof that the underlying text itself was supplied.
- The benchmark's own evaluator can revise gold documents and answers using multiple judges. The proposal instead requires a frozen sample and fact-based comparison; document any decision to use upstream corrections rather than changing gold mid-comparison.
- The proposed fallback is one common fixed sub-corpus containing gold documents of the 200 questions, their top BM25 neighbours, and random documents from all sources. This is conditional on the 5,000-document extraction pilot and budget assessment. Do not treat the fallback as approved or quietly build a gold-only corpus. A sub-corpus selected using gold IDs also changes the task and should be disclosed.
- The final corpus must be selected after the graph-cost pilot and team budget decision; the collision and duplicate-content policy should be recorded for that corpus.
