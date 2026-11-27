# One paired preliminary evaluation — Artem's revised scope

Status: accepted user direction · Date: 2026-11-27 · Owner: Artem

## Problem Statement
The immediate priority is an honest paired comparison, not a learned router or an unreviewed full 200-question run. Existing baselines have 24 development and 80 preliminary attempts, while graph answers and the four-system preliminary comparison remain missing. The user's staff feedback replaces the previous execution priority.

## Solution
Verify the frozen question IDs, corpus, generator, prompt and context budget against the delivered baseline experiment. Save graph-local and graph-global attempts for the same twelve development and forty preliminary questions: 104 graph attempts total, with contexts, provenance, timings, token counts and explicit failures. Supply measurements and blinded review batches to Safina; the team agrees failure/N/A policy and grades before deciding readiness for the 200-question run. Use the user's Kaggle for a real T4 run, with private artifacts and recorded hardware, never relabeling 5080 measurements.

## User Stories
1. As Artem, I want both graph retrievers to answer exactly the baseline question IDs under identical shared settings, so that the comparison is paired rather than confounded.
2. As a reviewer, I want actual contexts, source links, answers and explicit failures, so that every correctness label and failure diagnosis traces to a saved attempt.
3. As Safina, I want blinded graph development/preliminary handoffs and separate coverage, correctness, context-size, indexing-cost and query-cost records, so that I can assemble an honest per-question/per-type comparison.
4. As a teammate, I want real T4 hardware/version/log receipts and immutable experiment identities, so that GPU feasibility claims are reproducible.
5. As the team, we want unresolved grades, failure policy and missing baseline artifacts to remain explicit, so that no unavailable score is silently fabricated or used to train routing.

## Implementation Decisions
- Reuse baseline_rag.corpus.prepare and baseline_rag.run.run, not a new answer pipeline. The local configuration changes only artifact locations; shared frozen experiment fields must equal the recorded baseline.
- Exact shared settings: original 15,000 documents / 45,142 chunks, retrieval k=10, context=2560 answer-model tokens, prompt input cap=4096, output cap=384, seed=42; Qwen2.5-3B-Instruct tokenizer revision aa8e72537993ba99e69dfaafa59ed015b17504d1; Ollama 0.18.0, qwen2.5:3b Q4_K_M digest 357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b; prompt/decoding verbatim from the baseline configuration.
- Both retrievers use exactly the existing development/preliminary IDs, not a newly selected sample. Gold answers/facts are absent from inference inputs. The twelve development and original evaluation folds stay unchanged.
- export_graph_development.export_development reuses preliminary_grading.export for blinded development fact-review forms with blank labels and immutable run identities. It must not stand in for human grades.
- report_graph_delivery.build_report consumes saved artifacts only, keeps metrics separate and counts extraction/summary retries once. Historical incomplete session walls and missing usage remain explicit. Optional fold outputs are not required by this revised scope.
- Keep generated claims distinct from raw-source evidence and coverage distinct from correctness. No aggregate correctness winner is reported before an agreed failure/N/A policy and paired labels.
- Kaggle runs must be private, explicitly request NvidiaTeslaT4, assert the actual GPU name and save a separate hardware/config receipt. Query-time use of the prebuilt graph is not proof of constructing the graph on T4. No competition submission is requested.

## Testing Decisions
Existing unittest contracts cover shared generation, label-free inputs, resumption, context packing, provenance and frozen selection. New public seams are export_development(config_path, output) and build_report(config_path, extraction_path); tests use synthetic data only for software validation, not reported benchmark scores. Acceptance is 24 development + 80 preliminary graph attempts saved with explicit statuses and preserved run/configuration hashes, with real tokenizer budgets checked. Correctness is satisfied atomic gold facts / total facts after independent grading; no minimum answer-accuracy threshold or learned-router claim is invented.

## Out of Scope
Learned routing, router fitting, optional depth ablation and the full 200-question core run are deferred until paired correctness data exist and the team reviews failures/readiness. The earlier saved-output routing evaluator work is deferred, not presented as a trained router or a priority deliverable. Safina owns aggregate scoring/failure-policy agreement and paired analysis; Karina owns the baseline delivery/shared-generation settings. No fabricated author approval, human grading, signed AI-use statement or unmeasured T4 construction claim.

## Open Questions
- Kaggle authentication is resolved: account-specific access for m1r0tvorxc works and the quota check shows29.97 GPU hours remaining. The authorized private smoke and104-attempt T4 run remain to execute; no GPU result is inferred from API access.
- Baseline transfer is resolved: the user supplied the immutable baseline_results_15000 archive. All 264 payload hashes, 104 answers, and exact frozen experiment/query parity passed verification. No baseline rerun or extra delivery is currently requested from Karina.
- Safina/team: agree aggregate handling of failed and N/A answers, coordinate grading of all 160 preliminary answers plus 24 graph development answers and adjudicate disagreements.
- Whole team: review paired failures and agree readiness before any full 200-question execution. Corpus-global development coverage remains a future routing issue, not a reason to alter this preliminary sample.

## Further Notes
The user's 27 November staff-feedback image gives six ordered handoffs: verify parity; generate 104 graph attempts; agree failure/N/A policy and blinded batches; grade 160 preliminary +24 graph development answers; export separate paired metrics; review readiness. The existing weekly project plan remains historical background, with its immediate next-action section updated to this priority. See the revised paired-preliminary execution plan and append-only experiment log for exact commands/receipts.
