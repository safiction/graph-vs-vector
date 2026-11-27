# Paired preliminary delivery — revised execution plan

Authority: `docs/specs/2026-11-27-paired-preliminary.md` (user staff feedback, 27 November). This supersedes the earlier broad completion plan. Integration baseline remains `6b6455eb66cb37c7bdb91871e89ea5f8b5ed088d` on `artem/complete-graph-delivery`.

## Architecture and constraints
Reuse the existing complete graph and the merged shared generator. Produce only 24 development and 80 preliminary graph attempts, not the core folds. Keep identical baseline IDs, 15k/45142 corpus hashes, k=10, 2560 context tokens, exact prompt/model/digest/tokenizer/decoding. No learned-router training. Keep coverage/correctness/context-size/index-cost/query-cost separate. Preserve failures and do not invent a failure aggregation policy.

Dependency summary: P1 → P2 → P3 → P4. Software tests/reporting run independently of private T4 preparation. Authentication and baseline delivery are resolved; execution now waits on the real private T4 smoke/run, and the scored comparison waits on independent grading/failure-policy agreement, not invented data.

## P1 — verify parity and pinned runtime
Files: `configs/graph_local_15000.json`, `requirements-graph-answers.txt`, append-only experiment log; local-only model/runtime assets. Public interface: `baseline_rag.corpus.prepare` and `SharedGenerator`.
Steps: assert every frozen experiment field equals baseline; verify corpus/index/source audit hashes; install exact Ollama 0.18.0 and pinned Qwen2.5:3b; verify actual GPU residency; record hardware. CPU Torch is only for tokenizer tensor materialization, never model inference.
Verify: `PYTHONPATH=src data/.venv-graph-answers/bin/python -m baseline_rag --config configs/graph_local_15000.json prepare` exits 0 with 15000/45142 and no raw-source misses. Full suite exits 0. Commit: `chore(graph): freeze paired preliminary execution`.

## P2 — real T4 and exactly 104 graph attempts
Files: local-only private Kaggle input/kernel payload and separate T4 outputs; graph run outputs; experiment log. No source changes during inference.
Steps: authenticate Kaggle securely; inspect private account capability/quota; upload only project inputs as a private dataset; push a private `NvidiaTeslaT4` kernel with bounded timeout; assert actual T4 before inference; run unchanged development then preliminary only; collect notebook status/log/output and configuration hashes. First smoke timeout 1200 seconds; enlarge only within the user's authorized run and documented budget if a successful smoke establishes need. Do not run baseline inference just to reconstruct existing outputs.
Verify: saved graph attempts=24 development and80 preliminary; exact question-ID pairing; actual contexts, sources, statuses/errors, tokens and timing; T4 name in saved hardware receipt. Query-time use of a prebuilt graph is not T4 graph construction. Commit: `docs(graph): record paired graph run receipts`.

## P3 — grading handoff and separate measured costs
Files: `scripts/export_graph_development.py`, `tests/test_graph_development_export.py`, `scripts/report_graph_delivery.py`, `tests/test_graph_delivery_report.py`, `reports/graph/completion.json` and status docs. Public seams: `export_development(config_path, output)` and `build_report(config_path, extraction_path)`.
Steps: red/green tests for blinding, split/pair validation, overwrite protection and failures; export all graph development successes with blank human labels and explicit failed rows; aggregate saved extraction/summary/query costs including retries; validate real artifacts and budgets; provide Safina all preliminary outputs for her full-160 grading batches. Preserve correctness as pending until external labels arrive.
Verify: `PYTHONPATH=src data/.venv-graph-answers/bin/python -m unittest discover -s tests -p 'test_graph_*export.py' -v`; `PYTHONPATH=src data/.venv-graph-answers/bin/python -m unittest discover -s tests -p test_graph_delivery_report.py -v`; run real report command; observe exit0 and actual counts. Missing folds are deferred, not failures or blockers in this scope. Commit: `feat(graph): export auditable paired handoff and actual costs`.

## P4 — independent review and readiness handoff
Read final diff, run two fresh reviewers (standards/spec), resolve Critical/Important findings with tests; run full suite and compile checks; update plan/status to exact observed outcomes. Push only verified integration branch, not main. No claim the four-system comparison is complete without Karina's original package and Safina's paired correctness/policy. The whole team, not the assistant alone, decides readiness for the full200 run.
Verify: `PYTHONPATH=src data/.venv-graph-answers/bin/python -m unittest discover -s tests -v`; `data/.venv-graph-answers/bin/python -m compileall -q src scripts tests`; review no open Critical/Important; clean tracked Git status. Commit: `docs(graph): finalize verified preliminary handoff and blockers`.

## Current external receipts
- Kaggle authentication resolved during preparation: account-specific API access for m1r0tvorxc succeeds; `kaggle quota` reports29.97 GPU hours remaining. Upload/run must stay private and request actual NvidiaTeslaT4; smoke timeout1200 seconds.
- Baseline blocker resolved: user-provided archive e19bfbc170a207a7271c241c84bd3a551e29009c9ba1a137306a1c0d4cf6b2e2 unpacked privately under data/packages/baseline_results_15000. Verification passed all264 hashes,104 answers and exact frozen experiment/development/preliminary query equality.
- Human correctness and aggregate failed/N/A policy remain Safina/team handoffs. Corpus-global development coverage is deferred with routing, not a blocker to this fixed preliminary sample.
