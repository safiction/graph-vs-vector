# Paired preliminary delivery — revised execution plan

Authority: `docs/specs/2026-11-27-paired-preliminary.md` (user staff feedback, 27 November). This supersedes the earlier broad completion plan. Integration baseline remains `6b6455eb66cb37c7bdb91871e89ea5f8b5ed088d` on `artem/complete-graph-delivery`.

## Architecture and constraints
Reuse the existing complete graph and the merged shared generator. Produce only 24 development and 80 preliminary graph attempts, not the core folds. Keep identical baseline IDs, 15k/45142 corpus hashes, k=10, 2560 context tokens, exact prompt/model/digest/tokenizer/decoding. No learned-router training. Keep coverage/correctness/context-size/index-cost/query-cost separate. Preserve failures and do not invent a failure aggregation policy.

Dependency summary: P1 → P2 → P3 → P4. P1–P3 completed: the user explicitly replaced the T4 requirement with RTX5080;104 graph attempts and the complete grading packet are verified. P4 publishes the checked branch. The scored comparison still waits on independent grading/failure-policy agreement.

## P1 — verify parity and pinned runtime
Files: `configs/graph_local_15000.json`, `requirements-graph-answers.txt`, append-only experiment log; local-only model/runtime assets. Public interface: `baseline_rag.corpus.prepare` and `SharedGenerator`.
Steps: assert every frozen experiment field equals baseline; verify corpus/index/source audit hashes; install exact Ollama 0.18.0 and pinned Qwen2.5:3b; verify actual GPU residency; record hardware. CPU Torch is only for tokenizer tensor materialization, never model inference.
Verify: `PYTHONPATH=src data/.venv-graph-answers/bin/python -m baseline_rag --config configs/graph_local_15000.json prepare` exits 0 with 15000/45142 and no raw-source misses. Full suite exits 0. Commit: `chore(graph): freeze paired preliminary execution`.

## P2 — actual RTX5080 and exactly104 graph attempts — complete
Files: original local graph outputs, hardware/smoke receipts and append-only experiment log. The user's later direct instruction authorizes RTX5080 instead of T4. Historical private Kaggle failures remain separate.
Execution: pinned Ollama0.18.0/Qwen2.5-3B shared adapter passed a real GPU-resident smoke, then ran only graph-local/global on development and preliminary with unchanged frozen settings. No baseline inference was repeated and no src inference code changed.
Verified:24 development +80 preliminary attempts,104 successful calls,0 technical failures,6 output-cap flags. All104 actual contexts/full prompts were independently recounted; maximum2558/2683, within2560/4096. Actual hardware is RTX5080, not T4; no T4 model/construction claim.

## P3 — grading handoff and separate measured costs
Files: `scripts/export_graph_development.py`, `tests/test_graph_development_export.py`, `scripts/report_graph_delivery.py`, `tests/test_graph_delivery_report.py`, `reports/graph/completion.json` and status docs. Public seams: `export_development(config_path, output)` and `build_report(config_path, extraction_path)`.
Steps: red/green tests for blinding, split/pair validation, overwrite protection and failures; export all graph development successes with blank human labels and explicit failed rows; aggregate saved extraction/summary/query costs including retries; validate real artifacts and budgets; provide Safina all preliminary outputs for her full-160 grading batches. Preserve correctness as pending until external labels arrive.
Verify: `PYTHONPATH=src data/.venv-graph-answers/bin/python -m unittest discover -s tests -p 'test_graph_*export.py' -v`; `PYTHONPATH=src data/.venv-graph-answers/bin/python -m unittest discover -s tests -p test_graph_delivery_report.py -v`; run real report command; observe exit0 and actual counts. Missing folds are deferred, not failures or blockers in this scope. Commit: `feat(graph): export auditable paired handoff and actual costs`.

## P4 — independent review and readiness handoff
Read final diff, run two fresh reviewers (standards/spec), resolve Critical/Important findings with tests; run full suite and compile checks; update plan/status to exact observed outcomes. Push only verified integration branch, not main. No claim the four-system comparison is complete without Karina's original package and Safina's paired correctness/policy. The whole team, not the assistant alone, decides readiness for the full200 run.
Verify: `PYTHONPATH=src data/.venv-graph-answers/bin/python -m unittest discover -s tests -v`; `data/.venv-graph-answers/bin/python -m compileall -q src scripts tests`; review no open Critical/Important; clean tracked Git status. Commit: `docs(graph): finalize verified preliminary handoff and blockers`.

## Current external receipts
- Compute complete on user-authorized RTX5080. Private Kaggle attempts stopped before answer generation; their input/zstd/virtualenv failures are historical and no longer delivery blockers.
- Baseline blocker resolved: user-provided archive e19bfbc170a207a7271c241c84bd3a551e29009c9ba1a137306a1c0d4cf6b2e2 unpacked privately under data/packages/baseline_results_15000. Verification passed all264 hashes,104 answers and exact frozen experiment/development/preliminary query equality.
- Human correctness and aggregate failed/N/A policy remain Safina/team handoffs. Corpus-global development coverage is deferred with routing, not a blocker to this fixed preliminary sample.
