# Final baselines and teammate reproduction

As of 28 November 2026, baseline/shared-generation integration is complete.
The 104 baseline and 104 graph attempts form the paired preliminary/dev delivery.
Preserve these outputs and the 24 adjudicated baseline development answers.
Independent grading, failure/N/A policy and a team readiness decision remain
required before full evaluation. No final inference or teammate reproduction
is claimed by adding these instructions.

## Obtain the existing inputs

Ask Karina for the private baseline artifacts/indexes and portable corpus;
ask Artem for the graph/grading delivery. No public payload URL is recorded.
A Git clone alone is insufficient. Use these paths from the repository root:

| Destination | Existing payload |
| --- | --- |
| `data/corpus_15000/` | Verified corpus/chunks/questions; see [package verification](PORTABLE_CORPUS_15000.md) |
| `data/baselines/corpus_15000/experiment.json` | Original frozen experiment |
| `data/baselines/corpus_15000/questions/selection.json` | Original selection/input hashes |
| `data/baselines/corpus_15000/indexes/` | Complete BM25 and vector directories including metadata/settings |
| `data/baselines/corpus_15000/runs/` | Original predictions for review/demo |
| `data/models/` | Pinned models/tokenizer, copied or downloaded with existing setup |

The compact result ZIP excludes index binaries: transfer the original index
directories separately. Do not rebuild them to inspect saved results. Verify
archive checksums and preserve the corpus package's exact files.

`configs/baseline_final_200.json` retains the original generator, embedding,
prompt, seed, k=10 and 2560-token context cap. It reads original indexes and
writes separately to `data/baselines/final_200/`. External `index_root` is
read-only to the runner; `build`/`all` are rejected. Use `prepare`/`run`.
For another machine, make a local config copy changing filesystem paths only.
Keep the server endpoint identical to the frozen experiment; changing generation
settings needs a separately agreed experiment, never an edited historical receipt.

## Preflight and saved-output demo — available now

Create an environment using [baseline setup](VECTOR_RAG.md). Exact observed
dependencies are in `reports/baselines/requirements-lock.txt`. On Linux replace
`Scripts/python.exe` with `bin/python`. Run from the repository root:

```powershell
data/.venv-baselines/Scripts/python.exe scripts/report_final_baselines.py --preflight
data/.venv-baselines/Scripts/python.exe scripts/demo_baseline.py --question-id qst_0025
```

Preflight validates original corpus/configuration, 200 evaluation IDs and input
hashes, split balance/disjointness and existing index checksums. It writes
nothing and needs no answer server/GPU; it is not an inference feasibility or
team readiness decision. The demo displays a saved development answer, actual
context, sources and finish reason. Use `--predictions PATH` for another JSONL,
or `--json` for the original record and prediction-file SHA256.

## Reproduction by another participant

Another participant creates a clean environment, obtains/verifies the inputs,
runs preflight and inspects the demo. To verify actual inference, copy the final
config to `data/reproduction.json` and change its `output` to a fresh directory
such as `data/baselines/reproduction_YYYYMMDD/`. Provision pinned model assets
with the setup/download/pull commands first if necessary. Then:

```powershell
data/.venv-baselines/Scripts/python.exe scripts/manage_baseline_ollama.py start
data/.venv-baselines/Scripts/python.exe scripts/manage_baseline_ollama.py status
data/.venv-baselines/Scripts/python.exe scripts/run_baselines.py --config data/reproduction.json prepare
data/.venv-baselines/Scripts/python.exe scripts/run_baselines.py --config data/reproduction.json run --retrievers bm25 vector --splits development
data/.venv-baselines/Scripts/python.exe scripts/manage_baseline_ollama.py stop
```

This development-only reproduction can precede the full-evaluation gate;
preserve separate outputs and the original graded answers. Record participant,
date, commit, commands/config, input hashes, Python/dependencies, physical
GPU/driver, server/model identity, results and deviations in a reproduction
receipt. Preserve raw logs privately. Preflight/fixture tests alone are not
full reproduction; cross-hardware word-for-word answer equality is not promised.

## Full evaluation — after team readiness

Complete preliminary grading/adjudication and record team policy/readiness first.
Keep the original question IDs/settings. The two 100-question folds include the
40 preliminary questions: the new full run evaluates them in a separate output
rather than mixing historical/new execution costs. It preserves all originals.

```powershell
data/.venv-baselines/Scripts/python.exe scripts/report_final_baselines.py --preflight
data/.venv-baselines/Scripts/python.exe scripts/manage_baseline_ollama.py start
data/.venv-baselines/Scripts/python.exe scripts/run_baselines.py --config configs/baseline_final_200.json prepare
data/.venv-baselines/Scripts/python.exe scripts/run_baselines.py --config configs/baseline_final_200.json run --retrievers bm25 vector --splits fold_a fold_b
data/.venv-baselines/Scripts/python.exe scripts/report_final_baselines.py
data/.venv-baselines/Scripts/python.exe scripts/manage_baseline_ollama.py stop
```

Unchanged `run` resumes saved attempts; failures remain explicit and cause the
runner to exit nonzero. Do not selectively rerun capped/wrong answers. Agree a
technical retry policy before `--retry-errors`. Code changes invalidate resume:
use a new output/config when changing code or experimental settings.

## Audit and handoff

The final auditor uses only the locally cached pinned tokenizer, with no server
contact or answer-model loading. It validates both folds/systems against the
reference experiment, question texts/hashes, index identities and checkpoints;
recounts context/full-prompt tokens; checks source text/provenance, timing
components, output budgets/cap flags and pinned Ollama model GPU residency.

Outputs: `reports/baselines/final_200.json` and
`reports/baselines/final_200.per_question.csv`. Use `--output PATH.json` to change
their location; inputs/experiment directories are protected. Missing/corrupt
attempts produce nonzero exit. All 400 attempts may be saved even with explicit
failures: `all_attempts_saved` differs from `all_answers_successful`. Missing
top-level inputs fail before writing a new report; an old report does not prove
that a subsequent audit passed. The legacy 104-answer audit remains unchanged.

Coverage, actual context size, query timings and reused indexing receipts are
separate from correctness, which stays null until independent grading. Failed
calls/no-gold IDs retain N/A coverage. Indexing receipts describe the historical
build, not new indexing work. Record physical hardware and warm/cold state;
historical baseline and RTX 5080 graph times do not establish an architecture
speedup under controlled conditions.

Give predictions/measurements to Safina for grading, adjudication and paired
analysis/uncertainty. Karina writes baseline/shared-pipeline/cost methods and
limitations, then final report/slides after results are approved. Optional
k=3/5/10 experiments wait until the main evaluation is complete.
