# Hand-worked document coverage check

Use development question `qst_0344` (Project Related). Its gold source IDs are:

1. `dsid_e33c122264214846a8b7df7701d31482`
2. `dsid_409618a3958944f9b78362ed7e01bb33`
3. `dsid_a5a5235d973e421c870e11ceea4887ed`
4. `dsid_4b2ac6984bdf4737a18520e158b04453`

For an illustrative final context containing two chunks from the first gold document, one from the second, and one from another document, the distinct source IDs are `{dsid_e33c122264214846a8b7df7701d31482, dsid_409618a3958944f9b78362ed7e01bb33, dsid_other}`. Two of four distinct gold sources are represented: **coverage = 2/4 = 0.5**, **full coverage = false**, and **source count = 3**. This is a constructed example, not an actual retrieval result.

For High Level questions with no gold document IDs, and for `qst_0413` per the documented ID collision, coverage and full coverage are N/A. A successful empty context for a question with nonempty gold IDs has zero coverage. A failed run is recorded separately with N/A coverage, not treated as an empty successful context.

Use source IDs from the final context after packing and truncation. For graph summaries, record the IDs of underlying supporting documents as provenance; this does not mean their original text was supplied. Mean coverage and full-coverage rate use applicable questions as the denominator; report the number applicable, N/A cases, and failures. Report mean distinct source count separately to expose broad provenance. Document coverage does not prove that the relevant passage or fact was present.

`src/coverage.py` expects JSONL context records with `question_id`, `status` (`success` or `failed`), and, for successful runs, `final_source_document_ids`. It requires one record per question, including failed attempts. Its input is a proposed interface to agree with the shared runner; it must reflect the actual final supplied context.

For one retriever and one question file at a time, once context records are available:

```powershell
python src/coverage.py --questions data/questions/development.jsonl --contexts data/runs/example_contexts.jsonl --output data/results/example_coverage.json
```

This output includes per-question results and per-type aggregates. A missing or repeated question record, missing provenance on a successful run, or an existing output file causes an error; failures must be explicit. No context records or result files are created by this example command until the shared runner supplies them.
