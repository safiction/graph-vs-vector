# Shared data preparation

The dataset clone is expected at `../EnterpriseRAG-Bench`. Settings are in `src/config.py`.

The question selection uses seed 42. `manifests/questions.json` records the input checksum and exact IDs for both folds and the development set. Each evaluation fold contains 100 questions, 25 per project type. Development contains four questions from each available type and none from corpus-global, because all of its benchmark categories are used for evaluation.

Generated question records and corpus data are in the Git-ignored `data/` directory. `python src/load_data.py` creates the question files from the original `questions.jsonl` and writes `manifests/questions.json`; it refuses to overwrite existing question files and checks an existing selection manifest for a match. Reference answers and answer facts are for evaluation and are not generator inputs.

The shared chunks use 512 BGE tokens including special tokens and no overlap, with `BAAI/bge-base-en-v1.5` at revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`. The final corpus has not yet been frozen. The shared generation runner will measure its final context with the answer generator's tokenizer. The final corpus and chunk manifest will be published after the pilot and corpus decision.

The 5,000-document pilot uses source-proportional quotas and includes the development questions' gold documents. Its IDs, sampling method, source manifest, and chunk configuration are in `manifests/pilot_document_ids.txt`, `manifests/pilot_selection.json`, `manifests/pilot_corpus.jsonl`, and `manifests/pilot_chunk_config.json`. The generated chunks belong under `data/pilot/`; they are separate from the later final corpus. Install the dependencies from `requirements.txt` and make the pinned tokenizer available locally to run the chunker.

```powershell
python src/prepare_chunks.py --manifest manifests/pilot_corpus.jsonl --output-directory data/pilot/chunks
python src/verify_chunks.py --manifest manifests/pilot_corpus.jsonl --chunks data/pilot/chunks/chunks.jsonl
```

The chunker checks each source file against its manifest and refuses to overwrite existing output. The verifier checks that chunks reconstruct their source text and fit the tokenizer's limit. A failed or interrupted chunking run leaves a partial file and no `chunk_config.json`; preserve or remove that file before retrying.
