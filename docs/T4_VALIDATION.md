# T4 validation handoff — not yet measured

No T4 is attached to the current host. Its only GPU is an RTX 5080. Neither a 16 GB memory estimate nor a successful 5080 run establishes T4 feasibility.

## Keep the two validation claims separate
1. **Consume the completed graph:** load the frozen corpus/index/summaries, run CPU retrieval and the pinned Qwen2.5-3B Q4_K_M answer generator on an actual T4. This demonstrates query-time use of the delivered graph, not construction of the graph on T4.
2. **Rebuild graph extraction/summaries on T4:** run a separately identified development-only pilot and measure extraction/summarization throughput, GPU and host RAM, checkpoint/resume and audit. Do not reuse 5080 checkpoints as T4 measurements or mix new extraction settings into the frozen index.

The [vLLM quantization documentation](https://github.com/vllm-project/vllm/blob/main/docs/features/quantization/llm_compressor/fp8.md) distinguishes native FP8 W8A8 on compute capability >=8.9 from weight-only FP8 W8A16 via Marlin on Turing (>=7.5). A T4 is Turing. The availability of that path is not proof that this exact model architecture, vLLM build and checkpoint work on T4. Validate the actual kernel/backend; record any conversion or fallback as a different execution profile. Do not copy the 5080's bfloat16/backend assumptions blindly.

## Acceptance receipt to collect on the T4 host
- UTC timestamp, Git commit/dirty marker, `nvidia-smi` output explicitly naming **Tesla T4/NVIDIA T4**, driver, VRAM, Python/package lock and server binary hash.
- Original corpus/chunk/index/config hashes, model revision/digest, actual quantization/kernels, dtype, GPU residency and command line.
- For query-time smoke, an unchanged development-only query file and frozen shared prompt/2,560-token budget. Save actual answer, context, final sources, token counts, elapsed wall time, host RAM and GPU peak. No evaluation tuning.
- For construction pilot, use an explicitly separate output directory and model/profile identity. Save selected development/pilot chunk IDs, all extraction/summary attempts, resumability result, source audit and measured throughput. A pilot proves the recorded scope, not that full 15k indexing fits an unmeasured budget.
- Exit status, failure/OOM details, completion counts and limitations. Do not silently raise memory limits or claim native FP8 when a fallback ran.

Use the existing shared runner with a T4-local configuration copied from `configs/graph_local_15000.json`; change only local artifact/output paths for query-time validation. Install CPU Torch if it is used only for tokenizer tensors; Ollama must still report actual GPU residency. Preserve the generator's pinned Ollama version and model digest. If that release cannot load on T4, retain the failed receipt and obtain a team-approved new common experiment rather than quietly swapping just the graph generator.

## Current status
**PREPARING: Kaggle access is now authenticated.** The user authorized a private Kaggle T4 run; account-specific access succeeds for m1r0tvorxc and29.97 GPU hours remain. The private notebook must explicitly request `NvidiaTeslaT4`, check actual GPU names and use only GPU0. Smoke version1 failed before model inference/input discovery. The corrected private input revision2 is byte-preserving and smoke notebook version2 is queued. No successful T4 model call has been recorded yet. Prepared instructions and source compatibility documentation are not measured evidence; no T4 performance number is reported in the graph completion record.
