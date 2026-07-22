# Candidates from NVIDIA/TensorRT-LLM

Source repo (cloned): `/work/trt/repos/TensorRT-LLM`
Cookbook target: `/work/trt-samples-for-hackathon-cn/cookbook/09-TensorRT-LLM/` (currently a near-empty stub — only a `README.md` with one line).

## Context / key finding

This checkout of TensorRT-LLM has **moved to the PyTorch-backend high-level `LLM` API**. The old
"convert HF -> build TRT engine with `trtllm-build` -> run" workflow is largely gone; there is **no
Whisper example and no per-model `build.py`/`run.py`** anymore. The learnable, self-contained surface
is now the Python `LLM` API under `examples/llm-api/` plus `trtllm-serve`. Almost every example uses
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` (small, CPU-downloadable, ideal for a cookbook) and is a single
short `main()` — trivial to trim into a cookbook leaf directory with a `main.py` + `log-main.py.log`.

Most examples are self-contained one-file scripts; the main caveat is that running them requires a full
TensorRT-LLM install (GPU), so cookbook leaves should ship the script + expected-output log even if CI
skips execution (`.skip_unit_test` / `unit_test.yaml enabled: false`).

Cross-checked cookbook sections (`00-Data` … `10-TensorRT-RTX`): none contain any LLM `LLM`-API
content, so nothing below duplicates existing material.

---

## Ranked candidates

### 1. LLM API quickstart — "hello world" generate  — Priority: HIGH
- Path: `examples/llm-api/quickstart_example.py` (also literal-included in `docs/source/quick-start-guide.md`)
- Demonstrates: minimal `LLM(model=...)` + `SamplingParams` + `llm.generate(prompts)` loop. ~30 lines.
- Cookbook section: `09-TensorRT-LLM/01-Quickstart` (the anchor example the section is missing).
- Self-contained: YES, fully. Best first thing to adopt.

### 2. Async + streaming generation — Priority: HIGH
- Paths: `examples/llm-api/llm_inference_async.py`, `examples/llm-api/llm_inference_async_streaming.py`
- Demonstrates: `llm.generate_async(...)` with `asyncio`, and `streaming=True` token-by-token output.
- Cookbook section: `09-TensorRT-LLM/02-AsyncStreaming`.
- Self-contained: YES (43 / 64 lines each). Good pair after quickstart.

### 3. Sampling techniques showcase — Priority: HIGH
- Path: `examples/llm-api/llm_sampling.py` (248 lines, self-documenting)
- Demonstrates: greedy, temperature, top-k/top-p, beam search, n / best_of, penalties — all via
  `SamplingParams`. Maps directly to a "how decoding params work" cookbook lesson.
- Cookbook section: `09-TensorRT-LLM/03-Sampling`.
- Self-contained: YES; trim the `click` CLI into a plain sequential demo.

### 4. FP8 / INT4-AWQ quantization of an LLM — Priority: HIGH
- Paths: `examples/quantization/quantize.py` (208 lines) + `examples/quantization/README.md`
- Demonstrates: `quantize_and_export()` (ModelOpt) to produce FP8 / int4_awq / int8_sq / auto-quant
  checkpoints; also the simpler path of just passing a pre-quantized HF repo (e.g.
  `nvidia/Llama-3.1-8B-Instruct-FP8`) straight to `LLM(...)`.
- Cookbook section: `09-TensorRT-LLM/04-Quantization` (fills the biggest gap vs. cookbook `04-Feature`
  quant, which is TRT-core only).
- Self-contained: MOSTLY — `quantize.py` has a large argparse; better to author a trimmed
  `main.py` that (a) loads a pre-quantized FP8 checkpoint via `LLM`, and (b) shows one
  `quantize_and_export` call. Note: full calibration needs a real model + dataset (heavy).

### 5. Distributed inference (TP / PP / EP) — Priority: HIGH
- Path: `examples/llm-api/llm_inference_distributed.py` (44 lines)
- Demonstrates: multi-GPU via `tensor_parallel_size` / `pipeline_parallel_size` /
  `moe_expert_parallel_size` — one keyword arg each. Very teachable.
- Cookbook section: `09-TensorRT-LLM/05-MultiGPU` (parallels cookbook `08-Advance` multi-device).
- Self-contained: YES as source; execution needs ≥2 GPUs (mark CI-skip).

### 6. Speculative decoding (NGram / Draft-Target / EAGLE3 / MTP) — Priority: HIGH
- Paths: `examples/llm-api/llm_speculative_decoding.py` (94 lines) + recipes in
  `examples/llm-api/README.md` (NGram & Draft-Target via `quickstart_advanced.py`)
- Demonstrates: `speculative_config=` with `NGramDecodingConfig` / `Eagle3DecodingConfig` /
  `MTPDecodingConfig`. NGram variant needs only a single base model (no draft model) — pick that for
  the minimal cookbook version.
- Cookbook section: `09-TensorRT-LLM/06-SpeculativeDecoding`.
- Self-contained: YES; NGram path is the lightest.

### 7. KV-cache runtime configuration — Priority: HIGH
- Path: `examples/llm-api/llm_runtime.py` (144 lines, heavily commented)
- Demonstrates: `KvCacheConfig` (`free_gpu_memory_fraction`, `enable_block_reuse`) and
  `CudaGraphConfig`. This is the simplest concrete way to teach paged-KV-cache concepts the prompt
  asked for, without diving into C++ kernels.
- Cookbook section: `09-TensorRT-LLM/07-KVCacheAndRuntime`.
- Self-contained: YES.

### 8. trtllm-serve + OpenAI-compatible client — Priority: HIGH
- Paths: `docs/source/quick-start-guide.md` (serve section) + `examples/serve/openai_completion_client.py`
  + `examples/serve/openai_chat_client.py` + `examples/serve/curl_chat_client.sh`
- Demonstrates: launch `trtllm-serve "TinyLlama/..."`, then hit `v1/chat/completions` /
  `v1/completions` with the `openai` Python client or curl. The standard "deploy as a server" lesson.
- Cookbook section: `09-TensorRT-LLM/08-Serve` (parallels cookbook `07-Tool`).
- Self-contained: YES; client scripts are ~15 lines. Package as server-launch + client pair.

### 9. Guided / structured (JSON-schema) decoding — Priority: MED
- Path: `examples/llm-api/llm_guided_decoding.py` (47 lines)
- Demonstrates: `guided_decoding_backend='xgrammar'` + `GuidedDecodingParams(json=schema)` to force
  valid-JSON output. Popular, self-contained, uses TinyLlama.
- Cookbook section: `09-TensorRT-LLM/09-GuidedDecoding`.
- Self-contained: YES.

### 10. Custom logits processor — Priority: MED
- Path: `examples/llm-api/llm_logits_processor.py` (128 lines)
- Demonstrates: subclassing `LogitsProcessor` to bias/steer generation (e.g. length control by
  adjusting EOS logits). Good "extending the API" lesson.
- Cookbook section: `09-TensorRT-LLM/10-LogitsProcessor`.
- Self-contained: YES (needs `transformers` tokenizer, already a dep).

### 11. Multi-LoRA serving — Priority: MED
- Path: `examples/llm-api/llm_multilora.py` (89 lines)
- Demonstrates: `LoraConfig` + `LoRARequest` to load and switch among multiple LoRA adapters at
  request time (auto-downloads 3 small TinyLlama adapters from HF).
- Cookbook section: `09-TensorRT-LLM/11-MultiLoRA`.
- Self-contained: YES (downloads adapters via `huggingface_hub`).

### 12. Multimodal (image / video / audio) inference — Priority: MED
- Paths: `examples/llm-api/quickstart_multimodal.py` (340 lines) + `examples/llm-api/README.md`
- Demonstrates: `default_multimodal_input_loader` + `LLM` on a VLM (e.g. NVILA-8B, Qwen2-VL,
  Phi-4-multimodal for audio). Covers the "multimodal / Whisper-like" ask, though no dedicated Whisper.
- Cookbook section: `09-TensorRT-LLM/12-Multimodal`.
- Self-contained: PARTIALLY — depends on `quickstart_advanced.py` (`setup_llm`/`add_llm_args`) and
  large VLM weights. Higher trim effort; larger download. Good but not the first pick.

### 13. KV-cache host offloading demo — Priority: MED
- Path: `examples/llm-api/llm_kv_cache_offloading.py` (134 lines, excellent docstring)
- Demonstrates: measurable cache reuse — offload evicted KV blocks to host RAM vs. recompute; verify
  via `reused blocks` / `cache hit rate` in DEBUG logs. Concrete, observable perf lesson.
- Cookbook section: `09-TensorRT-LLM/07-KVCacheAndRuntime` (companion to #7).
- Self-contained: YES.

### 14. AutoDeploy one-command HF->deploy — Priority: MED
- Paths: `examples/auto_deploy/build_and_run_ad.py` (406 lines) + `examples/auto_deploy/README.md`
- Demonstrates: `python build_and_run_ad.py --model "TinyLlama/..."` — graph-transformation pipeline
  that auto-shards + deploys a raw HF checkpoint. Alternative "zero-config" entry point.
- Cookbook section: `09-TensorRT-LLM/13-AutoDeploy`.
- Self-contained: script is large/config-heavy; adopt as a "pointer + minimal invocation" lesson
  rather than a full port.

### 15. Interactive chat REPL app — Priority: LOW
- Path: `examples/apps/chat.py` (97 lines) + `examples/apps/fastapi_server.py`
- Demonstrates: multi-turn chat loop with `apply_chat_template` + `KvCacheConfig`; FastAPI wrapper.
- Cookbook section: `09-TensorRT-LLM/14-ChatApp`.
- Self-contained: YES but overlaps #8; lower priority.

### 16. Guided-decoding / eval utilities (summarize, mmlu) — Priority: LOW
- Paths: `examples/summarize.py` (944 lines), `examples/mmlu.py` (479 lines)
- Demonstrates: accuracy/eval harnesses (CNN/DailyMail ROUGE, MMLU). Useful but large and
  dataset-heavy; better referenced than ported.
- Cookbook section: `09-TensorRT-LLM/15-Evaluation` (optional).
- Self-contained: NO — heavy deps/datasets. Low priority.

---

## Suggested adoption order (minimal, high-value LLM section)

1. #1 Quickstart  →  2. #2 Async/Streaming  →  3. #3 Sampling  →  4. #7 KV-cache/Runtime
5. #5 Multi-GPU  →  6. #6 Speculative (NGram)  →  7. #4 Quantization (FP8 pre-quantized path)
8. #8 trtllm-serve + OpenAI client

These 8 give a complete, mostly self-contained LLM-API cookbook track using TinyLlama, filling the
`09-TensorRT-LLM` gap without any C++/kernel internals. Ship each as `main.py` + expected `log-main.py.log`;
gate CI execution behind GPU availability (`unit_test.yaml enabled: false` or `.skip_unit_test`).

## Excluded (require full framework build or are library internals)
- `cpp/`, `triton_backend/`, `triton_kernels/`, `benchmarks/`, `3rdparty/` — build/kernel internals.
- `examples/disaggregated/`, `examples/wide_ep/`, `examples/ray_orchestrator/` — multi-node infra;
  too heavy for a runnable cookbook leaf.
- Per-model dirs under `examples/models/core/*` — mostly READMEs/config, no longer standalone build scripts.
