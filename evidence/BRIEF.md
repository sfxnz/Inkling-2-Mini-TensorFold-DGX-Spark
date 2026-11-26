# Inkling-2-Mini (TensorFold v1.3.5 + DFlash2, 1x DGX Spark)

## Target
- **Model:** `thinkingmachines/Inkling-2-Mini` (rev `c9d028302cd0f035fdf8e7e91f089a4ce94d2195`, 20.19 GB ModelOpt NVFP4 layers 0-19 / MXFP8 layers 20-23 + BF16 dense/vision/unembed, Apache-2.0 + Thinking Machines AUP; source BF16/ModelOpt verified via safetensors headers)
- **Drafter:** `incoai/Inkling-2-Mini-DFlash2` (rev `4f9088ea66c21f2b99ed657d1941c306b03e84d2`, 0.99 GB BF16, 5 layers, block_size=8, taps `[5, 23, 41, 48]`, Apache-2.0)
- **Hardware:** 1x NVIDIA DGX Spark (GB10, `sm_121`, 121.6 GiB / ~121 GiB unified LPDDR5X, 224.3 GB/s measured copy BW on `dgx-spark`, 95.4 TFLOPS BF16 peak, 2,200 MHz GPU clock lock)
- **Target repo:** `https://github.com/sfxnz/Inkling-2-Mini-TensorFold-DGX-Spark`
- **Winning wedge (1 sentence: what will this recipe be #1 at?):** Fastest single-DGX-Spark Apache-2.0 TensorFold v1.3.5 + DFlash2 recipe across c1 prose, c4 aggregate decode, shared paged 1M KV capacity, TTFT/prefill, and fast startup.

## Hard Constraints
- Min context per request: `1,048,576` (full 1M shared paged FP8 KV pool; also verify `262,144` default tier)
- Concurrent resident users: up to `8` concurrent streams at 262K (and `3+` concurrent full 1M streams in FP8 paged pool)
- Host memory reserve: `>= 10.0 GiB` (out of 121 GiB unified RAM; never exceed ~111 GiB total GPU + process footprint)
- Quality gate: Target verification preserves greedy/sampled distributions; any lossy target feature (`FP8 KV`, `FP8 attn projections`, `TC prefill`) is clearly labelled, passes paired NLL <= `+0.002 nats/tok`, 50%/95% long-context needle retrieval, tool-call/structured-output conformance, and multilingual checks
- Licensing rule: Apache-2.0 / MIT only (with complete attribution in `NOTICE` and `README.md`)
- Required features: `vision`, `tool_calls`, `structured_output`, `reasoning_effort`, `prefix_cache_resume`, `dflash2`
- Allowed scope: TensorFold `v1.3.5` engine/kernel patches, CUDA graphs, shared paged KV pool, drafter optimizations, parallel fast load, and launch/harness tuning (no lossy Q4 dense weights)

## Phase 0 Checklist (complete BEFORE starting the optimization loop)
- [x] Hardware limits pinned from `knowledge/hardware-2x-spark.md` (224.3 GB/s BW, 95.4 TFLOPS BF16, 2,200 MHz clock lock; live quiet-host check verified by `@bench-opt`).
- [x] Architecture X-Ray in `THEORY.md`: dense vs. active-expert byte split, KV bytes/tok + sliding/conv/drafter state, ceilings via `ceiling.py` (`ceiling.json`).
- [x] Ecosystem Recon: `MiaAI-Lab/Inkling-2-Mini-DGX-Spark-TensorFold`, `Maxrio9/TensorFold` (`inkling2-mini-exl3-v2`), `sxuff/inkling-2-mini-spark`, and `tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark` classified in `PATCH_MAP.md` (+ licenses).
- [x] Engine + Weight Fast-Path Audit: `ashhart/TensorFold` `v1.3.5` audited against Mia `0001`..`0018` and Maxrio9 in `PATCH_MAP.md`.
- [ ] Day-1 Baseline + Stage/Kernel Profiler run on decode, prefill, and startup; Roofline Gap Table printed (owned by `@bench-opt`).
