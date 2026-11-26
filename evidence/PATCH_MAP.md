# Patch & Ecosystem Audit: TensorFold v1.3.5 vs MiaAI-Lab, Maxrio9, sxuff, tonyd2wild

Base target engine: **`ashhart/TensorFold` tag `v1.3.5`** (tag OID `d6b3be891b5fb539a705d9d3b5279008045900e8`, commit `02b324ae9767253d9911d753d5c835a5070eed1b`).

## 1. Architectural Landscape (Critical Finding)

1. **`ashhart/TensorFold` `v1.3.5`** ships the refactored `loop_engine.zig` / `cuda_native.zig` / `modelopt.cu` Inkling-2-Mini CUDA path (#4387) plus major v1.3.5 decode speedups (#4697 two-halves loop gate, #4728 warp-staged 128-byte expert weights, #4729 fused gate/up `kDecodeTiles=8`, #4744 bounded top-1024 heap sampler, #4876 thinking effort 0, #4898 tool-call format). However, in `v1.3.5`, `inkling2/cuda_native.zig` is still **text-only, no drafter (`Inkling2DSparkPending`), BF16 unpaged per-stream KV, no CUDA graphs, no lazy head, no TC prefill, and no structured output (`StructuresUnsupported`)**.
2. **`MiaAI-Lab/Inkling-2-Mini-DGX-Spark-TensorFold`** (`9e3cf20b76e0140bebbbe6ed13f6e3a00e881f12`) targets the older PR #4441 (`dc2ca8a1785df6ee5c8489ee7cd18e29cf7b4568`, `mini_native.zig` / `mini_model.zig`) with `patches/0001`..`0018`. Its `patches/0001`..`0018` cannot `git apply` directly to `v1.3.5`'s `loop_engine.zig` tree without porting, and several of its server/kernel features already landed in `v1.3.5`.
3. **`Maxrio9/TensorFold` branch `inkling2-mini-exl3-v2`** (`353717987a736b47f87039c89cfa675c820506c7`) already ported Mia's core runtime features directly onto `loop_engine.zig` / `cuda_native.zig`, AND added **shared paged FP8 KV pool (`loop_pages.zig` + `core/kv_pool`, 1M default context)**, **FP8 DFlash2 weights & borrowed FP8 head (`draft_fp8.zig`)**, **CUDA decode + shared verify graphs (`loop_graphs.zig`)**, **lazy head (`loop_lazy_head.zig`)**, **CUDA vision (`mini_vision.zig`)**, and **EXL3 support**. However, Maxrio9 branched pre-v1.3.5 and therefore **lacks v1.3.5's #4728/#4729 `modelopt.cu` decode expert kernels, #4876, #4898, and several Mia/new wins**.

---

## 2. Complete Patch & Feature Classification

### Category (a): Already in `ashhart/TensorFold` `v1.3.5`

| Source | Feature / Patch | v1.3.5 Status |
|---|---|---|
| Mia `0004` | Inkling-2 reply split (`reasoning`, `content`, `tool_calls`) | **In v1.3.5** (`chat_inkling2.zig`, `tool_parse.zig`) |
| Mia `0005` (partial) | Loop gate two-halves kernel + `TF_INKLING2_LOOP_TILED` equivalent | **In v1.3.5** (#4697 in `loop_math.cu`) |
| Mia `0007` (superseded) | NVFP4 routed expert warp split / down split | **Superseded in v1.3.5** by #4728 (warp-staged 128B rows) + #4729 (`kDecodeTiles`, fused gate/up) in `modelopt.cu` |
| Mia `0010` | `reasoning_effort` / `enable_thinking: false` + `reasoning` output field | **In v1.3.5** (#4876, `chat_inkling2.zig`) |
| Mia `0017` (partial) | Server compat, tool-call guard, multi-call required/named tool format | **In v1.3.5** (#4898 + `server/` updates; verify `TF_TOOL_CALL_GUARD` parity) |
| TensorFold `v1.3.5` | Bounded top-1024 heap sampler (`0.11 ms` vs `15 ms` on 201K vocab) | **In v1.3.5** (#4744) |
| TensorFold `v1.3.5` | Interleaved window-cost sweeps & startup repair | **In v1.3.5** (#4755) |
| TensorFold `v1.3.5` | `vision.zig` + `vision.cu` kernel primitives (used by Metal in v1.3.5) | **Kernels in v1.3.5**, needs CUDA runtime wiring (`mini_vision.zig` from Maxrio9) |

### Category (b): Needed to Port to `v1.3.5` (`@engine-dev` Milestone 1)

*Note: Items marked `[Maxrio9-loop_engine]` are already written against `loop_engine.zig` / `cuda_native.zig` in `Maxrio9/TensorFold` (`35371798`), making them clean, direct ports onto `v1.3.5`!*

| Priority | Source | Feature | Porting Notes |
|---|---|---|---|
| **P0** | Maxrio9 `inkling2-mini-exl3-v2` + Mia `0003` | **Shared paged FP8 KV pool (`1M` context, LRU prefix cache)** | Port `loop_pages.zig`, `cuda_capacity.zig`, `loop.zig` (`CacheFormat.fp8`, `initPaged`), `attention.cu` / `prompt_attention.cu` FP8 paged support, and `cuda_native.zig` `kvPool`/`share_*`. **Do not overwrite** `v1.3.5` `modelopt.cu` (#4728/#4729)! |
| **P0** | Mia `0018` + Maxrio9 `dflash2_*` + `draft_fp8.zig` + `tonyd2wild` `0018` | **DFlash2 native CUDA drafter + FP8/NVFP4 head + FP8 layers** | Port `dflash2_cuda.zig`, `dflash2_weights.zig`, `draft_fp8.zig`, `dspark_ops.zig` (taps), and `dspark_cuda.zig` (fallback). Combine Maxrio9's `borrow_head` + `FP8_LAYERS` with Mia/tonyd2wild's NVFP4 draft head option + calibrated chances (`TF_INKLING2_DFLASH2_CHANCES`). |
| **P0** | Maxrio9 `loop_graphs.zig` + Mia `0001`/`0005`/`0014` | **CUDA decode graphs + separate shared verify graph cache** | Port `loop_graphs.zig` (`TF_INKLING2_VERIFY_GRAPHS=1`) and `loop_round.zig` (`prepareDecode`, slot-stable page table pointers). |
| **P0** | Maxrio9 `loop_lazy_head.zig` + Mia `0008` | **Certified greedy FP8 Lazy Head** | Port `loop_lazy_head.zig` (`TF_INKLING2_LAZY_HEAD=1`) + `ops.cu` lazy kernels. Cuts target LM head reads by ~50% on greedy decode. |
| **P0** | Maxrio9 `mini_vision.zig` + Mia `0016` | **Native CUDA Vision (`--vision`)** | Wire `v1.3.5`'s existing `vision.zig` / `vision.cu` via Maxrio9's `mini_vision.zig` into `cuda_native.zig` and `loop_round.zig`. |
| **P0** | Maxrio9 `loop_masked_draw.zig` + Mia `0013` | **Structured output & `tool_choice` (`libtfgrammar.so`)** | Port `framed_structures = true` + `loop_masked_draw.zig` + xgrammar compile-cache disable fix. |
| **P1** | Mia `0012` | **Opt-in FP8 attention projections (`TF_INKLING2_ATTN_FP8=1`)** | Not in Maxrio9 for `modelopt-mixed`! Quantize `wq_du`, `wk_dv`, `wv_dv`, `wo_ud` to per-tensor E4M3 at load via `cuda.qmmf.Lane`. Cuts 48-visit dense weights from `2.79 GB -> 1.88 GB` (-32.5% dense BW). |
| **P1** | Mia `0015` | **Blackwell native FP4/MXFP8 Tensor-Core prefill (`TF_INKLING2_PREFILL_TC=1`)** | Not in Maxrio9 (`modelopt.cu` decodes to BF16)! Port Mia `0015`'s `mxf4nvf4` / `mxf8f6f4` MMA prefill path for `prompt_rows=4096` (`modelopt-mixed` too, not just `exl3`). |
| **P1** | Mia `0009` + `0011` + `0014` | **Fused prompt rows + sliding prompt attention prefetch + `4096` prompt rows** | Port `TF_INKLING2_PREFILL_SLIDE` (in Maxrio9 `kernels.zig`), `TF_INKLING2_ATTN_DB` (double-buffered K/V tile prefetch), and allow `prompt_rows=4096` on `modelopt-mixed` (`TF_INKLING2_PREFILL_ROWS`). |
| **P2** | Mia `0005` | **PDL (`TF_INKLING2_PDL`), `TF_INKLING2_MIX_NARROW`, `loop_math` gatePrepare** | Port Maxrio9's `gatePrepare`/`gateDownPrepared` in `loop_math.cu` and Mia `0005`'s PDL dependent launches + narrow mix kernel. |

### Category (c): New Optimization Opportunities Beyond Mia & Maxrio9 (`@engine-dev` Milestone 2)

1. **Batched `c4`/`c8` DFlash2 Drafting (`propose` + `absorb` in 1 pass):**
   - Both Mia `0018` and Maxrio9 `dflash2_cuda.zig` run DFlash2 `absorb` and `propose` **one stream at a time**. At `c4` (`4` streams), reading DFlash2's 5 layers + head `4x` wastes ~`2.0-2.7 GB/round`. Batching active streams into a single `absorb` + 5-layer `propose` pass (as in `tonyd2wild`'s `kMaxBatch=8` `dflash2.cu` and TF 1.3.5 `TF_Q35_DFLASH_ABSORB_BATCH`) cuts drafter weight traffic by `4x`.
2. **Zero-Thrash Shared-Window CUDA Graphs (`rows=4..32`):**
   - Maxrio9's `loop_graphs.zig` keys `verify_cache` (max `32` entries) by exact `round.Shape` (`Parts[16]` with exact `{slot, row0, rows, tiles, record}`), which thrashes whenever `4-8` streams have varying draft counts (`0..7`). Using device-side descriptor tables / slot-canonicalized or bucketed `rows` shapes captures shared rounds once with zero cache thrashing.
3. **Full DFlash2 Graph Capture (Block + Absorb + Selector):**
   - Maxrio9 graphs `dflash2_cuda.zig` `Block` per `(slot, n, rule)`, while Mia `0018` runs DFlash2 eagerly. Graph-capturing batched `absorb + block + top16 + selector` eliminates CPU launch gaps and host syncs between target verify and next draft.
4. **Dynamic `c4`/`c8` Draft Caps & Calibrated GB10 Cost Curves:**
   - Mia `0018` turns calibrated chances off at `>= 2` streams, and Maxrio9 hardcodes `shared_block_streams = 4`. Because MoE union climbs from `21.5%` (`R=4`) to `57.2%` (`R=16`) to `79.0%` (`R=32`), dynamically capping per-stream drafts to `d=1..3` at `c4`/`c8` based on per-stream acceptance probability + calibrated GB10 shared verify cost curves maximizes aggregate tok/s without over-pulling experts.
5. **Tensor-Core MMA Multi-Row Verify Experts (`rows >= threshold`):**
   - When `c4`/`c8` verify rounds hit `R=16..32` rows, `v1.3.5` `modelopt.cu` still runs the BF16-decoded decode kernel (`kDecodeTiles=1`). Routing wide verify windows (`R >= 8` or `16`) through TC MMA (`mxf4nvf4`/`mxf8f6f4`) cuts verify latency when compute/dequant bottlenecks wide windows.
6. **Parallel Fast Weight Load (`FAST_LOAD`) + Persistent Shard-Hash & Cost Cache:**
   - Multi-threaded eager safetensors reader overlapped with async HtoD upload and FP8/lazy-head GPU packing, plus mounting `/root/.cache/tensorfold` so warm starts target `< 10-15 s`.
7. **Short-Prompt TTFT Fast Path:**
   - Avoid 4K prefill scratch/tile overhead on short prompts (`<= 64-256` tokens) and optionally graph small prompt shapes (`<= 64` rows).

---

## 3. Ecosystem Recon Summary (Competitors)

| Recipe | Engine & Draft | Headline Numbers (1x DGX Spark) | Framing vs Quality Trade-off vs Real Wins | License |
|---|---|---|---|---|
| **MiaAI-Lab** (`Inkling-2-Mini-DGX-Spark-TensorFold`, `9e3cf20b`) | TF PR #4441 + `0001`..`0018`, DFlash2 | `c1`: **52.1** prose (think on) / **65.1** greedy / **106.7** code; `c4`: **115.6** prose / **159.2** code; `c8`: **168.1** prose / **237.5** code; prefill: **3,497** @ 4K / **3,072** @ 32K / **2,267** @ 128K; KV: **2.09M** (8x262K unpaged FP8) | **Framing:** unpaged per-stream 262K limits 1M concurrency. **Lossy opt-ins:** FP8 KV, `ATTN_FP8=1`, `PREFILL_TC=1` (732/734 accuracy). **Real wins:** native DFlash2 + NVFP4 head + chances, `PREFILL_TC`, lazy head, CUDA vision, CUDA graphs. | **Apache-2.0** |
| **Maxrio9** (`Maxrio9/TensorFold` `inkling2-mini-exl3-v2`, `35371798`) | TF (`#4786`-era) + `loop_engine` ports + paged FP8 KV + EXL3 | Full 1M context default, shared paged FP8 KV pool (~5M+ tok), FP8 DFlash2 weights | **Real wins:** shared paged FP8 KV pool on `loop_engine`, borrowed FP8 lazy head for DFlash2, `loop_graphs` separate shared verify cache. **Missing:** `v1.3.5` #4728/#4729, `ATTN_FP8`, `PREFILL_TC`, batched c4 DFlash2. | **Apache-2.0** (code pre-0.6.0 MIT) |
| **tonyd2wild** (`Inkling-2-Mini-NVFP4-DGX-Spark`, `2fe0eb57`) | vLLM + DFlash2 (`k=7`) & TF + patch 0018 | vLLM DFlash2: **88** mean `c1` (`73` prose / `95` code), **227** @ `c6`; TF DFlash2: **105.5** mean `c1`, **191** @ `c6`; KV (vLLM): **3.31M** FP8 (1M needle PASS) | **Framing:** greedy temp=0, 74-80 GiB KV leaves 7-8 GiB OS headroom (< 10 GiB reserve). **Real wins:** vLLM 1M FP8 KV pool math, `kMaxBatch=8` batched `dflash2.cu` design, NVFP4 draft head + landing-table chances. | **MIT** (repo & `dflash2` patches) / **Apache-2.0** (`tensorfold/`) |
| **sxuff** (`inkling-2-mini-spark`, `35a4fd83`) | vLLM (`humming` + INT8 DSpark) | `c1` greedy: **66.7** median (`45.6` prose / `73.3` code); warm startup **42-47 s** (`--safetensors-load-strategy eager`) | **Framing:** single-stream greedy 160 tok, DSpark only. **Real wins:** `--safetensors-load-strategy eager` (-65% startup), INT8 draft head/layers (lossless via rejection sampling), 16x512 BF16 GEMV tiles. | **Apache-2.0** |

---

## 4. Attribution & License Ledger (For `NOTICE` and `README.md`)

| Asset / Project | Author / Org | Repository / URL | Pinned Commit / Revision | SPDX License | Scope / Notes |
|---|---|---|---|---|---|
| **Inkling-2-Mini** | Thinking Machines Lab | `https://huggingface.co/thinkingmachines/Inkling-2-Mini` | `c9d028302cd0f035fdf8e7e91f089a4ce94d2195` | `Apache-2.0` (+ Thinking Machines Model AUP) | Target model weights & `config.json`; mounted read-only, never redistributed |
| **Inkling-2-Mini-DFlash2** | Inco AI | `https://huggingface.co/incoai/Inkling-2-Mini-DFlash2` | `4f9088ea66c21f2b99ed657d1941c306b03e84d2` | `Apache-2.0` | Block-diffusion draft model; mounted read-only |
| **DFlash** | Jian Chen, Yesheng Liang, Zhijian Liu (z-lab) | `https://github.com/z-lab/dflash` | Method / reference | `MIT` | Block-diffusion speculative decoding algorithm |
| **TensorFold** | Ash Hart & TensorFold contributors (incl. Jürgen Schmied `#4364/#4365/#4387/#4441/#4640/#4697/#4728/#4729`, ajensenwaud `#4124-#4128`, BlivionIaG `#4337/#4704`, sethforprivacy `#4447`, gilby, akol1 `#4274`) | `https://github.com/ashhart/TensorFold` | Tag `v1.3.5` (tag `d6b3be891b5fb539a705d9d3b5279008045900e8`, commit `02b324ae9767253d9911d753d5c835a5070eed1b`) | `Apache-2.0` (code prior to `0.6.0` retains `MIT`) | Core inference engine (`tensorfold-native`), `LICENSES/`, `THIRD_PARTY_NOTICES.md` |
| **Inkling-2-Mini-DGX-Spark-TensorFold** | MiaAI-Lab | `https://github.com/MiaAI-Lab/Inkling-2-Mini-DGX-Spark-TensorFold` | `9e3cf20b76e0140bebbbe6ed13f6e3a00e881f12` | `Apache-2.0` | `patches/0001`..`0018` (lazy head, FP8 attn, TC prefill, vision, DFlash2, chances, prompt/server fixes) & harness ideas |
| **Maxrio9/TensorFold** (`inkling2-mini-exl3-v2`) | Maxrio9 | `https://github.com/Maxrio9/TensorFold/tree/inkling2-mini-exl3-v2` | `353717987a736b47f87039c89cfa675c820506c7` | `Apache-2.0` (pre-0.6.0 `MIT`) | `loop_engine` ports of shared paged FP8 KV pool, 1M cap, FP8 DFlash2 weights, `loop_graphs`, `loop_lazy_head`, `mini_vision`, `loop_masked_draw` |
| **Inkling-2-Mini-NVFP4-DGX-Spark** | tonyd2wild (Tech2wild) | `https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark` | `2fe0eb57429013c0ccb1c6535339ae61a9e44d9a` | `MIT` (root & `dflash2/`), `Apache-2.0` (`tensorfold/`) | NVFP4 draft-head quantization (`fp4Of`), calibrated lone-stream chances, batched `dflash2.cu` design |
| **inkling-2-mini-spark** | sxuff | `https://github.com/sxuff/inkling-2-mini-spark` | `35a4fd8344f01148d35c899cc107aab0c537ce3d` | `Apache-2.0` | Benchmark framing insights & eager safetensors load inspiration |
| **xgrammar & dlpack** | MLC-AI team & DLPack contributors | `https://github.com/mlc-ai/xgrammar`, `https://github.com/dmlc/dlpack` | `v0.2.8` (`97787376faee5ed8466cfad57c99855e4ce2f6aa`) | `Apache-2.0` | `libtfgrammar.so` structured output & tool-choice grammar engine |
| **Zig** | Zig Software Foundation & contributors | `https://ziglang.org` | `0.17.0` | `MIT` | Build-time compiler only |
| **libjpeg-turbo & libwebp** | IJG / libjpeg-turbo & Chromium authors | `https://libjpeg-turbo.org`, `https://chromium.googlesource.com/webm/libwebp` | `3.1.4.1` / `1.6.0` | `IJG` / `BSD-3-Clause` / `zlib` | Image decoding for `--vision` |
| **NVIDIA PyTorch Container** | NVIDIA | `nvcr.io/nvidia/pytorch:26.07-py3` | `26.07-py3` | NVIDIA Software License Agreement | Base CUDA container (referenced in `Dockerfile`, never redistributed) |
