# Inkling-2-Mini (NVFP4/MXFP8 + DFlash2) on 1x DGX Spark (TensorFold v1.3.5 Patched)

Serve [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) with [incoai/Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) on a **single NVIDIA DGX Spark (GB10, `sm_121`, 128 GB unified LPDDR5X)** using patched `ashhart/TensorFold` `v1.3.5` (`engine-patches/`, `0001..0023`).

What this recipe delivers out of the box:
- **1M context with 4.73M shared paged FP8 KV tokens** (`4,726,272` tokens, `CONTEXT=1048576`, `PARALLEL=8`, `MEMORY_RESERVE_GIB=13.0`, `min_mem_avail=10.45 GiB`, `peak_anon=1.43 GiB`, `pswpout_delta=0`) and zero-copy LRU prefix caching (`cached_tokens` verified).
- **High-throughput NVFP4 MoE & batched multi-stream decode** via `Nvfp4Experts` BLOCK4 repack with fused split-half softcap-SwiGLU, stream-batched `conv`/`attn` with bucketed global `last_pos` and slot-agnostic `Shape` CUDA graphs, `qmmf_group` dense fusion (`TF_QMMF_HALF_BN=1`), and wave-scheduled MoE (`TF_INKLING2_MOE_WAVES=1`).
- **NVFP4 DFlash2 speculative decoding** (`--drafter-bits 4`, `333.2 MB` drafter weights), uniform batched `shared_guard` policy with calibrated `CHANCES` + `Landing`, single-launch fused keyed selector, BF16 Tensor Core `tf_ink2_df_attn_batch` (eliminating the `>=2K` context decode cliff: `49.76 -> 54.55 tok/s` at 4K ctx), and greedy exactness matching plain serving (`4/4` true `plain == drafted` exact on both `content` and `reasoning_content`).
- **Blackwell tensor-core MoE prefill** (`TF_INKLING2_PREFILL_TC=1`) reaching **`2,924..3,024 tok/s`** at 4K-16K and **`2,215 tok/s`** at 128K (strict `cached_tokens == 0`).
- **Clearly labelled opt-in lossy defaults:** paged **FP8 KV**, **`TF_INKLING2_ATTN_FP8=1`**, **`TF_INKLING2_GATE_FP8=1`**, **`TF_INKLING2_PREFILL_TC=1`**, and **`TF_INKLING2_SAMPLED_FP8_HEAD=1`** (reusing resident FP8 `lazy_head` on sampled decode for `+13%` c1 sampled speed at 0 B extra VRAM). Set `TF_INKLING2_SAMPLED_FP8_HEAD=0` for lossless BF16 Tensor Core sampled unembed (`47.92 / 105.13 tok/s`, or `49.33 / 106.67 tok/s` with `TF_QMMF_HALF_BN=1`; full comparison below).
- **Multimodal vision (`--vision`)** with pinned `libjpeg-turbo 3.1.4.1` and `libwebp 1.6.0` (PNG, JPEG, WebP), structured outputs / `tool_choice` (`libtfgrammar.so`), and fast persistent startup (`16.80 s` warm / `17.05 s` cold).

```sh
git clone https://github.com/sfxnz/Inkling-2-Mini-TensorFold-DGX-Spark
cd Inkling-2-Mini-TensorFold-DGX-Spark
./docker/build-patched.sh      # builds tf-inkling2:1.3.5 base + tf-inkling2:m3-p9 patched overlay
VALIDATE_ONLY=1 ./run.sh       # read-only preflight check
./run.sh                       # http://localhost:8000/v1, model inkling-2-mini
./stop.sh                      # graceful stop (60s drain) by label/name
```

---

## Measured Results (1x DGX Spark GB10, 2,200 MHz clock lock)

All runs below were measured on a single NVIDIA DGX Spark (GB10 `sm_121`, 48 SMs, 121.69 GiB unified memory, `nvidia-lock-clocks.service` at 2,200 MHz, `MEMORY_RESERVE_GIB=13.0`, `pswpout_delta=0` throughout). Inner-loop and outer-loop/gate rulers are reported separately because thinking mode, sampling temperature, and concurrency change acceptance and throughput.

### Outer-Loop & Qualification Ruler (`tools/bench.py --ruler outer`, `tools/prefill.py`, `tools/startup.py`, `tools/gates.py`)

| Metric | M2 (`tf-inkling2:m2-p9`) | M3 Qualified (`tf-inkling2:m3-p9` / `clean-verify-m3`) | Notes / Harness |
|---|---:|---:|---|
| **Startup (Cold / Warm)** | `15.30 s` / `14.29 s` | **`17.05 s`** / **`16.80 s`** | `tools/startup.py --mode both` until `/v1/models` 200 OK |
| **KV Pool (`CONTEXT=1M, P=8`)** | `4,736,000` tok | **`4,726,272` tok** (`18,462` pages) | Shared paged FP8 KV (`4x 1M` streams, NVFP4 drafter `333.2 MB`, `0` swap-out) |
| **Sampled Prose (thinking off, c1/c2/c4/c8)** | `42.25 / 53.75 / 72.26 / 94.40` | **`54.48 / 73.49 / 114.11 / 151.32` tok/s** | `tools/bench.py --ruler outer` (temp 1.0, median of 6; c1 `41.42 ms/rnd`, TTFT `0.143 s`, c4 `30.04/strm`, c8 `19.99/strm`; `SAMPLED_FP8_HEAD=1`) |
| **Sampled Prose (lossless BF16 TC unembed, `SAMPLED_FP8_HEAD=0`)** | `42.25 / - / 72.26 / -` | **`47.92 / 105.13`** (`m3-p9` base)<br>**`49.33 / 106.67`** (`+HALF_BN=1`)<br>**`48.56 / 106.13`** (`+MOE_WAVES=1`) | `c1 / c4` agg sampled prose (think off) with exact BF16 Tensor Core unembed (`evidence/profiles/m3-lossless-bf16-head-ref.json`) |
| **Thinking-On Prose (`T=0.7, k=20`, c1/c2/c4/c8)** | `43.75 / 51.46 / 73.60 / 92.05` | **`56.76 / 70.25 / 112.60 / 151.43` tok/s** | `tools/bench.py --ruler outer` (median of 6; `+29.7%` c1, `+53.0%` c4, `+64.5%` c8) |
| **Code (thinking off, default sampling, c1/c2/c4/c8)** | `48.37 / 56.88 / 80.09 / 98.37` | **`65.55 / 79.95 / 122.05 / 156.75` tok/s** | `tools/bench.py --ruler outer` (median of 6; `+35.5%` c1, `+52.4%` c4, `+59.4%` c8) |
| **Greedy Prose (thinking off, c1/c2/c4/c8)** | `46.21 / 58.85 / 74.27 / 95.62` | **`56.95 / 73.43 / 110.70 / 154.10` tok/s** | `tools/bench.py --ruler outer` (`c1` TTFT `0.143 s`, `44.54 ms/rnd`) |
| **4K-Context Decode (Lever 9 A/B, greedy prose)** | `49.76 tok/s` (`m3-p8`, `50.19 ms/rnd`) | **`54.55 tok/s`** (`41.67 ms/rnd`, `-8.52 ms/rnd`) | BF16 TC `tf_ink2_df_attn_batch` eliminates `>=2K` ctx cliff (`+9.6%`) |
| **Cold Prefill `4K` / `8K` / `16K`** (`cached=0`) | `2,347.1 / 2,522.7 / 2,563.8` | **`2,923.7 / 3,012.5 / 3,024.4` tok/s** | `tools/prefill.py` (`runs=2` median, strict `cached_tokens == 0`; `+24.6%` at 4K) |
| **Cold Prefill `32K` / `65K` / `128K`** (`cached=0`) | `2,464.6 / 2,275.9 / 1,980.4` | **`2,900.3 / 2,627.2 / 2,214.6` tok/s** | `tools/prefill.py` (`runs=2` median, strict `cached_tokens == 0`; `+17.7%` at 32K, `+11.8%` at 128K) |
| **Acceptance Gates (`tools/gates.py`)** | `12 / 12 PASS` | **12 / 12 PASS** | `needle_256k` (`302.5 s`), `needle_1m` (`2,869.9 s`, `1,023,701`/`1,023,881` exact), `soak` `c4`+`c8` (`252.3 s`), `vision` (PNG/JPEG/WebP), `tools`, `structured`, `prefix_cache` (`8704/8726`), `multilingual` |
| **Drafted vs Plain Exactness** | `4 / 4 EXACT` | **4 / 4 EXACT** | True `plain` (`DRAFTER="" EXTRA_ARGS="--no-drafts"`, `drafted=0`) vs `DFlash2` greedy `T=0` (`all_exact: true`, content + reasoning) |
| **Unified Memory Peak (`12/12` gates incl. `needle_1m` + `soak`)** | `min_avail 8.36 GiB`, `anon 2.93 GiB`, `pswpout 0` | **`min_avail 10.45 GiB`, `anon 1.43 GiB`, `pswpout 0`** | `tools/memwatch.sh` (684 samples, peak GPU `102.89 GiB`, `alerts: []`) |

### Inner-Loop Deterministic Ruler (`tf-inkling2:clean-verify-m3`, `tools/bench.py --ruler inner`, `T=0`, `runs=6`)

| Workload | `c1` | `c4` agg | `c8` agg |
|---|---:|---:|---:|
| Prose (greedy, thinking off) | **56.49 tok/s** (`44.60 ms/rnd`, `2.51 tok/rnd`, `+22.2%` vs M2) | **113.60 tok/s** (`29.66/strm`, `+45.5%` vs M2) | **151.40 tok/s** (`19.79/strm`, `+56.1%` vs M2) |
| Code (greedy, thinking off) | **74.53 tok/s** (`46.97 ms/rnd`, `3.48 tok/rnd`, `+50.3%` vs M2) | **132.29 tok/s** (`37.22/strm`, `+57.5%` vs M2) | **162.00 tok/s** (`21.52/strm`, `+38.8%` vs M2) |

### Clean-Checkout Build Verification (`tf-inkling2:clean-verify-m3`, `evidence/profiles/clean-verify-m3-ab.json`)

Rebuilt from scratch via `./docker/build-patched.sh tf-inkling2:clean-verify-m3` (`0001..0023` on `v1.3.5` `02b324ae`, image `sha256:9ae0ffd1de2b271d9407523b30db4a30b6c918f5c7a349b6cfb1085ba92e0217`, binary `12a9c9ca413736ecc618f111c82977f921ab09da87588905901ba43b795565ec`) and bracketed `A1 / B / A2` (`runs=4`, `pswpout_delta=0`, `min_mem_avail=10.95..11.30 GiB`) against `tf-inkling2:m3-p9`:
- **Sampled prose (`c1 / c4` agg):** `B (clean-verify-m3)` `54.04 / 111.97 tok/s` vs `A1/A2` avg `52.99 / 108.07 tok/s`.
- **Greedy prose (`c1 / c4` agg):** `B (clean-verify-m3)` `56.78 / 109.67 tok/s` vs `A1/A2` avg `56.07 / 106.22 tok/s`.
- **Cold prefill (`4K / 32K`):** `B (clean-verify-m3)` `2,841.0 / 2,893.0 tok/s` vs `A1/A2` avg `2,944.7 / 2,866.4 tok/s`.
- **Full qualification (`m3-qual-*`):** `12/12` gates PASS, `4/4` pure `plain == drafted` exact, `min_mem_avail = 10.45 GiB`, `peak_anon = 1.43 GiB`, `pswpout_delta = 0`.

*Note on opt-in lossy defaults, thinking effort, and NLL harness:*
- **Opt-in lossy defaults:** The default `recipe.conf` enables paged **FP8 KV**, **`TF_INKLING2_ATTN_FP8=1`**, **`TF_INKLING2_GATE_FP8=1`**, **`TF_INKLING2_PREFILL_TC=1`**, and **`TF_INKLING2_SAMPLED_FP8_HEAD=1`**. Greedy decode uses lossless exact BF16 verification via `LAZY_HEAD`; sampled decode uses the resident FP8 `lazy_head` weight when `TF_INKLING2_SAMPLED_FP8_HEAD=1` (`54.48 / 114.11 tok/s` c1/c4). To use lossless BF16 Tensor Core unembed on sampled decode too (`47.92 / 105.13 tok/s`, or `49.33 / 106.67 tok/s` with `TF_QMMF_HALF_BN=1`), remove `TF_INKLING2_SAMPLED_FP8_HEAD=1` (or set `TF_INKLING2_SAMPLED_FP8_HEAD=0`) in `EXTRA_ENV`. To also use pure BF16 attention/gate projections, drop `TF_INKLING2_ATTN_FP8=1` and `TF_INKLING2_GATE_FP8=1`.
- **Thinking effort:** `Inkling-2-Mini`'s chat template emits `Thinking effort level: 0` when `enable_thinking=False`; the model weights may still emit a brief separated `<|content_thinking|>` segment before the final answer, which TensorFold places in `reasoning_content` cleanly.
- **NLL harness:** Paired teacher-forced NLL (`tools/nll.py`) is currently unsupported by TensorFold's `inkling2` backend (`prompt_logprobs` is unimplemented in `loop_backend.zig` / `cuda.zig`, returning HTTP 400); quality parity is verified instead via true `plain` (`DRAFTER="" EXTRA_ARGS="--no-drafts"`) vs `drafted` (`DFlash2`) 4/4 greedy exactness and 12/12 `tools/gates.py` (including 256K and 1M needle retrieval).

---

## Requirements

- **Hardware:** 1x NVIDIA DGX Spark (GB10, `sm_121`, 128 GB unified LPDDR5X).
- **Driver & Runtime:** NVIDIA R580+ driver (`580.178.04`, CUDA 13.0) and Docker with the NVIDIA Container Toolkit (`--gpus all`).
- **Host Tools:** `docker`, `curl`, `hf` (Hugging Face CLI), `ss`, `free`, `stat`, `awk`, `grep` (and `jq` for `docker/verify-image.sh`, `shellcheck` for script lint).
- **Disk & Memory:** ~22 GB free under `HF_CACHE` (`thinkingmachines/Inkling-2-Mini` is 20.16 GB; `incoai/Inkling-2-Mini-DFlash2` is 0.99 GB) and `MemAvailable >= 108 GiB` before startup (`MEM_GATE_GIB`).

---

## Quick Start & Building from Source

### 1. Build the Patched Image

`engine-patches/` holds the 23 Apache-2.0 patches (`0001..0023` in `engine-patches/series`) against `ashhart/TensorFold` `v1.3.5` (`02b324ae9767253d9911d753d5c835a5070eed1b`).

```sh
# Builds tf-inkling2:1.3.5 base runtime (CUDA 13.0, libcublas-13-0, libjpeg-turbo 3.1.4.1, libwebp 1.6.0)
# then compiles patched tensorfold-native in Docker and overlays it via docker/Dockerfile.patched.
./docker/build-patched.sh
# Optional: rebuild libtfgrammar.so from xgrammar v0.2.8 too:
BUILD_GRAMMAR=1 ./docker/build-patched.sh
# Verify image capabilities on GB10:
./docker/verify-image.sh
```

### 2. Validate, Launch, and Stop

`run.sh` downloads missing pinned snapshots into `$HF_CACHE` (if not already present), checks ports, memory, and GPU exclusivity, mounts a persistent `/root/.cache/tensorfold` Docker volume (`ENGINE_CACHE`), and waits for `/health`.

```sh
VALIDATE_ONLY=1 ./run.sh   # read-only preflight + full docker argv
./run.sh                   # starts tf-inkling2 on port 8000
./stop.sh                  # graceful 60s drain + remove by label/name
```

Common overrides:
```sh
# Plain serving (no drafter at all):
DRAFTER="" EXTRA_ARGS="--no-drafts" ./run.sh

# Text-only (saves vision buffers):
VISION=0 ./run.sh

# Lossless BF16 TC sampled unembed (keeps ATTN_FP8/GATE_FP8/PREFILL_TC):
EXTRA_ENV="TF_INKLING2_ATTN_FP8=1 TF_INKLING2_PREFILL_TC=1 TF_INKLING2_GATE_FP8=1 TF_QMMF_HALF_BN=1 TF_INKLING2_MOE_WAVES=1 TF_INKLING2_SAMPLED_FP8_HEAD=0" ./run.sh

# BF16 attention, gate, and sampled unembed projections:
EXTRA_ENV="TF_INKLING2_PREFILL_TC=1 TF_QMMF_HALF_BN=1 TF_INKLING2_MOE_WAVES=1" ./run.sh
```

---

## Configuration (`recipe.conf`)

Any variable in `recipe.conf` can be overridden from the environment (e.g., `PORT=8001 PARALLEL=4 ./run.sh`).

| Variable | Default | Description |
|---|---|---|
| `MODEL`, `REVISION` | `thinkingmachines/Inkling-2-Mini` @ `c9d02830...` | Pinned target model (`1` model shard `19,526,019,168 B` + `mtp.safetensors` `635,801,416 B`). |
| `DRAFTER`, `DRAFTER_REVISION`, `DRAFTER_BITS` | `incoai/Inkling-2-Mini-DFlash2` @ `4f9088ea...`, `4` | Pinned DFlash2 drafter (`989,975,616 B`, packed to `333.2 MB` NVFP4 at load). Set `DRAFTER=""` to disable. |
| `BASE_IMAGE`, `IMAGE` | `tf-inkling2:1.3.5`, `tf-inkling2:m3-p9` | Base runtime and patched serving image. |
| `SERVED_NAME`, `PORT` | `inkling-2-mini`, `8000` | Model ID and HTTP listen port (`0.0.0.0`). |
| `CONTEXT`, `PARALLEL` | `1048576`, `8` | 1M context window (`--context`) and max concurrent streams (`--parallel`). |
| `VISION`, `VIDEO_TOKENS` | `1`, `64` | Enables `--vision` with `--vision-video-tokens 64`. |
| `MEMORY_RESERVE_GIB` | `13.0` | `TENSORFOLD_MEMORY_RESERVE_GIB` host safety reserve. |
| `ENGINE_CACHE` | `tf-inkling2-cache-135` | Persistent Docker volume at `/root/.cache/tensorfold` (shard hashes & window cost cache). |
| `EXTRA_ENV` | `TF_INKLING2_ATTN_FP8=1 TF_INKLING2_PREFILL_TC=1 TF_INKLING2_GATE_FP8=1 TF_QMMF_HALF_BN=1 TF_INKLING2_MOE_WAVES=1 TF_INKLING2_SAMPLED_FP8_HEAD=1` | Patched engine environment flags (set `TF_INKLING2_SAMPLED_FP8_HEAD=0` for lossless BF16 TC sampled unembed). |
| `API_KEY_FILE` | empty | Optional mode-600 key file (`--api-key-file`). |
| `MEM_GATE_GIB`, `FORCE` | `108`, `0` | Startup `MemAvailable` floor (`FORCE=1` overrides). |

---

## Evaluation & Gate Harness (`tools/`)

- **Two-ruler benchmark (`tools/bench.py`):**
  ```sh
  python3 tools/bench.py --ruler inner --concurrency 1,4,8 --runs 6 --out inner.json
  python3 tools/bench.py --ruler outer --concurrency 1,2,4,8 --runs 6 --out outer.json
  ```
- **Cold prefill (`tools/prefill.py`):**
  ```sh
  python3 tools/prefill.py --lengths 4096,8192,16384,32768,65536,131072 --runs 2 --out prefill.json
  ```
- **Cold & warm startup (`tools/startup.py`):**
  ```sh
  python3 tools/startup.py --mode both --out startup.json
  ```
- **Acceptance gates (`tools/gates.py`):**
  ```sh
  python3 tools/gates.py --out gates.json
  # Fast subset without 256K/1M needles:
  python3 tools/gates.py --skip needle_256k,needle_1m,soak
  ```
- **Paired NLL check (`tools/nll.py`, currently unsupported on `inkling2`):**
  The `inkling2` backend in `loop_backend.zig` / `cuda.zig` does not implement `prompt_logprobs` yet (HTTP 400: `"prompt_logprobs are not supported by this model or backend"`). Once upstream/backend adds `prompt_logprobs` support, capture and compare against a reference with:
  ```sh
  python3 tools/nll.py capture --label ref --out /tmp/nll-ref
  python3 tools/nll.py capture --label cand --out /tmp/nll-cand
  python3 tools/nll.py compare /tmp/nll-ref /tmp/nll-cand --margin 0.002
  ```
- **Unified-memory watchdog (`tools/memwatch.sh`):**
  ```sh
  ./tools/memwatch.sh /tmp/memwatch.tsv --interval 2 --floor-gib 8.0
  ```

---

## API Usage

TensorFold serves OpenAI-compatible (`/v1/chat/completions`, `/v1/completions`, `/v1/models`), Anthropic-compatible (`/v1/messages`), `/health`, and `/metrics` endpoints on `PORT` (default `8000`).

```sh
curl -s http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "inkling-2-mini",
    "messages": [{"role": "user", "content": "What is 17 * 23?"}],
    "max_tokens": 512,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

---

## Licenses and credits

This recipe repository (scripts, Dockerfiles, harness, and `engine-patches/`) is licensed under the **Apache License, Version 2.0** ([LICENSE](LICENSE)). Full third-party notices and exact pinned revisions are recorded in [NOTICE](NOTICE):

- **Target Weights:** Thinking Machines Lab, [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) @ `c9d028302cd0f035fdf8e7e91f089a4ce94d2195` (Apache-2.0 + Thinking Machines Model AUP).
- **DFlash2 Drafter:** Inco AI, [incoai/Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) @ `4f9088ea66c21f2b99ed657d1941c306b03e84d2` (Apache-2.0), implementing DFlash (Jian Chen, Yesheng Liang, Zhijian Liu, [z-lab/dflash](https://github.com/z-lab/dflash), MIT).
- **Inference Engine:** Ash Hart & TensorFold contributors, [ashhart/TensorFold](https://github.com/ashhart/TensorFold) tag `v1.3.5` (`02b324ae9767253d9911d753d5c835a5070eed1b`, Apache-2.0 / pre-0.6.0 MIT).
- **Adapted / Inspiring Community Recipes:**
  - [Maxrio9/TensorFold](https://github.com/Maxrio9/TensorFold/tree/inkling2-mini-exl3-v2) @ `353717987a736b47f87039c89cfa675c820506c7` (Apache-2.0 / pre-0.6.0 MIT): shared paged FP8 KV pool, 1M context cap, DFlash2 FP8 weights & borrowed head, CUDA graphs, lazy head, CUDA vision, and masked grammar.
  - [MiaAI-Lab/Inkling-2-Mini-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Inkling-2-Mini-DGX-Spark-TensorFold) @ `9e3cf20b76e0140bebbbe6ed13f6e3a00e881f12` (Apache-2.0, patches `0001/0005/0006/0007/0009/0012/0014/0015/0018`): server/prompt fixes, sliding AR deduplication & `mix_rows`/`rel_table` speedups, draft FP8/FP4 packing & shared fixed-block pricing, lazy head/vision, 4096-row chunks, `ATTN_FP8`, slot-agnostic wide decode graphs, Blackwell `PREFILL_TC`, and batched `hold_costs` / fused keyed selector / `CHANCES` + `Landing` calibration.
  - [tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark) @ `2fe0eb57429013c0ccb1c6535339ae61a9e44d9a` (MIT root/`dflash2/`, Apache-2.0 `tensorfold/`, patch `0018`): `shared_guard` / `CHANCES` + `Landing` / `hold_costs`, NVFP4 draft-head quantization, and batched DFlash2 kernel layout concepts.
  - [sxuff/inkling-2-mini-spark](https://github.com/sxuff/inkling-2-mini-spark) @ `35a4fd8344f01148d35c899cc107aab0c537ce3d` (Apache-2.0): benchmark framing and eager safetensors loading insights.
- **Runtime Libraries & Toolchain:** MLC-AI team & DLPack contributors ([mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar) `v0.2.8` @ `97787376faee5ed8466cfad57c99855e4ce2f6aa` Apache-2.0, `dlpack` Apache-2.0, `picojson` BSD-2-Clause), IJG / libjpeg-turbo contributors (`libjpeg-turbo` `3.1.4.1`, https://libjpeg-turbo.org, IJG / BSD-3-Clause / zlib), Chromium authors (`libwebp` `1.6.0`, https://chromium.googlesource.com/webm/libwebp, BSD-3-Clause), Zig Software Foundation & contributors (`Zig` `0.17.0`, https://ziglang.org, MIT, build-time only), and NVIDIA (CUDA `13.0.0`, `libcublas-13-0`, NVIDIA Software License Agreement).
