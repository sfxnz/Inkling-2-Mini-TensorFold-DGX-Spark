# Inkling-2-Mini (NVFP4/MXFP8 + DFlash2) on 1x DGX Spark (TensorFold v1.3.5 Patched)

Serve [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) with [incoai/Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) on a **single NVIDIA DGX Spark (GB10, `sm_121`, 128 GB unified LPDDR5X)** using patched `ashhart/TensorFold` `v1.3.5` (`engine-patches/`, `0001..0014`).

What this recipe delivers out of the box:
- **1M context with 4.74M shared paged FP8 KV tokens** (`CONTEXT=1048576`, `PARALLEL=8`, `MEMORY_RESERVE_GIB=13.0`) and zero-copy LRU prefix caching (`cached_tokens` verified).
- **Batched DFlash2 speculative decoding** across active concurrent streams (`c1..c8`), CUDA decode/verify/drafter graphs, and greedy exactness matching plain serving (`4/4` exact).
- **Blackwell tensor-core MoE prefill** (`TF_INKLING2_PREFILL_TC=1`) with 4096-row chunks, 8-warp `prompt_tc_coop`, and parallelized `tf_ink2moe_plan`.
- **Opt-in FP8 attention and gate projections** (`TF_INKLING2_ATTN_FP8=1`, `TF_INKLING2_GATE_FP8=1`) for higher decode throughput (disable by setting `EXTRA_ENV="TF_INKLING2_PREFILL_TC=1"` if strict BF16 dense projections are required).
- **Multimodal vision (`--vision`)** with pinned `libjpeg-turbo 3.1.4.1` and `libwebp 1.6.0` (PNG, JPEG, WebP), structured outputs / `tool_choice` (`libtfgrammar.so`), and fast persistent startup (`14.3 s` warm / `15.3 s` cold).

```sh
git clone https://github.com/sfxnz/Inkling-2-Mini-TensorFold-DGX-Spark
cd Inkling-2-Mini-TensorFold-DGX-Spark
./docker/build-patched.sh      # builds tf-inkling2:1.3.5 base + tf-inkling2:m2-p9 patched overlay
VALIDATE_ONLY=1 ./run.sh       # read-only preflight check
./run.sh                       # http://localhost:8000/v1, model inkling-2-mini
./stop.sh                      # graceful stop (60s drain) by label/name
```

---

## Measured Results (1x DGX Spark GB10, 2,200 MHz clock lock)

All runs below were measured on a single NVIDIA DGX Spark (GB10 `sm_121`, 48 SMs, 121.69 GiB unified memory, `nvidia-lock-clocks.service` at 2,200 MHz, `MEMORY_RESERVE_GIB=13.0`, `pswpout_delta=0` throughout). Inner-loop and outer-loop/gate rulers are reported separately because thinking mode, sampling temperature, and concurrency change acceptance and throughput.

### Outer-Loop & Qualification Ruler (`tools/bench.py --ruler outer`, `tools/prefill.py`, `tools/startup.py`, `tools/gates.py`)

| Metric | M1 Baseline (`tf-inkling2:m1`) | Qualified (`tf-inkling2:m2-p9`) | Notes / Harness |
|---|---|---|---|
| **Startup (Cold / Warm)** | `40.63 s` / `32.59 s` | **`15.30 s`** / **`14.29 s`** | `tools/startup.py --mode both` until `/v1/models` 200 OK |
| **KV Pool (`CONTEXT=1M, P=8`)** | `4,923,648` tok (`10G` res) | **`4,736,000` tok** (`13G` res) | Shared paged FP8 KV (`~4.5x` 1M streams, `0` swap-out) |
| **Sampled Prose (thinking off, c1/c2/c4/c8)** | `36.39` / `-` / `61.13` / `-` | **`42.25` / `53.75` / `72.26` / `94.40` tok/s** | `tools/bench.py --ruler outer` (default temp 1.0, median of 6) |
| **Thinking-On Prose (`T=0.7, k=20`, c1/c2/c4/c8)** | `40.74` / `-` / `60.82` / `-` | **`43.75` / `51.46` / `73.60` / `92.05` tok/s** | `tools/bench.py --ruler outer` (median of 6) |
| **Code (thinking off, default sampling, c1/c2/c4/c8)** | `-` | **`48.37` / `56.88` / `80.09` / `98.37` tok/s** | `tools/bench.py --ruler outer` (median of 6) |
| **Greedy Prose (thinking off, c1/c2/c4/c8)** | `39.78` / `-` / `63.09` / `82.75` | **`46.21` / `58.85` / `74.27` / `95.62` tok/s** | `tools/bench.py --ruler outer` (`c1` TTFT `0.177 s`, `51.25 ms/rnd`) |
| **Cold Prefill `4K` / `8K` / `16K`** (`cached=0`) | `701.9` / `-` / `-` tok/s | **`2,347.1` / `2,522.7` / `2,563.8` tok/s** | `tools/prefill.py` (`runs=2` median, strict `cached_tokens == 0`) |
| **Cold Prefill `32K` / `65K` / `128K`** (`cached=0`) | `696.6` / `-` / `649.3` tok/s | **`2,464.6` / `2,275.9` / `1,980.4` tok/s** | `tools/prefill.py` (`runs=2` median, strict `cached_tokens == 0`) |
| **Acceptance Gates (`tools/gates.py`)** | - | **12 / 12 PASS** | Includes `needle_256k` (`255K`), `needle_1m` (`1,023,701`/`1,023,881`), `soak` `c4`+`c8`, `vision` (PNG/JPEG/WebP), `tools`, `structured`, `prefix_cache` (`8704/8726`), `multilingual` |
| **Drafted vs Plain Exactness** | - | **4 / 4 EXACT** | Greedy `T=0`, `DFlash2` vs `DRAFTER=""` (`all_exact: true`) |
| **Unified Memory Peak (`needle_1m` + `soak`)** | `7.2 GiB` avail (`+2G` swap) | `min_mem_avail 8.36 GiB`, `pswpout 0` | `tools/memwatch.sh` (GPU res `103.91 GiB`, proc anon `2.93 GiB`) |

### Inner-Loop Deterministic Ruler (`cand-25-m2-p9`, `tools/bench.py --ruler inner`, `T=0`, `runs=6`)

| Workload | `c1` | `c4` agg | `c8` agg |
|---|---:|---:|---:|
| Prose (greedy, thinking off) | **45.64 tok/s** (`52.94 ms/rnd`, `1.426 tok/rnd` at c4) | **78.05 tok/s** | **96.97 tok/s** |
| Code (greedy, thinking off) | **49.59 tok/s** (`52.51 ms/rnd`) | **83.99 tok/s** | **116.74 tok/s** |

*Note on thinking effort, dense FP8 flags, and NLL harness:* `Inkling-2-Mini`'s chat template emits `Thinking effort level: 0` when `enable_thinking=False`; the model weights may still emit a brief separated `<|content_thinking|>` segment before the final answer, which TensorFold places in `reasoning_content` cleanly. `TF_INKLING2_ATTN_FP8=1` and `TF_INKLING2_GATE_FP8=1` trade a small amount of greedy code acceptance for +17-19% prose decode speed; unset them in `EXTRA_ENV` if you want pure BF16 dense projections. Note that paired teacher-forced NLL (`tools/nll.py`) is currently unsupported by TensorFold's `inkling2` backend (`prompt_logprobs` is unimplemented in `loop_backend.zig` / `cuda.zig`, returning HTTP 400); quality parity is verified instead via `plain` (`DRAFTER=""`) vs `drafted` (`DFlash2`) 4/4 greedy exactness and 12/12 `tools/gates.py` (including 256K and 1M needle retrieval).

---

## Requirements

- **Hardware:** 1x NVIDIA DGX Spark (GB10, `sm_121`, 128 GB unified LPDDR5X).
- **Driver & Runtime:** NVIDIA R580+ driver (`580.178.04`, CUDA 13.0) and Docker with the NVIDIA Container Toolkit (`--gpus all`).
- **Host Tools:** `docker`, `curl`, `hf` (Hugging Face CLI), `ss`, `free`, `stat`, `awk`, `grep` (and `jq` for `docker/verify-image.sh`, `shellcheck` for script lint).
- **Disk & Memory:** ~22 GB free under `HF_CACHE` (`thinkingmachines/Inkling-2-Mini` is 20.16 GB; `incoai/Inkling-2-Mini-DFlash2` is 0.99 GB) and `MemAvailable >= 108 GiB` before startup (`MEM_GATE_GIB`).

---

## Quick Start & Building from Source

### 1. Build the Patched Image

`engine-patches/` holds the 14 Apache-2.0 patches (`0001..0014` in `engine-patches/series`) against `ashhart/TensorFold` `v1.3.5` (`02b324ae9767253d9911d753d5c835a5070eed1b`).

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

To run without DFlash2 or without vision:
```sh
DRAFTER="" ./run.sh        # plain serving (no drafter)
VISION=0 ./run.sh          # text-only (saves vision buffers)
EXTRA_ENV="TF_INKLING2_PREFILL_TC=1" ./run.sh   # BF16 attention/gate projections
```

---

## Configuration (`recipe.conf`)

Any variable in `recipe.conf` can be overridden from the environment (e.g., `PORT=8001 PARALLEL=4 ./run.sh`).

| Variable | Default | Description |
|---|---|---|
| `MODEL`, `REVISION` | `thinkingmachines/Inkling-2-Mini` @ `c9d02830...` | Pinned target model (`1` model shard `19,526,019,168 B` + `mtp.safetensors` `635,801,416 B`). |
| `DRAFTER`, `DRAFTER_REVISION`, `DRAFTER_BITS` | `incoai/Inkling-2-Mini-DFlash2` @ `4f9088ea...`, `4` | Pinned DFlash2 drafter (`989,975,616 B`). Set `DRAFTER=""` to disable. |
| `BASE_IMAGE`, `IMAGE` | `tf-inkling2:1.3.5`, `tf-inkling2:m2-p9` | Base runtime and patched serving image. |
| `SERVED_NAME`, `PORT` | `inkling-2-mini`, `8000` | Model ID and HTTP listen port (`0.0.0.0`). |
| `CONTEXT`, `PARALLEL` | `1048576`, `8` | 1M context window (`--context`) and max concurrent streams (`--parallel`). |
| `VISION`, `VIDEO_TOKENS` | `1`, `64` | Enables `--vision` with `--vision-video-tokens 64`. |
| `MEMORY_RESERVE_GIB` | `13.0` | `TENSORFOLD_MEMORY_RESERVE_GIB` host safety reserve. |
| `ENGINE_CACHE` | `tf-inkling2-cache-135` | Persistent Docker volume at `/root/.cache/tensorfold` (shard hashes & window cost cache). |
| `EXTRA_ENV` | `TF_INKLING2_ATTN_FP8=1 TF_INKLING2_PREFILL_TC=1 TF_INKLING2_GATE_FP8=1` | Patched engine environment flags. |
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
  - [MiaAI-Lab/Inkling-2-Mini-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Inkling-2-Mini-DGX-Spark-TensorFold) @ `9e3cf20b76e0140bebbbe6ed13f6e3a00e881f12` (Apache-2.0): `ATTN_FP8`, Blackwell `PREFILL_TC`, lazy head, vision, DFlash2 chances, and server compatibility fixes.
  - [tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark) @ `2fe0eb57429013c0ccb1c6535339ae61a9e44d9a` (MIT / Apache-2.0): NVFP4 draft head & batched DFlash2 kernel concepts.
  - [sxuff/inkling-2-mini-spark](https://github.com/sxuff/inkling-2-mini-spark) @ `35a4fd8344f01148d35c899cc107aab0c537ce3d` (Apache-2.0): benchmark framing and eager safetensors loading insights.
- **Runtime Libraries & Toolchain:** MLC-AI team & DLPack contributors (`xgrammar` `v0.2.8` Apache-2.0, `dlpack` Apache-2.0, `picojson` BSD-2-Clause), IJG / libjpeg-turbo contributors (`libjpeg-turbo` `3.1.4.1`, https://libjpeg-turbo.org, IJG / BSD-3-Clause / zlib), Chromium authors (`libwebp` `1.6.0`, https://chromium.googlesource.com/webm/libwebp, BSD-3-Clause), Zig Software Foundation & contributors (`Zig` `0.17.0`, https://ziglang.org, MIT, build-time only), and NVIDIA (CUDA `13.0.0`, `libcublas-13-0`, NVIDIA Software License Agreement).
