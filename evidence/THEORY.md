# Architecture X-Ray & Theoretical Ceilings: Inkling-2-Mini + DFlash2 on 1x DGX Spark

## 1. Hardware Ground Truth (`dgx-spark`, GB10 `sm_121`)

From `knowledge/hardware-2x-spark.md`:
- **Architecture:** NVIDIA GB10 Grace Blackwell (`sm_121`), single rank (`TP=1`).
- **Unified Memory:** **121.6 GiB** visible to Linux (`121 GiB` usable cap; **>= 10.0 GiB host reserve** leaves **<= 111.0 GiB** total budget for weights + KV + scratch + CUDA graphs).
- **Clock Lock:** **2,200 MHz** via `nvidia-lock-clocks.service` (never modify).
- **Measured Device Copy Bandwidth (read + write):** **224.3 GB/s** (`dgx-spark`; spec 273 GB/s). Used as the practical bandwidth anchor for all decode ceilings below. (Note: read-only weight streaming on GB10 can reach ~245-255 GB/s in isolated GEMMs, so real kernels may land near 90-105% of the 224.3 GB/s R+W copy ruler.)
- **Measured BF16 GEMM Peak (`8192^3`):** **95.4 TFLOPS** (practical 50% MFU = **47.7 TFLOPS** for BF16 linear + attention FLOPs; FP8/FP4 Tensor Cores raise linear MoE/GEMM peak on `sm_121`).

---

## 2. Exact Checkpoint & Architecture X-Ray

Verified directly from Hugging Face `config.json` and safetensors headers:
- **Target (`thinkingmachines/Inkling-2-Mini`, rev `c9d028302cd0f035fdf8e7e91f089a4ce94d2195`, 20.19 GB total on disk):**
  - `model-00001-of-00001.safetensors`: `19,525,870,752` B tensor payload (`19.53 GB`); `mtp.safetensors` (DSpark, unused when DFlash2 runs): `635,795,744` B (`0.64 GB`).
  - **Looped MoE Schedule:** `24` physical layers; prelude `0..5` (1x), core `6..17` (looped **3x** = 36 visits), epilogue `18..23` (1x) = **48 layer visits per token**.
  - **Attention Geometry:** `hidden_size = 2304`, `24` Q heads, `8` KV heads, `head_dim = 128`, `d_rel = 16`, `rel_extent = 1024`, `sconv_kernel_size = 4`, `gate_rank = 64` controllers on core layers `6..17`.
  - **Hybrid Global / Sliding Schedule:** Physical layers `[5, 11, 17, 23]` are global -> **8 global visits** (`5`, `11`x3, `17`x3, `23`) and **40 sliding-window visits** (`sliding_window_size = 1024`).
  - **MoE:** `128` experts per layer, `top_k = 8`, `intermediate_size = 1152`, no shared experts.
    - Layers `0..19`: **NVFP4** (`11.4662 GB` unique on disk; `25.2256 GB` across 48 visits).
    - Layers `20..23`: **MXFP8** (`4.2043 GB` unique on disk; `4.2043 GB` across 48 visits).
    - Total 48-visit routed expert pool: **`29.4299 GB`**.
  - **Vocabulary:** `vocab_size = 201024`, `unpadded_vocab_size = 200058`.
    - `embed.weight` / `unembed.weight` (BF16): `0.9263 GB` padded (`0.9219 GB` unpadded `200058 * 2304 * 2`).
  - **Vision Tower:** `23`-layer `inkling2_vision` ViT (`0.6365 GB` BF16).

- **Drafter (`incoai/Inkling-2-Mini-DFlash2`, rev `4f9088ea66c21f2b99ed657d1941c306b03e84d2`, `0.9900 GB` payload):**
  - `5` Qwen3-style non-causal sliding layers (`window = 2048`), `hidden_size = 2304`, `intermediate_size = 6912`, `32` Q heads / `8` KV heads (`head_dim = 128`), two-tap dynamic convs (`conv_kernel_size = 2`, `conv_group_size = 16`), `block_size = 8` (`7` draft tokens/step), taps `[5, 23, 41, 48]`.
  - Candidate selector: `selector_rank = 256`, `selector_top_k = 16`, `predecessor_codebook` + `successor_codebook` (`2 * 201024 * 256` BF16 = `0.2058 GB`, of which only `16` rows/step are read in the selector walk).

### Exact Per-Step Byte Breakdown (48 Target Visits + Unembed + DFlash2)

| Component | Unique On Disk (GB) | 48-Visit BF16 Traffic (GB) | 48-Visit FP8 Opt Traffic (GB) | Notes |
|---|---:|---:|---:|---|
| Attention `wq/wk/wv/wo` | 0.9059 | 1.8119 | 0.9060 | `TF_INKLING2_ATTN_FP8=1` (per-tensor FP8) cuts in half |
| Attention `gate_attn` | 0.3397 | 0.6795 | 0.6795 (or 0.3397) | `0.3397 GB` if also FP8 |
| `wr`, `sink`, `conv`, `norm`, `router`, `ctrl`, `cross_pass` | 0.1201 | 0.2968 | 0.2968 | Loop controllers (`0.1699 GB`), `wr` (`0.0849 GB`), `router` (`0.0283 GB`) |
| **Total Target Dense (excl. LM head)** | **1.3657** | **2.7882** | **1.8822** | Core layers `6..17` exceed L2 (~24 MB), re-read per pass |
| Routed Experts (`R=1`, 8/128 = 6.25%) | 15.6705 (pool) | 1.8394 | 1.8394 | NVFP4 (L0-19) + MXFP8 (L20-23) |
| Target `unembed` (LM head) | 0.9219 | 0.9219 | 0.4609 | `TF_INKLING2_LAZY_HEAD=1` (greedy FP8 proxy) |
| **Target `R=1` Total (no KV)** | **-** | **5.5495** | **4.1825** | Dense = **50.2%** (BF16) / **45.0%** (FP8+lazy) of `R=1` bytes |
| DFlash2 5 layers + `fc` + `sel_proj` | 0.7841 | 0.7841 | 0.4166 | `TF_INKLING2_DRAFT_FP8_LAYERS=1` (`~0.531x`) |
| DFlash2 base head (7 draft rows) | 0 (shared) or 0.2593-0.4609 | 0.9219 | 0.4609 (FP8) / 0.2593 (NVFP4) | Borrowed FP8 lazy head saves VRAM; NVFP4 saves step BW |

---

## 3. MoE Active Expert Union (1..32 Verify Rows)

For `128` experts, `top_k = 8`, uniform routing union is `1 - (1 - 8/128)^R`. Real prose/code routing is skewed across experts (modeled via Zipf `alpha = 0.55` across 128 experts, reflecting shared syntax/function-word routing across adjacent tokens in a block or batch).

| Verify Rows `R` | Uniform Union (%) | Uniform Expert Bytes (GB) | Skewed Union (%) | Skewed Expert Bytes (GB) | Notes |
|---:|---:|---:|---:|---:|---|
| 1 | 6.2% | 1.839 | 6.2% | 1.839 | Autoregressive c1 (`8/128`) |
| 2 | 12.1% | 3.564 | 11.8% | 3.479 | `c2` autoregressive or `c1` 1-draft |
| 4 | 22.8% | 6.696 | 21.5% | 6.315 | `c4` autoregressive or `c1` capped `d=3` |
| 8 | 40.3% | 11.869 | 36.6% | 10.775 | `c1` full DFlash2 (`d=7`) or `c8` autoregressive |
| 12 | 53.9% | 15.864 | 48.1% | 14.168 | `c4` with `d=2` drafts (`4 * 3`) |
| 16 | 64.4% | 18.951 | 57.2% | 16.832 | `c4` with `d=3` drafts (`4 * 4`) |
| 24 | 78.8% | 23.177 | 70.3% | 20.688 | `c4` with `d=5` or `c8` with `d=2` |
| 32 | 87.3% | 25.699 | 79.0% | 23.261 | `c4` full `d=7` (`4 * 8`) or `c8` with `d=3` |

**Key Architectural Insight:** Because `R=8` pulls **36.6%-40.3%** of all 48-visit experts (`10.8-11.9 GB`), DFlash2 at `c1` only wins when acceptance `a` grows faster than `(verify_bytes + draft_bytes) / c1_bytes`, AND **dynamic draft caps / chances (`d=3..5` on prose, `d=7` on high-acceptance code)** pay hugely by shrinking `R`. At `c4` (`R=16..32`), unbatched DFlash2 would read drafter weights `4x` (`1.67-3.6 GB` extra); **batching DFlash2 across streams** reads drafter weights once (`0.42-0.68 GB`).

---

## 4. Decode Ceilings (Bandwidth = 224.3 GB/s, 4K vs 256K vs 1M FP8 KV)

Using `scripts/ceiling.py` (`ceiling.json`, FP8 attn `1.8822 GB` + lazy FP8 head `0.4609 GB` + FP8 DFlash2 layers `0.4166 GB` + NVFP4 draft head `0.2593 GB`, `kv_bytes_per_token = 16896` B/tok):

### `ceiling.py` Output (Unbatched Drafter, Uniform MoE Union)

| Mode / Workload | Drafts `k` | Mean Accept `a` | `c1` @ 4K | `c1` @ 32K | `c1` @ 256K | `c1` @ 1M | `c4` @ 4K | `c4` @ 256K | `c8` @ 4K | `c8` @ 256K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Autoregressive (FP8 attn + lazy) | 0 | 1.0 | **53** | 47 | 26 | 10 | **96** | 34 | **122** | 36 |
| DFlash2 Prose (typical, `d=7`) | 7 | 2.9 | **43** | 42 | 34 | 20 | **84** | 54 | **140** | 72 |
| DFlash2 Prose (capped `d=4`) | 4 | 2.6 | **53** | 51 | 38 | 20 | **91** | 54 | **141** | 69 |
| DFlash2 MT-Bench Ref (`d=7`) | 7 | 3.9 | **58** | 57 | 45 | 27 | **113** | 72 | **188** | 97 |
| DFlash2 Code / HumanEval (`d=7`) | 7 | 4.8 | **72** | 70 | 56 | 33 | **139** | 89 | **231** | 119 |

### Baseline BF16 vs Batched DFlash2 + Skewed MoE Comparison (`4K` / `256K` tok/s)

1. **BF16 Dense (`2.7882 GB`) + BF16 Head (`0.9219 GB`) + BF16 DFlash2 (`1.7060 GB`):**
   - `c1` @ 4K / 256K: Autoregressive **40 / 22**, DFlash2 Prose (`a=2.9`) **37 / 30**, MT-Bench (`a=3.9`) **50 / 40**, Code (`a=4.8`) **62 / 50**.
   - `c4` @ 4K / 256K: Autoregressive **84 / 32**, DFlash2 Prose (`a=2.9`) **71 / 48**, MT-Bench (`a=3.9`) **96 / 65**, Code (`a=4.8`) **118 / 80**.
   - `c8` @ 4K / 256K: Autoregressive **111 / 35**, DFlash2 Prose (`a=2.9`) **111 / 64**, MT-Bench (`a=3.9`) **149 / 86**, Code (`a=4.8`) **184 / 105**.
2. **Batched DFlash2 + FP8 Attn + Lazy Head + Skewed MoE (`Zipf alpha=0.55`):**
   - Batching DFlash2 across streams reads drafter weights **once** per round instead of `N` times; combined with skewed MoE expert sharing:
   - `c1` @ 4K / 256K: Prose (`d=4, a=2.6`) **55 / 39**, MT-Bench (`a=3.9`) **63 / 48**, Code (`a=4.8`) **78 / 59** (and on repetitive code with ~22-25% MoE union + read-only ~250 GB/s streaming, `c1` reaches ~100-115 tok/s).
   - `c4` @ 4K / 256K: Autoregressive **100 / 34**, Prose (`d=4, a=2.6`) **105 / 59**, MT-Bench (`a=3.9`) **132 / 80**, Code (`a=4.8`) **162 / 98**.
   - `c8` @ 4K / 256K: Autoregressive **131 / 37**, Prose (`d=7, a=2.9`) **166 / 79**, MT-Bench (`a=3.9`) **223 / 106**, Code (`a=4.8`) **275 / 130**.
3. **Why KV Matters at 256K-1M:** Each of the 8 global visits reads `16,896 B/tok` in FP8 (`4.43 GB` at 256K, `17.72 GB` at 1M per stream). Speculative decoding amortizes that global KV read across `a` accepted tokens, lifting `c1` @ 1M from `10 tok/s` (autoregressive) to `33 tok/s` (DFlash2 `a=4.8`).

---

## 5. Prefill, TTFT & Startup Ceilings

- **Effective Active Linear Params (48 visits + unembed):** `4.909B` (`3.058B` MoE + `1.851B` dense/unembed) -> `9.818e9` FLOPs/token.
- **Attention FLOPs/token:**
  - `40` sliding visits (`W=1024`, `24` Q heads, `dim=128`): `4 * 40 * 24 * 128 * 1024 = 0.503e9` FLOPs/token (+ `~0.060e9` relative-pos/conv = `~0.563e9` fixed).
  - `8` global visits (causal over `L`, mean `L/2`): `4 * 8 * 24 * 128 * (L / 2) = 49,152 * L` FLOPs/token.
- **Practical Prefill Ceiling (`0.5 * 95.4 TFLOPS = 47.7 TFLOPS` BF16-equivalent):**
  - `4K` (`4,096`): `10.58e9` FLOPs/tok -> **`4,508 tok/s`** (`~4,500 tok/s` BF16; higher with FP4/FP8 TC prefill `TF_INKLING2_PREFILL_TC=1`)
  - `32K` (`32,768`): `11.99e9` FLOPs/tok -> **`3,978 tok/s`**
  - `128K` (`131,072`): `16.82e9` FLOPs/tok -> **`2,835 tok/s`**
  - `256K` (`262,144`): `23.27e9` FLOPs/tok -> **`2,050 tok/s`**
  - `1M` (`1,048,576`): `61.92e9` FLOPs/tok -> **`770 tok/s`**
- **Short-Prompt TTFT Floor (e.g., `<= 256` tokens):**
  - One forward pass at `R=1..256`: at `R=64` (short chat prompt), MoE union approaches `98.4%` (`~29 GB`), dense + unembed `~2.3 GB` -> ~`31 GB / 224.3 GB/s` = **`~140 ms`** bandwidth floor at `R=64` (or **`~20-25 ms`** at `R=1..2`, **`~50-75 ms`** at `R=16-32` + tokenize/schedule). Skipping 4K chunk scratch/allocation overhead on short prompts keeps TTFT `< 80-150 ms`.
- **Startup Floor:**
  - Disk payload: `19.53 GB` (target) + `0.99 GB` (DFlash2) = `20.52 GB` (skipping `mtp.safetensors` when DFlash2 is active).
  - Warm page cache / NVMe (~6-10 GB/s): **`2.1-3.4 s`** raw read + **`~2-4 s`** CUDA arena alloc/FP8 packing + **`~3-5 s`** cached graph/cost load -> **`~8-12 s` warm startup floor** (with parallel `FAST_LOAD`, persistent `/root/.cache/tensorfold` shard hashes + window costs, and minimal graph warmup).

---

## 6. Exact KV Capacity & Memory Budget (121 GiB Unified RAM, >= 10 GiB Reserve)

**Available Budget:** `121.6 GiB - 10.6 GiB reserve = ~111.0 GiB` (`119.2 GB` decimal).

### Fixed Weights & Shared Scratch (approximate; measured on boot by `@bench-opt`):
- Target weights (ModelOpt NVFP4/MXFP8 + BF16 dense/embed/unembed): `~17.60 GiB` (text-only; `+0.59 GiB` with `vision` tower = `~18.19 GiB`).
- Lazy head FP8 copy: `+0.43 GiB`; DFlash2 weights (FP8 layers + borrowed FP8 head, or NVFP4 head): `~0.65-0.85 GiB`.
- Shared engine scratch + CUDA graphs + vision working buffers: `~2.5-3.5 GiB` (at `prompt_rows=4096`).
- **Total Fixed Non-Pool Footprint:** **`~22.0-23.0 GiB`**, leaving **`~88.0-89.0 GiB` (`~94.5 GB`)** for active streams + shared paged global KV pool!

### Per-Stream Fixed Overhead (Independent of Context Length):
1. **Sliding-window KV (40 visits):** `align64(1024 + prompt_rows - 1)` tokens.
   - At `prompt_rows = 2048` (`3,072` tokens): **BF16 = `0.469 GiB`**, **TF FP8 (`132 B/head`) = `0.242 GiB`**.
   - At `prompt_rows = 4096` (`5,120` tokens): **BF16 = `0.781 GiB`**, **TF FP8 (`132 B/head`) = `0.403 GiB`**.
2. **Conv state + backup + 16-row rewind history (48 visits x 4 sites):** `~26.8 MiB` (`28.11 MB`).
3. **DFlash2 fixed ring (5 layers x 2055 tokens BF16, 8 KV heads x 128):** **`40.14 MiB`** (`42.09 MB`).
4. **Lane logits + page table (`1M / 256 = 4,096` pages x `4 B`) + vision slot:** `~5-10 MiB`.
- **Total Fixed Per-Stream Overhead (TF FP8, `prompt_rows=4096`):** **`~0.48 GiB/stream`** (`~3.85 GiB` for `PARALLEL=8`).

### Global KV Bytes Per Token (8 Global Visits, 2 (K+V), 8 KV Heads, `head_dim = 128`):
- **BF16:** `8 * 2 * 8 * 128 * 2 = 32,768 B/tok` (`32 KiB/tok` = **`32.00 GiB` per 1M stream**, `8.00 GiB` per 256K stream).
- **TensorFold FP8 (`128 B` code + `4 B` fp32 scale per head-token):** `8 * 2 * 8 * 132 = 16,896 B/tok` (**`16.50 GiB` per 1M stream**, `4.125 GiB` per 256K stream).

### Maximum Resident Capacity Under `~88 GiB` Remaining Pool (`>= 10 GiB` Host Reserve):
- **Shared Paged FP8 Pool (`~84 GiB` pool after `8 x 0.48 = 3.85 GiB` per-stream fixed state):**
  - `84 GiB / 16,896 B/tok` = **`~5.34M` paged global KV tokens**!
  - Holds **`5.09x` full 1,048,576-token streams** simultaneously (exceeding `3+` 1M streams comfortably, or **`8` full `262K` streams (`2.10M` tokens) + `~51 GiB` (`~3.2M` tokens) of live LRU prefix prompt cache**!).
- **BF16 KV Comparison (`~81.5 GiB` pool after `8 x 0.85 = 6.8 GiB` BF16 sliding+conv):**
  - `81.5 GiB / 32,768 B/tok` = **`~2.67M` tokens** (`2.55x` 1M streams, or `8 x 262K` + `~17.5 GiB` prefix cache).
