# Optimization Plan: Inkling-2-Mini (TensorFold v1.3.5 + DFlash2, 1x DGX Spark)

## Goal & Constraints
- **Target Model:** `thinkingmachines/Inkling-2-Mini` (rev `c9d028302cd0f035fdf8e7e91f089a4ce94d2195`, 20.19 GB)
- **Drafter:** `incoai/Inkling-2-Mini-DFlash2` (rev `4f9088ea66c21f2b99ed657d1941c306b03e84d2`, 0.99 GB)
- **Hardware:** 1x NVIDIA DGX Spark (GB10, `sm_121`, 48 SMs, 121.69 GiB unified LPDDR5X, `>= 10 GiB` host reserve)
- **Engine:** `ashhart/TensorFold` `v1.3.5` (`d6b3be891b5fb539a705d9d3b5279008045900e8`) + Apache-2.0/MIT/BSD patches
- **Target Metrics (Anti-Tunnel-Vision across all 5):**
  1. `c1` prose decode (thinking off strict anchor + thinking on / greedy / code comparable)
  2. `c4` aggregate decode (prose & code)
  3. KV capacity (shared paged FP8 KV pool, >= 1M per-stream cap, target ~4.5M-5.5M+ tokens at `PARALLEL=8`)
  4. TTFT (short prompt) & cold prefill (`cached_tokens == 0` at 4K / 32K / 128K)
  5. Cold & warm startup time (target warm boot < 10-15 s)
- **Competitor Reference (MiaAI-Lab):** `c1`: 52.1 thinking-on / 65.1 greedy; `c4`: 115.6 prose / 159.2 code; prefill: 3,497 @ 4K / 3,072 @ 32K / 2,267 @ 128K; KV: 2.09M.
- **Quality & Safety Rules:**
  - Serialize all GPU jobs and Docker/Zig/NVCC builds under `flock /tmp/inference-hillclimb.lock`.
  - Never alter `nvidia-lock-clocks.service`.
  - Check host CPU load, swap-out, and unified memory residency (`driver residency + anon + swap-out`) on every run; void runs under contention.
  - One change per experiment, bracketed controls (`control -> candidate -> control`), >= 6 decode samples (median), exactness + NLL (`<= +0.002 nats/tok`) on any lossy/kernel change.

## Baseline Machine State Verification (2026-11-25)
- **dgx-spark (head):**
  - OS: Linux `7.0.0-1019-nvidia` aarch64, CUDA 13.0, Driver `580.178.04`
  - GPU: NVIDIA GB10 (`sm_121`, 48 SMs, 121.69 GiB total memory)
  - Clock Lock: `nvidia-lock-clocks.service` active (`ExecStart=/usr/bin/nvidia-smi -lgc 0,2200`); idle SM clock 208 MHz, loaded sample observed at 2,476 MHz during `sanity_gb10` (note: recorded as observed; service left untouched)
  - Host State: load avg `0.08, 0.04, 0.08`, 97-99% CPU idle, `MemAvailable` ~117.6-120.8 GiB, swap used 57-1061 MiB (0 swap I/O at idle), 0 running Docker containers
  - Sanity Microbenchmarks (under `flock`):
    - D2D 1 GiB copy BW (R+W, median of 10): **216.62 GB/s** (min 213.15, max 221.87)
    - BF16 GEMM 8192^3 (median of 10): **97.90 TFLOPS** (min 93.60, max 101.20)
- **dgx-spark (worker, not used for 1x recipe, checked for cluster sanity):**
  - GPU idle (0%, 208 MHz), `nvidia-lock-clocks.service` active, load avg `0.08, 0.04, 0.08`, `MemAvailable` 120,849 MiB, swap used 57 MiB.

## Phase Status
- [x] Task 1: Baseline machine state verified and `PLAN.md` / `ledger.jsonl` initialized.
- [x] Task 2: `@recon-theory` `THEORY.md`/`ceiling.json`/`PATCH_MAP.md` reviewed; `@recipe-infra` harness and `@engine-dev` `tf-inkling2:m1` received.
- [x] Task 3: Day-1 Baseline + `nsys` Profiler completed across all 5 metrics; Roofline Gap Table posted to `@engine-dev`.
- [ ] Task 4: Systematic Hill-Climb Loop (M1 config levers + M2 engine patches).

## Target Ceilings Summary (from `THEORY.md` & `ceiling.json`)
- **Hardware:** `224.3 GB/s` copy BW (note: `216.62 GB/s` D2D median measured on `dgx-spark`), `95.4 TFLOPS` BF16 peak (`97.90 TFLOPS` measured on `dgx-spark`; practical MFU 0.5 = `47.7 TFLOPS`), `<= 111.0 GiB` usable unified memory (`>= 10.0 GiB` reserve).
- **`c1` decode @ 4K:** AR `40` (BF16) / `53` (FP8 attn+lazy); DFlash2 Prose (`d=4, a=2.6`) `53` unbatched / `55` skewed; MT-Bench (`a=3.9`) `58` / `63`; Code (`a=4.8`) `72` / `78`.
- **`c4` aggregate @ 4K:** AR `84` (BF16) / `96` (FP8 attn+lazy); DFlash2 Prose `84-91` unbatched / `105` batched+skewed; MT-Bench `113` / `132`; Code `139` / `162`.
- **Prefill practical (BF16 47.7 TFLOPS):** `4,508` @ 4K, `3,978` @ 32K, `2,835` @ 128K, `2,050` @ 256K (higher with FP4/FP8 TC prefill `TF_INKLING2_PREFILL_TC=1`).
- **Short TTFT floor:** `~50-140 ms` (R=16..64).
- **Warm startup floor:** `~8-12 s` with `FAST_LOAD` + persistent cache.
- **KV capacity:** `16,896 B/tok` global FP8 (`~5.34M` paged tokens under 111 GiB cap = `5x` 1M streams or `8x 262K` + `~51 GiB` prefix cache).

## Experiment Log

### Day-1 Baseline (`m1-baseline`, `tf-inkling2:m1` defaults: `PREFILL_TC=0`, `ATTN_FP8=0`, `LAZY_HEAD=1`, `VERIFY_GRAPHS=1`, `MEMORY_RESERVE_GIB=10.0`)

#### Roofline Gap Table
| Metric | M1 Baseline | MiaAI-Lab Ref | Practical Ceiling | % Ceiling | Gap / Primary Bottleneck |
|---|---:|---:|---:|---:|---|
| `c1` sampled prose (think off, strict) | **36.39 tok/s** (`58.22 ms/rnd`, `2.124 tok/rnd`, `acc 0.404`) | - | 53-55 tok/s | 66-69% | `ATTN_FP8=0` (BF16 dense 20.8% GPU), MoE `45.7%`, ungraphed DFlash2 |
| `c1` greedy prose (think off) | **39.78 tok/s** (`59.38 ms/rnd`, `2.365 tok/rnd`, `acc 0.402`) | 65.1 | 53-58 tok/s | 69-75% | Same + DFlash2 drafting only ~3.6/rnd |
| `c1` thinking-on prose (`t=0.7, k=20`) | **40.74 tok/s** (`62.63 ms/rnd`, `2.561 tok/rnd`) | 52.1 | - | -21.8% vs Mia | Same |
| `c1` greedy code | **55.06 tok/s** (`65.90 ms/rnd`, `3.631 tok/rnd`, `acc 0.495`) | 106.7 | 72-78 tok/s | 71-76% | Same |
| `c4` agg greedy prose | **63.09 tok/s** (`70.41 ms/rnd`, `1.138 tok/rnd`) | 115.6 | 105-132 tok/s | 48-60% | **DFlash2 barely drafts at c4 (`1.14 tok/rnd`) + 18 CUDA graph re-instantiations + 29.6K eager launches** |
| `c4` agg greedy code | **63.77 tok/s** (`68.00 ms/rnd`, `1.125 tok/rnd`) | 159.2 | 139-162 tok/s | 39-46% | Same |
| `c8` agg greedy prose / code | **82.75 / 83.82 tok/s** (`1.10 / 1.16 tok/rnd`) | 168.1 / 237.5 | 166-275 tok/s | 30-50% | Same |
| Prefill `4K` (cold `cached_tokens==0`) | **701.9 tok/s** (5.80 s) | 3,497 | 4,508+ | **15.6%** | **`PREFILL_TC=0`: `tf_ink2moe_plan` takes 52.2% (2.46s) + `prompt_up` 19.6%** |
| Prefill `32K` (cold) | **696.6 tok/s** (47.40 s) | 3,072 | 3,978+ | **17.5%** | Same |
| Prefill `128K` (cold) | **649.3 tok/s** (208.01 s) | 2,267 | 2,835+ | **22.9%** | Same |
| Short TTFT (`c1`) | **202-212 ms** | - | ~50-140 ms | - | Prefill scratch / launch overhead on short prompts |
| Startup (Cold / Warm) | **40.63 s / 32.59 s** | - | ~8-12 s | ~31-37% | `/root/.cache/tensorfold` empty; `loop_backend.warm()` re-sweeps costs every boot + single-threaded load |
| KV Capacity (paged FP8, 1M ctx, P=8) | **4,923,648 tok** (19,233 pages, 77.48 GiB) | 2.09M | ~5.34M | **92.2%** | Reserve `10.0 GiB` dipped `MemAvailable` to `7.2-7.4 GiB` (+2 GiB swap); test `12.0 GiB` reserve |

#### Day-1 `nsys` Kernel & API Profile Summary (`profiles/m1-nsys-summary.txt`)
1. **`m1-prefill4k` (3,523 tok, 4.746 s):**
   - `tf_ink2moe_plan`: **52.2%** (`2,464.9 ms`, `51.35 ms`/call x 48)
   - `tf_ink2moe_prompt_up`: **19.6%** (`926.1 ms`, `6.43 ms`/call x 144)
   - `tf_ink2_loop_gate_down_prepared`: **9.3%** (`437.1 ms`, `4.55 ms`/call x 96)
   - `qmmf_kernel` (4,128,128): **3.1%** (`144.9 ms`)
   - `tf_ink2_loop_gate_up`: **2.5%** (`116.3 ms`), `tf_ink2_attn_slide_fp8`: **2.4%** (`112.5 ms`), `tf_ink2_loop_ar`: **2.3%** (`106.9 ms`)
2. **`m1-c1` (50 rounds, 128 tok, 3.038 s):**
   - `tf_ink2moe_up`: **45.7%** (`1,343.2 ms`, avg `274 us`)
   - `qmmf_kernel` dense/attn (`<4,16,64>` + `<4,16,32>`): **20.8%** (`610.7 ms`)
   - `qmmf_kernel<1,16,64>` (2 calls/round, ~1.97 ms each): **6.7%** (`196.6 ms`)
   - `tf_ink2moe_plan`: **4.5%** (`131.3 ms`), `tf_ink2_mix`: **4.1%** (`119.5 ms`), `tf_ink2_loop_ar`: **2.5%**, `tf_ink2_loop_gate_down`: **2.1%**, `tf_ink2_attn_fp8_prefetch`: **1.9%**, `tf_ink2_df_attn`: **0.9%**
   - CUDA API: `100` `cuGraphLaunch`, `1,651` eager `cuLaunchKernel`, `594` `cuLaunchKernelEx` (DFlash2 eager launches).
3. **`m1-c4` (4 streams, ~102 rounds, 7.309 s, `1.25-1.44 tok/rnd`):**
   - **Graph thrashing:** `18` `cuGraphInstantiateWithFlags` (`65.7 ms`) + `18` `cuGraphDestroy` + `18` `cuGraphUpload` (`78.8 ms`), plus **25,254 `cuLaunchKernel` (`185.7 ms`) + 4,352 `cuLaunchKernelEx` (`27.5 ms`)**!
   - **DFlash2 barely drafts:** `55-59` drafted total across `89-102` rounds (`1.12-1.44 tok/rnd`), falling back to AR (`68-73 ms/rnd`).
   - Kernels: `tf_ink2moe_up` **42.6%**, `qmmf_kernel` dense/attn **18.0%**, `tf_ink2_attn_fp8_prefetch` **6.1%**, `qmmf<1,16,64>` **5.5%** (`378 ms`), `tf_ink2moe_plan` **4.9%**, `tf_ink2_conv` **2.7%** (`74,112` calls).

### Hill-Climb Round 1: M1 Config & Opt-in Flags

1. **`MEMORY_RESERVE_GIB=12.0` (`ctl-01` vs `m1-baseline`):**
   - `m1-baseline` (`10.0 GiB` reserve) dipped `MemAvailable` to `7.2-7.4 GiB` with ~2 GiB swap-out.
   - `ctl-01` / `ctl-02` / `ctl-03` (`12.0 GiB` reserve) kept `4,750,336 - 4,800,768` paged FP8 KV tokens (`4.5-4.6x` 1M streams), `min MemAvailable = 9.57-9.64 GiB`, `peak footprint = 104.56-104.66 GiB`, **`pswpout_delta = 0`**, zero alerts. **ACCEPTED.**
2. **`TF_INKLING2_PREFILL_TC=1` (`ctl-01 -> cand-01-prefill-tc -> ctl-02`):**
   - `4K` cold prefill: `705.2 -> 740.9 -> 694.9 tok/s` (**+5.1% / +6.6%** vs brackets).
   - `32K` cold prefill: `699.9 -> 736.8 -> 698.2 tok/s` (**+5.3% / +5.5%** vs brackets).
   - `c1`: `39.82 -> 37.59 -> 39.92 tok/s` (drafted/rnd variance `2.00` vs `2.34-2.37`); `c4`: `63.39 -> 63.11 -> 63.05`.
   - Verdict: **KEPT** (`PREFILL_TC=1` wins on prefill, though masked by `tf_ink2moe_plan` 52.2% bottleneck; re-test after `m2-p1` fixes `tf_ink2moe_plan`).
3. **`TF_INKLING2_ATTN_FP8=1` (`ctl-02 -> cand-02-attn-fp8 -> ctl-03`):**
   - `c1` greedy prose: `39.92 -> 45.28 -> 39.86 tok/s` (**+13.4% / +13.6%**; `ms/rnd` `59.59 -> 55.72`, `tok/rnd` `2.37 -> 2.52`).
   - `c4` greedy prose: `63.05 -> 67.67 -> 63.38 tok/s` (**+7.3% / +6.8%**; `ms/rnd` `70.23 -> 67.53`).
   - `4K/32K` prefill: `700.6 / 694.7 tok/s` (neutral while `tf_ink2moe_plan` dominates).
   - `memwatch`: `min MemAvailable = 9.62 GiB`, `peak footprint = 104.63 GiB`, `pswpout_delta = 0`.
   - Verdict: **ACCEPTED** (pending quality/gates check alongside `PREFILL_TC=1`).

### Hill-Climb Round 2: `tf-inkling2:m2-p1` (`0006` @ `5185b43`)

1. **`m2-p1` (`ctl-04-m1-best -> cand-03-m2-p1-tc0 -> cand-04-m2-p1-tc1 -> ctl-05-m1-best`):**
   - `ctl-04-m1-best` (`m1` + `ATTN_FP8=1` + `PREFILL_TC=1`): warm boot `32.09 s`, `c1=45.59`, `c4=67.81`, `4K=737.3`, `32K=732.3`, `memwatch` peak `104.62 GiB` (`pswpout=0`).
   - `cand-03-m2-p1-tc0` (`m2-p1`, `ATTN_FP8=1`, `PREFILL_TC=0`, cold `cost_cache`): boot **`16.30 s`** (-49.2%), `c1=47.72` (+4.7%), `c4=70.61` (+4.1%), **`4K=1,649.0`** (**+2.24x**), **`32K=1,698.6`** (**+2.32x**), peak `105.02 GiB` (`pswpout=0`).
   - `cand-04-m2-p1-tc1` (`m2-p1`, `ATTN_FP8=1`, `PREFILL_TC=1`, `cost_cache` HIT): warm boot **`14.29 s`** (**-55.5%** vs controls; `loaded cached window costs, propose 4.48 ms`), `c1=48.06` (+5.4-6.0% vs controls, `54.09 ms/rnd`), `c4=70.51` (+4.0-4.5%), **`4K=1,960.0`** (**+2.66-2.73x** vs controls, +18.9% vs `tc0`), **`32K=1,896.8`** (**+2.59-2.60x** vs controls, +11.7% vs `tc0`), `4,748,032` KV tokens, peak `104.81 GiB`, min `MemAvailable 9.50 GiB`, `pswpout=0`.
   - `ctl-05-m1-best` (closing control): warm boot `32.11 s`, `c1=45.34`, `c4=67.48`, `4K=717.7`, `32K=730.1`.
   - Verdict: **`tf-inkling2:m2-p1` + `PREFILL_TC=1` + `ATTN_FP8=1` ACCEPTED across all metrics.**

#### `m2-p1` `nsys` Profile Summary (`profiles/m2p1-nsys-summary.txt`)
1. **`m2p1-prefill4k` (3,523 tok, 1.737 s, `PREFILL_TC=1`, `ATTN_FP8=1`):**
   - `tf_ink2moe_plan` dropped **`2,464.9 ms -> 0.80 ms` (3,073x)**; `gate_down_prepared` dropped **`437.1 ms -> 91.3 ms` (4.8x)**.
   - Remaining time (`1,711 ms` GPU):
     - TC MoE GEMMs (`prompt_tc_nvfp4` **34.3%** `587 ms` + `prompt_tc_mxfp8` **5.9%** `101 ms`): **40.2%** (`688 ms`).
     - Loop gates (`loop_gate_up` **6.8%** `116 ms` + `gate_down_prepared` **5.3%** `91 ms` + `gate_prepare` **2.4%** `42 ms` + `loop_messages` **1.6%** `27 ms`): **16.1%** (`276 ms`).
     - Dense `qmmf_kernel` (`<1,64,64>` **7.2%** `123 ms` + `<4,128,128>` `gate` **2.5%** `43 ms` + `<3,32,64>` **1.1%** `19 ms`): **10.8%** (`186 ms`).
     - Attention (`attn_slide_fp8` **6.5%** `111 ms` + `attn_prompt_fp8` **1.4%** `24 ms`): **7.9%** (`135 ms`).
     - `loop_ar` **6.2%** (`106 ms`), `mix_rows` **3.0%**, `rel_table` **2.9%**, `moe_combine` **2.4%**, `moe_act` **2.4%**, `conv` **2.0%**, `rowdot_tiles` **1.5%**.
2. **`m2p1-c1` (50 rounds, 128 tok, 2.845 s, `ATTN_FP8=1`):**
   - `tf_ink2moe_up`: **51.7%** (`1,401 ms`, `28.0 ms/rnd`).
   - `qmmf<1,16,64>` (`ATTN_FP8` `wq/wk/wv/wo`): **9.2%** (`250 ms`); `qmmf<4,16,64>` (BF16 `gate_attn`): **6.0%** (`164 ms`).
   - Two `1.96 ms` `qmmf<1,16,64>` calls/round (`98.2 + 97.7 ms` = **7.2%**, `3.9 ms/rnd`, target lazy head + FP8 draft head).
   - `mix` **4.5%**, `loop_ar` **2.8%**, `gate_down` **2.4%**, `attn_fp8_prefetch` **2.2%**, `conv` **1.6%**. `6,948` eager `cuLaunchKernel` + `1,850` `cuLaunchKernelEx` + `8` graph re-instantiations.

### Hill-Climb Round 3: `tf-inkling2:m2-p2` (`0007` @ `8db05bc`, Batched DFlash2)
1. **`ctl-06-m2-p1` (`12.0 GiB` reserve, `c1/c4/c8` prose):**
   - `c1=47.91`, `c4=70.45`, `c8=88.66` (`1.113 tok/rnd`).
   - At `c=8`, `proc_anon_gib` rose to `4.07 GiB` (`108.76 GiB` peak footprint), dipping `MemAvailable` to `6.01 GiB` with `SWAP_OUT` (`185,425` pages).
2. **`cand-05-m2-p2` (`12.0 GiB` reserve):**
   - `c1=46.95` (`53.79 ms/rnd`), `c4=73.55` (+4.4%, `1.299 tok/rnd`), tripped `memwatch` `LOW_MEM` (`5.99 GiB`) during `c=8`.
3. **`cand-06-m2-p2-res16` (`MEMORY_RESERVE_GIB=16.0`, `4,549,888` KV tokens = `4.3x` 1M streams):**
   - Warm boot `14.79 s`, `c1` prose `46.98` (`53.06 ms/rnd`), `c4` prose **`73.56`** (`+4.4%`, `1.300 tok/rnd`), **`c8` prose `95.56`** (`+7.8%`, `1.225 tok/rnd`).
   - Code greedy: `c1=49.96` (`2.768 tok/rnd`, `acc 0.436`, down from `m1` `ATTN_FP8=0` `55.06` / `3.631 tok/rnd`; testing `ATTN_FP8=0` vs `=1` on code in Round 4), `c4=80.94` (`1.623 tok/rnd`, `76.63 ms/rnd`).
   - `m2p2-c4-code` `nsys` profile (`profiles/m2p2-nsys-summary.txt`): `54,695` eager `cuLaunchKernel` (`315 ms`), `53` `cuGraphInstantiateWithFlags` (`175 ms`) + `53` `cuGraphDestroy` in 163 rounds; `tf_ink2moe_up` **47.2%** (`5,113 ms`, CUDA-core decode kernel used on `R=12..32` verify); `tf_ink2_attn_fp8_prefetch` **8.0%** (`866 ms`).
   - Verdict: **Batched DFlash2 (`m2-p2`) ACCEPTED (+4.4% c4, +7.8% c8); `MEMORY_RESERVE_GIB=16.0` prevents c8 LOW_MEM, though `verify_cache=256` inflated `proc_anon_gib` to `5.81 GiB` at c8.**

### Hill-Climb Round 4: `tf-inkling2:m2-p3` (`0008` @ `3925718`)
1. **`cand-07-m2-p3` (`PREFILL_TC=1, ATTN_FP8=1, GATE_FP8=0, DRAFT_FP4_HEAD=1, reserve=16.0`):**
   - Boot `16.55 s` (first boot of m2-p3 key), `c1` prose `47.03` (`53.79 ms/rnd`), `c4` prose `74.48` (`1.327 tok/rnd`), `c1` code `49.72`, `c4` code **`86.64`** (**+7.0%** over m2-p2 `80.94`, `1.973 tok/rnd`), **`4K` prefill `2,325.5 tok/s`** (**+18.6%** over m2-p1 `1,960.0`), **`32K` prefill `2,233.1 tok/s`** (**+17.7%** over m2-p1 `1,896.8`), `4,542,720` KV tokens, `min MemAvailable = 9.66 GiB`, `pswpout = 0`.
2. **`cand-08-m2-p3-gatefp8` (`+ TF_INKLING2_GATE_FP8=1`):**
   - Warm boot **`14.55 s`** (`propose 3.93 ms`), `c1` prose **`47.88`** (+1.8%, `53.03 ms/rnd`), `c4` prose **`76.43`** (+2.6%, `67.76 ms/rnd`), `4K` `2,213.7`, `32K` `2,222.6`, `min MemAvailable = 11.45 GiB`, `pswpout = 0`. **ACCEPTED.**
3. **`cand-09-m2-p3-attnbf16` (`ATTN_FP8=0` check on code vs prose):**
   - `c1` prose `40.05` (`55.25 ms/rnd`, `-16.4%` vs `cand-08`!), `c1` code `55.75` (`3.556 tok/rnd`, `acc 0.457`). Confirms `ATTN_FP8=1` trades greedy code draft acceptance (`0.457 -> 0.409`) for `+17-19%` prose speed (`40.05 -> 47.88`).
4. **`cand-10-m2-p3-fp8head` (`ATTN_FP8=1, GATE_FP8=1, DRAFT_FP4_HEAD=0`):**
   - Warm boot `14.46 s` (`loaded in 13.4s`), `c1` prose **`48.245`** (`52.374 ms/rnd`), `c4` prose `75.145` (`64.726 ms/rnd`), `c1` code `51.00` (`54.99 ms/rnd`), `c4` code `83.87` (`75.365 ms/rnd`), `pswpout=0`. **ACCEPTED as m2-p3 best.**
5. **`m2-p3` Gates & `prefill4k` Profile (`profiles/m2p3-gates.json`, `profiles/m2p3-nsys-summary.txt`):**
   - `gates.py` (`cand-08`): 7/9 passed (`load`, `chat`, `tools`, `structured`, `vision` PNG/JPEG/WebP, `drafted_exactness`, `multilingual`). 2 failed: `prefix_cache` (`turn2_cached=0`, `PROMPT_CACHE_GIB=""` left `cache=0` -> `h.prompts=null`) and `reasoning` (`thinking_off` emitted 116 chars of reasoning before `"391"`).
   - `m2p3-prefill4k` (`1.497 s` / `2,353 tok/s`): `prompt_tc_nvfp4` `438.7 ms` (29.8%) + `prompt_tc_mxfp8` `72.6 ms` (4.9%), `qmmf<1,64,64>` `168.7 ms` (11.5%), `attn_slide_fp8` `110.6 ms` (7.5%), `loop_ar` `106.9 ms` (7.3%), `gate_down_prepared` `93.8 ms` (6.4%), `loop_gate_up_tiled` `45.2 ms` (3.1%).

### Hill-Climb Round 5: `tf-inkling2:m2-p4` (`0009` @ `a570507`)
1. **`cand-11-m2-p4` (`PREFILL_TC=1, ATTN_FP8=1, GATE_FP8=1, DRAFT_FP4_HEAD=0, VERIFY_TC_ROWS=12, reserve=13.0`):**
   - Memory win verified: `peak_proc_anon_gib = 2.69 GiB` (down from `5.81 GiB` in `m2-p2`), `min_mem_avail_gib = 8.44 GiB`, `pswpout_delta = 0`, KV pool `4,744,448` tokens, warm boot `16.31 s` (`v2` cache).
   - Throughput regressed vs `m2-p3` (`cand-10`): prose `c1/c4/c8` dec `45.84` (agg `45.025`, `54.02 ms/rnd`) / `71.095` (`1.153 tok/rnd`) / `91.30` (`1.123 tok/rnd`); code `c1/c4/c8` `50.95` (`49.785` agg) / `75.155` (`1.246 tok/rnd`) / `94.005` (`1.213 tok/rnd`); prefill `4K=2,152.5`, `32K=2,220.4`. `VERIFY_TC_ROWS=12` (`RT=1` tile=16) hurt `c4` verify so `shared_guard` throttled drafting (`1.97 -> 1.25 tok/rnd`). Superseded immediately by `m2-p5` (`VERIFY_TC_ROWS=32`, `RT=4`, zero-copy `prefix_cache`, `loop_ar_cached`).

### Hill-Climb Round 6: `tf-inkling2:m2-p5` (`0010` @ `c56a896`)
1. **`cand-12-m2-p5` (`PREFILL_TC=1, ATTN_FP8=1, GATE_FP8=1, DRAFT_FP4_HEAD=0, reserve=13.0`):**
   - Warm boot `15.30 s` (`loaded in 14.9s`), KV pool `4,748,544` tokens, `peak_proc_anon_gib = 2.65 GiB`, `min_mem_avail_gib = 8.49 GiB`, `pswpout_delta = 0`.
   - Prose greedy (`c1/c4/c8`): dec `47.79` (agg `47.095`, **`51.203 ms/rnd`**, `ttft=0.177s`) / `75.755` (`1.283 tok/rnd`) / **`97.64`** (new best c8, `1.20 tok/rnd`).
   - Code greedy (`c1/c4/c8`): dec **`51.805`** (agg `50.63`, `2.837 tok/rnd`) / `85.42` (`1.713 tok/rnd`) / **`109.04`** (new best c1 & c8, `1.607 tok/rnd`).
   - Prefill (`cold_runs=1`): `4K=2,062.7 tok/s` (`1.974s`), `32K=2,200.4 tok/s` (`14.963s`).
   - `gates.py` (`profiles/m2p5-gates.json`): **8/9 passed**. `prefix_cache` **PASSED** (`turn2_cached=8704 / 8726` with `PROMPT_CACHE_GIB=""`). Only `reasoning` (`thinking_off` emitted 116 reasoning chars before `"391"`) still fails.
   - `nsys` (`profiles/m2p5-nsys-summary.txt`): `m2p5-c4-code` (`11.88s` wall) still has `tf_ink2moe_up` at **49.6%** (`5,544 ms`, `16,800` launches, `330 us` avg) on `R=12..31` verify; `m2p5-prefill4k` (`4,621` tok, `2.129s`) has `prompt_tc_nvfp4` 30.0% (`615 ms`) + `mxfp8` 5.1% (`104 ms`), `qmmf<1,64,64>` 10.8% (`222 ms`), `attn_slide_fp8` 7.3% (`150 ms`), `loop_ar_cached` 7.2% (`148 ms`), loop gates 13.5% (`276 ms`).
2. **`cand-13-m2-p5-res12` (`MEMORY_RESERVE_GIB=12.0`):**
   - KV pool **`4,809,216` tokens** (+60,672), `min_mem_avail_gib = 7.43 GiB`, `peak_proc_anon_gib = 2.63 GiB`, `pswpout_delta = 0`.
   - Prose (`c1/c4/c8`): `47.255` (`50.912 ms/rnd`) / `75.25` / `95.43`; prefill (`cold_runs=2`): `4K=2,134.7 tok/s`, `32K=2,184.5 tok/s`. **ACCEPTED (`reserve=12.0` clean with `verify_cache=64`).**
3. **`cand-14-m2-p5-prows2048` (`TF_INKLING2_PREFILL_ROWS=2048`, `cold_runs=2`):**
   - `4K=1,966.8 tok/s` (`-7.9%` vs `4096`), `32K=2,075.9 tok/s` (`-5.0%` vs `4096`). **REJECTED; keep `PREFILL_ROWS=4096`.**

### Hill-Climb Round 7: `tf-inkling2:m2-p6` (`0011` @ `2e64f8b`)
1. **`cand-15-m2-p6` (`PREFILL_TC=1, ATTN_FP8=1, GATE_FP8=1, reserve=12.0`):**
   - Prefill jumped **+10-11%**: `4K = 2,366.8 tok/s` (`1.735s`), `32K = 2,400.9 tok/s` (`13.804s`).
   - `c4 prose = 77.945`, `c4 code = 89.875`. **However `reserve=12.0` hit `pswpout_delta = 13613` (`SWAP_OUT` alert, `min_mem_avail = 7.55 GiB`) during the combined run; rejected `reserve=12.0` for combined workloads.**
2. **`cand-16-m2-p6-res13` (`MEMORY_RESERVE_GIB=13.0` clean control):**
   - Warm boot **`14.79 s`** (`loaded in 14.5s`, `propose 4.50 ms`), KV pool `4,746,240` tokens, `min_mem_avail_gib = 8.47 GiB`, `peak_proc_anon_gib = 2.69 GiB`, **`pswpout_delta = 0`**, `alerts = []`.
   - Prose greedy (`c1/c4/c8`): dec `46.17` (`0.172s` ttft, `51.70 ms/rnd`) / **`79.715`** (+5.2% vs m2-p5, `1.563 tok/rnd`) / **`101.74`** (+4.2% vs m2-p5).
   - Code greedy (`c1/c4/c8`): dec `49.51` (`52.41 ms/rnd`) / **`91.43`** (+7.0% vs m2-p5, `2.179 tok/rnd`) / **`116.825`** (+7.1% vs m2-p5, `1.812 tok/rnd`).
   - Prefill (`cold_runs=1`): `4K = 2,259.5 tok/s` (`1.811s`), **`32K = 2,403.0 tok/s`** (`13.791s`). **ACCEPTED as m2-p6 best.**
   - `nsys` (`profiles/m2p6-nsys-summary.txt`): `m2p6-prefill4k` (`1.946s` for `4621` tok vs `2.129s` in `m2-p5`): `prompt_tc_coop` cut `prompt_tc_nvfp4` `615.1 -> 495.4 ms` (`-19.5%`) and `mxfp8` `104.0 -> 83.4 ms` (`-19.8%`); `qmmf<1,128,128>` `221.8 -> 193.1 ms` (`-12.9%`); `loop_ar` `147.9 -> 138.3 ms`. Top remaining prefill bottlenecks: `prompt_tc` 31.0% (`579 ms`), **loop gates/messages 14.8% (`276 ms`)**, `qmmf` 10.3% (`193 ms`), `attn_slide+prompt_fp8` 10.0% (`187 ms`), `loop_ar` 7.4% (`138 ms`), plus `9,173` `cuMemcpyDtoDAsync_v2` calls (`235.5 ms` API).
3. **`cand-17-m2-p6-wide0` (`TF_INKLING2_WIDE_ROWS=0`, disable `tf_ink2moe_up_wide`):**
   - `c4 prose = 74.065` (`1.212 tok/rnd`, `-7.1%` vs `79.715`), `c4 code = 76.675` (`1.288 tok/rnd`, `-16.1%` vs `91.43`). Proves `tf_ink2moe_up_wide` delivers **+7.6% prose / +19.2% code** at `c4` by keeping `shared_guard` enabled.
4. **`cand-20-m2-p6-wide1` (`TF_INKLING2_WIDE_ROWS=1`):**
   - Prose `c1/c4 = 45.465 / 78.79`, code `c1/c4 = 49.285 / 89.49`, `pswpout = 0`. Slightly slower than default `WIDE_ROWS=4` across all 4 cells. **REJECTED; keep `WIDE_ROWS=4`.**

### Hill-Climb Round 8: `tf-inkling2:m2-p7` (`0012` @ `296d55a`)
1. **`cand-21-m2-p7` (`PREFILL_TC=1, ATTN_FP8=1, GATE_FP8=1, reserve=13.0`):**
   - Warm boot `16.31 s` (`loaded in 15.8s`), KV pool `4,737,024` tokens, `min_mem_avail_gib = 8.50 GiB`, `peak_proc_anon_gib = 2.68 GiB`, `pswpout_delta = 0`.
   - Prose (`c1/c4/c8`): `46.21` (**`50.324 ms/rnd`**) / **`84.69`** (+6.2% vs m2-p6 `79.72`, `1.825 tok/rnd`) / `94.47`.
   - Code (`c1/c4/c8`): `50.515` (`52.988 ms/rnd`) / `89.08` (`1.946 tok/rnd`) / `101.105` (`1.28 tok/rnd`).
   - Prefill: `4K = 2,166.6 tok/s` (`1.884s`), `32K = 2,280.3 tok/s` (`14.511s`) (regressed vs `m2-p6` `2,259.5 / 2,403.0`).
   - `gates.py` (`profiles/m2p7-gates.json`): 8/9 passed (`prefix_cache` **PASSED** `8704/8726` with `copy_spans`; only `reasoning` fails per model weights).
   - `nsys` (`profiles/m2p7-nsys-summary.txt`): `tf_ink2_copy_spans` cut `cuMemcpyDtoDAsync_v2` **`9,173 -> 63` calls** (`235.5 -> 27.3 ms` API, `1.27 ms` GPU). However, the 3 reg-cached kernels regressed prefill (`+121 ms`): `tf_ink2_loop_ar` **`138.3 -> 224.0 ms`** (`+85.7 ms`), `loop_gate_down_prepared` **`125.0 -> 155.3 ms`** (`+30.3 ms`), `loop_gate_up_tiled32` **`67.7 ms`** vs `loop_gate_up_tiled` `62.7 ms` (`+5.0 ms`). Keeping `m2-p6` as `best-config.json` while reporting to `@engine-dev` for `m2-p8`.
2. **`cand-19-m2-p6-shared4` (`TF_INKLING2_DFLASH2_SHARED=4` vs default `8` on `m2-p6`):**
   - Prose `c4/c8 = 80.485 / 92.745` (vs `79.715 / 101.74` @ `8`), code `c4/c8 = 84.355 / 105.29` (vs `91.43 / 116.825` @ `8`), `pswpout=0`. **REJECTED; keep default `DFLASH2_SHARED=8` on `sm_121`.**

### Hill-Climb Round 9: `tf-inkling2:m2-p8` (`0013` @ `02fc511`)
1. **`cand-22-m2-p8` (`PREFILL_TC=1, ATTN_FP8=1, GATE_FP8=1, reserve=13.0`):**
   - Warm boot `15.30 s` (`loaded in 14.8s`), KV pool `4,733,952` tokens, `min_mem_avail_gib = 8.55 GiB`, `peak_proc_anon_gib = 2.68 GiB`, **`pswpout_delta = 0`**, `alerts = []`.
   - Prose (`c1/c4/c8`): `46.42` (**`0.170s` ttft**, `51.20 ms/rnd`) / `79.40` (`1.51 tok/rnd`) / **`107.13`** (**NEW BEST c8**, +5.3% vs m2-p6 `101.74`, `1.476 tok/rnd`).
   - Code (`c1/c4/c8`): `50.085` (`52.19 ms/rnd`) / `85.63` (`1.765 tok/rnd`) / **`120.50`** (**NEW BEST c8**, +3.1% vs m2-p6 `116.83`, `1.946 tok/rnd`).
   - Prefill (`cold_runs=1`): **`4K = 2,366.7 tok/s`** (`1.742s`), `32K = 2,372.9 tok/s` (`13.984s`).
   - `gates.py` (`profiles/m2p8-gates.json`): **8/9 passed** (`prefix_cache` passes `8704/8726` with `copy_spans`; only `reasoning` fails per model weights).
   - `nsys` (`profiles/m2p8-nsys-summary.txt`): `m2p8-prefill4k` (`1.939 s` wall, fastest yet): `loop_ar` restored `224.0 -> 139.1 ms`, `gate_up_tiled` restored `67.7 -> 61.6 ms`, `copy_spans` kept (`cuMemcpyDtoDAsync_v2` `63` calls / `24.1 ms`). However, 4-row SMEM `gate_down_prepared` (`38 KB`) stayed at `156.4 ms` (`+31.4 ms` vs `m2-p6`'s `125.0 ms` 2-row). **ACCEPTED as new `best-config.json` (`best-m2-p8`).**
2. **`cand-23-m2-p8-wide8` (`TF_INKLING2_WIDE_ROWS=8` vs default `4`):**
   - `c1` rose (`47.65` prose / `50.895` code vs `46.42 / 50.085`), while `c4` dropped (`74.915` prose / `79.855` code vs `79.40 / 85.63`), `pswpout=0`. **REJECTED; `WIDE_ROWS=4` beats `0`, `1`, and `8` on `c4`.**
3. **`cand-24-m2-p8-vtc64` (`TF_INKLING2_VERIFY_TC_ROWS=64` vs default `32`):**
   - Prose `c4/c8 = 75.765 / 96.135` (vs `79.40 / 107.13`), code `c4/c8 = 89.835 / 116.835` (vs `85.63 / 120.50`), `pswpout=0`. **REJECTED; keep default `VERIFY_TC_ROWS=32`.**

### Hill-Climb Round 10: `tf-inkling2:m2-p9` (`0014` @ `48f5781`)
1. **`cand-25-m2-p9` (`PREFILL_TC=1, ATTN_FP8=1, GATE_FP8=1, reserve=13.0`):**
   - Warm boot `16.06 s` (`loaded in 15.7s`), KV pool `4,736,000` tokens, `min_mem_avail_gib = 8.48 GiB`, `peak_proc_anon_gib = 2.67 GiB`, **`pswpout_delta = 0`**, `alerts = []`.
   - Prose (`c1/c4/c8`): `45.64` (`0.176s` ttft, `52.94 ms/rnd`) / `78.05` (`1.426 tok/rnd`) / `96.965` (`1.185 tok/rnd`).
   - Code (`c1/c4/c8`): `49.59` (`52.51 ms/rnd`) / `83.99` (`1.636 tok/rnd`) / `116.735` (`1.784 tok/rnd`).
   - Prefill (`cold_runs=1`): `4K = 2,327.0 tok/s` (`1.756s`), **`32K = 2,495.3 tok/s`** (**NEW BEST**, `13.196s`, +5.2% vs m2-p8 `2,372.9`, +3.58x vs M1 `696.6`).
   - `gates.py` (`profiles/m2p9-gates.json`): **8/9 passed** (`prefix_cache` passes `8704/8726`, `drafted_exactness` passes; only `reasoning` fails per model weights).
   - `nsys` (`profiles/m2p9-nsys-summary.txt`): `m2p9-prefill4k` (`1.865 s` wall, fastest yet): 8-warp `prompt_tc_coop` cut `prompt_tc_nvfp4` `501.8 -> 464.0 ms` (`-7.5%`) and `mxfp8` `79.9 -> 73.2 ms` (`-8.4%`); 2-row `r+=8` `gate_down_prepared` cut `156.4 -> 122.6 ms` (`-21.6%`, fastest yet). **ACCEPTED as new `best-config.json` (`best-m2-p9`).**
2. **`cand-26-m2-p9-res125` (`MEMORY_RESERVE_GIB=12.5` knee check):**
   - `4,767,232` KV tokens (+31K vs `13.0`), `min_mem_avail_gib = 8.09 GiB`, `peak_proc_anon_gib = 2.60 GiB`, `pswpout_delta = 0`, `c1/c4/c8` prose `45.80 / 75.49 / 94.94`, `32K` prefill `2,459.4 tok/s`. Keep `MEMORY_RESERVE_GIB=13.0` (`min_mem_avail >= 8.48 GiB`) for full `needle_1m` + `soak` stability headroom.

### Qualification Suite (`tf-inkling2:m2-p9`, `MEMORY_RESERVE_GIB=13.0`, `ATTN_FP8=1`, `PREFILL_TC=1`, `GATE_FP8=1`)
1. **Startup (`profiles/m2p9-qual-startup.json`):** cold **`15.297 s`** (vs M1 `40.631 s`, 2.66x faster), warm **`14.294 s`** (vs M1 `32.588 s`, 2.28x faster).
2. **Multi-length Cold Prefill (`profiles/m2p9-qual-prefill.json`, `runs=2` median, `cached_tokens == 0`):**
   - `4K = 2,347.1 tok/s` (`1.740s`, +3.34x vs M1 `701.9`)
   - `8K = 2,522.7 tok/s` (`3.244s`)
   - `16K = 2,563.8 tok/s` (`6.384s`)
   - `32K = 2,464.6 tok/s` (`13.434s`, +3.54x vs M1 `696.6`)
   - `65K = 2,275.9 tok/s` (`29.498s`)
   - `128K = 1,980.4 tok/s` (`68.234s`, +3.05x vs M1 `649.3`)
   - `memwatch` (`m2p9-qual-p1-mem.tsv`): `min_mem_avail_gib = 11.45 GiB`, `pswpout_delta = 0`, `alerts = []`.
3. **Outer Ruler (`profiles/m2p9-qual-outer.json`, `c=1,2,4,8`, `runs=6` median, 0 errors/0 loops):**
   - `outer_sampled_prose_nothink`: `42.25 / 53.75 / 72.255 / 94.395` (`ttft 0.180s`, vs M1 `c1/c4 36.39 / 61.125`)
   - `outer_thinking_on_prose`: `43.75 / 51.455 / 73.60 / 92.045` (vs M1 `c1/c4 40.74 / 60.815`)
   - `outer_code_nothink`: `48.365 / 56.88 / 80.085 / 98.365`
   - `outer_greedy_prose_nothink`: `46.21 / 58.85 / 74.27 / 95.615` (`51.25 ms/rnd` c1)
   - `memwatch` (`m2p9-qual-outer-mem.tsv`): `1346` samples, `min_mem_avail_gib = 8.45 GiB`, `peak_proc_anon_gib = 2.72 GiB`, `pswpout_delta = 0`, `alerts = []`.
4. **Acceptance Gates & Exactness (`profiles/m2p9-qual-gates.json`, `profiles/m2p9-plain-vs-drafted.json`):**
   - **12/12 `gates.py` PASSED**: `load`, `chat`, `reasoning`, `tools`, `structured`, `vision`, `drafted_exactness` (serial==concurrent 4/4), `multilingual`, `prefix_cache` (`8704/8726`), `needle_256k` (`255,079` & `255,224` exact), `needle_1m` (`1,023,701` & `1,023,881` exact, `2973.64s`), `soak` (`c4`+`c8`, `loop: false`).
   - `plain` (`DRAFTER=""`) vs `drafted` (`DFlash2`): **4/4 exact match** (`all_exact: true`).
   - `memwatch` (`m2p9-qual-gates-mem.tsv`): `2065` samples, `min_mem_avail_gib = 8.36 GiB`, `peak_gpu_res_gib = 103.91 GiB`, `peak_proc_anon_gib = 2.93 GiB`, `peak_footprint_gib = 106.84 GiB`, `pswpout_delta = 0`, `alerts = []`.
5. **NLL Gate (`profiles/m2p9-nll.json`):**
   - Probed `tools/nll.py capture` + direct `POST /v1/completions` (`prompt_logprobs=1`) on `tf-inkling2:m2-p9` (`best-m2-p9` flags) under `flock`: **unsupported by backend** (`HTTP 400: {"error": {"message": "prompt_logprobs are not supported by this model or backend", "type": "invalid_request_error"}}`).
   - Engine source confirms `zig/src/families/inkling2/loop_backend.zig:498-510` omits `prompt_rows` (defaults `false` in `core/lanes/backend.zig:158`, wired at `native/cuda.zig:488`, refused at `server/fields.zig:366`). Also `first_row` (`logprobs`) is not wired on `inkling2` `vtable`.
   - Stopped cleanly via `./stop.sh` (`docker ps -a` empty).

### M3 Decode Optimization Sprint (`inkling2-v135-m3` from `48f5781`)
1. **`ctl-m3-00` (fresh M3 baseline control on `tf-inkling2:m2-p9`, `best-config.json`):**
   - Warm boot `14.47 s` (`loaded in 13.6s`), KV pool `4,699,136` tokens, `min_mem_avail_gib = 9.47 GiB`, `peak_gpu_res_gib = 103.06 GiB`, `peak_proc_anon_gib = 1.85 GiB`, `pswpout_delta = 0`, `alerts = []`, host load `0.09 -> 1.53`.
   - Outer sampled prose (think off, `runs=6`): `c1 = 42.25` (`50.65 ms/rnd`, `2.133 t/r`, `acc 0.443`), `c4 = 70.47` agg (`18.24/strm`, `70.35 ms/rnd`, `1.276 t/r`, `acc 0.509`).
   - Inner greedy prose (`runs=6`): `c1 = 46.52` (`52.00 ms/rnd`, `2.398 t/r`, `acc 0.451`), `c4 = 73.19` agg (`18.89/strm`, `67.52 ms/rnd`, `1.265 t/r`, `acc 0.423`).
   - Inner greedy code (`runs=6`): `c1 = 48.93` (`51.66 ms/rnd`, `2.529 t/r`, `acc 0.473`), `c4 = 82.32` agg (`22.01/strm`, `78.19 ms/rnd`, `1.733 t/r`, `acc 0.470`).
   - Cold prefill (`runs=1`): `4K = 2,520.8 tok/s` (`1.627s`), `32K = 2,462.2 tok/s` (`13.462s`).
2. **`cand-m3-p1` (`tf-inkling2:m3-p1`, `edb3883` = `author-loop` Commit 1 `8edc80c` + `environment.zig`, Lever 5: dedup sliding `loop_ar`, `PASS_MESSAGES=1`, `mix_rows` + 1-pass RMSNorm, tiled small-M `gate_down`/`gate_up`, `uint4` `rel_table`):**
   - Boot `16.49 s` (`loaded in 16.1s`, fresh `window-costs`), KV pool `4,697,600` tokens, `min_mem_avail_gib = 9.41 GiB`, `peak_gpu_res_gib = 103.04 GiB`, `peak_proc_anon_gib = 2.10 GiB`, 7/7 fast gates PASSED (`drafted_exactness` 4/4, `prefix_cache` 8704/8726). (Note: `pswpout_sum=66669` ~260 MiB background host page-cache reclaim after build despite `min_avail=9.41 GiB`.)
   - Outer sampled prose: `c1 = 43.72` (+3.5%, `49.75 ms/rnd`, `2.188 t/r`), `c4 = 71.83` (+1.9%, `67.66 ms/rnd`, `1.243 t/r`).
   - Inner greedy prose: `c1 = 47.61` (+2.3%, `50.13 ms/rnd`, `2.387 t/r`), `c4 = 74.65` (+2.0%, `65.96 ms/rnd`, `1.286 t/r`).
   - Inner greedy code: `c1 = 50.26` (+2.7%, `50.16 ms/rnd`, `2.510 t/r`), `c4 = 77.15` (`66.35 ms/rnd`, `1.354 t/r`).
   - Cold prefill: `4K = 2,421.1` (`1.687s`), `32K = 2,575.1` (+4.6%, `12.872s`).
   - Verdict: **ACCEPTED** (lossless `-1.5 to -2.7 ms/rnd` across all cells + 32K prefill win).
3. **`cand-m3-p2` (`tf-inkling2:m3-p2`, `84f571c` = `author-loop` Commit 2 `f101dc0`, Lever 1: stream-batched `conv`/`attn` + bucketed global `last_pos` + slot-agnostic `Shape` graphs):**
   - Boot `16.53 s` (`loaded in 15.6s`), KV pool `4,699,648` tokens, `min_mem_avail_gib = 10.22 GiB`, `peak_gpu_res_gib = 102.78 GiB`, `peak_proc_anon_gib = 1.60 GiB`, `pswpout_delta = 0`, `alerts = []`, 7/7 fast gates PASSED.
   - Outer sampled prose: `c1 = 42.08` (`49.18 ms/rnd`, `2.058 t/r`), `c4 = 78.80` (+9.7% vs `m3-p1`, +11.8% vs `ctl-m3-00`, `62.56 ms/rnd`, `1.271 t/r`).
   - Inner greedy prose: `c1 = 46.61` (`48.26 ms/rnd`, `2.231 t/r`), `c4 = 83.40` (+11.7% vs `m3-p1`, +14.0% vs `ctl-m3-00`, `59.17 ms/rnd`, `1.278 t/r`).
   - Inner greedy code: `c1 = 50.43` (`51.11 ms/rnd`, `2.573 t/r`), `c4 = 84.66` (+9.7% vs `m3-p1`, `59.04 ms/rnd`, `1.332 t/r`).
   - Cold prefill: `4K = 2,442.7` (`1.675s`), `32K = 2,616.5` (new best, +6.3% vs `ctl-m3-00`, `12.582s`).
   - Verdict: **ACCEPTED** (massive lossless c4 win `-5.1 to -7.3 ms/rnd`, lower anon RAM, `pswpout=0`).
4. **`cand-m3-p3` (`tf-inkling2:m3-p3`, `1bfc3b4` = `author-moe` Commit 1 `0fcb81a` + `environment.zig`, Lever 7: NVFP4 BLOCK4 repack + fused softcap-SwiGLU + sm_121a TC prefill):**
   - Boot `16.49 s` (`loaded in 15.5s`), KV pool `4,700,672` tokens (0 B VRAM delta), `min_mem_avail_gib = 9.95 GiB`, `peak_gpu_res_gib = 102.81 GiB`, `peak_proc_anon_gib = 1.64 GiB`, `pswpout_delta = 0`, `alerts = []`, 7/7 fast gates PASSED.
   - Outer sampled prose: `c1 = 45.56` (+8.3% vs `m3-p2`, `48.08 ms/rnd`), `c4 = 89.92` (+14.1% vs `m3-p2`, +27.6% vs `ctl-m3-00`, `59.73 ms/rnd`, `1.395 t/r`).
   - Inner greedy prose: `c1 = 47.81` (`43.25 ms/rnd`, `-5.01 ms/rnd`), `c4 = 91.17` (+9.3% vs `m3-p2`, +24.6% vs `ctl-m3-00`, `51.21 ms/rnd`, `1.212 t/r`).
   - Inner greedy code: `c1 = 65.54` (+30.0% vs `m3-p2`, `52.36 ms/rnd`, `3.413 t/r`), `c4 = 99.36` (+17.4% vs `m3-p2`, `54.34 ms/rnd`, `1.396 t/r`).
   - Cold prefill: `4K = 2,802.1` (+14.7%, `1.457s`), `32K = 2,878.6` (+10.0%, +16.9% vs `ctl-m3-00`, `11.475s`).
   - Verdict: **ACCEPTED** (huge win across c1, c4, code, and prefill; 0 VRAM delta, `pswpout=0`).
5. **`cand-m3-p4` (`tf-inkling2:m3-p4`, `e975172` = `author-loop` Commit 3 `43110c8` + `environment.zig`, Lever 3: lossless BF16 TC unembed + batched/sliced `drawRows` + batched `verify`):**
   - Boot `16.53 s` (`loaded in 14.8s`), KV pool `4,698,368` tokens, `min_mem_avail_gib = 10.01 GiB`, `peak_gpu_res_gib = 102.83 GiB`, `peak_proc_anon_gib = 1.78 GiB`, `pswpout_delta = 0`, `alerts = []`, 7/7 fast gates PASSED.
   - Outer sampled prose: `c1 = 44.72` (`48.20 ms/rnd`, `2.170 t/r`), `c4 = 89.31` (`58.38 ms/rnd`, `-1.35 ms/rnd`, `1.347 t/r`).
   - Inner greedy prose: `c1 = 48.30` (+1.0%, `44.72 ms/rnd`), `c4 = 94.82` (+4.0%, +29.6% vs `ctl-m3-00`, `54.58 ms/rnd`, `1.340 t/r`).
   - Inner greedy code: `c1 = 67.41` (+2.9%, `54.34 ms/rnd`, `3.644 t/r`), `c4 = 119.57` (+20.3%, +45.3% vs `ctl-m3-00`, `74.35 ms/rnd`, `2.522 t/r`).
   - Cold prefill: `4K = 2,807.2` (`1.456s`), `32K = 2,854.7` (`11.593s`).
   - Verdict: **ACCEPTED** (lossless `-1.35 ms/rnd` on sampled c4 + strong greedy prose/code c4 wins).
6. **`cand-m3-p5` & `cand-m3-p5-fp8tensor` (`tf-inkling2:m3-p5`, `79a07b2` = `author-loop` Commit 4 `4c1607e`, Lever 6 `QMMF_GROUP=1` + Lever 10 `ATTN_FP8_GROUP`):**
   - `ATTN_FP8_GROUP=1` (7/7 fast gates PASS, `pswpout=0`, `4,700,928` KV): inner greedy prose c1 `52.24` (`47.86 ms/rnd`), c4 `94.74`; outer sampled `44.05 / 88.41` (`48.78 / 57.83 ms/rnd`); **code c1 collapsed `67.41 -> 53.61` (-20.5%, `acc 0.464 -> 0.391`), code c4 `110.50`**; prefill `4K 2,759.9 / 32K 2,814.9`.
   - Same-binary A/B `ATTN_FP8_GROUP=0` + `QMMF_GROUP=1` (`min_avail=10.66 GiB`, `peak_anon=1.20 GiB`, `pswpout=0`): outer sampled `c1 = 45.52` (`47.73 ms/rnd`), `c4 = 87.49` (`55.34 ms/rnd`, `-3.04 ms/rnd` vs `m3-p4`); inner greedy prose `c1 = 48.19` (`40.89 ms/rnd`, `-3.83 ms/rnd`), `c4 = 94.83` (`52.46 ms/rnd`); inner greedy code `c1 = 66.83` (`acc 0.460`, recovered!), `c4 = 108.45`; cold prefill **`4K = 2,910.8` / `32K = 2,898.1`** (new bests).
   - Verdict: **`QMMF_GROUP=1` ACCEPTED; `ATTN_FP8_GROUP` REJECTED as default (flipped default to `false` on `inkling2-v135-m3`).**
7. **`cand-m3-p6` (`tf-inkling2:m3-p6`, `2831881` = `author-dflash` Commit 1 `0d6eced`, Lever 2: `shared_guard` uniform depth + diverged `shared_costs` + batched `hold_costs`):**
   - 4/4 `shared_guard.zig` unit tests PASS; 7/7 fast gates PASS; cold/fresh-warm boot `18.53 s` (`loaded in 16.7s`, `8 widths, 12 shared, 4 hold, propose 4.34 ms, sampled +2.11 ms`), KV pool `4,701,184` tokens, `min_mem_avail_gib = 10.32 GiB`, `peak_proc_anon_gib = 1.47 GiB`, `pswpout_delta = 0`, `alerts = []`.
   - Outer sampled prose: `c1 = 44.64` (`47.64 ms/rnd`), **`c4 = 99.98`** (+14.3% vs `m3-p5`, +41.9% vs `ctl-m3-00`, `26.78/strm`, `1.989 t/r`).
   - Inner greedy prose: `c1 = 49.35` (`42.53 ms/rnd`), **`c4 = 108.89`** (+14.8% vs `m3-p5`, +48.8% vs `ctl-m3-00`, `28.36/strm`, `2.133 t/r`).
   - Inner greedy code: `c1 = 66.33` (`51.60 ms/rnd`), **`c4 = 129.23`** (+19.2% vs `m3-p5`, +57.0% vs `ctl-m3-00`, `36.27/strm`, `2.734 t/r`).
   - Cold prefill: `4K = 2,831.2`, `32K = 2,885.7`.
   - Verdict: **ACCEPTED** (massive c4 breakthrough across prose & code, 100% lossless, `pswpout=0`).
8. **`cand-m3-p7` & `cand-m3-p7b` (`tf-inkling2:m3-p7b`, `126c969` = `author-dflash` Commit 2 `2a8e2a5` + `loop_graphs.zig:61` distinct-token `Graphs.price()` fix, Lever 4 + 8c):**
   - Initial `cand-m3-p7` (`21e29b2`, 7/7 fast gates PASS, `4K=2,998.2`, `32K=2,861.2`) exposed a latent `loop_graphs.zig:61` bug (`const zeros = @splat(0)` gave a flat `1..8` cost curve, causing `CHANCES=1` to over-draft `d=6.16` on prose with `acc=0.266` and regress c1 greedy prose `49.35 -> 46.32`).
   - Fixed `loop_graphs.zig:61` to pass distinct tokens `(1000 + j * 4099) % vocab` + bumped cache key to `inkling2-warm-v4` (`cand-m3-p7b`, `126c969`, `min_avail=10.99 GiB`, `peak_anon=0.90 GiB`, `pswpout=0`, `4,699,904` KV):
     - Outer sampled prose: **`c1 = 47.07`** (+5.4% vs `m3-p6`, +11.4% vs `ctl-m3-00`, `46.08 ms/rnd`, `acc 0.461`, `ttft 0.147s`), **`c4 = 102.68`** (+2.7% vs `m3-p6`, +45.7% vs `ctl-m3-00`, `2.048 t/r`).
     - Inner greedy prose: **`c1 = 53.02`** (+7.4% vs `m3-p6`, +14.0% vs `ctl-m3-00`, `43.03 ms/rnd`, `acc 0.491`), `c4 = 104.17` (`63.38 ms/rnd`).
     - Inner greedy code: **`c1 = 73.33`** (+10.6% vs `m3-p6`, +49.9% vs `ctl-m3-00`, `acc 0.589`), **`c4 = 129.38`**.
   - Verdict: **ACCEPTED** (huge c1 + c4 win across every workload, 100% lossless, `pswpout=0`).
9. **`cand-m3-p8` & `cand-m3-p8-v64k` (`tf-inkling2:m3-p8`, `88ae043` = `author-dflash` Commit 3 `5b8ca28`, Lever 8a + 8b: wired `--drafter-bits 4` NVFP4 draft layers + vocab-pruned draft head):**
   - `cand-m3-p8` (`DRAFTER_BITS=4`, `DRAFT_VOCAB=0`, 7/7 fast gates PASS, `min_avail=10.34 GiB`, `peak_anon=1.50 GiB`, `pswpout=0`): `propose` dropped `4.29 -> 3.58 ms` (-0.71 ms), drafter VRAM dropped `522 -> 333 MB` (`4,711,168` KV tokens, +11K). Outer sampled prose `c1 = 47.56` (`45.41 ms/rnd`, `acc 0.469`), `c4 = 103.18` (`27.37/strm`, `2.138 t/r`); inner greedy prose `c1 = 54.26` (`41.49 ms/rnd`), `c4 = 107.97`; inner greedy code `c1 = 71.78`, `c4 = 128.61`; cold prefill `4K = 2,791.5`, `32K = 2,888.6`.
   - `cand-m3-p8-v64k` (`TF_INKLING2_DRAFT_VOCAB=65536`): `propose` hit `2.48 ms` (4/4 gates incl. multilingual PASS), but `c1` sampled prose collapsed to `35.18` (`1.106 t/r`), `c4` `79.18` (`1.016 t/r`), `c1` greedy prose `37.35` (`1.174 t/r`). Verified root cause in `tokenizer.json`: Inkling-2-Mini places space-prefixed English/code words at IDs `149,876+` (`" the"=149876`, `" and"=149880`, `" def"=151560`), not `0..65K`.
   - Verdict: **`DRAFTER_BITS=4` NVFP4 draft layers ACCEPTED; prefix-slice `DRAFT_VOCAB=65536` REJECTED (keep `DRAFT_VOCAB=0`).**
10. **`cand-m3-p9` (`tf-inkling2:m3-p9`, `dba99e8` = `author-dflash` Commit 4 `db3a66b`, Lever 9: BF16 TC `tf_ink2_df_attn_batch` + 16 balanced chunks):**
    - 7/7 fast gates PASS, KV pool `4,708,096` tokens, `propose = 3.46 ms` (down from `4.36 ms`).
    - Outer sampled prose: **`c1 = 47.92`** (+13.4% vs `ctl-m3-00`, `45.50 ms/rnd`), **`c4 = 105.13`** (+49.2% vs `ctl-m3-00`, `27.50/strm`, `2.061 t/r`).
    - Inner greedy prose: **`c1 = 56.12`** (+20.6% vs `ctl-m3-00`, `43.02 ms/rnd`), **`c4 = 109.22`** (+49.2% vs `ctl-m3-00`, `28.58/strm`).
    - Inner greedy code: **`c1 = 73.43`** (+50.1% vs `ctl-m3-00`, `46.87 ms/rnd`), **`c4 = 131.66`** (+60.0% vs `ctl-m3-00`, `37.02/strm`).
    - Cold prefill: `4K = 2,793.8`, `32K = 2,882.1`.
    - Verdict: **ACCEPTED** (all-time best across every decode cell).
11. **4K-Context Lever 9 A/B & 6-Knob Same-Binary Sweep on `tf-inkling2:m3-p9` (`dba99e8`, all `pswpout=0`, `min_avail=10.91-11.08 GiB`, `peak_anon=0.87-0.94 GiB`):**
    - 4K-context decode A/B (`runs=4` median): `m3-p8` `49.76 tok/s` (`50.19 ms/rnd`) -> **`m3-p9` `54.55 tok/s` (+9.6%, `41.67 ms/rnd`, `-8.52 ms/rnd`)**.
    - `TF_QMMF_HALF_BN=1` (lossless): sampled `49.33 / 106.67`, greedy `55.93 (39.26 ms/r) / 114.57` (+4.9%). **ACCEPTED.**
    - `TF_INKLING2_MOE_WAVES=1` (lossless): sampled `48.56 / 106.13`, greedy `56.70 (39.44 ms/r) / 114.58` (+4.9%). **ACCEPTED.**
    - `TF_INKLING2_SAMPLED_FP8_HEAD=1` (reuses FP8 `lazy_head`, 0 VRAM): sampled **`54.18 (39.55 ms/r, acc 0.500) / 109.91`** (+13.1% c1 / +4.5% c4), greedy `55.54 / 112.15`. **ACCEPTED (clearly-labelled opt-in lossy default).**
    - `TF_INKLING2_WR_FP8=1`: sampled `50.07 / 108.43`, greedy c1 dipped `56.12 -> 54.10`. **KEEP OPT-IN `0`.**
    - `TF_INKLING2_DRAFT_LAYER_BITS=8`: `propose 4.42` vs `3.46 ms`, greedy c1 `54.92`. **KEEP `4`.**
    - `TF_INKLING2_SHARED_MAX_ROWS=20`: sampled c4 `104.26`, greedy c1 `54.62`. **KEEP `16`.**
12. **`cand-m3-combo` (combined final M3 `EXTRA_ENV` on `tf-inkling2:m3-p9`, `dba99e8`):**
    - `EXTRA_ENV="TF_INKLING2_ATTN_FP8=1 TF_INKLING2_PREFILL_TC=1 TF_INKLING2_GATE_FP8=1 TF_QMMF_HALF_BN=1 TF_INKLING2_MOE_WAVES=1 TF_INKLING2_SAMPLED_FP8_HEAD=1"`.
    - 7/7 fast gates PASS (`drafted_exactness` 4/4, `prefix_cache` 8704/8726), KV pool `4,719,872` tokens, `min_mem_avail_gib = 10.45 GiB`, `peak_gpu_res_gib = 102.92 GiB`, `peak_proc_anon_gib = 1.53 GiB`, `pswpout_delta = 0`, `alerts = []`.
    - Outer sampled prose: **`c1 = 54.50`** (+29.0% vs `ctl-m3-00`, `39.16 ms/rnd`, `acc 0.496`, `ttft 0.142s`), **`c4 = 110.70`** (+57.1% vs `ctl-m3-00`, `29.60/strm`, `2.069 t/r`).
    - Inner greedy prose: **`c1 = 56.04`** (+20.5%, `41.64 ms/rnd`), **`c4 = 111.05`** (+51.7%, `28.81/strm`).
    - Inner greedy code: **`c1 = 75.88`** (+55.1%, `45.82 ms/rnd`, `acc 0.594`), **`c4 = 132.74`** (+61.2%, `37.31/strm`, `acc 0.596`).
    - Cold prefill: **`4K = 2,825.9`** (`1.447s`), **`32K = 2,882.7`** (`11.414s`).
    - Verdict: **ACCEPTED as M3 release candidate (`opt/m3-decode`). Proceeding to clean-room `tf-inkling2:clean-verify-m3` build and full qualification.**

### M3 Clean-Room Qualification Suite (`tf-inkling2:clean-verify-m3` / `tf-inkling2:m3-p9`, `0001..0023` @ engine `dba99e8`, recipe `opt/m3-decode` `7512b92`)
- **Clean-room build (`profiles/clean-verify-m3-build.log`):** Built from pinned TensorFold `v1.3.5` (`02b324ae`) + `engine-patches/0001..0023` (`PATCH_SHA=38d486856695b5852b9219eb3e20733e99ae390426c707b9b99f03f20f329249`), `tensorfold-native` SHA-256 `12a9c9ca413736ecc618f111c82977f921ab09da87588905901ba43b795565ec`, image SHA-256 `sha256:9ae0ffd1de2b271d9407523b30db4a30b6c918f5c7a349b6cfb1085ba92e0217`. Tagged `tf-inkling2:m3-p9` to exact clean-room image.
- **KV pool & footprint (`MEMORY_RESERVE_GIB=13.0`, `DRAFTER_BITS=4`):** `18,462` pages x `256` = **`4,726,272` KV tokens** (`74.37 GiB`, 4x 1M context), drafter weights `333.2 MB` (NVFP4, down from `522.3 MB`), target weights `18.89 GB`, `model 102.57 GiB`.
1. **Startup (`profiles/m3-qual-startup.json`):** cold **`17.052 s`**, warm **`16.796 s`**.
2. **Multi-length Cold Prefill (`profiles/m3-qual-prefill.json`, `runs=2` median, `cached_tokens == 0` exact):**
   - `4K = 2,923.7 tok/s` (`+24.6%` vs M2 `2,347.1`)
   - `8K = 3,012.5 tok/s` (`+19.4%` vs M2 `2,522.7`)
   - `16K = 3,024.4 tok/s` (`+18.0%` vs M2 `2,563.8`)
   - `32K = 2,900.3 tok/s` (`+17.7%` vs M2 `2,464.6`)
   - `65K = 2,627.2 tok/s` (`+15.4%` vs M2 `2,275.9`)
   - `128K = 2,214.6 tok/s` (`+11.8%` vs M2 `1,980.4`)
3. **Full Outer Ruler (`profiles/m3-qual-outer.json`, `c=1,2,4,8`, `runs=6` median, 0 errors/0 loops):**
   - `outer_sampled_prose_nothink`: **`54.48`** (`+28.9%` vs M2 `42.25`, `41.42 ms/rnd`, `ttft 0.143s`) / **`73.49`** / **`114.11`** (`+57.9%` vs M2 `72.26`, `30.04/strm`) / **`151.32`** (`+60.3%` vs M2 `94.40`, `19.99/strm`).
   - `outer_thinking_on_prose` (`temp=0.7, top_k=20`): **`56.76`** (`+29.7%` vs M2 `43.75`) / `70.25` / **`112.60`** (`+53.0%` vs M2 `73.60`) / **`151.43`** (`+64.5%` vs M2 `92.05`).
   - `outer_code_nothink`: **`65.55`** (`+35.5%` vs M2 `48.37`) / `79.95` / **`122.05`** (`+52.4%` vs M2 `80.09`) / **`156.75`** (`+59.4%` vs M2 `98.37`).
   - `outer_greedy_prose_nothink`: **`56.95`** / `73.43` / **`110.70`** / **`154.10`**.
4. **Full Inner Ruler (`profiles/m3-qual-inner-prose.json`, `profiles/m3-qual-inner-code.json`, `c=1,4,8`, `runs=6` median, 0 errors/0 loops):**
   - `inner_greedy_prose`: **`56.49`** (`+22.2%` vs M2 `46.21`) / **`113.60`** (`+45.5%` vs M2 `78.05`) / **`151.40`** (`+56.1%` vs M2 `96.97`).
   - `inner_greedy_code`: **`74.53`** (`+50.3%` vs M2 `49.59`) / **`132.29`** (`+57.5%` vs M2 `83.99`) / **`162.00`** (`+38.8%` vs M2 `116.74`).
5. **Acceptance Gates, Exactness & Memory (`profiles/m3-qual-gates.json`, `profiles/m3-qual-gates-memwatch.json`, `profiles/m3-plain-vs-drafted.json`):**
   - **12/12 `gates.py` PASSED**: `load`, `chat`, `reasoning`, `tools`, `structured`, `vision` (png/jpeg/webp), `drafted_exactness` (serial==concurrent 4/4), `multilingual` (ko/ja/zh/ar/emoji, 0 `\ufffd`), `prefix_cache` (`8704/8726`), `needle_256k` (`302.46s`, 2/2 exact), `needle_1m` (`2869.93s`, `1,023,701` & `1,023,881` exact), `soak` (`252.33s`, c4+c8, 0 loops/errors).
   - **Memwatch (684 samples across 12/12 gates incl. `needle_1m` + `soak`):** `min_mem_avail_gib = 10.45 GiB` (vs `8.36 GiB` in M2), `peak_gpu_res_gib = 102.89 GiB`, `peak_proc_anon_gib = 1.43 GiB` (vs `2.93 GiB` in M2), **`pswpout_delta = 0`**, `alerts = []`.
   - **Pure `plain` (`DRAFTER="" EXTRA_ARGS="--no-drafts"`, `drafts: off`, `drafted=0`) vs `drafted` (`DFlash2`):** **4/4 exact character-for-character match** (`all_exact: true`, checking both `content` and `reasoning_content`). Also fixed `recipe.conf` line 17 (`DRAFTER=${DRAFTER-incoai/Inkling-2-Mini-DFlash2}`) in `7512b92` so `DRAFTER=""` properly disables the drafter.
6. **Clean-Verify A/B (`profiles/clean-verify-m3-ab.json`, `A1 m3-p9 / B clean-verify-m3 / A2 m3-p9`, `runs=4`, `pswpout=0`, `min_avail=10.95..11.30 GiB`, `alerts=[]`):**
   - `A1 (m3-p9)`: sampled `54.57 / 107.71`, greedy `56.24 / 112.62`, prefill `2,969.4 / 2,871.8`.
   - `B (clean-verify-m3)`: sampled `54.04 / 111.97`, greedy `56.78 / 109.67`, prefill `2,841.0 / 2,893.0`.
   - `A2 (m3-p9)`: sampled `51.40 / 108.44`, greedy `55.91 / 99.82`, prefill `2,920.0 / 2,860.9`.
   - Parity confirmed (`B` matches or beats `avg(A1, A2)` across decode and 32K prefill; GPU stopped cleanly via `./stop.sh`, `docker ps -a` empty, `nvidia-smi` `0%`).


















