# ngspice-47 Performance Optimization Changelog (`SPEEDUP.md`)

**Branch:** `opt/zen4-transient` (based on `ngspice-47`, commit `a80f6e3e9`)
**Reference Host:** AMD Ryzen 9 7940HX (Zen 4, 16c/32t, 2×8-core CCDs, AVX-512, FMA, 32 KiB L1d / 1 MiB L2 per core, 64 MiB L3, GCC 13.3.0, GLIBC `libmvec.so.1`)
**Numerical Policy:** Strict IEEE-754 double precision throughout — **no `-ffast-math`**, no physics approximations. Every stage is verified against stock `ngspice-47` golden `.raw` waveforms and built-in `tests/bsim4` (`nmos/qaSpec`, `pmos/qaSpec`).

---

## Compile-Time Feature Flags (Porting Controls)

All optimizations are cleanly separated into two independent compile-time flags defined in `src/include/ngspice/opt_config.h` and wired into `configure.ac`:

| Flag | `./configure` Option | Default | Scope & Portability |
|---|---|---|---|
| `NGSPICE_OPT_GENERAL` | `--enable-opt-general` / `--disable-opt-general` | `1` (enabled) | **100% portable C99 / OpenMP** (zero ISA intrinsics — works on x86-64, ARM64/Apple Silicon, RISC-V, etc.): O(1) `INPgetModBin` bin index, `--precompile`/`.uselib` PDK caching, unified KLU TLS parallel load, pre-baked RES, flat CAP & fast `CAPtrunc`, KLU refactor lazy scatter-map + branchless scaling + alias-free 4× unroll, Sparse slab-linear `spClear` & `spFactor` zero-skip, `CKTterr` precomputed weights, `NIconvTest` flat byte cache, `OldCKTstate0` guard, BSIM4v5/v6 scalar fast-paths. |
| `NGSPICE_OPT_ZEN4` | `--enable-opt-zen4` / `--disable-opt-zen4` | `auto` (`1` iff `__AVX512F__ && __GNUC__`, else `0`) | **AMD Zen 4 / AVX-512 SIMD kernels** (`bsim4_avx512_core.h`, `bsim4_avx512_glue.h`): 8-wide AVX-512 SoA BSIM4v5/v6 evaluator, inline Zen 4 `vrndscalepd`/`vscalefpd`/`vgetexppd`/`vgetmantpd` `exp`/`log` (≤2 ULP), 8-wide trap-assisted tunneling, model-sorted packing, 8-wide AVX-512 `fetlim`/`limvds`/`pnjlim`, `convTestAvx`, `truncAvx`, and two-level stationary-voltage block cache with exact first-order Taylor charge sync. |
| `NGSPICE_OPT_APPLE` | `--enable-opt-apple` / `--disable-opt-apple` | `auto` (`1` iff `__aarch64__ && __ARM_NEON`, else `0`) | **Apple Silicon (ARM64 NEON) optimizations** (`bsim4_apple_core.h`, `bsim4_apple_glue.h`, KLU, Sparse, `nghash`/`numparam`, `niconv`): 3-phase 4-wide (`2× float64x2_t`) NEON BSIM4v5/v6 kernel shared across both models, deduplicated L1D-resident scalar `BSIM4ApplePParam` table, 128B cache-line `BSIM4AppleCheck` with `W=1` stationary bypass + dynamic miss compaction + zero-`s1`/`s2` algebraic L1 charge sync, zero-division KLU solve (`invUdiag`/`invRs_perm`) & unsigned `GET_POINTER`, pre-baked constant CAP base, flat contiguous CSC Sparse refactor/solve fast path, fast power-of-two in-place-resize `nghash` & zero-copy `numparam` subcircuit parameter swap-merge. |

Both flags can also be overridden directly via `CFLAGS` (e.g. `CFLAGS="-O3 -DNGSPICE_OPT_GENERAL=1 -DNGSPICE_OPT_ZEN4=0"`). Building with `--enable-opt-general --disable-opt-zen4 CFLAGS="-O3 -march=x86-64 -mno-avx512f"` is verified to emit **0** `%zmm` instructions.

Runtime diagnostic switches (no rebuild required):
- `NGSPICE_NO_AVX512=1` — disable AVX-512 BSIM4 kernels at runtime
- `NGSPICE_NO_BLOCK_BYPASS=1` — disable the two-level stationary BSIM4 block cache
- `NGSPICE_AVX_BYPASS_TOL=<volts>` — override the stationary block cache voltage threshold (default: `min(CKTvoltTol, 1e-6)`)
- `NGSPICE_NO_CHORD=1` — disable Shamanskii / Modified Chord Newton KLU Jacobian reuse
- `NGSPICE_NO_PARTIAL_KLU=1` — disable incremental / partial KLU column refactorization & reciprocal-multiply `KLU_solve`
- `NGSPICE_NO_LANE_COMPACT=1` — disable dynamic active-lane SIMD compaction inside AVX-512 BSIM4 evaluation

---

## Summary of Final Speedups (Stages 1–8)

### Hero >10-Minute Benchmark (`sar12_hero_16cyc`: GF180 12-bit Post-PEX Redundant SAR ADC, 16 Cycles, >206k Newton Iters)

| Configuration | Stock `ngspice-47` | Stage 4 | **Final (Stage 7/8)** | Speedup (Same Config) | Speedup vs Stock Sparse 1t | Accuracy vs Golden |
|---|---:|---:|---:|---:|---:|---|
| **KLU 1t (Primary ST Target)** | — | 128.99 s | **69.40 s** (`1m 09s`) | — | **9.06×** (**1.91× faster than stock KLU 8t!**) | 16/16 SAR codes exact, max EOC Δt `110.74 ps` |
| **Sparse 1t** | **628.48 s** (`10m 28s`) | 200.71 s | **143.20 s** (`2m 23s`) | **4.39×** | **4.39×** | 16/16 SAR codes exact, max EOC Δt `99.16 ps` |
| **KLU 8t (1 CCD)** | **132.71 s** (`2m 13s`) | 39.19 s | **31.81 s** | **4.17×** | **19.76×** | 16/16 SAR codes exact, max EOC Δt `110.74 ps` |
| **Sparse 8t** | — | 113.76 s | **105.16 s** (`1m 45s`) | — | **5.98×** | 16/16 SAR codes exact, max EOC Δt `99.16 ps` |

### Probe Suite Progression Across All Stages (Wall Time, Seconds)

All 4 circuits are real GF180 post-PEX mixed-signal circuits from `@aicdb`:
- `sar12_probe_2cyc` (1,559 eqns, 2 SAR cycles, ~26k–27k iters)
- `ti12_full_probe_30ns` (7,221 eqns, full-chip 3-ch TI 12b SAR + PLL, ~2.1k iters)
- `pll1p2g_probe_20ns` (1,307 eqns, 1.2 GHz AD-PLL, ~2.9k iters)
- `sdm16_probe_8cyc` (3,609 eqns, 16-bit switched-capacitor SDM, ~5.6k iters)

| Deck | Config | Stock Sp 1t | Portable (`GENERAL=1, ZEN4=0`, `-mno-avx512f`) | Stage 4 (Solver) | Stage 6 (ST AVX-512) | **Final Stage 7/8** | Speedup vs Stock Sp 1t | Final `rel_rms` |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `sar12_probe_2cyc` | **KLU 1t** | — | 48.37 | 16.77 | 12.72 | **9.16** | **10.80×** | `9.27e-03` (2/2 codes exact) |
| | Sparse 1t | 98.94 | — | 28.57 | 24.65 | **20.92** | **4.73×** | `3.64e-02` (2/2 codes exact) |
| | KLU 8t | — | — | 5.34 | 4.96 | **4.37** | **22.64×** | `9.27e-03` (2/2 codes exact) |
| `ti12_full_probe_30ns` | **KLU 1t** | — | 29.69 | 16.36 | 13.69 | **8.86** (~7.4 w/ `.uselib`) | **6.89×** (~8.3× w/ `.uselib`) | `9.99e-13` (exact grid) |
| | Sparse 1t | 61.07 | — | 22.48 | — | **14.43** | **4.23×** | `1.08e-12` (exact grid) |
| | KLU 8t | — | — | 7.26 | 6.00 | **5.04** | **12.12×** | `9.12e-13` (exact grid) |
| `pll1p2g_probe_20ns` | **KLU 1t** | — | 6.70 | 3.39 | 2.09 | **1.51** | **7.28×** | `1.52e-11` (exact grid) |
| | Sparse 1t | 11.00 | — | 3.76 | 2.42 | **1.92** | **5.73×** | `1.52e-11` (exact grid) |
| | KLU 8t | — | — | 1.34 | — | **0.92** | **11.96×** | `1.53e-11` (exact grid) |
| `sdm16_probe_8cyc` | **KLU 1t** | — | 3.88 | 3.17 | 2.07 | **1.89** | **4.39×** | `4.75e-09` |
| | Sparse 1t | 8.29 | — | 5.07 | 3.90 | **3.81** | **2.18×** | `6.67e-10` |
| | KLU 8t | — | — | 1.88 | — | **1.64** | **5.05×** | `5.56e-09` |

### Explicit PDK Precompilation & Netlist Load Speedup (Stage 5 — Top-Level Netlist Uncached)

| Deck | Stock Netlist Load | Opt (No `.uselib`, O(1) Bin Index) | Opt + `.uselib` (`pdk_gf180_typ.nglib`) | Pure Load Speedup |
|---|---:|---:|---:|---:|
| `ti12_full_probe_30ns` | 3.804 s | 2.405 s (1.58×) | **1.429 s** | **2.66×** |
| `pll1p2g_probe_20ns` | 0.733 s | 0.562 s (1.30×) | **0.235 s** | **3.13×** |
| `sar12_probe_2cyc` | 0.460 s | 0.340 s (1.35×) | **0.200 s** | **2.30×** |
| `sdm16_probe_8cyc` | 0.227 s | 0.168 s (1.35×) | **0.097 s** | **2.34×** |

Precompiling the full GF180 typical PDK (`design.ngspice` + `sm141064.ngspice` corners + `gf180mcu_fd_sc_mcu7t5v0.spice`) takes **21.8 ms** and writes a **428.1 KiB** binary `NGLIB001` blob.

---

## Detailed Change Log & Porting Guide

### Stage 1: Compiler & Target-Specific Build Configuration (No Source Changes)
- **Flags:**
  ```sh
  CFLAGS="-O3 -march=znver4 -mtune=znver4 -fno-math-errno -fno-trapping-math -fno-semantic-interposition -flto=8 -g1"
  LDFLAGS="-flto=8 -O3 -march=znver4 -mtune=znver4 -lmvec"
  ```
- **Gain:** **1.16×–1.20×** across all 4 decks (`sar12` Sparse 1t `98.94s -> 83.10s`).
- **Flag:** Compiler flags (on non-Zen4 hosts, replace `-march=znver4 -mtune=znver4` with `-march=native`).

### Stage 2: Algorithmic & Dead-Work Eliminations (`fdd0b3244`, 21 files) — `NGSPICE_OPT_GENERAL`
- **Gain:** **1.18×–1.48×** over Stage 1 (`sar12` Sparse 1t `83.10s -> 64.23s`, KLU 1t `61.46s -> 52.52s`, KLU 8t `17.59s -> 11.87s`).
- **Major changes:**
  1. Guarded dead `OldCKTstate0` allocation & `memcpy` in `src/maths/ni/niiter.c` (only used for DC/TRANOP node damping).
  2. Precomputed `CKTterr` divided-difference linear weights once per timestep in `src/spicelib/analysis/ckttrunc.c` / `cktterr.c`.
  3. Flat `CKTnodeIsVolt` byte array cache in `src/maths/ni/niconv.c` replacing linked-list `CKTnodes` walk.
  4. Slab-linear `spClear` (`src/maths/sparse/spalloc.c`, `spbuild.c`) & `memset` `SMPclear` (`src/maths/KLU/klusmp.c`).
  5. Linear passive fast paths (`RESload` skipping unused transient `REScurrent`; hoisted `CAPload`).
  6. Scalar BSIM4v5/v6 fast paths (`hasTrapTunnel` flag, precomputed `temp`-invariant factors, inlined `integrate_fast()` & `polyDepletion()`, 4-terminal `LoadRhsMat`, OpenMP `BSIM4v*trunc` / `convTest`).

### Stage 3A: Unified Parallel KLU TLS Load, Pre-Baked RES, Flat CAP & KLU Refactor Fast Scaling (`d426e7c58`, 35 files) — `NGSPICE_OPT_GENERAL`
- **Gain:** **1.05×–1.13×** single-thread, **1.23×–1.41×** multi-thread (`sar12` KLU 1t `52.52s -> 46.50s`, KLU 8t `11.87s -> 9.22s`).
- **Major changes:**
  1. Added `DEVkluPrep` & `DEVkluLoadPar` to `SPICEdev` (`devdefs.h`, `cktload.c`, `klusmp.c`): single `#pragma omp parallel` region per `CKTload` with 64B-aligned TLS `Ax`/`rhs` buffers and deterministic 256-wide tiled reduction; zero TLS overhead on `nthreads == 1`.
  2. Pre-baked constant linear resistors (`RESkluPrep` -> `KLUmatrixAxResConst` copied via `memcpy` during load) and contiguous `CAPfastArr` (`CAPloadKluPar`).
  3. Direct BSIM4v5/v6 stamping (skipping `_34.._61` buffer copies) and KLU refactor `invRs` / `inv_ukk` / `IS_NONZERO(ujk)` zero-skip.

### Stage 3B: 8-Wide AVX-512 SoA BSIM4v5 & BSIM4v6 Kernel (`31e0bc6cb`, 10 files) — `NGSPICE_OPT_ZEN4`
- **Gain:** **1.60×–2.67×** single-thread, **1.31×–1.68×** multi-thread (`sar12` KLU 1t `46.50s -> 17.44s`, Sparse 1t `61.54s -> 31.07s`).
- **Major changes:**
  - `src/include/ngspice/bsim4_avx512_core.h` & `bsim4_avx512_glue.h`: 8-wide double-precision (`v8d`) SoA evaluator for standard 4-terminal BSIM4v5/v6 transient MOSFETs with seamless scalar fallback.

### Stage 4: Linear Solver Inner-Loop Optimization (`d00bb7708`, 4 files) — `NGSPICE_OPT_GENERAL`
- **Gain:** **1.04×–1.16×** Sparse, **1.03×–1.04×** KLU (`sar12` Sparse 1t `31.07s -> 28.57s`, KLU 1t `17.44s -> 16.77s`).
- **Major changes:** `restrict` + 4× unrolled loops in `klu_refactor.c`, `klu.c`, `klu_solve.c`; exact `Mult != 0.0` zero-skip in `spfactor.c`.

### Stage 5: Explicit PDK Precompiler (`--precompile` / `precompile` / `.uselib`) & O(1) `INPgetModBin` (`a1acb9080`, 10 files) — `NGSPICE_OPT_GENERAL`
- **Gain:** **2.30×–3.13×** pure netlist load speedup with `.uselib` (and **1.30×–1.58×** even without `.uselib`), bit-identical transient waveforms.
- **Major changes:**
  1. `src/spicelib/parser/{inpgmod.c,inpmkmod.c,inpkmods.c,inpxx.h}`: Single-pass zero-malloc `scan_net_tok()` + generation-versioned hash index `modbin_hash` turning `INPgetModBin` from O(M×N) linear scan into O(1) lookup.
  2. `src/frontend/nglib_cache.h`, `inpcom.c`, `commands.c`, `fteext.h`, `main.c`:
     - CLI: `ngspice --precompile <out.nglib> <in_pdk.spice>`
     - Interactive/control: `precompile <out.nglib> <in_pdk.spice>`
     - Netlist directive: `.uselib "<path/to/pdk.nglib>"` (top-level netlist is NOT cached; only the fixed PDK blob is precompiled).
     - Binary `NGLIB001` format: resolves `.lib`/`.include`/continuations, pre-evaluates top-level `.model` parameters, folds zero-switch mismatch expressions, and eliminates dead local `.param`s inside `.subckt`s (140 dead local params eliminated on GF180 typical PDK).

### Stage 6: Single-Thread BSIM4 AVX-512 Overhaul (`e7d4af7d8`, 11 files) — `NGSPICE_OPT_ZEN4`
- **Gain:** **1.20×–1.62×** single-thread KLU (`sar12` `16.77s -> 12.72s`, `pll1p2g` `3.39s -> 2.09s`, `sdm16` `3.17s -> 2.07s`, `ti12_full` `16.36s -> 13.69s`).
- **Major changes:**
  1. Replaced all 37 external `_ZGVeN8v_exp`/`_ZGVeN8v_log` calls in `bsim4_avx512_core.h` with inline Zen 4 hardware AVX-512 `v8_exp` (`vrndscalepd`+`vscalefpd`, max 2 ULP, `4.30e-16`) and `v8_log` (`vgetexppd`+`vgetmantpd`, max 1 ULP, `2.22e-16`) — eliminating cross-call ZMM register spilling.
  2. Added 8-wide AVX-512 reverse-bias trap-assisted tunneling (`hasTrapTunnel`), moving **100% of GF180 `pfet_03v3`** out of scalar fallback.
  3. Model-sorted block packing (`cmpEligIdx` by model/tunnel/dint/sint/instIdx), 8-wide AVX-512 `v8_fetlim`/`v8_limvds`/`v8_pnjlim`, compact 64B-aligned `BSIM4Avx512Dyn` eliminating 24 AoS `here->...` writes per iteration unless `b4v*ask.c` is triggered, 8-wide AVX-512 `BSIM4v*convTestAvx`, and `omp_get_max_threads() > 1` guards eliminating `GOMP_parallel` overhead on 1t.

### Stage 7: Stationary Block Cache, AVX-512 `BSIM4trunc`, Fast `CAP` & KLU Refactor Scatter Map (`8bffde13a`, 14 files) — `NGSPICE_OPT_ZEN4` + `NGSPICE_OPT_GENERAL`
- **Gain:** **1.39×–1.55×** single-thread KLU (`sar12` `12.72s -> 9.16s`, `ti12_full` `13.69s -> 8.86s`, `pll1p2g` `2.09s -> 1.51s`; `sar12_hero_16cyc` `128.99s -> 69.40s` KLU 1t, `39.19s -> 31.81s` KLU 8t).
- **Major changes:**
  1. **Two-level stationary-voltage BSIM4 block cache with exact first-order Taylor charge sync** (`bsim4_avx512_core.h`, `bsim4_avx512_glue.h`, `NGSPICE_OPT_ZEN4`):
     - Guarded by `|dv| <= min(CKTvoltTol, 1e-6)`, same `vds` polarity, AND 10× tighter-than-`convTest` current tolerance (`0.1 * (reltol * max(|Ihat|, |I|) + abstol)`) so Newton convergence can never stall.
     - **Level 1 (same timestep & integration):** skips physics, integration, and matrix/RHS assembly; replays cached 20 stamp vectors directly (with first-order `dq = C * dv` `CKTstate0` sync if `|dv| > 1e-13`).
     - **Level 2 (cross-timestep stationary):** skips all 1,500 lines of BSIM4 DC+CV+junction+overlap physics; reuses cached DC/C matrices + charges and runs only charge integration + `ag0` stamp assembly (~30 vector ops).
  2. **Dead `CKTstate0` write elimination** (`bsim4_avx512_core.h`): skipped 11 unused state slots (`st+4..10, 19, 21, 27, 28`) unless `need_aos != 0`.
  3. **8-wide AVX-512 `BSIM4v5truncAvx` / `BSIM4v6truncAvx`** (`bsim4_avx512_glue.h`, `b4v5trunc.c`, `b4v6trunc.c`, `NGSPICE_OPT_ZEN4`): eliminates AoS `InstArray` cache misses and skips `sqrt` unless a lane beats `minStep` (`trn` 2.7×–2.9× faster).
  4. **Fast `CAPtrunc` & hoisted `CAPloadKluPar`** (`captrunc.c`, `capload.c`, `NGSPICE_OPT_GENERAL`).
  5. **KLU refactor lazy scatter map, branchless `fabs` scaling, and alias-free 4× unroll** (`klu.h`, `klu_factor.c`, `klu_free_numeric.c`, `klu_refactor.c`, `klu.c`, `klu_solve.c`, `NGSPICE_OPT_GENERAL`): precomputes `(ref_off_p, ref_blk_col, ref_blk_p, ref_blk_row)` once per numeric object (eliminating `Pinv[Ai[p]] - k1` and `newrow < 0` branches on every refactor) and hoists all 4 `Xr[i0..i3]` loads before stores to eliminate false compiler aliasing stalls.

### Stage 8: Portable vs Zen 4 Compile-Time Feature Flags (`78a7359f0`, 39 files)
- Added `src/include/ngspice/opt_config.h` and `configure.ac` (`--enable-opt-general`, `--enable-opt-zen4`).
- Verified clean compilation, linking, and waveform PASS across all 3 build profiles:
  1. Default Zen 4 (`NGSPICE_OPT_GENERAL=1, NGSPICE_OPT_ZEN4=1`)
  2. Portable General-Only (`NGSPICE_OPT_GENERAL=1, NGSPICE_OPT_ZEN4=0`, `-march=x86-64 -mno-avx512f`, 0 `%zmm` instructions)
  3. Stock-Equivalent (`NGSPICE_OPT_GENERAL=0, NGSPICE_OPT_ZEN4=0`, `-march=x86-64 -mno-avx512f`)

---

## Apple Silicon Optimizations (`opt/apple-silicon-transient` — `NGSPICE_OPT_APPLE`)

**Branch:** `opt/apple-silicon-transient` (based on `opt/zen4-transient` @ `53747dbda`, preserving `NGSPICE_OPT_GENERAL` + `NGSPICE_OPT_ZEN4` alongside `--enable-opt-apple`)
**Reference Host:** Apple M6 (`arm64`, 2 Super + 4 Performance + 6 Efficiency cores; Super core: 192 KiB L1I, 128 KiB L1D, 20 MiB L2, 128-byte cache lines; Apple Clang 21.0.0)
**Numerical Policy:** Strict IEEE-754 double precision throughout — **no `-ffast-math`**, verified 100% PASS against golden `.raw` waveforms (`compare_raw.py`) and built-in `tests/bsim4` (`nmos/qaSpec`, `pmos/qaSpec`).

### Hardware & Workload Profiling Findings (Apple M6)
1. **NEON FP64 vs SME2 Streaming SVE:** Hardware microbenchmarking on M6 Super core measured NEON FP64 at **27.9 GFMA/s (55.7 GFLOP/s)** (`fdiv.2d` 8.90 Gdiv/s, `fsqrt.2d` 4.61 Gsqrt/s) vs SME2 Streaming SVE (`SVL=512b`, `FEAT_SME_F64F64`) at only **8.8 GFMA/s** (3.2× slower) without `FEAT_SME_FA64` (`svdiv`/`svsqrt` raise `SIGILL`). NEON FP64 is strictly superior for BSIM4.
2. **4-Wide (`2× float64x2_t`) 3-Phase Kernel:** Splitting BSIM4 into 3 `noinline` phases (`vth/mob`, `ids/diode`, `cv/jct`) at `B4A_W = 4` keeps M6's 4 FP64 NEON pipes saturated across serial dependency chains (measured **1.27× faster per transistor** than `W=2`) while staying well inside M6's 128 KiB L1D and 192 KiB L1I.
3. **Massive `pParam` Redundancy:** Across the 4 GF180 circuits (789–16,680 MOSFETs), there are only **49–147 unique `(model, pParam, channel)` parameter sets** (16×–113× sharing), fitting completely in L1D as scalar structs (`54–160 KiB`) instead of duplicated per-block SIMD parameter arrays (`11–22 MiB`).
4. **Per-Instance (`W=1`) Bypass & Compaction:** Checking stationary bypass per transistor (`W=1`) over exact 128-byte (1 M6 cache line) `BSIM4AppleCheck` structs raises bypass hit rates by +6.5% to +11.2% over static `W=4` blocks (`ti12_full`: 63.4% -> 71.6%, `sar12`: 51.7% -> 62.9%), eliminating 18%–23% of full physics evaluations.
5. **128 KiB L1D Discipline:** A/B profiling proved that per-model KLU shadow delta-stamping buffers (`2 × 112 KiB` on `sar12`) evicted `my_ax` from M6's 128 KiB L1D and regressed `sar12` load by 0.39 s; removing them and computing L1 charge sync algebraically via `cq = base_cq + ag0 * dq` (zero `s1`/`s2` loads) restored full L1D residency.

### Summary of Apple M6 Single-Core Speedups (vs Stock `ngspice-47` & Zen 4 AVX-512)

| Deck | Solver | Stock `ngspice-47` (M6) | `opt/zen4` Portable (M6) | **`opt/apple` PGO+LTO (M6, std)** | **`opt/apple` + `.uselib` (M6)** | Speedup vs Stock M6 | Ref: Zen 4 AVX-512 (7940HX std) | Accuracy vs Golden |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `sdm16_probe_8cyc` | **KLU 1c** | 2.97 s | 2.34 s | **1.53 s** | **1.50 s** | **1.94×** (1.98×) | 1.91 s | `rel_rms = 4.63e-09` |
| | **Sparse 1c** | 4.48 s | 3.51 s | **1.74 s** | **1.71 s** | **2.57×** (2.62×) | 3.67 s | `rel_rms = 6.00e-10` |
| `pll1p2g_probe_20ns` | **KLU 1c** | 4.37 s | 3.42 s | **1.17 s** | **1.09 s** | **3.74×** (4.01×) | 1.52 s | `rel_rms = 1.30e-05` |
| | **Sparse 1c** | 4.55 s | 3.53 s | **1.26 s** | **1.18 s** | **3.61×** (3.86×) | 1.83 s | `rel_rms = 1.30e-05` |
| `ti12_full_probe_30ns` | **KLU 1c** | 21.48 s | 14.01 s | **3.67 s** | **3.34 s** | **5.85×** (6.43×) | 9.09 s | `rel_rms = 2.25e-06` |
| | **Sparse 1c** | 24.12 s | 16.00 s | **5.86 s** | **5.13 s** | **4.12×** (4.70×) | 13.97 s | `rel_rms = 2.25e-06` |
| `sar12_probe_2cyc` | **KLU 1c** | 32.95 s | 24.35 s | **7.50 s** | **7.37 s** | **4.39×** (4.47×) | 9.17 s | `2/2` exact, EOC Δt `76.4 ps` |
| | **Sparse 1c** | 42.23 s | 31.28 s | **10.03 s** | **9.79 s** | **4.21×** (4.31×) | 20.15 s | `2/2` exact, EOC Δt `30.4 ps` |
| **`sar12_hero_16cyc`** | **KLU 1c** | — | — | **58.30 s** | — | — | 69.40 s | `16/16` exact, EOC Δt `94.7 ps` |
| | **Sparse 1c** | — | — | **73.47 s** | — | — | 143.92 s | `16/16` exact, EOC Δt `110.5 ps` |

### Apple Silicon Stage Log (`NGSPICE_OPT_APPLE`)
- **Stage A (`4b88f4748`):** `--enable-opt-apple` / `NGSPICE_OPT_APPLE` build plumbing; zero-division KLU `klu_solve` (`invUdiag` + `invRs_perm` cached in `klu_refactor`, inlined `L`+`U` solve for `nr==1`, zero-copy identity `SMPsolve`); pre-baked constant `KLUmatrixAxCapUnit` + per-timestep cached `KLUmatrixAxResCapBase`; 2-wide NEON `NIconvTest`.
- **Stage B (`f6f8c7725`):** Ground-up Apple Silicon BSIM4v5/v6 redesign (`bsim4_apple_core.h`, `bsim4_apple_glue.h`): single compiled kernel shared by v5 & v6, 3-phase `B4A_W=4` (`2× float64x2_t`) NEON evaluator with inline polynomial `exp`/`log`, deduplicated L1D-resident scalar `BSIM4ApplePParam` table with predictable scalar feature branches, exact 128B cache-line `BSIM4AppleCheck` + pack-level step validation, and per-instance (`W=1`) L1/L2 bypass with dynamic miss compaction.
- **Stage C (`eca5fb44c`):** Flat contiguous CSC numeric refactor & solve fast path for Sparse (`spdefs.h`, `spalloc.c`, `spbuild.c`, `spfactor.c`, `spsolve.c`, `sputils.c`): eliminates pointer-chasing linked lists in `spFactor` and `spSolve` (toggleable via `NGSPICE_NO_SPARSE_FLAT=1`), cutting `sar12` Sparse `fac` from `6.06 s -> 1.70 s` (**3.56× faster**) and `slv` from `1.00 s -> 0.28 s` (**3.57× faster**).
- **Stage D (`2d06a40dc`):** Fast power-of-two FNV-1a `nghash` with cached 32-bit hash pre-filtering and zero-malloc in-place node relinking in `nghash_resize`; stack-buffered `dicostack_pop` and swap-merge `nupa_copy_inst_dico` cutting `ti12_full` parse/setup overhead by 40% (`1.55 s -> 0.93 s`).
- **Stage E (`0d2e1106b` + Clang 21 PGO):** Algebraic `cq = base_cq + ag0 * dq` L1 charge sync (zero `s1`/`s2` loads), removal of L1D-evicting KLU shadow buffers, unsigned `size_t` shift in KLU `GET_POINTER`, and Apple Clang 21 PGO + ThinLTO (`-O3 -mcpu=native -fno-math-errno -fno-trapping-math -flto=thin -fprofile-instr-use=...`).

---

### Stage 9: Scattered-Task Frontend/Cache, `INPparseTree` Folding + Affine `ASRC`, & AVX-512 BSIM4 DCOP (`NGSPICE_OPT_GENERAL` + `NGSPICE_OPT_ZEN4`)
- **Gain (single-core 1t on Ryzen 9 7940HX):**
  - `opa1k_pvt16` (1,000 post-PEX GF180 op-amps × 16 PVT corners = 16,000 `OP + AC + Tran` tasks):
    - Standard Sparse: **`483.27 s -> 133.66 s`** (**3.62×** vs Stage 8, **4.69×** vs stock `626.47 s`, `rel_rms = 2.90e-6`)
    - Standard KLU: **`497.39 s -> 138.61 s`** (**3.59×** vs Stage 8, **4.59×** vs stock KLU `636.08 s`, `rel_rms = 5.30e-5`)
    - `.uselib` KLU: **`366.78 s -> 132.37 s`** (**2.77×** vs Stage 8, **4.73×** vs stock `626.47 s`, `rel_rms = 5.30e-5`)
  - Also accelerates all 5 hero/probe transient workloads (`pll_ring_150m_4us` KLU `31.59 s -> 29.29 s`, `ti12_full_300ns` KLU `48.56 s -> 45.34 s`, `sar12_hero_16cyc` Sparse `149.74 s -> 137.49 s`, `sdm16_120cyc` `.uselib` KLU `30.55 s -> 29.96 s`), with **18/18 golden PASS** and bit-exact determinism between Standard and `.uselib`.
- **Major changes:**
  1. **`INPparseTree` constant-folding & affine `ASRC` fast-path** (`inpptree.c`, `inpptree.h`, `asrc{defs.h,set,load,conv,del,par}.c`, `NGSPICE_OPT_GENERAL`):
     - Added recursive finite-safe `PTsimplify()` before `PTdifferentiate()` in `INPgetTree()`, eliminating `0 * abs(v(11,21))` bloat in GF180 `ppolyf_u_1k` behavioral resistors and folding constant subexpressions.
     - Added structural `INPtreeIsAffine()` detection (including `temper`-parameterized coefficients) with cached `(c0, g_i)` evaluation in `ASRCload` and `ASRCconvTest`, eliminating the `14.5%` `PTeval` hotspot completely.
  2. **Stat-validated in-memory `.nglib` & `.lib` caches + transitive `.subckt`/`.model` filtering** (`nglib_cache.h`, `inpcom.c`, `NGSPICE_OPT_GENERAL`):
     - Shared `NgDeckIndex` pre-indexes top-level `.subckt` spans (FNV-1a hash table) and `.model` base names once per cached library, validated on every hit via `stat()` `(st_dev, st_ino, st_size, mtime, compat_mask)` across the file and all transitive `.lib`/`.include` dependencies.
     - Materializes only top-level cards + transitively referenced `.subckt`/`.model` entries via multi-anchor BFS (cutting GF180 PDK materialization from `5,757` cards to `~100` cards per op-amp), spliced after comment-stripping/continuation-stitching of the top-level netlist.
     - Direct absolute-path `orig_path` lookup in `read_a_lib()` eliminates redundant `realpath()`/`readlink()` syscalls.
  3. **Fast frontend string & I/O primitives** (`ngspice.h`, `stringutil.h`, `string.c`, `inpcom.c`, `inpgmod.c`, `NGSPICE_OPT_GENERAL`):
     - Inline pure-ASCII `tolower_c`/`toupper_c`/`isspace_c`/`isdigit_c`/`isalpha_c`/`isalnum_c` and `extern inline` `ciprefix()`/`cieq()` eliminating glibc locale TLS overhead.
     - Buffered `fgets()` + 1 KiB stack-buffer `readline()` replacing byte-by-byte `getc()` + `TREALLOC`; in-place `inp_remove_ws()`; fast `.` guard in `inp_cider_models()`; early-exit `parse_line()` in `inpgmod.c`.
  4. **8-wide AVX-512 BSIM4v5/v6 DCOP/TRANOP fast path + deterministic packing + inline scalar Zen 4 `exp`/`log`** (`bsim4_avx512_core.h`, `bsim4_avx512_glue.h`, `b4v5ld.c`, `b4v6ld.c`, `NGSPICE_OPT_ZEN4`):
     - Extended AVX-512 SoA evaluation and `convTestAvx` to `MODEDCOP`/`MODETRANOP` (`MODEINITJCT`, `MODEINITFIX`, `MODEINITFLOAT`), skipping the C-V charge/integration block (`dc_only`) while preserving all `CKTstate0[st+0..10, 27]` and `here->von` handoffs for `MODEINITSMSIG`/`MODEINITTRAN`.
     - Fixed latent ASLR sensitivity in `cmpEligIdx` by sorting on `GENmodName` instead of raw model pointer address.
     - Added inline scalar Zen 4 `b4_s_exp`/`b4_s_log` (`<= 2 ULP`, hardware `vscalefsd`/`vgetexpsd`/`vgetmantsd`) for scalar `MODEINITSMSIG`/`MODEINITTRAN` paths.
  5. **Compact active-device dispatch, DCOP KLU `AxResConst`, & lazy-calibrated invariant-TSC `seconds()`** (`cktdefs.h`, `cktsetup.c`, `cktload.c`, `acan.c`, `resload.c`, `misc_time.c`):
     - Compact `CKTactiveDevs[]` list avoids scanning 70 empty device slots per `CKTload`/`CKTacLoad`; enabled KLU `AxResConst` pre-baked resistor stamping during `MODEDCOP`/`MODETRANOP` (computing `REScurrent` only on `MODEINITSMSIG` where noise analysis reads it); removed dead duplicate `IFseconds()` in `CKTload`; lazy-calibrated invariant-TSC `seconds()` under `NGSPICE_OPT_ZEN4`.

### Stage 10: Literature-Driven Advanced Solvers & Dynamic SIMD Compaction (`NGSPICE_OPT_GENERAL` + `NGSPICE_OPT_ZEN4`)
- **Literature basis:**
  - **Shamanskii / Modified Chord Newton** (Hindmarsh et al., SUNDIALS IDA, *ACM TOMS* 2005; Spectre / Xyce transient Newton reuse)
  - **Incremental / Partial KLU Refactorization** (Dinkelbach et al. 2021; NICSLU / Chen et al. *IEEE TCAD*)
  - **Division-free KLU Triangular Solve** (`invUdiag` + `invRsPerm` eliminating `2n` serial double divisions per solve)
  - **Dynamic Active-Lane SIMD Stream Compaction** (Kapre & DeHon; Gulati & Khatri; Shen et al. *DAC/TCAD*)
- **Gain (single-core 1t on Ryzen 9 7940HX vs Stage 9 / vs Stock Sparse 1t):**
  - `sar12_hero_16cyc`: KLU **`70.30 s -> 60.64 s`** (**-13.7%**, **10.36×** vs stock Sparse `628.48 s` — breaks 10× single-core on the 10-minute hero benchmark; 16/16 SAR codes exact!)
  - `pll_ring_150m_4us`: KLU **`29.29 s -> 22.70 s`** (**-22.5%**, **7.46×** vs stock `169.36 s`); Sparse **`36.00 s -> 30.54 s`** (**-15.2%**)
  - `sdm16_120cyc`: KLU **`30.60 s -> 24.70 s`** (**-19.3%**, **10.17×** vs stock `251.23 s`; 115/115 bit-exact PDM, `rel_rms = 6.77e-3`); Sparse **`70.26 s -> 69.33 s`** (`rel_rms = 5.25e-3`)
  - `ti12_full_300ns`: KLU **`45.34 s -> 39.16 s`** (**-13.6%**, **5.55×** vs stock `217.50 s`, `rel_rms = 1.72e-6`; `.uselib` **`38.87 s`**); Sparse **`84.10 s -> 77.01 s`** (**-8.4%**, `rel_rms = 3.04e-11`)
  - `sar12_probe_2cyc`: KLU **`9.01 s -> 8.13 s`** (**-9.8%**, **12.17×** vs stock `98.94 s`; `.uselib` **`8.00 s`**, 2/2 SAR codes exact)
  - `opa1k_pvt16`: KLU **`138.61 s -> 134.76 s`** (`-2.8%`; `.uselib` **`132.37 s -> 127.31 s`**, **4.92×** vs stock `626.47 s`)
  - **18/18** golden PASS across `opt_std (klu + sparse)` and `opt_uselib (klu)`, bit-identical KLU `rel_rms` between Standard and `.uselib`, all 3 compile-flag profiles verified (`GENERAL=1/ZEN4=1`, portable `GENERAL=1/ZEN4=0` with `0` `%zmm` instructions, and `GENERAL=0/ZEN4=0`), and built-in `tests/bsim4` (`test1..14`) PASS.
- **Major changes:**
  1. **Shamanskii / Modified Chord Newton for KLU** (`cktdefs.h`, `smpdefs.h`, `klusmp.c`, `nireinit.c`, `niiter.c`, `NGSPICE_OPT_GENERAL`, toggle `NGSPICE_NO_CHORD=1`):
     - Reuses untouched `KLU_numeric` on polishing Newton iterations (`iterno >= 2`, `ag0 == lu_ag0`, `prev_dx <= 0.05`, `CKTnoncon == 0`) and smooth constant-step first iterations (`iterno == 1`, `0.90 <= ag0/lu_ag0 <= 1.11`, `dx < 0.015`, `CKTkluChordCount < 10`) by computing exact residual `r = RHS - (A + Gmin*I)*Xold` and solving `L*U*dx = r` without overwriting `RHS` or `Ax`. Seamless same-iteration fallback to `SMPluFac + SMPsolve` whenever contraction fails (`dx >= 0.35 * prev_dx && dx > 5e-5`). Cuts `klu_refactor` calls by 37–49%.
  2. **Exact Incremental / Partial KLU Refactorization & Division-Free `KLU_solve`** (`klu.h`, `klu_internal.h`, `klu_version.h`, `klu_factor.c`, `klu_free_numeric.c`, `klu_refactor.c`, `klu.c`, `klu_solve.c`, `NGSPICE_OPT_GENERAL`, toggle `NGSPICE_NO_PARTIAL_KLU=1`):
     - Freezes row scaling `Rs` after the first refactor (keeping exact numerical LU via frozen `invRs`), tracks `ref_az_prev[0..nz-1]` and per-column dirty propagation through `Ui` dependencies, and skips every unchanged BTF column `k` bit-exactly.
     - Caches reciprocal pivot diagonals `ref_inv_udiag[k] = 1.0 / Udiag[k]` and permuted inverse row scale `ref_inv_rs_perm[k] = invRs[Pnum[k]]`, replacing all `2n` serial double divisions per solve with multiplies in `KLU_solve` and `KLU_usolve_inv`.
  3. **Complex Sparse `FactorComplexMatrix` Zero-Skip** (`spfactor.c`, `NGSPICE_OPT_GENERAL`):
     - Skips complex multiply-subtract inner loops when `dc.Real == 0.0 && dc.Imag == 0.0` (accelerates `opa1k_pvt16` AC analysis).
  4. **Zero-Bloat 4-Way Ground-Skip Stamping, Hot-in-`Dyn` L2 Co-location, & Cached Charge History** (`bsim4_avx512_core.h`, `bsim4_avx512_glue.h`, `NGSPICE_OPT_ZEN4`):
     - Mirrors hot read-only topology/stamp metadata (`type_v`, `d/g/s/bNodePrime`, `stateBase`, `kluOff16`) into `BSIM4Avx512Dyn` so L1 hits, `convTestAvx`, `terrBlock8`, and KLU stamping never stride across cold 10.5 KiB `BSIM4Avx512Block` structs.
     - 4-way `(sn != 0, bn != 0)` ground-skip stamping eliminates the 12-deep serialized `my_ax[inz]` store-forwarding stall chain on grounded MOSFETs with zero struct bloat.
     - Caches timestep integration history `hist_b/g/d` in `dyn` so `bsim4_avx512_l1_sync_state` updates `CKTstate0` in pure AVX-512 with zero `CKTstate1`/`CKTstate2` reads.
  5. **Dynamic Active-Lane SIMD Compaction** (`bsim4_avx512_core.h`, `bsim4_avx512_glue.h`, `b4v5ld.c`, `b4v6ld.c`, `NGSPICE_OPT_ZEN4`, toggle `NGSPICE_NO_LANE_COMPACT=1`):
     - In partially active 8-wide BSIM4 blocks (`1..5` moving lanes with `>=1` stationary lanes), stationary lanes take their exact L1/L2 bypass path while moving lanes are packed across blocks into dense 8-wide AVX-512 batches using an in-register 8×8 AVX-512 `double` transpose (`b4_transpose_8x8`) over contiguous per-instance `BSIM4Avx512Inst` records (allocated contiguously after `blocks[nBlocks]` when `nBlocks > 4`) and 1-cycle `_mm512_test_epi64_mask` (`vptestmq`) mask packing.
     - Preserves exact `0..nBlocks-1` stamping summation order on Sparse 1t (`stamp_mode != 0`) while eliminating 25–47% of redundant BSIM4 lane evaluations on large mixed-signal workloads (`ti12_full_300ns`, `pll_ring_150m_4us`, `sar12_hero_16cyc`).

### Stage 11: Portable Solver & Frontend Convergence from `opt/apple-silicon-transient` (`NGSPICE_OPT_GENERAL`)
- **Gain (single-core 1t on Ryzen 9 7940HX vs Stage 10 / vs Stock Sparse 1t):**
  - `sar12_hero_16cyc`: KLU **`60.64 s -> 55.56 s`** (**11.28×** vs stock Sparse `626.75 s`); Sparse **`142.57 s -> 77.84 s`** (**-45.4%**, `spFactor` **3.4×** faster, `spSolve` **4.0×** faster!)
  - `ti12_full_300ns`: KLU **`39.16 s -> 33.30 s`** (**-15.0%**, **13.95×** vs stock Sparse `464.51 s`); Sparse **`77.01 s -> 56.77 s`** (**-26.3%**, `rel_rms = 3.04e-11`)
  - `pll_ring_150m_4us`: KLU **`22.70 s -> 21.37 s`** (**10.86×** vs stock Sparse `232.14 s`); Sparse **`30.54 s -> 24.04 s`** (**-21.3%**)
  - `sdm16_120cyc`: KLU **`24.70 s -> 23.05 s`** (**-6.7%**); Sparse **`69.33 s -> 32.07 s`** (**-53.7%**, 115/115 exact PDM bits, `rel_rms = 5.25e-3`)
  - `sar12_probe_2cyc`: KLU **`8.13 s -> 7.48 s`** (**10.89×** vs stock Sparse `81.46 s`); Sparse **`18.91 s -> 11.20 s`** (**-40.8%**)
  - `opa1k_pvt16`: KLU **`134.76 s -> 130.88 s`** (`-2.9%`); Sparse **`134.72 s -> 123.34 s`** (**-8.4%**, `5.08×` vs stock `626.47 s`)
  - **12/12** golden PASS across KLU & Sparse with **bit-identical** Sparse iteration counts and `rel_rms` vs linked-list Sparse.
- **Major changes (all 100% portable C99 under `NGSPICE_OPT_GENERAL`):**
  1. **Flat contiguous CSC numeric refactor & exact row-oriented CSR backward solve for Sparse** (`spdefs.h`, `spalloc.c`, `spbuild.c`, `spfactor.c`, `spsolve.c`, `sputils.c`):
     - Compiles the Sparse linked-list LU structure into contiguous CSC arrays `(FlatColStart, FlatDiagIdx, FlatRow, FlatVal, FlatElemPtr)` plus a row-oriented strict-upper CSR index `(FlatURowStart, FlatUCol, FlatUValIdx)` that preserves exact stock `NextInRow` floating-point summation order in backward substitution while eliminating all pointer chasing.
  2. **Pre-baked unit-capacitance KLU matrix + per-timestep CAP RHS cache + `KLUmatrixIdentityMap`** (`capdefs.h`, `capload.c`, `klu.h`, `klusmp.c`, `cktload.c`):
     - Pre-bakes constant linear capacitance stamps into `KLUmatrixAxCapUnit` and combines `res_base + ag0 * cap_unit` once per `ag0` into `KLUmatrixAxResCapBase`, turning 100k PEX capacitor stamps on `ti12_full_300ns` into a single `memcpy`.
     - Detects identity `KLUmatrixNodeCollapsingNewToOld` mapping once per matrix to eliminate `RHS -> Intermediate -> RHS` scatter/gather copies in `SMPsolve`.
  3. **Fast power-of-2 `nghash` + zero-copy subcircuit parameter transfer** (`hash.h`, `hash.c`, `spicenum.c`, `xpressn.c`):
     - Power-of-2 mask sizing, cached 32-bit FNV-1a hash pre-filter, in-place node relinking on resize, and swap-merge subcircuit symbol dictionaries.
  4. **Unsigned `size_t` KLU `GET_POINTER`/`GET_X_POINTER`** (`klu_version.h`) & **`ft_numparse()` FP-to-int overflow UB fix** (`numparse.c`).

### Stage 12: Multirate & Latency-Partitioned Transient Integration + Opt-In Time-Axis Control (`NGSPICE_OPT_GENERAL` + `NGSPICE_OPT_ZEN4`)
- **Literature basis:**
  - **Compound-Fast Multirate Integration** (Gear & Wells, *BIT* 1984; Verhoeven et al., *IEEE TCAD* 2008; Gunther & Rentrop)
  - **Dynamic Latency Exploitation & Nodal Linear-Capacitor Compression** (Rabbat, Sangiovanni-Vincentelli & Hsieh, *IEEE TCAS* 1979)
  - **Time-Axis Predictor / PI+Deadband Step Controller / Adaptive Gear-3** (Gustafsson 1991; Soderlind H211PI *ACM TOMS* 2003; SUNDIALS IDA Hindmarsh et al. 2005)
- **Gain — Mixed-Signal Benchmarks (single-core KLU-1t, `cpu0`, `0` env vars default ON vs Stage 11 `9795d5b03`):**
  - `ti12_full_300ns` (`N=7,220`, 100k PEX caps): **`29.56 s -> 13.17 s` tran** (**2.24×** `std`, `rel_rms = 4.52e-6` PASS); with `nostepsizelimit`: **`8.96 s` tran** (**1.73×** vs `nostep` S11 / **3.30×** vs `std` S11, `2.69e-4` PASS).
  - `sar12_probe_2cyc`: **`7.01 s -> 6.37 s`** (**1.10×**, PASS `141.1 ps`); `sar12_hero_16cyc`: **`55.00 s -> 50.47 s`** (**1.090×**, **16/16** exact SAR codes, `143.7 ps`).
  - `pll_ring_150m_4us`: **`21.22 s -> 19.34 s`** (**1.10×**, PASS `c31`, `vc_err = 0.26 mV`).
  - `sdm16_120cyc`: **`22.49 s -> 22.47 s`** (**1.00×**, **0/115** bit-exact PDM, `r = 0.99932` PASS).
  - **4-Deck `std` Geomean:** **`1.284×`** over Stage 11 (**5/5 strict golden PASS** incl. Hero SAR).
  - **Opt-In Time-Axis Synergy (`NGSPICE_TA_*`, default `0`):** `TA_P123` (`PRED=2 CTRL=1 BKPT=1`) compounds to **`1.338×` `std` / `1.314×` `nostep` geomean** (`ti12` **`12.34 s` [2.39×]** `std`, **`8.01 s` [3.69×]** `nostep`; `sdm16` **`19.93 s` [1.13×]**); `TA_P234` (`CTRL=1 BKPT=1 ORD3=1`, native Gear-3) reaches **`7.63 s` (`3.88×` vs `std` S11, PASS `2.44e-4`)** on `ti12 nostep`.
- **Gain — VLSI Scaling on GF180 Post-APR 32-bit RISC-V SoC (`picorv32_gf180_soc`, `N=51,670` eqs, `58,320` BSIM4 MOSFETs, `21,955` R + `73,964` C [49.6k coupling], `2.68M` total KLU NNZ, single-core KLU-1t):**
  - Exact `120 ns` (3 cycles, `cpu8`, `0` env vars): **`942.88 s` Stock KLU-1t / `397.69 s` Stage 11 -> `243.30 s` tran** (**3.88×** vs Stock KLU-1t, **1.63×** vs Stage 11; KLU `factor` **`569.87 s / 329.15 s -> 176.12 s`** [**3.24× / 1.87×**], `62.3%` micro-steps, `avg_nF = 8,796 [17.0%]`, `fb = 0`, **3/3 cycles bit-exact PASS**, `9.50 mV` RMS vs Stock golden).
  - Full `260 ns` (6.5 cycles, PC `0x00 -> 0x04`, `cpu2`): Stage 11 **`802.68 s` -> `499.72 s` tran** default (**1.61×** vs Stage 11, **~4.00×** vs Stock KLU-1t; `factor` `660.97 s -> 360.14 s` [**1.84×**], `65.2%` micro-steps, **7/7 cycles bit-exact PASS**, `4.75 mV` RMS, `I_vdd = 3.629 mA` vs `3.63 mA`); with opt-in `TA_P123`: **`449.32 s` tran** (**1.79×** vs Stage 11, **~4.45×** vs Stock KLU-1t; `factor` `319.76 s` [**2.07×**], **7/7 cycles PASS**).
- **Major changes (all 3 build profiles verified clean with 0 warnings, 0 `%zmm` on `ZEN4=0`, `tests/bsim4` 2/2 PASS, LSan 0 new leaks):**
  1. **Deferred Nodal Linear-CAP Integration & Active-Node `CKTtrunc` (Default ON)** (`capdefs.h`, `capload.c`, `captrunc.c`, `capask.c`, `capinit.c`, `ckttrunc.c`, `NGSPICE_OPT_GENERAL`; opt-out `NGSPICE_NO_MR_DEFCAP=1`, `NGSPICE_NO_MR_TRUNC=1`):
     - Compresses `O(N_cap)` linear capacitor integration into `O(N_node)` nodal capacitance history (`CAPmrVhist[4]` / `CAPmrQnode[3]`, supporting BDF-1, Trap-2, Gear-2, and Gear-3) computed once per converged step in `CAPaccept` and materialized in `CAPkluPrep`, plus active-node `CAPtrunc` and `BSIM4truncAvx` stationary-block skipping.
  2. **Dynamic Latent-Partition BSIM4 Companion Freezing (Default ON)** (`bsim4_avx512_core.h`, `bsim4_avx512_glue.h`, `b4v5ld.c`, `b4v6ld.c`, `klu.h`, `klusmp.c`, `cktload.c`, `NGSPICE_OPT_ZEN4` + `NGSPICE_OPT_GENERAL`; opt-out `NGSPICE_NO_MR_LATENT=1`):
     - Freezes stationary 8-wide BSIM4 blocks into pre-accumulated `(KLUmatrixAxLatentDC, KLUmatrixAxLatentCap, KLUmatrixRhsLatentDC/Cap)` arrays with exact node-indexed wake-up (`0.5 * vtol`), state resynchronization (`s0..s4`), and `KLUmatrixResCapValid` coherency.
  3. **2-Level Compound-Fast Active-Submatrix KLU Multirate Integration (Auto-gated at `N >= 2,500`, Default ON)** (`dctran.c`, `klusmp.c`, `niiter.c`, `cktload.c`, `niconv.c`, `NGSPICE_OPT_GENERAL`; opt-out `NGSPICE_NO_MR_COMPFAST=1`, override `NGSPICE_MR_COMPFAST=0|1`, threshold `NGSPICE_MR_CF_MIN_N`):
     - Dynamically partitions active vs. slow nodes into a cached `nF × nF` KLU submatrix (`16`-entry LRU superset cache with union hysteresis, pre-baked `SubBaseAx`/`BndBaseAx`/`RailBaseAx`, full-row grounded DC rail KCL for exact `i(Vvdd)`, Diode terminal pinning, and zero-lag pre-accept boundary wake-and-redo).
     - At VLSI scale (`N >= 10,000`, strictly zero change for `N < 10,000`): uses directional strong-coupling 1-hop expansion (`|A_{row,j}| >= 0.02 * |A_{row,row}|`), exact first-order sensitivity-weighted wake (`WakeScale[i] = max_{j in S} |A_{j,i}|/|A_{j,j}|`), `vntol`-scaled thresholds, and VLSI-adaptive exact Chord Newton (`niiter.c` + `SMPkluSubSolve`).
  4. **Opt-In Time-Axis Predictor, PI+Deadband+NL Controller, Breakpoint Smoothing, Gear-3 & `CKTnodeIsVolt` Leak Fix (Default OFF)** (`cktdefs.h`, `cktdest.c`, `dctran.c`, `niiter.c`, `dioload.c`, `b4v5ld.c`, `b4v6ld.c`, `bsim4_avx512_core.h`, `bsim4_avx512_glue.h`, `capload.c`, `captrunc.c`, `NGSPICE_OPT_GENERAL`):
     - Opt-in toggles `NGSPICE_TA_PRED=0..3` (slope-limited 2nd-order Lagrange predictor, fast-only during `COMPFAST` bursts), `NGSPICE_TA_CTRL=1` (`TA_PI` Soderlind H211PI + `TA_DB=1.35` `ag0`-hold deadband + `TA_NL` Newton-aware damping + `TA_CHORD`), `NGSPICE_TA_BKPT=1`, and `NGSPICE_TA_ORD3=1` (native Gear-3 across AVX-512 BSIM4 and deferred KLU CAP); fixes unconditional `CKTnodeIsVolt` free in `cktdest.c`.

### Stage G (Apple Silicon): Zen 4 Stages 11–12 Convergence & Per-Instance NEON Multirate Port (`170df01bf`..`2cdc02f45`, `NGSPICE_OPT_GENERAL` + `NGSPICE_OPT_APPLE`)
- **Scope:**
  - **Stages 11–12 Merge (`170df01bf`):** Merged `origin/opt/zen4-transient@c9300bfb9` cleanly into `opt/apple-silicon-transient`, unifying all 16 portable Stage 11/12 files under `NGSPICE_OPT_GENERAL` (including exact row-oriented CSR backward solve in `spSolve`, deferred nodal `CAP` integration, active-node `CKTtrunc`, KLU latent accumulators, and 2-level `COMPFAST` active-submatrix multirate), combining Apple NEON full-scan with Stage 12 `sub_only` in `NIconvTest` (`niconv.c`), and preserving all Apple Silicon invariants (strict IEEE-754 exact division in `klu_solve.c` under `&& !NGSPICE_OPT_APPLE`, Clang 21 `ft_numparse()` UB fix, `CKTdiagGmin == 0.0` `MODETRANOP` guard, macOS `st_mtimespec` & `cntvct_el0` timer).
  - **Apple Silicon BSIM4 NEON Port (`18f241351`):** Ported all 4 Stage 12 BSIM4 optimizations into `bsim4_apple_core.h` and `bsim4_apple_glue.h` at **per-instance (`W=1`)** granularity:
    1. Per-instance dynamic latent-partition companion freezing (`MR_LATENT`, default ON) + `COMPFAST` active-submatrix filtering (`CKTmrSubActive`), keeping `BSIM4AppleCheck` at exact 128 bytes (`last_active_step` in `pad0`, `_Static_assert`), partitioning `grpActiveList` per `BSIM4AppleGroup` slice for O(1) freeze/wake while preserving 2-wide NEON batching, bucketing `cfFastList` by group on `cfEpoch` change via zero-allocation `pack->eval_q` scratch, exact CSR `nodeInstPtr`/`nodeInstIdx` wake-up, `stampFrozenDelta` (including internal `has_dint`/`has_sint` offsets via `ptrToKluOff`), and `s0..s4` state resync on wake/burst-exit.
    2. Active-node `BSIM4truncAvx` stationary skipping (`NGSPICE_NO_MR_TRUNC`, default ON) with dense 2-wide NEON `terrPairNeon` pairing of remaining active instances.
    3. Opt-in time-axis predictor (`CKTtaPredActive`) in `evalAppleGroupList` `MODEINITPRED`.
    4. Opt-in native Gear-3 (`NGSPICE_TA_ORD3=1`) across `isAvxEligibleTran`, `BSIM4ApplePack` step cache, `bsim4_apple_l2_assemble_one`, `bsim4_apple_eval_batch`, and 2-wide NEON `terrPairNeon`/`truncAvx`.
  - **L1-Resident Sparse DCOP Stamping Fix (`0e4339961`..`2cdc02f45`):** For L1-resident circuits (`nFb == 0 && nElig <= 64`, `<= 24 KB` working set), Sparse 1t `dc_only` stamps in exact `InstArray[0..nElig-1]` order (`pack->orig_order`) to match stock Sparse summation order on high-gain analog cells (`opa1k_pvt16` Sparse fixed to `2.57e-06` across all 16,000 tasks), while large circuits (`nElig > 64`) use single-pass inline grouped stamping (avoiding second-pass L1D spill and preserving 115/115 bit-exact PDM on `sdm16_120cyc` Sparse).
- **Verified Results — Apple M6 Super Core, Fresh PGO + ThinLTO (`2cdc02f45` vs Stage F `f54075591`, 12/12 Strict Golden PASS, 0 env vars, strict IEEE-754):**

| Workload | Solver | `f54075591` PGO | **`2cdc02f45` PGO** | `tran_s` | Speedup | Golden Result & Key Metric |
|---|---|---:|---:|---:|---:|---|
| `opa1k_pvt16` | **KLU** | 99.86 s | **103.45 s** | — | 0.965× | **PASS** `rel_rms = 5.31e-05 (ac_pk)`, 16k tasks |
| `opa1k_pvt16` | Sparse | 99.05 s *(FAIL)* | **101.10 s** | — | 0.980× | **PASS** `rel_rms = 2.57e-06 (ac_dc)`, 16k tasks *(fixed)* |
| `sdm16_120cyc` | **KLU** | 19.94 s | **19.52 s** | 19.23 s | **1.022×** | **PASS** 115/115 bit-exact PDM, `6.33e-03`, `r = 0.99932` |
| `sdm16_120cyc` | Sparse | 26.18 s | **27.05 s** | 26.68 s | 0.968× | **PASS** 115/115 bit-exact PDM, `6.77e-03`, `r = 0.99932` |
| `pll_ring_150m_4us` | **KLU** | 21.57 s | **17.69 s** | 17.59 s | **1.219×** | **PASS** `pll = 31/31`, `vc_err = 0.952 mV`, `8.76e-03` |
| `pll_ring_150m_4us` | Sparse | 23.54 s | **23.70 s** | 23.60 s | 0.993× | **PASS** `pll = 31/31`, `vc_err = 0.139 mV`, `5.89e-04` |
| `ti12_full_300ns` | **KLU** | 14.97 s | **8.93 s** | 8.14 s | **1.677×** | **PASS** `rel_rms = 6.96e-06 (v(dout_0))` |
| `ti12_full_300ns` | Sparse | 27.47 s | **30.13 s** | 29.17 s | 0.912× | **PASS** `rel_rms = 3.12e-11 (v(dout_6))` |
| `sar12_probe_2cyc` | **KLU** | 6.89 s | **5.66 s** | 5.36 s | **1.219×** | **PASS** 2/2 codes, `eoc_dt = 119.6 ps`, `2.48e-02` |
| `sar12_probe_2cyc` | Sparse | 8.62 s | **10.47 s** | 8.60 s | 0.823× | **PASS** 2/2 codes, `eoc_dt = 27.7 ps`, `1.66e-02` |
| `sar12_hero_16cyc` | **KLU** | 53.74 s | **43.14 s** | 42.83 s | **1.246×** | **PASS** 16/16 codes, `eoc_dt = 119.6 ps`, `1.50e-02` |
| `sar12_hero_16cyc` | Sparse | 68.84 s | **69.68 s** | 67.79 s | 0.988× | **PASS** 16/16 codes, `eoc_dt = 75.8 ps`, `8.27e-03` |

- **Highlights:** **1.260×** geomean across the 5 KLU transient workloads (**1.205×** across all 6 KLU workloads), **12/12 `all_ok=True`** (upgraded from 11/12 on `f54075591`), `tests/bsim4` **2/2 PASS** on ThinLTO and PGO+ThinLTO, `-DNGSPICE_OPT_APPLE=0 -fsyntax-only` **12/12 PASS**, and opt-in `NGSPICE_TA_*` verified PASS (`ti12` KLU `TA_P123+ORD3`: `8.99 s` ThinLTO, `7.65e-05`; `sar12_probe_2cyc` KLU `TA_P123`: `5.70 s` ThinLTO, `17.5 ps`, 2/2 codes; NEON Gear-3 matches pure scalar Gear-3 identically).
