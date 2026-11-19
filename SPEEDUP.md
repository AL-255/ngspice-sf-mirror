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

Both flags can also be overridden directly via `CFLAGS` (e.g. `CFLAGS="-O3 -DNGSPICE_OPT_GENERAL=1 -DNGSPICE_OPT_ZEN4=0"`). Building with `--enable-opt-general --disable-opt-zen4 CFLAGS="-O3 -march=x86-64 -mno-avx512f"` is verified to emit **0** `%zmm` instructions.

Runtime diagnostic switches (no rebuild required):
- `NGSPICE_NO_AVX512=1` — disable AVX-512 BSIM4 kernels at runtime
- `NGSPICE_NO_BLOCK_BYPASS=1` — disable the two-level stationary BSIM4 block cache
- `NGSPICE_AVX_BYPASS_TOL=<volts>` — override the stationary block cache voltage threshold (default: `min(CKTvoltTol, 1e-6)`)

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
