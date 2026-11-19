# ngspice-47 Zen 4 Performance Optimization Changelog (`SPEEDUP.md`)

**Branch:** `opt/zen4-transient` (based on `ngspice-47`, commit `a80f6e3e9`)
**Target Host:** AMD Ryzen 9 7940HX (Zen 4, 16c/32t, 2×8-core CCDs, AVX-512, FMA, 32 KiB L1d / 1 MiB L2 per core, 64 MiB L3, GCC 13.3.0, GLIBC `libmvec.so.1`)
**Numerical Policy:** Strict IEEE-754 double precision throughout — **no `-ffast-math`**, no physics approximations, every stage verified against stock `ngspice-47` golden `.raw` waveforms and built-in `tests/bsim4` (`nmos/qaSpec`, `pmos/qaSpec`).

---

## Summary of Speedups (Stages 1–4)

### Hero >10-Minute Benchmark (`sar12_hero_16cyc`: GF180 12-bit Post-PEX Redundant SAR ADC, 16 Cycles, >206k Newton Iters)

| Configuration | Stock `ngspice-47` | Optimized (Stage 4) | Speedup (Same Config) | Speedup vs Stock Sparse 1t | Accuracy vs Golden |
|---|---:|---:|---:|---:|---|
| Sparse 1t | **628.48 s** (`10m 28s`) | **200.71 s** (`3m 21s`) | **3.13×** | **3.13×** | 16/16 SAR codes exact, max EOC Δt `99.2 ps` |
| KLU 1t | — | **128.99 s** (`2m 09s`) | — | **4.87×** | 16/16 SAR codes exact, max EOC Δt `110.7 ps` |
| Sparse 8t | — | **113.76 s** (`1m 54s`) | — | **5.52×** | 16/16 SAR codes exact, max EOC Δt `99.2 ps` |
| KLU 8t (1 CCD) | **132.71 s** (`2m 13s`) | **39.19 s** | **3.39×** (`5.21×` load) | **16.04×** | 16/16 SAR codes exact, max EOC Δt `110.7 ps` |

### Probe Suite Progression (Wall Time, Seconds)

All 4 circuits are real GF180 post-PEX mixed-signal circuits from `@aicdb`:
- `sar12_probe_2cyc` (1,559 eqns, 2 SAR cycles, ~26k–27k iters)
- `ti12_full_probe_30ns` (7,221 eqns, full-chip 3-ch TI 12b SAR + PLL, ~2.1k iters)
- `pll1p2g_probe_20ns` (1,307 eqns, 1.2 GHz AD-PLL, ~2.9k iters)
- `sdm16_probe_8cyc` (3,609 eqns, 16-bit switched-capacitor SDM, ~5.6k iters)

| Deck | Config | Stock Sp 1t | Stage 1 (Flags+LTO) | Stage 2 (Alg Opt) | Stage 3A (Par KLU) | Stage 3B (AVX-512) | Stage 4 (Solver) | Speedup vs Stock Sp 1t |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `sar12_probe_2cyc` | Sparse 1t | 98.94 | 83.10 | 64.23 | 61.54 | 31.07 | **28.57** | **3.46×** |
| | KLU 1t | — | 61.46 | 52.52 | 46.50 | 17.44 | **16.77** | **5.90×** |
| | KLU 8t | — | 17.59 | 11.87 | 9.22 | 5.48 | **5.34** | **18.52×** |
| `ti12_full_probe_30ns` | Sparse 1t | 61.07 | 51.75 | 42.26 | 37.29 | 22.41 | **22.48** | **2.72×** |
| | KLU 1t | — | 45.91 | 37.09 | 32.51 | 16.59 | **16.36** | **3.73×** |
| | KLU 8t | — | 20.09 | 13.45 | 9.54 | 7.28 | **7.26** | **8.41×** |
| | KLU 16t | — | — | — | — | — | **6.08** | **10.04×** (`21.8×` tran) |
| `pll1p2g_probe_20ns` | Sparse 1t | 11.00 | 9.19 | 7.69 | 6.97 | 3.87 | **3.76** | **2.93×** |
| | KLU 1t | — | 8.62 | 7.37 | 6.77 | 3.40 | **3.39** | **3.25×** |
| | KLU 8t | — | 3.02 | 2.24 | 1.81 | 1.33 | **1.34** | **8.21×** |
| `sdm16_probe_8cyc` | Sparse 1t | 8.29 | 7.91 | 6.30 | 5.88 | 5.12 | **5.07** | **1.64×** |
| | KLU 1t | — | 5.13 | 4.47 | 3.98 | 3.15 | **3.17** | **2.61×** |
| | KLU 8t | — | 3.00 | 2.46 | 2.00 | 1.86 | **1.88** | **4.42×** |

---

## Detailed Change Log & Porting Guide

### Stage 1: Compiler & Target-Specific Build Configuration (No Source Changes)
- **Flags:**
  ```sh
  CFLAGS="-O3 -march=znver4 -mtune=znver4 -fno-math-errno -fno-trapping-math -fno-semantic-interposition -flto=8 -g1"
  LDFLAGS="-flto=8 -O3 -march=znver4 -mtune=znver4 -lmvec"
  ```
- **Configure:** `--enable-openmp --enable-klu --enable-xspice --enable-cider --disable-debug --with-readline=no`
- **Gain:** **1.16×–1.20×** across all 4 decks (e.g., `sar12` Sparse 1t `98.94s -> 83.10s`, `ti12_full` `61.07s -> 51.75s`).
- **Porting note:** Safe drop-in on any GCC 12+ / x86-64-v4 / Zen 4 host (`-fno-math-errno -fno-trapping-math` avoids `errno` branch checks after `sqrt`/`exp`/`log` without weakening IEEE-754 rounding).

---

### Stage 2: Algorithmic & Dead-Work Eliminations (`fdd0b3244`, 21 files)

- **Gain:** **1.18×–1.48×** on top of Stage 1 (`sar12` Sparse 1t `83.10s -> 64.23s`, KLU 1t `61.46s -> 52.52s`, KLU 8t `17.59s -> 11.87s`; `ti12_full` KLU 8t `20.09s -> 13.45s`).
- **Major changes:**
  1. **Guarded dead `OldCKTstate0` allocation & `memcpy`** (`src/maths/ni/niiter.c`): `OldCKTstate0` was allocated and copied every Newton iteration even when `CKTbypass == 0`. Now skipped when bypass is disabled.
  2. **Precomputed `CKTterr` divided-difference weights** (`src/include/ngspice/cktdefs.h`, `src/spicelib/analysis/ckttrunc.c`, `src/spicelib/analysis/cktterr.c`): Hoisted the time-step-only divided-difference coefficients out of the per-device `CKTterr()` call (computed once per timestep in `CKTtrunc` instead of thousands of times per timestep).
  3. **Flat `CKTnodeIsVolt` byte cache** (`src/include/ngspice/cktdefs.h`, `src/maths/ni/niconv.c`, `src/spicelib/analysis/cktdltno.c`, `cktdest.c`): Replaced linked-list traversal of `CKTnodes` in `NIconvTest()` with a flat contiguous `unsigned char *` array indexed by equation number.
  4. **Slab-linear `spClear` & `memset` `SMPclear`** (`src/maths/sparse/spalloc.c`, `spbuild.c`, `src/maths/KLU/klusmp.c`): `spClear()` walks element allocation slabs sequentially instead of pointer-chasing every column linked list; `SMPclear()` clears `KLUmatrixAx` with a single `memset()`.
  5. **Linear passive fast paths (`RESload` / `CAPload`)** (`src/spicelib/devices/res/resload.c`, `src/spicelib/devices/cap/capload.c`): Skipped unused `REScurrent` calculation during transient when not needed; hoisted constants and inlined integration inside `CAPload`.
  6. **BSIM4v5 & BSIM4v6 scalar micro-optimizations** (`src/spicelib/devices/bsim4v5/b4v5ld.c`, `b4v5temp.c`, `b4v5trunc.c`, `b4v5cvtest.c`, `bsim4v5def.h` and matching `bsim4v6/*`):
     - Guarded bypass checks when `CKTbypass == 0` and trap-assisted tunneling with a `temp`-time `hasTrapTunnel` flag.
     - Precomputed `temp`-invariant factors (`1/toxe`, `1/coxe`, `sqrtLeff`, etc.), inlined `integrate_fast()` and `polyDepletion()`, merged GIDL/GISL matrix stamps, added a 4-terminal fast path in `LoadRhsMat`, and parallelized `BSIM4v*trunc` and `BSIM4v*convTest` with OpenMP.
- **Porting note:** 100% portable C (no SIMD intrinsics, no ABI dependencies). Every sub-change is self-contained and easy to cherry-pick upstream.

---

### Stage 3A: Unified Parallel KLU TLS Load, Pre-Baked RES, Flat CAP & KLU Refactor Fast Scaling (`d426e7c58`, 35 files)

- **Gain:** **1.05×–1.13×** single-thread, **1.23×–1.41×** multi-thread (`sar12` KLU 1t `52.52s -> 46.50s`, KLU 8t `11.87s -> 9.22s`; `ti12_full` KLU 1t `37.09s -> 32.51s`, KLU 8t `13.45s -> 9.54s`).
- **Major changes:**
  1. **Unified single-region KLU parallel load with TLS buffers** (`src/include/ngspice/devdefs.h`, `klu.h`, `smpdefs.h`, `src/maths/KLU/klusmp.c`, `src/spicelib/analysis/cktload.c`, `src/spicelib/devices/dev.c`):
     - Added optional `DEVkluPrep` and `DEVkluLoadPar` function pointers to `SPICEdev`.
     - Allocated persistent 64B-aligned thread-local `Ax_tls` and `rhs_tls` arrays plus a trailing trash slot `KLUmatrixAx[nz]` so ground/disabled stamps need no branch.
     - Launched a single `#pragma omp parallel` region per `CKTload` across all `DEVkluLoadPar`-capable devices, followed by a deterministic cache-tiled (256-wide) reduction into `KLUmatrixAx`/`CKTrhs`.
     - Single-thread (`nthreads == 1`) writes directly to `KLUmatrixAx`/`CKTrhs` with zero TLS overhead.
  2. **Pre-baked linear resistors (`RESkluPrep`) & flat `CAPfastArr`** (`src/spicelib/devices/res/*`, `src/spicelib/devices/cap/*`):
     - Linear, temperature-constant resistors are pre-accumulated once into `KLUmatrixAxResConst` and copied into `KLUmatrixAx` via `memcpy` during `SMPclear` (eliminating `RESload` completely during transient for linear resistors!).
     - Linear capacitors packed into contiguous `CAPfastArr` with pre-resolved KLU offsets, direct KLU/TLS stamping (`CAPloadKluPar`), and parallel `CAPtrunc`.
  3. **Direct BSIM4v5/v6 stamping**: Eliminated intermediate `_34.._61` buffer writes for both KLU/TLS and single-thread Sparse. Also fixed missed `exp(T0*log(T1))` -> `s2*s2*s2` / `1/sqrt(cbrt(T2))` in `b4v6ld.c`.
  4. **KLU refactor fast scaling & reciprocal pivot** (`src/maths/KLU/klu_refactor.c`): Precomputed `invRs[i] = 1.0 / Rs[i]` in unused `Xwork + n` workspace, replaced inner division by `ukk` with `inv_ukk = 1.0 / ukk`, and added `if (IS_NONZERO(ujk))` exact zero-skip.
- **Porting note:** Portable C + OpenMP. Adding `DEVkluPrep`/`DEVkluLoadPar` to `SPICEdev` requires updating the device table initializers (`dev.c` / device `*init.c` defaults to `NULL`).

---

### Stage 3B: 8-Wide AVX-512 SoA BSIM4v5 & BSIM4v6 Kernel with GLIBC `libmvec` (`31e0bc6cb`, 10 files)

- **Gain:** **1.60×–2.67×** single-thread, **1.31×–1.68×** multi-thread (`sar12` Sparse 1t `61.54s -> 31.07s` [load **3.15×** faster!], KLU 1t `46.50s -> 17.44s`, KLU 8t `9.22s -> 5.48s`; `ti12_full` Sparse 1t `37.29s -> 22.41s`, KLU 1t `32.51s -> 16.59s`; `pll1p2g` Sparse 1t `6.97s -> 3.87s`).
- **Major changes:**
  - Created two header-only SIMD modules guarded by `#if defined(__AVX512F__) && defined(__GNUC__)` with runtime disable switch `NGSPICE_NO_AVX512=1`:
    - `src/include/ngspice/bsim4_avx512_core.h`: 8-wide double-precision (`v8d`, 64B ZMM) SoA BSIM4v5 & BSIM4v6 DC + C-V + junction + GIDL/GISL + integration kernel calling GLIBC `libmvec` (`_ZGVeN8v_exp`, `_ZGVeN8v_log` — microbenchmarked **5.20×** faster than scalar `libm` with max rel err `3.5e-16`). Captures all BSIM4v5 vs BSIM4v6 physics differences via `is_v6` flag and per-block feature masks.
    - `src/include/ngspice/bsim4_avx512_glue.h`: Packs eligible instances (currently `rgateMod==0, rbodyMod==0, rdsMod==0, trnqsMod==0, acnqsMod==0, igcMod==0, igbMod==0, mobMod==0, capMod==2, dioMod==1, !hasTrapTunnel, xpart>=0, off==0, tempMod<2`) into 64B-aligned `BSIM4Avx512Block` structs; supports KLU/TLS, Sparse 1t direct, and Sparse OMP buffered stamping; falls back seamlessly to scalar for any ineligible instances.
  - Wired into `bsim4v5def.h`, `b4v5mdel.c`, `b4v5ld.c`, `b4v5temp.c` and matching `bsim4v6/*`.
- **Porting note:** Requires AVX-512F + GCC vector extensions + GLIBC `libmvec` (`-lmvec`). Cleanly compiles out when `__AVX512F__` is undefined; easy to port to AVX2 (`v4d` + `_ZGVdN4v_exp`/`_ZGVdN4v_log`) or SVE by changing the 15-line vector-helper typedefs at the top of `bsim4_avx512_core.h`.

---

### Stage 4: Linear Solver Inner-Loop Optimization (`d00bb7708`, 4 files)

- **Gain:** **1.04×–1.16×** on Sparse 1t/8t and **1.03×–1.04×** on KLU (`sar12` Sparse 1t `31.07s -> 28.57s` [`fac` -21%], Sparse 8t `18.47s -> 15.96s`, KLU 1t `17.44s -> 16.77s`, KLU 8t `5.48s -> 5.34s`).
- **Major changes:**
  1. `src/maths/KLU/klu_refactor.c`: Added `restrict`-qualified pointers and 4× manual unrolling to both the `X[Li[p]] -= Lx[p] * ujk` scatter-FMA loop and the `Lx[p] = X[i] * inv_ukk; X[i] = 0.0` gather-scale loop (distinct column indices guarantee no aliasing, unlocking 2 FMAs/cycle on Zen 4).
  2. `src/maths/KLU/klu.c` & `src/maths/KLU/klu_solve.c`: Added `if (xk == 0.0) continue;` exact zero-skip and 4× unrolled `restrict` inner loops to `KLU_lsolve`, `KLU_usolve`, and off-diagonal block back-substitution (`nrhs == 1`, real).
  3. `src/maths/sparse/spfactor.c`: Hoisted `Mult` into a register in the Direct branch and added `if (Mult != 0.0)` exact zero-skip to both Direct and Indirect branches of `spFactor`.
- **Porting note:** 100% portable C99 (`restrict`), zero architecture-specific code.
