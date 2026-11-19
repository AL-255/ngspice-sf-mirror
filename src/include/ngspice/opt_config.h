/*
 * Compile-time feature flags separating portable general optimizations from
 * AMD Zen 4 / AVX-512 specific SIMD optimizations for easier cross-platform porting.
 *
 * Flags:
 *   NGSPICE_OPT_GENERAL (0 or 1, default 1):
 *       Portable C99 / OpenMP algorithmic & data-structure optimizations
 *       (no ISA-specific intrinsics; portable to x86-64, ARM64, RISC-V, etc.):
 *       - O(1) INPgetModBin hash index & zero-malloc scan_net_tok
 *       - Explicit PDK precompiler (--precompile / precompile / .uselib)
 *       - Unified KLU TLS parallel load, pre-baked RES, flat CAP & fast CAPtrunc
 *       - KLU refactor lazy scatter map, branchless scaling, alias-free 4x unroll
 *       - Sparse slab-linear spClear & spFactor Mult != 0 zero-skip
 *       - CKTterr precomputed weights, NIconvTest CKTnodeIsVolt cache, OldCKTstate0 guard
 *       - BSIM4v5/v6 scalar fast-path, direct 1t stamping, OMP 1t guard
 *
 *   NGSPICE_OPT_ZEN4 (0 or 1, default 1 iff __AVX512F__ && __GNUC__, else 0):
 *       AMD Zen 4 / AVX-512 specific SIMD kernels & hardware intrinsics:
 *       - 8-wide AVX-512 SoA BSIM4v5 & BSIM4v6 evaluator
 *       - Inline Zen 4 vrndscalepd/vscalefpd/vgetexppd/vgetmantpd exp & log
 *       - 8-wide AVX-512 voltage limiting (fetlim/limvds/pnjlim), convTest, trunc
 *       - Two-level stationary-voltage BSIM4 block cache with Taylor charge sync
 *
 * Control via ./configure:
 *   --enable-opt-general / --disable-opt-general
 *   --enable-opt-zen4    / --disable-opt-zen4
 * or override at compile time via CFLAGS:
 *   CFLAGS="-DNGSPICE_OPT_GENERAL=1 -DNGSPICE_OPT_ZEN4=0"
 */
#ifndef NGSPICE_OPT_CONFIG_H
#define NGSPICE_OPT_CONFIG_H

#ifndef NGSPICE_OPT_GENERAL
#  ifdef CONFIG_OPT_GENERAL
#    define NGSPICE_OPT_GENERAL CONFIG_OPT_GENERAL
#  else
#    define NGSPICE_OPT_GENERAL 1
#  endif
#endif

#ifndef NGSPICE_OPT_ZEN4
#  ifdef CONFIG_OPT_ZEN4
#    define NGSPICE_OPT_ZEN4 CONFIG_OPT_ZEN4
#  elif defined(__AVX512F__) && defined(__GNUC__)
#    define NGSPICE_OPT_ZEN4 1
#  else
#    define NGSPICE_OPT_ZEN4 0
#  endif
#endif

/* Safety: Zen 4 AVX-512 kernels require AVX-512F compiler support */
#if NGSPICE_OPT_ZEN4 && !(defined(__AVX512F__) && defined(__GNUC__))
#  undef NGSPICE_OPT_ZEN4
#  define NGSPICE_OPT_ZEN4 0
#endif

#ifndef NGSPICE_OPT_APPLE
#  ifdef CONFIG_OPT_APPLE
#    define NGSPICE_OPT_APPLE CONFIG_OPT_APPLE
#  elif defined(__aarch64__) && defined(__ARM_NEON)
#    define NGSPICE_OPT_APPLE 1
#  else
#    define NGSPICE_OPT_APPLE 0
#  endif
#endif

/* Safety: Apple Silicon / ARM64 NEON kernels require AArch64 NEON support */
#if NGSPICE_OPT_APPLE && !(defined(__aarch64__) && defined(__ARM_NEON))
#  undef NGSPICE_OPT_APPLE
#  define NGSPICE_OPT_APPLE 0
#endif

#define NGSPICE_OPT_BSIM4_SIMD (NGSPICE_OPT_ZEN4 || NGSPICE_OPT_APPLE)

#endif /* NGSPICE_OPT_CONFIG_H */
