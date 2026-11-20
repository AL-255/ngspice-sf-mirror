/*
 * Shared 8-wide AVX-512 SoA evaluator for BSIM4v5 & BSIM4v6 standard 4-terminal
 * transient MOSFETs (rgateMod=0, rbodyMod=0, rdsMod=0, trnqsMod=0, acnqsMod=0,
 * igcMod=0, igbMod=0, mobMod=0, capMod=2, dioMod=1, mtrlMod=0,
 * cvchargeMod=0, tempMod<2, xpart>=0, off=0; supports hasTrapTunnel).
 *
 * Uses exact IEEE-754 double-precision AVX-512 with inline Zen 4 hardware
 * vscalefpd / vrndscalepd / vgetexppd / vgetmantpd exp & log (<= 2 ULP,
 * max_rel <= 4.3e-16) to avoid external call ZMM register spilling.
 */
#ifndef NGSPICE_BSIM4_AVX512_CORE_H
#define NGSPICE_BSIM4_AVX512_CORE_H

#if NGSPICE_OPT_ZEN4

#include <immintrin.h>
#include <stdint.h>
#include <math.h>

typedef double v8d __attribute__((vector_size(64), aligned(64)));

#define B4V_MAX_EXP        5.834617425e14
#define B4V_MIN_EXP        1.713908431e-15
#define B4V_EXP_THRESHOLD  34.0
#define B4V_LOG1P_MIN_EXP  1.713908431e-15
#define B4V_EPSSI          1.03594e-10
#define B4V_CHARGE_Q       1.60219e-19
#define B4V_DELTA_1        0.02
#define B4V_DELTA_3        0.02
#define B4V_DELTA_4        0.02

static inline v8d v8_set1(double x) { return (v8d)_mm512_set1_pd(x); }
static inline v8d v8_load(const double *p) { return (v8d)_mm512_loadu_pd(p); }
static inline void v8_store(double *p, v8d v) { _mm512_storeu_pd(p, (__m512d)v); }
static inline v8d v8_sqrt(v8d x) { return (v8d)_mm512_sqrt_pd((__m512d)x); }
static inline v8d v8_abs(v8d x) { return (v8d)_mm512_abs_pd((__m512d)x); }
static inline v8d v8_min(v8d a, v8d b) { return (v8d)_mm512_min_pd((__m512d)a, (__m512d)b); }
static inline v8d v8_max(v8d a, v8d b) { return (v8d)_mm512_max_pd((__m512d)a, (__m512d)b); }

static inline __mmask8 v8_gt(v8d a, v8d b) { return _mm512_cmp_pd_mask((__m512d)a, (__m512d)b, _CMP_GT_OQ); }
static inline __mmask8 v8_ge(v8d a, v8d b) { return _mm512_cmp_pd_mask((__m512d)a, (__m512d)b, _CMP_GE_OQ); }
static inline __mmask8 v8_lt(v8d a, v8d b) { return _mm512_cmp_pd_mask((__m512d)a, (__m512d)b, _CMP_LT_OQ); }
static inline __mmask8 v8_le(v8d a, v8d b) { return _mm512_cmp_pd_mask((__m512d)a, (__m512d)b, _CMP_LE_OQ); }
static inline __mmask8 v8_eq(v8d a, v8d b) { return _mm512_cmp_pd_mask((__m512d)a, (__m512d)b, _CMP_EQ_OQ); }

/* Select t where m==1, f where m==0 */
static inline v8d v8_sel(__mmask8 m, v8d t, v8d f) {
    return (v8d)_mm512_mask_blend_pd(m, (__m512d)f, (__m512d)t);
}

/*
 * Inline IEEE-754 double AVX-512 exp(x) for |x| <= 700 (BSIM4 clamps to [-100, 100]).
 * Verified max_ulp = 2, max_rel = 4.30e-16 across [-100, +100].
 * Uses Zen 4 vrndscalepd + vscalefpd -> zero external calls, zero ZMM spills.
 */
static inline v8d v8_exp(v8d vx) {
    __m512d x = (__m512d)vx;
    const __m512d inv_ln2 = _mm512_set1_pd(1.44269504088896340736);
    const __m512d ln2_hi  = _mm512_set1_pd(6.93147180369123816490e-01);
    const __m512d ln2_lo  = _mm512_set1_pd(1.90821492927058770002e-10);
    __m512d n = _mm512_roundscale_pd(_mm512_mul_pd(x, inv_ln2),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512d r = _mm512_fnmadd_pd(n, ln2_hi, x);
    r = _mm512_fnmadd_pd(n, ln2_lo, r);

    __m512d r2 = _mm512_mul_pd(r, r);
    __m512d r4 = _mm512_mul_pd(r2, r2);
    __m512d r8 = _mm512_mul_pd(r4, r4);

    __m512d p01 = _mm512_add_pd(_mm512_set1_pd(1.0), r);
    __m512d p23 = _mm512_fmadd_pd(r, _mm512_set1_pd(1.0 / 6.0), _mm512_set1_pd(0.5));
    __m512d p45 = _mm512_fmadd_pd(r, _mm512_set1_pd(1.0 / 120.0), _mm512_set1_pd(1.0 / 24.0));
    __m512d p67 = _mm512_fmadd_pd(r, _mm512_set1_pd(1.0 / 5040.0), _mm512_set1_pd(1.0 / 720.0));
    __m512d p89 = _mm512_fmadd_pd(r, _mm512_set1_pd(1.0 / 362880.0), _mm512_set1_pd(1.0 / 40320.0));
    __m512d pAB = _mm512_fmadd_pd(r, _mm512_set1_pd(1.0 / 39916800.0), _mm512_set1_pd(1.0 / 3628800.0));
    __m512d pCD = _mm512_fmadd_pd(r, _mm512_set1_pd(1.0 / 6227020800.0), _mm512_set1_pd(1.0 / 479001600.0));

    __m512d q0 = _mm512_fmadd_pd(r2, p23, p01);
    __m512d q1 = _mm512_fmadd_pd(r2, p67, p45);
    __m512d q2 = _mm512_fmadd_pd(r2, pAB, p89);

    __m512d s0 = _mm512_fmadd_pd(r4, q1, q0);
    __m512d s1 = _mm512_fmadd_pd(r4, pCD, q2);
    __m512d poly = _mm512_fmadd_pd(r8, s1, s0);
    return (v8d)_mm512_scalef_pd(poly, n);
}

/*
 * Inline IEEE-754 double AVX-512 log(x) for positive normal doubles x > 0.
 * Verified max_ulp = 1, max_rel = 2.22e-16 across [e^-500, e^200].
 * Uses Zen 4 vgetexppd + vgetmantpd -> zero external calls, zero ZMM spills.
 */
static inline v8d v8_log(v8d vx) {
    __m512d x = (__m512d)vx;
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d ln2_hi = _mm512_set1_pd(6.93147180369123816490e-01);
    const __m512d ln2_lo = _mm512_set1_pd(1.90821492927058770002e-10);

    __m512d k = _mm512_getexp_pd(x);
    __m512d m = _mm512_getmant_pd(x, _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_zero);
    __mmask8 m_lt1 = _mm512_cmp_pd_mask(m, one, _CMP_LT_OQ);
    k = _mm512_mask_add_pd(k, m_lt1, k, one);

    __m512d f = _mm512_sub_pd(m, one);
    __m512d u = _mm512_div_pd(f, _mm512_add_pd(m, one));
    __m512d u2 = _mm512_mul_pd(u, u);
    __m512d u4 = _mm512_mul_pd(u2, u2);
    __m512d u8 = _mm512_mul_pd(u4, u4);

    __m512d p0 = _mm512_fmadd_pd(u2, _mm512_set1_pd(2.0 / 5.0), _mm512_set1_pd(2.0 / 3.0));
    __m512d p1 = _mm512_fmadd_pd(u2, _mm512_set1_pd(2.0 / 9.0), _mm512_set1_pd(2.0 / 7.0));
    __m512d p2 = _mm512_fmadd_pd(u2, _mm512_set1_pd(2.0 / 13.0), _mm512_set1_pd(2.0 / 11.0));
    __m512d p3 = _mm512_fmadd_pd(u2, _mm512_set1_pd(2.0 / 17.0), _mm512_set1_pd(2.0 / 15.0));
    __m512d p4 = _mm512_fmadd_pd(u2, _mm512_set1_pd(2.0 / 21.0), _mm512_set1_pd(2.0 / 19.0));

    __m512d q0 = _mm512_fmadd_pd(u4, p1, p0);
    __m512d q1 = _mm512_fmadd_pd(u4, p3, p2);
    __m512d poly = _mm512_fmadd_pd(u8, _mm512_fmadd_pd(u8, p4, q1), q0);

    __m512d u3 = _mm512_mul_pd(u, u2);
    __m512d logm_lo = _mm512_fnmadd_pd(f, u, _mm512_mul_pd(u3, poly));
    __m512d res = _mm512_fmadd_pd(k, ln2_lo, logm_lo);
    res = _mm512_add_pd(f, res);
    return (v8d)_mm512_fmadd_pd(k, ln2_hi, res);
}

static inline v8d v8_exp_clamped(v8d x, double lo, double hi) {
    return v8_exp(v8_min(v8_max(x, v8_set1(lo)), v8_set1(hi)));
}

/*
 * Inline IEEE-754 double scalar exp(x) & log(x) using Zen 4 AVX-512 scalar
 * hardware instructions (vroundscalesd/vscalefsd/vgetexpsd/vgetmantsd).
 * <= 2 ULP over normal BSIM4 range; falls back to libm for out-of-range/NaN.
 */
static inline double b4_s_exp(double x) {
    if (__builtin_expect(!(x >= -700.0 && x <= 700.0), 0))
        return (exp)(x);
    __m128d vx = _mm_set_sd(x);
    __m128d n = _mm_roundscale_sd(vx, _mm_mul_sd(vx, _mm_set_sd(1.44269504088896340736)),
                                  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    double nd = _mm_cvtsd_f64(n);
    double r = (x - nd * 6.93147180369123816490e-01) - nd * 1.90821492927058770002e-10;
    double r2 = r * r;
    double r4 = r2 * r2;
    double r8 = r4 * r4;
    double p01 = 1.0 + r;
    double p23 = 0.5 + r * (1.0 / 6.0);
    double p45 = (1.0 / 24.0) + r * (1.0 / 120.0);
    double p67 = (1.0 / 720.0) + r * (1.0 / 5040.0);
    double p89 = (1.0 / 40320.0) + r * (1.0 / 362880.0);
    double pAB = (1.0 / 3628800.0) + r * (1.0 / 39916800.0);
    double pCD = (1.0 / 479001600.0) + r * (1.0 / 6227020800.0);
    double q0 = p01 + r2 * p23;
    double q1 = p45 + r2 * p67;
    double q2 = p89 + r2 * pAB;
    double s0 = q0 + r4 * q1;
    double s1 = q2 + r4 * pCD;
    double poly = s0 + r8 * s1;
    return _mm_cvtsd_f64(_mm_scalef_sd(_mm_set_sd(poly), n));
}

static inline double b4_s_log(double x) {
    if (__builtin_expect(!(x >= 1.0e-300 && x <= 1.0e300), 0))
        return (log)(x);
    __m128d vx = _mm_set_sd(x);
    double k = _mm_cvtsd_f64(_mm_getexp_sd(vx, vx));
    double m = _mm_cvtsd_f64(_mm_getmant_sd(vx, vx, _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_zero));
    if (m < 1.0)
        k += 1.0;
    double u = (m - 1.0) / (m + 1.0);
    double u2 = u * u;
    double u4 = u2 * u2;
    double u8 = u4 * u4;
    double p0 = (2.0 / 3.0) + u2 * (2.0 / 5.0);
    double p1 = (2.0 / 7.0) + u2 * (2.0 / 9.0);
    double p2 = (2.0 / 11.0) + u2 * (2.0 / 13.0);
    double p3 = (2.0 / 15.0) + u2 * (2.0 / 17.0);
    double p4 = (2.0 / 19.0) + u2 * (2.0 / 21.0);
    double q0 = p0 + u4 * p1;
    double q1 = p2 + u4 * p3;
    double poly = q0 + u8 * (q1 + u8 * p4);
    double res = (k * 1.90821492927058770002e-10 + u * u2 * poly) + 2.0 * u;
    return k * 6.93147180369123816490e-01 + res;
}

static inline v8d v8_log_pos(v8d x) {
    return v8_log(v8_max(x, v8_set1(1.0e-300)));
}

/* 8-wide exact DEVlimvds (matches devsup.c:DEVlimvds) */
static inline v8d v8_limvds(v8d vnew, v8d vold) {
    __mmask8 m_ge35 = v8_ge(vold, v8_set1(3.5));
    __mmask8 m_inc  = v8_gt(vnew, vold);
    v8d hi_inc = v8_min(vnew, v8_set1(3.0) * vold + v8_set1(2.0));
    __mmask8 m_lt35 = v8_lt(vnew, v8_set1(3.5));
    v8d hi_dec = v8_sel(m_lt35, v8_max(vnew, v8_set1(2.0)), vnew);
    v8d hi_res = v8_sel(m_inc, hi_inc, hi_dec);

    v8d lo_inc = v8_min(vnew, v8_set1(4.0));
    v8d lo_dec = v8_max(vnew, v8_set1(-0.5));
    v8d lo_res = v8_sel(m_inc, lo_inc, lo_dec);
    return v8_sel(m_ge35, hi_res, lo_res);
}

/* 8-wide exact DEVfetlim (matches devsup.c:DEVfetlim) */
static inline v8d v8_fetlim(v8d vnew, v8d vold, v8d vto) {
    const v8d z0 = v8_set1(0.0);
    v8d diff_o = vold - vto;
    v8d vtsthi = v8_abs(v8_set1(2.0) * diff_o) + v8_set1(2.0);
    v8d vtstlo = v8_abs(diff_o) + v8_set1(1.0);
    v8d vtox   = vto + v8_set1(3.5);
    v8d delv   = vnew - vold;

    __mmask8 m_on  = v8_ge(vold, vto);
    __mmask8 m_hi  = v8_ge(vold, vtox);
    __mmask8 m_dec = v8_le(delv, z0);

    /* vold >= vtox */
    __mmask8 m_n_ge_x = v8_ge(vnew, vtox);
    v8d hi_dec_ge = v8_sel(v8_gt(-delv, vtstlo), vold - vtstlo, vnew);
    v8d hi_dec_lt = v8_max(vnew, vto + v8_set1(2.0));
    v8d hi_dec = v8_sel(m_n_ge_x, hi_dec_ge, hi_dec_lt);
    v8d hi_inc = v8_sel(v8_ge(delv, vtsthi), vold + vtsthi, vnew);
    v8d on_hi  = v8_sel(m_dec, hi_dec, hi_inc);

    /* vto <= vold < vtox (middle region) */
    v8d on_mid = v8_sel(m_dec, v8_max(vnew, vto - v8_set1(0.5)), v8_min(vnew, vto + v8_set1(4.0)));
    v8d on_res = v8_sel(m_hi, on_hi, on_mid);

    /* vold < vto (off) */
    v8d off_dec = v8_sel(v8_gt(-delv, vtsthi), vold - vtsthi, vnew);
    v8d vtemp = vto + v8_set1(0.5);
    v8d off_inc_le = v8_sel(v8_gt(delv, vtstlo), vold + vtstlo, vnew);
    v8d off_inc = v8_sel(v8_le(vnew, vtemp), off_inc_le, vtemp);
    v8d off_res = v8_sel(m_dec, off_dec, off_inc);

    return v8_sel(m_on, on_res, off_res);
}

/* 8-wide exact DEVpnjlim (matches devsup.c:DEVpnjlim); returns mask of limited lanes */
static inline v8d v8_pnjlim(v8d vnew, v8d vold, double vt_s, v8d vcrit, __mmask8 *limited_mask) {
    const v8d z0 = v8_set1(0.0);
    const v8d vt = v8_set1(vt_s);
    const v8d two = v8_set1(2.0);
    __mmask8 m_fwd = v8_gt(vnew, vcrit) & v8_gt(v8_abs(vnew - vold), v8_set1(vt_s + vt_s));
    v8d res = vnew;
    __mmask8 chk = m_fwd;

    if (m_fwd) {
        __mmask8 m_vo_pos = v8_gt(vold, z0);
        v8d arg = (vnew - vold) * v8_set1(1.0 / vt_s);
        __mmask8 m_arg_pos = v8_gt(arg, z0);
        /* Avoid non-positive log argument on inactive lanes */
        v8d larg_p = v8_max(arg - two, v8_set1(1.0e-300));
        v8d larg_n = v8_max(two - arg, v8_set1(1.0e-300));
        v8d v_pos = v8_sel(m_arg_pos,
                           vold + vt * (two + v8_log(larg_p)),
                           vold - vt * (two + v8_log(larg_n)));
        v8d v_neg = vt * v8_log(v8_max(vnew * v8_set1(1.0 / vt_s), v8_set1(1.0e-300)));
        v8d v_fwd = v8_sel(m_vo_pos, v_pos, v_neg);
        res = v8_sel(m_fwd, v_fwd, res);
    }

    __mmask8 m_rem_neg = (~m_fwd) & v8_lt(vnew, z0);
    if (m_rem_neg) {
        v8d arg = v8_sel(v8_gt(vold, z0), -vold - v8_set1(1.0), two * vold - v8_set1(1.0));
        __mmask8 m_clamp = m_rem_neg & v8_lt(vnew, arg);
        res = v8_sel(m_clamp, arg, res);
        chk |= m_clamp;
    }

    *limited_mask = chk;
    return res;
}

/* 8-wide BSIM4v5/v6 reverse-bias trap-assisted tunneling term (exact match to DEXP) */
static inline void
v8_trap_tunnel_term(__mmask8 mask, v8d vj, v8d vt, v8d invNvtm, v8d revSat,
                    v8d *g_io, v8d *c_io)
{
    const v8d one = v8_set1(1.0);
    const v8d exp_th = v8_set1(B4V_EXP_THRESHOLD);
    const v8d nexp_th = v8_set1(-B4V_EXP_THRESHOLD);
    const v8d max_exp = v8_set1(B4V_MAX_EXP);
    const v8d min_exp = v8_set1(B4V_MIN_EXP);

    v8d diff = vt - vj;
    __mmask8 m_small = v8_lt(diff, vt * v8_set1(1.0e-3));
    v8d T9_reg = one / v8_sel(m_small, one, diff);
    v8d T0_sm  = -vj * invNvtm * v8_set1(1.0e3);
    v8d T0_reg = -vj * invNvtm * vt * T9_reg;
    v8d T0     = v8_sel(m_small, T0_sm, T0_reg);

    __mmask8 m_hi  = v8_gt(T0, exp_th);
    __mmask8 m_lo  = v8_lt(T0, nexp_th);
    __mmask8 m_mid = (~m_hi) & (~m_lo) & mask;

    v8d B = v8_sel(m_hi, max_exp * (one + T0 - exp_th), min_exp);
    v8d C = v8_sel(m_hi, max_exp, v8_set1(0.0));
    if (m_mid) {
        v8d ex = v8_exp_clamped(T0, -B4V_EXP_THRESHOLD, B4V_EXP_THRESHOLD);
        B = v8_sel(m_mid, ex, B);
        C = v8_sel(m_mid, ex, C);
    }

    v8d dT0_sm  = invNvtm * v8_set1(1.0e3);
    v8d dT0_reg = vt * invNvtm * (T9_reg + vj * T9_reg * T9_reg);
    v8d dB_dVb  = C * v8_sel(m_small, dT0_sm, dT0_reg);

    *g_io = v8_sel(mask, *g_io + revSat * dB_dVb, *g_io);
    *c_io = v8_sel(mask, *c_io - revSat * (B - one), *c_io);
}

/* 8-wide poly gate depletion (exact match to BSIM4v5/v6 polyDepletion with mtrlMod=0) */
static inline void
v8_poly_depletion(v8d phi_vfb, v8d polyT1, __mmask8 m_ngate_valid,
                  v8d Vgs, v8d *Vgs_eff, v8d *dVgs_eff_dVg)
{
    __mmask8 m = m_ngate_valid & v8_gt(Vgs, phi_vfb);
    if (!m) {
        *Vgs_eff = Vgs;
        *dVgs_eff_dVg = v8_set1(1.0);
        return;
    }
    v8d one = v8_set1(1.0);
    v8d T8 = v8_max(Vgs - phi_vfb, v8_set1(0.0));
    v8d invT1 = one / polyT1;
    v8d T4 = v8_sqrt(one + v8_set1(2.0) * T8 * invT1);
    v8d T2 = v8_set1(2.0) * T8 / (T4 + one);
    v8d T3 = v8_set1(0.5) * T2 * T2 * invT1;
    v8d T7 = v8_set1(1.07) - T3; /* 1.12 - T3 - 0.05 */
    v8d T6 = v8_sqrt(T7 * T7 + v8_set1(0.224));
    v8d T5 = v8_set1(1.12) - v8_set1(0.5) * (T7 + T6);
    v8d eff = Vgs - T5;
    v8d deff = one - (v8_set1(0.5) - v8_set1(0.5) / T4) * (one + T7 / T6);
    *Vgs_eff = v8_sel(m, eff, Vgs);
    *dVgs_eff_dVg = v8_sel(m, deff, one);
}

/*
 * 64-byte aligned 8-wide SoA constant block for 8 BSIM4v5 or BSIM4v6 MOSFETs.
 */
typedef struct __attribute__((aligned(64))) {
    /* Core geometry / model / bias-independent terms */
    v8d type;              /* +1.0 (NMOS) or -1.0 (PMOS) */
    v8d m;
    v8d nf;
    v8d vth0_signed;       /* type * vth0 */
    v8d vfb_plus_phi;      /* vfb + phi */
    v8d vfbzb;
    v8d vbsc;
    v8d phi;
    v8d phi_095;           /* 0.95 * phi */
    v8d sqrtPhi;
    v8d V0;                /* vbi - phi */
    v8d vtm;
    v8d coxe;
    v8d invCoxe;
    v8d coxp;
    v8d toxe;
    v8d invToxe;
    v8d toxp;
    v8d tox_ctm;           /* 1e8 * toxp */
    v8d inv_tox_ctm;       /* 1.0 / (1e8 * toxp) */
    v8d inv_2e8_toxp;      /* 1.0 / (2e8 * toxp) */
    v8d link_ctm;          /* 1e-3 * toxp */
    v8d leff;
    v8d invLeff;
    v8d sqrtLeff;
    v8d weff;
    v8d weffCV;
    v8d leffCV;
    v8d CoxWL;             /* coxe * weffCV * leffCV * nf */
    v8d CoxWL_ov_coxe;     /* weffCV * leffCV * nf */
    v8d drainCond;
    v8d sourceCond;
    v8d mgd;               /* m * drainCond */
    v8d mgs;               /* m * sourceCond */
    v8d ados1p9;           /* 1.9e-9 * ados (1.9e-9 in v4.5) */
    v8d bdos0p7;           /* 0.7 * bdos (0.7 in v4.5) */

    /* Diode (dioMod == 1) & Trap-Assisted Tunneling */
    v8d invNvtms;
    v8d invNvtmd;
    v8d SourceSatCurrent;
    v8d DrainSatCurrent;
    v8d vjsmFwd;
    v8d vjdmFwd;
    v8d IVjsmFwd;
    v8d IVjdmFwd;
    v8d invNvtmrs_s, invNvtmrs_d;
    v8d invNvtmrssw_s, invNvtmrssw_d;
    v8d invNvtmrsswg_s, invNvtmrsswg_d;
    v8d vtss, vtsd, vtssws, vtsswd, vtsswgs, vtsswgd;
    v8d SjctRevSat, DjctRevSat, SswRevSat, DswRevSat, SswgRevSat, DswgRevSat;

    /* Vth / SCE / DIBL / n / Pocket */
    v8d xdep0OverSqrtPhi;
    v8d factor1;
    v8d dvt0;
    v8d dvt1_leff;         /* dvt1 * leff */
    v8d dvt2;
    v8d dvt0w;
    v8d dvt1w_weff_leff;   /* dvt1w * weff * leff */
    v8d dvt2w;
    v8d k1;
    v8d k1ox;
    v8d k2ox;
    v8d k3;
    v8d k3b;
    v8d lpebFactor;
    v8d vthNarrowW;
    v8d vth_const_T1;      /* k1ox*(lpe0Factor-1)*sqrtPhi + (kt1 + kt1l/leff)*TempRatio - k1*sqrtPhi*lpebFactor + type*vth0 + k3*vthNarrowW */
    v8d kt2_TempRatio;     /* kt2 * TempRatio */
    v8d eta0;
    v8d etab;
    v8d theta0vb0;
    v8d nfactor;
    v8d cit;
    v8d cdsc;
    v8d cdscb;
    v8d cdscd;
    v8d dvtp0;
    v8d dvtp1;

    /* Poly depletion / Vgsteff */
    v8d polyT1;            /* 1e6 * CHARGE * EPSSI * ngate / (coxe*coxe), 1.0 if invalid */
    v8d mstar;
    v8d voffcbn;
    v8d coxe_ov_cdep0;

    /* Weff / Rds / Abulk */
    v8d dwg;
    v8d dwb;
    v8d prwg;
    v8d prwb;
    v8d rds0_half;
    v8d rdswmin;
    v8d xj;
    v8d a0;
    v8d b0_ov_wb1;         /* b0 / (weff + b1) */
    v8d ags_a0;            /* ags * a0 */
    v8d keta;
    v8d abulkCVfactor;

    /* Mobility / Saturation / CLM / DIBL / DITS / SCBE / Isub / Fsevl */
    v8d ua;
    v8d ub;
    v8d uc;
    v8d ud;
    v8d u0temp;
    v8d vsattemp;
    v8d a1;
    v8d a2;
    v8d delta;
    v8d lambda;
    v8d litl;
    v8d fprout;
    v8d pvag;
    v8d pclm;
    v8d inv_pclm_litl;     /* 1.0 / (pclm * litl) when pclm > MIN_EXP */
    v8d thetaRout;
    v8d inv_thetaRout;
    v8d pdiblb;
    v8d pdits;
    v8d inv_pdits;
    v8d pditsd;
    v8d pditsl_term;       /* 1.0 + pditsl * leff */
    v8d pscbe1_litl;
    v8d inv_pscbe2;        /* 1.0 / pscbe2 when active */
    v8d alpha_ov_leff;     /* (alpha0 + alpha1*leff) / leff */
    v8d beta0;
    v8d inv_vtl_tfactor;   /* 1.0 / (vtl * tfactor) when active */
    v8d vtfbphi2;

    /* GIDL / GISL */
    v8d inv_3toxe;         /* 1.0 / (3.0 * toxe) */
    v8d agidl_weffCJ;
    v8d bgidl;
    v8d cgidl;
    v8d egidl;
    v8d agisl_weffCJ;
    v8d bgisl;
    v8d cgisl;
    v8d egisl;

    /* C-V (capMod == 2) & Overlap */
    v8d noff;
    v8d voffcv;
    v8d acde;
    v8d ldeb;
    v8d moin;
    v8d cgdo;
    v8d cgso;
    v8d cgbo;
    v8d weffCV_cgdl;
    v8d ckappad;
    v8d inv_ckappad;
    v8d weffCV_cgsl;
    v8d ckappas;
    v8d inv_ckappas;

    /* Junction C-V */
    v8d czbs;
    v8d czbssw;
    v8d czbsswg;
    v8d czbd;
    v8d czbdsw;
    v8d czbdswg;
    v8d PhiBS, invPhiBS;
    v8d PhiBSWS, invPhiBSWS;
    v8d PhiBSWGS, invPhiBSWGS;
    v8d PhiBD, invPhiBD;
    v8d PhiBSWD, invPhiBSWD;
    v8d PhiBSWGD, invPhiBSWGD;
    v8d MJS, inv1mMJS;
    v8d MJSWS, inv1mMJSWS;
    v8d MJSWGS, inv1mMJSWGS;
    v8d MJD, inv1mMJD;
    v8d MJSWD, inv1mMJSWD;
    v8d MJSWGD, inv1mMJSWGD;
    v8d jctS_sum;          /* czbs + czbssw + czbsswg */
    v8d jctD_sum;          /* czbd + czbdsw + czbdswg */
    v8d jctS_lin;          /* czbs*MJS/PhiBS + czbssw*MJSWS/PhiBSWS + czbsswg*MJSWGS/PhiBSWGS */
    v8d jctD_lin;          /* czbd*MJD/PhiBD + czbdsw*MJSWD/PhiBSWD + czbdswg*MJSWGD/PhiBSWGD */

    /* Per-lane bitmasks */
    __mmask8 m_ngate_valid;
    __mmask8 m_dvt0w;
    __mmask8 m_dvtp0;
    __mmask8 m_fprout;
    __mmask8 m_pclm;
    __mmask8 m_thetaRout;
    __mmask8 m_pdits;
    __mmask8 m_pscbe;
    __mmask8 m_isub;
    __mmask8 m_vtl;
    __mmask8 m_lambda;
    __mmask8 m_gidl_en;
    __mmask8 m_gisl_en;
    __mmask8 m_xpart_gt05;
    __mmask8 m_xpart_lt05;
    __mmask8 m_src_sat_pos;
    __mmask8 m_drn_sat_pos;
    __mmask8 m_czbs, m_czbssw, m_czbsswg;
    __mmask8 m_czbd, m_czbdsw, m_czbdswg;
    __mmask8 m_mjs_half, m_mjsws_half, m_mjswgs_half;
    __mmask8 m_mjd_half, m_mjswd_half, m_mjswgd_half;
    __mmask8 m_has_dint, m_has_sint;
    __mmask8 m_k1ox_zero, m_k1ox_le0;
    __mmask8 m_vfbzb_le0;
    __mmask8 m_nf_ne1;
    __mmask8 m_has_a1;
    __mmask8 m_tt_s, m_tt_d, m_tt_ssw, m_tt_dsw, m_tt_sswg, m_tt_dswg;

    /* Block-level summary flags */
    uint8_t n_active;      /* 1..8 */
    uint8_t is_v6;         /* 0 = BSIM4v5, 1 = BSIM4v6 */
    uint8_t has_ud;
    uint8_t xpart_mode;    /* 0=all <0.5 (40/60), 1=all >0.5 (0/100), 2=all ==0.5 (50/50), 3=mixed */
    uint8_t has_trap_tunnel;

    /* Per-lane scalar metadata for gather/scatter & state access */
    v8d vcrit_v;
    double vcrit[8];
    double type_s[8];
    int dNodePrime[8];
    int gNodePrime[8];
    int sNodePrime[8];
    int bNodePrime[8];
    int stateBase[8];
    int instIdx[8];        /* index into model->BSIM4v*InstanceArray */
    int kluOff[8][22];     /* KLU Ax offsets (valid when KLU mode active) */
} BSIM4Avx512Block;

/* Compact 64B-aligned per-block dynamic state, Level-1 stamp cache & Level-2 physics cache */
typedef struct __attribute__((aligned(64))) {
    /* Hot Level-1 / convTest / limiting fields (first cache lines) */
    v8d von;
    v8d last_vds, last_vgs, last_vbs;
    v8d Idtot, cd_g, cd_d, cd_b;
    v8d Ibtot, cb_g, cb_d, cb_b;

    /* Level-1 cached matrix/RHS stamps (already multiplied by m) */
    double st_rhsd[8], st_rhsg[8], st_rhsb[8], st_rhss[8];
    double st_v34[8], st_v35[8], st_v36[8], st_v37[8];
    double st_v44[8], st_v46[8], st_v47[8], st_v48[8];
    double st_v51[8], st_v52[8], st_v53[8], st_v55[8];
    double st_v58[8], st_v59[8], st_v60[8], st_v61[8];

    /* Cache validity metadata */
    double last_time;
    double last_ag0, last_ag1, last_ag2;
    double last_gmin;
    const double *last_s0;
    const double *last_s1;
    const double *last_s2;
    int last_int_mode;
    uint8_t phys_valid;
    uint8_t stamp_valid;

    /* Level-2 cached time-independent physics (DC + unscaled C + charges) */
    v8d Cggb, Cgdb, Cgsb;
    v8d Cdgb, Cddb, Cdsb;
    v8d Csgb, Csdb, Cssb;
    v8d Cbgb, Cbdb, Cbsb;
    v8d mdc44, mdc46, mdc47, mdc48;
    v8d mdc51, mdc52, mdc53, mdc55;
    v8d mdc58, mdc59, mdc60, mdc61;
    v8d mdc_rhsd, mdc_rhsb, mdc_rhss;
    v8d q_lin_g, q_lin_d, q_lin_b;
    v8d qb_st, qg_st, qd_st;
    double a_vbd[8], a_vbs[8], a_vgs[8], a_vds[8];
    double a_qb[8], a_qg[8], a_qd[8];
} BSIM4Avx512Dyn;

typedef struct __attribute__((aligned(64))) {
    /* Instance state fields needed only when need_aos != 0 (e.g., .save @m[...]) */
    double von[8];
    int    mode[8];
    double cd[8], gds[8], gm[8], gmbs[8];
    double gbd[8], gbs[8], cbs[8], cbd[8];
    double csub[8], gbds[8], gbgs[8], gbbs[8];
    double Igidl[8], ggidld[8], ggidlg[8], ggidlb[8];
    double Igisl[8], ggisls[8], ggislg[8], ggislb[8];
} BSIM4Avx512Out;

/*
 * Fast Level-1 state sync helper: when a block hits Level-1 with small nonzero dv,
 * update CKTstate0 voltages & first-order Taylor-corrected charges/currents so
 * the next timestep's s1/s2 and CKTtrunc stay accurate to O(dv^2), without
 * re-evaluating BSIM4 physics or matrix/RHS stamps (which are invariant to O(dv^2)).
 */
static inline void
bsim4_avx512_l1_sync_state(
    const BSIM4Avx512Block * restrict b,
    const BSIM4Avx512Dyn * restrict dyn,
    v8d vds, v8d vgs, v8d vbs,
    double ag0, double ag1, double ag2,
    int is_gear2, int is_trap2,
    double * restrict s0,
    const double * restrict s1,
    const double * restrict s2)
{
    v8d dvds = vds - dyn->last_vds;
    v8d dvgs = vgs - dyn->last_vgs;
    v8d dvbs = vbs - dyn->last_vbs;
    v8d dvgb = dvgs - dvbs;
    v8d dvbd = dvbs - dvds;

    v8d qg_new = dyn->qg_st + (dyn->Cggb * dvgb - dyn->Cgdb * dvbd - dyn->Cgsb * dvbs);
    v8d qd_new = dyn->qd_st + (dyn->Cdgb * dvgb - dyn->Cddb * dvbd - dyn->Cdsb * dvbs);
    v8d qb_new = dyn->qb_st + (dyn->Cbgb * dvgb - dyn->Cbdb * dvbd - dyn->Cbsb * dvbs);

    double lvds[8], lvgs[8], lvbs[8], lvbd[8];
    double lqb[8], lqg[8], lqd[8];
    v8_store(lvds, vds); v8_store(lvgs, vgs);
    v8_store(lvbs, vbs); v8_store(lvbd, vbs - vds);
    v8_store(lqb, qb_new); v8_store(lqg, qg_new); v8_store(lqd, qd_new);

    const int nact = b->n_active;
    for (int lane = 0; lane < nact; lane++) {
        int st = b->stateBase[lane];
        double qb = lqb[lane], qg = lqg[lane], qd = lqd[lane];
        s0[st + 0]  = lvbd[lane];
        s0[st + 1]  = lvbs[lane];
        s0[st + 2]  = lvgs[lane];
        s0[st + 3]  = lvds[lane];
        s0[st + 11] = qb;
        s0[st + 13] = qg;
        s0[st + 15] = qd;

        double cqb, cqg, cqd;
        if (is_gear2) {
            cqb = ag2 * s2[st + 11] + ag1 * s1[st + 11] + ag0 * qb;
            cqg = ag2 * s2[st + 13] + ag1 * s1[st + 13] + ag0 * qg;
            cqd = ag2 * s2[st + 15] + ag1 * s1[st + 15] + ag0 * qd;
        } else if (is_trap2) {
            cqb = -s1[st + 12] * ag1 + ag0 * (qb - s1[st + 11]);
            cqg = -s1[st + 14] * ag1 + ag0 * (qg - s1[st + 13]);
            cqd = -s1[st + 16] * ag1 + ag0 * (qd - s1[st + 15]);
        } else {
            cqb = ag0 * qb + ag1 * s1[st + 11];
            cqg = ag0 * qg + ag1 * s1[st + 13];
            cqd = ag0 * qd + ag1 * s1[st + 15];
        }
        s0[st + 12] = cqb;
        s0[st + 14] = cqg;
        s0[st + 16] = cqd;
    }
}

/*
 * Fast Level-2 helper: integrate cached charges (with optional first-order
 * Taylor correction dq = C * dv when has_dv != 0) into CKTstate0 and assemble
 * the 20 matrix/RHS stamps in dyn->st_* from cached DC + C matrices.
 * Used on BOTH full evaluations (has_dv=0) and Level-2 cross-step hits (has_dv=1).
 */
static inline void
bsim4_avx512_integrate_and_assemble(
    const BSIM4Avx512Block * restrict b,
    BSIM4Avx512Dyn * restrict dyn,
    v8d vds, v8d vgs, v8d vbs,
    int has_dv,
    double ag0, double ag1, double ag2,
    int is_gear2, int is_trap2,
    double * restrict s0,
    const double * restrict s1,
    const double * restrict s2)
{
    double a_cqb[8], a_cqg[8], a_cqd[8];
    const int nact = b->n_active;
    v8d dqg = v8_set1(0.0), dqd = v8_set1(0.0), dqb = v8_set1(0.0);

    if (has_dv) {
        v8d dvds = vds - dyn->last_vds;
        v8d dvgs = vgs - dyn->last_vgs;
        v8d dvbs = vbs - dyn->last_vbs;
        v8d dvgb = dvgs - dvbs;
        v8d dvbd = dvbs - dvds;
        dqg = dyn->Cggb * dvgb - dyn->Cgdb * dvbd - dyn->Cgsb * dvbs;
        dqd = dyn->Cdgb * dvgb - dyn->Cddb * dvbd - dyn->Cdsb * dvbs;
        dqb = dyn->Cbgb * dvgb - dyn->Cbdb * dvbd - dyn->Cbsb * dvbs;

        double lvds[8], lvgs[8], lvbs[8], lvbd[8];
        double lqb[8], lqg[8], lqd[8];
        v8_store(lvds, vds); v8_store(lvgs, vgs);
        v8_store(lvbs, vbs); v8_store(lvbd, vbs - vds);
        v8_store(lqb, dyn->qb_st + dqb);
        v8_store(lqg, dyn->qg_st + dqg);
        v8_store(lqd, dyn->qd_st + dqd);

        for (int lane = 0; lane < nact; lane++) {
            int st = b->stateBase[lane];
            double qb = lqb[lane], qg = lqg[lane], qd = lqd[lane];
            s0[st + 0]  = lvbd[lane];
            s0[st + 1]  = lvbs[lane];
            s0[st + 2]  = lvgs[lane];
            s0[st + 3]  = lvds[lane];
            s0[st + 11] = qb;
            s0[st + 13] = qg;
            s0[st + 15] = qd;

            double cqb, cqg, cqd;
            if (is_gear2) {
                cqb = ag2 * s2[st + 11] + ag1 * s1[st + 11] + ag0 * qb;
                cqg = ag2 * s2[st + 13] + ag1 * s1[st + 13] + ag0 * qg;
                cqd = ag2 * s2[st + 15] + ag1 * s1[st + 15] + ag0 * qd;
            } else if (is_trap2) {
                cqb = -s1[st + 12] * ag1 + ag0 * (qb - s1[st + 11]);
                cqg = -s1[st + 14] * ag1 + ag0 * (qg - s1[st + 13]);
                cqd = -s1[st + 16] * ag1 + ag0 * (qd - s1[st + 15]);
            } else {
                cqb = ag0 * qb + ag1 * s1[st + 11];
                cqg = ag0 * qg + ag1 * s1[st + 13];
                cqd = ag0 * qd + ag1 * s1[st + 15];
            }
            s0[st + 12] = a_cqb[lane] = cqb;
            s0[st + 14] = a_cqg[lane] = cqg;
            s0[st + 16] = a_cqd[lane] = cqd;
        }
    } else {
        for (int lane = 0; lane < nact; lane++) {
            int st = b->stateBase[lane];
            double qb = dyn->a_qb[lane];
            double qg = dyn->a_qg[lane];
            double qd = dyn->a_qd[lane];
            s0[st + 0]  = dyn->a_vbd[lane];
            s0[st + 1]  = dyn->a_vbs[lane];
            s0[st + 2]  = dyn->a_vgs[lane];
            s0[st + 3]  = dyn->a_vds[lane];
            s0[st + 11] = qb;
            s0[st + 13] = qg;
            s0[st + 15] = qd;

            double cqb, cqg, cqd;
            if (is_gear2) {
                cqb = ag2 * s2[st + 11] + ag1 * s1[st + 11] + ag0 * qb;
                cqg = ag2 * s2[st + 13] + ag1 * s1[st + 13] + ag0 * qg;
                cqd = ag2 * s2[st + 15] + ag1 * s1[st + 15] + ag0 * qd;
            } else if (is_trap2) {
                cqb = -s1[st + 12] * ag1 + ag0 * (qb - s1[st + 11]);
                cqg = -s1[st + 14] * ag1 + ag0 * (qg - s1[st + 13]);
                cqd = -s1[st + 16] * ag1 + ag0 * (qd - s1[st + 15]);
            } else {
                cqb = ag0 * qb + ag1 * s1[st + 11];
                cqg = ag0 * qg + ag1 * s1[st + 13];
                cqd = ag0 * qd + ag1 * s1[st + 15];
            }
            s0[st + 12] = a_cqb[lane] = cqb;
            s0[st + 14] = a_cqg[lane] = cqg;
            s0[st + 16] = a_cqd[lane] = cqd;
        }
    }
    for (int lane = nact; lane < 8; lane++) {
        a_cqb[lane] = a_cqg[lane] = a_cqd[lane] = 0.0;
    }

    v8d v_ag0 = v8_set1(ag0);
    v8d vm_ag0 = b->m * v_ag0;
    v8d cqbody = v8_load(a_cqb);
    v8d cqgate = v8_load(a_cqg);
    v8d cqdrn  = v8_load(a_cqd);

    v8d mtype = b->m * b->type;
    v8d mceqqg = (cqgate + v_ag0 * (dyn->q_lin_g - dqg)) * mtype;
    v8d mceqqd = (cqdrn  + v_ag0 * (dyn->q_lin_d - dqd)) * mtype;
    v8d mceqqb = (cqbody + v_ag0 * (dyn->q_lin_b - dqb)) * mtype;

    v8_store(dyn->st_rhsd, dyn->mdc_rhsd - mceqqd);
    v8_store(dyn->st_rhsg, mceqqg);
    v8_store(dyn->st_rhsb, dyn->mdc_rhsb - mceqqb);
    v8_store(dyn->st_rhss, dyn->mdc_rhss + mceqqg + mceqqb + mceqqd);

    v8d v34 = dyn->Cggb * vm_ag0;
    v8d v35 = dyn->Cgdb * vm_ag0;
    v8d v36 = dyn->Cgsb * vm_ag0;
    v8_store(dyn->st_v34, v34);
    v8_store(dyn->st_v35, v35);
    v8_store(dyn->st_v36, v36);
    v8_store(dyn->st_v37, -(v34 + v35 + v36));

    v8d mgcdgb = dyn->Cdgb * vm_ag0;
    v8d mgcddb = dyn->Cddb * vm_ag0;
    v8d mgcdsb = dyn->Cdsb * vm_ag0;
    v8_store(dyn->st_v44, dyn->mdc44 + mgcddb);
    v8_store(dyn->st_v46, dyn->mdc46 + mgcdgb);
    v8_store(dyn->st_v47, dyn->mdc47 - mgcdsb);
    v8_store(dyn->st_v48, dyn->mdc48 + (mgcdgb + mgcddb + mgcdsb));

    v8d mgcsgb = dyn->Csgb * vm_ag0;
    v8d mgcsdb = dyn->Csdb * vm_ag0;
    v8d mgcssb = dyn->Cssb * vm_ag0;
    v8_store(dyn->st_v51, dyn->mdc51 - mgcsdb);
    v8_store(dyn->st_v52, dyn->mdc52 + mgcsgb);
    v8_store(dyn->st_v53, dyn->mdc53 + mgcssb);
    v8_store(dyn->st_v55, dyn->mdc55 + (mgcsgb + mgcsdb + mgcssb));

    v8d mgcbgb = dyn->Cbgb * vm_ag0;
    v8d mgcbdb = dyn->Cbdb * vm_ag0;
    v8d mgcbsb = dyn->Cbsb * vm_ag0;
    v8_store(dyn->st_v58, dyn->mdc58 + mgcbdb);
    v8_store(dyn->st_v59, dyn->mdc59 + mgcbgb);
    v8_store(dyn->st_v60, dyn->mdc60 + mgcbsb);
    v8_store(dyn->st_v61, dyn->mdc61 - (mgcbgb + mgcbdb + mgcbsb));
}

/*
 * Evaluate 8 MOSFETs in parallel given prepared/limited terminal voltages
 * vds_in, vgs_in, vbs_in (already type-signed, matching scalar BSIM4v*LoadOMP).
 * Writes integrated charges/currents into CKTstate0 and fills `out`.
 */
static inline void
bsim4_avx512_eval_block(
    const BSIM4Avx512Block * restrict b,
    BSIM4Avx512Dyn * restrict dyn,
    v8d vds, v8d vgs, v8d vbs,
    double ckt_gmin,
    double ag0, double ag1, double ag2,
    int is_gear2, int is_trap2,
    double * restrict s0,
    const double * restrict s1,
    const double * restrict s2,
    BSIM4Avx512Out * restrict out,
    int need_aos,
    int dc_only)
{
    const v8d z0 = v8_set1(0.0);
    const v8d one = v8_set1(1.0);
    const v8d half = v8_set1(0.5);
    const v8d two = v8_set1(2.0);
    const v8d gmin = v8_set1(ckt_gmin);
    const v8d max_exp = v8_set1(B4V_MAX_EXP);
    const v8d min_exp = v8_set1(B4V_MIN_EXP);
    const v8d exp_th = v8_set1(B4V_EXP_THRESHOLD);
    const v8d nexp_th = v8_set1(-B4V_EXP_THRESHOLD);
    const v8d epssi = v8_set1(B4V_EPSSI);

    v8d vbd = vbs - vds;
    v8d vgd = vgs - vds;
    v8d vgb = vgs - vbs;

    dyn->last_vds = vds;
    dyn->last_vgs = vgs;
    dyn->last_vbs = vbs;

    /* ===================================================================== */
    /* 1. Source/Drain Junction Diodes (dioMod == 1)                         */
    /* ===================================================================== */
    v8d gbs, cbs, gbd, cbd;
    {
        v8d T2 = vbs * b->invNvtms;
        __mmask8 m_low = v8_lt(T2, nexp_th);
        __mmask8 m_mid = (~m_low) & v8_le(vbs, b->vjsmFwd) & b->m_src_sat_pos;
        v8d evbs = z0;
        if (m_mid)
            evbs = v8_exp_clamped(T2, -B4V_EXP_THRESHOLD, 100.0);
        v8d g_low = gmin;
        v8d c_low = b->SourceSatCurrent * (min_exp - one) + gmin * vbs;
        v8d g_mid = b->SourceSatCurrent * evbs * b->invNvtms + gmin;
        v8d c_mid = b->SourceSatCurrent * (evbs - one) + gmin * vbs;
        v8d T0h = b->IVjsmFwd * b->invNvtms;
        v8d g_hi = T0h + gmin;
        v8d c_hi = b->IVjsmFwd - b->SourceSatCurrent + T0h * (vbs - b->vjsmFwd) + gmin * vbs;
        gbs = v8_sel(m_low, g_low, v8_sel(m_mid, g_mid, g_hi));
        cbs = v8_sel(m_low, c_low, v8_sel(m_mid, c_mid, c_hi));
        gbs = v8_sel(b->m_src_sat_pos, gbs, gmin);
        cbs = v8_sel(b->m_src_sat_pos, cbs, gmin * vbs);
    }
    {
        v8d T2 = vbd * b->invNvtmd;
        __mmask8 m_low = v8_lt(T2, nexp_th);
        __mmask8 m_mid = (~m_low) & v8_le(vbd, b->vjdmFwd) & b->m_drn_sat_pos;
        v8d evbd = z0;
        if (m_mid)
            evbd = v8_exp_clamped(T2, -B4V_EXP_THRESHOLD, 100.0);
        v8d g_low = gmin;
        v8d c_low = b->DrainSatCurrent * (min_exp - one) + gmin * vbd;
        v8d g_mid = b->DrainSatCurrent * evbd * b->invNvtmd + gmin;
        v8d c_mid = b->DrainSatCurrent * (evbd - one) + gmin * vbd;
        v8d T0h = b->IVjdmFwd * b->invNvtmd;
        v8d g_hi = T0h + gmin;
        v8d c_hi = b->IVjdmFwd - b->DrainSatCurrent + T0h * (vbd - b->vjdmFwd) + gmin * vbd;
        gbd = v8_sel(m_low, g_low, v8_sel(m_mid, g_mid, g_hi));
        cbd = v8_sel(m_low, c_low, v8_sel(m_mid, c_mid, c_hi));
        gbd = v8_sel(b->m_drn_sat_pos, gbd, gmin);
        cbd = v8_sel(b->m_drn_sat_pos, cbd, gmin * vbd);
    }
    if (b->has_trap_tunnel) {
        if (b->m_tt_s)
            v8_trap_tunnel_term(b->m_tt_s, vbs, b->vtss, b->invNvtmrs_s, b->SjctRevSat, &gbs, &cbs);
        if (b->m_tt_d)
            v8_trap_tunnel_term(b->m_tt_d, vbd, b->vtsd, b->invNvtmrs_d, b->DjctRevSat, &gbd, &cbd);
        if (b->m_tt_ssw)
            v8_trap_tunnel_term(b->m_tt_ssw, vbs, b->vtssws, b->invNvtmrssw_s, b->SswRevSat, &gbs, &cbs);
        if (b->m_tt_dsw)
            v8_trap_tunnel_term(b->m_tt_dsw, vbd, b->vtsswd, b->invNvtmrssw_d, b->DswRevSat, &gbd, &cbd);
        if (b->m_tt_sswg)
            v8_trap_tunnel_term(b->m_tt_sswg, vbs, b->vtsswgs, b->invNvtmrsswg_s, b->SswgRevSat, &gbs, &cbs);
        if (b->m_tt_dswg)
            v8_trap_tunnel_term(b->m_tt_dswg, vbd, b->vtsswgd, b->invNvtmrsswg_d, b->DswgRevSat, &gbd, &cbd);
    }

    /* ===================================================================== */
    /* 2. Forward / Reverse mode swap                                        */
    /* ===================================================================== */
    __mmask8 m_fwd = v8_ge(vds, z0);
    v8d Vds = v8_sel(m_fwd, vds, -vds);
    v8d Vgs = v8_sel(m_fwd, vgs, vgd);
    v8d Vbs = v8_sel(m_fwd, vbs, vbd);

    /* ===================================================================== */
    /* 3. Vbseff, Phis, Xdep, Vth                                            */
    /* ===================================================================== */
    v8d Vbseff, dVbseff_dVb;
    {
        v8d T0 = Vbs - b->vbsc - v8_set1(0.001);
        v8d T1 = v8_sqrt(T0 * T0 - v8_set1(0.004) * b->vbsc);
        __mmask8 m_ge0 = v8_ge(T0, z0);
        v8d v_pos = b->vbsc + half * (T0 + T1);
        v8d dv_pos = half * (one + T0 / T1);
        v8d T2 = v8_set1(-0.002) / (T1 - T0);
        v8d v_neg = b->vbsc * (one + T2);
        v8d dv_neg = T2 * b->vbsc / T1;
        Vbseff = v8_sel(m_ge0, v_pos, v_neg);
        dVbseff_dVb = v8_sel(m_ge0, dv_pos, dv_neg);

        v8d T9 = b->phi_095;
        T0 = T9 - Vbseff - v8_set1(0.001);
        T1 = v8_sqrt(T0 * T0 + v8_set1(0.004) * T9);
        Vbseff = T9 - half * (T0 + T1);
        dVbseff_dVb = dVbseff_dVb * (half * (one + T0 / T1));
    }

    v8d Phis = b->phi - Vbseff;
    v8d sqrtPhis = v8_sqrt(Phis);
    v8d dsqrtPhis_dVb = v8_set1(-0.5) / sqrtPhis;
    v8d Xdep = b->xdep0OverSqrtPhi * sqrtPhis;
    v8d dXdep_dVb = b->xdep0OverSqrtPhi * dsqrtPhis_dVb;

    v8d sqrtXdep = v8_sqrt(Xdep);
    v8d invSqrtXdep = one / sqrtXdep;

    /* lt1 */
    v8d lt1, dlt1_dVb;
    {
        v8d T0 = b->dvt2 * Vbseff;
        __mmask8 m_ge = v8_ge(T0, v8_set1(-0.5));
        v8d T4 = one / (v8_set1(3.0) + v8_set1(8.0) * T0);
        v8d T1 = v8_sel(m_ge, one + T0, (one + v8_set1(3.0) * T0) * T4);
        v8d T2 = v8_sel(m_ge, b->dvt2, b->dvt2 * T4 * T4);
        lt1 = b->factor1 * sqrtXdep * T1;
        dlt1_dVb = b->factor1 * (half * invSqrtXdep * T1 * dXdep_dVb + sqrtXdep * T2);
    }

    /* Theta0 */
    v8d Theta0, dTheta0_dVb;
    {
        v8d T0 = b->dvt1_leff / lt1;
        __mmask8 m_lt = v8_lt(T0, exp_th);
        v8d th_hi = v8_set1(1.0 / (B4V_MAX_EXP - 2.0));
        if (m_lt) {
            v8d T1 = v8_exp_clamped(T0, -100.0, B4V_EXP_THRESHOLD);
            v8d T2 = T1 - one;
            v8d T4 = T2 * T2 + two * T1 * min_exp;
            v8d invT4 = one / T4;
            v8d th = T1 * invT4;
            v8d dT1 = -T0 * T1 * dlt1_dVb / lt1;
            v8d dth = dT1 * (T4 - two * T1 * (T2 + min_exp)) * (invT4 * invT4);
            Theta0 = v8_sel(m_lt, th, th_hi);
            dTheta0_dVb = v8_sel(m_lt, dth, z0);
        } else {
            Theta0 = th_hi;
            dTheta0_dVb = z0;
        }
    }
    v8d Delt_vth = b->dvt0 * Theta0 * b->V0;
    v8d dDelt_vth_dVb = b->dvt0 * dTheta0_dVb * b->V0;

    /* dvt0w narrow-width SCE (skip completely if all lanes have dvt0w == 0) */
    v8d T2_dvt0w = z0, dT2_dvt0w_dVb = z0;
    if (b->m_dvt0w) {
        v8d T0w = b->dvt2w * Vbseff;
        __mmask8 m_ge = v8_ge(T0w, v8_set1(-0.5));
        v8d T4w = one / (v8_set1(3.0) + v8_set1(8.0) * T0w);
        v8d T1w = v8_sel(m_ge, one + T0w, (one + v8_set1(3.0) * T0w) * T4w);
        v8d T2w = v8_sel(m_ge, b->dvt2w, b->dvt2w * T4w * T4w);
        v8d ltw = b->factor1 * sqrtXdep * T1w;
        v8d dltw_dVb = b->factor1 * (half * invSqrtXdep * T1w * dXdep_dVb + sqrtXdep * T2w);
        v8d T0 = b->dvt1w_weff_leff / ltw;
        __mmask8 m_lt = v8_lt(T0, exp_th) & b->m_dvt0w;
        v8d t5_hi = v8_set1(1.0 / (B4V_MAX_EXP - 2.0));
        v8d T5 = t5_hi, dT5_dVb = z0;
        if (m_lt) {
            v8d T1 = v8_exp_clamped(T0, -100.0, B4V_EXP_THRESHOLD);
            v8d T2 = T1 - one;
            v8d T4 = T2 * T2 + two * T1 * min_exp;
            v8d invT4 = one / T4;
            v8d dT1 = -T0 * T1 * dltw_dVb / ltw;
            T5 = v8_sel(m_lt, T1 * invT4, t5_hi);
            dT5_dVb = v8_sel(m_lt, dT1 * (T4 - two * T1 * (T2 + min_exp)) * (invT4 * invT4), z0);
        }
        T2_dvt0w = v8_sel(b->m_dvt0w, b->dvt0w * T5 * b->V0, z0);
        dT2_dvt0w_dVb = v8_sel(b->m_dvt0w, b->dvt0w * dT5_dVb * b->V0, z0);
    }

    /* DIBL shift */
    v8d dDIBL_Sft_dVd, DIBL_Sft, T4_dibl;
    {
        v8d T3 = b->eta0 + b->etab * Vbseff;
        __mmask8 m_lt = v8_lt(T3, v8_set1(1.0e-4));
        v8d T9 = one / (v8_set1(3.0) - v8_set1(2.0e4) * T3);
        T3 = v8_sel(m_lt, (v8_set1(2.0e-4) - T3) * T9, T3);
        T4_dibl = v8_sel(m_lt, T9 * T9, one);
        dDIBL_Sft_dVd = T3 * b->theta0vb0;
        DIBL_Sft = dDIBL_Sft_dVd * Vds;
    }

    v8d Vth = b->vth_const_T1 + b->k1ox * sqrtPhis * b->lpebFactor
            - b->k2ox * Vbseff - Delt_vth - T2_dvt0w
            + b->k3b * Vbseff * b->vthNarrowW
            + b->kt2_TempRatio * Vbseff - DIBL_Sft;
    v8d dVth_dVb = b->lpebFactor * b->k1ox * dsqrtPhis_dVb - b->k2ox
                 - dDelt_vth_dVb - dT2_dvt0w_dVb + b->k3b * b->vthNarrowW
                 - b->etab * Vds * b->theta0vb0 * T4_dibl + b->kt2_TempRatio;
    v8d dVth_dVd = -dDIBL_Sft_dVd;

    /* Subthreshold swing n */
    v8d n, dn_dVb, dn_dVd;
    {
        v8d tmp1 = epssi / Xdep;
        v8d tmp2 = b->nfactor * tmp1;
        v8d tmp3 = b->cdsc + b->cdscb * Vbseff + b->cdscd * Vds;
        v8d tmp4 = (tmp2 + tmp3 * Theta0 + b->cit) * b->invCoxe;
        v8d base_dVb = (-tmp2 / Xdep * dXdep_dVb + tmp3 * dTheta0_dVb + b->cdscb * Theta0) * b->invCoxe;
        v8d base_dVd = b->cdscd * Theta0 * b->invCoxe;
        __mmask8 m_ge = v8_ge(tmp4, v8_set1(-0.5));
        v8d T0 = one / (v8_set1(3.0) + v8_set1(8.0) * tmp4);
        v8d T0sq = T0 * T0;
        n = v8_sel(m_ge, one + tmp4, (one + v8_set1(3.0) * tmp4) * T0);
        dn_dVb = v8_sel(m_ge, base_dVb, base_dVb * T0sq);
        dn_dVd = v8_sel(m_ge, base_dVd, base_dVd * T0sq);
    }

    /* Pocket implant Vth correction (dvtp0 > 0) */
    if (b->m_dvtp0) {
        v8d T0 = -b->dvtp1 * Vds;
        __mmask8 m_ge = v8_ge(T0, nexp_th) & b->m_dvtp0;
        v8d T2 = min_exp, dT2_dVd = z0;
        if (m_ge) {
            v8d ex = v8_exp_clamped(T0, -B4V_EXP_THRESHOLD, 100.0);
            T2 = v8_sel(m_ge, ex, min_exp);
            dT2_dVd = v8_sel(m_ge, -b->dvtp1 * ex, z0);
        }
        v8d T3 = b->leff + b->dvtp0 * (one + T2);
        v8d dT3_dVd = b->dvtp0 * dT2_dVd;
        v8d T4 = b->vtm * v8_log_pos(b->leff / T3);
        v8d dT4_dVd = -b->vtm * dT3_dVd / T3;
        v8d dDITS_dVd = dn_dVd * T4 + n * dT4_dVd;
        v8d dDITS_dVb = T4 * dn_dVb;
        Vth      = v8_sel(b->m_dvtp0, Vth - n * T4, Vth);
        dVth_dVd = v8_sel(b->m_dvtp0, dVth_dVd - dDITS_dVd, dVth_dVd);
        dVth_dVb = v8_sel(b->m_dvtp0, dVth_dVb - dDITS_dVb, dVth_dVb);
    }
    dyn->von = Vth;
    if (need_aos || dc_only)
        v8_store(out->von, Vth);

    /* ===================================================================== */
    /* 4. Poly Depletion & Vgsteff                                           */
    /* ===================================================================== */
    v8d vgs_eff, dvgs_eff_dvg, vgd_eff, dvgd_eff_dvg;
    v8_poly_depletion(b->vfb_plus_phi, b->polyT1, b->m_ngate_valid, vgs, &vgs_eff, &dvgs_eff_dvg);
    v8_poly_depletion(b->vfb_plus_phi, b->polyT1, b->m_ngate_valid, vgd, &vgd_eff, &dvgd_eff_dvg);
    v8d Vgs_eff = v8_sel(m_fwd, vgs_eff, vgd_eff);
    v8d dVgs_eff_dVg = v8_sel(m_fwd, dvgs_eff_dvg, dvgd_eff_dvg);

    v8d Vgst = Vgs_eff - Vth;

    v8d Vgsteff, dVgsteff_dVg, dVgsteff_dVd, dVgsteff_dVb;
    {
        v8d T0 = n * b->vtm;
        v8d T1 = b->mstar * Vgst;
        v8d T2 = T1 / T0;
        __mmask8 m_hi = v8_gt(T2, exp_th);
        __mmask8 m_lo = v8_lt(T2, nexp_th);
        __mmask8 m_mid = (~m_hi) & (~m_lo);

        v8d T10_hi = T1;
        v8d dT10g_hi = b->mstar * dVgs_eff_dVg;
        v8d dT10d_hi = -dVth_dVd * b->mstar;
        v8d dT10b_hi = -dVth_dVb * b->mstar;

        v8d t10_base = b->vtm * v8_set1(B4V_LOG1P_MIN_EXP);
        v8d T10_lo = t10_base * n;
        v8d dT10d_lo = t10_base * dn_dVd;
        v8d dT10b_lo = t10_base * dn_dVb;

        v8d T10 = v8_sel(m_hi, T10_hi, T10_lo);
        v8d dT10g = v8_sel(m_hi, dT10g_hi, z0);
        v8d dT10d = v8_sel(m_hi, dT10d_hi, dT10d_lo);
        v8d dT10b = v8_sel(m_hi, dT10b_hi, dT10b_lo);
        if (m_mid) {
            v8d ex = v8_exp_clamped(T2, -B4V_EXP_THRESHOLD, B4V_EXP_THRESHOLD);
            v8d T3 = b->vtm * v8_log_pos(one + ex);
            v8d dg0 = b->mstar * ex / (one + ex);
            v8d inv_n = one / n;
            T10   = v8_sel(m_mid, n * T3, T10);
            dT10b = v8_sel(m_mid, T3 * dn_dVb - dg0 * (dVth_dVb + Vgst * dn_dVb * inv_n), dT10b);
            dT10d = v8_sel(m_mid, T3 * dn_dVd - dg0 * (dVth_dVd + Vgst * dn_dVd * inv_n), dT10d);
            dT10g = v8_sel(m_mid, dg0 * dVgs_eff_dVg, dT10g);
        }

        v8d T1b = b->voffcbn - (one - b->mstar) * Vgst;
        v8d T2b = T1b / T0;
        __mmask8 m2_lo = v8_lt(T2b, nexp_th);
        __mmask8 m2_hi = v8_gt(T2b, exp_th);
        __mmask8 m2_mid = (~m2_lo) & (~m2_hi);

        v8d T3_cl = b->coxe_ov_cdep0 * v8_sel(m2_lo, min_exp, max_exp);
        v8d T9 = b->mstar + T3_cl * n;
        v8d dT9g = z0;
        v8d dT9d = dn_dVd * T3_cl;
        v8d dT9b = dn_dVb * T3_cl;
        if (m2_mid) {
            v8d ex = v8_exp_clamped(T2b, -B4V_EXP_THRESHOLD, B4V_EXP_THRESHOLD);
            v8d T4 = b->coxe_ov_cdep0 * ex;
            v8d T5 = T1b * T4 / T0;
            v8d dg0 = b->coxe_ov_cdep0 * (b->mstar - one) * ex / b->vtm;
            T9   = v8_sel(m2_mid, b->mstar + n * T4, T9);
            dT9b = v8_sel(m2_mid, T4 * dn_dVb - dg0 * dVth_dVb - T5 * dn_dVb, dT9b);
            dT9d = v8_sel(m2_mid, T4 * dn_dVd - dg0 * dVth_dVd - T5 * dn_dVd, dT9d);
            dT9g = v8_sel(m2_mid, dg0 * dVgs_eff_dVg, dT9g);
        }

        v8d invT9 = one / T9;
        v8d invT9sq = invT9 * invT9;
        Vgsteff      = T10 * invT9;
        dVgsteff_dVg = (T9 * dT10g - T10 * dT9g) * invT9sq;
        dVgsteff_dVd = (T9 * dT10d - T10 * dT9d) * invT9sq;
        dVgsteff_dVb = (T9 * dT10b - T10 * dT9b) * invT9sq;
    }

    /* ===================================================================== */
    /* 5. Weff, Rds (rdsMod == 0), Abulk                                     */
    /* ===================================================================== */
    v8d T9_phi = sqrtPhis - b->sqrtPhi;
    v8d Weff = b->weff - two * (b->dwg * Vgsteff + b->dwb * T9_phi);
    v8d dWeff_dVg = -two * b->dwg;
    v8d dWeff_dVb = -two * b->dwb * dsqrtPhis_dVb;
    {
        __mmask8 m_w = v8_lt(Weff, v8_set1(2.0e-8));
        if (m_w) {
            v8d T0 = one / (v8_set1(6.0e-8) - two * Weff);
            v8d T0s = T0 * T0 * v8_set1(4.0e-16);
            Weff = v8_sel(m_w, v8_set1(2.0e-8) * (v8_set1(4.0e-8) - Weff) * T0, Weff);
            dWeff_dVg = v8_sel(m_w, dWeff_dVg * T0s, dWeff_dVg);
            dWeff_dVb = v8_sel(m_w, dWeff_dVb * T0s, dWeff_dVb);
        }
    }

    v8d Rds, dRds_dVg, dRds_dVb;
    {
        v8d T0 = one + b->prwg * Vgsteff;
        v8d invT0 = one / T0;
        v8d dT0_dVg = -b->prwg * invT0 * invT0;
        v8d T1 = b->prwb * T9_phi;
        v8d dT1_dVb = b->prwb * dsqrtPhis_dVb;
        v8d T2 = invT0 + T1;
        v8d T3 = T2 + v8_sqrt(T2 * T2 + v8_set1(0.01));
        v8d dT3_base = one + T2 / (T3 - T2);
        Rds = b->rdswmin + T3 * b->rds0_half;
        dRds_dVg = b->rds0_half * dT3_base * dT0_dVg;
        dRds_dVb = b->rds0_half * dT3_base * dT1_dVb;
    }

    v8d Abulk0, dAbulk0_dVb, Abulk, dAbulk_dVg, dAbulk_dVb;
    {
        v8d T9 = half * b->k1ox * b->lpebFactor / sqrtPhis;
        v8d T1 = T9 + b->k2ox - b->k3b * b->vthNarrowW;
        v8d dT1_dVb = -T9 / sqrtPhis * dsqrtPhis_dVb;

        v8d T9x = v8_sqrt(b->xj * Xdep);
        v8d tmp1 = b->leff + two * T9x;
        v8d T5 = b->leff / tmp1;
        v8d tmp2 = b->a0 * T5;
        v8d T2 = tmp2 + b->b0_ov_wb1;
        v8d dT2_dVb = -T9x / (tmp1 * Xdep) * dXdep_dVb;
        v8d T7 = T5 * T5 * T5;

        Abulk0 = one + T1 * T2;
        dAbulk0_dVb = T1 * tmp2 * dT2_dVb + T2 * dT1_dVb;

        v8d T8 = b->ags_a0 * T7;
        dAbulk_dVg = -T1 * T8;
        Abulk = Abulk0 + dAbulk_dVg * Vgsteff;
        dAbulk_dVb = dAbulk0_dVb - T8 * Vgsteff * (dT1_dVb + v8_set1(3.0) * T1 * dT2_dVb);

        __mmask8 m_a0 = v8_lt(Abulk0, v8_set1(0.1));
        if (m_a0) {
            v8d Ta = one / (v8_set1(3.0) - v8_set1(20.0) * Abulk0);
            Abulk0 = v8_sel(m_a0, (v8_set1(0.2) - Abulk0) * Ta, Abulk0);
            dAbulk0_dVb = v8_sel(m_a0, dAbulk0_dVb * Ta * Ta, dAbulk0_dVb);
        }
        __mmask8 m_ab = v8_lt(Abulk, v8_set1(0.1));
        if (m_ab) {
            v8d Ta = one / (v8_set1(3.0) - v8_set1(20.0) * Abulk);
            v8d Tasq = Ta * Ta;
            Abulk = v8_sel(m_ab, (v8_set1(0.2) - Abulk) * Ta, Abulk);
            dAbulk_dVb = v8_sel(m_ab, dAbulk_dVb * Tasq, dAbulk_dVb);
            dAbulk_dVg = v8_sel(m_ab, dAbulk_dVg * Tasq, dAbulk_dVg);
        }

        v8d Tk = b->keta * Vbseff;
        __mmask8 m_k = v8_ge(Tk, v8_set1(-0.9));
        v8d T0k = one / (one + Tk);
        v8d T1k = one / (v8_set1(0.8) + Tk);
        v8d T0 = v8_sel(m_k, T0k, (v8_set1(17.0) + v8_set1(20.0) * Tk) * T1k);
        v8d dT0_dVb = v8_sel(m_k, -b->keta * T0k * T0k, -b->keta * T1k * T1k);
        dAbulk_dVg = dAbulk_dVg * T0;
        dAbulk_dVb = dAbulk_dVb * T0 + Abulk * dT0_dVb;
        dAbulk0_dVb = dAbulk0_dVb * T0 + Abulk0 * dT0_dVb;
        Abulk = Abulk * T0;
        Abulk0 = Abulk0 * T0;
    }

    /* ===================================================================== */
    /* 6. Mobility (mobMod == 0), EsatL, Vdsat, Vdseff                       */
    /* ===================================================================== */
    v8d ueff, dueff_dVg, dueff_dVd, dueff_dVb;
    {
        v8d T0 = Vgsteff + Vth + Vth;
        v8d T2 = b->ua + b->uc * Vbseff;
        v8d T3 = T0 * b->invToxe;
        v8d T5, dDen_dVg, dDen_dVd, dDen_dVb;
        if (!b->has_ud) {
            T5 = T3 * (T2 + b->ub * T3);
            dDen_dVg = (T2 + two * b->ub * T3) * b->invToxe;
            dDen_dVd = dDen_dVg * two * dVth_dVd;
            dDen_dVb = dDen_dVg * two * dVth_dVb + b->uc * T3;
        } else if (!b->is_v6) {
            v8d invT3 = one / T3;
            v8d T6 = b->ud * invT3 * invT3 * Vth * Vth;
            T5 = T3 * (T2 + b->ub * T3) + T6;
            v8d T7 = -two * T6 / T0;
            dDen_dVg = (T2 + two * b->ub * T3) * b->invToxe + T7;
            dDen_dVd = dDen_dVg * two * dVth_dVd;
            dDen_dVb = dDen_dVg * two * dVth_dVb + b->uc * T3;
        } else {
            v8d T12 = v8_sqrt(Vth * Vth + v8_set1(0.0001));
            v8d T9 = one / (Vgsteff + two * T12);
            v8d T10 = T9 * b->toxe;
            v8d T8 = b->ud * T10 * T10 * Vth;
            v8d T6 = T8 * Vth;
            T5 = T3 * (T2 + b->ub * T3) + T6;
            v8d T7 = -two * T6 * T9;
            v8d T11 = T7 * Vth / T12;
            v8d base_g = (T2 + two * b->ub * T3) * b->invToxe;
            v8d T13 = two * (base_g + T11 + T8);
            dDen_dVd = T13 * dVth_dVd;
            dDen_dVb = T13 * dVth_dVb + b->uc * T3;
            dDen_dVg = base_g + T7;
        }

        __mmask8 m_den = v8_ge(T5, v8_set1(-0.8));
        v8d T9 = one / (v8_set1(7.0) + v8_set1(10.0) * T5);
        v8d Denomi = v8_sel(m_den, one + T5, (v8_set1(0.6) + T5) * T9);
        v8d T9sq = T9 * T9;
        dDen_dVg = v8_sel(m_den, dDen_dVg, dDen_dVg * T9sq);
        dDen_dVd = v8_sel(m_den, dDen_dVd, dDen_dVd * T9sq);
        dDen_dVb = v8_sel(m_den, dDen_dVb, dDen_dVb * T9sq);

        v8d invDen = one / Denomi;
        ueff = b->u0temp * invDen;
        v8d m_u_inv = -ueff * invDen;
        dueff_dVg = m_u_inv * dDen_dVg;
        dueff_dVd = m_u_inv * dDen_dVd;
        dueff_dVb = m_u_inv * dDen_dVb;
    }

    v8d WVCox = Weff * b->vsattemp * b->coxe;
    v8d WVCoxRds = WVCox * Rds;

    v8d Esat = two * b->vsattemp / ueff;
    v8d EsatL = Esat * b->leff;
    v8d T0_es = -EsatL / ueff;
    v8d dEsatL_dVg = T0_es * dueff_dVg;
    v8d dEsatL_dVd = T0_es * dueff_dVd;
    v8d dEsatL_dVb = T0_es * dueff_dVb;

    v8d Lambda = b->a2, dLambda_dVg = z0;
    if (b->m_has_a1) {
        __mmask8 m_pos = v8_gt(b->a1, z0);
        __mmask8 m_neg = v8_lt(b->a1, z0);
        if (m_pos) {
            v8d T0 = one - b->a2;
            v8d T1 = T0 - b->a1 * Vgsteff - v8_set1(0.0001);
            v8d T2 = v8_sqrt(v8_max(T1 * T1 + v8_set1(0.0004) * T0, v8_set1(1.0e-300)));
            Lambda = v8_sel(m_pos, b->a2 + T0 - half * (T1 + T2), Lambda);
            dLambda_dVg = v8_sel(m_pos, half * b->a1 * (one + T1 / T2), dLambda_dVg);
        }
        if (m_neg) {
            v8d T1 = b->a2 + b->a1 * Vgsteff - v8_set1(0.0001);
            v8d T2 = v8_sqrt(v8_max(T1 * T1 + v8_set1(0.0004) * b->a2, v8_set1(1.0e-300)));
            Lambda = v8_sel(m_neg, half * (T1 + T2), Lambda);
            dLambda_dVg = v8_sel(m_neg, half * b->a1 * (one + T1 / T2), dLambda_dVg);
        }
    }

    v8d Vgst2Vtm = Vgsteff + two * b->vtm;
    v8d invWeff = one / Weff;
    __mmask8 m_rds_pos = v8_gt(Rds, z0);
    v8d invRds_safe = v8_sel(m_rds_pos, one / v8_max(Rds, v8_set1(1.0e-300)), z0);
    v8d tmp2_rw = dRds_dVg * invRds_safe + dWeff_dVg * invWeff;
    v8d tmp3_rw = dRds_dVb * invRds_safe + dWeff_dVb * invWeff;

    v8d Vdsat, dVdsat_dVg, dVdsat_dVd, dVdsat_dVb, tmp1_lam;
    {
        __mmask8 m_simple = v8_eq(Rds, z0) & v8_eq(Lambda, one);
        if (m_simple == 0xFF) {
            v8d T0 = one / (Abulk * EsatL + Vgst2Vtm);
            tmp1_lam = z0;
            v8d T1 = T0 * T0;
            v8d T2 = Vgst2Vtm * T0;
            v8d T3 = EsatL * Vgst2Vtm;
            Vdsat = T3 * T0;
            v8d dT0_dVg = -(Abulk * dEsatL_dVg + EsatL * dAbulk_dVg + one) * T1;
            v8d dT0_dVd = -(Abulk * dEsatL_dVd) * T1;
            v8d dT0_dVb = -(Abulk * dEsatL_dVb + dAbulk_dVb * EsatL) * T1;
            dVdsat_dVg = T3 * dT0_dVg + T2 * dEsatL_dVg + EsatL * T0;
            dVdsat_dVd = T3 * dT0_dVd + T2 * dEsatL_dVd;
            dVdsat_dVb = T3 * dT0_dVb + T2 * dEsatL_dVb;
        } else {
            v8d invLam = one / Lambda;
            v8d twoInvLamM1 = two * invLam - one;
            v8d invLamM1 = invLam - one;
            tmp1_lam = dLambda_dVg * invLam * invLam;
            v8d T9 = Abulk * WVCoxRds;
            v8d T8 = Abulk * T9;
            v8d T7 = Vgst2Vtm * T9;
            v8d T6 = Vgst2Vtm * WVCoxRds;
            v8d T0 = two * Abulk * (T9 - one + invLam);
            v8d dT0_dVg = two * (T8 * tmp2_rw - Abulk * tmp1_lam + (two * T9 + invLamM1) * dAbulk_dVg);
            v8d dT0_dVb = two * (T8 * (two / Abulk * dAbulk_dVb + tmp3_rw) + invLamM1 * dAbulk_dVb);
            v8d T1 = Vgst2Vtm * twoInvLamM1 + Abulk * EsatL + v8_set1(3.0) * T7;
            v8d dT1_dVg = twoInvLamM1 - two * Vgst2Vtm * tmp1_lam
                        + Abulk * dEsatL_dVg + EsatL * dAbulk_dVg
                        + v8_set1(3.0) * (T9 + T7 * tmp2_rw + T6 * dAbulk_dVg);
            v8d dT1_dVb = Abulk * dEsatL_dVb + EsatL * dAbulk_dVb
                        + v8_set1(3.0) * (T6 * dAbulk_dVb + T7 * tmp3_rw);
            v8d dT1_dVd = Abulk * dEsatL_dVd;
            v8d T2 = Vgst2Vtm * (EsatL + two * T6);
            v8d dT2_dVg = EsatL + Vgst2Vtm * dEsatL_dVg + T6 * (v8_set1(4.0) + two * Vgst2Vtm * tmp2_rw);
            v8d dT2_dVb = Vgst2Vtm * (dEsatL_dVb + two * T6 * tmp3_rw);
            v8d dT2_dVd = Vgst2Vtm * dEsatL_dVd;
            v8d T3 = v8_sqrt(v8_max(T1 * T1 - two * T0 * T2, v8_set1(1.0e-300)));
            v8d invT0 = one / v8_sel(m_simple, one, T0);
            v8d invT3 = one / T3;
            Vdsat = (T1 - T3) * invT0;
            dVdsat_dVg = (dT1_dVg - (T1 * dT1_dVg - dT0_dVg * T2 - T0 * dT2_dVg) * invT3 - Vdsat * dT0_dVg) * invT0;
            dVdsat_dVb = (dT1_dVb - (T1 * dT1_dVb - dT0_dVb * T2 - T0 * dT2_dVb) * invT3 - Vdsat * dT0_dVb) * invT0;
            dVdsat_dVd = (dT1_dVd - (T1 * dT1_dVd - T0 * dT2_dVd) * invT3) * invT0;
            if (m_simple) {
                v8d sT0 = one / (Abulk * EsatL + Vgst2Vtm);
                v8d sT1 = sT0 * sT0;
                v8d sT2 = Vgst2Vtm * sT0;
                v8d sT3 = EsatL * Vgst2Vtm;
                v8d sdT0g = -(Abulk * dEsatL_dVg + EsatL * dAbulk_dVg + one) * sT1;
                v8d sdT0d = -(Abulk * dEsatL_dVd) * sT1;
                v8d sdT0b = -(Abulk * dEsatL_dVb + dAbulk_dVb * EsatL) * sT1;
                Vdsat      = v8_sel(m_simple, sT3 * sT0, Vdsat);
                dVdsat_dVg = v8_sel(m_simple, sT3 * sdT0g + sT2 * dEsatL_dVg + EsatL * sT0, dVdsat_dVg);
                dVdsat_dVd = v8_sel(m_simple, sT3 * sdT0d + sT2 * dEsatL_dVd, dVdsat_dVd);
                dVdsat_dVb = v8_sel(m_simple, sT3 * sdT0b + sT2 * dEsatL_dVb, dVdsat_dVb);
                tmp1_lam   = v8_sel(m_simple, z0, tmp1_lam);
            }
        }
    }

    /* Vdseff */
    v8d Vdseff, dVdseff_dVg, dVdseff_dVd, dVdseff_dVb;
    {
        v8d T1 = Vdsat - Vds - b->delta;
        v8d dT1_dVg = dVdsat_dVg;
        v8d dT1_dVd = dVdsat_dVd - one;
        v8d dT1_dVb = dVdsat_dVb;
        v8d T2 = v8_sqrt(T1 * T1 + v8_set1(4.0) * b->delta * Vdsat);
        v8d invT2 = one / T2;
        v8d T0 = T1 * invT2;
        v8d T9 = two * b->delta;
        v8d T3 = T9 * invT2;
        v8d dT2_dVg = T0 * dT1_dVg + T3 * dVdsat_dVg;
        v8d dT2_dVd = T0 * dT1_dVd + T3 * dVdsat_dVd;
        v8d dT2_dVb = T0 * dT1_dVb + T3 * dVdsat_dVb;

        __mmask8 m_ge = v8_ge(T1, z0);
        v8d ve_p = Vdsat - half * (T1 + T2);
        v8d dg_p = dVdsat_dVg - half * (dT1_dVg + dT2_dVg);
        v8d dd_p = dVdsat_dVd - half * (dT1_dVd + dT2_dVd);
        v8d db_p = dVdsat_dVb - half * (dT1_dVb + dT2_dVb);

        v8d inv_diff = one / (T2 - T1);
        v8d T4 = T9 * inv_diff;
        v8d T5 = one - T4;
        v8d T6 = Vdsat * T4 * inv_diff;
        v8d ve_n = Vdsat * T5;
        v8d dg_n = dVdsat_dVg * T5 + T6 * (dT2_dVg - dT1_dVg);
        v8d dd_n = dVdsat_dVd * T5 + T6 * (dT2_dVd - dT1_dVd);
        v8d db_n = dVdsat_dVb * T5 + T6 * (dT2_dVb - dT1_dVb);

        Vdseff      = v8_sel(m_ge, ve_p, ve_n);
        dVdseff_dVg = v8_sel(m_ge, dg_p, dg_n);
        dVdseff_dVd = v8_sel(m_ge, dd_p, dd_n);
        dVdseff_dVb = v8_sel(m_ge, db_p, db_n);

        __mmask8 m_vds0 = v8_eq(Vds, z0);
        Vdseff      = v8_sel(m_vds0, z0, Vdseff);
        dVdseff_dVg = v8_sel(m_vds0, z0, dVdseff_dVg);
        dVdseff_dVb = v8_sel(m_vds0, z0, dVdseff_dVb);
        Vdseff      = v8_min(Vdseff, Vds);
    }
    v8d diffVds = Vds - Vdseff;

    /* Velocity Overshoot (lambda > 0) */
    if (b->m_lambda) {
        v8d T1 = b->leff * ueff;
        v8d T2 = b->lambda / T1;
        v8d T3 = -T2 / T1 * b->leff;
        v8d dT2_dVd = T3 * dueff_dVd;
        v8d dT2_dVg = T3 * dueff_dVg;
        v8d dT2_dVb = T3 * dueff_dVb;
        v8d T5 = one / (Esat * b->litl);
        v8d T4 = -T5 / EsatL;
        v8d dT5_dVg = dEsatL_dVg * T4;
        v8d dT5_dVd = dEsatL_dVd * T4;
        v8d dT5_dVb = dEsatL_dVb * T4;
        v8d T6 = one + diffVds * T5;
        v8d dT6_dVg = dT5_dVg * diffVds - dVdseff_dVg * T5;
        v8d dT6_dVd = dT5_dVd * diffVds + (one - dVdseff_dVd) * T5;
        v8d dT6_dVb = dT5_dVb * diffVds - dVdseff_dVb * T5;
        v8d T7 = two / (T6 * T6 + one);
        v8d T8 = one - T7;
        v8d T9 = T6 * T7 * T7;
        v8d T10 = one + T2 * T8;
        __mmask8 m_eq1 = v8_eq(T10, one);
        v8d dT10_dVg = v8_sel(m_eq1, z0, dT2_dVg * T8 + T2 * T9 * dT6_dVg);
        v8d dT10_dVd = v8_sel(m_eq1, z0, dT2_dVd * T8 + T2 * T9 * dT6_dVd);
        v8d dT10_dVb = v8_sel(m_eq1, z0, dT2_dVb * T8 + T2 * T9 * dT6_dVb);
        dEsatL_dVg = v8_sel(b->m_lambda, dEsatL_dVg * T10 + EsatL * dT10_dVg, dEsatL_dVg);
        dEsatL_dVd = v8_sel(b->m_lambda, dEsatL_dVd * T10 + EsatL * dT10_dVd, dEsatL_dVd);
        dEsatL_dVb = v8_sel(b->m_lambda, dEsatL_dVb * T10 + EsatL * dT10_dVb, dEsatL_dVb);
        EsatL      = v8_sel(b->m_lambda, EsatL * T10, EsatL);
        if (b->is_v6)
            Esat = v8_sel(b->m_lambda, EsatL * b->invLeff, Esat);
    }

    /* ===================================================================== */
    /* 7. Vasat, Idl, CLM, DIBL, DITS, SCBE, Isub, Ids, Fsevl                */
    /* ===================================================================== */
    v8d Vasat, dVasat_dVg, dVasat_dVb, dVasat_dVd;
    {
        v8d invVgst2Vtm = one / Vgst2Vtm;
        v8d tmp4 = one - half * Abulk * Vdsat * invVgst2Vtm;
        v8d T9 = WVCoxRds * Vgsteff;
        v8d T8 = T9 * invVgst2Vtm;
        v8d T0 = EsatL + Vdsat + two * T9 * tmp4;
        v8d T7 = two * WVCoxRds * tmp4;
        v8d dT0_dVg = dEsatL_dVg + dVdsat_dVg + T7 * (one + tmp2_rw * Vgsteff)
                    - T8 * (Abulk * dVdsat_dVg - Abulk * Vdsat * invVgst2Vtm + Vdsat * dAbulk_dVg);
        v8d dT0_dVb = dEsatL_dVb + dVdsat_dVb + T7 * tmp3_rw * Vgsteff
                    - T8 * (dAbulk_dVb * Vdsat + Abulk * dVdsat_dVb);
        v8d dT0_dVd = dEsatL_dVd + dVdsat_dVd - T8 * Abulk * dVdsat_dVd;
        v8d T9b = WVCoxRds * Abulk;
        v8d T1 = two / Lambda - one + T9b;
        v8d dT1_dVg = -two * tmp1_lam + WVCoxRds * (Abulk * tmp2_rw + dAbulk_dVg);
        v8d dT1_dVb = dAbulk_dVb * WVCoxRds + T9b * tmp3_rw;
        v8d invT1 = one / T1;
        Vasat = T0 * invT1;
        dVasat_dVg = (dT0_dVg - Vasat * dT1_dVg) * invT1;
        dVasat_dVb = (dT0_dVb - Vasat * dT1_dVb) * invT1;
        dVasat_dVd = dT0_dVd * invT1;
    }

    /* Idl & CoxeffWovL */
    v8d CoxeffWovL, Idl, dIdl_dVg, dIdl_dVd, dIdl_dVb;
    {
        v8d T0 = (Vgsteff + b->vtfbphi2) * b->inv_2e8_toxp;
        v8d tmp3 = v8_exp(b->bdos0p7 * v8_log_pos(T0));
        v8d T1 = one + tmp3;
        v8d invT1 = one / T1;
        v8d T2 = b->bdos0p7 * tmp3 / T0;
        v8d Tcen = b->ados1p9 * invT1;
        v8d dTcen_dVg = -Tcen * T2 * b->inv_2e8_toxp * invT1;
        v8d Coxeff = epssi * b->coxp / (epssi + b->coxp * Tcen);
        v8d dCoxeff_dVg = -Coxeff * Coxeff * dTcen_dVg * v8_set1(1.0 / B4V_EPSSI);
        CoxeffWovL = Coxeff * Weff * b->invLeff;
        v8d beta = ueff * CoxeffWovL;
        v8d T3 = ueff * b->invLeff;
        v8d dbeta_dVg = CoxeffWovL * dueff_dVg + T3 * (Weff * dCoxeff_dVg + Coxeff * dWeff_dVg);
        v8d dbeta_dVd = CoxeffWovL * dueff_dVd;
        v8d dbeta_dVb = CoxeffWovL * dueff_dVb + T3 * Coxeff * dWeff_dVb;

        v8d invVgst2Vtm = one / Vgst2Vtm;
        v8d T0a = one - half * Vdseff * Abulk * invVgst2Vtm;
        v8d dT0a_dVg = -half * (Abulk * dVdseff_dVg - Abulk * Vdseff * invVgst2Vtm + Vdseff * dAbulk_dVg) * invVgst2Vtm;
        v8d dT0a_dVd = -half * Abulk * dVdseff_dVd * invVgst2Vtm;
        v8d dT0a_dVb = -half * (Abulk * dVdseff_dVb + dAbulk_dVb * Vdseff) * invVgst2Vtm;

        v8d fgche1 = Vgsteff * T0a;
        v8d dfgche1_dVg = Vgsteff * dT0a_dVg + T0a;
        v8d dfgche1_dVd = Vgsteff * dT0a_dVd;
        v8d dfgche1_dVb = Vgsteff * dT0a_dVb;

        v8d invEsatL = one / EsatL;
        v8d T9 = Vdseff * invEsatL;
        v8d fgche2 = one + T9;
        v8d invFg2 = one / fgche2;
        v8d dfgche2_dVg = (dVdseff_dVg - T9 * dEsatL_dVg) * invEsatL;
        v8d dfgche2_dVd = (dVdseff_dVd - T9 * dEsatL_dVd) * invEsatL;
        v8d dfgche2_dVb = (dVdseff_dVb - T9 * dEsatL_dVb) * invEsatL;

        v8d gche = beta * fgche1 * invFg2;
        v8d dgche_dVg = (beta * dfgche1_dVg + fgche1 * dbeta_dVg - gche * dfgche2_dVg) * invFg2;
        v8d dgche_dVd = (beta * dfgche1_dVd + fgche1 * dbeta_dVd - gche * dfgche2_dVd) * invFg2;
        v8d dgche_dVb = (beta * dfgche1_dVb + fgche1 * dbeta_dVb - gche * dfgche2_dVb) * invFg2;

        v8d T0g = one + gche * Rds;
        v8d invT0g = one / T0g;
        Idl = gche * invT0g;
        v8d T1g = (one - Idl * Rds) * invT0g;
        v8d T2g = Idl * Idl;
        dIdl_dVg = T1g * dgche_dVg - T2g * dRds_dVg;
        dIdl_dVd = T1g * dgche_dVd;
        dIdl_dVb = T1g * dgche_dVb - T2g * dRds_dVb;
    }

    /* FP (pocket degradation) */
    v8d FP = one, dFP_dVg = z0;
    if (b->m_fprout) {
        v8d T9 = b->fprout * b->sqrtLeff / Vgst2Vtm;
        v8d fp = one / (one + T9);
        FP = v8_sel(b->m_fprout, fp, one);
        dFP_dVg = v8_sel(b->m_fprout, fp * fp * T9 / Vgst2Vtm, z0);
    }

    /* PvagTerm */
    v8d PvagTerm, dPvagTerm_dVg, dPvagTerm_dVb, dPvagTerm_dVd;
    {
        v8d invEsatL = one / EsatL;
        v8d T8 = b->pvag * invEsatL;
        v8d T9 = T8 * Vgsteff;
        __mmask8 m_gt = v8_gt(T9, v8_set1(-0.9));
        v8d T4 = one / (v8_set1(17.0) + v8_set1(20.0) * T9);
        v8d T4sq = T4 * T4;
        v8d scale_d = v8_sel(m_gt, one, T4sq);
        PvagTerm = v8_sel(m_gt, one + T9, (v8_set1(0.8) + T9) * T4);
        dPvagTerm_dVg = T8 * (one - Vgsteff * dEsatL_dVg * invEsatL) * scale_d;
        v8d f_bd = -T9 * invEsatL * scale_d;
        dPvagTerm_dVb = f_bd * dEsatL_dVb;
        dPvagTerm_dVd = f_bd * dEsatL_dVd;
    }

    /* VACLM & Cclm */
    v8d Cclm = max_exp, dCclm_dVg = z0, dCclm_dVb = z0, dCclm_dVd = z0;
    v8d VACLM = max_exp, dVACLM_dVg = z0, dVACLM_dVb = z0, dVACLM_dVd = z0;
    {
        __mmask8 m_clm = b->m_pclm & v8_gt(diffVds, v8_set1(1.0e-10));
        if (m_clm) {
            v8d T0 = one + Rds * Idl;
            v8d dT0_dVg = dRds_dVg * Idl + Rds * dIdl_dVg;
            v8d dT0_dVd = Rds * dIdl_dVd;
            v8d dT0_dVb = dRds_dVb * Idl + Rds * dIdl_dVb;
            v8d invEsat = one / Esat;
            v8d T2 = Vdsat * invEsat;
            v8d T1 = b->leff + T2;
            v8d dT1_dVg = (dVdsat_dVg - T2 * dEsatL_dVg * b->invLeff) * invEsat;
            v8d dT1_dVd = (dVdsat_dVd - T2 * dEsatL_dVd * b->invLeff) * invEsat;
            v8d dT1_dVb = (dVdsat_dVb - T2 * dEsatL_dVb * b->invLeff) * invEsat;
            v8d cc = FP * PvagTerm * T0 * T1 * b->inv_pclm_litl;
            v8d invPv = one / PvagTerm, invT0 = one / T0, invT1 = one / T1;
            v8d dcc_g = cc * (dFP_dVg / FP + dPvagTerm_dVg * invPv + dT0_dVg * invT0 + dT1_dVg * invT1);
            v8d dcc_b = cc * (dPvagTerm_dVb * invPv + dT0_dVb * invT0 + dT1_dVb * invT1);
            v8d dcc_d = cc * (dPvagTerm_dVd * invPv + dT0_dVd * invT0 + dT1_dVd * invT1);
            Cclm = v8_sel(m_clm, cc, max_exp);
            dCclm_dVg = v8_sel(m_clm, dcc_g, z0);
            dCclm_dVb = v8_sel(m_clm, dcc_b, z0);
            dCclm_dVd = v8_sel(m_clm, dcc_d, z0);
            VACLM = v8_sel(m_clm, cc * diffVds, max_exp);
            dVACLM_dVg = v8_sel(m_clm, dcc_g * diffVds - dVdseff_dVg * cc, z0);
            dVACLM_dVb = v8_sel(m_clm, dcc_b * diffVds - dVdseff_dVb * cc, z0);
            dVACLM_dVd = v8_sel(m_clm, dcc_d * diffVds + (one - dVdseff_dVd) * cc, z0);
        }
    }

    /* VADIBL */
    v8d VADIBL = max_exp, dVADIBL_dVg = z0, dVADIBL_dVb = z0, dVADIBL_dVd = z0;
    if (b->m_thetaRout) {
        v8d T8 = Abulk * Vdsat;
        v8d T0 = Vgst2Vtm * T8;
        v8d dT0_dVg = Vgst2Vtm * Abulk * dVdsat_dVg + T8 + Vgst2Vtm * Vdsat * dAbulk_dVg;
        v8d dT0_dVb = Vgst2Vtm * (dAbulk_dVb * Vdsat + Abulk * dVdsat_dVb);
        v8d dT0_dVd = Vgst2Vtm * Abulk * dVdsat_dVd;
        v8d T1 = Vgst2Vtm + T8;
        v8d dT1_dVg = one + Abulk * dVdsat_dVg + Vdsat * dAbulk_dVg;
        v8d dT1_dVb = Abulk * dVdsat_dVb + dAbulk_dVb * Vdsat;
        v8d dT1_dVd = Abulk * dVdsat_dVd;
        v8d invT1 = one / T1;
        v8d invT1sq = invT1 * invT1;
        v8d va = (Vgst2Vtm - T0 * invT1) * b->inv_thetaRout;
        v8d dva_g = (one - dT0_dVg * invT1 + T0 * dT1_dVg * invT1sq) * b->inv_thetaRout;
        v8d dva_b = (-dT0_dVb * invT1 + T0 * dT1_dVb * invT1sq) * b->inv_thetaRout;
        v8d dva_d = (-dT0_dVd * invT1 + T0 * dT1_dVd * invT1sq) * b->inv_thetaRout;

        v8d T7 = b->pdiblb * Vbseff;
        __mmask8 m_ge = v8_ge(T7, v8_set1(-0.9));
        v8d T3_p = one / (one + T7);
        v8d T4_n = one / (v8_set1(0.8) + T7);
        v8d T3_n = (v8_set1(17.0) + v8_set1(20.0) * T7) * T4_n;
        v8d T3 = v8_sel(m_ge, T3_p, T3_n);
        v8d va_p = va * T3_p;
        v8d dvab_p = (dva_b - va_p * b->pdiblb) * T3_p;
        v8d dvab_n = dva_b * T3_n - va * b->pdiblb * T4_n * T4_n;
        va = va * T3;
        dva_g = dva_g * T3;
        dva_d = dva_d * T3;
        dva_b = v8_sel(m_ge, dvab_p, dvab_n);

        dva_g = dva_g * PvagTerm + va * dPvagTerm_dVg;
        dva_b = dva_b * PvagTerm + va * dPvagTerm_dVb;
        dva_d = dva_d * PvagTerm + va * dPvagTerm_dVd;
        va = va * PvagTerm;

        VADIBL      = v8_sel(b->m_thetaRout, va, max_exp);
        dVADIBL_dVg = v8_sel(b->m_thetaRout, dva_g, z0);
        dVADIBL_dVb = v8_sel(b->m_thetaRout, dva_b, z0);
        dVADIBL_dVd = v8_sel(b->m_thetaRout, dva_d, z0);
    }

    v8d Va = Vasat + VACLM;
    v8d dVa_dVg = dVasat_dVg + dVACLM_dVg;
    v8d dVa_dVb = dVasat_dVb + dVACLM_dVb;
    v8d dVa_dVd = dVasat_dVd + dVACLM_dVd;

    /* VADITS (only compute exp when pdits > MIN_EXP!) */
    v8d VADITS = max_exp, dVADITS_dVg = z0, dVADITS_dVd = z0;
    if (b->m_pdits) {
        v8d T0 = b->pditsd * Vds;
        __mmask8 m_le = v8_le(T0, exp_th) & b->m_pdits;
        v8d T1 = max_exp, dT1_dVd = z0;
        if (m_le) {
            v8d ex = v8_exp_clamped(T0, -100.0, B4V_EXP_THRESHOLD);
            T1 = v8_sel(m_le, ex, max_exp);
            dT1_dVd = v8_sel(m_le, ex * b->pditsd, z0);
        }
        v8d vd0 = (one + b->pditsl_term * T1) * b->inv_pdits;
        VADITS      = v8_sel(b->m_pdits, vd0 * FP, max_exp);
        dVADITS_dVg = v8_sel(b->m_pdits, vd0 * dFP_dVg, z0);
        dVADITS_dVd = v8_sel(b->m_pdits, FP * b->pditsl_term * dT1_dVd * b->inv_pdits, z0);
    }

    /* VASCBE */
    v8d VASCBE = max_exp, dVASCBE_dVg = z0, dVASCBE_dVd = z0, dVASCBE_dVb = z0;
    if (b->m_pscbe) {
        v8d scbe_hi = max_exp * b->leff * b->inv_pscbe2;
        v8d thr = b->pscbe1_litl * v8_set1(1.0 / B4V_EXP_THRESHOLD);
        __mmask8 m_act = b->m_pscbe & v8_gt(diffVds, thr);
        VASCBE = v8_sel(b->m_pscbe, scbe_hi, max_exp);
        if (m_act) {
            v8d diff_safe = v8_max(diffVds, v8_set1(1.0e-30));
            v8d T0 = b->pscbe1_litl / diff_safe;
            v8d vs = b->leff * v8_exp_clamped(T0, -100.0, B4V_EXP_THRESHOLD) * b->inv_pscbe2;
            v8d T1 = T0 * vs / diff_safe;
            VASCBE      = v8_sel(m_act, vs, VASCBE);
            dVASCBE_dVg = v8_sel(m_act, T1 * dVdseff_dVg, z0);
            dVASCBE_dVd = v8_sel(m_act, -T1 * (one - dVdseff_dVd), z0);
            dVASCBE_dVb = v8_sel(m_act, T1 * dVdseff_dVb, z0);
        }
    }

    /* Add DIBL */
    v8d invVADIBL = one / VADIBL;
    v8d T9_dibl = diffVds * invVADIBL;
    v8d T0_dibl = one + T9_dibl;
    v8d Idsa = Idl * T0_dibl;
    v8d dIdsa_dVg = T0_dibl * dIdl_dVg - Idl * (dVdseff_dVg + T9_dibl * dVADIBL_dVg) * invVADIBL;
    v8d dIdsa_dVd = T0_dibl * dIdl_dVd + Idl * (one - dVdseff_dVd - T9_dibl * dVADIBL_dVd) * invVADIBL;
    v8d dIdsa_dVb = T0_dibl * dIdl_dVb - Idl * (dVdseff_dVb + T9_dibl * dVADIBL_dVb) * invVADIBL;

    /* Add DITS */
    {
        v8d invVADITS = one / VADITS;
        v8d T9 = diffVds * invVADITS;
        v8d T0 = one + T9;
        dIdsa_dVg = T0 * dIdsa_dVg - Idsa * (dVdseff_dVg + T9 * dVADITS_dVg) * invVADITS;
        dIdsa_dVd = T0 * dIdsa_dVd + Idsa * (one - dVdseff_dVd - T9 * dVADITS_dVd) * invVADITS;
        dIdsa_dVb = T0 * dIdsa_dVb - Idsa * dVdseff_dVb * invVADITS;
        Idsa = Idsa * T0;
    }

    /* Add CLM */
    {
        v8d T0 = v8_log_pos(Va / Vasat);
        v8d invVa = one / Va, invVasat = one / Vasat, invCclm = one / Cclm;
        v8d dT0_dVg = dVa_dVg * invVa - dVasat_dVg * invVasat;
        v8d dT0_dVb = dVa_dVb * invVa - dVasat_dVb * invVasat;
        v8d dT0_dVd = dVa_dVd * invVa - dVasat_dVd * invVasat;
        v8d T1 = T0 * invCclm;
        v8d T9 = one + T1;
        v8d dT9_dVg = (dT0_dVg - T1 * dCclm_dVg) * invCclm;
        v8d dT9_dVb = (dT0_dVb - T1 * dCclm_dVb) * invCclm;
        v8d dT9_dVd = (dT0_dVd - T1 * dCclm_dVd) * invCclm;
        dIdsa_dVg = dIdsa_dVg * T9 + Idsa * dT9_dVg;
        dIdsa_dVb = dIdsa_dVb * T9 + Idsa * dT9_dVb;
        dIdsa_dVd = dIdsa_dVd * T9 + Idsa * dT9_dVd;
        Idsa = Idsa * T9;
    }

    /* Substrate current Isub */
    v8d Isub = z0, Gbg = z0, Gbd = z0, Gbb = z0;
    if (b->m_isub) {
        v8d thr = b->beta0 * v8_set1(1.0 / B4V_EXP_THRESHOLD);
        __mmask8 m_exp = b->m_isub & v8_gt(diffVds, thr);
        v8d T3_lo = b->alpha_ov_leff * min_exp;
        v8d T1 = T3_lo * diffVds;
        v8d dT1_dVg = -T3_lo * dVdseff_dVg;
        v8d dT1_dVd = T3_lo * (one - dVdseff_dVd);
        v8d dT1_dVb = -T3_lo * dVdseff_dVb;
        if (m_exp) {
            v8d diff_safe = v8_max(diffVds, v8_set1(1.0e-30));
            v8d T0 = -b->beta0 / diff_safe;
            v8d T1e = b->alpha_ov_leff * diff_safe * v8_exp_clamped(T0, -B4V_EXP_THRESHOLD, 0.0);
            v8d T3e = T1e / diff_safe * (T0 - one);
            T1      = v8_sel(m_exp, T1e, T1);
            dT1_dVg = v8_sel(m_exp, T3e * dVdseff_dVg, dT1_dVg);
            dT1_dVd = v8_sel(m_exp, T3e * (dVdseff_dVd - one), dT1_dVd);
            dT1_dVb = v8_sel(m_exp, T3e * dVdseff_dVb, dT1_dVb);
        }
        v8d T4 = Idsa * Vdseff;
        v8d gbg0 = T1 * (dIdsa_dVg * Vdseff + Idsa * dVdseff_dVg) + T4 * dT1_dVg;
        v8d gbd0 = T1 * (dIdsa_dVd * Vdseff + Idsa * dVdseff_dVd) + T4 * dT1_dVd;
        v8d gbb0 = T1 * (dIdsa_dVb * Vdseff + Idsa * dVdseff_dVb) + T4 * dT1_dVb;
        gbd0 = gbd0 + gbg0 * dVgsteff_dVd;
        gbb0 = (gbb0 + gbg0 * dVgsteff_dVb) * dVbseff_dVb;
        gbg0 = gbg0 * dVgsteff_dVg;
        Isub = v8_sel(b->m_isub, T1 * T4, z0);
        Gbg  = v8_sel(b->m_isub, gbg0, z0);
        Gbd  = v8_sel(b->m_isub, gbd0, z0);
        Gbb  = v8_sel(b->m_isub, gbb0, z0);
    }

    /* Add SCBE & assemble Ids, Gm, Gds, Gmb */
    v8d Ids, Gm, Gds, Gmb, cdrain;
    {
        v8d invVASCBE = one / VASCBE;
        v8d T9 = diffVds * invVASCBE;
        v8d T0 = one + T9;
        Ids = Idsa * T0;
        v8d gm0  = T0 * dIdsa_dVg - Idsa * (dVdseff_dVg + T9 * dVASCBE_dVg) * invVASCBE;
        v8d gds0 = T0 * dIdsa_dVd + Idsa * (one - dVdseff_dVd - T9 * dVASCBE_dVd) * invVASCBE;
        v8d gmb0 = T0 * dIdsa_dVb - Idsa * (dVdseff_dVb + T9 * dVASCBE_dVb) * invVASCBE;

        v8d tmp1 = gds0 + gm0 * dVgsteff_dVd;
        v8d tmp2 = gmb0 + gm0 * dVgsteff_dVb;
        Gm  = (Ids * dVdseff_dVg + Vdseff * gm0) * dVgsteff_dVg;
        Gds = Ids * (dVdseff_dVd + dVdseff_dVg * dVgsteff_dVd) + Vdseff * tmp1;
        Gmb = (Ids * (dVdseff_dVb + dVdseff_dVg * dVgsteff_dVb) + Vdseff * tmp2) * dVbseff_dVb;
        cdrain = Ids * Vdseff;
    }

    /* Source End Velocity Limit (vtl > 0) */
    if (b->m_vtl) {
        v8d T12 = b->invLeff / CoxeffWovL;
        v8d T11 = T12 / Vgsteff;
        v8d T10 = -T11 / Vgsteff;
        v8d vs = cdrain * T11;
        v8d dvs_dVg = Gm * T11 + cdrain * T10 * dVgsteff_dVg;
        v8d dvs_dVd = Gds * T11 + cdrain * T10 * dVgsteff_dVd;
        v8d dvs_dVb = Gmb * T11 + cdrain * T10 * dVgsteff_dVb;
        v8d T1 = vs * b->inv_vtl_tfactor;
        __mmask8 m_act = b->m_vtl & v8_gt(T1, z0);
        if (m_act) {
            v8d s2 = T1 * T1;
            v8d T2 = one + s2 * s2 * s2;
            v8d vs_safe = v8_sel(m_act, vs, one);
            v8d T3 = (T2 - one) * v8_set1(6.0) / vs_safe;
            /* T2^(-1/6) = exp((-1/6) * log(T2)) */
            v8d Fsevl = v8_exp(v8_set1(-1.0 / 6.0) * v8_log_pos(T2));
            v8d T4 = v8_set1(-1.0 / 6.0) * Fsevl / T2;
            v8d dF_g = T4 * T3 * dvs_dVg;
            v8d dF_d = T4 * T3 * dvs_dVd;
            v8d dF_b = T4 * T3 * dvs_dVb;
            Gm     = v8_sel(m_act, Gm * Fsevl + cdrain * dF_g, Gm);
            Gmb    = v8_sel(m_act, Gmb * Fsevl + cdrain * dF_b, Gmb);
            Gds    = v8_sel(m_act, Gds * Fsevl + cdrain * dF_d, Gds);
            cdrain = v8_sel(m_act, cdrain * Fsevl, cdrain);
        }
    }

    /* ===================================================================== */
    /* 8. GIDL & GISL                                                        */
    /* ===================================================================== */
    v8d Igidl = z0, Ggidld = z0, Ggidlg = z0, Ggidlb = z0;
    if (b->m_gidl_en) {
        v8d T1 = (vds - vgs_eff - b->egidl) * b->inv_3toxe;
        __mmask8 m_act = b->m_gidl_en & v8_gt(T1, z0) & v8_le(vbd, z0);
        if (m_act) {
            v8d T1s = v8_max(T1, v8_set1(1.0e-30));
            v8d dT1_dVd = b->inv_3toxe;
            v8d dT1_dVg = -dvgs_eff_dvg * dT1_dVd;
            v8d T2 = b->bgidl / T1s;
            __mmask8 m_e = m_act & v8_lt(T2, v8_set1(100.0));
            v8d ig_lo = b->agidl_weffCJ * v8_set1(3.720075976e-44);
            v8d gd_l = ig_lo * dT1_dVd;
            v8d gg_l = ig_lo * dT1_dVg;
            v8d ig = ig_lo * T1s;
            v8d gd = gd_l, gg = gg_l;
            if (m_e) {
                v8d ige = b->agidl_weffCJ * T1s * v8_exp_clamped(-T2, -100.0, 0.0);
                v8d T3 = ige * (one + T2) / T1s;
                ig = v8_sel(m_e, ige, ig);
                gd = v8_sel(m_e, T3 * dT1_dVd, gd);
                gg = v8_sel(m_e, T3 * dT1_dVg, gg);
            }
            v8d T4 = vbd * vbd;
            v8d T5 = -vbd * T4;
            v8d T6 = b->cgidl + T5;
            v8d invT6 = one / T6;
            v8d T7 = T5 * invT6;
            v8d T8 = v8_set1(3.0) * b->cgidl * T4 * invT6 * invT6;
            Ggidld = v8_sel(m_act, gd * T7 + ig * T8, z0);
            Ggidlg = v8_sel(m_act, gg * T7, z0);
            Ggidlb = v8_sel(m_act, -ig * T8, z0);
            Igidl  = v8_sel(m_act, ig * T7, z0);
        }
    }

    v8d Igisl = z0, Ggisls = z0, Ggislg = z0, Ggislb = z0;
    if (b->m_gisl_en) {
        v8d T1 = (-vds - vgd_eff - b->egisl) * b->inv_3toxe;
        __mmask8 m_act = b->m_gisl_en & v8_gt(T1, z0) & v8_le(vbs, z0);
        if (m_act) {
            v8d T1s = v8_max(T1, v8_set1(1.0e-30));
            v8d dT1_dVd = b->inv_3toxe;
            v8d dT1_dVg = -dvgd_eff_dvg * dT1_dVd;
            v8d T2 = b->bgisl / T1s;
            __mmask8 m_e = m_act & v8_lt(T2, v8_set1(100.0));
            v8d ig_lo = b->agisl_weffCJ * v8_set1(3.720075976e-44);
            v8d gs_l = ig_lo * dT1_dVd;
            v8d gg_l = ig_lo * dT1_dVg;
            v8d ig = ig_lo * T1s;
            v8d gs = gs_l, gg = gg_l;
            if (m_e) {
                v8d ige = b->agisl_weffCJ * T1s * v8_exp_clamped(-T2, -100.0, 0.0);
                v8d T3 = ige * (one + T2) / T1s;
                ig = v8_sel(m_e, ige, ig);
                gs = v8_sel(m_e, T3 * dT1_dVd, gs);
                gg = v8_sel(m_e, T3 * dT1_dVg, gg);
            }
            v8d T4 = vbs * vbs;
            v8d T5 = -vbs * T4;
            v8d T6 = b->cgisl + T5;
            v8d invT6 = one / T6;
            v8d T7 = T5 * invT6;
            v8d T8 = v8_set1(3.0) * b->cgisl * T4 * invT6 * invT6;
            Ggisls = v8_sel(m_act, gs * T7 + ig * T8, z0);
            Ggislg = v8_sel(m_act, gg * T7, z0);
            Ggislb = v8_sel(m_act, -ig * T8, z0);
            Igisl  = v8_sel(m_act, ig * T7, z0);
        }
    }

    /* Scale DC currents/conductances by nf */
    if (b->m_nf_ne1) {
        cdrain = cdrain * b->nf;
        Gds = Gds * b->nf;
        Gm  = Gm  * b->nf;
        Gmb = Gmb * b->nf;
        Gbb = Gbb * b->nf;
        Gbg = Gbg * b->nf;
        Gbd = Gbd * b->nf;
        Isub = Isub * b->nf;
        Igidl = Igidl * b->nf; Ggidld = Ggidld * b->nf; Ggidlg = Ggidlg * b->nf; Ggidlb = Ggidlb * b->nf;
        Igisl = Igisl * b->nf; Ggisls = Ggisls * b->nf; Ggislg = Ggislg * b->nf; Ggislb = Ggislb * b->nf;
    }

    /* ===================================================================== */
    /* 9. Intrinsic C-V (capMod == 2, Charge-Thickness Model)                */
    /* ===================================================================== */
    v8d Cggb = z0, Cgdb = z0, Cgsb = z0;
    v8d Cdgb = z0, Cddb = z0, Cdsb = z0;
    v8d Csgb = z0, Csdb = z0, Cssb = z0;
    v8d Cbgb = z0, Cbdb = z0, Cbsb = z0;

    if (!dc_only) {
    /* Note: if Vbseff < 0 -> VbseffCV = Vbseff, dVbseffCV_dVb = 1.0;
     *       else -> VbseffCV = phi - Phis = Vbseff, dVbseffCV_dVb = -(-1.0) = 1.0! */
    v8d VbseffCV = Vbseff;
    v8d VgsteffCV, dVgCV_dVg, dVgCV_dVd, dVgCV_dVb;
    {
        v8d noff = n * b->noff;
        v8d dnoff_dVd = b->noff * dn_dVd;
        v8d dnoff_dVb = b->noff * dn_dVb;
        v8d T0 = b->vtm * noff;
        v8d dVg_off = Vgst - b->voffcv;
        v8d VgstNVt = dVg_off / T0;
        __mmask8 m_hi = v8_gt(VgstNVt, exp_th);
        __mmask8 m_lo = v8_lt(VgstNVt, nexp_th);
        __mmask8 m_mid = (~m_hi) & (~m_lo);

        v8d vg_lo = T0 * v8_set1(B4V_LOG1P_MIN_EXP);
        v8d inv_noff = one / noff;
        v8d ratio_lo = vg_lo * inv_noff;
        VgsteffCV = v8_sel(m_hi, dVg_off, vg_lo);
        dVgCV_dVg = v8_sel(m_hi, dVgs_eff_dVg, z0);
        dVgCV_dVd = v8_sel(m_hi, -dVth_dVd, ratio_lo * dnoff_dVd);
        dVgCV_dVb = v8_sel(m_hi, -dVth_dVb, ratio_lo * dnoff_dVb);
        if (m_mid) {
            v8d ex = v8_exp_clamped(VgstNVt, -B4V_EXP_THRESHOLD, B4V_EXP_THRESHOLD);
            v8d vg = T0 * v8_log_pos(one + ex);
            v8d dg0 = ex / (one + ex);
            VgsteffCV = v8_sel(m_mid, vg, VgsteffCV);
            dVgCV_dVd = v8_sel(m_mid, -dg0 * (dVth_dVd + dVg_off * inv_noff * dnoff_dVd) + vg * inv_noff * dnoff_dVd, dVgCV_dVd);
            dVgCV_dVb = v8_sel(m_mid, -dg0 * (dVth_dVb + dVg_off * inv_noff * dnoff_dVb) + vg * inv_noff * dnoff_dVb, dVgCV_dVb);
            dVgCV_dVg = v8_sel(m_mid, dg0 * dVgs_eff_dVg, dVgCV_dVg);
        }
    }

    v8d qgate, qbulk, qdrn;
    v8d cggb, cgsb, cgdb, cdgb, cdsb, cddb, cbgb, cbsb, cbdb;
    {
        v8d V3 = b->vfbzb - Vgs_eff + VbseffCV - v8_set1(B4V_DELTA_3);
        v8d f4 = v8_sel(b->m_vfbzb_le0, v8_set1(-4.0 * B4V_DELTA_3), v8_set1(4.0 * B4V_DELTA_3));
        v8d T0_fb = v8_sqrt(V3 * V3 + f4 * b->vfbzb);
        v8d T1_fb = half * (one + V3 / T0_fb);
        v8d Vfbeff = b->vfbzb - half * (V3 + T0_fb);
        v8d dVfbeff_dVg = T1_fb * dVgs_eff_dVg;
        v8d dVfbeff_dVb = -T1_fb;

        v8d T0_ox = (Vgs_eff - VbseffCV - b->vfbzb) * b->inv_tox_ctm;
        v8d dT0_ox_dVg = dVgs_eff_dVg * b->inv_tox_ctm;
        v8d dT0_ox_dVb = -b->inv_tox_ctm;
        v8d tmp_ac = T0_ox * b->acde;
        __mmask8 m_ac_lo = v8_le(tmp_ac, nexp_th);
        __mmask8 m_ac_hi = v8_ge(tmp_ac, exp_th);
        __mmask8 m_ac_mid = (~m_ac_lo) & (~m_ac_hi);
        v8d Tcen = b->ldeb * v8_sel(m_ac_lo, min_exp, max_exp);
        v8d dTcen_dVg = z0, dTcen_dVb = z0;
        if (m_ac_mid) {
            v8d tc = b->ldeb * v8_exp_clamped(tmp_ac, -B4V_EXP_THRESHOLD, B4V_EXP_THRESHOLD);
            v8d dtc = b->acde * tc;
            Tcen = v8_sel(m_ac_mid, tc, Tcen);
            dTcen_dVb = v8_sel(m_ac_mid, dtc * dT0_ox_dVb, z0);
            dTcen_dVg = v8_sel(m_ac_mid, dtc * dT0_ox_dVg, z0);
        }

        v8d V3l = b->ldeb - Tcen - b->link_ctm;
        v8d V4l = v8_sqrt(V3l * V3l + v8_set1(4.0) * b->link_ctm * b->ldeb);
        Tcen = b->ldeb - half * (V3l + V4l);
        v8d T1l = half * (one + V3l / V4l);
        dTcen_dVg = dTcen_dVg * T1l;
        dTcen_dVb = dTcen_dVb * T1l;

        v8d Ccen = epssi / Tcen;
        v8d T2c = b->coxp / (b->coxp + Ccen);
        v8d Coxeff = T2c * Ccen;
        v8d dCox_base = T2c * T2c * (-Ccen / Tcen);
        v8d dCoxeff_dVg = dCox_base * dTcen_dVg;
        v8d dCoxeff_dVb = dCox_base * dTcen_dVb;
        v8d CoxWLcen = b->CoxWL_ov_coxe * Coxeff;

        v8d Qac0 = CoxWLcen * (Vfbeff - b->vfbzb);
        v8d QovCox_ac = Qac0 / Coxeff;
        v8d dQac0_dVg = CoxWLcen * dVfbeff_dVg + QovCox_ac * dCoxeff_dVg;
        v8d dQac0_dVb = CoxWLcen * dVfbeff_dVb + QovCox_ac * dCoxeff_dVb;

        v8d T0s = half * b->k1ox;
        v8d T3s = Vgs_eff - Vfbeff - VbseffCV - VgsteffCV;
        __mmask8 m_t3neg = v8_lt(T3s, z0);
        v8d T1_pos = v8_sqrt(v8_max(T0s * T0s + T3s, v8_set1(1.0e-300)));
        v8d T1_neg = T0s + T3s / v8_sel(b->m_k1ox_zero, one, b->k1ox);
        v8d T1s = v8_sel(b->m_k1ox_zero, z0, v8_sel(m_t3neg, T1_neg, T1_pos));
        v8d T2s = v8_sel(b->m_k1ox_zero, z0, v8_sel(m_t3neg, CoxWLcen, CoxWLcen * T0s / T1_pos));
        v8d Qsub0 = CoxWLcen * b->k1ox * (T1s - T0s);
        v8d QovCox_sub = Qsub0 / Coxeff;
        v8d dQsub0_dVg = T2s * (dVgs_eff_dVg - dVfbeff_dVg - dVgCV_dVg) + QovCox_sub * dCoxeff_dVg;
        v8d dQsub0_dVd = -T2s * dVgCV_dVd;
        v8d dQsub0_dVb = -T2s * (dVfbeff_dVb + one + dVgCV_dVb) + QovCox_sub * dCoxeff_dVb;

        /* DeltaPhi & VgDP */
        v8d Den_dp = v8_sel(b->m_k1ox_le0, v8_set1(0.25) * b->moin * b->vtm,
                            b->moin * b->vtm * b->k1ox * b->k1ox);
        v8d T0_dp = v8_sel(b->m_k1ox_le0, half * b->sqrtPhi, b->k1ox * b->sqrtPhi);
        v8d T1_dp = two * T0_dp + VgsteffCV;
        v8d DeltaPhi = b->vtm * v8_log_pos(one + T1_dp * VgsteffCV / Den_dp);
        v8d dDeltaPhi_dVg = two * b->vtm * (T1_dp - T0_dp) / (Den_dp + T1_dp * VgsteffCV);

        v8d T0_vg = VgsteffCV - DeltaPhi - v8_set1(0.001);
        v8d dT0_vg = one - dDeltaPhi_dVg;
        v8d T1_vg = v8_sqrt(T0_vg * T0_vg + VgsteffCV * v8_set1(0.004));
        v8d VgDP = half * (T0_vg + T1_vg);
        v8d dVgDP_dVg = half * (dT0_vg + (T0_vg * dT0_vg + v8_set1(0.002)) / T1_vg);

        /* Inversion Tcen & Coxeff */
        v8d T0_i = (VgsteffCV + b->vtfbphi2) * b->inv_2e8_toxp;
        v8d tmp_i = v8_exp(b->bdos0p7 * v8_log_pos(T0_i));
        v8d T1_i = one + tmp_i;
        v8d invT1_i = one / T1_i;
        v8d T2_i = b->bdos0p7 * tmp_i * b->inv_2e8_toxp / T0_i;
        v8d Tcen_i = b->ados1p9 * invT1_i;
        v8d dTcen_i0 = -Tcen_i * T2_i * invT1_i;
        v8d dTcen_id = dTcen_i0 * dVgCV_dVd;
        v8d dTcen_ib = dTcen_i0 * dVgCV_dVb;
        v8d dTcen_ig = dTcen_i0 * dVgCV_dVg;

        v8d Ccen_i = epssi / Tcen_i;
        v8d T0_ci = b->coxp / (b->coxp + Ccen_i);
        v8d Coxeff_i = T0_ci * Ccen_i;
        v8d dCox_i0 = T0_ci * T0_ci * (-Ccen_i / Tcen_i);
        v8d dCoxeff_id = dCox_i0 * dTcen_id;
        v8d dCoxeff_ib = dCox_i0 * dTcen_ib;
        v8d dCoxeff_ig = dCox_i0 * dTcen_ig;
        v8d CoxWLcen_i = b->CoxWL_ov_coxe * Coxeff_i;

        /* VdsatCV & VdseffCV */
        v8d AbulkCV = Abulk0 * b->abulkCVfactor;
        v8d dAbulkCV_dVb = b->abulkCVfactor * dAbulk0_dVb;
        v8d invAbCV = one / AbulkCV;
        v8d VdsatCV = VgDP * invAbCV;

        v8d T0_v = VdsatCV - Vds - v8_set1(B4V_DELTA_4);
        v8d dT0_vg_cv = dVgDP_dVg * invAbCV;
        v8d dT0_vb_cv = -VdsatCV * dAbulkCV_dVb * invAbCV;
        v8d T1_v = v8_sqrt(T0_v * T0_v + v8_set1(4.0 * B4V_DELTA_4) * VdsatCV);
        v8d invT1_v = one / T1_v;
        v8d dT1_base = (T0_v + v8_set1(2.0 * B4V_DELTA_4)) * invT1_v;
        v8d dT1_vd_cv = -T0_v * invT1_v;
        v8d dT1_vb_cv = dT1_base * dT0_vb_cv;
        v8d dT1_vg_cv = dT1_base * dT0_vg_cv;

        __mmask8 m_ge = v8_ge(T0_v, z0);
        v8d inv_diff_v = one / (T1_v - T0_v);
        v8d T3_v = v8_set1(2.0 * B4V_DELTA_4) * inv_diff_v;
        v8d T4_v = one - T3_v;
        v8d T5_v = VdsatCV * T3_v * inv_diff_v;

        v8d VdseffCV     = v8_sel(m_ge, VdsatCV - half * (T0_v + T1_v), VdsatCV * T4_v);
        v8d dVdseffCV_dVg = v8_sel(m_ge, half * (dT0_vg_cv - dT1_vg_cv), dT0_vg_cv * T4_v + T5_v * (dT1_vg_cv - dT0_vg_cv));
        v8d dVdseffCV_dVd = v8_sel(m_ge, half * (one - dT1_vd_cv), T5_v * (dT1_vd_cv + one));
        v8d vb_fac = b->is_v6 ? (T4_v - T5_v) : (one - T5_v);
        v8d dVdseffCV_dVb = v8_sel(m_ge, half * (dT0_vb_cv - dT1_vb_cv), dT0_vb_cv * vb_fac + T5_v * dT1_vb_cv);

        __mmask8 m_vds0 = v8_eq(Vds, z0);
        VdseffCV      = v8_sel(m_vds0, z0, VdseffCV);
        dVdseffCV_dVg = v8_sel(m_vds0, z0, dVdseffCV_dVg);
        dVdseffCV_dVb = v8_sel(m_vds0, z0, dVdseffCV_dVb);

        /* Channel charges qgate, qbulk, qsrc */
        v8d T0_q = AbulkCV * VdseffCV;
        v8d T1_q = VgDP;
        v8d T2_q = v8_set1(12.0) * (T1_q - half * T0_q + v8_set1(1.0e-20));
        v8d invT2_q = one / T2_q;
        v8d T3_q = T0_q * invT2_q;
        v8d T4_q = one - v8_set1(12.0) * T3_q * T3_q;
        v8d T5_q = AbulkCV * (v8_set1(6.0) * T0_q * (v8_set1(4.0) * T1_q - T0_q) * (invT2_q * invT2_q) - half);
        v8d T6_q = T5_q * VdseffCV * invAbCV;

        v8d invCox_i = one / Coxeff_i;
        qgate = CoxWLcen_i * (T1_q - T0_q * (half - T3_q));
        v8d QovCox_g = qgate * invCox_i;
        v8d Cgg1_0 = CoxWLcen_i * (T4_q * dVgDP_dVg + T5_q * dVdseffCV_dVg);
        v8d Cgd1 = CoxWLcen_i * T5_q * dVdseffCV_dVd + Cgg1_0 * dVgCV_dVd + QovCox_g * dCoxeff_id;
        v8d Cgb1 = CoxWLcen_i * (T5_q * dVdseffCV_dVb + T6_q * dAbulkCV_dVb) + Cgg1_0 * dVgCV_dVb + QovCox_g * dCoxeff_ib;
        v8d Cgg1 = Cgg1_0 * dVgCV_dVg + QovCox_g * dCoxeff_ig;

        v8d T7_q = one - AbulkCV;
        v8d T8_q = T2_q * T2_q;
        v8d T9_q = v8_set1(12.0) * T7_q * T0_q * T0_q / (T8_q * AbulkCV);
        v8d T10_q = T9_q * dVgDP_dVg;
        v8d T11_q = -T7_q * T5_q * invAbCV;
        v8d T12_q = -(T9_q * T1_q * invAbCV + VdseffCV * (half - T3_q));

        qbulk = CoxWLcen_i * T7_q * (half * VdseffCV - T0_q * VdseffCV * invT2_q);
        v8d QovCox_b = qbulk * invCox_i;
        v8d Cbg1_0 = CoxWLcen_i * (T10_q + T11_q * dVdseffCV_dVg);
        v8d Cbd1 = CoxWLcen_i * T11_q * dVdseffCV_dVd + Cbg1_0 * dVgCV_dVd + QovCox_b * dCoxeff_id;
        v8d Cbb1 = CoxWLcen_i * (T11_q * dVdseffCV_dVb + T12_q * dAbulkCV_dVb) + Cbg1_0 * dVgCV_dVb + QovCox_b * dCoxeff_ib;
        v8d Cbg1 = Cbg1_0 * dVgCV_dVg + QovCox_b * dCoxeff_ig;

        v8d qsrc, Csg, Csd, Csb;
        if (b->xpart_mode == 0) {
            /* All 8 lanes use 40/60 partition (standard GF180 xpart=0) */
            v8d T2p = T2_q * v8_set1(1.0 / 12.0);
            v8d invT2p = one / T2p;
            v8d T3p = half * CoxWLcen_i * (invT2p * invT2p);
            v8d T4p = T1_q * (v8_set1(2.0 / 3.0) * T0_q * T0_q + T1_q * (T1_q - v8_set1(4.0 / 3.0) * T0_q))
                    - v8_set1(2.0 / 15.0) * T0_q * T0_q * T0_q;
            qsrc = -T3p * T4p;
            v8d QovCox_s = qsrc * invCox_i;
            v8d T8p = v8_set1(4.0 / 3.0) * T1_q * (T1_q - T0_q) + v8_set1(0.4) * T0_q * T0_q;
            v8d T5p = -two * qsrc * invT2p - T3p * (T1_q * (v8_set1(3.0) * T1_q - v8_set1(8.0 / 3.0) * T0_q)
                    + v8_set1(2.0 / 3.0) * T0_q * T0_q);
            v8d T6p = AbulkCV * (qsrc * invT2p + T3p * T8p);
            v8d T7p = T6p * VdseffCV * invAbCV;
            v8d Csg0 = T5p * dVgDP_dVg + T6p * dVdseffCV_dVg;
            Csd = Csg0 * dVgCV_dVd + T6p * dVdseffCV_dVd + QovCox_s * dCoxeff_id;
            Csb = Csg0 * dVgCV_dVb + T6p * dVdseffCV_dVb + T7p * dAbulkCV_dVb + QovCox_s * dCoxeff_ib;
            Csg = Csg0 * dVgCV_dVg + QovCox_s * dCoxeff_ig;
        } else if (b->xpart_mode == 2) {
            /* All 8 lanes use 50/50 partition */
            qsrc = -half * qgate;
            Csg  = -half * Cgg1;
            Csd  = -half * Cgd1;
            Csb  = -half * Cgb1;
        } else {
            /* 0/100 or mixed: start with 50/50 then blend 0/100 and 40/60 */
            qsrc = -half * qgate;
            Csg  = -half * Cgg1;
            Csd  = -half * Cgd1;
            Csb  = -half * Cgb1;
            if (b->m_xpart_gt05) {
                v8d qs0 = -CoxWLcen_i * (half * T1_q + v8_set1(0.25) * T0_q - half * T0_q * T0_q * invT2_q);
                v8d Qov = qs0 * invCox_i;
                v8d T2d = T2_q + T2_q;
                v8d invT3d = one / (T2d * T2d);
                v8d T7d = -(v8_set1(0.25) - v8_set1(12.0) * T0_q * (v8_set1(4.0) * T1_q - T0_q) * invT3d);
                v8d T4d = -(half + v8_set1(24.0) * T0_q * T0_q * invT3d) * dVgDP_dVg;
                v8d T5d = T7d * AbulkCV;
                v8d T6d = T7d * VdseffCV;
                v8d Csg0 = CoxWLcen_i * (T4d + T5d * dVdseffCV_dVg);
                qsrc = v8_sel(b->m_xpart_gt05, qs0, qsrc);
                Csd  = v8_sel(b->m_xpart_gt05, CoxWLcen_i * T5d * dVdseffCV_dVd + Csg0 * dVgCV_dVd + Qov * dCoxeff_id, Csd);
                Csb  = v8_sel(b->m_xpart_gt05, CoxWLcen_i * (T5d * dVdseffCV_dVb + T6d * dAbulkCV_dVb) + Csg0 * dVgCV_dVb + Qov * dCoxeff_ib, Csb);
                Csg  = v8_sel(b->m_xpart_gt05, Csg0 * dVgCV_dVg + Qov * dCoxeff_ig, Csg);
            }
            if (b->m_xpart_lt05) {
                v8d T2p = T2_q * v8_set1(1.0 / 12.0);
                v8d invT2p = one / T2p;
                v8d T3p = half * CoxWLcen_i * (invT2p * invT2p);
                v8d T4p = T1_q * (v8_set1(2.0 / 3.0) * T0_q * T0_q + T1_q * (T1_q - v8_set1(4.0 / 3.0) * T0_q))
                        - v8_set1(2.0 / 15.0) * T0_q * T0_q * T0_q;
                v8d qs4 = -T3p * T4p;
                v8d Qov = qs4 * invCox_i;
                v8d T8p = v8_set1(4.0 / 3.0) * T1_q * (T1_q - T0_q) + v8_set1(0.4) * T0_q * T0_q;
                v8d T5p = -two * qs4 * invT2p - T3p * (T1_q * (v8_set1(3.0) * T1_q - v8_set1(8.0 / 3.0) * T0_q)
                        + v8_set1(2.0 / 3.0) * T0_q * T0_q);
                v8d T6p = AbulkCV * (qs4 * invT2p + T3p * T8p);
                v8d T7p = T6p * VdseffCV * invAbCV;
                v8d Csg0 = T5p * dVgDP_dVg + T6p * dVdseffCV_dVg;
                qsrc = v8_sel(b->m_xpart_lt05, qs4, qsrc);
                Csd  = v8_sel(b->m_xpart_lt05, Csg0 * dVgCV_dVd + T6p * dVdseffCV_dVd + Qov * dCoxeff_id, Csd);
                Csb  = v8_sel(b->m_xpart_lt05, Csg0 * dVgCV_dVb + T6p * dVdseffCV_dVb + T7p * dAbulkCV_dVb + Qov * dCoxeff_ib, Csb);
                Csg  = v8_sel(b->m_xpart_lt05, Csg0 * dVgCV_dVg + Qov * dCoxeff_ig, Csg);
            }
        }

        qgate = qgate + Qac0 + Qsub0 - qbulk;
        qbulk = qbulk - (Qac0 + Qsub0);
        qdrn  = -(qgate + qbulk + qsrc);

        v8d Cbg = Cbg1 - dQac0_dVg - dQsub0_dVg;
        v8d Cbd = Cbd1 - dQsub0_dVd;
        v8d Cbb = (Cbb1 - dQac0_dVb - dQsub0_dVb) * dVbseff_dVb;
        v8d Cgg = Cgg1 - Cbg;
        v8d Cgd = Cgd1 - Cbd;
        v8d Cgb = (Cgb1 - (Cbb1 - dQac0_dVb - dQsub0_dVb)) * dVbseff_dVb;
        Csb = Csb * dVbseff_dVb;

        cggb = Cgg;
        cgsb = -(Cgg + Cgd + Cgb);
        cgdb = Cgd;
        cdgb = -(Cgg + Cbg + Csg);
        cdsb = Cgg + Cgd + Cgb + Cbg + Cbd + Cbb + Csg + Csd + Csb;
        cddb = -(Cgd + Cbd + Csd);
        cbgb = Cbg;
        cbsb = -(Cbg + Cbd + Cbb);
        cbdb = Cbd;
    }

    /* ===================================================================== */
    /* 10. Junction Depletion Capacitances (qbs, capbs, qbd, capbd)          */
    /* ===================================================================== */
    v8d qbs, capbs, qbd, capbd;
    {
        /* Source-bulk */
        __mmask8 m_neg = v8_lt(vbs, z0);
        v8d T1_f = vbs * b->jctS_lin;
        qbs   = vbs * (b->jctS_sum + half * T1_f);
        capbs = b->jctS_sum + T1_f;
        if (m_neg) {
            v8d q_n = z0, c_n = z0;
            if (b->m_czbs & m_neg) {
                __mmask8 m = b->m_czbs & m_neg;
                v8d arg = v8_sel(m, one - vbs * b->invPhiBS, one);
                v8d sarg;
                if ((b->m_mjs_half & m) == m)
                    sarg = one / v8_sqrt(arg);
                else {
                    sarg = v8_exp(-b->MJS * v8_log(arg));
                    if (b->m_mjs_half & m)
                        sarg = v8_sel(b->m_mjs_half & m, one / v8_sqrt(arg), sarg);
                }
                q_n = v8_sel(m, q_n + b->PhiBS * b->czbs * (one - arg * sarg) * b->inv1mMJS, q_n);
                c_n = v8_sel(m, c_n + b->czbs * sarg, c_n);
            }
            if (b->m_czbssw & m_neg) {
                __mmask8 m = b->m_czbssw & m_neg;
                v8d arg = v8_sel(m, one - vbs * b->invPhiBSWS, one);
                v8d sarg;
                if ((b->m_mjsws_half & m) == m)
                    sarg = one / v8_sqrt(arg);
                else {
                    sarg = v8_exp(-b->MJSWS * v8_log(arg));
                    if (b->m_mjsws_half & m)
                        sarg = v8_sel(b->m_mjsws_half & m, one / v8_sqrt(arg), sarg);
                }
                q_n = v8_sel(m, q_n + b->PhiBSWS * b->czbssw * (one - arg * sarg) * b->inv1mMJSWS, q_n);
                c_n = v8_sel(m, c_n + b->czbssw * sarg, c_n);
            }
            if (b->m_czbsswg & m_neg) {
                __mmask8 m = b->m_czbsswg & m_neg;
                v8d arg = v8_sel(m, one - vbs * b->invPhiBSWGS, one);
                v8d sarg;
                if ((b->m_mjswgs_half & m) == m)
                    sarg = one / v8_sqrt(arg);
                else {
                    sarg = v8_exp(-b->MJSWGS * v8_log(arg));
                    if (b->m_mjswgs_half & m)
                        sarg = v8_sel(b->m_mjswgs_half & m, one / v8_sqrt(arg), sarg);
                }
                q_n = v8_sel(m, q_n + b->PhiBSWGS * b->czbsswg * (one - arg * sarg) * b->inv1mMJSWGS, q_n);
                c_n = v8_sel(m, c_n + b->czbsswg * sarg, c_n);
            }
            qbs   = v8_sel(m_neg, q_n, qbs);
            capbs = v8_sel(m_neg, c_n, capbs);
        }
    }
    {
        /* Drain-bulk */
        __mmask8 m_neg = v8_lt(vbd, z0);
        v8d T1_f = vbd * b->jctD_lin;
        qbd   = vbd * (b->jctD_sum + half * T1_f);
        capbd = b->jctD_sum + T1_f;
        if (m_neg) {
            v8d q_n = z0, c_n = z0;
            if (b->m_czbd & m_neg) {
                __mmask8 m = b->m_czbd & m_neg;
                v8d arg = v8_sel(m, one - vbd * b->invPhiBD, one);
                v8d sarg;
                if ((b->m_mjd_half & m) == m)
                    sarg = one / v8_sqrt(arg);
                else {
                    sarg = v8_exp(-b->MJD * v8_log(arg));
                    if (b->m_mjd_half & m)
                        sarg = v8_sel(b->m_mjd_half & m, one / v8_sqrt(arg), sarg);
                }
                q_n = v8_sel(m, q_n + b->PhiBD * b->czbd * (one - arg * sarg) * b->inv1mMJD, q_n);
                c_n = v8_sel(m, c_n + b->czbd * sarg, c_n);
            }
            if (b->m_czbdsw & m_neg) {
                __mmask8 m = b->m_czbdsw & m_neg;
                v8d arg = v8_sel(m, one - vbd * b->invPhiBSWD, one);
                v8d sarg;
                if ((b->m_mjswd_half & m) == m)
                    sarg = one / v8_sqrt(arg);
                else {
                    sarg = v8_exp(-b->MJSWD * v8_log(arg));
                    if (b->m_mjswd_half & m)
                        sarg = v8_sel(b->m_mjswd_half & m, one / v8_sqrt(arg), sarg);
                }
                q_n = v8_sel(m, q_n + b->PhiBSWD * b->czbdsw * (one - arg * sarg) * b->inv1mMJSWD, q_n);
                c_n = v8_sel(m, c_n + b->czbdsw * sarg, c_n);
            }
            if (b->m_czbdswg & m_neg) {
                __mmask8 m = b->m_czbdswg & m_neg;
                v8d arg = v8_sel(m, one - vbd * b->invPhiBSWGD, one);
                v8d sarg;
                if ((b->m_mjswgd_half & m) == m)
                    sarg = one / v8_sqrt(arg);
                else {
                    sarg = v8_exp(-b->MJSWGD * v8_log(arg));
                    if (b->m_mjswgd_half & m)
                        sarg = v8_sel(b->m_mjswgd_half & m, one / v8_sqrt(arg), sarg);
                }
                q_n = v8_sel(m, q_n + b->PhiBSWGD * b->czbdswg * (one - arg * sarg) * b->inv1mMJSWGD, q_n);
                c_n = v8_sel(m, c_n + b->czbdswg * sarg, c_n);
            }
            qbd   = v8_sel(m_neg, q_n, qbd);
            capbd = v8_sel(m_neg, c_n, capbd);
        }
    }

    /* ===================================================================== */
    /* 11. Overlap Capacitances (capMod == 2) & Total Charge Assembly        */
    /* ===================================================================== */
    v8d cgdo, qgdo, cgso, qgso;
    {
        v8d T0 = vgd + v8_set1(B4V_DELTA_1);
        v8d T1 = v8_sqrt(T0 * T0 + v8_set1(4.0 * B4V_DELTA_1));
        v8d T2 = half * (T0 - T1);
        v8d T4 = v8_sqrt(one - v8_set1(4.0) * T2 * b->inv_ckappad);
        cgdo = b->cgdo + b->weffCV_cgdl - b->weffCV_cgdl * (one - one / T4) * (half - half * T0 / T1);
        qgdo = (b->cgdo + b->weffCV_cgdl) * vgd - b->weffCV_cgdl * (T2 + half * b->ckappad * (T4 - one));

        v8d T0s = vgs + v8_set1(B4V_DELTA_1);
        v8d T1s = v8_sqrt(T0s * T0s + v8_set1(4.0 * B4V_DELTA_1));
        v8d T2s = half * (T0s - T1s);
        v8d T4s = v8_sqrt(one - v8_set1(4.0) * T2s * b->inv_ckappas);
        cgso = b->cgso + b->weffCV_cgsl - b->weffCV_cgsl * (one - one / T4s) * (half - half * T0s / T1s);
        qgso = (b->cgso + b->weffCV_cgsl) * vgs - b->weffCV_cgsl * (T2s + half * b->ckappas * (T4s - one));

        if (b->m_nf_ne1) {
            cgdo = cgdo * b->nf;
            cgso = cgso * b->nf;
            qgdo = qgdo * b->nf;
            qgso = qgso * b->nf;
        }
    }

    v8d qgb = b->cgbo * vgb;
    v8d qgate_tot = qgate + qgdo + qgso + qgb;
    v8d qbulk_tot = qbulk - qgb;
    v8d qdrn_fwd  = qdrn - qgdo;
    v8d qsrc_fwd  = -(qgate_tot + qbulk_tot + qdrn_fwd);
    v8d qsrc_rev  = qdrn - qgso;
    v8d qdrn_rev  = -(qgate_tot + qbulk_tot + qsrc_rev);
    v8d qdrn_tot  = v8_sel(m_fwd, qdrn_fwd, qdrn_rev);
    v8d qsrc_tot  = v8_sel(m_fwd, qsrc_fwd, qsrc_rev);

    /* Unscaled (time-independent) terminal capacitances */
    Cggb = cggb + cgdo + cgso + b->cgbo;
    Cgdb = v8_sel(m_fwd, cgdb, cgsb) - cgdo;
    Cgsb = v8_sel(m_fwd, cgsb, cgdb) - cgso;

    v8d cgg_bg_dg = cggb + cbgb + cdgb;
    Cdgb = v8_sel(m_fwd, cdgb - cgdo, -(cgg_bg_dg + cgdo));
    Csgb = v8_sel(m_fwd, -(cgg_bg_dg + cgso), cdgb - cgso);
    Cbgb = cbgb - b->cgbo;

    v8d c_sum_d = cgdb + cbdb + cddb;
    v8d c_sum_s = cgsb + cbsb + cdsb;
    Cddb = v8_sel(m_fwd, cddb + capbd + cgdo, capbd + cgdo - c_sum_s);
    Cdsb = v8_sel(m_fwd, cdsb, -c_sum_d);
    Csdb = v8_sel(m_fwd, -c_sum_d, cdsb);
    Cssb = v8_sel(m_fwd, capbs + cgso - c_sum_s, cddb + capbs + cgso);

    Cbdb = v8_sel(m_fwd, cbdb, cbsb) - capbd;
    Cbsb = v8_sel(m_fwd, cbsb, cbdb) - capbs;

    /* ===================================================================== */
    /* 12. Cache Time-Independent Physics (DC + C + Charges) in dyn          */
    /* ===================================================================== */
    v8d qg_st = qgate_tot;
    v8d qd_st = qdrn_tot - qbd;
    v8d qs_st = qsrc_tot - qbs;
    v8d qb_st = qbulk_tot + qbd + qbs;

    v8_store(dyn->a_vds, vds); v8_store(dyn->a_vgs, vgs);
    v8_store(dyn->a_vbs, vbs); v8_store(dyn->a_vbd, vbd);
    v8_store(dyn->a_qg, qg_st); v8_store(dyn->a_qd, qd_st); v8_store(dyn->a_qb, qb_st);
    dyn->qb_st = qb_st; dyn->qg_st = qg_st; dyn->qd_st = qd_st;

    if (need_aos) {
        double a_qbs[8], a_qbd[8], a_qs[8];
        v8_store(a_qbs, qbs); v8_store(a_qbd, qbd); v8_store(a_qs, qs_st);
        for (int lane = 0; lane < b->n_active; lane++) {
            int st = b->stateBase[lane];
            s0[st + 4]  = dyn->a_vbs[lane]; /* vdbs */
            s0[st + 5]  = dyn->a_vbd[lane]; /* vdbd */
            s0[st + 6]  = dyn->a_vbs[lane]; /* vsbs */
            s0[st + 7]  = dyn->a_vgs[lane]; /* vges */
            s0[st + 8]  = dyn->a_vgs[lane]; /* vgms */
            s0[st + 9]  = 0.0;              /* vses */
            s0[st + 10] = dyn->a_vds[lane]; /* vdes */
            s0[st + 19] = a_qbs[lane];
            s0[st + 21] = a_qbd[lane];
            s0[st + 27] = 0.0;              /* qdef */
            s0[st + 28] = a_qs[lane];
        }
    }
    } else {
        v8_store(dyn->a_vds, vds); v8_store(dyn->a_vgs, vgs);
        v8_store(dyn->a_vbs, vbs); v8_store(dyn->a_vbd, vbd);
        for (int lane = 0; lane < b->n_active; lane++) {
            int st = b->stateBase[lane];
            s0[st + 0]  = dyn->a_vbd[lane];
            s0[st + 1]  = dyn->a_vbs[lane];
            s0[st + 2]  = dyn->a_vgs[lane];
            s0[st + 3]  = dyn->a_vds[lane];
            s0[st + 4]  = dyn->a_vbs[lane]; /* vdbs */
            s0[st + 5]  = dyn->a_vbd[lane]; /* vdbd */
            s0[st + 6]  = dyn->a_vbs[lane]; /* vsbs */
            s0[st + 7]  = dyn->a_vgs[lane]; /* vges */
            s0[st + 8]  = dyn->a_vgs[lane]; /* vgms */
            s0[st + 9]  = 0.0;              /* vses */
            s0[st + 10] = dyn->a_vds[lane]; /* vdes */
            s0[st + 27] = 0.0;              /* qdef */
        }
    }

    v8d ceqjs = (cbs - gbs * vbs) * b->type;
    v8d ceqjd = (cbd - gbd * vbd) * b->type;

    /* Forward vs Reverse DC stamping terms */
    v8d Gm_st   = v8_sel(m_fwd, Gm, -Gm);
    v8d Gmbs_st = v8_sel(m_fwd, Gmb, -Gmb);
    v8d FwdSum  = v8_sel(m_fwd, Gm + Gmb, z0);
    v8d RevSum  = v8_sel(m_fwd, z0, -(Gm_st + Gmbs_st)); /* == Gm + Gmb when rev */

    v8d ceqdrn_f = b->type * (cdrain - Gds * vds - Gm * vgs - Gmb * vbs);
    v8d ceqdrn_r = -b->type * (cdrain + Gds * vds - Gm * vgd - Gmb * vbd);
    v8d ceqdrn = v8_sel(m_fwd, ceqdrn_f, ceqdrn_r);

    v8d ceqbd_f = b->type * (Isub + Igidl - (Gbd + Ggidld) * vds - (Gbg + Ggidlg) * vgs - (Gbb + Ggidlb) * vbs);
    v8d ceqbs_f = b->type * (Igisl + Ggisls * vds - Ggislg * vgd - Ggislb * vbd);
    v8d ceqbd_r = b->type * (Igidl - Ggidld * vds - Ggidlg * vgs - Ggidlb * vbs);
    v8d ceqbs_r = b->type * (Isub + Igisl + (Gbd + Ggisls) * vds - (Gbg + Ggislg) * vgd - (Gbb + Ggislb) * vbd);
    v8d ceqbd = v8_sel(m_fwd, ceqbd_f, ceqbd_r);
    v8d ceqbs = v8_sel(m_fwd, ceqbs_f, ceqbs_r);

    v8d gb_sum = Gbd + Gbg + Gbb;
    v8d gbbdp  = v8_sel(m_fwd, -Gbd, gb_sum);
    v8d gbbsp  = v8_sel(m_fwd, gb_sum, -Gbd);
    v8d gbdpg  = v8_sel(m_fwd, Gbg, z0);
    v8d gbdpdp = v8_sel(m_fwd, Gbd, z0);
    v8d gbdpb  = v8_sel(m_fwd, Gbb, z0);
    v8d gbdpsp = v8_sel(m_fwd, -gb_sum, z0);
    v8d gbspg  = v8_sel(m_fwd, z0, Gbg);
    v8d gbspdp = v8_sel(m_fwd, z0, -gb_sum);
    v8d gbspb  = v8_sel(m_fwd, z0, Gbb);
    v8d gbspsp = v8_sel(m_fwd, z0, Gbd);

    v8d ggidl_sum = Ggidlg + Ggidld + Ggidlb;
    v8d ggisl_sum = Ggislg + Ggisls + Ggislb;

    v8d vm = b->m;
    dyn->Cggb = Cggb; dyn->Cgdb = Cgdb; dyn->Cgsb = Cgsb;
    dyn->Cdgb = Cdgb; dyn->Cddb = Cddb; dyn->Cdsb = Cdsb;
    dyn->Csgb = Csgb; dyn->Csdb = Csdb; dyn->Cssb = Cssb;
    dyn->Cbgb = Cbgb; dyn->Cbdb = Cbdb; dyn->Cbsb = Cbsb;

    dyn->mdc44 = vm * (b->drainCond + Gds + gbd + RevSum + gbdpdp + Ggidld);
    dyn->mdc46 = vm * (Gm_st + gbdpg + Ggidlg);
    dyn->mdc47 = vm * (Gds + FwdSum - gbdpsp + ggidl_sum);
    dyn->mdc48 = vm * (gbd - Gmbs_st - gbdpb - Ggidlb);
    dyn->mdc51 = vm * (Gds + RevSum - gbspdp + ggisl_sum);
    dyn->mdc52 = vm * (-Gm_st + gbspg + Ggislg);
    dyn->mdc53 = vm * (b->sourceCond + Gds + gbs + FwdSum + gbspsp + Ggisls);
    dyn->mdc55 = vm * (gbs + Gmbs_st - gbspb - Ggislb);
    dyn->mdc58 = vm * (-gbd + gbbdp - Ggidld + ggisl_sum);
    dyn->mdc59 = vm * (-Gbg - Ggidlg - Ggislg);
    dyn->mdc60 = vm * (-gbs + gbbsp + ggidl_sum - Ggisls);
    dyn->mdc61 = vm * (gbd + gbs - Gbb - Ggidlb - Ggislb);

    dyn->mdc_rhsd = vm * (ceqjd - ceqbd - ceqdrn);
    dyn->mdc_rhsb = vm * (ceqbd + ceqbs - ceqjd - ceqjs);
    dyn->mdc_rhss = vm * (ceqdrn - ceqbs + ceqjs);

    if (!dc_only) {
        dyn->q_lin_g = -Cggb * vgb + Cgdb * vbd + Cgsb * vbs;
        dyn->q_lin_d = -Cdgb * vgb + Cddb * vbd + Cdsb * vbs;
        dyn->q_lin_b = -Cbgb * vgb + Cbdb * vbd + Cbsb * vbs;

        /* ===================================================================== */
        /* 13. Integrate Charges & Assemble Matrix/RHS Stamps in dyn->st_*       */
        /* ===================================================================== */
        bsim4_avx512_integrate_and_assemble(b, dyn, vds, vgs, vbs, 0,
                                            ag0, ag1, ag2, is_gear2, is_trap2, s0, s1, s2);
    } else {
        v8_store(dyn->st_rhsd, dyn->mdc_rhsd);
        v8_store(dyn->st_rhsg, z0);
        v8_store(dyn->st_rhsb, dyn->mdc_rhsb);
        v8_store(dyn->st_rhss, dyn->mdc_rhss);
        v8_store(dyn->st_v34, z0);
        v8_store(dyn->st_v35, z0);
        v8_store(dyn->st_v36, z0);
        v8_store(dyn->st_v37, z0);
        v8_store(dyn->st_v44, dyn->mdc44);
        v8_store(dyn->st_v46, dyn->mdc46);
        v8_store(dyn->st_v47, dyn->mdc47);
        v8_store(dyn->st_v48, dyn->mdc48);
        v8_store(dyn->st_v51, dyn->mdc51);
        v8_store(dyn->st_v52, dyn->mdc52);
        v8_store(dyn->st_v53, dyn->mdc53);
        v8_store(dyn->st_v55, dyn->mdc55);
        v8_store(dyn->st_v58, dyn->mdc58);
        v8_store(dyn->st_v59, dyn->mdc59);
        v8_store(dyn->st_v60, dyn->mdc60);
        v8_store(dyn->st_v61, dyn->mdc61);
    }

    /* Pre-assemble compact SoA convTest coefficients in dyn (exact match to b4v*cvtest.c) */
    {
        v8d Ggidls = -ggidl_sum;
        v8d Ggisld = -ggisl_sum;

        v8d id_f = cdrain + Isub - cbd + Igidl;
        v8d id_r = cdrain + cbd - Igidl;
        dyn->Idtot = v8_sel(m_fwd, id_f, id_r);
        dyn->cd_g  = v8_sel(m_fwd, Gm + Gbg + Ggidlg, Gm - Ggidlg);
        dyn->cd_b  = v8_sel(m_fwd, Gmb + Gbb + Ggidlb - gbd, gbd + Gmb - Ggidlb);
        dyn->cd_d  = v8_sel(m_fwd, Gds + Gbd + Ggidld + gbd, -(gbd + Gmb + Gm + Gds + Ggidls));

        dyn->Ibtot = cbs + cbd - Igidl - Igisl - Isub;
        dyn->cb_g  = -(Gbg + Ggidlg + Ggislg);
        dyn->cb_b  = gbd + gbs - (Gbb + Ggidlb + Ggislb);
        v8d cbd_f  = -gbd - (Gbd + Ggidld) + Ggislg + Ggislb + Ggisls;
        v8d cbd_r  = -gbd + (Gbb + Ggislb) + (Gbg + Ggislg) + (Gbd + Ggisld - Ggidls);
        dyn->cb_d  = v8_sel(m_fwd, cbd_f, cbd_r);
    }

    /* Only scatter 24 AoS instance fields when explicitly requested (e.g., .save @m[...]) */
    if (need_aos) {
        for (int lane = 0; lane < b->n_active; lane++)
            out->mode[lane] = (m_fwd & (1u << lane)) ? 1 : -1;
        v8_store(out->cd, cdrain);
        v8_store(out->gds, Gds);
        v8_store(out->gm, Gm);
        v8_store(out->gmbs, Gmb);
        v8_store(out->gbd, gbd);
        v8_store(out->gbs, gbs);
        v8_store(out->cbs, cbs);
        v8_store(out->cbd, cbd);
        v8_store(out->csub, Isub);
        v8_store(out->gbds, Gbd);
        v8_store(out->gbgs, Gbg);
        v8_store(out->gbbs, Gbb);
        v8_store(out->Igidl, Igidl);
        v8_store(out->ggidld, Ggidld);
        v8_store(out->ggidlg, Ggidlg);
        v8_store(out->ggidlb, Ggidlb);
        v8_store(out->Igisl, Igisl);
        v8_store(out->ggisls, Ggisls);
        v8_store(out->ggislg, Ggislg);
        v8_store(out->ggislb, Ggislb);
    }
}

#endif /* NGSPICE_OPT_ZEN4 */
#endif /* NGSPICE_BSIM4_AVX512_CORE_H */
