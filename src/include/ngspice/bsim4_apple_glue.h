/*
 * Per-version glue (BSIM4v5 / BSIM4v6) for Apple Silicon (ARM64 NEON) BSIM4 kernel.
 *
 * Features:
 *   - Deduplicates shared BSIM4ApplePParam per unique (model, pParam, channel params)
 *   - Per-instance (W=1) Level-1 / Level-2 stationary bypass over 128B BSIM4AppleCheck
 *   - Dynamic compaction of active misses into dense 2-wide NEON pairs sharing one pp
 *   - Adaptive incremental KLU delta-stamping (zero matrix/RHS stamps on L1 hits!)
 *   - Fast convTest and 2-wide NEON trunc over compact BSIM4AppleCheck
 */
#if NGSPICE_OPT_APPLE

#include "ngspice/bsim4_apple_core.h"

int B4P(AvxNeedAosSync) = 0;

static inline int
B4P(isAvxEligibleTran)(CKTcircuit *ckt)
{
    static int disabled = -1;
    if (disabled < 0) {
        disabled = (getenv("NGSPICE_NO_NEON") != NULL ||
                    getenv("NGSPICE_NO_SIMD") != NULL ||
                    getenv("NGSPICE_NO_AVX512") != NULL) ? 1 : 0;
    }
    if (disabled)
        return 0;

    long mode = ckt->CKTmode;
    if (!(mode & MODETRAN))
        return 0;
    if (mode & (MODEINITSMSIG | MODEINITTRAN | MODEINITJCT | MODEINITFIX | MODEDCTRANCURVE | MODEDC))
        return 0;
    if (ckt->CKTbypass)
        return 0;
    if ((ckt->CKTintegrateMethod != GEAR && ckt->CKTintegrateMethod != TRAPEZOIDAL) ||
        !(ckt->CKTorder == 1 || ckt->CKTorder == 2 ||
          (ckt->CKTintegrateMethod == GEAR && ckt->CKTorder == 3)))
        return 0;
    return 1;
}

static inline int
B4P(isAvxEligibleDcOp)(CKTcircuit *ckt)
{
    static int disabled = -1;
    if (disabled < 0) {
        disabled = (getenv("NGSPICE_NO_NEON") != NULL ||
                    getenv("NGSPICE_NO_SIMD") != NULL ||
                    getenv("NGSPICE_NO_AVX512") != NULL) ? 1 : 0;
    }
    if (disabled)
        return 0;

    long mode = ckt->CKTmode;
    if (!(mode & (MODEDCOP | MODETRANOP)))
        return 0;
    if ((mode & MODETRANOP) && ckt->CKTdiagGmin != 0.0)
        return 0;
    if (!(mode & (MODEINITFLOAT | MODEINITJCT | MODEINITFIX)))
        return 0;
    if (mode & (MODEINITSMSIG | MODEINITTRAN | MODEUIC | MODEDCTRANCURVE | MODEAC | MODETRAN))
        return 0;
    if (ckt->CKTbypass)
        return 0;
    return 1;
}

void
B4P(freeAvxLatent)(void *p)
{
    BSIM4AppleLatentState *lat = (BSIM4AppleLatentState *)p;
    if (!lat) return;
    if (getenv("NGSPICE_MR_STATS") != NULL && lat->statCalls > 0) {
        double avg_f = (double)lat->statFrozenSum / (double)lat->statCalls;
        double pct = 100.0 * avg_f / (double)(lat->nElig > 0 ? lat->nElig : 1);
        fprintf(stderr,
                "[MR_LATENT] insts=%d calls=%llu avg_frozen=%.1f (%.1f%%) freezes=%llu wakes=%llu resets=%llu fail(val=%llu icap=%llu cache=%llu snap=%llu)\n",
                lat->nElig, lat->statCalls, avg_f, pct,
                lat->statFreezes, lat->statWakes, lat->statResets,
                lat->statFailValid, lat->statFailIcap, lat->statFailCache, lat->statFailSnap);
    }
    free(lat->latentFrozen);
    free(lat->latentStreak);
    free(lat->instGroup);
    free(lat->activeList);
    free(lat->activePos);
    free(lat->grpActiveList);
    free(lat->grpActivePos);
    free(lat->grpActiveCnt);
    free(lat->nodeInstPtr);
    free(lat->nodeInstIdx);
    free(lat->vSnap);
    free(lat->nodeFrozenCnt);
    free(lat->cfFastList);
    free(lat->cfGrpStart);
    free(lat->cfGrpCnt);
    free(lat->cfIsFastInst);
    free(lat);
}

#ifdef KLU
static void B4P(latentUnfreezeAll)(B4P(model) *head, CKTcircuit *ckt, KLUmatrix *km);
#endif

static inline int
B4P(mrLatentEnabled)(void)
{
    static int init = 0, en = 1;
    if (!init) {
        en = 1;
        if (getenv("NGSPICE_NO_MR_LATENT") != NULL) {
            en = 0;
        } else {
            const char *v = getenv("NGSPICE_MR_LATENT");
            if (v != NULL && v[0] != '\0')
                en = (v[0] != '0');
        }
        init = 1;
    }
    return en;
}

static inline double
B4P(mrLatentVtol)(const CKTcircuit *ckt)
{
    static int init = 0, has_ov = 0;
    static double ov = 1.0e-5;
    if (!init) {
        const char *v = getenv("NGSPICE_MR_LATENT_VTOL");
        if (v && v[0]) {
            ov = atof(v);
            has_ov = 1;
        }
        init = 1;
    }
    if (has_ov) return ov;
    double vt = ckt->CKTvoltTol;
    double vcap = (ckt->CKTmaxEqNum >= 10000) ? 1.0e-4 : 1.0e-5;
    return (vt < vcap) ? vt : vcap;
}

static inline double
B4P(mrLatentItol)(const CKTcircuit *ckt)
{
    static int init = 0, has_ov = 0;
    static double ov = 0.0;
    if (!init) {
        const char *v = getenv("NGSPICE_MR_LATENT_ITOL");
        if (v && v[0]) {
            ov = atof(v);
            has_ov = 1;
        }
        init = 1;
    }
    if (has_ov) return ov;
    double base = 2.0 * ckt->CKTabstol;
    if (ckt->CKTmaxEqNum >= 10000 && ckt->CKTvoltTol > 2.0e-6)
        base *= (ckt->CKTvoltTol / 1.0e-6);
    return base;
}

static inline int
B4P(mrLatentStreakReq)(void)
{
    static int init = 0, req = 3;
    if (!init) {
        const char *v = getenv("NGSPICE_MR_LATENT_STREAK");
        if (v && v[0]) {
            int k = atoi(v);
            if (k >= 1 && k <= 20) req = k;
        }
        init = 1;
    }
    return req;
}

void
B4P(freeApplePack)(B4P(model) *head)
{
    BSIM4ApplePack *pack = (BSIM4ApplePack *)head->B4P(AvxBlocks);
    if (pack) {
        free(pack->groups);
        free(pack->checks);
        free(pack->stamps);
        free(pack->l2s);
        free(pack->jcts);
        free(pack->eval_q);
        free(pack->orig_order);
        free(pack->eval_vds);
        free(pack->eval_vgs);
        free(pack->eval_vbs);
        free(pack);
        head->B4P(AvxBlocks) = NULL;
    }
    B4P(freeAvxLatent)(head->B4P(AvxLatent));
    head->B4P(AvxLatent) = NULL;
    head->B4P(AvxDyn) = NULL;
    head->B4P(AvxDynValid) = 0;
    head->B4P(AvxBlockCount) = 0;
}

typedef struct {
    const char *mod_name;
    double    l, w;
    uintptr_t mod_ptr;
    uintptr_t pp_ptr;
    int       has_trap;
    double    nf;
    double    vth0, vfb, vfbzb, vbsc, k2ox, eta0, u0temp, vsattemp, vtfbphi2;
    int       orig_idx;
} B4P(GroupSortKey);

static int
B4P(cmpGroupKey)(const void *pa, const void *pb)
{
    const B4P(GroupSortKey) *a = (const B4P(GroupSortKey) *)pa;
    const B4P(GroupSortKey) *b = (const B4P(GroupSortKey) *)pb;
    if (a->mod_ptr != b->mod_ptr) {
        int mc = strcmp(a->mod_name, b->mod_name);
        if (mc != 0) return mc;
        return (a->mod_ptr < b->mod_ptr) ? -1 : 1;
    }
    if (a->l != b->l) return (a->l < b->l) ? -1 : 1;
    if (a->w != b->w) return (a->w < b->w) ? -1 : 1;
    if (a->pp_ptr  != b->pp_ptr)  return (a->pp_ptr  < b->pp_ptr)  ? -1 : 1;
    if (a->has_trap != b->has_trap) return a->has_trap - b->has_trap;
    if (a->nf != b->nf) return (a->nf < b->nf) ? -1 : 1;
    int c = memcmp(&a->vth0, &b->vth0, 9 * sizeof(double));
    if (c != 0) return c;
    return a->orig_idx - b->orig_idx;
}

static int
B4P(sameGroupKey)(const B4P(GroupSortKey) *a, const B4P(GroupSortKey) *b)
{
    return (a->mod_ptr == b->mod_ptr &&
            a->pp_ptr  == b->pp_ptr &&
            a->has_trap == b->has_trap &&
            a->nf == b->nf &&
            memcmp(&a->vth0, &b->vth0, 9 * sizeof(double)) == 0);
}

static inline int
B4P(ptrToKluOff)(const double *ptr, uintptr_t ax_addr, size_t ax_bytes, int nz)
{
    uintptr_t d = (uintptr_t)ptr - ax_addr;
    if (d < ax_bytes)
        return (int)(d / sizeof(double));
    return nz;
}

static void
B4P(fillApplePParam)(BSIM4ApplePParam *pp, B4P(instance) *h, B4P(model) *m, CKTcircuit *ckt)
{
    memset(pp, 0, sizeof(*pp));
#if B4_IS_V6
    struct bsim4v6SizeDependParam *p = h->pParam;
    pp->is_v6 = 1;
#else
    struct bsim4v5SizeDependParam *p = h->pParam;
    pp->is_v6 = 0;
#endif
    double leff = p->B4P(leff);
    double TempRatio = ckt->CKTtemp / m->B4P(tnom) - 1.0;

    pp->type        = (double)m->B4P(type);
    pp->nf          = h->B4P(nf);
    pp->f_nf_ne1    = (h->B4P(nf) != 1.0) ? 1 : 0;
    pp->vth0_signed = (double)m->B4P(type) * h->B4P(vth0);
    pp->vfb_plus_phi= h->B4P(vfb) + p->B4P(phi);
    pp->vfbzb       = h->B4P(vfbzb);
    pp->f_vfbzb_le0 = (h->B4P(vfbzb) <= 0.0) ? 1 : 0;
    pp->vbsc        = h->B4P(vbsc);
    pp->phi         = p->B4P(phi);
    pp->phi_095     = 0.95 * p->B4P(phi);
    pp->sqrtPhi     = p->B4P(sqrtPhi);
    pp->V0          = p->B4P(vbi) - p->B4P(phi);
    pp->vtm         = m->B4P(vtm);
    pp->coxe        = m->B4P(coxe);
    pp->invCoxe     = 1.0 / m->B4P(coxe);
    pp->coxp        = m->B4P(coxp);
    pp->toxe        = m->B4P(toxe);
    pp->invToxe     = 1.0 / m->B4P(toxe);
    pp->toxp        = m->B4P(toxp);
    pp->tox_ctm     = 1.0e8 * m->B4P(toxp);
    pp->inv_tox_ctm = 1.0 / (1.0e8 * m->B4P(toxp));
    pp->inv_2e8_toxp= 1.0 / (2.0e8 * m->B4P(toxp));
    pp->link_ctm    = 1.0e-3 * m->B4P(toxp);
    pp->leff        = leff;
    pp->invLeff     = 1.0 / leff;
    pp->sqrtLeff    = p->B4P(sqrtLeff);
    pp->weff        = p->B4P(weff);
    pp->weffCV      = p->B4P(weffCV);
    pp->leffCV      = p->B4P(leffCV);
    pp->CoxWL_ov_coxe = p->B4P(weffCV) * p->B4P(leffCV) * h->B4P(nf);
    pp->CoxWL       = m->B4P(coxe) * pp->CoxWL_ov_coxe;
#if B4_IS_V6
    pp->ados1p9     = 1.9e-9 * m->B4P(ados);
    pp->bdos0p7     = 0.7 * m->B4P(bdos);
#else
    pp->ados1p9     = 1.9e-9;
    pp->bdos0p7     = 0.7;
#endif
    pp->invNvtms    = 1.0 / (m->B4P(vtm) * m->B4P(SjctEmissionCoeff));
    pp->invNvtmd    = 1.0 / (m->B4P(vtm) * m->B4P(DjctEmissionCoeff));
    pp->vcrit       = m->B4P(vcrit);

    if (h->B4P(hasTrapTunnel)) {
        pp->has_trap_tunnel = 1;
#if B4_IS_V6
        double nvs    = m->B4P(vtm0) * m->B4P(njtsstemp);
        double nvd    = m->B4P(vtm0) * m->B4P(njtsdtemp);
        double nvsws  = m->B4P(vtm0) * m->B4P(njtsswstemp);
        double nvswd  = m->B4P(vtm0) * m->B4P(njtsswdtemp);
        double nvswgs = m->B4P(vtm0) * m->B4P(njtsswgstemp);
        double nvswgd = m->B4P(vtm0) * m->B4P(njtsswgdtemp);
#else
        double nvs    = m->B4P(vtm0) * m->B4P(njtstemp);
        double nvd    = nvs;
        double nvsws  = m->B4P(vtm0) * m->B4P(njtsswtemp);
        double nvswd  = nvsws;
        double nvswgs = m->B4P(vtm0) * m->B4P(njtsswgtemp);
        double nvswgd = nvswgs;
#endif
        pp->invNvtmrs_s    = (nvs != 0.0)    ? (1.0 / nvs)    : 0.0;
        pp->invNvtmrs_d    = (nvd != 0.0)    ? (1.0 / nvd)    : 0.0;
        pp->invNvtmrssw_s  = (nvsws != 0.0)  ? (1.0 / nvsws)  : 0.0;
        pp->invNvtmrssw_d  = (nvswd != 0.0)  ? (1.0 / nvswd)  : 0.0;
        pp->invNvtmrsswg_s = (nvswgs != 0.0) ? (1.0 / nvswgs) : 0.0;
        pp->invNvtmrsswg_d = (nvswgd != 0.0) ? (1.0 / nvswgd) : 0.0;
        pp->vtss    = m->B4P(vtss);
        pp->vtsd    = m->B4P(vtsd);
        pp->vtssws  = m->B4P(vtssws);
        pp->vtsswd  = m->B4P(vtsswd);
        pp->vtsswgs = m->B4P(vtsswgs);
        pp->vtsswgd = m->B4P(vtsswgd);
    } else {
        pp->vtss = pp->vtsd = pp->vtssws = pp->vtsswd = pp->vtsswgs = pp->vtsswgd = 10.0;
    }

    pp->xdep0OverSqrtPhi = p->B4P(xdep0OverSqrtPhi);
    pp->factor1          = m->B4P(factor1);
    pp->dvt0             = p->B4P(dvt0);
    pp->dvt1_leff        = p->B4P(dvt1) * leff;
    pp->dvt2             = p->B4P(dvt2);
    pp->dvt0w            = p->B4P(dvt0w);
    pp->dvt1w_weff_leff  = p->B4P(dvt1w) * p->B4P(weff) * leff;
    pp->dvt2w            = p->B4P(dvt2w);
    pp->f_dvt0w          = (p->B4P(dvt0w) != 0.0) ? 1 : 0;

    pp->k1           = p->B4P(k1);
    pp->k1ox         = p->B4P(k1ox);
    pp->f_k1ox_zero  = (p->B4P(k1ox) == 0.0) ? 1 : 0;
    pp->f_k1ox_le0   = (p->B4P(k1ox) <= 0.0) ? 1 : 0;
    pp->k2ox         = h->B4P(k2ox);
    pp->k3           = p->B4P(k3);
    pp->k3b          = p->B4P(k3b);
    pp->lpebFactor   = p->B4P(lpebFactor);
    pp->vthNarrowW   = p->B4P(vthNarrowW);
    pp->vth_const_T1 = (double)m->B4P(type) * h->B4P(vth0)
                     - p->B4P(k1) * p->B4P(sqrtPhi) * p->B4P(lpebFactor)
                     + p->B4P(k3) * p->B4P(vthNarrowW)
                     + p->B4P(k1ox) * (p->B4P(lpe0Factor) - 1.0) * p->B4P(sqrtPhi)
                     + (p->B4P(kt1) + p->B4P(kt1l) / leff) * TempRatio;
    pp->kt2_TempRatio= p->B4P(kt2) * TempRatio;
    pp->eta0         = h->B4P(eta0);
    pp->etab         = p->B4P(etab);
    pp->theta0vb0    = p->B4P(theta0vb0);
    pp->nfactor      = p->B4P(nfactor);
    pp->cit          = p->B4P(cit);
    pp->cdsc         = p->B4P(cdsc);
    pp->cdscb        = p->B4P(cdscb);
    pp->cdscd        = p->B4P(cdscd);
    pp->dvtp0        = p->B4P(dvtp0);
    pp->dvtp1        = p->B4P(dvtp1);
    pp->f_dvtp0      = (p->B4P(dvtp0) > 0.0) ? 1 : 0;

    if (p->B4P(ngate) > 1.0e18 && p->B4P(ngate) < 1.0e25) {
        pp->f_ngate_valid = 1;
        pp->polyT1 = 1.0e6 * CHARGE * EPSSI * p->B4P(ngate) / (m->B4P(coxe) * m->B4P(coxe));
    } else {
        pp->polyT1 = 1.0;
    }
    pp->mstar         = p->B4P(mstar);
    pp->voffcbn       = p->B4P(voffcbn);
    pp->coxe_ov_cdep0 = m->B4P(coxe) / p->B4P(cdep0);

    pp->dwg          = p->B4P(dwg);
    pp->dwb          = p->B4P(dwb);
    pp->prwg         = p->B4P(prwg);
    pp->prwb         = p->B4P(prwb);
    pp->rds0_half    = 0.5 * p->B4P(rds0);
    pp->rdswmin      = p->B4P(rdswmin);
    pp->xj           = p->B4P(xj);
    pp->a0           = p->B4P(a0);
    pp->b0_ov_wb1    = p->B4P(b0) / (p->B4P(weff) + p->B4P(b1));
    pp->ags_a0       = p->B4P(ags) * p->B4P(a0);
    pp->keta         = p->B4P(keta);
    pp->abulkCVfactor= p->B4P(abulkCVfactor);

    pp->ua       = p->B4P(ua);
    pp->ub       = p->B4P(ub);
    pp->uc       = p->B4P(uc);
    pp->ud       = p->B4P(ud);
    pp->has_ud   = (p->B4P(ud) != 0.0) ? 1 : 0;
    pp->u0temp   = h->B4P(u0temp);
    pp->vsattemp = h->B4P(vsattemp);
    pp->a1       = p->B4P(a1);
    pp->f_has_a1 = (p->B4P(a1) != 0.0) ? 1 : 0;
    pp->a2       = p->B4P(a2);
    pp->delta    = p->B4P(delta);
    pp->lambda   = p->B4P(lambda);
    pp->litl     = p->B4P(litl);
    pp->f_lambda = (m->B4P(lambdaGiven) && m->B4P(lambda) > 0.0) ? 1 : 0;
    pp->fprout   = p->B4P(fprout);
    pp->f_fprout = (p->B4P(fprout) > 0.0) ? 1 : 0;
    pp->pvag     = p->B4P(pvag);
    pp->pclm     = p->B4P(pclm);
    if (p->B4P(pclm) > B4V_MIN_EXP) {
        pp->f_pclm = 1;
        pp->inv_pclm_litl = 1.0 / (p->B4P(pclm) * p->B4P(litl));
    }
    pp->thetaRout = p->B4P(thetaRout);
    if (p->B4P(thetaRout) > B4V_MIN_EXP) {
        pp->f_thetaRout = 1;
        pp->inv_thetaRout = 1.0 / p->B4P(thetaRout);
    }
    pp->pdiblb      = p->B4P(pdiblb);
    pp->pdits       = p->B4P(pdits);
    pp->pditsd      = p->B4P(pditsd);
    pp->pditsl_term = 1.0 + m->B4P(pditsl) * leff;
    if (p->B4P(pdits) > B4V_MIN_EXP) {
        pp->f_pdits = 1;
        pp->inv_pdits = 1.0 / p->B4P(pdits);
    }
    pp->pscbe1_litl = p->B4P(pscbe1) * p->B4P(litl);
#if B4_IS_V6
    if (p->B4P(pscbe2) > 0.0 && p->B4P(pscbe1) >= 0.0) {
#else
    if (p->B4P(pscbe2) > 0.0) {
#endif
        pp->f_pscbe = 1;
        pp->inv_pscbe2 = 1.0 / p->B4P(pscbe2);
    }
    double atmp = p->B4P(alpha0) + p->B4P(alpha1) * leff;
    if (atmp > 0.0 && p->B4P(beta0) > 0.0) {
        pp->f_isub = 1;
        pp->alpha_ov_leff = atmp / leff;
    }
    pp->beta0 = p->B4P(beta0);
    if (m->B4P(vtlGiven) && m->B4P(vtl) > 0.0) {
        pp->f_vtl = 1;
        pp->inv_vtl_tfactor = 1.0 / (p->B4P(vtl) * p->B4P(tfactor));
    }
    pp->vtfbphi2 = h->B4P(vtfbphi2);

    pp->inv_3toxe    = 1.0 / (3.0 * m->B4P(toxe));
    pp->agidl_weffCJ = p->B4P(agidl) * p->B4P(weffCJ);
    pp->bgidl        = p->B4P(bgidl);
    pp->cgidl        = p->B4P(cgidl);
    pp->egidl        = p->B4P(egidl);
    pp->f_gidl_en    = (p->B4P(agidl) > 0.0 && p->B4P(bgidl) > 0.0 && p->B4P(cgidl) > 0.0) ? 1 : 0;
#if B4_IS_V6
    pp->agisl_weffCJ = p->B4P(agisl) * p->B4P(weffCJ);
    pp->bgisl        = p->B4P(bgisl);
    pp->cgisl        = p->B4P(cgisl);
    pp->egisl        = p->B4P(egisl);
    pp->f_gisl_en    = (p->B4P(agisl) > 0.0 && p->B4P(bgisl) > 0.0 && p->B4P(cgisl) > 0.0) ? 1 : 0;
#else
    pp->agisl_weffCJ = pp->agidl_weffCJ;
    pp->bgisl        = pp->bgidl;
    pp->cgisl        = pp->cgidl;
    pp->egisl        = pp->egidl;
    pp->f_gisl_en    = pp->f_gidl_en;
#endif

    pp->noff   = p->B4P(noff);
    pp->voffcv = p->B4P(voffcv);
    pp->acde   = p->B4P(acde);
    pp->ldeb   = p->B4P(ldeb);
    pp->moin   = p->B4P(moin);
    if (m->B4P(xpart) > 0.5) {
        pp->f_xpart_gt05 = 1;
        pp->xpart_mode = 1;
    } else if (m->B4P(xpart) < 0.5) {
        pp->f_xpart_lt05 = 1;
        pp->xpart_mode = 0;
    } else {
        pp->xpart_mode = 2;
    }

    pp->cgdo        = p->B4P(cgdo);
    pp->cgso        = p->B4P(cgso);
    pp->cgbo        = p->B4P(cgbo);
    pp->weffCV_cgdl = p->B4P(weffCV) * p->B4P(cgdl);
    pp->ckappad     = p->B4P(ckappad);
    pp->inv_ckappad = 1.0 / p->B4P(ckappad);
    pp->weffCV_cgsl = p->B4P(weffCV) * p->B4P(cgsl);
    pp->ckappas     = p->B4P(ckappas);
    pp->inv_ckappas = 1.0 / p->B4P(ckappas);

    pp->PhiBS    = m->B4P(PhiBS);    pp->invPhiBS    = 1.0 / m->B4P(PhiBS);
    pp->PhiBSWS  = m->B4P(PhiBSWS);  pp->invPhiBSWS  = 1.0 / m->B4P(PhiBSWS);
    pp->PhiBSWGS = m->B4P(PhiBSWGS); pp->invPhiBSWGS = 1.0 / m->B4P(PhiBSWGS);
    pp->PhiBD    = m->B4P(PhiBD);    pp->invPhiBD    = 1.0 / m->B4P(PhiBD);
    pp->PhiBSWD  = m->B4P(PhiBSWD);  pp->invPhiBSWD  = 1.0 / m->B4P(PhiBSWD);
    pp->PhiBSWGD = m->B4P(PhiBSWGD); pp->invPhiBSWGD = 1.0 / m->B4P(PhiBSWGD);

    double mjs    = m->B4P(SbulkJctBotGradingCoeff);
    double mjsws  = m->B4P(SbulkJctSideGradingCoeff);
    double mjswgs = m->B4P(SbulkJctGateSideGradingCoeff);
    double mjd    = m->B4P(DbulkJctBotGradingCoeff);
    double mjswd  = m->B4P(DbulkJctSideGradingCoeff);
    double mjswgd = m->B4P(DbulkJctGateSideGradingCoeff);
    pp->MJS    = mjs;    pp->inv1mMJS    = 1.0 / (1.0 - mjs);
    pp->MJSWS  = mjsws;  pp->inv1mMJSWS  = 1.0 / (1.0 - mjsws);
    pp->MJSWGS = mjswgs; pp->inv1mMJSWGS = 1.0 / (1.0 - mjswgs);
    pp->MJD    = mjd;    pp->inv1mMJD    = 1.0 / (1.0 - mjd);
    pp->MJSWD  = mjswd;  pp->inv1mMJSWD  = 1.0 / (1.0 - mjswd);
    pp->MJSWGD = mjswgd; pp->inv1mMJSWGD = 1.0 / (1.0 - mjswgd);
    pp->f_mjs_half    = (mjs    == 0.5) ? 1 : 0;
    pp->f_mjsws_half  = (mjsws  == 0.5) ? 1 : 0;
    pp->f_mjswgs_half = (mjswgs == 0.5) ? 1 : 0;
    pp->f_mjd_half    = (mjd    == 0.5) ? 1 : 0;
    pp->f_mjswd_half  = (mjswd  == 0.5) ? 1 : 0;
    pp->f_mjswgd_half = (mjswgd == 0.5) ? 1 : 0;
}

static void
B4P(ensureAvxPack)(B4P(model) *head, CKTcircuit *ckt)
{
    double *ax = NULL;
    unsigned int nz = 0;
#ifdef KLU
    if (ckt->CKTmatrix && ckt->CKTmatrix->CKTkluMODE &&
        ckt->CKTmatrix->SMPkluMatrix &&
        ckt->CKTmatrix->SMPkluMatrix->KLUmatrixIsComplex == KLUmatrixReal)
    {
        ax = ckt->CKTmatrix->SMPkluMatrix->KLUmatrixAx;
        nz = ckt->CKTmatrix->SMPkluMatrix->KLUmatrixNZ;
    }
#endif
    if (head->B4P(AvxBuilt) &&
        head->B4P(AvxBuiltTemp) == ckt->CKTtemp &&
        head->B4P(AvxKluAxBase) == ax &&
        head->B4P(AvxKluNz) == nz)
        return;

#ifdef KLU
    if (head->B4P(AvxLatent) && ckt->CKTmatrix && ckt->CKTmatrix->CKTkluMODE &&
        ckt->CKTmatrix->SMPkluMatrix)
        B4P(latentUnfreezeAll)(head, ckt, ckt->CKTmatrix->SMPkluMatrix);
#endif
    B4P(freeApplePack)(head);
    FREE(head->B4P(AvxFallbackIdx));
    head->B4P(AvxFallbackCount) = 0;

    int count = head->B4P(InstCount);
    B4P(instance) **InstArray = head->B4P(InstanceArray);
    if (count <= 0) {
        head->B4P(AvxBuiltTemp) = ckt->CKTtemp;
        head->B4P(AvxKluAxBase) = ax;
        head->B4P(AvxKluNz) = nz;
        head->B4P(AvxBuilt) = 1;
        return;
    }

    B4P(GroupSortKey) *keys = TMALLOC(B4P(GroupSortKey), count);
    int *fbIdx = TMALLOC(int, count);
    int nElig = 0, nFb = 0;

    for (int idx = 0; idx < count; idx++) {
        B4P(instance) *h = InstArray[idx];
        B4P(model) *m = B4P(modPtr)(h);
        int ok = (h->B4P(rgateMod) == 0 &&
                  h->B4P(rbodyMod) == 0 &&
                  m->B4P(rdsMod) == 0 &&
                  h->B4P(trnqsMod) == 0 &&
                  h->B4P(acnqsMod) == 0 &&
                  m->B4P(igcMod) == 0 &&
                  m->B4P(igbMod) == 0 &&
                  m->B4P(mobMod) == 0 &&
                  m->B4P(capMod) == 2 &&
                  m->B4P(dioMod) == 1 &&
                  m->B4P(xpart) >= 0.0 &&
                  h->B4P(off) == 0 &&
                  m->B4P(tempMod) < 2);
#if B4_IS_V6
        ok = ok && (m->B4P(mtrlMod) == 0) && (m->B4P(cvchargeMod) == 0);
#endif
        if (ok) {
            B4P(GroupSortKey) *k = &keys[nElig++];
            k->mod_name = m->gen.GENmodName;
            k->l        = h->B4P(l);
            k->w        = h->B4P(w);
            k->mod_ptr  = (uintptr_t)m;
            k->pp_ptr   = (uintptr_t)h->pParam;
            k->has_trap = h->B4P(hasTrapTunnel) ? 1 : 0;
            k->nf       = h->B4P(nf);
            k->vth0     = h->B4P(vth0);
            k->vfb      = h->B4P(vfb);
            k->vfbzb    = h->B4P(vfbzb);
            k->vbsc     = h->B4P(vbsc);
            k->k2ox     = h->B4P(k2ox);
            k->eta0     = h->B4P(eta0);
            k->u0temp   = h->B4P(u0temp);
            k->vsattemp = h->B4P(vsattemp);
            k->vtfbphi2 = h->B4P(vtfbphi2);
            k->orig_idx = idx;
        } else {
            fbIdx[nFb++] = idx;
        }
    }

    if (nElig > 1) {
        qsort(keys, (size_t)nElig, sizeof(B4P(GroupSortKey)), B4P(cmpGroupKey));
    }

    int nGroups = 0;
    for (int i = 0; i < nElig; i++) {
        if (i == 0 || !B4P(sameGroupKey)(&keys[i], &keys[i - 1]))
            nGroups++;
    }

    BSIM4ApplePack *pack = (BSIM4ApplePack *)calloc(1, sizeof(BSIM4ApplePack));
    int alloc_ok = (pack != NULL);
    if (alloc_ok && nElig > 0) {
        if (posix_memalign((void **)&pack->groups, 128, (size_t)nGroups * sizeof(BSIM4AppleGroup)) != 0) alloc_ok = 0;
        if (alloc_ok && posix_memalign((void **)&pack->checks, 128, (size_t)nElig * sizeof(BSIM4AppleCheck)) != 0) alloc_ok = 0;
        if (alloc_ok && posix_memalign((void **)&pack->stamps, 64, (size_t)nElig * sizeof(BSIM4AppleStamp)) != 0) alloc_ok = 0;
        if (alloc_ok && posix_memalign((void **)&pack->l2s, 64, (size_t)nElig * sizeof(BSIM4AppleL2)) != 0) alloc_ok = 0;
        if (alloc_ok && posix_memalign((void **)&pack->jcts, 64, (size_t)nElig * sizeof(BSIM4AppleJct)) != 0) alloc_ok = 0;
        if (alloc_ok) {
            pack->eval_q     = (int *)malloc((size_t)(nElig + 2) * sizeof(int));
            pack->orig_order = (nFb == 0) ? (int *)malloc((size_t)(nElig + 2) * sizeof(int)) : NULL;
            pack->eval_vds   = (double *)malloc((size_t)(nElig + 2) * sizeof(double));
            pack->eval_vgs   = (double *)malloc((size_t)(nElig + 2) * sizeof(double));
            pack->eval_vbs   = (double *)malloc((size_t)(nElig + 2) * sizeof(double));
            if (!pack->eval_q || (nFb == 0 && !pack->orig_order) ||
                !pack->eval_vds || !pack->eval_vgs || !pack->eval_vbs)
                alloc_ok = 0;
        }
    }

    if (!alloc_ok) {
        if (pack) {
            free(pack->groups); free(pack->checks); free(pack->stamps);
            free(pack->l2s); free(pack->jcts);
            free(pack->eval_q); free(pack->orig_order);
            free(pack->eval_vds); free(pack->eval_vgs); free(pack->eval_vbs);
            free(pack);
        }
        FREE(keys);
        for (int idx = 0; idx < count; idx++) fbIdx[idx] = idx;
        head->B4P(AvxFallbackIdx) = fbIdx;
        head->B4P(AvxFallbackCount) = count;
        head->B4P(AvxBuiltTemp) = ckt->CKTtemp;
        head->B4P(AvxKluAxBase) = ax;
        head->B4P(AvxKluNz) = nz;
        head->B4P(AvxBuilt) = 1;
        return;
    }

    if (nElig > 0) {
        memset(pack->checks, 0, (size_t)nElig * sizeof(BSIM4AppleCheck));
        memset(pack->stamps, 0, (size_t)nElig * sizeof(BSIM4AppleStamp));
        memset(pack->l2s,    0, (size_t)nElig * sizeof(BSIM4AppleL2));
        memset(pack->jcts,   0, (size_t)nElig * sizeof(BSIM4AppleJct));
    }

    pack->nGroups = nGroups;
    pack->nElig   = nElig;

    uintptr_t ax_addr = (uintptr_t)ax;
    size_t ax_bytes = (size_t)nz * sizeof(double);

    int gi = -1;
    for (int i = 0; i < nElig; i++) {
        B4P(instance) *h = InstArray[keys[i].orig_idx];
        B4P(model) *m = B4P(modPtr)(h);
#if B4_IS_V6
        struct bsim4v6SizeDependParam *p = h->pParam;
#else
        struct bsim4v5SizeDependParam *p = h->pParam;
#endif
        if (i == 0 || !B4P(sameGroupKey)(&keys[i], &keys[i - 1])) {
            gi++;
            pack->groups[gi].offset = i;
            pack->groups[gi].count  = 0;
            B4P(fillApplePParam)(&pack->groups[gi].pp, h, m, ckt);
        }
        pack->groups[gi].count++;

        BSIM4AppleCheck *c  = &pack->checks[i];
        BSIM4AppleStamp *sp = &pack->stamps[i];
        BSIM4AppleL2    *l2 = &pack->l2s[i];
        BSIM4AppleJct   *jc = &pack->jcts[i];

        c->dNodePrime = h->B4P(dNodePrime);
        c->gNodePrime = h->B4P(gNodePrime);
        c->sNodePrime = h->B4P(sNodePrime);
        c->bNodePrime = h->B4P(bNodePrime);
        c->stateBase  = h->B4P(states);
        c->instIdx    = keys[i].orig_idx;
        if (pack->orig_order)
            pack->orig_order[keys[i].orig_idx] = i;
        c->has_dint   = (h->B4P(dNode) != h->B4P(dNodePrime)) ? 1 : 0;
        c->has_sint   = (h->B4P(sNode) != h->B4P(sNodePrime)) ? 1 : 0;
        c->type       = (double)m->B4P(type);

        sp->m          = h->B4P(m);
        sp->drainCond  = h->B4P(drainConductance);
        sp->sourceCond = h->B4P(sourceConductance);
        sp->mgd        = h->B4P(m) * h->B4P(drainConductance);
        sp->mgs        = h->B4P(m) * h->B4P(sourceConductance);
        sp->vcrit      = m->B4P(vcrit);
        sp->von        = h->B4P(von);

        l2->m     = h->B4P(m);
        l2->mtype = h->B4P(m) * (double)m->B4P(type);

        if (ax != NULL && nz > 0) {
            sp->kluOff[0]  = B4P(ptrToKluOff)(h->B4P(GPgpPtr), ax_addr, ax_bytes, (int)nz);
            sp->kluOff[1]  = B4P(ptrToKluOff)(h->B4P(GPdpPtr), ax_addr, ax_bytes, (int)nz);
            sp->kluOff[2]  = B4P(ptrToKluOff)(h->B4P(GPspPtr), ax_addr, ax_bytes, (int)nz);
            sp->kluOff[3]  = B4P(ptrToKluOff)(h->B4P(GPbpPtr), ax_addr, ax_bytes, (int)nz);
            sp->kluOff[4]  = B4P(ptrToKluOff)(h->B4P(DPdpPtr), ax_addr, ax_bytes, (int)nz);
            sp->kluOff[5]  = B4P(ptrToKluOff)(h->B4P(DPgpPtr), ax_addr, ax_bytes, (int)nz);
            sp->kluOff[6]  = B4P(ptrToKluOff)(h->B4P(DPspPtr), ax_addr, ax_bytes, (int)nz);
            sp->kluOff[7]  = B4P(ptrToKluOff)(h->B4P(DPbpPtr), ax_addr, ax_bytes, (int)nz);
            sp->kluOff[8]  = B4P(ptrToKluOff)(h->B4P(SPdpPtr), ax_addr, ax_bytes, (int)nz);
            sp->kluOff[9]  = B4P(ptrToKluOff)(h->B4P(SPgpPtr), ax_addr, ax_bytes, (int)nz);
            sp->kluOff[10] = B4P(ptrToKluOff)(h->B4P(SPspPtr), ax_addr, ax_bytes, (int)nz);
            sp->kluOff[11] = B4P(ptrToKluOff)(h->B4P(SPbpPtr), ax_addr, ax_bytes, (int)nz);
            sp->kluOff[12] = B4P(ptrToKluOff)(h->B4P(BPdpPtr), ax_addr, ax_bytes, (int)nz);
            sp->kluOff[13] = B4P(ptrToKluOff)(h->B4P(BPgpPtr), ax_addr, ax_bytes, (int)nz);
            sp->kluOff[14] = B4P(ptrToKluOff)(h->B4P(BPspPtr), ax_addr, ax_bytes, (int)nz);
            sp->kluOff[15] = B4P(ptrToKluOff)(h->B4P(BPbpPtr), ax_addr, ax_bytes, (int)nz);
        }

        /* Per-instance junction parameters */
        uint16_t jf = 0;
        jc->SourceSatCurrent = h->B4P(SourceSatCurrent);
        jc->DrainSatCurrent  = h->B4P(DrainSatCurrent);
        if (jc->SourceSatCurrent > 0.0) jf |= B4A_JF_SRC_SAT;
        if (jc->DrainSatCurrent  > 0.0) jf |= B4A_JF_DRN_SAT;
        jc->vjsmFwd  = h->B4P(vjsmFwd);
        jc->vjdmFwd  = h->B4P(vjdmFwd);
        jc->IVjsmFwd = h->B4P(IVjsmFwd);
        jc->IVjdmFwd = h->B4P(IVjdmFwd);

        double czbs    = m->B4P(SunitAreaTempJctCap) * h->B4P(Aseff);
        double czbssw  = m->B4P(SunitLengthSidewallTempJctCap) * h->B4P(Pseff);
        double czbsswg = m->B4P(SunitLengthGateSidewallTempJctCap) * p->B4P(weffCJ) * h->B4P(nf);
        double czbd    = m->B4P(DunitAreaTempJctCap) * h->B4P(Adeff);
        double czbdsw  = m->B4P(DunitLengthSidewallTempJctCap) * h->B4P(Pdeff);
        double czbdswg = m->B4P(DunitLengthGateSidewallTempJctCap) * p->B4P(weffCJ) * h->B4P(nf);
        jc->czbs = czbs; jc->czbssw = czbssw; jc->czbsswg = czbsswg;
        jc->czbd = czbd; jc->czbdsw = czbdsw; jc->czbdswg = czbdswg;
        if (czbs    > 0.0) jf |= B4A_JF_CZBS;
        if (czbssw  > 0.0) jf |= B4A_JF_CZBSSW;
        if (czbsswg > 0.0) jf |= B4A_JF_CZBSSWG;
        if (czbd    > 0.0) jf |= B4A_JF_CZBD;
        if (czbdsw  > 0.0) jf |= B4A_JF_CZBDSW;
        if (czbdswg > 0.0) jf |= B4A_JF_CZBDSWG;

        const BSIM4ApplePParam *pp = &pack->groups[gi].pp;
        jc->jctS_sum = czbs + czbssw + czbsswg;
        jc->jctD_sum = czbd + czbdsw + czbdswg;
        jc->jctS_lin = czbs * pp->MJS * pp->invPhiBS
                     + czbssw * pp->MJSWS * pp->invPhiBSWS
                     + czbsswg * pp->MJSWGS * pp->invPhiBSWGS;
        jc->jctD_lin = czbd * pp->MJD * pp->invPhiBD
                     + czbdsw * pp->MJSWD * pp->invPhiBSWD
                     + czbdswg * pp->MJSWGD * pp->invPhiBSWGD;

        if (h->B4P(hasTrapTunnel)) {
            jc->SjctRevSat = h->B4P(SjctTempRevSatCur);
            jc->DjctRevSat = h->B4P(DjctTempRevSatCur);
            jc->SswRevSat  = h->B4P(SswTempRevSatCur);
            jc->DswRevSat  = h->B4P(DswTempRevSatCur);
            jc->SswgRevSat = h->B4P(SswgTempRevSatCur);
            jc->DswgRevSat = h->B4P(DswgTempRevSatCur);
            if (jc->SjctRevSat != 0.0) jf |= B4A_JF_TT_S;
            if (jc->DjctRevSat != 0.0) jf |= B4A_JF_TT_D;
            if (jc->SswRevSat  != 0.0) jf |= B4A_JF_TT_SSW;
            if (jc->DswRevSat  != 0.0) jf |= B4A_JF_TT_DSW;
            if (jc->SswgRevSat != 0.0) jf |= B4A_JF_TT_SSWG;
            if (jc->DswgRevSat != 0.0) jf |= B4A_JF_TT_DSWG;
        }
        jc->flags = jf;
    }

    FREE(keys);
    head->B4P(AvxBlocks) = (void *)pack;
    head->B4P(AvxDyn) = NULL;
    head->B4P(AvxLatent) = NULL;
    head->B4P(AvxDynValid) = 0;
    head->B4P(AvxBlockCount) = nGroups;
    if (nFb > 0) {
        head->B4P(AvxFallbackIdx) = fbIdx;
        head->B4P(AvxFallbackCount) = nFb;
    } else {
        FREE(fbIdx);
        head->B4P(AvxFallbackIdx) = NULL;
        head->B4P(AvxFallbackCount) = 0;
    }
    head->B4P(AvxBuiltTemp) = ckt->CKTtemp;
    head->B4P(AvxKluAxBase) = ax;
    head->B4P(AvxKluNz) = nz;
    head->B4P(AvxBuilt) = 1;
}

static inline void
B4P(syncAvxDynFromInst)(B4P(model) *head)
{
    if (head->B4P(AvxDynValid) || !head->B4P(AvxBuilt))
        return;
    BSIM4ApplePack *pack = (BSIM4ApplePack *)head->B4P(AvxBlocks);
    if (!pack) return;
    B4P(instance) **InstArray = head->B4P(InstanceArray);
    int nElig = pack->nElig;
    for (int i = 0; i < nElig; i++) {
        BSIM4AppleCheck *c  = &pack->checks[i];
        BSIM4AppleStamp *sp = &pack->stamps[i];
        sp->von = InstArray[c->instIdx]->B4P(von);
        c->phys_valid = 0;
        c->stamp_valid = 0;
        c->last_active_step = -100;
    }
    pack->step_valid = 0;
    head->B4P(AvxDynValid) = 1;
}

static inline double
B4P(avxBypassTol)(const CKTcircuit *ckt)
{
    static int init = 0;
    static int mode = 0;
    static double override_tol = 1.0e-6;
    if (!init) {
        if (getenv("NGSPICE_NO_BLOCK_BYPASS") != NULL) {
            mode = 1;
        } else {
            const char *env = getenv("NGSPICE_AVX_BYPASS_TOL");
            if (env && env[0]) {
                override_tol = atof(env);
                mode = 2;
            }
        }
        init = 1;
    }
    if (mode == 1) return -1.0;
    if (mode == 2) return override_tol;
    double vt = ckt->CKTvoltTol;
    return (vt < 1.0e-6) ? vt : 1.0e-6;
}

static inline int
B4P(instCacheMatch)(
    const BSIM4AppleCheck * restrict c,
    double vds, double vgs, double vbs,
    double vtol, double reltol, double abstol,
    int *exact_zero_out)
{
    double delvds = vds - c->last_vds;
    double delvgs = vgs - c->last_vgs;
    double delvbs = vbs - c->last_vbs;
    double a0 = fabs(delvds), a1 = fabs(delvgs), a2 = fabs(delvbs);
    double dv = (a0 > a1) ? a0 : a1;
    if (a2 > dv) dv = a2;
    if (dv > vtol)
        return 0;
    if ((vds >= 0.0) != (c->last_vds >= 0.0))
        return 0;
    double r10 = 0.1 * reltol;
    double a10 = 0.1 * abstol;
    double cd_diff = c->cd_g * delvgs + c->cd_d * delvds + c->cd_b * delvbs;
    double cdhat   = c->Idtot + cd_diff;
    double m0 = fabs(cdhat), m1 = fabs(c->Idtot);
    double tol0 = r10 * ((m0 > m1) ? m0 : m1) + a10;
    if (fabs(cd_diff) >= tol0)
        return 0;
    double cb_diff = c->cb_g * delvgs + c->cb_d * delvds + c->cb_b * delvbs;
    double cbhat   = c->Ibtot + cb_diff;
    double m2 = fabs(cbhat), m3 = fabs(c->Ibtot);
    double tol6 = r10 * ((m2 > m3) ? m2 : m3) + a10;
    if (fabs(cb_diff) >= tol6)
        return 0;
    *exact_zero_out = (dv <= 1.0e-13);
    return 1;
}

static inline void
B4P(stampOneDirect)(
    const BSIM4AppleCheck * restrict c,
    const BSIM4AppleStamp * restrict sp,
    B4P(instance) **InstArray,
    double * restrict my_ax,
    double * restrict my_rhs,
    int stamp_mode)
{
    const double * restrict st = sp->st;
    if (stamp_mode == 1) {
        const int * restrict off = sp->kluOff;
        int dn = c->dNodePrime;
        int gn = c->gNodePrime;
        int sn = c->sNodePrime;
        int bn = c->bNodePrime;
        if (dn) my_rhs[dn] += st[0];
        if (gn) my_rhs[gn] -= st[1];
        if (bn) my_rhs[bn] += st[2];
        if (sn) my_rhs[sn] += st[3];
        /* 4-way ground-skip stamping: eliminates serialized my_ax[inz] store-forwarding stalls */
        if (sn != 0 && bn != 0) {
            my_ax[off[0]]  += st[4];
            my_ax[off[1]]  += st[5];
            my_ax[off[2]]  += st[6];
            my_ax[off[3]]  += st[7];
            my_ax[off[4]]  += st[8];
            my_ax[off[5]]  += st[9];
            my_ax[off[6]]  -= st[10];
            my_ax[off[7]]  -= st[11];
            my_ax[off[8]]  -= st[12];
            my_ax[off[9]]  += st[13];
            my_ax[off[10]] += st[14];
            my_ax[off[11]] -= st[15];
            my_ax[off[12]] += st[16];
            my_ax[off[13]] += st[17];
            my_ax[off[14]] += st[18];
            my_ax[off[15]] += st[19];
        } else if (sn != 0) {
            my_ax[off[0]]  += st[4];
            my_ax[off[1]]  += st[5];
            my_ax[off[2]]  += st[6];
            my_ax[off[4]]  += st[8];
            my_ax[off[5]]  += st[9];
            my_ax[off[6]]  -= st[10];
            my_ax[off[8]]  -= st[12];
            my_ax[off[9]]  += st[13];
            my_ax[off[10]] += st[14];
        } else if (bn != 0) {
            my_ax[off[0]]  += st[4];
            my_ax[off[1]]  += st[5];
            my_ax[off[3]]  += st[7];
            my_ax[off[4]]  += st[8];
            my_ax[off[5]]  += st[9];
            my_ax[off[7]]  -= st[11];
            my_ax[off[12]] += st[16];
            my_ax[off[13]] += st[17];
            my_ax[off[15]] += st[19];
        } else {
            my_ax[off[0]]  += st[4];
            my_ax[off[1]]  += st[5];
            my_ax[off[4]]  += st[8];
            my_ax[off[5]]  += st[9];
        }
        if (__builtin_expect(c->has_dint | c->has_sint, 0)) {
            B4P(instance) *here = InstArray[c->instIdx];
            if (c->has_dint) {
                double mgd = sp->mgd;
                *(here->B4P(DPdPtr)) -= mgd;
                *(here->B4P(DdpPtr)) -= mgd;
                *(here->B4P(DdPtr))  += mgd;
            }
            if (c->has_sint) {
                double mgs = sp->mgs;
                *(here->B4P(SPsPtr)) -= mgs;
                *(here->B4P(SspPtr)) -= mgs;
                *(here->B4P(SsPtr))  += mgs;
            }
        }
    } else if (stamp_mode == 2) {
        B4P(instance) *here = InstArray[c->instIdx];
        int dn = c->dNodePrime;
        int gn = c->gNodePrime;
        int sn = c->sNodePrime;
        int bn = c->bNodePrime;
        if (dn) my_rhs[dn] += st[0];
        if (gn) my_rhs[gn] -= st[1];
        if (bn) my_rhs[bn] += st[2];
        if (sn) my_rhs[sn] += st[3];
        if (sn != 0 && bn != 0) {
            *(here->B4P(GPgpPtr)) += st[4];
            *(here->B4P(GPdpPtr)) += st[5];
            *(here->B4P(GPspPtr)) += st[6];
            *(here->B4P(GPbpPtr)) += st[7];
            *(here->B4P(DPdpPtr)) += st[8];
            *(here->B4P(DPgpPtr)) += st[9];
            *(here->B4P(DPspPtr)) -= st[10];
            *(here->B4P(DPbpPtr)) -= st[11];
            *(here->B4P(SPdpPtr)) -= st[12];
            *(here->B4P(SPgpPtr)) += st[13];
            *(here->B4P(SPspPtr)) += st[14];
            *(here->B4P(SPbpPtr)) -= st[15];
            *(here->B4P(BPdpPtr)) += st[16];
            *(here->B4P(BPgpPtr)) += st[17];
            *(here->B4P(BPspPtr)) += st[18];
            *(here->B4P(BPbpPtr)) += st[19];
        } else if (sn != 0) {
            *(here->B4P(GPgpPtr)) += st[4];
            *(here->B4P(GPdpPtr)) += st[5];
            *(here->B4P(GPspPtr)) += st[6];
            *(here->B4P(DPdpPtr)) += st[8];
            *(here->B4P(DPgpPtr)) += st[9];
            *(here->B4P(DPspPtr)) -= st[10];
            *(here->B4P(SPdpPtr)) -= st[12];
            *(here->B4P(SPgpPtr)) += st[13];
            *(here->B4P(SPspPtr)) += st[14];
        } else if (bn != 0) {
            *(here->B4P(GPgpPtr)) += st[4];
            *(here->B4P(GPdpPtr)) += st[5];
            *(here->B4P(GPbpPtr)) += st[7];
            *(here->B4P(DPdpPtr)) += st[8];
            *(here->B4P(DPgpPtr)) += st[9];
            *(here->B4P(DPbpPtr)) -= st[11];
            *(here->B4P(BPdpPtr)) += st[16];
            *(here->B4P(BPgpPtr)) += st[17];
            *(here->B4P(BPbpPtr)) += st[19];
        } else {
            *(here->B4P(GPgpPtr)) += st[4];
            *(here->B4P(GPdpPtr)) += st[5];
            *(here->B4P(DPdpPtr)) += st[8];
            *(here->B4P(DPgpPtr)) += st[9];
        }
        if (__builtin_expect(c->has_dint, 0)) {
            double mgd = sp->mgd;
            *(here->B4P(DPdPtr)) -= mgd;
            *(here->B4P(DdpPtr)) -= mgd;
            *(here->B4P(DdPtr))  += mgd;
        }
        if (__builtin_expect(c->has_sint, 0)) {
            double mgs = sp->mgs;
            *(here->B4P(SPsPtr)) -= mgs;
            *(here->B4P(SspPtr)) -= mgs;
            *(here->B4P(SsPtr))  += mgs;
        }
    } else {
        B4P(instance) *here = InstArray[c->instIdx];
        here->B4P(rhsdPrime) = st[0];
        here->B4P(rhsgPrime) = st[1];
        here->B4P(rhsbPrime) = st[2];
        here->B4P(rhssPrime) = st[3];
        here->B4P(_34) = st[4];
        here->B4P(_35) = st[5];
        here->B4P(_36) = st[6];
        here->B4P(_37) = st[7];
        here->B4P(_44) = st[8];
        here->B4P(_46) = st[9];
        here->B4P(_47) = st[10];
        here->B4P(_48) = st[11];
        if (__builtin_expect(c->has_dint, 0)) {
            double mgd = sp->mgd;
            here->B4P(_45) = mgd;
            here->B4P(_49) = mgd;
            here->B4P(_50) = mgd;
        }
        here->B4P(_51) = st[12];
        here->B4P(_52) = st[13];
        here->B4P(_53) = st[14];
        here->B4P(_55) = st[15];
        if (__builtin_expect(c->has_sint, 0)) {
            double mgs = sp->mgs;
            here->B4P(_54) = mgs;
            here->B4P(_56) = mgs;
            here->B4P(_57) = mgs;
        }
        here->B4P(_58) = st[16];
        here->B4P(_59) = st[17];
        here->B4P(_60) = st[18];
        here->B4P(_61) = st[19];
    }
}

static inline void
B4P(scatterAosOne)(B4P(instance) *here, const BSIM4AppleOut *out, int lane)
{
    here->B4P(von)    = out->von[lane];
    here->B4P(mode)   = out->mode[lane];
    here->B4P(cd)     = out->cd[lane];
    here->B4P(gds)    = out->gds[lane];
    here->B4P(gm)     = out->gm[lane];
    here->B4P(gmbs)   = out->gmbs[lane];
    here->B4P(gbd)    = out->gbd[lane];
    here->B4P(gbs)    = out->gbs[lane];
    here->B4P(cbs)    = out->cbs[lane];
    here->B4P(cbd)    = out->cbd[lane];
    here->B4P(csub)   = out->csub[lane];
    here->B4P(gbds)   = out->gbds[lane];
    here->B4P(gbgs)   = out->gbgs[lane];
    here->B4P(gbbs)   = out->gbbs[lane];
    here->B4P(Igidl)  = out->Igidl[lane];
    here->B4P(ggidld) = out->ggidld[lane];
    here->B4P(ggidlg) = out->ggidlg[lane];
    here->B4P(ggidlb) = out->ggidlb[lane];
    here->B4P(ggidls) = -(out->ggidld[lane] + out->ggidlg[lane] + out->ggidlb[lane]);
    here->B4P(Igisl)  = out->Igisl[lane];
    here->B4P(ggisls) = out->ggisls[lane];
    here->B4P(ggislg) = out->ggislg[lane];
    here->B4P(ggislb) = out->ggislb[lane];
    here->B4P(ggisld) = -(out->ggisls[lane] + out->ggislg[lane] + out->ggislb[lane]);
}

#ifdef KLU
static inline int
B4P(collectUniqueNodesInst)(const BSIM4AppleCheck * restrict c, int N, int out[4])
{
    int nu = 0;
    int nd[4] = { c->dNodePrime, c->gNodePrime, c->sNodePrime, c->bNodePrime };
    for (int t = 0; t < 4; t++) {
        int u = nd[t];
        if (u <= 0 || u >= N) continue;
        int dup = 0;
        for (int k = 0; k < nu; k++) {
            if (out[k] == u) { dup = 1; break; }
        }
        if (!dup) out[nu++] = u;
    }
    return nu;
}

static inline void
B4P(stampFrozenDelta)(
    const BSIM4AppleCheck * restrict c,
    const BSIM4AppleStamp * restrict sp,
    const BSIM4AppleL2 * restrict l2,
    B4P(instance) **InstArray,
    KLUmatrix *km,
    double ag0,
    double sign)
{
    int inz = (int)km->KLUmatrixNZ;
    double * restrict ldc  = km->KLUmatrixAxLatentDC;
    double * restrict lcap = km->KLUmatrixAxLatentCap;
    double * restrict rdc  = km->KLUmatrixRhsLatentDC;
    double * restrict rcap = km->KLUmatrixRhsLatentCap;
    int upd_rcb = (km->KLUmatrixResCapValid && km->KLUmatrixResCapHasLatent &&
                   km->KLUmatrixResCapAg0 == ag0 && km->KLUmatrixAxResCapBase != NULL);
    if (!upd_rcb)
        km->KLUmatrixResCapValid = 0;
    double * restrict rcb = upd_rcb ? km->KLUmatrixAxResCapBase : NULL;

    double vm    = l2->m * sign;
    double mtype = l2->mtype * sign;

    double u_ceqqg = l2->q_lin_g * mtype;
    double u_ceqqd = l2->q_lin_d * mtype;
    double u_ceqqb = l2->q_lin_b * mtype;

    double drhsd = l2->mdc_rhsd * sign;
    double drhsb = l2->mdc_rhsb * sign;
    double drhss = l2->mdc_rhss * sign;
    double crhsd = -u_ceqqd;
    double crhsg = -u_ceqqg;
    double crhsb = -u_ceqqb;
    double crhss = u_ceqqg + u_ceqqb + u_ceqqd;

    double c34 = l2->Cggb * vm, c35 = l2->Cgdb * vm, c36 = l2->Cgsb * vm;
    double c37 = -(c34 + c35 + c36);
    double c44 = l2->Cddb * vm, c46 = l2->Cdgb * vm, c47 = l2->Cdsb * vm;
    double c48 = -(c46 + c44 + c47);
    double c51 = l2->Csdb * vm, c52 = l2->Csgb * vm, c53 = l2->Cssb * vm;
    double c55 = -(c52 + c51 + c53);
    double c58 = l2->Cbdb * vm, c59 = l2->Cbgb * vm, c60 = l2->Cbsb * vm;
    double c61 = -(c59 + c58 + c60);

    double d44 =  l2->mdc44 * sign, d46 =  l2->mdc46 * sign;
    double d47 = -l2->mdc47 * sign, d48 = -l2->mdc48 * sign;
    double d51 = -l2->mdc51 * sign, d52 =  l2->mdc52 * sign;
    double d53 =  l2->mdc53 * sign, d55 = -l2->mdc55 * sign;
    double d58 =  l2->mdc58 * sign, d59 =  l2->mdc59 * sign;
    double d60 =  l2->mdc60 * sign, d61 =  l2->mdc61 * sign;

#define B4A_LAT_ADD_CAP(o, cv) do { \
    int _o = (o); \
    if (_o < inz) { \
        lcap[_o] += (cv); \
        if (upd_rcb) rcb[_o] += ag0 * (cv); \
    } \
} while (0)

#define B4A_LAT_ADD_BOTH(o, dv, cv) do { \
    int _o = (o); \
    if (_o < inz) { \
        ldc[_o]  += (dv); \
        lcap[_o] += (cv); \
        if (upd_rcb) rcb[_o] += (dv) + ag0 * (cv); \
    } \
} while (0)

    const int * restrict off = sp->kluOff;
    int dn = c->dNodePrime;
    int gn = c->gNodePrime;
    int sn = c->sNodePrime;
    int bn = c->bNodePrime;
    if (dn) { rdc[dn] += drhsd; rcap[dn] += crhsd; }
    if (gn) { rcap[gn] += crhsg; }
    if (bn) { rdc[bn] += drhsb; rcap[bn] += crhsb; }
    if (sn) { rdc[sn] += drhss; rcap[sn] += crhss; }

    B4A_LAT_ADD_CAP(off[0], c34);
    B4A_LAT_ADD_CAP(off[1], c35);
    B4A_LAT_ADD_BOTH(off[4], d44, c44);
    B4A_LAT_ADD_BOTH(off[5], d46, c46);
    if (sn != 0 && bn != 0) {
        B4A_LAT_ADD_CAP(off[2], c36);
        B4A_LAT_ADD_CAP(off[3], c37);
        B4A_LAT_ADD_BOTH(off[6],  d47, c47);
        B4A_LAT_ADD_BOTH(off[7],  d48, c48);
        B4A_LAT_ADD_BOTH(off[8],  d51, c51);
        B4A_LAT_ADD_BOTH(off[9],  d52, c52);
        B4A_LAT_ADD_BOTH(off[10], d53, c53);
        B4A_LAT_ADD_BOTH(off[11], d55, c55);
        B4A_LAT_ADD_BOTH(off[12], d58, c58);
        B4A_LAT_ADD_BOTH(off[13], d59, c59);
        B4A_LAT_ADD_BOTH(off[14], d60, c60);
        B4A_LAT_ADD_BOTH(off[15], d61, c61);
    } else if (sn != 0) {
        B4A_LAT_ADD_CAP(off[2], c36);
        B4A_LAT_ADD_BOTH(off[6],  d47, c47);
        B4A_LAT_ADD_BOTH(off[8],  d51, c51);
        B4A_LAT_ADD_BOTH(off[9],  d52, c52);
        B4A_LAT_ADD_BOTH(off[10], d53, c53);
    } else if (bn != 0) {
        B4A_LAT_ADD_CAP(off[3], c37);
        B4A_LAT_ADD_BOTH(off[7],  d48, c48);
        B4A_LAT_ADD_BOTH(off[12], d58, c58);
        B4A_LAT_ADD_BOTH(off[13], d59, c59);
        B4A_LAT_ADD_BOTH(off[15], d61, c61);
    }

    if (__builtin_expect(c->has_dint | c->has_sint, 0)) {
        B4P(instance) *here = InstArray[c->instIdx];
        uintptr_t ax_addr = (uintptr_t)km->KLUmatrixAx;
        size_t ax_bytes = (size_t)inz * sizeof(double);
        if (c->has_dint) {
            double mgd = sp->mgd * sign;
            int o16 = B4P(ptrToKluOff)(here->B4P(DPdPtr), ax_addr, ax_bytes, inz);
            int o17 = B4P(ptrToKluOff)(here->B4P(DdpPtr), ax_addr, ax_bytes, inz);
            int o18 = B4P(ptrToKluOff)(here->B4P(DdPtr),  ax_addr, ax_bytes, inz);
            if (o16 < inz) { ldc[o16] -= mgd; if (upd_rcb) rcb[o16] -= mgd; }
            if (o17 < inz) { ldc[o17] -= mgd; if (upd_rcb) rcb[o17] -= mgd; }
            if (o18 < inz) { ldc[o18] += mgd; if (upd_rcb) rcb[o18] += mgd; }
        }
        if (c->has_sint) {
            double mgs = sp->mgs * sign;
            int o19 = B4P(ptrToKluOff)(here->B4P(SPsPtr), ax_addr, ax_bytes, inz);
            int o20 = B4P(ptrToKluOff)(here->B4P(SspPtr), ax_addr, ax_bytes, inz);
            int o21 = B4P(ptrToKluOff)(here->B4P(SsPtr),  ax_addr, ax_bytes, inz);
            if (o19 < inz) { ldc[o19] -= mgs; if (upd_rcb) rcb[o19] -= mgs; }
            if (o20 < inz) { ldc[o20] -= mgs; if (upd_rcb) rcb[o20] -= mgs; }
            if (o21 < inz) { ldc[o21] += mgs; if (upd_rcb) rcb[o21] += mgs; }
        }
    }
#undef B4A_LAT_ADD_CAP
#undef B4A_LAT_ADD_BOTH
    km->KLUmatrixLatentOps++;
}

static inline void
B4P(syncFrozenStateOnWake)(
    const BSIM4AppleCheck * restrict c,
    const BSIM4AppleL2 * restrict l2,
    CKTcircuit *ckt,
    int sync_s0)
{
    double *s0 = ckt->CKTstate0;
    double *s1 = ckt->CKTstate1;
    double *s2 = ckt->CKTstate2;
    double *s3 = ckt->CKTstates[3];
    double *s4 = ckt->CKTstates[4];
    int st = c->stateBase;
    double vbd = l2->a_vbd, vbs = l2->a_vbs, vgs = l2->a_vgs, vds = l2->a_vds;
    double qb  = l2->qb_st, qg  = l2->qg_st, qd  = l2->qd_st;
    if (s1) {
        s1[st + 0]  = vbd; s1[st + 1]  = vbs; s1[st + 2]  = vgs; s1[st + 3]  = vds;
        s1[st + 11] = qb;  s1[st + 12] = 0.0;
        s1[st + 13] = qg;  s1[st + 14] = 0.0;
        s1[st + 15] = qd;  s1[st + 16] = 0.0;
    }
    if (s2) {
        s2[st + 0]  = vbd; s2[st + 1]  = vbs; s2[st + 2]  = vgs; s2[st + 3]  = vds;
        s2[st + 11] = qb;  s2[st + 12] = 0.0;
        s2[st + 13] = qg;  s2[st + 14] = 0.0;
        s2[st + 15] = qd;  s2[st + 16] = 0.0;
    }
    if (s3) {
        s3[st + 11] = qb; s3[st + 13] = qg; s3[st + 15] = qd;
    }
    if (s4) {
        s4[st + 11] = qb; s4[st + 13] = qg; s4[st + 15] = qd;
    }
    if (sync_s0 && s0) {
        s0[st + 0]  = vbd; s0[st + 1]  = vbs; s0[st + 2]  = vgs; s0[st + 3]  = vds;
        s0[st + 11] = qb;  s0[st + 12] = 0.0;
        s0[st + 13] = qg;  s0[st + 14] = 0.0;
        s0[st + 15] = qd;  s0[st + 16] = 0.0;
    }
}

static void
B4P(latentUnfreezeAll)(B4P(model) *head, CKTcircuit *ckt, KLUmatrix *km)
{
    BSIM4AppleLatentState *lat = (BSIM4AppleLatentState *)head->B4P(AvxLatent);
    if (!lat || lat->nFrozen == 0)
        return;
    BSIM4ApplePack *pack = (BSIM4ApplePack *)head->B4P(AvxBlocks);
    if (!pack) return;
    BSIM4AppleCheck *checks = pack->checks;
    const BSIM4AppleStamp *stamps = pack->stamps;
    const BSIM4AppleL2 *l2s = pack->l2s;
    B4P(instance) **InstArray = head->B4P(InstanceArray);
    int nElig = lat->nElig;
    int nGroups = lat->nGroups;
    int sync_s0 = ((ckt->CKTmode & MODETRAN) && !(ckt->CKTmode & MODEINITPRED));
    double ag0 = ckt->CKTag[0];
    int acc = ckt->CKTstat->STATaccepted;
    int all_mine = (km && km->KLUmatrixLatentCount == lat->nFrozen);

    for (int i = 0; i < nElig; i++) {
        if (lat->latentFrozen[i]) {
            if (!all_mine && km && km->KLUmatrixAxLatentDC)
                B4P(stampFrozenDelta)(&checks[i], &stamps[i], &l2s[i], InstArray, km, ag0, -1.0);
            if (ckt->CKTmode & MODETRAN)
                B4P(syncFrozenStateOnWake)(&checks[i], &l2s[i], ckt, sync_s0);
            lat->latentFrozen[i] = 0;
            lat->latentStreak[i] = 0;
            checks[i].stamp_valid = 0;
            checks[i].last_active_step = acc;
        }
        lat->activeList[i] = i;
        lat->activePos[i] = i;
    }
    for (int gi = 0; gi < nGroups; gi++) {
        int goff = pack->groups[gi].offset;
        int gcnt = pack->groups[gi].count;
        lat->grpActiveCnt[gi] = gcnt;
        for (int k = 0; k < gcnt; k++) {
            lat->grpActiveList[goff + k] = goff + k;
            lat->grpActivePos[goff + k] = k;
        }
    }
    if (km) {
        km->KLUmatrixLatentCount -= lat->nFrozen;
        if (km->KLUmatrixLatentCount <= 0) {
            km->KLUmatrixLatentCount = 0;
            km->KLUmatrixLatentOps = 0;
            km->KLUmatrixResCapValid = 0;
            if (km->KLUmatrixAxLatentDC)
                memset(km->KLUmatrixAxLatentDC, 0, ((size_t)km->KLUmatrixNZ + 1) * sizeof(double));
            if (km->KLUmatrixAxLatentCap)
                memset(km->KLUmatrixAxLatentCap, 0, ((size_t)km->KLUmatrixNZ + 1) * sizeof(double));
            if (km->KLUmatrixRhsLatentDC && km->KLUmatrixRhsLatentLen > 0)
                memset(km->KLUmatrixRhsLatentDC, 0, (size_t)km->KLUmatrixRhsLatentLen * sizeof(double));
            if (km->KLUmatrixRhsLatentCap && km->KLUmatrixRhsLatentLen > 0)
                memset(km->KLUmatrixRhsLatentCap, 0, (size_t)km->KLUmatrixRhsLatentLen * sizeof(double));
        }
    }
    memset(lat->nodeFrozenCnt, 0, (size_t)(lat->nNodes + 1) * sizeof(int));
    lat->nActive = nElig;
    lat->nFrozen = 0;
    lat->cfEpoch = 0;
}

static int
B4P(ensureLatentBuffers)(KLUmatrix *km, int rhs_len)
{
    int nz = (int)km->KLUmatrixNZ;
    if (!km->KLUmatrixAxLatentDC) {
        if (posix_memalign((void **)&km->KLUmatrixAxLatentDC, 128, ((size_t)nz + 1) * sizeof(double)) != 0)
            km->KLUmatrixAxLatentDC = NULL;
        else
            memset(km->KLUmatrixAxLatentDC, 0, ((size_t)nz + 1) * sizeof(double));
    }
    if (!km->KLUmatrixAxLatentCap) {
        if (posix_memalign((void **)&km->KLUmatrixAxLatentCap, 128, ((size_t)nz + 1) * sizeof(double)) != 0)
            km->KLUmatrixAxLatentCap = NULL;
        else
            memset(km->KLUmatrixAxLatentCap, 0, ((size_t)nz + 1) * sizeof(double));
    }
    if (km->KLUmatrixRhsLatentLen < rhs_len || !km->KLUmatrixRhsLatentDC || !km->KLUmatrixRhsLatentCap) {
        free(km->KLUmatrixRhsLatentDC);
        free(km->KLUmatrixRhsLatentCap);
        km->KLUmatrixRhsLatentDC = NULL;
        km->KLUmatrixRhsLatentCap = NULL;
        km->KLUmatrixRhsLatentLen = 0;
        if (posix_memalign((void **)&km->KLUmatrixRhsLatentDC, 128, (size_t)rhs_len * sizeof(double)) == 0 &&
            posix_memalign((void **)&km->KLUmatrixRhsLatentCap, 128, (size_t)rhs_len * sizeof(double)) == 0)
        {
            memset(km->KLUmatrixRhsLatentDC, 0, (size_t)rhs_len * sizeof(double));
            memset(km->KLUmatrixRhsLatentCap, 0, (size_t)rhs_len * sizeof(double));
            km->KLUmatrixRhsLatentLen = rhs_len;
        }
    }
    return (km->KLUmatrixAxLatentDC && km->KLUmatrixAxLatentCap &&
            km->KLUmatrixRhsLatentDC && km->KLUmatrixRhsLatentCap);
}

static BSIM4AppleLatentState *
B4P(ensureLatentState)(B4P(model) *head, CKTcircuit *ckt)
{
    BSIM4ApplePack *pack = (BSIM4ApplePack *)head->B4P(AvxBlocks);
    if (!pack) return NULL;
    int nElig = pack->nElig;
    int nGroups = pack->nGroups;
    int N = ckt->CKTmaxEqNum;
    BSIM4AppleLatentState *lat = (BSIM4AppleLatentState *)head->B4P(AvxLatent);
    if (lat && lat->nElig == nElig && lat->nGroups == nGroups && lat->nNodes == N)
        return lat;

    B4P(freeAvxLatent)(lat);
    head->B4P(AvxLatent) = NULL;
    if (nElig <= 32 || nGroups <= 0 || N <= 1)
        return NULL;

    lat = (BSIM4AppleLatentState *)calloc(1, sizeof(BSIM4AppleLatentState));
    if (!lat) return NULL;
    lat->nElig = nElig;
    lat->nGroups = nGroups;
    lat->nNodes = N;
    lat->nActive = nElig;
    lat->nFrozen = 0;
    lat->lastAccepted = -1;
    lat->lastGmin = ckt->CKTgmin;

    lat->latentFrozen  = (uint8_t *)calloc((size_t)nElig, sizeof(uint8_t));
    lat->latentStreak  = (uint8_t *)calloc((size_t)nElig, sizeof(uint8_t));
    lat->instGroup     = (int *)malloc((size_t)nElig * sizeof(int));
    lat->activeList    = (int *)malloc((size_t)nElig * sizeof(int));
    lat->activePos     = (int *)malloc((size_t)nElig * sizeof(int));
    lat->grpActiveList = (int *)malloc((size_t)nElig * sizeof(int));
    lat->grpActivePos  = (int *)malloc((size_t)nElig * sizeof(int));
    lat->grpActiveCnt  = (int *)malloc((size_t)nGroups * sizeof(int));
    lat->nodeInstPtr   = (int *)calloc((size_t)(N + 2), sizeof(int));
    lat->vSnap         = (double *)calloc((size_t)(N + 1), sizeof(double));
    lat->nodeFrozenCnt = (int *)calloc((size_t)(N + 1), sizeof(int));
    lat->cfFastList    = (int *)malloc((size_t)nElig * sizeof(int));
    lat->cfGrpStart    = (int *)malloc((size_t)nGroups * sizeof(int));
    lat->cfGrpCnt      = (int *)calloc((size_t)nGroups, sizeof(int));
    lat->cfIsFastInst  = (unsigned char *)calloc((size_t)nElig, sizeof(unsigned char));

    if (!lat->latentFrozen || !lat->latentStreak || !lat->instGroup ||
        !lat->activeList || !lat->activePos || !lat->grpActiveList ||
        !lat->grpActivePos || !lat->grpActiveCnt || !lat->nodeInstPtr ||
        !lat->vSnap || !lat->nodeFrozenCnt || !lat->cfFastList ||
        !lat->cfGrpStart || !lat->cfGrpCnt || !lat->cfIsFastInst)
    {
        B4P(freeAvxLatent)(lat);
        return NULL;
    }

    const BSIM4AppleCheck *checks = pack->checks;
    for (int gi = 0; gi < nGroups; gi++) {
        int goff = pack->groups[gi].offset;
        int gcnt = pack->groups[gi].count;
        lat->grpActiveCnt[gi] = gcnt;
        for (int k = 0; k < gcnt; k++) {
            int idx = goff + k;
            lat->instGroup[idx] = gi;
            lat->grpActiveList[idx] = idx;
            lat->grpActivePos[idx] = k;
        }
    }

    int uniq[4];
    for (int i = 0; i < nElig; i++) {
        lat->activeList[i] = i;
        lat->activePos[i] = i;
        int nu = B4P(collectUniqueNodesInst)(&checks[i], N, uniq);
        for (int k = 0; k < nu; k++)
            lat->nodeInstPtr[uniq[k] + 1]++;
    }
    for (int i = 1; i <= N; i++)
        lat->nodeInstPtr[i] += lat->nodeInstPtr[i - 1];
    int total = lat->nodeInstPtr[N];
    lat->nodeInstIdx = (int *)malloc((size_t)(total > 0 ? total : 1) * sizeof(int));
    int *cur = (int *)malloc((size_t)(N + 1) * sizeof(int));
    if (!lat->nodeInstIdx || !cur) {
        free(cur);
        B4P(freeAvxLatent)(lat);
        return NULL;
    }
    memcpy(cur, lat->nodeInstPtr, (size_t)(N + 1) * sizeof(int));
    for (int i = 0; i < nElig; i++) {
        int nu = B4P(collectUniqueNodesInst)(&checks[i], N, uniq);
        for (int k = 0; k < nu; k++)
            lat->nodeInstIdx[cur[uniq[k]]++] = i;
    }
    free(cur);

    head->B4P(AvxLatent) = lat;
    return lat;
}

static void
B4P(latentSync)(B4P(model) *head, CKTcircuit *ckt, KLUmatrix *km)
{
    int lat_en = B4P(mrLatentEnabled)();
    int cf_active = ckt->CKTmrSubActive;
    BSIM4ApplePack *pack = (BSIM4ApplePack *)head->B4P(AvxBlocks);
    BSIM4AppleLatentState *lat_cur = (BSIM4AppleLatentState *)head->B4P(AvxLatent);

    /* Restore slow unfrozen BSIM4 instances' states when returning from a compound-fast micro-burst */
    int cf_just_synced = 0;
    if (!cf_active && lat_cur && lat_cur->cfNeedsSync && pack) {
        BSIM4AppleCheck *checks = pack->checks;
        const BSIM4AppleL2 *l2s = pack->l2s;
        int nElig = lat_cur->nElig;
        int sync_s0 = ((ckt->CKTmode & MODETRAN) && !(ckt->CKTmode & MODEINITPRED));
        for (int i = 0; i < nElig; i++) {
            if (!lat_cur->cfIsFastInst[i] && !lat_cur->latentFrozen[i]) {
                if (checks[i].phys_valid && (ckt->CKTmode & MODETRAN))
                    B4P(syncFrozenStateOnWake)(&checks[i], &l2s[i], ckt, sync_s0);
                checks[i].stamp_valid = 0;
            }
        }
        lat_cur->cfNeedsSync = 0;
        cf_just_synced = 1;
    }

    if (!lat_en && !cf_active)
        return;

    int nFb = head->B4P(AvxFallbackCount);
    int nthreads = 1;
#ifdef USE_OMP
    nthreads = omp_get_max_threads();
#endif
    int eligible = (head->B4P(AvxBuilt) && head->B4P(AvxDynValid) && pack &&
                    B4P(isAvxEligibleTran)(ckt) && !B4P(AvxNeedAosSync) &&
                    pack->nElig > 32 && nFb == 0 && nthreads == 1 &&
                    ckt->CKTtime > 0.0 && ckt->CKTsenInfo == NULL);
    if (!eligible) {
        if (lat_en)
            B4P(latentUnfreezeAll)(head, ckt, km);
        return;
    }

    BSIM4AppleLatentState *lat = B4P(ensureLatentState)(head, ckt);
    if (!lat) return;

    BSIM4AppleCheck *checks = pack->checks;
    const BSIM4AppleStamp *stamps = pack->stamps;
    const BSIM4AppleL2 *l2s = pack->l2s;
    B4P(instance) **InstArray = head->B4P(InstanceArray);
    int nElig = lat->nElig;
    int nGroups = lat->nGroups;
    int N = lat->nNodes;
    int acc = ckt->CKTstat->STATaccepted;
    double ag0 = ckt->CKTag[0];

    if (lat_en) {
        int rhs_len = (int)km->KLUmatrixNrhs + 1;
        if (rhs_len < ckt->CKTmaxEqNum + 1) rhs_len = ckt->CKTmaxEqNum + 1;
        if (!B4P(ensureLatentBuffers)(km, rhs_len))
            goto build_cf_list;

        double vtol = B4P(mrLatentVtol)(ckt);
        if (vtol < 0.0) {
            B4P(latentUnfreezeAll)(head, ckt, km);
            goto build_cf_list;
        }

        if (ckt->CKTgmin != lat->lastGmin || km->KLUmatrixLatentOps > 65536 || acc < lat->lastAccepted) {
            if (lat->nFrozen > 0)
                lat->statResets++;
            B4P(latentUnfreezeAll)(head, ckt, km);
            lat->lastGmin = ckt->CKTgmin;
            lat->lastAccepted = acc;
            lat->cfEpoch = 0;
        }

        const double * restrict rhsOld = ckt->CKTrhsOld;
        int is_pred = ((ckt->CKTmode & MODEINITPRED) != 0);

        /* 1. Exact node-based wake-up pass (fast nodes only during micro-steps!) */
        if (lat->nFrozen > 0) {
            double half_wake = 0.5 * vtol;
            const int * restrict fcnt = lat->nodeFrozenCnt;
            double * restrict vsnap = lat->vSnap;
            const int * restrict niptr = lat->nodeInstPtr;
            const int * restrict niidx = lat->nodeInstIdx;
            int uniq[4];
            int woke_any = 0;

            int sub_only = (cf_active && ckt->CKTmrFastNodes && ckt->CKTmrSubNFast > 0);
            int n_loop = sub_only ? ckt->CKTmrSubNFast : (N - 1);
            for (int li = 0; li < n_loop; li++) {
                int node = sub_only ? ckt->CKTmrFastNodes[li] : (li + 1);
                if (node <= 0 || node >= N || fcnt[node] == 0) continue;
                double vnow = rhsOld[node];
                if (fabs(vnow - vsnap[node]) <= half_wake) continue;
                vsnap[node] = vnow;
                int p0 = niptr[node], p1 = niptr[node + 1];
                for (int p = p0; p < p1; p++) {
                    int idx = niidx[p];
                    if (!lat->latentFrozen[idx]) continue;
                    BSIM4AppleCheck *c = &checks[idx];
                    B4P(stampFrozenDelta)(c, &stamps[idx], &l2s[idx], InstArray, km, ag0, -1.0);
                    int nu = B4P(collectUniqueNodesInst)(c, N, uniq);
                    for (int k = 0; k < nu; k++)
                        lat->nodeFrozenCnt[uniq[k]]--;
                    B4P(syncFrozenStateOnWake)(c, &l2s[idx], ckt, !is_pred);
                    lat->latentFrozen[idx] = 0;
                    lat->latentStreak[idx] = 0;
                    c->stamp_valid = 0;
                    c->last_active_step = acc;
                    lat->activePos[idx] = lat->nActive;
                    lat->activeList[lat->nActive++] = idx;
                    int gi = lat->instGroup[idx];
                    int goff = pack->groups[gi].offset;
                    int gpos = lat->grpActiveCnt[gi]++;
                    lat->grpActiveList[goff + gpos] = idx;
                    lat->grpActivePos[idx] = gpos;
                    lat->nFrozen--;
                    km->KLUmatrixLatentCount--;
                    lat->statWakes++;
                    woke_any = 1;
                }
            }
            if (woke_any)
                lat->cfEpoch = 0;
            if (km->KLUmatrixLatentCount <= 0) {
                km->KLUmatrixLatentCount = 0;
                km->KLUmatrixLatentOps = 0;
                km->KLUmatrixResCapValid = 0;
                memset(km->KLUmatrixAxLatentDC, 0, ((size_t)km->KLUmatrixNZ + 1) * sizeof(double));
                memset(km->KLUmatrixAxLatentCap, 0, ((size_t)km->KLUmatrixNZ + 1) * sizeof(double));
                memset(km->KLUmatrixRhsLatentDC, 0, (size_t)km->KLUmatrixRhsLatentLen * sizeof(double));
                memset(km->KLUmatrixRhsLatentCap, 0, (size_t)km->KLUmatrixRhsLatentLen * sizeof(double));
            }
        }

        /* 2. Freeze pass: only on full-circuit steps (not during compound-fast micro-bursts) */
        if (!cf_active && acc > lat->lastAccepted && acc >= 3 && is_pred) {
            const double * restrict v_acc =
                (ckt->CKTtaPredActive && ckt->CKTtaSol[0]) ? ckt->CKTtaSol[0] : rhsOld;
            int prev_acc = lat->lastAccepted;
            int delta_acc = (prev_acc >= 0) ? (acc - prev_acc) : 1;
            if (delta_acc < 1) delta_acc = 1;
            lat->lastAccepted = acc;
            int streak_req = B4P(mrLatentStreakReq)();
            const double * restrict s1 = ckt->CKTstate1;
            double reltol = ckt->CKTreltol;
            double abstol = ckt->CKTabstol;
            double icap_floor = B4P(mrLatentItol)(ckt);
            double q_snap_tol = 0.4 * vtol;
            int pack_gmin_ok = (pack->step_valid && pack->last_gmin == ckt->CKTgmin);
            int uniq[4];

            for (int ai = lat->nActive - 1; ai >= 0; ai--) {
                int idx = lat->activeList[ai];
                if (cf_just_synced && !lat->cfIsFastInst[idx])
                    continue;
                BSIM4AppleCheck *c = &checks[idx];
                if (!pack_gmin_ok || !c->phys_valid || !c->stamp_valid) {
                    lat->latentStreak[idx] = 0;
                    lat->statFailValid++;
                    continue;
                }
                int st = c->stateBase;
                double tol_i = 0.1 * (reltol * fabs(c->Idtot) + abstol);
                if (tol_i < icap_floor) tol_i = icap_floor;
                if (fabs(s1[st + 12]) > tol_i ||
                    fabs(s1[st + 14]) > tol_i ||
                    fabs(s1[st + 16]) > tol_i)
                {
                    lat->latentStreak[idx] = 0;
                    lat->statFailIcap++;
                    continue;
                }
                double vs = v_acc[c->sNodePrime];
                double t  = c->type;
                double vds_r = (v_acc[c->dNodePrime] - vs) * t;
                double vgs_r = (v_acc[c->gNodePrime] - vs) * t;
                double vbs_r = (v_acc[c->bNodePrime] - vs) * t;
                int ez = 0;
                if (!B4P(instCacheMatch)(c, vds_r, vgs_r, vbs_r, vtol, reltol, abstol, &ez)) {
                    lat->latentStreak[idx] = 0;
                    lat->statFailCache++;
                    continue;
                }
                int inc = (c->last_active_step <= prev_acc) ? delta_acc : 1;
                int ns = (int)lat->latentStreak[idx] + inc;
                lat->latentStreak[idx] = (uint8_t)(ns > 255 ? 255 : ns);
                if ((int)lat->latentStreak[idx] < streak_req)
                    continue;

                int nu = B4P(collectUniqueNodesInst)(c, N, uniq);
                int snap_ok = 1;
                for (int k = 0; k < nu; k++) {
                    int u = uniq[k];
                    if (lat->nodeFrozenCnt[u] > 0 && fabs(v_acc[u] - lat->vSnap[u]) > q_snap_tol) {
                        snap_ok = 0;
                        break;
                    }
                }
                if (!snap_ok) {
                    lat->statFailSnap++;
                    continue;
                }

                for (int k = 0; k < nu; k++) {
                    int u = uniq[k];
                    if (lat->nodeFrozenCnt[u] == 0)
                        lat->vSnap[u] = v_acc[u];
                    lat->nodeFrozenCnt[u]++;
                }
                B4P(stampFrozenDelta)(c, &stamps[idx], &l2s[idx], InstArray, km, ag0, +1.0);
                lat->latentFrozen[idx] = 1;
                int last_idx = lat->activeList[--lat->nActive];
                lat->activeList[ai] = last_idx;
                lat->activePos[last_idx] = ai;
                lat->activePos[idx] = -1;
                int gi = lat->instGroup[idx];
                int goff = pack->groups[gi].offset;
                int gpos = lat->grpActivePos[idx];
                int glast = lat->grpActiveList[goff + --lat->grpActiveCnt[gi]];
                lat->grpActiveList[goff + gpos] = glast;
                lat->grpActivePos[glast] = gpos;
                lat->grpActivePos[idx] = -1;
                lat->nFrozen++;
                km->KLUmatrixLatentCount++;
                lat->statFreezes++;
            }
        }
        lat->statCalls++;
        lat->statFrozenSum += (unsigned long long)lat->nFrozen;
    }

build_cf_list:
    if (cf_active && ckt->CKTmrFastNodes && ckt->CKTmrSubNFast > 0 &&
        lat->cfEpoch != ckt->CKTmrSubEpoch)
    {
        memset(lat->cfIsFastInst, 0, (size_t)nElig);
        memset(lat->cfGrpCnt, 0, (size_t)nGroups * sizeof(int));
        const int * restrict fn = ckt->CKTmrFastNodes;
        int nF = ckt->CKTmrSubNFast;
        const int * restrict niptr = lat->nodeInstPtr;
        const int * restrict niidx = lat->nodeInstIdx;
        int * restrict tmp = pack->eval_q;
        int n_fast_inst = 0;
        for (int fi = 0; fi < nF; fi++) {
            int node = fn[fi];
            if (node <= 0 || node >= N) continue;
            int p0 = niptr[node], p1 = niptr[node + 1];
            for (int p = p0; p < p1; p++) {
                int idx = niidx[p];
                if (!lat->cfIsFastInst[idx]) {
                    lat->cfIsFastInst[idx] = 1;
                    if (!lat->latentFrozen[idx]) {
                        tmp[n_fast_inst++] = idx;
                        lat->cfGrpCnt[lat->instGroup[idx]]++;
                    }
                }
            }
        }
        /* Include any non-frozen instance that has never been evaluated yet */
        for (int idx = 0; idx < nElig; idx++) {
            if (!lat->cfIsFastInst[idx] && !lat->latentFrozen[idx] && !checks[idx].phys_valid) {
                lat->cfIsFastInst[idx] = 1;
                tmp[n_fast_inst++] = idx;
                lat->cfGrpCnt[lat->instGroup[idx]]++;
            }
        }
        /* Bucket fast non-frozen instances by group for dense 2-wide NEON batching */
        int pos = 0;
        for (int gi = 0; gi < nGroups; gi++) {
            lat->cfGrpStart[gi] = pos;
            pos += lat->cfGrpCnt[gi];
            lat->cfGrpCnt[gi] = 0;
        }
        for (int k = 0; k < n_fast_inst; k++) {
            int idx = tmp[k];
            int gi = lat->instGroup[idx];
            lat->cfFastList[lat->cfGrpStart[gi] + lat->cfGrpCnt[gi]++] = idx;
        }
        lat->cfNFast = n_fast_inst;
        lat->cfEpoch = ckt->CKTmrSubEpoch;
        if (n_fast_inst < nElig - lat->nFrozen)
            lat->cfNeedsSync = 1;
    }
}
#endif /* KLU */

/*
 * Evaluate one BSIM4AppleGroup (all instances when idx_list == NULL, or a subset idx_list[0..idx_count-1]):
 *   1. Sequential W=1 scan over 128B BSIM4AppleCheck entries:
 *      - L1 stationary hits -> optional scalar Taylor charge sync + direct stamp
 *      - L2 stationary hits -> scalar L2 integrate/assemble + direct stamp
 *      - Active misses      -> compact into eval_q[]
 *   2. Flush eval_q[] in dense 2-wide NEON pairs sharing identical BSIM4ApplePParam *pp!
 */
static void
B4P(evalAppleGroupList)(
    BSIM4ApplePack * restrict pack,
    int gi,
    const int * restrict idx_list,
    int idx_count,
    B4P(instance) **InstArray,
    CKTcircuit *ckt,
    double * restrict my_ax,
    double * restrict my_rhs,
    int stamp_mode,
    int dc_only,
    int force_aos,
    int l1_step_ok,
    int l2_gmin_ok,
    int * restrict eval_q,
    double * restrict eval_vds,
    double * restrict eval_vgs,
    double * restrict eval_vbs)
{
    const BSIM4AppleGroup * restrict grp = &pack->groups[gi];
    const BSIM4ApplePParam * restrict pp = &grp->pp;
    BSIM4AppleCheck * restrict checks = pack->checks;
    BSIM4AppleStamp * restrict stamps = pack->stamps;
    BSIM4AppleL2    * restrict l2s    = pack->l2s;
    const BSIM4AppleJct * restrict jcts = pack->jcts;

    const double * restrict rhsOld = ckt->CKTrhsOld;
    double * restrict s0 = ckt->CKTstate0;
    const double * restrict s1 = dc_only ? NULL : ckt->CKTstate1;
    const double * restrict s2 = dc_only ? NULL : ckt->CKTstate2;
    const double * restrict s3 = dc_only ? NULL : ckt->CKTstate3;
    const double cur_gmin = ckt->CKTgmin;
    const double reltol   = ckt->CKTreltol;
    const double abstol   = ckt->CKTabstol;
    const double ag0 = dc_only ? 0.0 : ckt->CKTag[0];
    const double ag1 = dc_only ? 0.0 : ckt->CKTag[1];
    const double ag2 = dc_only ? 0.0 : ckt->CKTag[2];
    const double ag3 = dc_only ? 0.0 : ckt->CKTag[3];
    const int is_pred    = (!dc_only && (ckt->CKTmode & MODEINITPRED) != 0);
    const int is_initjct = (dc_only && (ckt->CKTmode & MODEINITJCT) != 0);
    const int is_gear3   = (!dc_only && ckt->CKTintegrateMethod == GEAR && ckt->CKTorder == 3);
    const int is_gear2   = (!dc_only && ckt->CKTintegrateMethod == GEAR && ckt->CKTorder == 2);
    const int is_trap2   = (!dc_only && ckt->CKTintegrateMethod == TRAPEZOIDAL && ckt->CKTorder == 2);
    const int need_aos   = B4P(AvxNeedAosSync) || force_aos;
    const double vtol    = dc_only ? -1.0 : B4P(avxBypassTol)(ckt);
    const double xfact   = is_pred ? (ckt->CKTdelta / ckt->CKTdeltaOld[1]) : 0.0;
    const double one_p_xf = 1.0 + xfact;
    const int do_l1 = (!dc_only && !is_pred && !need_aos && vtol >= 0.0 && l1_step_ok);
    const int do_l2 = (!dc_only && !need_aos && vtol >= 0.0 && l2_gmin_ok);
    const int acc   = ckt->CKTstat->STATaccepted;

    int start = grp->offset;
    int n_iter = idx_list ? idx_count : grp->count;
    int n_eval = 0;
    int noncon_add = 0;

    for (int ii = 0; ii < n_iter; ii++) {
        int i = idx_list ? idx_list[ii] : (start + ii);
        BSIM4AppleCheck * restrict c = &checks[i];

        double vds, vgs, vbs;
        if (is_initjct) {
            B4P(instance) *h = InstArray[c->instIdx];
            double t = c->type;
            vds = t * h->B4P(icVDS);
            vgs = t * h->B4P(icVGS);
            vbs = t * h->B4P(icVBS);
            if (vds == 0.0 && vgs == 0.0 && vbs == 0.0) {
                vds = 0.1;
                vgs = pp->vth0_signed + 0.1;
                vbs = 0.0;
            }
            eval_q[n_eval]   = i;
            eval_vds[n_eval] = vds;
            eval_vgs[n_eval] = vgs;
            eval_vbs[n_eval] = vbs;
            n_eval++;
            continue;
        }

        double s0_vds, s0_vgs, s0_vbs, s0_vbd;

        if (!is_pred) {
            double vs = rhsOld[c->sNodePrime];
            double t  = c->type;
            vds = (rhsOld[c->dNodePrime] - vs) * t;
            vgs = (rhsOld[c->gNodePrime] - vs) * t;
            vbs = (rhsOld[c->bNodePrime] - vs) * t;

            /* Level-1 per-instance stationary check: zero s1/s2 loads via base_cq + ag0*dq! */
            if (do_l1 && c->stamp_valid) {
                int exact_zero = 0;
                if (B4P(instCacheMatch)(c, vds, vgs, vbs, vtol, reltol, abstol, &exact_zero)) {
                    if (!exact_zero) {
                        bsim4_apple_l1_sync_one(c, &l2s[i], vds, vgs, vbs, ag0, s0);
                        c->last_active_step = acc;
                    }
                    B4P(stampOneDirect)(c, &stamps[i], InstArray, my_ax, my_rhs, stamp_mode);
                    continue;
                }
            }

            int st = c->stateBase;
            s0_vbd = s0[st + 0];
            s0_vbs = s0[st + 1];
            s0_vgs = s0[st + 2];
            s0_vds = s0[st + 3];
        } else {
            int st = c->stateBase;
            double s1_vds = s1[st + 3], s1_vgs = s1[st + 2], s1_vbs = s1[st + 1];
            double s1_vbd = s1_vbs - s1_vds;
            s0[st + 3] = s0_vds = s1_vds;
            s0[st + 2] = s0_vgs = s1_vgs;
            s0[st + 1] = s0_vbs = s1_vbs;
            s0[st + 0] = s0_vbd = s1_vbd;
#if NGSPICE_OPT_GENERAL
            if (ckt->CKTtaPredActive) {
                double vs = rhsOld[c->sNodePrime];
                double t  = c->type;
                vds = (rhsOld[c->dNodePrime] - vs) * t;
                vgs = (rhsOld[c->gNodePrime] - vs) * t;
                vbs = (rhsOld[c->bNodePrime] - vs) * t;
            } else
#endif
            {
                vds = one_p_xf * s1_vds - xfact * s2[st + 3];
                vgs = one_p_xf * s1_vgs - xfact * s2[st + 2];
                vbs = one_p_xf * s1_vbs - xfact * s2[st + 1];
            }
        }

        BSIM4AppleStamp * restrict sp = &stamps[i];

        /* Voltage limiting: fetlim + limvds + pnjlim */
        double vgdo = s0_vgs - s0_vds;
        double vbd  = vbs - vds;
        double vgd  = vgs - vds;
        double von  = sp->von;
        if (s0_vds >= 0.0) {
            vgs = b4a_fetlim_scalar(vgs, s0_vgs, von);
            vds = vgs - vgd;
            vds = b4a_limvds_scalar(vds, s0_vds);
        } else {
            vgd = b4a_fetlim_scalar(vgd, vgdo, von);
            vds = vgs - vgd;
            vds = -b4a_limvds_scalar(-vds, -s0_vds);
            vgs = vgd + vds;
        }
        int icheck = 0;
        if (vds >= 0.0) {
            vbs = b4a_pnjlim_scalar(vbs, s0_vbs, CONSTvt0, sp->vcrit, &icheck);
        } else {
            vbd = b4a_pnjlim_scalar(vbd, s0_vbd, CONSTvt0, sp->vcrit, &icheck);
            vbs = vbd + vds;
        }
        if (icheck) noncon_add++;

        /* Level-2 per-instance cross-timestep stationary check */
        if (do_l2 && !icheck && c->phys_valid) {
            int exact_zero = 0;
            if (B4P(instCacheMatch)(c, vds, vgs, vbs, vtol, reltol, abstol, &exact_zero)) {
                bsim4_apple_l2_assemble_one(c, &l2s[i], sp, vds, vgs, vbs, !exact_zero,
                                            ag0, ag1, ag2, ag3, is_gear3, is_gear2, is_trap2,
                                            s0, s1, s2, s3);
                if (!exact_zero)
                    c->last_active_step = acc;
                c->stamp_valid = 1;
                B4P(stampOneDirect)(c, sp, InstArray, my_ax, my_rhs, stamp_mode);
                continue;
            }
        }

        /* Active miss: compact into dense B4A_W-wide NEON batch for this pParam group */
        eval_q[n_eval]   = i;
        eval_vds[n_eval] = vds;
        eval_vgs[n_eval] = vgs;
        eval_vbs[n_eval] = vbs;
        n_eval++;
    }

    if (noncon_add) {
#ifdef USE_OMP
        if (omp_in_parallel()) {
            #pragma omp atomic
            ckt->CKTnoncon += noncon_add;
        } else
#endif
        {
            ckt->CKTnoncon += noncon_add;
        }
    }

    /* Flush compacted active misses in dense B4A_W-wide NEON batches sharing identical pp */
    BSIM4AppleOut out;
    for (int q = 0; q < n_eval; q += B4A_W) {
        int nact = n_eval - q;
        if (nact > B4A_W) nact = B4A_W;

        const BSIM4AppleJct *jpts[B4A_W];
        BSIM4AppleCheck     *cpts[B4A_W];
        BSIM4AppleStamp     *spts[B4A_W];
        BSIM4AppleL2        *l2pts[B4A_W];
        double bvds[B4A_W], bvgs[B4A_W], bvbs[B4A_W];

        for (int l = 0; l < B4A_W; l++) {
            int ql = (l < nact) ? (q + l) : q;
            int idx = eval_q[ql];
            jpts[l]  = &jcts[idx];
            cpts[l]  = &checks[idx];
            spts[l]  = &stamps[idx];
            l2pts[l] = &l2s[idx];
            bvds[l]  = eval_vds[ql];
            bvgs[l]  = eval_vgs[ql];
            bvbs[l]  = eval_vbs[ql];
        }

        bsim4_apple_eval_batch(
            pp, jpts, cpts, spts, l2pts, nact,
            bvds, bvgs, bvbs,
            cur_gmin, ag0, ag1, ag2, ag3, is_gear3, is_gear2, is_trap2,
            s0, s1, s2, s3, need_aos ? &out : NULL, need_aos, dc_only);

        for (int l = 0; l < nact; l++) {
            if (dc_only) {
                cpts[l]->phys_valid = 0;
                cpts[l]->stamp_valid = 0;
                if (need_aos)
                    B4P(scatterAosOne)(InstArray[cpts[l]->instIdx], &out, l);
                else
                    InstArray[cpts[l]->instIdx]->B4P(von) = spts[l]->von;
            } else {
                cpts[l]->last_active_step = acc;
                cpts[l]->phys_valid = 1;
                cpts[l]->stamp_valid = !need_aos;
                if (need_aos)
                    B4P(scatterAosOne)(InstArray[cpts[l]->instIdx], &out, l);
            }
            if (stamp_mode != 3)
                B4P(stampOneDirect)(cpts[l], spts[l], InstArray, my_ax, my_rhs, stamp_mode);
        }
    }
}

/*
 * Top-level single-thread / OpenMP driver for BSIM4ApplePack.
 * Uses pack-level step validation (checked ONCE per CKTload instead of per transistor!).
 */
static void
B4P(runApplePack)(
    B4P(model) *head,
    CKTcircuit *ckt,
    double * restrict my_ax,
    double * restrict my_rhs,
    int stamp_mode,
    int dc_only,
    int force_aos)
{
    BSIM4ApplePack *pack = (BSIM4ApplePack *)head->B4P(AvxBlocks);
    if (!pack || pack->nElig <= 0) return;
    B4P(instance) **InstArray = head->B4P(InstanceArray);
    int nGroups = pack->nGroups;

    const double cur_time = ckt->CKTtime;
    const double cur_gmin = ckt->CKTgmin;
    const double ag0 = dc_only ? 0.0 : ckt->CKTag[0];
    const double ag1 = dc_only ? 0.0 : ckt->CKTag[1];
    const double ag2 = dc_only ? 0.0 : ckt->CKTag[2];
    const double ag3 = dc_only ? 0.0 : ckt->CKTag[3];
    const int is_gear3 = (!dc_only && ckt->CKTintegrateMethod == GEAR && ckt->CKTorder == 3);
    const int is_gear2 = (!dc_only && ckt->CKTintegrateMethod == GEAR && ckt->CKTorder == 2);
    const int is_trap2 = (!dc_only && ckt->CKTintegrateMethod == TRAPEZOIDAL && ckt->CKTorder == 2);
    const int int_mode = is_gear3 ? 3 : (is_gear2 ? 2 : (is_trap2 ? 1 : 0));
    const double *s0 = ckt->CKTstate0, *s1 = ckt->CKTstate1, *s2 = ckt->CKTstate2;
    const double *s3 = dc_only ? NULL : ckt->CKTstate3;

    int l2_gmin_ok = (!dc_only && pack->step_valid && pack->last_gmin == cur_gmin);
    int l1_step_ok = (l2_gmin_ok &&
                      pack->last_time == cur_time &&
                      pack->last_ag0 == ag0 && pack->last_ag1 == ag1 && pack->last_ag2 == ag2 &&
                      (!is_gear3 || pack->last_ag3 == ag3) &&
                      pack->last_int_mode == int_mode &&
                      pack->last_s0 == s0 && pack->last_s1 == s1 && pack->last_s2 == s2 &&
                      (!is_gear3 || pack->last_s3 == s3));

#ifdef USE_OMP
    if (omp_in_parallel()) {
        int max_g = 64;
        for (int g = 0; g < nGroups; g++)
            if (pack->groups[g].count > max_g) max_g = pack->groups[g].count;
        int *tq = (int *)malloc((size_t)(max_g + 4) * sizeof(int));
        double *tvd = (double *)malloc((size_t)(max_g + 4) * sizeof(double));
        double *tvg = (double *)malloc((size_t)(max_g + 4) * sizeof(double));
        double *tvb = (double *)malloc((size_t)(max_g + 4) * sizeof(double));
        if (tq && tvd && tvg && tvb) {
            #pragma omp for schedule(dynamic, 1) nowait
            for (int gi = 0; gi < nGroups; gi++) {
                B4P(evalAppleGroupList)(pack, gi, NULL, pack->groups[gi].count,
                                        InstArray, ckt, my_ax, my_rhs,
                                        stamp_mode, dc_only, force_aos,
                                        l1_step_ok, l2_gmin_ok, tq, tvd, tvg, tvb);
            }
        }
        free(tq); free(tvd); free(tvg); free(tvb);
        #pragma omp single nowait
        {
            if (dc_only) {
                pack->step_valid = 0;
            } else {
                pack->last_time = cur_time; pack->last_gmin = cur_gmin;
                pack->last_ag0 = ag0; pack->last_ag1 = ag1; pack->last_ag2 = ag2; pack->last_ag3 = ag3;
                pack->last_int_mode = int_mode;
                pack->last_s0 = s0; pack->last_s1 = s1; pack->last_s2 = s2; pack->last_s3 = s3;
                pack->step_valid = !(B4P(AvxNeedAosSync) || force_aos);
            }
        }
        return;
    }
#endif

    const BSIM4AppleLatentState *lat = (const BSIM4AppleLatentState *)head->B4P(AvxLatent);
    if (stamp_mode == 1 && !dc_only && ckt->CKTmrSubActive && lat && lat->cfEpoch == ckt->CKTmrSubEpoch) {
        for (int gi = 0; gi < nGroups; gi++) {
            int gcnt = lat->cfGrpCnt[gi];
            if (gcnt <= 0) continue;
            B4P(evalAppleGroupList)(pack, gi, lat->cfFastList + lat->cfGrpStart[gi], gcnt,
                                    InstArray, ckt, my_ax, my_rhs,
                                    stamp_mode, dc_only, force_aos,
                                    l1_step_ok, l2_gmin_ok,
                                    pack->eval_q, pack->eval_vds, pack->eval_vgs, pack->eval_vbs);
        }
    } else if (stamp_mode == 1 && !dc_only && lat && lat->nFrozen > 0) {
        for (int gi = 0; gi < nGroups; gi++) {
            int gcnt = lat->grpActiveCnt[gi];
            if (gcnt <= 0) continue;
            B4P(evalAppleGroupList)(pack, gi, lat->grpActiveList + pack->groups[gi].offset, gcnt,
                                    InstArray, ckt, my_ax, my_rhs,
                                    stamp_mode, dc_only, force_aos,
                                    l1_step_ok, l2_gmin_ok,
                                    pack->eval_q, pack->eval_vds, pack->eval_vgs, pack->eval_vbs);
        }
    } else if (stamp_mode == 2 && dc_only && pack->orig_order) {
        /* Preserve exact stock InstArray[0..nElig-1] Sparse stamping order during DCOP/TRANOP */
        for (int gi = 0; gi < nGroups; gi++) {
            B4P(evalAppleGroupList)(pack, gi, NULL, pack->groups[gi].count,
                                    InstArray, ckt, my_ax, my_rhs,
                                    3, dc_only, force_aos,
                                    l1_step_ok, l2_gmin_ok,
                                    pack->eval_q, pack->eval_vds, pack->eval_vgs, pack->eval_vbs);
        }
        const int * restrict ord = pack->orig_order;
        int nElig = pack->nElig;
        for (int oi = 0; oi < nElig; oi++) {
            int idx = ord[oi];
            B4P(stampOneDirect)(&pack->checks[idx], &pack->stamps[idx], InstArray, my_ax, my_rhs, 2);
        }
    } else {
        for (int gi = 0; gi < nGroups; gi++) {
            B4P(evalAppleGroupList)(pack, gi, NULL, pack->groups[gi].count,
                                    InstArray, ckt, my_ax, my_rhs,
                                    stamp_mode, dc_only, force_aos,
                                    l1_step_ok, l2_gmin_ok,
                                    pack->eval_q, pack->eval_vds, pack->eval_vgs, pack->eval_vbs);
        }
    }

    if (dc_only) {
        pack->step_valid = 0;
    } else {
        pack->last_time = cur_time; pack->last_gmin = cur_gmin;
        pack->last_ag0 = ag0; pack->last_ag1 = ag1; pack->last_ag2 = ag2; pack->last_ag3 = ag3;
        pack->last_int_mode = int_mode;
        pack->last_s0 = s0; pack->last_s1 = s1; pack->last_s2 = s2; pack->last_s3 = s3;
        pack->step_valid = !(B4P(AvxNeedAosSync) || force_aos);
    }
}

/*
 * Fast convergence test over 128B BSIM4AppleCheck (zero AoS chase!).
 */
int
B4P(convTestAvx)(GENmodel *inModel, CKTcircuit *ckt)
{
    B4P(model) *head = (B4P(model) *)inModel;
    if (!head->B4P(AvxBuilt) || !head->B4P(AvxDynValid) ||
        (!B4P(isAvxEligibleTran)(ckt) && !B4P(isAvxEligibleDcOp)(ckt)))
        return -1;
    if (head->B4P(AvxFallbackCount) != 0)
        return -1;
    BSIM4ApplePack *pack = (BSIM4ApplePack *)head->B4P(AvxBlocks);
    if (!pack) return -1;

    const BSIM4AppleCheck * restrict checks = pack->checks;
    int nElig = pack->nElig;
    const double * restrict rhsOld = ckt->CKTrhsOld;
    const double reltol = ckt->CKTreltol;
    const double abstol = ckt->CKTabstol;

    const BSIM4AppleLatentState *lat = (const BSIM4AppleLatentState *)head->B4P(AvxLatent);
    const int *act_list = NULL;
    int n_iter = nElig;
    if (ckt->CKTmrSubActive && lat && lat->cfEpoch == ckt->CKTmrSubEpoch) {
        act_list = lat->cfFastList;
        n_iter = lat->cfNFast;
    } else if (lat && lat->nFrozen > 0) {
        act_list = lat->activeList;
        n_iter = lat->nActive;
    }

    for (int ii = 0; ii < n_iter; ii++) {
        int i = act_list ? act_list[ii] : ii;
        const BSIM4AppleCheck * restrict c = &checks[i];
        double vs = rhsOld[c->sNodePrime];
        double t  = c->type;
        double delvds = (rhsOld[c->dNodePrime] - vs) * t - c->last_vds;
        double delvgs = (rhsOld[c->gNodePrime] - vs) * t - c->last_vgs;
        double delvbs = (rhsOld[c->bNodePrime] - vs) * t - c->last_vbs;

        double cd_diff = c->cd_g * delvgs + c->cd_d * delvds + c->cd_b * delvbs;
        double cdhat = c->Idtot + cd_diff;
        double m0 = fabs(cdhat), m1 = fabs(c->Idtot);
        double tol0 = reltol * ((m0 > m1) ? m0 : m1) + abstol;
        if (fabs(cd_diff) >= tol0)
            return 1;

        double cb_diff = c->cb_g * delvgs + c->cb_d * delvds + c->cb_b * delvbs;
        double cbhat = c->Ibtot + cb_diff;
        double m2 = fabs(cbhat), m3 = fabs(c->Ibtot);
        double tol6 = reltol * ((m2 > m3) ? m2 : m3) + abstol;
        if (fabs(cb_diff) > tol6)
            return 1;
    }
    return 0;
}

/*
 * 2-wide NEON truncation error over BSIM4AppleCheck (zero AoS chase, zero fdiv/sqrt on 99.9% of pairs!).
 */
static inline float64x2_t b4a_t2_pair(double a, double b) { float64x2_t r = { a, b }; return r; }

static inline double
B4P(terrPairNeon)(
    int st0, int st1, int nact,
    const double * restrict s0, const double * restrict s1,
    const double * restrict s2, const double * restrict s3,
    const double * restrict s4,
    int ord,
    float64x2_t v_abstol, float64x2_t v_reltol, float64x2_t v_relInvD, float64x2_t v_chgtol,
    float64x2_t v_trtol, float64x2_t v_tfac,
    float64x2_t v_w0, float64x2_t v_w1, float64x2_t v_w2, float64x2_t v_w3, float64x2_t v_w4,
    double cur)
{
    static const int qoffs[3] = { 11, 13, 15 };
    if (ord == 3) {
        double cur_cu = (cur < 1.0e100) ? (cur * cur * cur) : 1.0e300;
        for (int c = 0; c < 3; c++) {
            int i0 = st0 + qoffs[c];
            int i1 = (nact > 1) ? (st1 + qoffs[c]) : i0;
            float64x2_t q0 = b4a_t2_pair(s0[i0], s0[i1]);
            float64x2_t q1 = b4a_t2_pair(s1[i0], s1[i1]);
            float64x2_t q2 = b4a_t2_pair(s2[i0], s2[i1]);
            float64x2_t q3 = b4a_t2_pair(s3[i0], s3[i1]);
            float64x2_t q4 = b4a_t2_pair(s4[i0], s4[i1]);
            float64x2_t c0 = vabsq_f64(b4a_t2_pair(s0[i0 + 1], s0[i1 + 1]));
            float64x2_t c1 = vabsq_f64(b4a_t2_pair(s1[i0 + 1], s1[i1 + 1]));
            float64x2_t volttol = vfmaq_f64(v_abstol, v_reltol, vmaxnmq_f64(c0, c1));
            float64x2_t qmax = vmaxnmq_f64(vabsq_f64(q0), vabsq_f64(q1));
            float64x2_t chargetol = vmulq_f64(v_relInvD, vmaxnmq_f64(qmax, v_chgtol));
            float64x2_t tol = vmaxnmq_f64(volttol, chargetol);
            float64x2_t d0 = vfmaq_f64(
                vfmaq_f64(vfmaq_f64(vfmaq_f64(vmulq_f64(v_w0, q0), v_w1, q1), v_w2, q2), v_w3, q3),
                v_w4, q4);
            float64x2_t den = vmaxnmq_f64(v_abstol, vmulq_f64(v_tfac, vabsq_f64(d0)));
            float64x2_t num = vmulq_f64(v_trtol, tol);
            if (__builtin_expect(cur >= 1.0e100 || vmaxvq_u32(vreinterpretq_u32_u64(vcltq_f64(num, vmulq_f64(vdupq_n_f64(cur_cu), den)))) != 0, 0)) {
                float64x2_t ratio = vdivq_f64(num, den);
                double r0 = vgetq_lane_f64(ratio, 0);
                if (r0 < cur_cu) { cur = cbrt(r0); cur_cu = r0; }
                if (nact > 1) {
                    double r1 = vgetq_lane_f64(ratio, 1);
                    if (r1 < cur_cu) { cur = cbrt(r1); cur_cu = r1; }
                }
            }
        }
    } else if (ord == 2) {
        double cur_sq = (cur < 1.0e150) ? (cur * cur) : 1.0e300;
        for (int c = 0; c < 3; c++) {
            int i0 = st0 + qoffs[c];
            int i1 = (nact > 1) ? (st1 + qoffs[c]) : i0;
            float64x2_t q0 = b4a_t2_pair(s0[i0], s0[i1]);
            float64x2_t q1 = b4a_t2_pair(s1[i0], s1[i1]);
            float64x2_t q2 = b4a_t2_pair(s2[i0], s2[i1]);
            float64x2_t q3 = b4a_t2_pair(s3[i0], s3[i1]);
            float64x2_t c0 = vabsq_f64(b4a_t2_pair(s0[i0 + 1], s0[i1 + 1]));
            float64x2_t c1 = vabsq_f64(b4a_t2_pair(s1[i0 + 1], s1[i1 + 1]));
            float64x2_t volttol = vfmaq_f64(v_abstol, v_reltol, vmaxnmq_f64(c0, c1));
            float64x2_t qmax = vmaxnmq_f64(vabsq_f64(q0), vabsq_f64(q1));
            float64x2_t chargetol = vmulq_f64(v_relInvD, vmaxnmq_f64(qmax, v_chgtol));
            float64x2_t tol = vmaxnmq_f64(volttol, chargetol);
            float64x2_t d0 = vfmaq_f64(vfmaq_f64(vfmaq_f64(vmulq_f64(v_w0, q0), v_w1, q1), v_w2, q2), v_w3, q3);
            float64x2_t den = vmaxnmq_f64(v_abstol, vmulq_f64(v_tfac, vabsq_f64(d0)));
            float64x2_t num = vmulq_f64(v_trtol, tol);
            if (__builtin_expect(cur >= 1.0e150 || vmaxvq_u32(vreinterpretq_u32_u64(vcltq_f64(num, vmulq_f64(vdupq_n_f64(cur_sq), den)))) != 0, 0)) {
                float64x2_t ratio = vdivq_f64(num, den);
                double r0 = vgetq_lane_f64(ratio, 0);
                if (r0 < cur_sq) { cur = sqrt(r0); cur_sq = r0; }
                if (nact > 1) {
                    double r1 = vgetq_lane_f64(ratio, 1);
                    if (r1 < cur_sq) { cur = sqrt(r1); cur_sq = r1; }
                }
            }
        }
    } else {
        for (int c = 0; c < 3; c++) {
            int i0 = st0 + qoffs[c];
            int i1 = (nact > 1) ? (st1 + qoffs[c]) : i0;
            float64x2_t q0 = b4a_t2_pair(s0[i0], s0[i1]);
            float64x2_t q1 = b4a_t2_pair(s1[i0], s1[i1]);
            float64x2_t q2 = b4a_t2_pair(s2[i0], s2[i1]);
            float64x2_t c0 = vabsq_f64(b4a_t2_pair(s0[i0 + 1], s0[i1 + 1]));
            float64x2_t c1 = vabsq_f64(b4a_t2_pair(s1[i0 + 1], s1[i1 + 1]));
            float64x2_t volttol = vfmaq_f64(v_abstol, v_reltol, vmaxnmq_f64(c0, c1));
            float64x2_t qmax = vmaxnmq_f64(vabsq_f64(q0), vabsq_f64(q1));
            float64x2_t chargetol = vmulq_f64(v_relInvD, vmaxnmq_f64(qmax, v_chgtol));
            float64x2_t tol = vmaxnmq_f64(volttol, chargetol);
            float64x2_t d0 = vfmaq_f64(vfmaq_f64(vmulq_f64(v_w0, q0), v_w1, q1), v_w2, q2);
            float64x2_t den = vmaxnmq_f64(v_abstol, vmulq_f64(v_tfac, vabsq_f64(d0)));
            float64x2_t num = vmulq_f64(v_trtol, tol);
            if (__builtin_expect(cur >= 1.0e150 || vmaxvq_u32(vreinterpretq_u32_u64(vcltq_f64(num, vmulq_f64(vdupq_n_f64(cur), den)))) != 0, 0)) {
                float64x2_t del = vdivq_f64(num, den);
                double d_0 = vgetq_lane_f64(del, 0);
                if (d_0 < cur) cur = d_0;
                if (nact > 1) {
                    double d_1 = vgetq_lane_f64(del, 1);
                    if (d_1 < cur) cur = d_1;
                }
            }
        }
    }
    return cur;
}

int
B4P(truncAvx)(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
    B4P(model) *head = (B4P(model) *)inModel;
    int ord = ckt->CKTorder;
    if (!head->B4P(AvxBuilt) || !ckt->CKTterrCacheValid || (ord != 1 && ord != 2 && ord != 3))
        return -1;
    BSIM4ApplePack *pack = (BSIM4ApplePack *)head->B4P(AvxBlocks);
    if (!pack) return -1;

    const BSIM4AppleCheck * restrict checks = pack->checks;
    int nElig = pack->nElig;
    const double * restrict s0 = ckt->CKTstate0;
    const double * restrict s1 = ckt->CKTstate1;
    const double * restrict s2 = ckt->CKTstates[2];
    const double * restrict s3 = (ord >= 2) ? ckt->CKTstates[3] : NULL;
    const double * restrict s4 = (ord >= 3) ? ckt->CKTstates[4] : NULL;
    const double *w = ckt->CKTterrWeights;

    float64x2_t v_abstol  = vdupq_n_f64(ckt->CKTabstol);
    float64x2_t v_reltol  = vdupq_n_f64(ckt->CKTreltol);
    float64x2_t v_relInvD = vdupq_n_f64(ckt->CKTterrReltolInvDelta);
    float64x2_t v_chgtol  = vdupq_n_f64(ckt->CKTchgtol);
    float64x2_t v_trtol   = vdupq_n_f64(ckt->CKTtrtol);
    float64x2_t v_tfac    = vdupq_n_f64(ckt->CKTterrFactor);
    float64x2_t v_w0      = vdupq_n_f64(w[0]);
    float64x2_t v_w1      = vdupq_n_f64(w[1]);
    float64x2_t v_w2      = vdupq_n_f64(w[2]);
    float64x2_t v_w3      = vdupq_n_f64(ord >= 2 ? w[3] : 0.0);
    float64x2_t v_w4      = vdupq_n_f64(ord >= 3 ? w[4] : 0.0);

    static int mr_trunc_init = 0, mr_trunc_en = 1;
    if (!mr_trunc_init) {
        const char *no_v = getenv("NGSPICE_NO_MR_TRUNC");
        const char *v = getenv("NGSPICE_MR_TRUNC");
        if (no_v != NULL && no_v[0] != '\0' && no_v[0] != '0')
            mr_trunc_en = 0;
        else if (v != NULL && v[0] != '\0')
            mr_trunc_en = (v[0] != '0');
        else
            mr_trunc_en = 1;
        mr_trunc_init = 1;
    }
    int active_cutoff = mr_trunc_en ? (ckt->CKTstat->STATaccepted - (ord + 2)) : -1000000;

    double minStep = *timeStep;
#ifdef USE_OMP
    if (nElig > 512 && omp_get_max_threads() > 1) {
        #pragma omp parallel for reduction(min:minStep)
        for (int i = 0; i < nElig; i += 2) {
            int a0 = !(mr_trunc_en && checks[i].phys_valid && checks[i].last_active_step < active_cutoff);
            int a1 = (i + 1 < nElig) &&
                     !(mr_trunc_en && checks[i + 1].phys_valid && checks[i + 1].last_active_step < active_cutoff);
            if (!a0 && !a1)
                continue;
            int st0 = a0 ? checks[i].stateBase : checks[i + 1].stateBase;
            int st1 = (a0 && a1) ? checks[i + 1].stateBase : st0;
            int nact = (a0 && a1) ? 2 : 1;
            minStep = B4P(terrPairNeon)(
                st0, st1, nact,
                s0, s1, s2, s3, s4, ord,
                v_abstol, v_reltol, v_relInvD, v_chgtol, v_trtol, v_tfac,
                v_w0, v_w1, v_w2, v_w3, v_w4, minStep);
        }
    } else
#endif
    {
        const BSIM4AppleLatentState *lat = (const BSIM4AppleLatentState *)head->B4P(AvxLatent);
        const int *act_list = NULL;
        int n_iter = nElig;
        if (ckt->CKTmrSubActive && lat && lat->cfEpoch == ckt->CKTmrSubEpoch) {
            act_list = lat->cfFastList;
            n_iter = lat->cfNFast;
        } else if (lat && lat->nFrozen > 0) {
            act_list = lat->activeList;
            n_iter = lat->nActive;
        }
        int pend_st = -1;
        for (int ii = 0; ii < n_iter; ii++) {
            int i = act_list ? act_list[ii] : ii;
            if (mr_trunc_en && checks[i].phys_valid && checks[i].last_active_step < active_cutoff)
                continue;
            int st = checks[i].stateBase;
            if (pend_st < 0) {
                pend_st = st;
            } else {
                minStep = B4P(terrPairNeon)(
                    pend_st, st, 2,
                    s0, s1, s2, s3, s4, ord,
                    v_abstol, v_reltol, v_relInvD, v_chgtol, v_trtol, v_tfac,
                    v_w0, v_w1, v_w2, v_w3, v_w4, minStep);
                pend_st = -1;
            }
        }
        if (pend_st >= 0) {
            minStep = B4P(terrPairNeon)(
                pend_st, pend_st, 1,
                s0, s1, s2, s3, s4, ord,
                v_abstol, v_reltol, v_relInvD, v_chgtol, v_trtol, v_tfac,
                v_w0, v_w1, v_w2, v_w3, v_w4, minStep);
        }
    }

    int nFb = head->B4P(AvxFallbackCount);
    if (nFb > 0) {
        const int *fbIdx = head->B4P(AvxFallbackIdx);
        B4P(instance) **InstArray = head->B4P(InstanceArray);
        for (int fi = 0; fi < nFb; fi++) {
            B4P(instance) *here = InstArray[fbIdx[fi]];
            CKTterr(here->B4P(qb), ckt, &minStep);
            CKTterr(here->B4P(qg), ckt, &minStep);
            CKTterr(here->B4P(qd), ckt, &minStep);
        }
    }

    *timeStep = minStep;
    return OK;
}

#endif /* NGSPICE_OPT_APPLE */
