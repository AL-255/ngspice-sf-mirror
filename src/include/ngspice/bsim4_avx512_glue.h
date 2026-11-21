/*
 * Per-version glue (BSIM4v5 / BSIM4v6) for 8-wide AVX-512 SoA evaluation.
 * Included inside b4v5ld.c (with B4_IS_V6=0) and b4v6ld.c (with B4_IS_V6=1).
 *
 * Expects macros:
 *   B4P(name) -> BSIM4v5##name  or  BSIM4v6##name
 *   B4_IS_V6  -> 0 or 1
 */
#if NGSPICE_OPT_ZEN4

#include "ngspice/bsim4_avx512_core.h"

int B4P(AvxNeedAosSync) = 0;

static B4P(instance) **B4P(sort_inst_arr) = NULL;

void
B4P(freeAvxLatent)(void *p)
{
    BSIM4AvxLatentState *lat = (BSIM4AvxLatentState *)p;
    if (!lat) return;
    if (getenv("NGSPICE_MR_STATS") != NULL && lat->statCalls > 0) {
        double avg_f = (double)lat->statFrozenSum / (double)lat->statCalls;
        double pct = 100.0 * avg_f / (double)(lat->nBlocks > 0 ? lat->nBlocks : 1);
        fprintf(stderr,
                "[MR_LATENT] blocks=%d calls=%llu avg_frozen=%.1f (%.1f%%) freezes=%llu wakes=%llu resets=%llu fail(val=%llu icap=%llu cache=%llu snap=%llu)\n",
                lat->nBlocks, lat->statCalls, avg_f, pct,
                lat->statFreezes, lat->statWakes, lat->statResets,
                lat->statFailValid, lat->statFailIcap, lat->statFailCache, lat->statFailSnap);
    }
    free(lat->activeList);
    free(lat->activePos);
    free(lat->nodeBlkPtr);
    free(lat->nodeBlkIdx);
    free(lat->vSnap);
    free(lat->nodeFrozenCnt);
    free(lat->cfFastList);
    free(lat->cfIsFastBlk);
    free(lat);
}

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

static int B4P(sort_by_name) = 0;

static int
B4P(cmpEligIdx)(const void *pa, const void *pb)
{
    int ia = *(const int *)pa;
    int ib = *(const int *)pb;
    B4P(instance) *ha = B4P(sort_inst_arr)[ia];
    B4P(instance) *hb = B4P(sort_inst_arr)[ib];
    B4P(model) *ma = B4P(modPtr)(ha);
    B4P(model) *mb = B4P(modPtr)(hb);
    if (ma != mb) {
        int mc = strcmp(ma->gen.GENmodName, mb->gen.GENmodName);
        if (mc != 0) return mc;
    }
    int ta = ha->B4P(hasTrapTunnel) ? 1 : 0;
    int tb = hb->B4P(hasTrapTunnel) ? 1 : 0;
    if (ta != tb) return ta - tb;
    int da = (ha->B4P(dNode) != ha->B4P(dNodePrime)) ? 1 : 0;
    int db = (hb->B4P(dNode) != hb->B4P(dNodePrime)) ? 1 : 0;
    if (da != db) return da - db;
    int sa = (ha->B4P(sNode) != ha->B4P(sNodePrime)) ? 1 : 0;
    int sb = (hb->B4P(sNode) != hb->B4P(sNodePrime)) ? 1 : 0;
    if (sa != sb) return sa - sb;
    if (B4P(sort_by_name) && ha->B4P(name) && hb->B4P(name)) {
        int nc = strcmp(ha->B4P(name), hb->B4P(name));
        if (nc != 0) return nc;
    }
    return ia - ib;
}

static inline int
B4P(isAvxEligibleTran)(CKTcircuit *ckt)
{
    static int disabled = -1;
    if (disabled < 0)
        disabled = (getenv("NGSPICE_NO_AVX512") != NULL) ? 1 : 0;
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
    if (disabled < 0)
        disabled = (getenv("NGSPICE_NO_AVX512") != NULL) ? 1 : 0;
    if (disabled)
        return 0;

    long mode = ckt->CKTmode;
    if (!(mode & (MODEDCOP | MODETRANOP)))
        return 0;
    if (!(mode & (MODEINITFLOAT | MODEINITJCT | MODEINITFIX)))
        return 0;
    if (mode & (MODEINITSMSIG | MODEINITTRAN | MODEUIC | MODEDCTRANCURVE | MODEAC | MODETRAN))
        return 0;
    if (ckt->CKTbypass)
        return 0;
    return 1;
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

    free(head->B4P(AvxBlocks));
    head->B4P(AvxBlocks) = NULL;
    free(head->B4P(AvxDyn));
    head->B4P(AvxDyn) = NULL;
    B4P(freeAvxLatent)(head->B4P(AvxLatent));
    head->B4P(AvxLatent) = NULL;
    head->B4P(AvxDynValid) = 0;
    FREE(head->B4P(AvxFallbackIdx));
    head->B4P(AvxBlockCount) = 0;
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

    int *eligIdx = TMALLOC(int, count);
    int *fbIdx   = TMALLOC(int, count);
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
        if (ok)
            eligIdx[nElig++] = idx;
        else
            fbIdx[nFb++] = idx;
    }

    if (nElig > 1) {
        B4P(sort_inst_arr) = InstArray;
        B4P(sort_by_name) = (getenv("NGSPICE_MR_PACK") != NULL);
        qsort(eligIdx, (size_t)nElig, sizeof(int), B4P(cmpEligIdx));
    }

    int nBlocks = (nElig + 7) / 8;
    BSIM4Avx512Block *blocks = NULL;
    BSIM4Avx512Dyn *dyn = NULL;
    BSIM4Avx512Inst *insts = NULL;
    if (nBlocks > 0) {
        size_t blk_bytes = (size_t)nBlocks * sizeof(BSIM4Avx512Block);
        size_t inst_bytes = (nBlocks > 4) ? ((size_t)nBlocks * 8u * sizeof(BSIM4Avx512Inst)) : 0u;
        if (posix_memalign((void **)&blocks, 64, blk_bytes + inst_bytes) != 0)
            blocks = NULL;
        if (blocks && posix_memalign((void **)&dyn, 64, (size_t)nBlocks * sizeof(BSIM4Avx512Dyn)) != 0) {
            free(blocks);
            blocks = NULL;
            dyn = NULL;
        }
        if (blocks && inst_bytes > 0)
            insts = (BSIM4Avx512Inst *)(blocks + nBlocks);
        if (dyn)
            memset(dyn, 0, (size_t)nBlocks * sizeof(BSIM4Avx512Dyn));
    }
    if (nBlocks > 0 && blocks == NULL) {
        /* Fallback all if allocation failed */
        FREE(eligIdx);
        for (int idx = 0; idx < count; idx++) fbIdx[idx] = idx;
        head->B4P(AvxFallbackIdx) = fbIdx;
        head->B4P(AvxFallbackCount) = count;
        head->B4P(AvxBuiltTemp) = ckt->CKTtemp;
        head->B4P(AvxKluAxBase) = ax;
        head->B4P(AvxKluNz) = nz;
        head->B4P(AvxBuilt) = 1;
        return;
    }

    uintptr_t ax_addr = (uintptr_t)ax;
    size_t ax_bytes = (size_t)nz * sizeof(double);
    int inz = (int)nz;

    for (int bi = 0; bi < nBlocks; bi++) {
        BSIM4Avx512Block *blk = &blocks[bi];
        memset(blk, 0, sizeof(*blk));
        int base = bi * 8;
        int nact = nElig - base;
        if (nact > 8) nact = 8;
        blk->n_active = (uint8_t)nact;
        blk->is_v6 = B4_IS_V6;

        double t_type[8], t_m[8], t_nf[8], t_vth0s[8], t_vfbphi[8], t_vfbzb[8], t_vbsc[8];
        double t_phi[8], t_phi095[8], t_sqrtPhi[8], t_V0[8], t_vtm[8], t_coxe[8], t_invCoxe[8];
        double t_coxp[8], t_toxe[8], t_invToxe[8], t_toxp[8], t_toxctm[8], t_invtoxctm[8];
        double t_inv2e8[8], t_link[8], t_leff[8], t_invLeff[8], t_sqrtLeff[8], t_weff[8];
        double t_weffCV[8], t_leffCV[8], t_CoxWL[8], t_CoxWL_oc[8], t_gdpr[8], t_gspr[8];
        double t_mgd[8], t_mgs[8], t_ados[8], t_bdos[8];
        double t_iNvtms[8], t_iNvtmd[8], t_Ssat[8], t_Dsat[8], t_vjsm[8], t_vjdm[8], t_IVjsm[8], t_IVjdm[8];
        double t_iNvrs[8], t_iNvrd[8], t_iNvrssw[8], t_iNvrdsw[8], t_iNvrsswg[8], t_iNvrdswg[8];
        double t_vtss[8], t_vtsd[8], t_vtssws[8], t_vtsswd[8], t_vtsswgs[8], t_vtsswgd[8];
        double t_SjRev[8], t_DjRev[8], t_SswRev[8], t_DswRev[8], t_SswgRev[8], t_DswgRev[8];
        double t_xdep0s[8], t_fac1[8], t_dvt0[8], t_dvt1l[8], t_dvt2[8], t_dvt0w[8], t_dvt1wwl[8], t_dvt2w[8];
        double t_k1[8], t_k1ox[8], t_k2ox[8], t_k3[8], t_k3b[8], t_lpeb[8], t_vthNW[8], t_vthC[8], t_kt2T[8];
        double t_eta0[8], t_etab[8], t_th0vb0[8], t_nfac[8], t_cit[8], t_cdsc[8], t_cdscb[8], t_cdscd[8];
        double t_dvtp0[8], t_dvtp1[8], t_polyT1[8], t_mstar[8], t_voffcbn[8], t_coxeCdep[8];
        double t_dwg[8], t_dwb[8], t_prwg[8], t_prwb[8], t_rds0h[8], t_rdswmin[8], t_xj[8], t_a0[8];
        double t_b0wb1[8], t_agsa0[8], t_keta[8], t_abCVf[8];
        double t_ua[8], t_ub[8], t_uc[8], t_ud[8], t_u0t[8], t_vsatt[8], t_a1[8], t_a2[8], t_delta[8];
        double t_lam[8], t_litl[8], t_fprout[8], t_pvag[8], t_pclm[8], t_ipclml[8], t_thRout[8], t_ithRout[8];
        double t_pdiblb[8], t_pdits[8], t_ipdits[8], t_pditsd[8], t_pditsl[8], t_pscbe1l[8], t_ipscbe2[8];
        double t_alpl[8], t_beta0[8], t_ivtl[8], t_vtfb2[8];
        double t_i3toxe[8], t_agidlw[8], t_bgidl[8], t_cgidl[8], t_egidl[8];
        double t_agislw[8], t_bgisl[8], t_cgisl[8], t_egisl[8];
        double t_noff[8], t_voffcv[8], t_acde[8], t_ldeb[8], t_moin[8], t_cgdo[8], t_cgso[8], t_cgbo[8];
        double t_wcgdl[8], t_ckd[8], t_ickd[8], t_wcgsl[8], t_cks[8], t_icks[8];
        double t_czbs[8], t_czbssw[8], t_czbsswg[8], t_czbd[8], t_czbdsw[8], t_czbdswg[8];
        double t_PBS[8], t_iPBS[8], t_PBSWS[8], t_iPBSWS[8], t_PBSWGS[8], t_iPBSWGS[8];
        double t_PBD[8], t_iPBD[8], t_PBSWD[8], t_iPBSWD[8], t_PBSWGD[8], t_iPBSWGD[8];
        double t_MJS[8], t_i1MJS[8], t_MJSWS[8], t_i1MJSWS[8], t_MJSWGS[8], t_i1MJSWGS[8];
        double t_MJD[8], t_i1MJD[8], t_MJSWD[8], t_i1MJSWD[8], t_MJSWGD[8], t_i1MJSWGD[8];
        double t_jSsum[8], t_jDsum[8], t_jSlin[8], t_jDlin[8];

        int all_xp_lt05 = 1, all_xp_gt05 = 1, all_xp_eq05 = 1;
        int has_ud = 0;

        for (int lane = 0; lane < 8; lane++) {
            int src_idx = (lane < nact) ? eligIdx[base + lane] : eligIdx[base];
            B4P(instance) *h = InstArray[src_idx];
            B4P(model) *m = B4P(modPtr)(h);
#if B4_IS_V6
            struct bsim4v6SizeDependParam *p = h->pParam;
#else
            struct bsim4v5SizeDependParam *p = h->pParam;
#endif
            __mmask8 bit = (__mmask8)(1u << lane);

            double leff = p->B4P(leff);
            double TempRatio = ckt->CKTtemp / m->B4P(tnom) - 1.0;

            t_type[lane]     = (double)m->B4P(type);
            t_m[lane]        = h->B4P(m);
            t_nf[lane]       = h->B4P(nf);
            if (h->B4P(nf) != 1.0) blk->m_nf_ne1 |= bit;
            t_vth0s[lane]    = (double)m->B4P(type) * h->B4P(vth0);
            t_vfbphi[lane]   = h->B4P(vfb) + p->B4P(phi);
            t_vfbzb[lane]    = h->B4P(vfbzb);
            if (h->B4P(vfbzb) <= 0.0) blk->m_vfbzb_le0 |= bit;
            t_vbsc[lane]     = h->B4P(vbsc);
            t_phi[lane]      = p->B4P(phi);
            t_phi095[lane]   = 0.95 * p->B4P(phi);
            t_sqrtPhi[lane]  = p->B4P(sqrtPhi);
            t_V0[lane]       = p->B4P(vbi) - p->B4P(phi);
            t_vtm[lane]      = m->B4P(vtm);
            t_coxe[lane]     = m->B4P(coxe);
            t_invCoxe[lane]  = 1.0 / m->B4P(coxe);
            t_coxp[lane]     = m->B4P(coxp);
            t_toxe[lane]     = m->B4P(toxe);
            t_invToxe[lane]  = 1.0 / m->B4P(toxe);
            t_toxp[lane]     = m->B4P(toxp);
            t_toxctm[lane]   = 1.0e8 * m->B4P(toxp);
            t_invtoxctm[lane]= 1.0 / (1.0e8 * m->B4P(toxp));
            t_inv2e8[lane]   = 1.0 / (2.0e8 * m->B4P(toxp));
            t_link[lane]     = 1.0e-3 * m->B4P(toxp);
            t_leff[lane]     = leff;
            t_invLeff[lane]  = 1.0 / leff;
            t_sqrtLeff[lane] = p->B4P(sqrtLeff);
            t_weff[lane]     = p->B4P(weff);
            t_weffCV[lane]   = p->B4P(weffCV);
            t_leffCV[lane]   = p->B4P(leffCV);
            t_CoxWL_oc[lane] = p->B4P(weffCV) * p->B4P(leffCV) * h->B4P(nf);
            t_CoxWL[lane]    = m->B4P(coxe) * t_CoxWL_oc[lane];
            t_gdpr[lane]     = h->B4P(drainConductance);
            t_gspr[lane]     = h->B4P(sourceConductance);
            t_mgd[lane]      = h->B4P(m) * h->B4P(drainConductance);
            t_mgs[lane]      = h->B4P(m) * h->B4P(sourceConductance);
#if B4_IS_V6
            t_ados[lane]     = 1.9e-9 * m->B4P(ados);
            t_bdos[lane]     = 0.7 * m->B4P(bdos);
#else
            t_ados[lane]     = 1.9e-9;
            t_bdos[lane]     = 0.7;
#endif

            t_iNvtms[lane] = 1.0 / (m->B4P(vtm) * m->B4P(SjctEmissionCoeff));
            t_iNvtmd[lane] = 1.0 / (m->B4P(vtm) * m->B4P(DjctEmissionCoeff));
            t_Ssat[lane]   = h->B4P(SourceSatCurrent);
            t_Dsat[lane]   = h->B4P(DrainSatCurrent);
            if (h->B4P(SourceSatCurrent) > 0.0) blk->m_src_sat_pos |= bit;
            if (h->B4P(DrainSatCurrent) > 0.0)  blk->m_drn_sat_pos |= bit;
            t_vjsm[lane]   = h->B4P(vjsmFwd);
            t_vjdm[lane]   = h->B4P(vjdmFwd);
            t_IVjsm[lane]  = h->B4P(IVjsmFwd);
            t_IVjdm[lane]  = h->B4P(IVjdmFwd);

            if (h->B4P(hasTrapTunnel)) {
                blk->has_trap_tunnel = 1;
#if B4_IS_V6
                double nvs   = m->B4P(vtm0) * m->B4P(njtsstemp);
                double nvd   = m->B4P(vtm0) * m->B4P(njtsdtemp);
                double nvsws = m->B4P(vtm0) * m->B4P(njtsswstemp);
                double nvswd = m->B4P(vtm0) * m->B4P(njtsswdtemp);
                double nvswgs= m->B4P(vtm0) * m->B4P(njtsswgstemp);
                double nvswgd= m->B4P(vtm0) * m->B4P(njtsswgdtemp);
#else
                double nvs   = m->B4P(vtm0) * m->B4P(njtstemp);
                double nvd   = nvs;
                double nvsws = m->B4P(vtm0) * m->B4P(njtsswtemp);
                double nvswd = nvsws;
                double nvswgs= m->B4P(vtm0) * m->B4P(njtsswgtemp);
                double nvswgd= nvswgs;
#endif
                t_iNvrs[lane]    = (nvs != 0.0) ? (1.0 / nvs) : 0.0;
                t_iNvrd[lane]    = (nvd != 0.0) ? (1.0 / nvd) : 0.0;
                t_iNvrssw[lane]  = (nvsws != 0.0) ? (1.0 / nvsws) : 0.0;
                t_iNvrdsw[lane]  = (nvswd != 0.0) ? (1.0 / nvswd) : 0.0;
                t_iNvrsswg[lane] = (nvswgs != 0.0) ? (1.0 / nvswgs) : 0.0;
                t_iNvrdswg[lane] = (nvswgd != 0.0) ? (1.0 / nvswgd) : 0.0;
                t_vtss[lane]     = m->B4P(vtss);
                t_vtsd[lane]     = m->B4P(vtsd);
                t_vtssws[lane]   = m->B4P(vtssws);
                t_vtsswd[lane]   = m->B4P(vtsswd);
                t_vtsswgs[lane]  = m->B4P(vtsswgs);
                t_vtsswgd[lane]  = m->B4P(vtsswgd);
                t_SjRev[lane]    = h->B4P(SjctTempRevSatCur);
                t_DjRev[lane]    = h->B4P(DjctTempRevSatCur);
                t_SswRev[lane]   = h->B4P(SswTempRevSatCur);
                t_DswRev[lane]   = h->B4P(DswTempRevSatCur);
                t_SswgRev[lane]  = h->B4P(SswgTempRevSatCur);
                t_DswgRev[lane]  = h->B4P(DswgTempRevSatCur);
                if (t_SjRev[lane]   != 0.0) blk->m_tt_s    |= bit;
                if (t_DjRev[lane]   != 0.0) blk->m_tt_d    |= bit;
                if (t_SswRev[lane]  != 0.0) blk->m_tt_ssw  |= bit;
                if (t_DswRev[lane]  != 0.0) blk->m_tt_dsw  |= bit;
                if (t_SswgRev[lane] != 0.0) blk->m_tt_sswg |= bit;
                if (t_DswgRev[lane] != 0.0) blk->m_tt_dswg |= bit;
            } else {
                t_iNvrs[lane] = t_iNvrd[lane] = t_iNvrssw[lane] = t_iNvrdsw[lane] = 0.0;
                t_iNvrsswg[lane] = t_iNvrdswg[lane] = 0.0;
                t_vtss[lane] = t_vtsd[lane] = t_vtssws[lane] = t_vtsswd[lane] = t_vtsswgs[lane] = t_vtsswgd[lane] = 10.0;
                t_SjRev[lane] = t_DjRev[lane] = t_SswRev[lane] = t_DswRev[lane] = t_SswgRev[lane] = t_DswgRev[lane] = 0.0;
            }

            t_xdep0s[lane]  = p->B4P(xdep0OverSqrtPhi);
            t_fac1[lane]    = m->B4P(factor1);
            t_dvt0[lane]    = p->B4P(dvt0);
            t_dvt1l[lane]   = p->B4P(dvt1) * leff;
            t_dvt2[lane]    = p->B4P(dvt2);
            t_dvt0w[lane]   = p->B4P(dvt0w);
            t_dvt1wwl[lane] = p->B4P(dvt1w) * p->B4P(weff) * leff;
            t_dvt2w[lane]   = p->B4P(dvt2w);
            if (p->B4P(dvt0w) != 0.0) blk->m_dvt0w |= bit;

            t_k1[lane]     = p->B4P(k1);
            t_k1ox[lane]   = p->B4P(k1ox);
            if (p->B4P(k1ox) == 0.0) blk->m_k1ox_zero |= bit;
            if (p->B4P(k1ox) <= 0.0) blk->m_k1ox_le0 |= bit;
            t_k2ox[lane]   = h->B4P(k2ox);
            t_k3[lane]     = p->B4P(k3);
            t_k3b[lane]    = p->B4P(k3b);
            t_lpeb[lane]   = p->B4P(lpebFactor);
            t_vthNW[lane]  = p->B4P(vthNarrowW);
            t_vthC[lane]   = (double)m->B4P(type) * h->B4P(vth0)
                           - p->B4P(k1) * p->B4P(sqrtPhi) * p->B4P(lpebFactor)
                           + p->B4P(k3) * p->B4P(vthNarrowW)
                           + p->B4P(k1ox) * (p->B4P(lpe0Factor) - 1.0) * p->B4P(sqrtPhi)
                           + (p->B4P(kt1) + p->B4P(kt1l) / leff) * TempRatio;
            t_kt2T[lane]   = p->B4P(kt2) * TempRatio;
            t_eta0[lane]   = h->B4P(eta0);
            t_etab[lane]   = p->B4P(etab);
            t_th0vb0[lane] = p->B4P(theta0vb0);
            t_nfac[lane]   = p->B4P(nfactor);
            t_cit[lane]    = p->B4P(cit);
            t_cdsc[lane]   = p->B4P(cdsc);
            t_cdscb[lane]  = p->B4P(cdscb);
            t_cdscd[lane]  = p->B4P(cdscd);
            t_dvtp0[lane]  = p->B4P(dvtp0);
            t_dvtp1[lane]  = p->B4P(dvtp1);
            if (p->B4P(dvtp0) > 0.0) blk->m_dvtp0 |= bit;

            if (p->B4P(ngate) > 1.0e18 && p->B4P(ngate) < 1.0e25) {
                blk->m_ngate_valid |= bit;
                t_polyT1[lane] = 1.0e6 * CHARGE * EPSSI * p->B4P(ngate) / (m->B4P(coxe) * m->B4P(coxe));
            } else {
                t_polyT1[lane] = 1.0;
            }
            t_mstar[lane]    = p->B4P(mstar);
            t_voffcbn[lane]  = p->B4P(voffcbn);
            t_coxeCdep[lane] = m->B4P(coxe) / p->B4P(cdep0);

            t_dwg[lane]    = p->B4P(dwg);
            t_dwb[lane]    = p->B4P(dwb);
            t_prwg[lane]   = p->B4P(prwg);
            t_prwb[lane]   = p->B4P(prwb);
            t_rds0h[lane]  = 0.5 * p->B4P(rds0);
            t_rdswmin[lane]= p->B4P(rdswmin);
            t_xj[lane]     = p->B4P(xj);
            t_a0[lane]     = p->B4P(a0);
            t_b0wb1[lane]  = p->B4P(b0) / (p->B4P(weff) + p->B4P(b1));
            t_agsa0[lane]  = p->B4P(ags) * p->B4P(a0);
            t_keta[lane]   = p->B4P(keta);
            t_abCVf[lane]  = p->B4P(abulkCVfactor);

            t_ua[lane]     = p->B4P(ua);
            t_ub[lane]     = p->B4P(ub);
            t_uc[lane]     = p->B4P(uc);
            t_ud[lane]     = p->B4P(ud);
            if (p->B4P(ud) != 0.0) has_ud = 1;
            t_u0t[lane]    = h->B4P(u0temp);
            t_vsatt[lane]  = h->B4P(vsattemp);
            t_a1[lane]     = p->B4P(a1);
            if (p->B4P(a1) != 0.0) blk->m_has_a1 |= bit;
            t_a2[lane]     = p->B4P(a2);
            t_delta[lane]  = p->B4P(delta);
            t_lam[lane]    = p->B4P(lambda);
            t_litl[lane]   = p->B4P(litl);
            if (m->B4P(lambdaGiven) && m->B4P(lambda) > 0.0) blk->m_lambda |= bit;
            t_fprout[lane] = p->B4P(fprout);
            if (p->B4P(fprout) > 0.0) blk->m_fprout |= bit;
            t_pvag[lane]   = p->B4P(pvag);
            t_pclm[lane]   = p->B4P(pclm);
            if (p->B4P(pclm) > B4V_MIN_EXP) {
                blk->m_pclm |= bit;
                t_ipclml[lane] = 1.0 / (p->B4P(pclm) * p->B4P(litl));
            } else {
                t_ipclml[lane] = 0.0;
            }
            t_thRout[lane] = p->B4P(thetaRout);
            if (p->B4P(thetaRout) > B4V_MIN_EXP) {
                blk->m_thetaRout |= bit;
                t_ithRout[lane] = 1.0 / p->B4P(thetaRout);
            } else {
                t_ithRout[lane] = 0.0;
            }
            t_pdiblb[lane] = p->B4P(pdiblb);
            t_pdits[lane]  = p->B4P(pdits);
            t_pditsd[lane] = p->B4P(pditsd);
            t_pditsl[lane] = 1.0 + m->B4P(pditsl) * leff;
            if (p->B4P(pdits) > B4V_MIN_EXP) {
                blk->m_pdits |= bit;
                t_ipdits[lane] = 1.0 / p->B4P(pdits);
            } else {
                t_ipdits[lane] = 0.0;
            }
            t_pscbe1l[lane] = p->B4P(pscbe1) * p->B4P(litl);
#if B4_IS_V6
            if (p->B4P(pscbe2) > 0.0 && p->B4P(pscbe1) >= 0.0) {
#else
            if (p->B4P(pscbe2) > 0.0) {
#endif
                blk->m_pscbe |= bit;
                t_ipscbe2[lane] = 1.0 / p->B4P(pscbe2);
            } else {
                t_ipscbe2[lane] = 0.0;
            }
            double atmp = p->B4P(alpha0) + p->B4P(alpha1) * leff;
            if (atmp > 0.0 && p->B4P(beta0) > 0.0) {
                blk->m_isub |= bit;
                t_alpl[lane] = atmp / leff;
            } else {
                t_alpl[lane] = 0.0;
            }
            t_beta0[lane] = p->B4P(beta0);
            if (m->B4P(vtlGiven) && m->B4P(vtl) > 0.0) {
                blk->m_vtl |= bit;
                t_ivtl[lane] = 1.0 / (p->B4P(vtl) * p->B4P(tfactor));
            } else {
                t_ivtl[lane] = 0.0;
            }
            t_vtfb2[lane] = h->B4P(vtfbphi2);

            t_i3toxe[lane] = 1.0 / (3.0 * m->B4P(toxe));
            t_agidlw[lane] = p->B4P(agidl) * p->B4P(weffCJ);
            t_bgidl[lane]  = p->B4P(bgidl);
            t_cgidl[lane]  = p->B4P(cgidl);
            t_egidl[lane]  = p->B4P(egidl);
            if (p->B4P(agidl) > 0.0 && p->B4P(bgidl) > 0.0 && p->B4P(cgidl) > 0.0)
                blk->m_gidl_en |= bit;
#if B4_IS_V6
            t_agislw[lane] = p->B4P(agisl) * p->B4P(weffCJ);
            t_bgisl[lane]  = p->B4P(bgisl);
            t_cgisl[lane]  = p->B4P(cgisl);
            t_egisl[lane]  = p->B4P(egisl);
            if (p->B4P(agisl) > 0.0 && p->B4P(bgisl) > 0.0 && p->B4P(cgisl) > 0.0)
                blk->m_gisl_en |= bit;
#else
            t_agislw[lane] = t_agidlw[lane];
            t_bgisl[lane]  = t_bgidl[lane];
            t_cgisl[lane]  = t_cgidl[lane];
            t_egisl[lane]  = t_egidl[lane];
            if (p->B4P(agidl) > 0.0 && p->B4P(bgidl) > 0.0 && p->B4P(cgidl) > 0.0)
                blk->m_gisl_en |= bit;
#endif

            t_noff[lane]   = p->B4P(noff);
            t_voffcv[lane] = p->B4P(voffcv);
            t_acde[lane]   = p->B4P(acde);
            t_ldeb[lane]   = p->B4P(ldeb);
            t_moin[lane]   = p->B4P(moin);
            if (m->B4P(xpart) > 0.5) {
                blk->m_xpart_gt05 |= bit;
                all_xp_lt05 = 0; all_xp_eq05 = 0;
            } else if (m->B4P(xpart) < 0.5) {
                blk->m_xpart_lt05 |= bit;
                all_xp_gt05 = 0; all_xp_eq05 = 0;
            } else {
                all_xp_lt05 = 0; all_xp_gt05 = 0;
            }
            t_cgdo[lane]  = p->B4P(cgdo);
            t_cgso[lane]  = p->B4P(cgso);
            t_cgbo[lane]  = p->B4P(cgbo);
            t_wcgdl[lane] = p->B4P(weffCV) * p->B4P(cgdl);
            t_ckd[lane]   = p->B4P(ckappad);
            t_ickd[lane]  = 1.0 / p->B4P(ckappad);
            t_wcgsl[lane] = p->B4P(weffCV) * p->B4P(cgsl);
            t_cks[lane]   = p->B4P(ckappas);
            t_icks[lane]  = 1.0 / p->B4P(ckappas);

            double czbs    = m->B4P(SunitAreaTempJctCap) * h->B4P(Aseff);
            double czbssw  = m->B4P(SunitLengthSidewallTempJctCap) * h->B4P(Pseff);
            double czbsswg = m->B4P(SunitLengthGateSidewallTempJctCap) * p->B4P(weffCJ) * h->B4P(nf);
            double czbd    = m->B4P(DunitAreaTempJctCap) * h->B4P(Adeff);
            double czbdsw  = m->B4P(DunitLengthSidewallTempJctCap) * h->B4P(Pdeff);
            double czbdswg = m->B4P(DunitLengthGateSidewallTempJctCap) * p->B4P(weffCJ) * h->B4P(nf);
            t_czbs[lane] = czbs; t_czbssw[lane] = czbssw; t_czbsswg[lane] = czbsswg;
            t_czbd[lane] = czbd; t_czbdsw[lane] = czbdsw; t_czbdswg[lane] = czbdswg;
            if (czbs > 0.0)    blk->m_czbs |= bit;
            if (czbssw > 0.0)  blk->m_czbssw |= bit;
            if (czbsswg > 0.0) blk->m_czbsswg |= bit;
            if (czbd > 0.0)    blk->m_czbd |= bit;
            if (czbdsw > 0.0)  blk->m_czbdsw |= bit;
            if (czbdswg > 0.0) blk->m_czbdswg |= bit;

            t_PBS[lane]    = m->B4P(PhiBS);    t_iPBS[lane]    = 1.0 / m->B4P(PhiBS);
            t_PBSWS[lane]  = m->B4P(PhiBSWS);  t_iPBSWS[lane]  = 1.0 / m->B4P(PhiBSWS);
            t_PBSWGS[lane] = m->B4P(PhiBSWGS); t_iPBSWGS[lane] = 1.0 / m->B4P(PhiBSWGS);
            t_PBD[lane]    = m->B4P(PhiBD);    t_iPBD[lane]    = 1.0 / m->B4P(PhiBD);
            t_PBSWD[lane]  = m->B4P(PhiBSWD);  t_iPBSWD[lane]  = 1.0 / m->B4P(PhiBSWD);
            t_PBSWGD[lane] = m->B4P(PhiBSWGD); t_iPBSWGD[lane] = 1.0 / m->B4P(PhiBSWGD);

            double mjs = m->B4P(SbulkJctBotGradingCoeff);
            double mjsws = m->B4P(SbulkJctSideGradingCoeff);
            double mjswgs = m->B4P(SbulkJctGateSideGradingCoeff);
            double mjd = m->B4P(DbulkJctBotGradingCoeff);
            double mjswd = m->B4P(DbulkJctSideGradingCoeff);
            double mjswgd = m->B4P(DbulkJctGateSideGradingCoeff);
            t_MJS[lane] = mjs; t_i1MJS[lane] = 1.0 / (1.0 - mjs);
            t_MJSWS[lane] = mjsws; t_i1MJSWS[lane] = 1.0 / (1.0 - mjsws);
            t_MJSWGS[lane] = mjswgs; t_i1MJSWGS[lane] = 1.0 / (1.0 - mjswgs);
            t_MJD[lane] = mjd; t_i1MJD[lane] = 1.0 / (1.0 - mjd);
            t_MJSWD[lane] = mjswd; t_i1MJSWD[lane] = 1.0 / (1.0 - mjswd);
            t_MJSWGD[lane] = mjswgd; t_i1MJSWGD[lane] = 1.0 / (1.0 - mjswgd);
            if (mjs == 0.5)    blk->m_mjs_half |= bit;
            if (mjsws == 0.5)  blk->m_mjsws_half |= bit;
            if (mjswgs == 0.5) blk->m_mjswgs_half |= bit;
            if (mjd == 0.5)    blk->m_mjd_half |= bit;
            if (mjswd == 0.5)  blk->m_mjswd_half |= bit;
            if (mjswgd == 0.5) blk->m_mjswgd_half |= bit;

            t_jSsum[lane] = czbs + czbssw + czbsswg;
            t_jDsum[lane] = czbd + czbdsw + czbdswg;
            t_jSlin[lane] = czbs * mjs * t_iPBS[lane] + czbssw * mjsws * t_iPBSWS[lane] + czbsswg * mjswgs * t_iPBSWGS[lane];
            t_jDlin[lane] = czbd * mjd * t_iPBD[lane] + czbdsw * mjswd * t_iPBSWD[lane] + czbdswg * mjswgd * t_iPBSWGD[lane];

            if (h->B4P(dNode) != h->B4P(dNodePrime)) blk->m_has_dint |= bit;
            if (h->B4P(sNode) != h->B4P(sNodePrime)) blk->m_has_sint |= bit;

            blk->vcrit[lane]      = m->B4P(vcrit);
            blk->type_s[lane]     = (double)m->B4P(type);
            blk->dNodePrime[lane] = h->B4P(dNodePrime);
            blk->gNodePrime[lane] = h->B4P(gNodePrime);
            blk->sNodePrime[lane] = h->B4P(sNodePrime);
            blk->bNodePrime[lane] = h->B4P(bNodePrime);
            blk->stateBase[lane]  = h->B4P(states);
            blk->instIdx[lane]    = src_idx;

            if (ax != NULL && nz > 0) {
                const double *ptrs[22] = {
                    h->B4P(GPgpPtr), h->B4P(GPdpPtr), h->B4P(GPspPtr), h->B4P(GPbpPtr),
                    h->B4P(DPdpPtr), h->B4P(DPgpPtr), h->B4P(DPspPtr), h->B4P(DPbpPtr),
                    h->B4P(SPdpPtr), h->B4P(SPgpPtr), h->B4P(SPspPtr), h->B4P(SPbpPtr),
                    h->B4P(BPdpPtr), h->B4P(BPgpPtr), h->B4P(BPspPtr), h->B4P(BPbpPtr),
                    h->B4P(DPdPtr),  h->B4P(DdpPtr),  h->B4P(DdPtr),
                    h->B4P(SPsPtr),  h->B4P(SspPtr),  h->B4P(SsPtr)
                };
                for (int k = 0; k < 22; k++) {
                    uintptr_t d = (uintptr_t)ptrs[k] - ax_addr;
                    blk->kluOff[lane][k] = (d < ax_bytes) ? (int)(d / sizeof(double)) : inz;
                }
            }
        }

        blk->has_ud = (uint8_t)has_ud;
        blk->xpart_mode = all_xp_lt05 ? 0 : (all_xp_gt05 ? 1 : (all_xp_eq05 ? 2 : 3));

        blk->type = v8_load(t_type); blk->m = v8_load(t_m); blk->nf = v8_load(t_nf);
        blk->vth0_signed = v8_load(t_vth0s); blk->vfb_plus_phi = v8_load(t_vfbphi);
        blk->vfbzb = v8_load(t_vfbzb); blk->vbsc = v8_load(t_vbsc);
        blk->phi = v8_load(t_phi); blk->phi_095 = v8_load(t_phi095);
        blk->sqrtPhi = v8_load(t_sqrtPhi); blk->V0 = v8_load(t_V0);
        blk->vtm = v8_load(t_vtm); blk->coxe = v8_load(t_coxe); blk->invCoxe = v8_load(t_invCoxe);
        blk->coxp = v8_load(t_coxp); blk->toxe = v8_load(t_toxe); blk->invToxe = v8_load(t_invToxe);
        blk->toxp = v8_load(t_toxp); blk->tox_ctm = v8_load(t_toxctm); blk->inv_tox_ctm = v8_load(t_invtoxctm);
        blk->inv_2e8_toxp = v8_load(t_inv2e8); blk->link_ctm = v8_load(t_link);
        blk->leff = v8_load(t_leff); blk->invLeff = v8_load(t_invLeff); blk->sqrtLeff = v8_load(t_sqrtLeff);
        blk->weff = v8_load(t_weff); blk->weffCV = v8_load(t_weffCV); blk->leffCV = v8_load(t_leffCV);
        blk->CoxWL = v8_load(t_CoxWL); blk->CoxWL_ov_coxe = v8_load(t_CoxWL_oc);
        blk->drainCond = v8_load(t_gdpr); blk->sourceCond = v8_load(t_gspr);
        blk->mgd = v8_load(t_mgd); blk->mgs = v8_load(t_mgs);
        blk->ados1p9 = v8_load(t_ados); blk->bdos0p7 = v8_load(t_bdos);

        blk->invNvtms = v8_load(t_iNvtms); blk->invNvtmd = v8_load(t_iNvtmd);
        blk->SourceSatCurrent = v8_load(t_Ssat); blk->DrainSatCurrent = v8_load(t_Dsat);
        blk->vjsmFwd = v8_load(t_vjsm); blk->vjdmFwd = v8_load(t_vjdm);
        blk->IVjsmFwd = v8_load(t_IVjsm); blk->IVjdmFwd = v8_load(t_IVjdm);
        blk->invNvtmrs_s = v8_load(t_iNvrs); blk->invNvtmrs_d = v8_load(t_iNvrd);
        blk->invNvtmrssw_s = v8_load(t_iNvrssw); blk->invNvtmrssw_d = v8_load(t_iNvrdsw);
        blk->invNvtmrsswg_s = v8_load(t_iNvrsswg); blk->invNvtmrsswg_d = v8_load(t_iNvrdswg);
        blk->vtss = v8_load(t_vtss); blk->vtsd = v8_load(t_vtsd);
        blk->vtssws = v8_load(t_vtssws); blk->vtsswd = v8_load(t_vtsswd);
        blk->vtsswgs = v8_load(t_vtsswgs); blk->vtsswgd = v8_load(t_vtsswgd);
        blk->SjctRevSat = v8_load(t_SjRev); blk->DjctRevSat = v8_load(t_DjRev);
        blk->SswRevSat = v8_load(t_SswRev); blk->DswRevSat = v8_load(t_DswRev);
        blk->SswgRevSat = v8_load(t_SswgRev); blk->DswgRevSat = v8_load(t_DswgRev);
        blk->vcrit_v = v8_load(blk->vcrit);

        blk->xdep0OverSqrtPhi = v8_load(t_xdep0s); blk->factor1 = v8_load(t_fac1);
        blk->dvt0 = v8_load(t_dvt0); blk->dvt1_leff = v8_load(t_dvt1l); blk->dvt2 = v8_load(t_dvt2);
        blk->dvt0w = v8_load(t_dvt0w); blk->dvt1w_weff_leff = v8_load(t_dvt1wwl); blk->dvt2w = v8_load(t_dvt2w);
        blk->k1 = v8_load(t_k1); blk->k1ox = v8_load(t_k1ox); blk->k2ox = v8_load(t_k2ox);
        blk->k3 = v8_load(t_k3); blk->k3b = v8_load(t_k3b);
        blk->lpebFactor = v8_load(t_lpeb); blk->vthNarrowW = v8_load(t_vthNW);
        blk->vth_const_T1 = v8_load(t_vthC); blk->kt2_TempRatio = v8_load(t_kt2T);
        blk->eta0 = v8_load(t_eta0); blk->etab = v8_load(t_etab); blk->theta0vb0 = v8_load(t_th0vb0);
        blk->nfactor = v8_load(t_nfac); blk->cit = v8_load(t_cit);
        blk->cdsc = v8_load(t_cdsc); blk->cdscb = v8_load(t_cdscb); blk->cdscd = v8_load(t_cdscd);
        blk->dvtp0 = v8_load(t_dvtp0); blk->dvtp1 = v8_load(t_dvtp1);

        blk->polyT1 = v8_load(t_polyT1); blk->mstar = v8_load(t_mstar);
        blk->voffcbn = v8_load(t_voffcbn); blk->coxe_ov_cdep0 = v8_load(t_coxeCdep);

        blk->dwg = v8_load(t_dwg); blk->dwb = v8_load(t_dwb);
        blk->prwg = v8_load(t_prwg); blk->prwb = v8_load(t_prwb);
        blk->rds0_half = v8_load(t_rds0h); blk->rdswmin = v8_load(t_rdswmin);
        blk->xj = v8_load(t_xj); blk->a0 = v8_load(t_a0);
        blk->b0_ov_wb1 = v8_load(t_b0wb1); blk->ags_a0 = v8_load(t_agsa0);
        blk->keta = v8_load(t_keta); blk->abulkCVfactor = v8_load(t_abCVf);

        blk->ua = v8_load(t_ua); blk->ub = v8_load(t_ub); blk->uc = v8_load(t_uc); blk->ud = v8_load(t_ud);
        blk->u0temp = v8_load(t_u0t); blk->vsattemp = v8_load(t_vsatt);
        blk->a1 = v8_load(t_a1); blk->a2 = v8_load(t_a2); blk->delta = v8_load(t_delta);
        blk->lambda = v8_load(t_lam); blk->litl = v8_load(t_litl);
        blk->fprout = v8_load(t_fprout); blk->pvag = v8_load(t_pvag);
        blk->pclm = v8_load(t_pclm); blk->inv_pclm_litl = v8_load(t_ipclml);
        blk->thetaRout = v8_load(t_thRout); blk->inv_thetaRout = v8_load(t_ithRout);
        blk->pdiblb = v8_load(t_pdiblb);
        blk->pdits = v8_load(t_pdits); blk->inv_pdits = v8_load(t_ipdits);
        blk->pditsd = v8_load(t_pditsd); blk->pditsl_term = v8_load(t_pditsl);
        blk->pscbe1_litl = v8_load(t_pscbe1l); blk->inv_pscbe2 = v8_load(t_ipscbe2);
        blk->alpha_ov_leff = v8_load(t_alpl); blk->beta0 = v8_load(t_beta0);
        blk->inv_vtl_tfactor = v8_load(t_ivtl); blk->vtfbphi2 = v8_load(t_vtfb2);

        blk->inv_3toxe = v8_load(t_i3toxe);
        blk->agidl_weffCJ = v8_load(t_agidlw); blk->bgidl = v8_load(t_bgidl);
        blk->cgidl = v8_load(t_cgidl); blk->egidl = v8_load(t_egidl);
        blk->agisl_weffCJ = v8_load(t_agislw); blk->bgisl = v8_load(t_bgisl);
        blk->cgisl = v8_load(t_cgisl); blk->egisl = v8_load(t_egisl);

        blk->noff = v8_load(t_noff); blk->voffcv = v8_load(t_voffcv);
        blk->acde = v8_load(t_acde); blk->ldeb = v8_load(t_ldeb); blk->moin = v8_load(t_moin);
        blk->cgdo = v8_load(t_cgdo); blk->cgso = v8_load(t_cgso); blk->cgbo = v8_load(t_cgbo);
        blk->weffCV_cgdl = v8_load(t_wcgdl); blk->ckappad = v8_load(t_ckd); blk->inv_ckappad = v8_load(t_ickd);
        blk->weffCV_cgsl = v8_load(t_wcgsl); blk->ckappas = v8_load(t_cks); blk->inv_ckappas = v8_load(t_icks);

        blk->czbs = v8_load(t_czbs); blk->czbssw = v8_load(t_czbssw); blk->czbsswg = v8_load(t_czbsswg);
        blk->czbd = v8_load(t_czbd); blk->czbdsw = v8_load(t_czbdsw); blk->czbdswg = v8_load(t_czbdswg);
        blk->PhiBS = v8_load(t_PBS); blk->invPhiBS = v8_load(t_iPBS);
        blk->PhiBSWS = v8_load(t_PBSWS); blk->invPhiBSWS = v8_load(t_iPBSWS);
        blk->PhiBSWGS = v8_load(t_PBSWGS); blk->invPhiBSWGS = v8_load(t_iPBSWGS);
        blk->PhiBD = v8_load(t_PBD); blk->invPhiBD = v8_load(t_iPBD);
        blk->PhiBSWD = v8_load(t_PBSWD); blk->invPhiBSWD = v8_load(t_iPBSWD);
        blk->PhiBSWGD = v8_load(t_PBSWGD); blk->invPhiBSWGD = v8_load(t_iPBSWGD);
        blk->MJS = v8_load(t_MJS); blk->inv1mMJS = v8_load(t_i1MJS);
        blk->MJSWS = v8_load(t_MJSWS); blk->inv1mMJSWS = v8_load(t_i1MJSWS);
        blk->MJSWGS = v8_load(t_MJSWGS); blk->inv1mMJSWGS = v8_load(t_i1MJSWGS);
        blk->MJD = v8_load(t_MJD); blk->inv1mMJD = v8_load(t_i1MJD);
        blk->MJSWD = v8_load(t_MJSWD); blk->inv1mMJSWD = v8_load(t_i1MJSWD);
        blk->MJSWGD = v8_load(t_MJSWGD); blk->inv1mMJSWGD = v8_load(t_i1MJSWGD);
        blk->jctS_sum = v8_load(t_jSsum); blk->jctD_sum = v8_load(t_jDsum);
        blk->jctS_lin = v8_load(t_jSlin); blk->jctD_lin = v8_load(t_jDlin);

        /* Mirror hot read-only topology/stamp metadata into dyn[bi] so L1 hits,
         * convTest, terrBlock8, and KLU stamping never touch the 10 KiB cold block! */
        dyn[bi].type_v     = blk->type;
        dyn[bi].n_active   = blk->n_active;
        dyn[bi].m_has_dint = blk->m_has_dint;
        dyn[bi].m_has_sint = blk->m_has_sint;
        for (int lane = 0; lane < 8; lane++) {
            dyn[bi].dNodePrime[lane] = blk->dNodePrime[lane];
            dyn[bi].gNodePrime[lane] = blk->gNodePrime[lane];
            dyn[bi].sNodePrime[lane] = blk->sNodePrime[lane];
            dyn[bi].bNodePrime[lane] = blk->bNodePrime[lane];
            dyn[bi].stateBase[lane]  = blk->stateBase[lane];
            for (int k = 0; k < 16; k++)
                dyn[bi].kluOff16[lane][k] = blk->kluOff[lane][k];
        }

        /* Stage 10E: populate AoS per-instance parameter table for active-lane compaction */
        if (insts) {
            const double (*bvecs)[8] = (const double (*)[8])blk;
            const uint8_t *bmasks = (const uint8_t *)&blk->m_ngate_valid;
            for (int lane = 0; lane < 8; lane++) {
                BSIM4Avx512Inst *ins = &insts[bi * 8 + lane];
                for (size_t f = 0; f < B4_NUM_PARAM_V8; f++)
                    ins->p[f] = bvecs[f][lane];
                for (size_t f = B4_NUM_PARAM_V8; f < B4_PARAM_PADDED; f++)
                    ins->p[f] = 0.0;
                uint64_t mb = 0;
                for (int mk = 0; mk < B4_NUM_MASKS; mk++) {
                    if (bmasks[mk] & (1u << lane))
                        mb |= (1ULL << mk);
                }
                if (blk->has_ud)
                    mb |= (1ULL << B4_NUM_MASKS);
                ins->mask_bits = mb;
            }
        }
    }

    FREE(eligIdx);
    head->B4P(AvxBlocks) = blocks;
    head->B4P(AvxDyn) = dyn;
    head->B4P(AvxDynValid) = 0;
    head->B4P(AvxBlockCount) = nBlocks;
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
    const BSIM4Avx512Block *blocks = (const BSIM4Avx512Block *)head->B4P(AvxBlocks);
    BSIM4Avx512Dyn *dyn = (BSIM4Avx512Dyn *)head->B4P(AvxDyn);
    B4P(instance) **InstArray = head->B4P(InstanceArray);
    int nBlocks = head->B4P(AvxBlockCount);
    for (int bi = 0; bi < nBlocks; bi++) {
        double von_a[8];
        int nact = blocks[bi].n_active;
        for (int lane = 0; lane < nact; lane++)
            von_a[lane] = InstArray[blocks[bi].instIdx[lane]]->B4P(von);
        for (int lane = nact; lane < 8; lane++)
            von_a[lane] = von_a[0];
        dyn[bi].von = v8_load(von_a);
        dyn[bi].phys_valid = 0;
        dyn[bi].stamp_valid = 0;
        dyn[bi].last_active_step = -100;
        dyn[bi].latent_frozen = 0;
        dyn[bi].latent_streak = 0;
    }
    head->B4P(AvxDynValid) = 1;
}

static inline double
B4P(avxBypassTol)(const CKTcircuit *ckt)
{
    static int init = 0;
    static int mode = 0; /* 0=auto(min(CKTvoltTol,1e-6)), 1=disabled, 2=override */
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

static inline __mmask8
B4P(laneCacheHitMask)(
    const BSIM4Avx512Dyn * restrict d,
    v8d vds, v8d vgs, v8d vbs,
    double vtol, double reltol, double abstol,
    __mmask8 active_mask,
    __mmask8 *exact_zero_mask_out)
{
    v8d delvds = vds - d->last_vds;
    v8d delvgs = vgs - d->last_vgs;
    v8d delvbs = vbs - d->last_vbs;
    v8d dv = v8_max(v8_abs(delvds), v8_max(v8_abs(delvgs), v8_abs(delvbs)));
    __mmask8 ok = v8_le(dv, v8_set1(vtol)) & active_mask;
    if (!ok) {
        *exact_zero_mask_out = 0;
        return 0;
    }
    __mmask8 m_flip = (v8_ge(vds, v8_set1(0.0)) ^ v8_ge(d->last_vds, v8_set1(0.0))) & ok;
    ok &= ~m_flip;
    if (!ok) {
        *exact_zero_mask_out = 0;
        return 0;
    }
    const v8d v_rel = v8_set1(0.1 * reltol);
    const v8d v_abs = v8_set1(0.1 * abstol);
    v8d cd_diff = d->cd_g * delvgs + d->cd_d * delvds + d->cd_b * delvbs;
    v8d cdhat   = d->Idtot + cd_diff;
    v8d tol0    = v_rel * v8_max(v8_abs(cdhat), v8_abs(d->Idtot)) + v_abs;
    v8d cb_diff = d->cb_g * delvgs + d->cb_d * delvds + d->cb_b * delvbs;
    v8d cbhat   = d->Ibtot + cb_diff;
    v8d tol6    = v_rel * v8_max(v8_abs(cbhat), v8_abs(d->Ibtot)) + v_abs;
    __mmask8 fail = (v8_ge(v8_abs(cd_diff), tol0) | v8_ge(v8_abs(cb_diff), tol6)) & ok;
    __mmask8 hit = ok & ~fail;
    *exact_zero_mask_out = v8_le(dv, v8_set1(1.0e-13)) & hit;
    return hit;
}

static inline int
B4P(blockCacheMatch)(
    const BSIM4Avx512Dyn * restrict d,
    v8d vds, v8d vgs, v8d vbs,
    double vtol, double reltol, double abstol,
    __mmask8 active_mask,
    int *exact_zero_out)
{
    v8d delvds = vds - d->last_vds;
    v8d delvgs = vgs - d->last_vgs;
    v8d delvbs = vbs - d->last_vbs;
    v8d dv = v8_max(v8_abs(delvds), v8_max(v8_abs(delvgs), v8_abs(delvbs)));
    if ((v8_le(dv, v8_set1(vtol)) & active_mask) != active_mask)
        return 0;
    /* Require identical forward/reverse sign across vds == 0 */
    __mmask8 m_flip = (v8_ge(vds, v8_set1(0.0)) ^ v8_ge(d->last_vds, v8_set1(0.0))) & active_mask;
    if (m_flip)
        return 0;
    /* Require 10x tighter than convTest current tolerance so cache can NEVER stall Newton */
    const v8d v_rel = v8_set1(0.1 * reltol);
    const v8d v_abs = v8_set1(0.1 * abstol);
    v8d cd_diff = d->cd_g * delvgs + d->cd_d * delvds + d->cd_b * delvbs;
    v8d cdhat   = d->Idtot + cd_diff;
    v8d tol0    = v_rel * v8_max(v8_abs(cdhat), v8_abs(d->Idtot)) + v_abs;
    v8d cb_diff = d->cb_g * delvgs + d->cb_d * delvds + d->cb_b * delvbs;
    v8d cbhat   = d->Ibtot + cb_diff;
    v8d tol6    = v_rel * v8_max(v8_abs(cbhat), v8_abs(d->Ibtot)) + v_abs;
    __mmask8 fail = (v8_ge(v8_abs(cd_diff), tol0) | v8_ge(v8_abs(cb_diff), tol6)) & active_mask;
    if (fail)
        return 0;
    *exact_zero_out = ((v8_le(dv, v8_set1(1.0e-13)) & active_mask) == active_mask);
    return 1;
}

static inline void
B4P(stampAvxDyn)(
    const BSIM4Avx512Block * restrict blk,
    const BSIM4Avx512Dyn * restrict dyn,
    B4P(instance) **InstArray,
    double * restrict my_ax,
    double * restrict my_rhs,
    int stamp_mode)
{
    int nact = dyn->n_active;
    double mgd_arr[8], mgs_arr[8];
    if (dyn->m_has_dint) v8_store(mgd_arr, blk->mgd);
    if (dyn->m_has_sint) v8_store(mgs_arr, blk->mgs);

    if (stamp_mode == 0) {
        for (int lane = 0; lane < nact; lane++) {
            const int *off = dyn->kluOff16[lane];
            int dn = dyn->dNodePrime[lane];
            int gn = dyn->gNodePrime[lane];
            int sn = dyn->sNodePrime[lane];
            int bn = dyn->bNodePrime[lane];
            if (dn) my_rhs[dn] += dyn->st_rhsd[lane];
            if (gn) my_rhs[gn] -= dyn->st_rhsg[lane];
            if (bn) my_rhs[bn] += dyn->st_rhsb[lane];
            if (sn) my_rhs[sn] += dyn->st_rhss[lane];

            /* Skip grounded S/B matrix stamps to eliminate serialized my_ax[inz]
             * store-to-load forwarding stalls with zero extra struct footprint! */
            if (sn != 0 && bn != 0) {
                my_ax[off[0]]  += dyn->st_v34[lane];
                my_ax[off[1]]  += dyn->st_v35[lane];
                my_ax[off[2]]  += dyn->st_v36[lane];
                my_ax[off[3]]  += dyn->st_v37[lane];
                my_ax[off[4]]  += dyn->st_v44[lane];
                my_ax[off[5]]  += dyn->st_v46[lane];
                my_ax[off[6]]  -= dyn->st_v47[lane];
                my_ax[off[7]]  -= dyn->st_v48[lane];
                my_ax[off[8]]  -= dyn->st_v51[lane];
                my_ax[off[9]]  += dyn->st_v52[lane];
                my_ax[off[10]] += dyn->st_v53[lane];
                my_ax[off[11]] -= dyn->st_v55[lane];
                my_ax[off[12]] += dyn->st_v58[lane];
                my_ax[off[13]] += dyn->st_v59[lane];
                my_ax[off[14]] += dyn->st_v60[lane];
                my_ax[off[15]] += dyn->st_v61[lane];
            } else if (sn != 0) {
                /* Bulk grounded (b==0): only 9 (g,d,s)x(g,d,s) entries active */
                my_ax[off[0]]  += dyn->st_v34[lane];
                my_ax[off[1]]  += dyn->st_v35[lane];
                my_ax[off[2]]  += dyn->st_v36[lane];
                my_ax[off[4]]  += dyn->st_v44[lane];
                my_ax[off[5]]  += dyn->st_v46[lane];
                my_ax[off[6]]  -= dyn->st_v47[lane];
                my_ax[off[8]]  -= dyn->st_v51[lane];
                my_ax[off[9]]  += dyn->st_v52[lane];
                my_ax[off[10]] += dyn->st_v53[lane];
            } else if (bn != 0) {
                /* Source grounded (s==0): only 9 (g,d,b)x(g,d,b) entries active */
                my_ax[off[0]]  += dyn->st_v34[lane];
                my_ax[off[1]]  += dyn->st_v35[lane];
                my_ax[off[3]]  += dyn->st_v37[lane];
                my_ax[off[4]]  += dyn->st_v44[lane];
                my_ax[off[5]]  += dyn->st_v46[lane];
                my_ax[off[7]]  -= dyn->st_v48[lane];
                my_ax[off[12]] += dyn->st_v58[lane];
                my_ax[off[13]] += dyn->st_v59[lane];
                my_ax[off[15]] += dyn->st_v61[lane];
            } else {
                /* Both source & bulk grounded (common bottom NMOS): only 4 entries! */
                my_ax[off[0]]  += dyn->st_v34[lane];
                my_ax[off[1]]  += dyn->st_v35[lane];
                my_ax[off[4]]  += dyn->st_v44[lane];
                my_ax[off[5]]  += dyn->st_v46[lane];
            }
        }
        if (dyn->m_has_dint) {
            for (int lane = 0; lane < nact; lane++) {
                if (dyn->m_has_dint & (1u << lane)) {
                    const int *off = blk->kluOff[lane];
                    double mgd = mgd_arr[lane];
                    my_ax[off[16]] -= mgd;
                    my_ax[off[17]] -= mgd;
                    my_ax[off[18]] += mgd;
                }
            }
        }
        if (dyn->m_has_sint) {
            for (int lane = 0; lane < nact; lane++) {
                if (dyn->m_has_sint & (1u << lane)) {
                    const int *off = blk->kluOff[lane];
                    double mgs = mgs_arr[lane];
                    my_ax[off[19]] -= mgs;
                    my_ax[off[20]] -= mgs;
                    my_ax[off[21]] += mgs;
                }
            }
        }
    } else if (stamp_mode == 1) {
        for (int lane = 0; lane < nact; lane++) {
            B4P(instance) *here = InstArray[blk->instIdx[lane]];
            __mmask8 bit = (__mmask8)(1u << lane);
            int dn = blk->dNodePrime[lane];
            int gn = blk->gNodePrime[lane];
            int sn = blk->sNodePrime[lane];
            int bn = blk->bNodePrime[lane];
            if (dn) my_rhs[dn] += dyn->st_rhsd[lane];
            if (gn) my_rhs[gn] -= dyn->st_rhsg[lane];
            if (bn) my_rhs[bn] += dyn->st_rhsb[lane];
            if (sn) my_rhs[sn] += dyn->st_rhss[lane];

            *(here->B4P(GPgpPtr)) += dyn->st_v34[lane];
            *(here->B4P(GPdpPtr)) += dyn->st_v35[lane];
            *(here->B4P(DPdpPtr)) += dyn->st_v44[lane];
            *(here->B4P(DPgpPtr)) += dyn->st_v46[lane];
            if (sn != 0 && bn != 0) {
                *(here->B4P(GPspPtr)) += dyn->st_v36[lane];
                *(here->B4P(GPbpPtr)) += dyn->st_v37[lane];
                *(here->B4P(DPspPtr)) -= dyn->st_v47[lane];
                *(here->B4P(DPbpPtr)) -= dyn->st_v48[lane];
                *(here->B4P(SPdpPtr)) -= dyn->st_v51[lane];
                *(here->B4P(SPgpPtr)) += dyn->st_v52[lane];
                *(here->B4P(SPspPtr)) += dyn->st_v53[lane];
                *(here->B4P(SPbpPtr)) -= dyn->st_v55[lane];
                *(here->B4P(BPdpPtr)) += dyn->st_v58[lane];
                *(here->B4P(BPgpPtr)) += dyn->st_v59[lane];
                *(here->B4P(BPspPtr)) += dyn->st_v60[lane];
                *(here->B4P(BPbpPtr)) += dyn->st_v61[lane];
            } else if (sn != 0) {
                *(here->B4P(GPspPtr)) += dyn->st_v36[lane];
                *(here->B4P(DPspPtr)) -= dyn->st_v47[lane];
                *(here->B4P(SPdpPtr)) -= dyn->st_v51[lane];
                *(here->B4P(SPgpPtr)) += dyn->st_v52[lane];
                *(here->B4P(SPspPtr)) += dyn->st_v53[lane];
            } else if (bn != 0) {
                *(here->B4P(GPbpPtr)) += dyn->st_v37[lane];
                *(here->B4P(DPbpPtr)) -= dyn->st_v48[lane];
                *(here->B4P(BPdpPtr)) += dyn->st_v58[lane];
                *(here->B4P(BPgpPtr)) += dyn->st_v59[lane];
                *(here->B4P(BPbpPtr)) += dyn->st_v61[lane];
            }
            if (blk->m_has_dint & bit) {
                double mgd = mgd_arr[lane];
                *(here->B4P(DPdPtr)) -= mgd;
                *(here->B4P(DdpPtr)) -= mgd;
                *(here->B4P(DdPtr))  += mgd;
            }
            if (blk->m_has_sint & bit) {
                double mgs = mgs_arr[lane];
                *(here->B4P(SPsPtr)) -= mgs;
                *(here->B4P(SspPtr)) -= mgs;
                *(here->B4P(SsPtr))  += mgs;
            }
        }
    } else {
        for (int lane = 0; lane < nact; lane++) {
            B4P(instance) *here = InstArray[blk->instIdx[lane]];
            __mmask8 bit = (__mmask8)(1u << lane);
            here->B4P(rhsdPrime) = dyn->st_rhsd[lane];
            here->B4P(rhsgPrime) = dyn->st_rhsg[lane];
            here->B4P(rhsbPrime) = dyn->st_rhsb[lane];
            here->B4P(rhssPrime) = dyn->st_rhss[lane];
            here->B4P(_34) = dyn->st_v34[lane];
            here->B4P(_35) = dyn->st_v35[lane];
            here->B4P(_36) = dyn->st_v36[lane];
            here->B4P(_37) = dyn->st_v37[lane];
            here->B4P(_44) = dyn->st_v44[lane];
            here->B4P(_46) = dyn->st_v46[lane];
            here->B4P(_47) = dyn->st_v47[lane];
            here->B4P(_48) = dyn->st_v48[lane];
            if (blk->m_has_dint & bit) {
                double mgd = mgd_arr[lane];
                here->B4P(_45) = mgd;
                here->B4P(_49) = mgd;
                here->B4P(_50) = mgd;
            }
            here->B4P(_51) = dyn->st_v51[lane];
            here->B4P(_52) = dyn->st_v52[lane];
            here->B4P(_53) = dyn->st_v53[lane];
            here->B4P(_55) = dyn->st_v55[lane];
            if (blk->m_has_sint & bit) {
                double mgs = mgs_arr[lane];
                here->B4P(_54) = mgs;
                here->B4P(_56) = mgs;
                here->B4P(_57) = mgs;
            }
            here->B4P(_58) = dyn->st_v58[lane];
            here->B4P(_59) = dyn->st_v59[lane];
            here->B4P(_60) = dyn->st_v60[lane];
            here->B4P(_61) = dyn->st_v61[lane];
        }
    }
}

/*
 * Evaluate one 8-wide block with 8-wide AVX-512 voltage limiting, two-level
 * stationary-voltage caching (Level 1 same-step stamp replay & Level 2
 * cross-step physics reuse), and direct stamping:
 *   stamp_mode == 0: KLU / TLS direct stamp into my_ax & my_rhs (zero AoS chase when !need_aos!)
 *   stamp_mode == 1: Sparse 1t direct stamp into *(here->*Ptr) & my_rhs
 *   stamp_mode == 2: Sparse OMP buffered write into here->B4P(_34.._61) & here->B4P(rhs*)
 */
static inline void
B4P(evalAvxBlock)(
    const BSIM4Avx512Block * restrict blk,
    BSIM4Avx512Dyn * restrict dyn,
    B4P(instance) **InstArray,
    CKTcircuit *ckt,
    double * restrict my_ax,
    double * restrict my_rhs,
    int stamp_mode,
    int dc_only,
    int force_aos)
{
    const double * restrict rhsOld = ckt->CKTrhsOld;
    double * restrict s0 = ckt->CKTstate0;
    const double * restrict s1 = dc_only ? NULL : ckt->CKTstate1;
    const double * restrict s2 = dc_only ? NULL : ckt->CKTstate2;
    const double * restrict s3 = dc_only ? NULL : ckt->CKTstate3;
    int is_pred = (!dc_only && (ckt->CKTmode & MODEINITPRED) != 0);
    int is_gear3 = (!dc_only && ckt->CKTintegrateMethod == GEAR && ckt->CKTorder == 3);
    int is_gear2 = (!dc_only && ckt->CKTintegrateMethod == GEAR && ckt->CKTorder == 2);
    int is_trap2 = (!dc_only && ckt->CKTintegrateMethod == TRAPEZOIDAL && ckt->CKTorder == 2);
    int int_mode = is_gear3 ? 3 : (is_gear2 ? 2 : (is_trap2 ? 1 : 0));
    int need_aos = B4P(AvxNeedAosSync) || force_aos;
    double ag0 = dc_only ? 0.0 : ckt->CKTag[0];
    double ag1 = dc_only ? 0.0 : ckt->CKTag[1];
    double ag2 = dc_only ? 0.0 : ckt->CKTag[2];
    double ag3 = dc_only ? 0.0 : ckt->CKTag[3];
    double vtol = dc_only ? -1.0 : B4P(avxBypassTol)(ckt);
    int nact = dyn->n_active;
    __mmask8 active_mask = (__mmask8)((1u << nact) - 1u);

    double a_vds[8], a_vgs[8], a_vbs[8];
    double a_s0_vds[8], a_s0_vgs[8], a_s0_vbs[8], a_s0_vbd[8];

    /* Level-1 fast path: exact same timestep/integration & stationary voltages ->
     * skip CKTstate0 read/write, voltage limiting, physics, and integration! */
    if (!dc_only && !is_pred && !need_aos && vtol >= 0.0 && dyn->stamp_valid &&
        dyn->last_time == ckt->CKTtime &&
        dyn->last_ag0 == ag0 && dyn->last_ag1 == ag1 && dyn->last_ag2 == ag2 &&
        (!is_gear3 || dyn->last_ag3 == ag3) &&
        dyn->last_gmin == ckt->CKTgmin && dyn->last_int_mode == int_mode &&
        dyn->last_s0 == s0 && dyn->last_s1 == s1 && dyn->last_s2 == s2 &&
        (!is_gear3 || dyn->last_s3 == s3))
    {
        for (int lane = 0; lane < nact; lane++) {
            double vs_node = rhsOld[dyn->sNodePrime[lane]];
            a_vds[lane] = rhsOld[dyn->dNodePrime[lane]] - vs_node;
            a_vgs[lane] = rhsOld[dyn->gNodePrime[lane]] - vs_node;
            a_vbs[lane] = rhsOld[dyn->bNodePrime[lane]] - vs_node;
        }
        for (int lane = nact; lane < 8; lane++) {
            a_vds[lane] = a_vds[0];
            a_vgs[lane] = a_vgs[0];
            a_vbs[lane] = a_vbs[0];
        }
        v8d vds_r = v8_load(a_vds) * dyn->type_v;
        v8d vgs_r = v8_load(a_vgs) * dyn->type_v;
        v8d vbs_r = v8_load(a_vbs) * dyn->type_v;
        int exact_zero = 0;
        if (B4P(blockCacheMatch)(dyn, vds_r, vgs_r, vbs_r, vtol,
                                 ckt->CKTreltol, ckt->CKTabstol, active_mask, &exact_zero)) {
            if (!exact_zero) {
                bsim4_avx512_l1_sync_state(dyn, vds_r, vgs_r, vbs_r, ag0, s0);
                dyn->last_active_step = ckt->CKTstat->STATaccepted;
            }
            B4P(stampAvxDyn)(blk, dyn, InstArray, my_ax, my_rhs, stamp_mode);
            return;
        }
    }

    double xfact = is_pred ? (ckt->CKTdelta / ckt->CKTdeltaOld[1]) : 0.0;
    double one_p_xf = 1.0 + xfact;
    int is_initjct = (dc_only && (ckt->CKTmode & MODEINITJCT) != 0);

    v8d vds, vgs, vbs;
    __mmask8 chk_mask = 0;

    if (is_initjct) {
        int any_ic = 0;
        for (int lane = 0; lane < nact; lane++) {
            B4P(instance) *h = InstArray[blk->instIdx[lane]];
            if (h->B4P(icVDS) != 0.0 || h->B4P(icVGS) != 0.0 || h->B4P(icVBS) != 0.0) {
                any_ic = 1;
                break;
            }
        }
        if (!any_ic) {
            vds = v8_set1(0.1);
            vgs = blk->vth0_signed + v8_set1(0.1);
            vbs = v8_set1(0.0);
        } else {
            double t_vth0[8], t_type[8];
            v8_store(t_vth0, blk->vth0_signed);
            v8_store(t_type, blk->type);
            for (int lane = 0; lane < nact; lane++) {
                B4P(instance) *h = InstArray[blk->instIdx[lane]];
                double d = t_type[lane] * h->B4P(icVDS);
                double g = t_type[lane] * h->B4P(icVGS);
                double b = t_type[lane] * h->B4P(icVBS);
                if (d == 0.0 && g == 0.0 && b == 0.0) {
                    d = 0.1; g = t_vth0[lane] + 0.1; b = 0.0;
                }
                a_vds[lane] = d; a_vgs[lane] = g; a_vbs[lane] = b;
            }
            for (int lane = nact; lane < 8; lane++) {
                a_vds[lane] = a_vds[0]; a_vgs[lane] = a_vgs[0]; a_vbs[lane] = a_vbs[0];
            }
            vds = v8_load(a_vds);
            vgs = v8_load(a_vgs);
            vbs = v8_load(a_vbs);
        }
    } else {
#ifndef PREDICTOR
    if (is_pred) {
#if NGSPICE_OPT_GENERAL
        if (ckt->CKTtaPredActive) {
            for (int lane = 0; lane < nact; lane++) {
                int st = blk->stateBase[lane];
                double s1_vds = s1[st + 3], s1_vgs = s1[st + 2], s1_vbs = s1[st + 1];
                double s1_vbd = s1_vbs - s1_vds;
                s0[st + 3] = a_s0_vds[lane] = s1_vds;
                s0[st + 2] = a_s0_vgs[lane] = s1_vgs;
                s0[st + 1] = a_s0_vbs[lane] = s1_vbs;
                s0[st + 0] = a_s0_vbd[lane] = s1_vbd;
                double vs_node = rhsOld[blk->sNodePrime[lane]];
                a_vds[lane] = rhsOld[blk->dNodePrime[lane]] - vs_node;
                a_vgs[lane] = rhsOld[blk->gNodePrime[lane]] - vs_node;
                a_vbs[lane] = rhsOld[blk->bNodePrime[lane]] - vs_node;
            }
        } else
#endif
        {
            for (int lane = 0; lane < nact; lane++) {
                int st = blk->stateBase[lane];
                double s1_vds = s1[st + 3], s1_vgs = s1[st + 2], s1_vbs = s1[st + 1];
                double s1_vbd = s1_vbs - s1_vds;
                s0[st + 3] = a_s0_vds[lane] = s1_vds;
                s0[st + 2] = a_s0_vgs[lane] = s1_vgs;
                s0[st + 1] = a_s0_vbs[lane] = s1_vbs;
                s0[st + 0] = a_s0_vbd[lane] = s1_vbd;
                a_vds[lane] = one_p_xf * s1_vds - xfact * s2[st + 3];
                a_vgs[lane] = one_p_xf * s1_vgs - xfact * s2[st + 2];
                a_vbs[lane] = one_p_xf * s1_vbs - xfact * s2[st + 1];
            }
        }
    } else
#endif
    {
        for (int lane = 0; lane < nact; lane++) {
            int st = blk->stateBase[lane];
            double vs_node = rhsOld[blk->sNodePrime[lane]];
            a_vds[lane] = rhsOld[blk->dNodePrime[lane]] - vs_node;
            a_vgs[lane] = rhsOld[blk->gNodePrime[lane]] - vs_node;
            a_vbs[lane] = rhsOld[blk->bNodePrime[lane]] - vs_node;
            a_s0_vbd[lane] = s0[st + 0];
            a_s0_vbs[lane] = s0[st + 1];
            a_s0_vgs[lane] = s0[st + 2];
            a_s0_vds[lane] = s0[st + 3];
        }
    }
    for (int lane = nact; lane < 8; lane++) {
        a_vds[lane] = a_vds[0];
        a_vgs[lane] = a_vgs[0];
        a_vbs[lane] = a_vbs[0];
        a_s0_vbd[lane] = a_s0_vbd[0];
        a_s0_vbs[lane] = a_s0_vbs[0];
        a_s0_vgs[lane] = a_s0_vgs[0];
        a_s0_vds[lane] = a_s0_vds[0];
    }

    vds = v8_load(a_vds);
    vgs = v8_load(a_vgs);
    vbs = v8_load(a_vbs);
#if NGSPICE_OPT_GENERAL
    if (!is_pred || ckt->CKTtaPredActive) {
#else
    if (!is_pred) {
#endif
        vds = vds * blk->type;
        vgs = vgs * blk->type;
        vbs = vbs * blk->type;
    }

    v8d s0_vds = v8_load(a_s0_vds);
    v8d s0_vgs = v8_load(a_s0_vgs);
    v8d s0_vbs = v8_load(a_s0_vbs);
    v8d s0_vbd = v8_load(a_s0_vbd);
    v8d vgdo   = s0_vgs - s0_vds;
    v8d vbd    = vbs - vds;
    v8d vgd    = vgs - vds;
    v8d von    = dyn->von;

    /* 8-wide AVX-512 DEVfetlim + DEVlimvds */
    {
        __mmask8 m_s0_fwd = v8_ge(s0_vds, v8_set1(0.0));
        v8d vgs_f = v8_fetlim(vgs, s0_vgs, von);
        v8d vds_f = v8_limvds(vgs_f - vgd, s0_vds);

        v8d vgd_r = v8_fetlim(vgd, vgdo, von);
        v8d vds_r = -v8_limvds(-(vgs - vgd_r), -s0_vds);
        v8d vgs_r = vgd_r + vds_r;

        vgs = v8_sel(m_s0_fwd, vgs_f, vgs_r);
        vds = v8_sel(m_s0_fwd, vds_f, vds_r);
    }

    /* 8-wide AVX-512 DEVpnjlim */
    {
        __mmask8 m_vds_fwd = v8_ge(vds, v8_set1(0.0));
        v8d vj_in  = v8_sel(m_vds_fwd, vbs, vbd);
        v8d vj_old = v8_sel(m_vds_fwd, s0_vbs, s0_vbd);
        v8d vj_lim = v8_pnjlim(vj_in, vj_old, CONSTvt0, blk->vcrit_v, &chk_mask);
        vbs = v8_sel(m_vds_fwd, vj_lim, vj_lim + vds);
        chk_mask &= active_mask;
        if (chk_mask) {
            int noncon_add = __builtin_popcount((unsigned int)chk_mask);
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
    }
    }

    /* Level-2 fast path: cross-timestep stationary voltages -> reuse cached DC+C+Q
     * physics and only run charge integration + ag0 stamp assembly! */
    if (!dc_only && !need_aos && !chk_mask && vtol >= 0.0 && dyn->phys_valid &&
        dyn->last_gmin == ckt->CKTgmin)
    {
        int exact_zero = 0;
        if (B4P(blockCacheMatch)(dyn, vds, vgs, vbs, vtol,
                                 ckt->CKTreltol, ckt->CKTabstol, active_mask, &exact_zero)) {
            bsim4_avx512_integrate_and_assemble(blk, dyn, vds, vgs, vbs, !exact_zero,
                                                ag0, ag1, ag2, ag3,
                                                is_gear3, is_gear2, is_trap2, s0, s1, s2, s3);
            if (!exact_zero)
                dyn->last_active_step = ckt->CKTstat->STATaccepted;
            dyn->last_time = ckt->CKTtime;
            dyn->last_ag0 = ag0; dyn->last_ag1 = ag1; dyn->last_ag2 = ag2; dyn->last_ag3 = ag3;
            dyn->last_int_mode = int_mode;
            dyn->last_s0 = s0; dyn->last_s1 = s1; dyn->last_s2 = s2; dyn->last_s3 = s3;
            dyn->stamp_valid = 1;
            B4P(stampAvxDyn)(blk, dyn, InstArray, my_ax, my_rhs, stamp_mode);
            return;
        }
    }

    BSIM4Avx512Out out;
    bsim4_avx512_eval_block(blk, dyn, vds, vgs, vbs,
                            ckt->CKTgmin, ag0, ag1, ag2, ag3,
                            is_gear3, is_gear2, is_trap2, s0, s1, s2, s3, &out, need_aos, dc_only);

    if (dc_only) {
        dyn->phys_valid = 0;
        dyn->stamp_valid = 0;
        if (!need_aos) {
            for (int lane = 0; lane < nact; lane++)
                InstArray[blk->instIdx[lane]]->B4P(von) = out.von[lane];
        }
    } else {
        dyn->last_time = ckt->CKTtime;
        dyn->last_ag0 = ag0; dyn->last_ag1 = ag1; dyn->last_ag2 = ag2; dyn->last_ag3 = ag3;
        dyn->last_gmin = ckt->CKTgmin;
        dyn->last_int_mode = int_mode;
        dyn->last_s0 = s0; dyn->last_s1 = s1; dyn->last_s2 = s2; dyn->last_s3 = s3;
        dyn->last_active_step = ckt->CKTstat->STATaccepted;
        dyn->phys_valid = 1;
        dyn->stamp_valid = !need_aos;
    }

    if (need_aos) {
        for (int lane = 0; lane < nact; lane++) {
            B4P(instance) *here = InstArray[blk->instIdx[lane]];
            here->B4P(von)    = out.von[lane];
            here->B4P(mode)   = out.mode[lane];
            here->B4P(cd)     = out.cd[lane];
            here->B4P(gds)    = out.gds[lane];
            here->B4P(gm)     = out.gm[lane];
            here->B4P(gmbs)   = out.gmbs[lane];
            here->B4P(gbd)    = out.gbd[lane];
            here->B4P(gbs)    = out.gbs[lane];
            here->B4P(cbs)    = out.cbs[lane];
            here->B4P(cbd)    = out.cbd[lane];
            here->B4P(csub)   = out.csub[lane];
            here->B4P(gbds)   = out.gbds[lane];
            here->B4P(gbgs)   = out.gbgs[lane];
            here->B4P(gbbs)   = out.gbbs[lane];
            here->B4P(Igidl)  = out.Igidl[lane];
            here->B4P(ggidld) = out.ggidld[lane];
            here->B4P(ggidlg) = out.ggidlg[lane];
            here->B4P(ggidlb) = out.ggidlb[lane];
            here->B4P(ggidls) = -(out.ggidld[lane] + out.ggidlg[lane] + out.ggidlb[lane]);
            here->B4P(Igisl)  = out.Igisl[lane];
            here->B4P(ggisls) = out.ggisls[lane];
            here->B4P(ggislg) = out.ggislg[lane];
            here->B4P(ggislb) = out.ggislb[lane];
            here->B4P(ggisld) = -(out.ggisls[lane] + out.ggislg[lane] + out.ggislb[lane]);
        }
    }

    B4P(stampAvxDyn)(blk, dyn, InstArray, my_ax, my_rhs, stamp_mode);
}

/*
 * Stage 10E: Dynamic Active-Lane SIMD Compaction (Stream Compaction for BSIM4 AVX-512).
 * When a transient block has 1..compact_max moving lanes and >=1 stationary lanes,
 * stationary lanes take their exact L1/L2 bypass path while moving lanes are packed
 * across blocks into dense 8-wide AVX-512 batches (using 8x8 in-register AVX-512
 * transpose of BSIM4Avx512Inst rows + vptestmq mask packing) before calling
 * bsim4_avx512_eval_block once per 8 active transistors.
 * Runtime toggles:
 *   NGSPICE_NO_LANE_COMPACT=1   disable compaction (fall back to evalAvxBlock)
 */
typedef struct {
    int     len;
    int     src_bi[8];
    uint8_t src_lane[8];
    uint8_t is_last[8];
    double  vds[8];
    double  vgs[8];
    double  vbs[8];
} B4P(CompactQueue);

static inline void
B4P(flushCompactQueue)(
    B4P(CompactQueue) * restrict q,
    int has_trap_tunnel,
    const BSIM4Avx512Block * restrict blocks,
    const BSIM4Avx512Inst * restrict insts,
    BSIM4Avx512Dyn * restrict dyn,
    BSIM4Avx512Block * restrict sblk,
    BSIM4Avx512Dyn * restrict sdyn,
    B4P(instance) **InstArray,
    CKTcircuit *ckt,
    double * restrict my_ax,
    double * restrict my_rhs,
    int stamp_mode,
    double ag0, double ag1, double ag2, double ag3,
    int is_gear3, int is_gear2, int is_trap2, int int_mode,
    double * restrict s0,
    const double * restrict s1,
    const double * restrict s2,
    const double * restrict s3)
{
    const int qlen = q->len;
    if (qlen <= 0)
        return;

    const BSIM4Avx512Inst *rows[8];
    for (int k = 0; k < qlen; k++) {
        int bi = q->src_bi[k];
        int ln = (int)q->src_lane[k];
        rows[k] = &insts[bi * 8 + ln];
        sdyn->stateBase[k] = dyn[bi].stateBase[ln];
    }
    for (int k = qlen; k < 8; k++) {
        rows[k] = rows[0];
        q->vds[k] = q->vds[0];
        q->vgs[k] = q->vgs[0];
        q->vbs[k] = q->vbs[0];
        sdyn->stateBase[k] = sdyn->stateBase[0];
    }

    /* 8x8 AVX-512 in-register transpose of all B4_NUM_PARAM_V8 AoS params -> SoA sblk */
    v8d *dst_v = (v8d *)sblk;
    const double *p0 = rows[0]->p, *p1 = rows[1]->p, *p2 = rows[2]->p, *p3 = rows[3]->p;
    const double *p4 = rows[4]->p, *p5 = rows[5]->p, *p6 = rows[6]->p, *p7 = rows[7]->p;
    size_t g = 0;
    for (; g + 8 <= B4_NUM_PARAM_V8; g += 8) {
        b4_transpose_8x8(p0 + g, p1 + g, p2 + g, p3 + g,
                         p4 + g, p5 + g, p6 + g, p7 + g,
                         dst_v + g);
    }
    if (g < B4_NUM_PARAM_V8) {
        v8d tmp[8];
        b4_transpose_8x8(p0 + g, p1 + g, p2 + g, p3 + g,
                         p4 + g, p5 + g, p6 + g, p7 + g,
                         tmp);
        size_t rem = B4_NUM_PARAM_V8 - g;
        for (size_t i = 0; i < rem; i++)
            dst_v[g + i] = tmp[i];
    }

    /* Pack all 42 per-lane bitmasks + has_ud using 1-cycle AVX-512 vptestmq */
    __m512i v_mb = _mm512_set_epi64(
        (long long)rows[7]->mask_bits, (long long)rows[6]->mask_bits,
        (long long)rows[5]->mask_bits, (long long)rows[4]->mask_bits,
        (long long)rows[3]->mask_bits, (long long)rows[2]->mask_bits,
        (long long)rows[1]->mask_bits, (long long)rows[0]->mask_bits);
    const __m512i v_one = _mm512_set1_epi64(1LL);
    uint8_t *dst_m = (uint8_t *)&sblk->m_ngate_valid;
    for (int mk = 0; mk < B4_NUM_MASKS; mk++) {
        dst_m[mk] = (uint8_t)_mm512_test_epi64_mask(v_mb, v_one);
        v_mb = _mm512_srli_epi64(v_mb, 1);
    }
    sblk->has_ud = (_mm512_test_epi64_mask(v_mb, v_one) != 0) ? 1 : 0;
    sblk->n_active = (uint8_t)qlen;
    sdyn->n_active = (uint8_t)qlen;
    sblk->has_trap_tunnel = (uint8_t)has_trap_tunnel;
    {
        __mmask8 m_lt = sblk->m_xpart_lt05;
        __mmask8 m_gt = sblk->m_xpart_gt05;
        if (m_lt == 0xFF) sblk->xpart_mode = 0;
        else if (m_gt == 0xFF) sblk->xpart_mode = 1;
        else if ((m_lt | m_gt) == 0) sblk->xpart_mode = 2;
        else sblk->xpart_mode = 3;
    }

    BSIM4Avx512Out out;
    bsim4_avx512_eval_block(sblk, sdyn,
                            v8_load(q->vds), v8_load(q->vgs), v8_load(q->vbs),
                            ckt->CKTgmin, ag0, ag1, ag2, ag3,
                            is_gear3, is_gear2, is_trap2, s0, s1, s2, s3, &out, 0, 0);

    /* Scatter all 75 updated per-lane dyn fields back to each source block */
    const double (*src1)[8] = (const double (*)[8])&sdyn->von;
    const double (*src2)[8] = (const double (*)[8])&sdyn->hist_b;
    for (int k = 0; k < qlen; k++) {
        int bi = q->src_bi[k];
        int dl = (int)q->src_lane[k];
        BSIM4Avx512Dyn *d = &dyn[bi];
        double (*dst1)[8] = (double (*)[8])&d->von;
        for (int f = 0; f < B4_DYN_R1_VECS; f++)
            dst1[f][dl] = src1[f][k];
        double (*dst2)[8] = (double (*)[8])&d->hist_b;
        for (int f = 0; f < B4_DYN_R2_VECS; f++)
            dst2[f][dl] = src2[f][k];

        if (q->is_last[k]) {
            d->last_time = ckt->CKTtime;
            d->last_ag0 = ag0; d->last_ag1 = ag1; d->last_ag2 = ag2; d->last_ag3 = ag3;
            d->last_gmin = ckt->CKTgmin;
            d->last_int_mode = int_mode;
            d->last_s0 = s0; d->last_s1 = s1; d->last_s2 = s2; d->last_s3 = s3;
            d->last_active_step = ckt->CKTstat->STATaccepted;
            d->phys_valid = 1;
            d->stamp_valid = 1;
            if (stamp_mode == 0)
                B4P(stampAvxDyn)(&blocks[bi], d, InstArray, my_ax, my_rhs, stamp_mode);
        }
    }
    q->len = 0;
}

#ifdef KLU
static inline int
B4P(collectUniqueNodes)(const BSIM4Avx512Dyn * restrict d, int N, int out[32])
{
    int nu = 0;
    int nact = d->n_active;
    for (int lane = 0; lane < nact; lane++) {
        int nd[4] = {
            d->dNodePrime[lane], d->gNodePrime[lane],
            d->sNodePrime[lane], d->bNodePrime[lane]
        };
        for (int t = 0; t < 4; t++) {
            int u = nd[t];
            if (u <= 0 || u >= N) continue;
            int dup = 0;
            for (int k = 0; k < nu; k++) {
                if (out[k] == u) { dup = 1; break; }
            }
            if (!dup) out[nu++] = u;
        }
    }
    return nu;
}

static inline void
B4P(stampFrozenDelta)(
    const BSIM4Avx512Block * restrict blk,
    const BSIM4Avx512Dyn * restrict d,
    KLUmatrix *km,
    double ag0,
    double sign)
{
    int nact = d->n_active;
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

    v8d vs = v8_set1(sign);
    v8d vm = blk->m * vs;
    v8d mtype = vm * blk->type;

    v8d u_ceqqg = d->q_lin_g * mtype;
    v8d u_ceqqd = d->q_lin_d * mtype;
    v8d u_ceqqb = d->q_lin_b * mtype;

    double drhsd[8], drhsb[8], drhss[8];
    double crhsd[8], crhsg[8], crhsb[8], crhss[8];
    v8_store(drhsd, d->mdc_rhsd * vs);
    v8_store(drhsb, d->mdc_rhsb * vs);
    v8_store(drhss, d->mdc_rhss * vs);
    v8_store(crhsd, -u_ceqqd);
    v8_store(crhsg, -u_ceqqg);
    v8_store(crhsb, -u_ceqqb);
    v8_store(crhss, u_ceqqg + u_ceqqb + u_ceqqd);

    double c34[8], c35[8], c36[8], c37[8];
    double c44[8], c46[8], c47[8], c48[8];
    double c51[8], c52[8], c53[8], c55[8];
    double c58[8], c59[8], c60[8], c61[8];
    v8d vc34 = d->Cggb * vm, vc35 = d->Cgdb * vm, vc36 = d->Cgsb * vm;
    v8_store(c34, vc34); v8_store(c35, vc35); v8_store(c36, vc36);
    v8_store(c37, -(vc34 + vc35 + vc36));
    v8d vc44 = d->Cddb * vm, vc46 = d->Cdgb * vm, vc47 = d->Cdsb * vm;
    v8_store(c44, vc44); v8_store(c46, vc46); v8_store(c47, vc47);
    v8_store(c48, -(vc46 + vc44 + vc47));
    v8d vc51 = d->Csdb * vm, vc52 = d->Csgb * vm, vc53 = d->Cssb * vm;
    v8_store(c51, vc51); v8_store(c52, vc52); v8_store(c53, vc53);
    v8_store(c55, -(vc52 + vc51 + vc53));
    v8d vc58 = d->Cbdb * vm, vc59 = d->Cbgb * vm, vc60 = d->Cbsb * vm;
    v8_store(c58, vc58); v8_store(c59, vc59); v8_store(c60, vc60);
    v8_store(c61, -(vc59 + vc58 + vc60));

    double d44[8], d46[8], d47[8], d48[8];
    double d51[8], d52[8], d53[8], d55[8];
    double d58[8], d59[8], d60[8], d61[8];
    v8_store(d44,  d->mdc44 * vs); v8_store(d46,  d->mdc46 * vs);
    v8_store(d47, -d->mdc47 * vs); v8_store(d48, -d->mdc48 * vs);
    v8_store(d51, -d->mdc51 * vs); v8_store(d52,  d->mdc52 * vs);
    v8_store(d53,  d->mdc53 * vs); v8_store(d55, -d->mdc55 * vs);
    v8_store(d58,  d->mdc58 * vs); v8_store(d59,  d->mdc59 * vs);
    v8_store(d60,  d->mdc60 * vs); v8_store(d61,  d->mdc61 * vs);

#define B4_LAT_ADD_CAP(o, cv) do { \
    int _o = (o); \
    if (_o < inz) { \
        lcap[_o] += (cv); \
        if (upd_rcb) rcb[_o] += ag0 * (cv); \
    } \
} while (0)

#define B4_LAT_ADD_BOTH(o, dv, cv) do { \
    int _o = (o); \
    if (_o < inz) { \
        ldc[_o]  += (dv); \
        lcap[_o] += (cv); \
        if (upd_rcb) rcb[_o] += (dv) + ag0 * (cv); \
    } \
} while (0)

    for (int lane = 0; lane < nact; lane++) {
        const int *off = d->kluOff16[lane];
        int dn = d->dNodePrime[lane];
        int gn = d->gNodePrime[lane];
        int sn = d->sNodePrime[lane];
        int bn = d->bNodePrime[lane];
        if (dn) { rdc[dn] += drhsd[lane]; rcap[dn] += crhsd[lane]; }
        if (gn) { rcap[gn] += crhsg[lane]; }
        if (bn) { rdc[bn] += drhsb[lane]; rcap[bn] += crhsb[lane]; }
        if (sn) { rdc[sn] += drhss[lane]; rcap[sn] += crhss[lane]; }

        B4_LAT_ADD_CAP(off[0], c34[lane]);
        B4_LAT_ADD_CAP(off[1], c35[lane]);
        B4_LAT_ADD_BOTH(off[4], d44[lane], c44[lane]);
        B4_LAT_ADD_BOTH(off[5], d46[lane], c46[lane]);
        if (sn != 0 && bn != 0) {
            B4_LAT_ADD_CAP(off[2], c36[lane]);
            B4_LAT_ADD_CAP(off[3], c37[lane]);
            B4_LAT_ADD_BOTH(off[6],  d47[lane], c47[lane]);
            B4_LAT_ADD_BOTH(off[7],  d48[lane], c48[lane]);
            B4_LAT_ADD_BOTH(off[8],  d51[lane], c51[lane]);
            B4_LAT_ADD_BOTH(off[9],  d52[lane], c52[lane]);
            B4_LAT_ADD_BOTH(off[10], d53[lane], c53[lane]);
            B4_LAT_ADD_BOTH(off[11], d55[lane], c55[lane]);
            B4_LAT_ADD_BOTH(off[12], d58[lane], c58[lane]);
            B4_LAT_ADD_BOTH(off[13], d59[lane], c59[lane]);
            B4_LAT_ADD_BOTH(off[14], d60[lane], c60[lane]);
            B4_LAT_ADD_BOTH(off[15], d61[lane], c61[lane]);
        } else if (sn != 0) {
            B4_LAT_ADD_CAP(off[2], c36[lane]);
            B4_LAT_ADD_BOTH(off[6],  d47[lane], c47[lane]);
            B4_LAT_ADD_BOTH(off[8],  d51[lane], c51[lane]);
            B4_LAT_ADD_BOTH(off[9],  d52[lane], c52[lane]);
            B4_LAT_ADD_BOTH(off[10], d53[lane], c53[lane]);
        } else if (bn != 0) {
            B4_LAT_ADD_CAP(off[3], c37[lane]);
            B4_LAT_ADD_BOTH(off[7],  d48[lane], c48[lane]);
            B4_LAT_ADD_BOTH(off[12], d58[lane], c58[lane]);
            B4_LAT_ADD_BOTH(off[13], d59[lane], c59[lane]);
            B4_LAT_ADD_BOTH(off[15], d61[lane], c61[lane]);
        }
    }
    if (d->m_has_dint) {
        double mgd_arr[8];
        v8_store(mgd_arr, blk->mgd * vs);
        for (int lane = 0; lane < nact; lane++) {
            if (d->m_has_dint & (1u << lane)) {
                const int *off = blk->kluOff[lane];
                double mgd = mgd_arr[lane];
                if (off[16] < inz) { ldc[off[16]] -= mgd; if (upd_rcb) rcb[off[16]] -= mgd; }
                if (off[17] < inz) { ldc[off[17]] -= mgd; if (upd_rcb) rcb[off[17]] -= mgd; }
                if (off[18] < inz) { ldc[off[18]] += mgd; if (upd_rcb) rcb[off[18]] += mgd; }
            }
        }
    }
    if (d->m_has_sint) {
        double mgs_arr[8];
        v8_store(mgs_arr, blk->mgs * vs);
        for (int lane = 0; lane < nact; lane++) {
            if (d->m_has_sint & (1u << lane)) {
                const int *off = blk->kluOff[lane];
                double mgs = mgs_arr[lane];
                if (off[19] < inz) { ldc[off[19]] -= mgs; if (upd_rcb) rcb[off[19]] -= mgs; }
                if (off[20] < inz) { ldc[off[20]] -= mgs; if (upd_rcb) rcb[off[20]] -= mgs; }
                if (off[21] < inz) { ldc[off[21]] += mgs; if (upd_rcb) rcb[off[21]] += mgs; }
            }
        }
    }
#undef B4_LAT_ADD_CAP
#undef B4_LAT_ADD_BOTH
    km->KLUmatrixLatentOps++;
}

static inline void
B4P(syncFrozenStateOnWake)(const BSIM4Avx512Dyn * restrict d, CKTcircuit *ckt, int sync_s0)
{
    double *s0 = ckt->CKTstate0;
    double *s1 = ckt->CKTstate1;
    double *s2 = ckt->CKTstate2;
    double *s3 = ckt->CKTstates[3];
    double *s4 = ckt->CKTstates[4];
    int nact = d->n_active;
    for (int lane = 0; lane < nact; lane++) {
        int st = d->stateBase[lane];
        double vbd = d->a_vbd[lane], vbs = d->a_vbs[lane], vgs = d->a_vgs[lane], vds = d->a_vds[lane];
        double qb  = d->a_qb[lane],  qg  = d->a_qg[lane],  qd  = d->a_qd[lane];
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
}

static void
B4P(latentUnfreezeAll)(B4P(model) *head, CKTcircuit *ckt, KLUmatrix *km)
{
    BSIM4AvxLatentState *lat = (BSIM4AvxLatentState *)head->B4P(AvxLatent);
    if (!lat || lat->nFrozen == 0)
        return;
    const BSIM4Avx512Block *blocks = (const BSIM4Avx512Block *)head->B4P(AvxBlocks);
    BSIM4Avx512Dyn *dyn = (BSIM4Avx512Dyn *)head->B4P(AvxDyn);
    int nBlocks = lat->nBlocks;
    int sync_s0 = ((ckt->CKTmode & MODETRAN) && !(ckt->CKTmode & MODEINITPRED));
    double ag0 = ckt->CKTag[0];
    int all_mine = (km && km->KLUmatrixLatentCount == lat->nFrozen);

    for (int bi = 0; bi < nBlocks; bi++) {
        if (dyn[bi].latent_frozen) {
            if (!all_mine && km && km->KLUmatrixAxLatentDC)
                B4P(stampFrozenDelta)(&blocks[bi], &dyn[bi], km, ag0, -1.0);
            if (ckt->CKTmode & MODETRAN)
                B4P(syncFrozenStateOnWake)(&dyn[bi], ckt, sync_s0);
            dyn[bi].latent_frozen = 0;
            dyn[bi].latent_streak = 0;
            dyn[bi].stamp_valid = 0;
            dyn[bi].last_time = -1.0;
            dyn[bi].last_active_step = ckt->CKTstat->STATaccepted;
        }
        lat->activeList[bi] = bi;
        lat->activePos[bi] = bi;
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
    lat->nActive = nBlocks;
    lat->nFrozen = 0;
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

static BSIM4AvxLatentState *
B4P(ensureLatentState)(B4P(model) *head, CKTcircuit *ckt)
{
    int nBlocks = head->B4P(AvxBlockCount);
    int N = ckt->CKTmaxEqNum;
    BSIM4AvxLatentState *lat = (BSIM4AvxLatentState *)head->B4P(AvxLatent);
    if (lat && lat->nBlocks == nBlocks && lat->nNodes == N)
        return lat;

    B4P(freeAvxLatent)(lat);
    head->B4P(AvxLatent) = NULL;
    if (nBlocks <= 4 || N <= 1)
        return NULL;

    lat = (BSIM4AvxLatentState *)calloc(1, sizeof(BSIM4AvxLatentState));
    if (!lat) return NULL;
    lat->nBlocks = nBlocks;
    lat->nNodes = N;
    lat->nActive = nBlocks;
    lat->nFrozen = 0;
    lat->lastAccepted = -1;
    lat->lastGmin = ckt->CKTgmin;

    lat->activeList    = (int *)malloc((size_t)nBlocks * sizeof(int));
    lat->activePos     = (int *)malloc((size_t)nBlocks * sizeof(int));
    lat->nodeBlkPtr    = (int *)calloc((size_t)(N + 2), sizeof(int));
    lat->vSnap         = (double *)calloc((size_t)(N + 1), sizeof(double));
    lat->nodeFrozenCnt = (int *)calloc((size_t)(N + 1), sizeof(int));
    lat->cfFastList    = (int *)malloc((size_t)nBlocks * sizeof(int));
    lat->cfIsFastBlk   = (unsigned char *)calloc((size_t)nBlocks, sizeof(unsigned char));

    const BSIM4Avx512Dyn *dyn = (const BSIM4Avx512Dyn *)head->B4P(AvxDyn);
    int uniq[32];
    for (int bi = 0; bi < nBlocks; bi++) {
        lat->activeList[bi] = bi;
        lat->activePos[bi] = bi;
        int nu = B4P(collectUniqueNodes)(&dyn[bi], N, uniq);
        for (int k = 0; k < nu; k++)
            lat->nodeBlkPtr[uniq[k] + 1]++;
    }
    for (int i = 1; i <= N; i++)
        lat->nodeBlkPtr[i] += lat->nodeBlkPtr[i - 1];
    int total = lat->nodeBlkPtr[N];
    lat->nodeBlkIdx = (int *)malloc((size_t)(total > 0 ? total : 1) * sizeof(int));
    int *cur = (int *)malloc((size_t)(N + 1) * sizeof(int));
    memcpy(cur, lat->nodeBlkPtr, (size_t)(N + 1) * sizeof(int));
    for (int bi = 0; bi < nBlocks; bi++) {
        int nu = B4P(collectUniqueNodes)(&dyn[bi], N, uniq);
        for (int k = 0; k < nu; k++)
            lat->nodeBlkIdx[cur[uniq[k]]++] = bi;
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
    BSIM4AvxLatentState *lat_cur = (BSIM4AvxLatentState *)head->B4P(AvxLatent);

    /* Restore slow unfrozen BSIM4 blocks' states when returning from a compound-fast micro-burst */
    int cf_just_synced = 0;
    if (!cf_active && lat_cur && lat_cur->cfNeedsSync) {
        BSIM4Avx512Dyn *dyn = (BSIM4Avx512Dyn *)head->B4P(AvxDyn);
        int nBlocks = lat_cur->nBlocks;
        int sync_s0 = ((ckt->CKTmode & MODETRAN) && !(ckt->CKTmode & MODEINITPRED));
        for (int bi = 0; bi < nBlocks; bi++) {
            if (!lat_cur->cfIsFastBlk[bi] && !dyn[bi].latent_frozen && dyn[bi].phys_valid) {
                if (ckt->CKTmode & MODETRAN)
                    B4P(syncFrozenStateOnWake)(&dyn[bi], ckt, sync_s0);
                dyn[bi].last_time = -1.0;
            }
        }
        lat_cur->cfNeedsSync = 0;
        cf_just_synced = 1;
    }

    if (!lat_en && !cf_active)
        return;

    int nBlocks = head->B4P(AvxBlockCount);
    int nFb = head->B4P(AvxFallbackCount);
    int nthreads = 1;
#ifdef USE_OMP
    nthreads = omp_get_max_threads();
#endif
    int eligible = (head->B4P(AvxBuilt) && head->B4P(AvxDynValid) &&
                    B4P(isAvxEligibleTran)(ckt) && !B4P(AvxNeedAosSync) &&
                    nBlocks > 4 && nFb == 0 && nthreads == 1 &&
                    ckt->CKTtime > 0.0 && ckt->CKTsenInfo == NULL);
    if (!eligible) {
        if (lat_en)
            B4P(latentUnfreezeAll)(head, ckt, km);
        return;
    }

    BSIM4AvxLatentState *lat = B4P(ensureLatentState)(head, ckt);
    if (!lat) return;

    const BSIM4Avx512Block *blocks = (const BSIM4Avx512Block *)head->B4P(AvxBlocks);
    BSIM4Avx512Dyn *dyn = (BSIM4Avx512Dyn *)head->B4P(AvxDyn);
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
            const int * restrict nbptr = lat->nodeBlkPtr;
            const int * restrict nbidx = lat->nodeBlkIdx;
            int uniq[32];
            int woke_any = 0;

            int sub_only = (cf_active && ckt->CKTmrFastNodes && ckt->CKTmrSubNFast > 0);
            int n_loop = sub_only ? ckt->CKTmrSubNFast : (N - 1);
            for (int li = 0; li < n_loop; li++) {
                int node = sub_only ? ckt->CKTmrFastNodes[li] : (li + 1);
                if (node <= 0 || node >= N || fcnt[node] == 0) continue;
                double vnow = rhsOld[node];
                if (fabs(vnow - vsnap[node]) <= half_wake) continue;
                vsnap[node] = vnow;
                int p0 = nbptr[node], p1 = nbptr[node + 1];
                for (int p = p0; p < p1; p++) {
                    int bi = nbidx[p];
                    BSIM4Avx512Dyn *d = &dyn[bi];
                    if (!d->latent_frozen) continue;
                    B4P(stampFrozenDelta)(&blocks[bi], d, km, ag0, -1.0);
                    int nu = B4P(collectUniqueNodes)(d, N, uniq);
                    for (int k = 0; k < nu; k++)
                        lat->nodeFrozenCnt[uniq[k]]--;
                    B4P(syncFrozenStateOnWake)(d, ckt, !is_pred);
                    d->latent_frozen = 0;
                    d->latent_streak = 0;
                    d->stamp_valid = 0;
                    d->last_time = -1.0;
                    d->last_active_step = acc;
                    lat->activePos[bi] = lat->nActive;
                    lat->activeList[lat->nActive++] = bi;
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
            int uniq[32];

            for (int ai = lat->nActive - 1; ai >= 0; ai--) {
                int bi = lat->activeList[ai];
                if (cf_just_synced && !lat->cfIsFastBlk[bi])
                    continue;
                BSIM4Avx512Dyn *d = &dyn[bi];
                if (!d->phys_valid || !d->stamp_valid || d->last_gmin != ckt->CKTgmin) {
                    d->latent_streak = 0;
                    lat->statFailValid++;
                    continue;
                }
                int nact = d->n_active;
                __mmask8 active_mask = (__mmask8)((1u << nact) - 1u);
                double a_vds[8], a_vgs[8], a_vbs[8], id_arr[8];
                v8_store(id_arr, d->Idtot);
                int icap_ok = 1;
                for (int lane = 0; lane < nact; lane++) {
                    int st = d->stateBase[lane];
                    double tol_i = 0.1 * (reltol * fabs(id_arr[lane]) + abstol);
                    if (tol_i < icap_floor) tol_i = icap_floor;
                    if (fabs(s1[st + 12]) > tol_i ||
                        fabs(s1[st + 14]) > tol_i ||
                        fabs(s1[st + 16]) > tol_i)
                    {
                        icap_ok = 0;
                        break;
                    }
                    double vs = v_acc[d->sNodePrime[lane]];
                    a_vds[lane] = v_acc[d->dNodePrime[lane]] - vs;
                    a_vgs[lane] = v_acc[d->gNodePrime[lane]] - vs;
                    a_vbs[lane] = v_acc[d->bNodePrime[lane]] - vs;
                }
                if (!icap_ok) {
                    d->latent_streak = 0;
                    lat->statFailIcap++;
                    continue;
                }
                for (int lane = nact; lane < 8; lane++) {
                    a_vds[lane] = a_vds[0]; a_vgs[lane] = a_vgs[0]; a_vbs[lane] = a_vbs[0];
                }
                v8d vds_r = v8_load(a_vds) * d->type_v;
                v8d vgs_r = v8_load(a_vgs) * d->type_v;
                v8d vbs_r = v8_load(a_vbs) * d->type_v;
                int ez = 0;
                if (!B4P(blockCacheMatch)(d, vds_r, vgs_r, vbs_r, vtol, reltol, abstol, active_mask, &ez)) {
                    d->latent_streak = 0;
                    lat->statFailCache++;
                    continue;
                }
                int inc = (d->last_active_step <= prev_acc) ? delta_acc : 1;
                int ns = (int)d->latent_streak + inc;
                d->latent_streak = (uint8_t)(ns > 255 ? 255 : ns);
                if ((int)d->latent_streak < streak_req)
                    continue;

                int nu = B4P(collectUniqueNodes)(d, N, uniq);
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
                B4P(stampFrozenDelta)(&blocks[bi], d, km, ag0, +1.0);
                d->latent_frozen = 1;
                int last_bi = lat->activeList[--lat->nActive];
                lat->activeList[ai] = last_bi;
                lat->activePos[last_bi] = ai;
                lat->activePos[bi] = -1;
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
        memset(lat->cfIsFastBlk, 0, (size_t)nBlocks);
        const int * restrict fn = ckt->CKTmrFastNodes;
        int nF = ckt->CKTmrSubNFast;
        const int * restrict nbptr = lat->nodeBlkPtr;
        const int * restrict nbidx = lat->nodeBlkIdx;
        int n_fast_blk = 0;
        for (int fi = 0; fi < nF; fi++) {
            int node = fn[fi];
            if (node <= 0 || node >= N) continue;
            int p0 = nbptr[node], p1 = nbptr[node + 1];
            for (int p = p0; p < p1; p++) {
                int bi = nbidx[p];
                if (!lat->cfIsFastBlk[bi]) {
                    lat->cfIsFastBlk[bi] = 1;
                    if (!dyn[bi].latent_frozen)
                        lat->cfFastList[n_fast_blk++] = bi;
                }
            }
        }
        /* Include any non-frozen block that has never been evaluated yet */
        for (int bi = 0; bi < nBlocks; bi++) {
            if (!lat->cfIsFastBlk[bi] && !dyn[bi].latent_frozen && !dyn[bi].phys_valid) {
                lat->cfIsFastBlk[bi] = 1;
                lat->cfFastList[n_fast_blk++] = bi;
            }
        }
        lat->cfNFast = n_fast_blk;
        lat->cfEpoch = ckt->CKTmrSubEpoch;
        if (n_fast_blk < nBlocks - lat->nFrozen)
            lat->cfNeedsSync = 1;
    }
}
#endif /* KLU */

static inline void
B4P(evalAvxRangeList)(
    const BSIM4Avx512Block * restrict blocks,
    BSIM4Avx512Dyn * restrict dyn,
    int nBlocks,
    const int * restrict active_list,
    int n_iter_blks,
    B4P(instance) **InstArray,
    CKTcircuit *ckt,
    double * restrict my_ax,
    double * restrict my_rhs,
    int stamp_mode,
    int dc_only,
    int force_aos)
{
    static int compact_init = 0;
    static int compact_max = 5;
    if (!compact_init) {
        if (getenv("NGSPICE_NO_LANE_COMPACT") != NULL)
            compact_max = 0;
        compact_init = 1;
    }

    int need_aos = B4P(AvxNeedAosSync) || force_aos;
    double vtol = dc_only ? -1.0 : B4P(avxBypassTol)(ckt);
    if (dc_only || need_aos || nBlocks <= 4 || vtol < 0.0 || compact_max <= 0) {
        for (int ii = 0; ii < n_iter_blks; ii++) {
            int bi = active_list ? active_list[ii] : ii;
            B4P(evalAvxBlock)(&blocks[bi], &dyn[bi], InstArray, ckt,
                              my_ax, my_rhs, stamp_mode, dc_only, force_aos);
        }
        return;
    }

    const BSIM4Avx512Inst * restrict insts = (const BSIM4Avx512Inst *)(blocks + nBlocks);
    const double * restrict rhsOld = ckt->CKTrhsOld;
    double * restrict s0 = ckt->CKTstate0;
    const double * restrict s1 = ckt->CKTstate1;
    const double * restrict s2 = ckt->CKTstate2;
    const double * restrict s3 = ckt->CKTstate3;
    const int is_pred = ((ckt->CKTmode & MODEINITPRED) != 0);
    const int is_gear3 = (ckt->CKTintegrateMethod == GEAR && ckt->CKTorder == 3);
    const int is_gear2 = (ckt->CKTintegrateMethod == GEAR && ckt->CKTorder == 2);
    const int is_trap2 = (ckt->CKTintegrateMethod == TRAPEZOIDAL && ckt->CKTorder == 2);
    const int int_mode = is_gear3 ? 3 : (is_gear2 ? 2 : (is_trap2 ? 1 : 0));
    const double ag0 = ckt->CKTag[0];
    const double ag1 = ckt->CKTag[1];
    const double ag2 = ckt->CKTag[2];
    const double ag3 = ckt->CKTag[3];
    const double reltol = ckt->CKTreltol;
    const double abstol = ckt->CKTabstol;
    const double xfact = is_pred ? (ckt->CKTdelta / ckt->CKTdeltaOld[1]) : 0.0;
    const double one_p_xf = 1.0 + xfact;

    B4P(CompactQueue) q[2];
    q[0].len = 0;
    q[1].len = 0;
    BSIM4Avx512Block sblk;
    BSIM4Avx512Dyn sdyn;
    sblk.is_v6 = B4_IS_V6;

    for (int ii = 0; ii < n_iter_blks; ii++) {
        int bi = active_list ? active_list[ii] : ii;
        const BSIM4Avx512Block * restrict blk = &blocks[bi];
        BSIM4Avx512Dyn * restrict d = &dyn[bi];
        const int nact = d->n_active;
        const __mmask8 active_mask = (__mmask8)((1u << nact) - 1u);

        double a_vds[8], a_vgs[8], a_vbs[8];
        double a_s0_vds[8], a_s0_vgs[8], a_s0_vbs[8], a_s0_vbd[8];
        v8d vds, vgs, vbs;
        __mmask8 chk_mask = 0;

        int same_step = (!is_pred && d->stamp_valid &&
            d->last_time == ckt->CKTtime &&
            d->last_ag0 == ag0 && d->last_ag1 == ag1 && d->last_ag2 == ag2 &&
            (!is_gear3 || d->last_ag3 == ag3) &&
            d->last_gmin == ckt->CKTgmin && d->last_int_mode == int_mode &&
            d->last_s0 == s0 && d->last_s1 == s1 && d->last_s2 == s2 &&
            (!is_gear3 || d->last_s3 == s3));

        if (same_step) {
            for (int lane = 0; lane < nact; lane++) {
                double vs_node = rhsOld[d->sNodePrime[lane]];
                a_vds[lane] = rhsOld[d->dNodePrime[lane]] - vs_node;
                a_vgs[lane] = rhsOld[d->gNodePrime[lane]] - vs_node;
                a_vbs[lane] = rhsOld[d->bNodePrime[lane]] - vs_node;
            }
            for (int lane = nact; lane < 8; lane++) {
                a_vds[lane] = a_vds[0];
                a_vgs[lane] = a_vgs[0];
                a_vbs[lane] = a_vbs[0];
            }
            vds = v8_load(a_vds) * d->type_v;
            vgs = v8_load(a_vgs) * d->type_v;
            vbs = v8_load(a_vbs) * d->type_v;
            __mmask8 ez_mask = 0;
            __mmask8 hm = B4P(laneCacheHitMask)(d, vds, vgs, vbs, vtol,
                                                reltol, abstol, active_mask, &ez_mask);
            if (hm == active_mask) {
                if (ez_mask != active_mask) {
                    bsim4_avx512_l1_sync_state(d, vds, vgs, vbs, ag0, s0);
                    d->last_active_step = ckt->CKTstat->STATaccepted;
                }
                if (stamp_mode == 0)
                    B4P(stampAvxDyn)(blk, d, InstArray, my_ax, my_rhs, stamp_mode);
                continue;
            }
            /* L1 missed on >=1 lane: read s0 only (vds/vgs/vbs already loaded!) */
            for (int lane = 0; lane < nact; lane++) {
                int st = d->stateBase[lane];
                a_s0_vbd[lane] = s0[st + 0];
                a_s0_vbs[lane] = s0[st + 1];
                a_s0_vgs[lane] = s0[st + 2];
                a_s0_vds[lane] = s0[st + 3];
            }
            for (int lane = nact; lane < 8; lane++) {
                a_s0_vbd[lane] = a_s0_vbd[0];
                a_s0_vbs[lane] = a_s0_vbs[0];
                a_s0_vgs[lane] = a_s0_vgs[0];
                a_s0_vds[lane] = a_s0_vds[0];
            }
        } else {
#ifndef PREDICTOR
            if (is_pred) {
#if NGSPICE_OPT_GENERAL
                if (ckt->CKTtaPredActive) {
                    for (int lane = 0; lane < nact; lane++) {
                        int st = d->stateBase[lane];
                        double s1_vds = s1[st + 3], s1_vgs = s1[st + 2], s1_vbs = s1[st + 1];
                        double s1_vbd = s1_vbs - s1_vds;
                        s0[st + 3] = a_s0_vds[lane] = s1_vds;
                        s0[st + 2] = a_s0_vgs[lane] = s1_vgs;
                        s0[st + 1] = a_s0_vbs[lane] = s1_vbs;
                        s0[st + 0] = a_s0_vbd[lane] = s1_vbd;
                        double vs_node = rhsOld[d->sNodePrime[lane]];
                        a_vds[lane] = rhsOld[d->dNodePrime[lane]] - vs_node;
                        a_vgs[lane] = rhsOld[d->gNodePrime[lane]] - vs_node;
                        a_vbs[lane] = rhsOld[d->bNodePrime[lane]] - vs_node;
                    }
                } else
#endif
                {
                    for (int lane = 0; lane < nact; lane++) {
                        int st = d->stateBase[lane];
                        double s1_vds = s1[st + 3], s1_vgs = s1[st + 2], s1_vbs = s1[st + 1];
                        double s1_vbd = s1_vbs - s1_vds;
                        s0[st + 3] = a_s0_vds[lane] = s1_vds;
                        s0[st + 2] = a_s0_vgs[lane] = s1_vgs;
                        s0[st + 1] = a_s0_vbs[lane] = s1_vbs;
                        s0[st + 0] = a_s0_vbd[lane] = s1_vbd;
                        a_vds[lane] = one_p_xf * s1_vds - xfact * s2[st + 3];
                        a_vgs[lane] = one_p_xf * s1_vgs - xfact * s2[st + 2];
                        a_vbs[lane] = one_p_xf * s1_vbs - xfact * s2[st + 1];
                    }
                }
            } else
#endif
            {
                for (int lane = 0; lane < nact; lane++) {
                    int st = d->stateBase[lane];
                    double vs_node = rhsOld[d->sNodePrime[lane]];
                    a_vds[lane] = rhsOld[d->dNodePrime[lane]] - vs_node;
                    a_vgs[lane] = rhsOld[d->gNodePrime[lane]] - vs_node;
                    a_vbs[lane] = rhsOld[d->bNodePrime[lane]] - vs_node;
                    a_s0_vbd[lane] = s0[st + 0];
                    a_s0_vbs[lane] = s0[st + 1];
                    a_s0_vgs[lane] = s0[st + 2];
                    a_s0_vds[lane] = s0[st + 3];
                }
            }
            for (int lane = nact; lane < 8; lane++) {
                a_vds[lane] = a_vds[0];
                a_vgs[lane] = a_vgs[0];
                a_vbs[lane] = a_vbs[0];
                a_s0_vbd[lane] = a_s0_vbd[0];
                a_s0_vbs[lane] = a_s0_vbs[0];
                a_s0_vgs[lane] = a_s0_vgs[0];
                a_s0_vds[lane] = a_s0_vds[0];
            }
            vds = v8_load(a_vds);
            vgs = v8_load(a_vgs);
            vbs = v8_load(a_vbs);
#if NGSPICE_OPT_GENERAL
            if (!is_pred || ckt->CKTtaPredActive) {
#else
            if (!is_pred) {
#endif
                vds = vds * d->type_v;
                vgs = vgs * d->type_v;
                vbs = vbs * d->type_v;
            }
        }

        /* 8-wide AVX-512 DEVfetlim + DEVlimvds + DEVpnjlim */
        {
            v8d s0_vds = v8_load(a_s0_vds);
            v8d s0_vgs = v8_load(a_s0_vgs);
            v8d s0_vbs = v8_load(a_s0_vbs);
            v8d s0_vbd = v8_load(a_s0_vbd);
            v8d vgdo   = s0_vgs - s0_vds;
            v8d vbd    = vbs - vds;
            v8d vgd    = vgs - vds;
            v8d von    = d->von;

            __mmask8 m_s0_fwd = v8_ge(s0_vds, v8_set1(0.0));
            v8d vgs_f = v8_fetlim(vgs, s0_vgs, von);
            v8d vds_f = v8_limvds(vgs_f - vgd, s0_vds);

            v8d vgd_r = v8_fetlim(vgd, vgdo, von);
            v8d vds_r = -v8_limvds(-(vgs - vgd_r), -s0_vds);
            v8d vgs_r = vgd_r + vds_r;

            vgs = v8_sel(m_s0_fwd, vgs_f, vgs_r);
            vds = v8_sel(m_s0_fwd, vds_f, vds_r);

            __mmask8 m_vds_fwd = v8_ge(vds, v8_set1(0.0));
            v8d vj_in  = v8_sel(m_vds_fwd, vbs, vbd);
            v8d vj_old = v8_sel(m_vds_fwd, s0_vbs, s0_vbd);
            v8d vj_lim = v8_pnjlim(vj_in, vj_old, CONSTvt0, blk->vcrit_v, &chk_mask);
            vbs = v8_sel(m_vds_fwd, vj_lim, vj_lim + vds);
            chk_mask &= active_mask;
            if (chk_mask)
                ckt->CKTnoncon += __builtin_popcount((unsigned int)chk_mask);
        }

        if (!chk_mask && d->phys_valid && d->last_gmin == ckt->CKTgmin) {
            __mmask8 ez_mask = 0;
            __mmask8 hit_mask = B4P(laneCacheHitMask)(d, vds, vgs, vbs, vtol,
                                                      reltol, abstol, active_mask, &ez_mask);
            if (hit_mask == active_mask) {
                /* Full L2 (or post-limit L1) hit */
                if (same_step) {
                    if (ez_mask != active_mask) {
                        bsim4_avx512_l1_sync_state(d, vds, vgs, vbs, ag0, s0);
                        d->last_active_step = ckt->CKTstat->STATaccepted;
                    }
                } else {
                    bsim4_avx512_integrate_and_assemble(blk, d, vds, vgs, vbs,
                                                        (ez_mask != active_mask),
                                                        ag0, ag1, ag2, ag3,
                                                        is_gear3, is_gear2, is_trap2,
                                                        s0, s1, s2, s3);
                    if (ez_mask != active_mask)
                        d->last_active_step = ckt->CKTstat->STATaccepted;
                    d->last_time = ckt->CKTtime;
                    d->last_ag0 = ag0; d->last_ag1 = ag1; d->last_ag2 = ag2; d->last_ag3 = ag3;
                    d->last_int_mode = int_mode;
                    d->last_s0 = s0; d->last_s1 = s1; d->last_s2 = s2; d->last_s3 = s3;
                    d->stamp_valid = 1;
                }
                if (stamp_mode == 0)
                    B4P(stampAvxDyn)(blk, d, InstArray, my_ax, my_rhs, stamp_mode);
                continue;
            }

            __mmask8 miss_mask = active_mask & ~hit_mask;
            int n_miss = __builtin_popcount((unsigned int)miss_mask);
            if (hit_mask && n_miss <= compact_max) {
                /* Partial miss: service stationary lanes in-place, compact moving lanes! */
                if (same_step) {
                    __mmask8 sync_m = hit_mask & ~ez_mask;
                    if (sync_m)
                        bsim4_avx512_l1_sync_state_mask(d, vds, vgs, vbs, ag0, s0, sync_m);
                } else {
                    bsim4_avx512_integrate_and_assemble_mask(blk, d, vds, vgs, vbs, 1,
                                                             ag0, ag1, ag2, ag3,
                                                             is_gear3, is_gear2, is_trap2,
                                                             s0, s1, s2, s3, hit_mask);
                }

                double lvds[8], lvgs[8], lvbs[8];
                v8_store(lvds, vds);
                v8_store(lvgs, vgs);
                v8_store(lvbs, vbs);
                int t = blk->has_trap_tunnel ? 1 : 0;
                B4P(CompactQueue) *qt = &q[t];
                unsigned int mm = (unsigned int)miss_mask;
                int last_lane = 31 - __builtin_clz(mm);
                while (mm) {
                    int lane = __builtin_ctz(mm);
                    mm &= mm - 1u;
                    int pos = qt->len++;
                    qt->src_bi[pos]   = bi;
                    qt->src_lane[pos] = (uint8_t)lane;
                    qt->is_last[pos]  = (uint8_t)(lane == last_lane);
                    qt->vds[pos]      = lvds[lane];
                    qt->vgs[pos]      = lvgs[lane];
                    qt->vbs[pos]      = lvbs[lane];
                    if (qt->len == 8) {
                        B4P(flushCompactQueue)(qt, t, blocks, insts, dyn, &sblk, &sdyn,
                                               InstArray, ckt, my_ax, my_rhs, stamp_mode,
                                               ag0, ag1, ag2, ag3,
                                               is_gear3, is_gear2, is_trap2, int_mode,
                                               s0, s1, s2, s3);
                    }
                }
                continue;
            }
        }

        /* Full or heavy miss: evaluate block in-place directly (zero copy overhead) */
        BSIM4Avx512Out out;
        bsim4_avx512_eval_block(blk, d, vds, vgs, vbs,
                                ckt->CKTgmin, ag0, ag1, ag2, ag3,
                                is_gear3, is_gear2, is_trap2, s0, s1, s2, s3, &out, 0, 0);
        d->last_time = ckt->CKTtime;
        d->last_ag0 = ag0; d->last_ag1 = ag1; d->last_ag2 = ag2; d->last_ag3 = ag3;
        d->last_gmin = ckt->CKTgmin;
        d->last_int_mode = int_mode;
        d->last_s0 = s0; d->last_s1 = s1; d->last_s2 = s2; d->last_s3 = s3;
        d->last_active_step = ckt->CKTstat->STATaccepted;
        d->phys_valid = 1;
        d->stamp_valid = 1;
        if (stamp_mode == 0)
            B4P(stampAvxDyn)(blk, d, InstArray, my_ax, my_rhs, stamp_mode);
    }

    if (q[0].len > 0) {
        B4P(flushCompactQueue)(&q[0], 0, blocks, insts, dyn, &sblk, &sdyn,
                               InstArray, ckt, my_ax, my_rhs, stamp_mode,
                               ag0, ag1, ag2, ag3,
                               is_gear3, is_gear2, is_trap2, int_mode,
                               s0, s1, s2, s3);
    }
    if (q[1].len > 0) {
        B4P(flushCompactQueue)(&q[1], 1, blocks, insts, dyn, &sblk, &sdyn,
                               InstArray, ckt, my_ax, my_rhs, stamp_mode,
                               ag0, ag1, ag2, ag3,
                               is_gear3, is_gear2, is_trap2, int_mode,
                               s0, s1, s2, s3);
    }
    if (stamp_mode != 0) {
        for (int bi = 0; bi < nBlocks; bi++)
            B4P(stampAvxDyn)(&blocks[bi], &dyn[bi], InstArray, my_ax, my_rhs, stamp_mode);
    }
}

static inline void
B4P(evalAvxRange)(
    const BSIM4Avx512Block * restrict blocks,
    BSIM4Avx512Dyn * restrict dyn,
    int nBlocks,
    B4P(instance) **InstArray,
    CKTcircuit *ckt,
    double * restrict my_ax,
    double * restrict my_rhs,
    int stamp_mode,
    int dc_only,
    int force_aos)
{
    B4P(evalAvxRangeList)(blocks, dyn, nBlocks, NULL, nBlocks, InstArray, ckt,
                          my_ax, my_rhs, stamp_mode, dc_only, force_aos);
}

/*
 * 8-wide AVX-512 convergence test over AvxBlocks + AvxDyn (zero AoS/state chase!).
 * Returns: -1 if AVX-512 convTest not applicable, 0 if all AVX blocks converged, 1 if noncon.
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

    const BSIM4Avx512Dyn *dyn = (const BSIM4Avx512Dyn *)head->B4P(AvxDyn);
    int nBlocks = head->B4P(AvxBlockCount);
    const double * restrict rhsOld = ckt->CKTrhsOld;
    const v8d v_reltol = v8_set1(ckt->CKTreltol);
    const v8d v_abstol = v8_set1(ckt->CKTabstol);

    int noncon = 0;
#ifdef USE_OMP
    if (nBlocks > 64 && omp_get_max_threads() > 1) {
        #pragma omp parallel for reduction(|:noncon)
        for (int bi = 0; bi < nBlocks; bi++) {
            if (noncon) continue;
            const BSIM4Avx512Dyn *d = &dyn[bi];
            int nact = d->n_active;
            double a_vds[8], a_vgs[8], a_vbs[8];
            for (int lane = 0; lane < nact; lane++) {
                double vs = rhsOld[d->sNodePrime[lane]];
                a_vds[lane] = rhsOld[d->dNodePrime[lane]] - vs;
                a_vgs[lane] = rhsOld[d->gNodePrime[lane]] - vs;
                a_vbs[lane] = rhsOld[d->bNodePrime[lane]] - vs;
            }
            for (int lane = nact; lane < 8; lane++) {
                a_vds[lane] = a_vds[0]; a_vgs[lane] = a_vgs[0]; a_vbs[lane] = a_vbs[0];
            }
            v8d delvds = v8_load(a_vds) * d->type_v - d->last_vds;
            v8d delvgs = v8_load(a_vgs) * d->type_v - d->last_vgs;
            v8d delvbs = v8_load(a_vbs) * d->type_v - d->last_vbs;

            v8d cd_diff = d->cd_g * delvgs + d->cd_d * delvds + d->cd_b * delvbs;
            v8d cdhat = d->Idtot + cd_diff;
            v8d tol0 = v_reltol * v8_max(v8_abs(cdhat), v8_abs(d->Idtot)) + v_abstol;

            v8d cb_diff = d->cb_g * delvgs + d->cb_d * delvds + d->cb_b * delvbs;
            v8d cbhat = d->Ibtot + cb_diff;
            v8d tol6 = v_reltol * v8_max(v8_abs(cbhat), v8_abs(d->Ibtot)) + v_abstol;

            __mmask8 active_mask = (__mmask8)((1u << nact) - 1u);
            __mmask8 fail = (v8_ge(v8_abs(cd_diff), tol0) | v8_gt(v8_abs(cb_diff), tol6)) & active_mask;
            if (fail)
                noncon = 1;
        }
        return noncon;
    }
#endif
    const BSIM4AvxLatentState *lat = (const BSIM4AvxLatentState *)head->B4P(AvxLatent);
    const int *act_list = NULL;
    int n_iter = nBlocks;
    if (ckt->CKTmrSubActive && lat && lat->cfEpoch == ckt->CKTmrSubEpoch) {
        act_list = lat->cfFastList;
        n_iter = lat->cfNFast;
    } else if (lat && lat->nFrozen > 0) {
        act_list = lat->activeList;
        n_iter = lat->nActive;
    }
    for (int ii = 0; ii < n_iter; ii++) {
        int bi = act_list ? act_list[ii] : ii;
        const BSIM4Avx512Dyn *d = &dyn[bi];
        int nact = d->n_active;
        double a_vds[8], a_vgs[8], a_vbs[8];
        for (int lane = 0; lane < nact; lane++) {
            double vs = rhsOld[d->sNodePrime[lane]];
            a_vds[lane] = rhsOld[d->dNodePrime[lane]] - vs;
            a_vgs[lane] = rhsOld[d->gNodePrime[lane]] - vs;
            a_vbs[lane] = rhsOld[d->bNodePrime[lane]] - vs;
        }
        for (int lane = nact; lane < 8; lane++) {
            a_vds[lane] = a_vds[0]; a_vgs[lane] = a_vgs[0]; a_vbs[lane] = a_vbs[0];
        }
        v8d delvds = v8_load(a_vds) * d->type_v - d->last_vds;
        v8d delvgs = v8_load(a_vgs) * d->type_v - d->last_vgs;
        v8d delvbs = v8_load(a_vbs) * d->type_v - d->last_vbs;

        v8d cd_diff = d->cd_g * delvgs + d->cd_d * delvds + d->cd_b * delvbs;
        v8d cdhat = d->Idtot + cd_diff;
        v8d tol0 = v_reltol * v8_max(v8_abs(cdhat), v8_abs(d->Idtot)) + v_abstol;

        v8d cb_diff = d->cb_g * delvgs + d->cb_d * delvds + d->cb_b * delvbs;
        v8d cbhat = d->Ibtot + cb_diff;
        v8d tol6 = v_reltol * v8_max(v8_abs(cbhat), v8_abs(d->Ibtot)) + v_abstol;

        __mmask8 active_mask = (__mmask8)((1u << nact) - 1u);
        __mmask8 fail = (v8_ge(v8_abs(cd_diff), tol0) | v8_gt(v8_abs(cb_diff), tol6)) & active_mask;
        if (fail)
            return 1;
    }
    return 0;
}

/*
 * 8-wide AVX-512 truncation error over AvxBlocks (zero AoS chase!).
 * Handles ord == 1 and ord == 2 when CKTterrCacheValid is set.
 * Returns -1 if not applicable (caller falls back to scalar), 0 if handled.
 */
static inline double
B4P(terrBlock8)(
    const BSIM4Avx512Dyn * restrict dyn,
    const double * restrict s0,
    const double * restrict s1,
    const double * restrict s2,
    const double * restrict s3,
    const double * restrict s4,
    int ord,
    v8d v_abstol, v8d v_reltol, v8d v_relInvD, v8d v_chgtol,
    v8d v_trtol, v8d v_tfac,
    v8d v_w0, v8d v_w1, v8d v_w2, v8d v_w3, v8d v_w4,
    double cur)
{
    int nact = dyn->n_active;
    __mmask8 active_mask = (__mmask8)((1u << nact) - 1u);
    double aq0[3][8], aq1[3][8], aq2[3][8], aq3[3][8], aq4[3][8], ac0[3][8], ac1[3][8];

    if (ord == 3) {
        for (int lane = 0; lane < nact; lane++) {
            int st = dyn->stateBase[lane];
            aq0[0][lane] = s0[st + 11]; ac0[0][lane] = s0[st + 12];
            aq0[1][lane] = s0[st + 13]; ac0[1][lane] = s0[st + 14];
            aq0[2][lane] = s0[st + 15]; ac0[2][lane] = s0[st + 16];
            aq1[0][lane] = s1[st + 11]; ac1[0][lane] = s1[st + 12];
            aq1[1][lane] = s1[st + 13]; ac1[1][lane] = s1[st + 14];
            aq1[2][lane] = s1[st + 15]; ac1[2][lane] = s1[st + 16];
            aq2[0][lane] = s2[st + 11];
            aq2[1][lane] = s2[st + 13];
            aq2[2][lane] = s2[st + 15];
            aq3[0][lane] = s3[st + 11];
            aq3[1][lane] = s3[st + 13];
            aq3[2][lane] = s3[st + 15];
            aq4[0][lane] = s4[st + 11];
            aq4[1][lane] = s4[st + 13];
            aq4[2][lane] = s4[st + 15];
        }
        for (int lane = nact; lane < 8; lane++) {
            for (int c = 0; c < 3; c++) {
                aq0[c][lane] = aq0[c][0]; ac0[c][lane] = ac0[c][0];
                aq1[c][lane] = aq1[c][0]; ac1[c][lane] = ac1[c][0];
                aq2[c][lane] = aq2[c][0]; aq3[c][lane] = aq3[c][0];
                aq4[c][lane] = aq4[c][0];
            }
        }
        for (int c = 0; c < 3; c++) {
            v8d q0 = v8_load(aq0[c]), q1 = v8_load(aq1[c]);
            v8d c0 = v8_abs(v8_load(ac0[c])), c1 = v8_abs(v8_load(ac1[c]));
            v8d volttol = v_abstol + v_reltol * v8_max(c0, c1);
            v8d qmax = v8_max(v8_abs(q0), v8_abs(q1));
            v8d chargetol = v_relInvD * v8_max(qmax, v_chgtol);
            v8d tol = v8_max(volttol, chargetol);
            v8d d0 = v_w0 * q0 + v_w1 * q1 + v_w2 * v8_load(aq2[c])
                   + v_w3 * v8_load(aq3[c]) + v_w4 * v8_load(aq4[c]);
            v8d den = v8_max(v_abstol, v_tfac * v8_abs(d0));
            v8d num = v_trtol * tol;
            if (cur >= 1e150 || (v8_lt(num, v8_set1(cur * cur * cur) * den) & active_mask)) {
                double ratio[8];
                v8_store(ratio, num / den);
                for (int lane = 0; lane < nact; lane++) {
                    double del = cbrt(ratio[lane]);
                    if (del < cur)
                        cur = del;
                }
            }
        }
    } else if (ord == 2) {
        for (int lane = 0; lane < nact; lane++) {
            int st = dyn->stateBase[lane];
            aq0[0][lane] = s0[st + 11]; ac0[0][lane] = s0[st + 12];
            aq0[1][lane] = s0[st + 13]; ac0[1][lane] = s0[st + 14];
            aq0[2][lane] = s0[st + 15]; ac0[2][lane] = s0[st + 16];
            aq1[0][lane] = s1[st + 11]; ac1[0][lane] = s1[st + 12];
            aq1[1][lane] = s1[st + 13]; ac1[1][lane] = s1[st + 14];
            aq1[2][lane] = s1[st + 15]; ac1[2][lane] = s1[st + 16];
            aq2[0][lane] = s2[st + 11];
            aq2[1][lane] = s2[st + 13];
            aq2[2][lane] = s2[st + 15];
            aq3[0][lane] = s3[st + 11];
            aq3[1][lane] = s3[st + 13];
            aq3[2][lane] = s3[st + 15];
        }
        for (int lane = nact; lane < 8; lane++) {
            for (int c = 0; c < 3; c++) {
                aq0[c][lane] = aq0[c][0]; ac0[c][lane] = ac0[c][0];
                aq1[c][lane] = aq1[c][0]; ac1[c][lane] = ac1[c][0];
                aq2[c][lane] = aq2[c][0]; aq3[c][lane] = aq3[c][0];
            }
        }
        for (int c = 0; c < 3; c++) {
            v8d q0 = v8_load(aq0[c]), q1 = v8_load(aq1[c]);
            v8d c0 = v8_abs(v8_load(ac0[c])), c1 = v8_abs(v8_load(ac1[c]));
            v8d volttol = v_abstol + v_reltol * v8_max(c0, c1);
            v8d qmax = v8_max(v8_abs(q0), v8_abs(q1));
            v8d chargetol = v_relInvD * v8_max(qmax, v_chgtol);
            v8d tol = v8_max(volttol, chargetol);
            v8d d0 = v_w0 * q0 + v_w1 * q1 + v_w2 * v8_load(aq2[c]) + v_w3 * v8_load(aq3[c]);
            v8d den = v8_max(v_abstol, v_tfac * v8_abs(d0));
            v8d num = v_trtol * tol;
            if (cur >= 1e150 || (v8_lt(num, v8_set1(cur * cur) * den) & active_mask)) {
                double del[8];
                v8_store(del, v8_sqrt(num / den));
                for (int lane = 0; lane < nact; lane++) {
                    if (del[lane] < cur)
                        cur = del[lane];
                }
            }
        }
    } else {
        for (int lane = 0; lane < nact; lane++) {
            int st = dyn->stateBase[lane];
            aq0[0][lane] = s0[st + 11]; ac0[0][lane] = s0[st + 12];
            aq0[1][lane] = s0[st + 13]; ac0[1][lane] = s0[st + 14];
            aq0[2][lane] = s0[st + 15]; ac0[2][lane] = s0[st + 16];
            aq1[0][lane] = s1[st + 11]; ac1[0][lane] = s1[st + 12];
            aq1[1][lane] = s1[st + 13]; ac1[1][lane] = s1[st + 14];
            aq1[2][lane] = s1[st + 15]; ac1[2][lane] = s1[st + 16];
            aq2[0][lane] = s2[st + 11];
            aq2[1][lane] = s2[st + 13];
            aq2[2][lane] = s2[st + 15];
        }
        for (int lane = nact; lane < 8; lane++) {
            for (int c = 0; c < 3; c++) {
                aq0[c][lane] = aq0[c][0]; ac0[c][lane] = ac0[c][0];
                aq1[c][lane] = aq1[c][0]; ac1[c][lane] = ac1[c][0];
                aq2[c][lane] = aq2[c][0];
            }
        }
        for (int c = 0; c < 3; c++) {
            v8d q0 = v8_load(aq0[c]), q1 = v8_load(aq1[c]);
            v8d c0 = v8_abs(v8_load(ac0[c])), c1 = v8_abs(v8_load(ac1[c]));
            v8d volttol = v_abstol + v_reltol * v8_max(c0, c1);
            v8d qmax = v8_max(v8_abs(q0), v8_abs(q1));
            v8d chargetol = v_relInvD * v8_max(qmax, v_chgtol);
            v8d tol = v8_max(volttol, chargetol);
            v8d d0 = v_w0 * q0 + v_w1 * q1 + v_w2 * v8_load(aq2[c]);
            v8d den = v8_max(v_abstol, v_tfac * v8_abs(d0));
            v8d num = v_trtol * tol;
            if (cur >= 1e150 || (v8_lt(num, v8_set1(cur) * den) & active_mask)) {
                double del[8];
                v8_store(del, num / den);
                for (int lane = 0; lane < nact; lane++) {
                    if (del[lane] < cur)
                        cur = del[lane];
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

    const BSIM4Avx512Dyn *dyn = (const BSIM4Avx512Dyn *)head->B4P(AvxDyn);
    int nBlocks = head->B4P(AvxBlockCount);
    const double * restrict s0 = ckt->CKTstate0;
    const double * restrict s1 = ckt->CKTstate1;
    const double * restrict s2 = ckt->CKTstates[2];
    const double * restrict s3 = (ord >= 2) ? ckt->CKTstates[3] : NULL;
    const double * restrict s4 = (ord >= 3) ? ckt->CKTstates[4] : NULL;
    const double *w = ckt->CKTterrWeights;

    v8d v_abstol  = v8_set1(ckt->CKTabstol);
    v8d v_reltol  = v8_set1(ckt->CKTreltol);
    v8d v_relInvD = v8_set1(ckt->CKTterrReltolInvDelta);
    v8d v_chgtol  = v8_set1(ckt->CKTchgtol);
    v8d v_trtol   = v8_set1(ckt->CKTtrtol);
    v8d v_tfac    = v8_set1(ckt->CKTterrFactor);
    v8d v_w0      = v8_set1(w[0]);
    v8d v_w1      = v8_set1(w[1]);
    v8d v_w2      = v8_set1(w[2]);
    v8d v_w3      = v8_set1(ord >= 2 ? w[3] : 0.0);
    v8d v_w4      = v8_set1(ord >= 3 ? w[4] : 0.0);

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
    if (nBlocks > 32 && omp_get_max_threads() > 1) {
        #pragma omp parallel for reduction(min:minStep)
        for (int bi = 0; bi < nBlocks; bi++) {
            if (mr_trunc_en && dyn[bi].phys_valid && dyn[bi].last_active_step < active_cutoff)
                continue;
            minStep = B4P(terrBlock8)(&dyn[bi], s0, s1, s2, s3, s4, ord,
                                      v_abstol, v_reltol, v_relInvD, v_chgtol,
                                      v_trtol, v_tfac, v_w0, v_w1, v_w2, v_w3, v_w4, minStep);
        }
    } else
#endif
    {
        const BSIM4AvxLatentState *lat = (const BSIM4AvxLatentState *)head->B4P(AvxLatent);
        const int *act_list = NULL;
        int n_iter = nBlocks;
        if (ckt->CKTmrSubActive && lat && lat->cfEpoch == ckt->CKTmrSubEpoch) {
            act_list = lat->cfFastList;
            n_iter = lat->cfNFast;
        } else if (lat && lat->nFrozen > 0) {
            act_list = lat->activeList;
            n_iter = lat->nActive;
        }
        for (int ii = 0; ii < n_iter; ii++) {
            int bi = act_list ? act_list[ii] : ii;
            if (mr_trunc_en && dyn[bi].phys_valid && dyn[bi].last_active_step < active_cutoff)
                continue;
            minStep = B4P(terrBlock8)(&dyn[bi], s0, s1, s2, s3, s4, ord,
                                      v_abstol, v_reltol, v_relInvD, v_chgtol,
                                      v_trtol, v_tfac, v_w0, v_w1, v_w2, v_w3, v_w4, minStep);
        }
    }

    int nFb = head->B4P(AvxFallbackCount);
    if (nFb > 0) {
        const int *fbIdx = head->B4P(AvxFallbackIdx);
        B4P(instance) **InstArray = head->B4P(InstanceArray);
        for (int k = 0; k < nFb; k++) {
            B4P(instance) *inst = InstArray[fbIdx[k]];
            CKTterr(inst->B4P(qb), ckt, &minStep);
            CKTterr(inst->B4P(qg), ckt, &minStep);
            CKTterr(inst->B4P(qd), ckt, &minStep);
            if (inst->B4P(trnqsMod))
                CKTterr(inst->B4P(qcdump), ckt, &minStep);
            if (inst->B4P(rbodyMod)) {
                CKTterr(inst->B4P(qbs), ckt, &minStep);
                CKTterr(inst->B4P(qbd), ckt, &minStep);
            }
            if (inst->B4P(rgateMod) == 3)
                CKTterr(inst->B4P(qgmid), ckt, &minStep);
        }
    }

    *timeStep = minStep;
    return 0;
}

#endif /* NGSPICE_OPT_ZEN4 */
