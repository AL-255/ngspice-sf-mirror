/*
 * Per-version glue (BSIM4v5 / BSIM4v6) for Apple Silicon / ARM64 NEON SoA evaluation.
 * Included inside b4v5ld.c (with B4_IS_V6=0) and b4v6ld.c (with B4_IS_V6=1).
 *
 * Expects macros:
 *   B4P(name) -> BSIM4v5##name  or  BSIM4v6##name
 *   B4_IS_V6  -> 0 or 1
 */
#if NGSPICE_OPT_APPLE

#include "ngspice/bsim4_neon_core.h"

int B4P(AvxNeedAosSync) = 0;

static B4P(instance) **B4P(sort_inst_arr) = NULL;

static int
B4P(cmpEligIdx)(const void *pa, const void *pb)
{
    int ia = *(const int *)pa;
    int ib = *(const int *)pb;
    B4P(instance) *ha = B4P(sort_inst_arr)[ia];
    B4P(instance) *hb = B4P(sort_inst_arr)[ib];
    uintptr_t ma = (uintptr_t)B4P(modPtr)(ha);
    uintptr_t mb = (uintptr_t)B4P(modPtr)(hb);
    if (ma != mb) return (ma < mb) ? -1 : 1;
    int ta = ha->B4P(hasTrapTunnel) ? 1 : 0;
    int tb = hb->B4P(hasTrapTunnel) ? 1 : 0;
    if (ta != tb) return ta - tb;
    int da = (ha->B4P(dNode) != ha->B4P(dNodePrime)) ? 1 : 0;
    int db = (hb->B4P(dNode) != hb->B4P(dNodePrime)) ? 1 : 0;
    if (da != db) return da - db;
    int sa = (ha->B4P(sNode) != ha->B4P(sNodePrime)) ? 1 : 0;
    int sb = (hb->B4P(sNode) != hb->B4P(sNodePrime)) ? 1 : 0;
    if (sa != sb) return sa - sb;
    return ia - ib;
}

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
        (ckt->CKTorder != 1 && ckt->CKTorder != 2))
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
    free(head->B4P(AvxL2));
    head->B4P(AvxL2) = NULL;
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
        qsort(eligIdx, (size_t)nElig, sizeof(int), B4P(cmpEligIdx));
    }

    int nBlocks = (nElig + B4N_W - 1) / B4N_W;
    BSIM4NeonConst *consts = NULL;
    BSIM4NeonHot   *hots   = NULL;
    BSIM4NeonL2    *l2s    = NULL;
    if (nBlocks > 0) {
        if (posix_memalign((void **)&consts, 128, (size_t)nBlocks * sizeof(BSIM4NeonConst)) != 0)
            consts = NULL;
        if (consts && posix_memalign((void **)&hots, 128, (size_t)nBlocks * sizeof(BSIM4NeonHot)) != 0) {
            free(consts); consts = NULL; hots = NULL;
        }
        if (consts && posix_memalign((void **)&l2s, 128, (size_t)nBlocks * sizeof(BSIM4NeonL2)) != 0) {
            free(consts); free(hots); consts = NULL; hots = NULL; l2s = NULL;
        }
        if (hots) memset(hots, 0, (size_t)nBlocks * sizeof(BSIM4NeonHot));
        if (l2s)  memset(l2s,  0, (size_t)nBlocks * sizeof(BSIM4NeonL2));
    }
    if (nBlocks > 0 && consts == NULL) {
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
        BSIM4NeonConst *blk = &consts[bi];
        BSIM4NeonHot   *hot = &hots[bi];
        BSIM4NeonL2    *l2  = &l2s[bi];
        memset(blk, 0, sizeof(*blk));

        int base = bi * B4N_W;
        int nact = nElig - base;
        if (nact > B4N_W) nact = B4N_W;
        hot->n_active = (uint8_t)nact;
        hot->active_mask = vn_mask_from_bits((uint8_t)((1u << nact) - 1u));
        blk->is_v6 = B4_IS_V6;

        double t_type[B4N_W], t_m[B4N_W], t_nf[B4N_W], t_vth0s[B4N_W], t_vfbphi[B4N_W], t_vfbzb[B4N_W], t_vbsc[B4N_W];
        double t_phi[B4N_W], t_phi095[B4N_W], t_sqrtPhi[B4N_W], t_V0[B4N_W], t_vtm[B4N_W], t_coxe[B4N_W], t_invCoxe[B4N_W];
        double t_coxp[B4N_W], t_toxe[B4N_W], t_invToxe[B4N_W], t_toxp[B4N_W], t_toxctm[B4N_W], t_invtoxctm[B4N_W];
        double t_inv2e8[B4N_W], t_link[B4N_W], t_leff[B4N_W], t_invLeff[B4N_W], t_sqrtLeff[B4N_W], t_weff[B4N_W];
        double t_weffCV[B4N_W], t_leffCV[B4N_W], t_CoxWL[B4N_W], t_CoxWL_oc[B4N_W], t_gdpr[B4N_W], t_gspr[B4N_W];
        double t_ados[B4N_W], t_bdos[B4N_W], t_vcrit[B4N_W];
        double t_iNvtms[B4N_W], t_iNvtmd[B4N_W], t_Ssat[B4N_W], t_Dsat[B4N_W], t_vjsm[B4N_W], t_vjdm[B4N_W], t_IVjsm[B4N_W], t_IVjdm[B4N_W];
        double t_iNvrs[B4N_W], t_iNvrd[B4N_W], t_iNvrssw[B4N_W], t_iNvrdsw[B4N_W], t_iNvrsswg[B4N_W], t_iNvrdswg[B4N_W];
        double t_vtss[B4N_W], t_vtsd[B4N_W], t_vtssws[B4N_W], t_vtsswd[B4N_W], t_vtsswgs[B4N_W], t_vtsswgd[B4N_W];
        double t_SjRev[B4N_W], t_DjRev[B4N_W], t_SswRev[B4N_W], t_DswRev[B4N_W], t_SswgRev[B4N_W], t_DswgRev[B4N_W];
        double t_xdep0s[B4N_W], t_fac1[B4N_W], t_dvt0[B4N_W], t_dvt1l[B4N_W], t_dvt2[B4N_W], t_dvt0w[B4N_W], t_dvt1wwl[B4N_W], t_dvt2w[B4N_W];
        double t_k1[B4N_W], t_k1ox[B4N_W], t_k2ox[B4N_W], t_k3[B4N_W], t_k3b[B4N_W], t_lpeb[B4N_W], t_vthNW[B4N_W], t_vthC[B4N_W], t_kt2T[B4N_W];
        double t_eta0[B4N_W], t_etab[B4N_W], t_th0vb0[B4N_W], t_nfac[B4N_W], t_cit[B4N_W], t_cdsc[B4N_W], t_cdscb[B4N_W], t_cdscd[B4N_W];
        double t_dvtp0[B4N_W], t_dvtp1[B4N_W], t_polyT1[B4N_W], t_mstar[B4N_W], t_voffcbn[B4N_W], t_coxeCdep[B4N_W];
        double t_dwg[B4N_W], t_dwb[B4N_W], t_prwg[B4N_W], t_prwb[B4N_W], t_rds0h[B4N_W], t_rdswmin[B4N_W], t_xj[B4N_W], t_a0[B4N_W];
        double t_b0wb1[B4N_W], t_agsa0[B4N_W], t_keta[B4N_W], t_abCVf[B4N_W];
        double t_ua[B4N_W], t_ub[B4N_W], t_uc[B4N_W], t_ud[B4N_W], t_u0t[B4N_W], t_vsatt[B4N_W], t_a1[B4N_W], t_a2[B4N_W], t_delta[B4N_W];
        double t_lam[B4N_W], t_litl[B4N_W], t_fprout[B4N_W], t_pvag[B4N_W], t_pclm[B4N_W], t_ipclml[B4N_W], t_thRout[B4N_W], t_ithRout[B4N_W];
        double t_pdiblb[B4N_W], t_pdits[B4N_W], t_ipdits[B4N_W], t_pditsd[B4N_W], t_pditsl[B4N_W], t_pscbe1l[B4N_W], t_ipscbe2[B4N_W];
        double t_alpl[B4N_W], t_beta0[B4N_W], t_ivtl[B4N_W], t_vtfb2[B4N_W];
        double t_i3toxe[B4N_W], t_agidlw[B4N_W], t_bgidl[B4N_W], t_cgidl[B4N_W], t_egidl[B4N_W];
        double t_agislw[B4N_W], t_bgisl[B4N_W], t_cgisl[B4N_W], t_egisl[B4N_W];
        double t_noff[B4N_W], t_voffcv[B4N_W], t_acde[B4N_W], t_ldeb[B4N_W], t_moin[B4N_W], t_cgdo[B4N_W], t_cgso[B4N_W], t_cgbo[B4N_W];
        double t_wcgdl[B4N_W], t_ckd[B4N_W], t_ickd[B4N_W], t_wcgsl[B4N_W], t_cks[B4N_W], t_icks[B4N_W];
        double t_czbs[B4N_W], t_czbssw[B4N_W], t_czbsswg[B4N_W], t_czbd[B4N_W], t_czbdsw[B4N_W], t_czbdswg[B4N_W];
        double t_PBS[B4N_W], t_iPBS[B4N_W], t_PBSWS[B4N_W], t_iPBSWS[B4N_W], t_PBSWGS[B4N_W], t_iPBSWGS[B4N_W];
        double t_PBD[B4N_W], t_iPBD[B4N_W], t_PBSWD[B4N_W], t_iPBSWD[B4N_W], t_PBSWGD[B4N_W], t_iPBSWGD[B4N_W];
        double t_MJS[B4N_W], t_i1MJS[B4N_W], t_MJSWS[B4N_W], t_i1MJSWS[B4N_W], t_MJSWGS[B4N_W], t_i1MJSWGS[B4N_W];
        double t_MJD[B4N_W], t_i1MJD[B4N_W], t_MJSWD[B4N_W], t_i1MJSWD[B4N_W], t_MJSWGD[B4N_W], t_i1MJSWGD[B4N_W];
        double t_jSsum[B4N_W], t_jDsum[B4N_W], t_jSlin[B4N_W], t_jDlin[B4N_W];

        int all_xp_lt05 = 1, all_xp_gt05 = 1, all_xp_eq05 = 1;
        int has_ud = 0;

        for (int lane = 0; lane < B4N_W; lane++) {
            int src_idx = (lane < nact) ? eligIdx[base + lane] : eligIdx[base];
            B4P(instance) *h = InstArray[src_idx];
            B4P(model) *m = B4P(modPtr)(h);
#if B4_IS_V6
            struct bsim4v6SizeDependParam *p = h->pParam;
#else
            struct bsim4v5SizeDependParam *p = h->pParam;
#endif
            uint8_t bit = (uint8_t)(1u << lane);

            double leff = p->B4P(leff);
            double TempRatio = ckt->CKTtemp / m->B4P(tnom) - 1.0;

            t_type[lane]     = (double)m->B4P(type);
            t_m[lane]        = h->B4P(m);
            t_nf[lane]       = h->B4P(nf);
            if (h->B4P(nf) != 1.0) blk->f_nf_ne1 |= bit;
            t_vth0s[lane]    = (double)m->B4P(type) * h->B4P(vth0);
            t_vfbphi[lane]   = h->B4P(vfb) + p->B4P(phi);
            t_vfbzb[lane]    = h->B4P(vfbzb);
            if (h->B4P(vfbzb) <= 0.0) blk->f_vfbzb_le0 |= bit;
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
            hot->mgd[lane]   = h->B4P(m) * h->B4P(drainConductance);
            hot->mgs[lane]   = h->B4P(m) * h->B4P(sourceConductance);
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
            if (h->B4P(SourceSatCurrent) > 0.0) blk->f_src_sat_pos |= bit;
            if (h->B4P(DrainSatCurrent) > 0.0)  blk->f_drn_sat_pos |= bit;
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
                if (t_SjRev[lane]   != 0.0) blk->f_tt_s    |= bit;
                if (t_DjRev[lane]   != 0.0) blk->f_tt_d    |= bit;
                if (t_SswRev[lane]  != 0.0) blk->f_tt_ssw  |= bit;
                if (t_DswRev[lane]  != 0.0) blk->f_tt_dsw  |= bit;
                if (t_SswgRev[lane] != 0.0) blk->f_tt_sswg |= bit;
                if (t_DswgRev[lane] != 0.0) blk->f_tt_dswg |= bit;
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
            if (p->B4P(dvt0w) != 0.0) blk->f_dvt0w |= bit;

            t_k1[lane]     = p->B4P(k1);
            t_k1ox[lane]   = p->B4P(k1ox);
            if (p->B4P(k1ox) == 0.0) blk->f_k1ox_zero |= bit;
            if (p->B4P(k1ox) <= 0.0) blk->f_k1ox_le0 |= bit;
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
            if (p->B4P(dvtp0) > 0.0) blk->f_dvtp0 |= bit;

            if (p->B4P(ngate) > 1.0e18 && p->B4P(ngate) < 1.0e25) {
                blk->f_ngate_valid |= bit;
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
            if (p->B4P(a1) != 0.0) blk->f_has_a1 |= bit;
            t_a2[lane]     = p->B4P(a2);
            t_delta[lane]  = p->B4P(delta);
            t_lam[lane]    = p->B4P(lambda);
            t_litl[lane]   = p->B4P(litl);
            if (m->B4P(lambdaGiven) && m->B4P(lambda) > 0.0) blk->f_lambda |= bit;
            t_fprout[lane] = p->B4P(fprout);
            if (p->B4P(fprout) > 0.0) blk->f_fprout |= bit;
            t_pvag[lane]   = p->B4P(pvag);
            t_pclm[lane]   = p->B4P(pclm);
            if (p->B4P(pclm) > B4V_MIN_EXP) {
                blk->f_pclm |= bit;
                t_ipclml[lane] = 1.0 / (p->B4P(pclm) * p->B4P(litl));
            } else {
                t_ipclml[lane] = 0.0;
            }
            t_thRout[lane] = p->B4P(thetaRout);
            if (p->B4P(thetaRout) > B4V_MIN_EXP) {
                blk->f_thetaRout |= bit;
                t_ithRout[lane] = 1.0 / p->B4P(thetaRout);
            } else {
                t_ithRout[lane] = 0.0;
            }
            t_pdiblb[lane] = p->B4P(pdiblb);
            t_pdits[lane]  = p->B4P(pdits);
            t_pditsd[lane] = p->B4P(pditsd);
            t_pditsl[lane] = 1.0 + m->B4P(pditsl) * leff;
            if (p->B4P(pdits) > B4V_MIN_EXP) {
                blk->f_pdits |= bit;
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
                blk->f_pscbe |= bit;
                t_ipscbe2[lane] = 1.0 / p->B4P(pscbe2);
            } else {
                t_ipscbe2[lane] = 0.0;
            }
            double atmp = p->B4P(alpha0) + p->B4P(alpha1) * leff;
            if (atmp > 0.0 && p->B4P(beta0) > 0.0) {
                blk->f_isub |= bit;
                t_alpl[lane] = atmp / leff;
            } else {
                t_alpl[lane] = 0.0;
            }
            t_beta0[lane] = p->B4P(beta0);
            if (m->B4P(vtlGiven) && m->B4P(vtl) > 0.0) {
                blk->f_vtl |= bit;
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
                blk->f_gidl_en |= bit;
#if B4_IS_V6
            t_agislw[lane] = p->B4P(agisl) * p->B4P(weffCJ);
            t_bgisl[lane]  = p->B4P(bgisl);
            t_cgisl[lane]  = p->B4P(cgisl);
            t_egisl[lane]  = p->B4P(egisl);
            if (p->B4P(agisl) > 0.0 && p->B4P(bgisl) > 0.0 && p->B4P(cgisl) > 0.0)
                blk->f_gisl_en |= bit;
#else
            t_agislw[lane] = t_agidlw[lane];
            t_bgisl[lane]  = t_bgidl[lane];
            t_cgisl[lane]  = t_cgidl[lane];
            t_egisl[lane]  = t_egidl[lane];
            if (p->B4P(agidl) > 0.0 && p->B4P(bgidl) > 0.0 && p->B4P(cgidl) > 0.0)
                blk->f_gisl_en |= bit;
#endif

            t_noff[lane]   = p->B4P(noff);
            t_voffcv[lane] = p->B4P(voffcv);
            t_acde[lane]   = p->B4P(acde);
            t_ldeb[lane]   = p->B4P(ldeb);
            t_moin[lane]   = p->B4P(moin);
            if (m->B4P(xpart) > 0.5) {
                blk->f_xpart_gt05 |= bit;
                all_xp_lt05 = 0; all_xp_eq05 = 0;
            } else if (m->B4P(xpart) < 0.5) {
                blk->f_xpart_lt05 |= bit;
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
            if (czbs > 0.0)    blk->f_czbs |= bit;
            if (czbssw > 0.0)  blk->f_czbssw |= bit;
            if (czbsswg > 0.0) blk->f_czbsswg |= bit;
            if (czbd > 0.0)    blk->f_czbd |= bit;
            if (czbdsw > 0.0)  blk->f_czbdsw |= bit;
            if (czbdswg > 0.0) blk->f_czbdswg |= bit;

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
            if (mjs == 0.5)    blk->f_mjs_half |= bit;
            if (mjsws == 0.5)  blk->f_mjsws_half |= bit;
            if (mjswgs == 0.5) blk->f_mjswgs_half |= bit;
            if (mjd == 0.5)    blk->f_mjd_half |= bit;
            if (mjswd == 0.5)  blk->f_mjswd_half |= bit;
            if (mjswgd == 0.5) blk->f_mjswgd_half |= bit;

            t_jSsum[lane] = czbs + czbssw + czbsswg;
            t_jDsum[lane] = czbd + czbdsw + czbdswg;
            t_jSlin[lane] = czbs * mjs * t_iPBS[lane] + czbssw * mjsws * t_iPBSWS[lane] + czbsswg * mjswgs * t_iPBSWGS[lane];
            t_jDlin[lane] = czbd * mjd * t_iPBD[lane] + czbdsw * mjswd * t_iPBSWD[lane] + czbdswg * mjswgd * t_iPBSWGD[lane];

            if (lane < nact) {
                if (h->B4P(dNode) != h->B4P(dNodePrime)) hot->m_has_dint |= bit;
                if (h->B4P(sNode) != h->B4P(sNodePrime)) hot->m_has_sint |= bit;
            }

            t_vcrit[lane]         = m->B4P(vcrit);
            hot->dNodePrime[lane] = h->B4P(dNodePrime);
            hot->gNodePrime[lane] = h->B4P(gNodePrime);
            hot->sNodePrime[lane] = h->B4P(sNodePrime);
            hot->bNodePrime[lane] = h->B4P(bNodePrime);
            hot->stateBase[lane]  = h->B4P(states);
            hot->instIdx[lane]    = src_idx;

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
                    hot->kluOff[lane][k] = (d < ax_bytes) ? (int)(d / sizeof(double)) : inz;
                }
            }
        }

        blk->has_ud = (uint8_t)has_ud;
        blk->xpart_mode = all_xp_lt05 ? 0 : (all_xp_gt05 ? 1 : (all_xp_eq05 ? 2 : 3));

        /* Expand scalar flags to NEON vector masks once */
#define B4N_EXP_MASK(name) blk->m_##name = vn_mask_from_bits(blk->f_##name)
        B4N_EXP_MASK(ngate_valid); B4N_EXP_MASK(dvt0w); B4N_EXP_MASK(dvtp0);
        B4N_EXP_MASK(fprout); B4N_EXP_MASK(pclm); B4N_EXP_MASK(thetaRout);
        B4N_EXP_MASK(pdits); B4N_EXP_MASK(pscbe); B4N_EXP_MASK(isub);
        B4N_EXP_MASK(vtl); B4N_EXP_MASK(lambda); B4N_EXP_MASK(gidl_en);
        B4N_EXP_MASK(gisl_en); B4N_EXP_MASK(xpart_gt05); B4N_EXP_MASK(xpart_lt05);
        B4N_EXP_MASK(src_sat_pos); B4N_EXP_MASK(drn_sat_pos);
        B4N_EXP_MASK(czbs); B4N_EXP_MASK(czbssw); B4N_EXP_MASK(czbsswg);
        B4N_EXP_MASK(czbd); B4N_EXP_MASK(czbdsw); B4N_EXP_MASK(czbdswg);
        B4N_EXP_MASK(mjs_half); B4N_EXP_MASK(mjsws_half); B4N_EXP_MASK(mjswgs_half);
        B4N_EXP_MASK(mjd_half); B4N_EXP_MASK(mjswd_half); B4N_EXP_MASK(mjswgd_half);
        B4N_EXP_MASK(k1ox_zero); B4N_EXP_MASK(k1ox_le0); B4N_EXP_MASK(vfbzb_le0);
        B4N_EXP_MASK(nf_ne1); B4N_EXP_MASK(has_a1);
        B4N_EXP_MASK(tt_s); B4N_EXP_MASK(tt_d); B4N_EXP_MASK(tt_ssw);
        B4N_EXP_MASK(tt_dsw); B4N_EXP_MASK(tt_sswg); B4N_EXP_MASK(tt_dswg);
#undef B4N_EXP_MASK

        /* Hot & L2 copies of type/m/vcrit so Level-1 & Level-2 NEVER touch Const */
        hot->type    = vn_load(t_type);
        hot->vcrit_v = vn_load(t_vcrit);
        l2->m        = vn_load(t_m);
        l2->mtype    = l2->m * hot->type;

        blk->type = hot->type; blk->m = l2->m; blk->nf = vn_load(t_nf);
        blk->vth0_signed = vn_load(t_vth0s); blk->vfb_plus_phi = vn_load(t_vfbphi);
        blk->vfbzb = vn_load(t_vfbzb); blk->vbsc = vn_load(t_vbsc);
        blk->phi = vn_load(t_phi); blk->phi_095 = vn_load(t_phi095);
        blk->sqrtPhi = vn_load(t_sqrtPhi); blk->V0 = vn_load(t_V0);
        blk->vtm = vn_load(t_vtm); blk->coxe = vn_load(t_coxe); blk->invCoxe = vn_load(t_invCoxe);
        blk->coxp = vn_load(t_coxp); blk->toxe = vn_load(t_toxe); blk->invToxe = vn_load(t_invToxe);
        blk->toxp = vn_load(t_toxp); blk->tox_ctm = vn_load(t_toxctm); blk->inv_tox_ctm = vn_load(t_invtoxctm);
        blk->inv_2e8_toxp = vn_load(t_inv2e8); blk->link_ctm = vn_load(t_link);
        blk->leff = vn_load(t_leff); blk->invLeff = vn_load(t_invLeff); blk->sqrtLeff = vn_load(t_sqrtLeff);
        blk->weff = vn_load(t_weff); blk->weffCV = vn_load(t_weffCV); blk->leffCV = vn_load(t_leffCV);
        blk->CoxWL = vn_load(t_CoxWL); blk->CoxWL_ov_coxe = vn_load(t_CoxWL_oc);
        blk->drainCond = vn_load(t_gdpr); blk->sourceCond = vn_load(t_gspr);
        blk->ados1p9 = vn_load(t_ados); blk->bdos0p7 = vn_load(t_bdos);

        blk->invNvtms = vn_load(t_iNvtms); blk->invNvtmd = vn_load(t_iNvtmd);
        blk->SourceSatCurrent = vn_load(t_Ssat); blk->DrainSatCurrent = vn_load(t_Dsat);
        blk->vjsmFwd = vn_load(t_vjsm); blk->vjdmFwd = vn_load(t_vjdm);
        blk->IVjsmFwd = vn_load(t_IVjsm); blk->IVjdmFwd = vn_load(t_IVjdm);
        blk->invNvtmrs_s = vn_load(t_iNvrs); blk->invNvtmrs_d = vn_load(t_iNvrd);
        blk->invNvtmrssw_s = vn_load(t_iNvrssw); blk->invNvtmrssw_d = vn_load(t_iNvrdsw);
        blk->invNvtmrsswg_s = vn_load(t_iNvrsswg); blk->invNvtmrsswg_d = vn_load(t_iNvrdswg);
        blk->vtss = vn_load(t_vtss); blk->vtsd = vn_load(t_vtsd);
        blk->vtssws = vn_load(t_vtssws); blk->vtsswd = vn_load(t_vtsswd);
        blk->vtsswgs = vn_load(t_vtsswgs); blk->vtsswgd = vn_load(t_vtsswgd);
        blk->SjctRevSat = vn_load(t_SjRev); blk->DjctRevSat = vn_load(t_DjRev);
        blk->SswRevSat = vn_load(t_SswRev); blk->DswRevSat = vn_load(t_DswRev);
        blk->SswgRevSat = vn_load(t_SswgRev); blk->DswgRevSat = vn_load(t_DswgRev);

        blk->xdep0OverSqrtPhi = vn_load(t_xdep0s); blk->factor1 = vn_load(t_fac1);
        blk->dvt0 = vn_load(t_dvt0); blk->dvt1_leff = vn_load(t_dvt1l); blk->dvt2 = vn_load(t_dvt2);
        blk->dvt0w = vn_load(t_dvt0w); blk->dvt1w_weff_leff = vn_load(t_dvt1wwl); blk->dvt2w = vn_load(t_dvt2w);
        blk->k1 = vn_load(t_k1); blk->k1ox = vn_load(t_k1ox); blk->k2ox = vn_load(t_k2ox);
        blk->k3 = vn_load(t_k3); blk->k3b = vn_load(t_k3b);
        blk->lpebFactor = vn_load(t_lpeb); blk->vthNarrowW = vn_load(t_vthNW);
        blk->vth_const_T1 = vn_load(t_vthC); blk->kt2_TempRatio = vn_load(t_kt2T);
        blk->eta0 = vn_load(t_eta0); blk->etab = vn_load(t_etab); blk->theta0vb0 = vn_load(t_th0vb0);
        blk->nfactor = vn_load(t_nfac); blk->cit = vn_load(t_cit);
        blk->cdsc = vn_load(t_cdsc); blk->cdscb = vn_load(t_cdscb); blk->cdscd = vn_load(t_cdscd);
        blk->dvtp0 = vn_load(t_dvtp0); blk->dvtp1 = vn_load(t_dvtp1);

        blk->polyT1 = vn_load(t_polyT1); blk->mstar = vn_load(t_mstar);
        blk->voffcbn = vn_load(t_voffcbn); blk->coxe_ov_cdep0 = vn_load(t_coxeCdep);

        blk->dwg = vn_load(t_dwg); blk->dwb = vn_load(t_dwb);
        blk->prwg = vn_load(t_prwg); blk->prwb = vn_load(t_prwb);
        blk->rds0_half = vn_load(t_rds0h); blk->rdswmin = vn_load(t_rdswmin);
        blk->xj = vn_load(t_xj); blk->a0 = vn_load(t_a0);
        blk->b0_ov_wb1 = vn_load(t_b0wb1); blk->ags_a0 = vn_load(t_agsa0);
        blk->keta = vn_load(t_keta); blk->abulkCVfactor = vn_load(t_abCVf);

        blk->ua = vn_load(t_ua); blk->ub = vn_load(t_ub); blk->uc = vn_load(t_uc); blk->ud = vn_load(t_ud);
        blk->u0temp = vn_load(t_u0t); blk->vsattemp = vn_load(t_vsatt);
        blk->a1 = vn_load(t_a1); blk->a2 = vn_load(t_a2); blk->delta = vn_load(t_delta);
        blk->lambda = vn_load(t_lam); blk->litl = vn_load(t_litl);
        blk->fprout = vn_load(t_fprout); blk->pvag = vn_load(t_pvag);
        blk->pclm = vn_load(t_pclm); blk->inv_pclm_litl = vn_load(t_ipclml);
        blk->thetaRout = vn_load(t_thRout); blk->inv_thetaRout = vn_load(t_ithRout);
        blk->pdiblb = vn_load(t_pdiblb);
        blk->pdits = vn_load(t_pdits); blk->inv_pdits = vn_load(t_ipdits);
        blk->pditsd = vn_load(t_pditsd); blk->pditsl_term = vn_load(t_pditsl);
        blk->pscbe1_litl = vn_load(t_pscbe1l); blk->inv_pscbe2 = vn_load(t_ipscbe2);
        blk->alpha_ov_leff = vn_load(t_alpl); blk->beta0 = vn_load(t_beta0);
        blk->inv_vtl_tfactor = vn_load(t_ivtl); blk->vtfbphi2 = vn_load(t_vtfb2);

        blk->inv_3toxe = vn_load(t_i3toxe);
        blk->agidl_weffCJ = vn_load(t_agidlw); blk->bgidl = vn_load(t_bgidl);
        blk->cgidl = vn_load(t_cgidl); blk->egidl = vn_load(t_egidl);
        blk->agisl_weffCJ = vn_load(t_agislw); blk->bgisl = vn_load(t_bgisl);
        blk->cgisl = vn_load(t_cgisl); blk->egisl = vn_load(t_egisl);

        blk->noff = vn_load(t_noff); blk->voffcv = vn_load(t_voffcv);
        blk->acde = vn_load(t_acde); blk->ldeb = vn_load(t_ldeb); blk->moin = vn_load(t_moin);
        blk->cgdo = vn_load(t_cgdo); blk->cgso = vn_load(t_cgso); blk->cgbo = vn_load(t_cgbo);
        blk->weffCV_cgdl = vn_load(t_wcgdl); blk->ckappad = vn_load(t_ckd); blk->inv_ckappad = vn_load(t_ickd);
        blk->weffCV_cgsl = vn_load(t_wcgsl); blk->ckappas = vn_load(t_cks); blk->inv_ckappas = vn_load(t_icks);

        blk->czbs = vn_load(t_czbs); blk->czbssw = vn_load(t_czbssw); blk->czbsswg = vn_load(t_czbsswg);
        blk->czbd = vn_load(t_czbd); blk->czbdsw = vn_load(t_czbdsw); blk->czbdswg = vn_load(t_czbdswg);
        blk->PhiBS = vn_load(t_PBS); blk->invPhiBS = vn_load(t_iPBS);
        blk->PhiBSWS = vn_load(t_PBSWS); blk->invPhiBSWS = vn_load(t_iPBSWS);
        blk->PhiBSWGS = vn_load(t_PBSWGS); blk->invPhiBSWGS = vn_load(t_iPBSWGS);
        blk->PhiBD = vn_load(t_PBD); blk->invPhiBD = vn_load(t_iPBD);
        blk->PhiBSWD = vn_load(t_PBSWD); blk->invPhiBSWD = vn_load(t_iPBSWD);
        blk->PhiBSWGD = vn_load(t_PBSWGD); blk->invPhiBSWGD = vn_load(t_iPBSWGD);
        blk->MJS = vn_load(t_MJS); blk->inv1mMJS = vn_load(t_i1MJS);
        blk->MJSWS = vn_load(t_MJSWS); blk->inv1mMJSWS = vn_load(t_i1MJSWS);
        blk->MJSWGS = vn_load(t_MJSWGS); blk->inv1mMJSWGS = vn_load(t_i1MJSWGS);
        blk->MJD = vn_load(t_MJD); blk->inv1mMJD = vn_load(t_i1MJD);
        blk->MJSWD = vn_load(t_MJSWD); blk->inv1mMJSWD = vn_load(t_i1MJSWD);
        blk->MJSWGD = vn_load(t_MJSWGD); blk->inv1mMJSWGD = vn_load(t_i1MJSWGD);
        blk->jctS_sum = vn_load(t_jSsum); blk->jctD_sum = vn_load(t_jDsum);
        blk->jctS_lin = vn_load(t_jSlin); blk->jctD_lin = vn_load(t_jDlin);
    }

    FREE(eligIdx);
    head->B4P(AvxBlocks) = consts;
    head->B4P(AvxDyn) = hots;
    head->B4P(AvxL2) = l2s;
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
    BSIM4NeonHot *hots = (BSIM4NeonHot *)head->B4P(AvxDyn);
    B4P(instance) **InstArray = head->B4P(InstanceArray);
    int nBlocks = head->B4P(AvxBlockCount);
    for (int bi = 0; bi < nBlocks; bi++) {
        double von_a[B4N_W];
        BSIM4NeonHot *hot = &hots[bi];
        for (int lane = 0; lane < B4N_W; lane++)
            von_a[lane] = InstArray[hot->instIdx[lane]]->B4P(von);
        hot->von = vn_load(von_a);
        hot->phys_valid = 0;
        hot->stamp_valid = 0;
    }
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

/* Branchless register-direct gather of (vds, vgs, vbs) for all B4N_W lanes
 * (inactive lanes share lane 0's node indices, so zero branches needed!). */
static inline void
B4P(gatherVoltages)(
    const BSIM4NeonHot * restrict hot,
    const double * restrict rhsOld,
    vnd *vds_out, vnd *vgs_out, vnd *vbs_out)
{
#if B4N_W == 2
    double vs0 = rhsOld[hot->sNodePrime[0]];
    double vs1 = rhsOld[hot->sNodePrime[1]];
    float64x2_t vd = { rhsOld[hot->dNodePrime[0]] - vs0, rhsOld[hot->dNodePrime[1]] - vs1 };
    float64x2_t vg = { rhsOld[hot->gNodePrime[0]] - vs0, rhsOld[hot->gNodePrime[1]] - vs1 };
    float64x2_t vb = { rhsOld[hot->bNodePrime[0]] - vs0, rhsOld[hot->bNodePrime[1]] - vs1 };
    *vds_out = vd * hot->type;
    *vgs_out = vg * hot->type;
    *vbs_out = vb * hot->type;
#else
    double a_vds[B4N_W], a_vgs[B4N_W], a_vbs[B4N_W];
    for (int lane = 0; lane < B4N_W; lane++) {
        double vs = rhsOld[hot->sNodePrime[lane]];
        a_vds[lane] = rhsOld[hot->dNodePrime[lane]] - vs;
        a_vgs[lane] = rhsOld[hot->gNodePrime[lane]] - vs;
        a_vbs[lane] = rhsOld[hot->bNodePrime[lane]] - vs;
    }
    *vds_out = vn_load(a_vds) * hot->type;
    *vgs_out = vn_load(a_vgs) * hot->type;
    *vbs_out = vn_load(a_vbs) * hot->type;
#endif
}

static inline int
B4P(blockCacheMatch)(
    const BSIM4NeonHot * restrict h,
    vnd vds, vnd vgs, vnd vbs,
    double vtol, double reltol, double abstol,
    int *exact_zero_out)
{
    vnd delvds = vds - h->last_vds;
    vnd delvgs = vgs - h->last_vgs;
    vnd delvbs = vbs - h->last_vbs;
    vnd dv = vn_max(vn_abs(delvds), vn_max(vn_abs(delvgs), vn_abs(delvbs)));
    if (vn_any(vn_gt(dv, vn_set1(vtol)) & h->active_mask))
        return 0;
    vnm m_flip = (vn_ge(vds, vn_set1(0.0)) ^ vn_ge(h->last_vds, vn_set1(0.0))) & h->active_mask;
    if (vn_any(m_flip))
        return 0;
    const vnd v_rel = vn_set1(0.1 * reltol);
    const vnd v_abs = vn_set1(0.1 * abstol);
    vnd cd_diff = h->cd_g * delvgs + h->cd_d * delvds + h->cd_b * delvbs;
    vnd cdhat   = h->Idtot + cd_diff;
    vnd tol0    = v_rel * vn_max(vn_abs(cdhat), vn_abs(h->Idtot)) + v_abs;
    vnd cb_diff = h->cb_g * delvgs + h->cb_d * delvds + h->cb_b * delvbs;
    vnd cbhat   = h->Ibtot + cb_diff;
    vnd tol6    = v_rel * vn_max(vn_abs(cbhat), vn_abs(h->Ibtot)) + v_abs;
    vnm fail = (vn_ge(vn_abs(cd_diff), tol0) | vn_ge(vn_abs(cb_diff), tol6)) & h->active_mask;
    if (vn_any(fail))
        return 0;
    *exact_zero_out = !vn_any(vn_gt(dv, vn_set1(1.0e-13)) & h->active_mask);
    return 1;
}

static inline void
B4P(stampNeonHot)(
    const BSIM4NeonHot * restrict hot,
    B4P(instance) **InstArray,
    double * restrict my_ax,
    double * restrict my_rhs,
    int stamp_mode)
{
    const int nact = hot->n_active;
    if (stamp_mode == 0) {
        for (int lane = 0; lane < nact; lane++) {
            const double * restrict st = hot->st_lane[lane];
            const int * restrict off = hot->kluOff[lane];
            my_rhs[hot->dNodePrime[lane]] += st[0];
            my_rhs[hot->gNodePrime[lane]] -= st[1];
            my_rhs[hot->bNodePrime[lane]] += st[2];
            my_rhs[hot->sNodePrime[lane]] += st[3];

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
        }
        if (hot->m_has_dint) {
            for (int lane = 0; lane < nact; lane++) {
                if (hot->m_has_dint & (1u << lane)) {
                    const int * restrict off = hot->kluOff[lane];
                    double mgd = hot->mgd[lane];
                    my_ax[off[16]] -= mgd;
                    my_ax[off[17]] -= mgd;
                    my_ax[off[18]] += mgd;
                }
            }
        }
        if (hot->m_has_sint) {
            for (int lane = 0; lane < nact; lane++) {
                if (hot->m_has_sint & (1u << lane)) {
                    const int * restrict off = hot->kluOff[lane];
                    double mgs = hot->mgs[lane];
                    my_ax[off[19]] -= mgs;
                    my_ax[off[20]] -= mgs;
                    my_ax[off[21]] += mgs;
                }
            }
        }
    } else if (stamp_mode == 1) {
        for (int lane = 0; lane < nact; lane++) {
            B4P(instance) *here = InstArray[hot->instIdx[lane]];
            const double * restrict st = hot->st_lane[lane];
            uint8_t bit = (uint8_t)(1u << lane);
            my_rhs[hot->dNodePrime[lane]] += st[0];
            my_rhs[hot->gNodePrime[lane]] -= st[1];
            my_rhs[hot->bNodePrime[lane]] += st[2];
            my_rhs[hot->sNodePrime[lane]] += st[3];

            *(here->B4P(GPgpPtr)) += st[4];
            *(here->B4P(GPdpPtr)) += st[5];
            *(here->B4P(GPspPtr)) += st[6];
            *(here->B4P(GPbpPtr)) += st[7];
            *(here->B4P(DPdpPtr)) += st[8];
            *(here->B4P(DPgpPtr)) += st[9];
            *(here->B4P(DPspPtr)) -= st[10];
            *(here->B4P(DPbpPtr)) -= st[11];
            if (hot->m_has_dint & bit) {
                double mgd = hot->mgd[lane];
                *(here->B4P(DPdPtr)) -= mgd;
                *(here->B4P(DdpPtr)) -= mgd;
                *(here->B4P(DdPtr))  += mgd;
            }
            *(here->B4P(SPdpPtr)) -= st[12];
            *(here->B4P(SPgpPtr)) += st[13];
            *(here->B4P(SPspPtr)) += st[14];
            *(here->B4P(SPbpPtr)) -= st[15];
            if (hot->m_has_sint & bit) {
                double mgs = hot->mgs[lane];
                *(here->B4P(SPsPtr)) -= mgs;
                *(here->B4P(SspPtr)) -= mgs;
                *(here->B4P(SsPtr))  += mgs;
            }
            *(here->B4P(BPdpPtr)) += st[16];
            *(here->B4P(BPgpPtr)) += st[17];
            *(here->B4P(BPspPtr)) += st[18];
            *(here->B4P(BPbpPtr)) += st[19];
        }
    } else {
        for (int lane = 0; lane < nact; lane++) {
            B4P(instance) *here = InstArray[hot->instIdx[lane]];
            const double * restrict st = hot->st_lane[lane];
            uint8_t bit = (uint8_t)(1u << lane);
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
            if (hot->m_has_dint & bit) {
                double mgd = hot->mgd[lane];
                here->B4P(_45) = mgd;
                here->B4P(_49) = mgd;
                here->B4P(_50) = mgd;
            }
            here->B4P(_51) = st[12];
            here->B4P(_52) = st[13];
            here->B4P(_53) = st[14];
            here->B4P(_55) = st[15];
            if (hot->m_has_sint & bit) {
                double mgs = hot->mgs[lane];
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
}

static inline void
B4P(evalNeonBlock)(
    const BSIM4NeonConst * restrict blk,
    BSIM4NeonHot * restrict hot,
    BSIM4NeonL2 * restrict l2,
    B4P(instance) **InstArray,
    CKTcircuit *ckt,
    double * restrict my_ax,
    double * restrict my_rhs,
    int stamp_mode)
{
    const double * restrict rhsOld = ckt->CKTrhsOld;
    double * restrict s0 = ckt->CKTstate0;
    const double * restrict s1 = ckt->CKTstate1;
    const double * restrict s2 = ckt->CKTstate2;
    int is_pred = (ckt->CKTmode & MODEINITPRED) != 0;
    int is_gear2 = (ckt->CKTintegrateMethod == GEAR && ckt->CKTorder == 2);
    int is_trap2 = (ckt->CKTintegrateMethod == TRAPEZOIDAL && ckt->CKTorder == 2);
    int int_mode = is_gear2 ? 2 : (is_trap2 ? 1 : 0);
    int need_aos = B4P(AvxNeedAosSync);
    double ag0 = ckt->CKTag[0], ag1 = ckt->CKTag[1], ag2 = ckt->CKTag[2];
    double vtol = B4P(avxBypassTol)(ckt);
    const int nact = hot->n_active;

    /* Level-1 fast path: NEVER touches BSIM4NeonConst (and skips L2 on exact_zero!) */
    if (!is_pred && !need_aos && vtol >= 0.0 && hot->stamp_valid &&
        hot->last_time == ckt->CKTtime &&
        hot->last_ag0 == ag0 && hot->last_ag1 == ag1 && hot->last_ag2 == ag2 &&
        hot->last_gmin == ckt->CKTgmin && hot->last_int_mode == int_mode &&
        hot->last_s0 == s0 && hot->last_s1 == s1 && hot->last_s2 == s2)
    {
        vnd vds_r, vgs_r, vbs_r;
        B4P(gatherVoltages)(hot, rhsOld, &vds_r, &vgs_r, &vbs_r);
        int exact_zero = 0;
        if (B4P(blockCacheMatch)(hot, vds_r, vgs_r, vbs_r, vtol,
                                 ckt->CKTreltol, ckt->CKTabstol, &exact_zero)) {
            if (!exact_zero) {
                bsim4_neon_l1_sync_state(hot, l2, vds_r, vgs_r, vbs_r,
                                         ag0, ag1, ag2, is_gear2, is_trap2, s0, s1, s2);
            }
            B4P(stampNeonHot)(hot, InstArray, my_ax, my_rhs, stamp_mode);
            return;
        }
    }

    vnd vds, vgs, vbs, s0_vds, s0_vgs, s0_vbs, s0_vbd;
    double a_s0_vds[B4N_W], a_s0_vgs[B4N_W], a_s0_vbs[B4N_W], a_s0_vbd[B4N_W];

#ifndef PREDICTOR
    if (is_pred) {
        double xfact = ckt->CKTdelta / ckt->CKTdeltaOld[1];
        double one_p_xf = 1.0 + xfact;
        double a_vds[B4N_W], a_vgs[B4N_W], a_vbs[B4N_W];
        for (int lane = 0; lane < nact; lane++) {
            int st = hot->stateBase[lane];
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
        for (int lane = nact; lane < B4N_W; lane++) {
            a_vds[lane] = a_vds[0]; a_vgs[lane] = a_vgs[0]; a_vbs[lane] = a_vbs[0];
            a_s0_vbd[lane] = a_s0_vbd[0]; a_s0_vbs[lane] = a_s0_vbs[0];
            a_s0_vgs[lane] = a_s0_vgs[0]; a_s0_vds[lane] = a_s0_vds[0];
        }
        vds = vn_load(a_vds); vgs = vn_load(a_vgs); vbs = vn_load(a_vbs);
    } else
#endif
    {
        B4P(gatherVoltages)(hot, rhsOld, &vds, &vgs, &vbs);
        for (int lane = 0; lane < B4N_W; lane++) {
            int st = hot->stateBase[lane];
            a_s0_vbd[lane] = s0[st + 0];
            a_s0_vbs[lane] = s0[st + 1];
            a_s0_vgs[lane] = s0[st + 2];
            a_s0_vds[lane] = s0[st + 3];
        }
    }

    s0_vds = vn_load(a_s0_vds);
    s0_vgs = vn_load(a_s0_vgs);
    s0_vbs = vn_load(a_s0_vbs);
    s0_vbd = vn_load(a_s0_vbd);
    vnd vgdo   = s0_vgs - s0_vds;
    vnd vbd    = vbs - vds;
    vnd vgd    = vgs - vds;
    vnd von    = hot->von;

    /* NEON DEVfetlim + DEVlimvds */
    {
        vnm m_s0_fwd = vn_ge(s0_vds, vn_set1(0.0));
        vnd vgs_f = vn_fetlim(vgs, s0_vgs, von);
        vnd vds_f = vn_limvds(vgs_f - vgd, s0_vds);

        vnd vgd_r = vn_fetlim(vgd, vgdo, von);
        vnd vds_r = -vn_limvds(-(vgs - vgd_r), -s0_vds);
        vnd vgs_r = vgd_r + vds_r;

        vgs = vn_sel(m_s0_fwd, vgs_f, vgs_r);
        vds = vn_sel(m_s0_fwd, vds_f, vds_r);
    }

    /* NEON DEVpnjlim */
    vnm chk_mask = vn_mask_zero();
    int any_chk = 0;
    {
        vnm m_vds_fwd = vn_ge(vds, vn_set1(0.0));
        vnd vj_in  = vn_sel(m_vds_fwd, vbs, vbd);
        vnd vj_old = vn_sel(m_vds_fwd, s0_vbs, s0_vbd);
        vnd vj_lim = vn_pnjlim(vj_in, vj_old, CONSTvt0, hot->vcrit_v, &chk_mask);
        vbs = vn_sel(m_vds_fwd, vj_lim, vj_lim + vds);
        chk_mask = chk_mask & hot->active_mask;
        any_chk = vn_any(chk_mask);
        if (any_chk) {
            int noncon_add = vn_popcount(chk_mask);
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

    /* Level-2 fast path: NEVER touches BSIM4NeonConst! */
    if (!need_aos && !any_chk && vtol >= 0.0 && hot->phys_valid &&
        hot->last_gmin == ckt->CKTgmin)
    {
        int exact_zero = 0;
        if (B4P(blockCacheMatch)(hot, vds, vgs, vbs, vtol,
                                 ckt->CKTreltol, ckt->CKTabstol, &exact_zero)) {
            bsim4_neon_integrate_and_assemble(hot, l2, vds, vgs, vbs, !exact_zero,
                                              ag0, ag1, ag2, is_gear2, is_trap2, s0, s1, s2);
            hot->last_time = ckt->CKTtime;
            hot->last_ag0 = ag0; hot->last_ag1 = ag1; hot->last_ag2 = ag2;
            hot->last_int_mode = int_mode;
            hot->last_s0 = s0; hot->last_s1 = s1; hot->last_s2 = s2;
            hot->stamp_valid = 1;
            B4P(stampNeonHot)(hot, InstArray, my_ax, my_rhs, stamp_mode);
            return;
        }
    }

    BSIM4NeonOut out;
    bsim4_neon_eval_block(blk, hot, l2, vds, vgs, vbs,
                          ckt->CKTgmin, ag0, ag1, ag2,
                          is_gear2, is_trap2, s0, s1, s2, &out, need_aos);

    hot->last_time = ckt->CKTtime;
    hot->last_ag0 = ag0; hot->last_ag1 = ag1; hot->last_ag2 = ag2;
    hot->last_gmin = ckt->CKTgmin;
    hot->last_int_mode = int_mode;
    hot->last_s0 = s0; hot->last_s1 = s1; hot->last_s2 = s2;
    hot->phys_valid = 1;
    hot->stamp_valid = !need_aos;

    if (need_aos) {
        for (int lane = 0; lane < nact; lane++) {
            B4P(instance) *here = InstArray[hot->instIdx[lane]];
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

    B4P(stampNeonHot)(hot, InstArray, my_ax, my_rhs, stamp_mode);
}

/*
 * NEON convergence test over BSIM4NeonHot ONLY (zero Const / L2 / AoS chase!).
 */
int
B4P(convTestAvx)(GENmodel *inModel, CKTcircuit *ckt)
{
    B4P(model) *head = (B4P(model) *)inModel;
    if (!head->B4P(AvxBuilt) || !head->B4P(AvxDynValid) || !B4P(isAvxEligibleTran)(ckt))
        return -1;
    if (head->B4P(AvxFallbackCount) != 0)
        return -1;

    const BSIM4NeonHot *hots = (const BSIM4NeonHot *)head->B4P(AvxDyn);
    int nBlocks = head->B4P(AvxBlockCount);
    const double * restrict rhsOld = ckt->CKTrhsOld;
    const vnd v_reltol = vn_set1(ckt->CKTreltol);
    const vnd v_abstol = vn_set1(ckt->CKTabstol);

    int noncon = 0;
#ifdef USE_OMP
    if (nBlocks > 256 && omp_get_max_threads() > 1) {
        #pragma omp parallel for reduction(|:noncon)
        for (int bi = 0; bi < nBlocks; bi++) {
            if (noncon) continue;
            const BSIM4NeonHot *h = &hots[bi];
            vnd vds, vgs, vbs;
            B4P(gatherVoltages)(h, rhsOld, &vds, &vgs, &vbs);
            vnd delvds = vds - h->last_vds;
            vnd delvgs = vgs - h->last_vgs;
            vnd delvbs = vbs - h->last_vbs;

            vnd cd_diff = h->cd_g * delvgs + h->cd_d * delvds + h->cd_b * delvbs;
            vnd cdhat = h->Idtot + cd_diff;
            vnd tol0 = v_reltol * vn_max(vn_abs(cdhat), vn_abs(h->Idtot)) + v_abstol;

            vnd cb_diff = h->cb_g * delvgs + h->cb_d * delvds + h->cb_b * delvbs;
            vnd cbhat = h->Ibtot + cb_diff;
            vnd tol6 = v_reltol * vn_max(vn_abs(cbhat), vn_abs(h->Ibtot)) + v_abstol;

            vnm fail = (vn_ge(vn_abs(cd_diff), tol0) | vn_gt(vn_abs(cb_diff), tol6)) & h->active_mask;
            if (vn_any(fail))
                noncon = 1;
        }
        return noncon;
    }
#endif
    for (int bi = 0; bi < nBlocks; bi++) {
        const BSIM4NeonHot *h = &hots[bi];
        vnd vds, vgs, vbs;
        B4P(gatherVoltages)(h, rhsOld, &vds, &vgs, &vbs);
        vnd delvds = vds - h->last_vds;
        vnd delvgs = vgs - h->last_vgs;
        vnd delvbs = vbs - h->last_vbs;

        vnd cd_diff = h->cd_g * delvgs + h->cd_d * delvds + h->cd_b * delvbs;
        vnd cdhat = h->Idtot + cd_diff;
        vnd tol0 = v_reltol * vn_max(vn_abs(cdhat), vn_abs(h->Idtot)) + v_abstol;

        vnd cb_diff = h->cb_g * delvgs + h->cb_d * delvds + h->cb_b * delvbs;
        vnd cbhat = h->Ibtot + cb_diff;
        vnd tol6 = v_reltol * vn_max(vn_abs(cbhat), vn_abs(h->Ibtot)) + v_abstol;

        vnm fail = (vn_ge(vn_abs(cd_diff), tol0) | vn_gt(vn_abs(cb_diff), tol6)) & h->active_mask;
        if (vn_any(fail))
            return 1;
    }
    return 0;
}

/*
 * NEON truncation error over BSIM4NeonHot ONLY (zero Const / L2 / AoS chase!).
 */
static inline double
B4P(terrBlockNeon)(
    const BSIM4NeonHot * restrict h,
    const double * restrict s0,
    const double * restrict s1,
    const double * restrict s2,
    const double * restrict s3,
    int ord,
    vnd v_abstol, vnd v_reltol, vnd v_relInvD, vnd v_chgtol,
    vnd v_trtol, vnd v_tfac,
    vnd v_w0, vnd v_w1, vnd v_w2, vnd v_w3,
    double cur)
{
    const int nact = h->n_active;
    double aq0[3][B4N_W], aq1[3][B4N_W], aq2[3][B4N_W], aq3[3][B4N_W], ac0[3][B4N_W], ac1[3][B4N_W];

    if (ord == 2) {
        for (int lane = 0; lane < B4N_W; lane++) {
            int st = h->stateBase[lane];
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
        for (int c = 0; c < 3; c++) {
            vnd q0 = vn_load(aq0[c]), q1 = vn_load(aq1[c]);
            vnd c0 = vn_abs(vn_load(ac0[c])), c1 = vn_abs(vn_load(ac1[c]));
            vnd volttol = v_abstol + v_reltol * vn_max(c0, c1);
            vnd qmax = vn_max(vn_abs(q0), vn_abs(q1));
            vnd chargetol = v_relInvD * vn_max(qmax, v_chgtol);
            vnd tol = vn_max(volttol, chargetol);
            vnd d0 = v_w0 * q0 + v_w1 * q1 + v_w2 * vn_load(aq2[c]) + v_w3 * vn_load(aq3[c]);
            vnd den = vn_max(v_abstol, v_tfac * vn_abs(d0));
            vnd num = v_trtol * tol;
            if (cur >= 1e150 || vn_any(vn_lt(num, vn_set1(cur * cur) * den) & h->active_mask)) {
                double del[B4N_W];
                vn_store(del, vn_sqrt(num / den));
                for (int lane = 0; lane < nact; lane++) {
                    if (del[lane] < cur)
                        cur = del[lane];
                }
            }
        }
    } else {
        for (int lane = 0; lane < B4N_W; lane++) {
            int st = h->stateBase[lane];
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
        for (int c = 0; c < 3; c++) {
            vnd q0 = vn_load(aq0[c]), q1 = vn_load(aq1[c]);
            vnd c0 = vn_abs(vn_load(ac0[c])), c1 = vn_abs(vn_load(ac1[c]));
            vnd volttol = v_abstol + v_reltol * vn_max(c0, c1);
            vnd qmax = vn_max(vn_abs(q0), vn_abs(q1));
            vnd chargetol = v_relInvD * vn_max(qmax, v_chgtol);
            vnd tol = vn_max(volttol, chargetol);
            vnd d0 = v_w0 * q0 + v_w1 * q1 + v_w2 * vn_load(aq2[c]);
            vnd den = vn_max(v_abstol, v_tfac * vn_abs(d0));
            vnd num = v_trtol * tol;
            if (cur >= 1e150 || vn_any(vn_lt(num, vn_set1(cur) * den) & h->active_mask)) {
                double del[B4N_W];
                vn_store(del, num / den);
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
    if (!head->B4P(AvxBuilt) || !ckt->CKTterrCacheValid || (ord != 1 && ord != 2))
        return -1;

    const BSIM4NeonHot *hots = (const BSIM4NeonHot *)head->B4P(AvxDyn);
    int nBlocks = head->B4P(AvxBlockCount);
    const double * restrict s0 = ckt->CKTstate0;
    const double * restrict s1 = ckt->CKTstate1;
    const double * restrict s2 = ckt->CKTstates[2];
    const double * restrict s3 = (ord == 2) ? ckt->CKTstates[3] : NULL;
    const double *w = ckt->CKTterrWeights;

    vnd v_abstol  = vn_set1(ckt->CKTabstol);
    vnd v_reltol  = vn_set1(ckt->CKTreltol);
    vnd v_relInvD = vn_set1(ckt->CKTterrReltolInvDelta);
    vnd v_chgtol  = vn_set1(ckt->CKTchgtol);
    vnd v_trtol   = vn_set1(ckt->CKTtrtol);
    vnd v_tfac    = vn_set1(ckt->CKTterrFactor);
    vnd v_w0      = vn_set1(w[0]);
    vnd v_w1      = vn_set1(w[1]);
    vnd v_w2      = vn_set1(w[2]);
    vnd v_w3      = vn_set1(ord == 2 ? w[3] : 0.0);

    double minStep = *timeStep;
#ifdef USE_OMP
    if (nBlocks > 128 && omp_get_max_threads() > 1) {
        #pragma omp parallel for reduction(min:minStep)
        for (int bi = 0; bi < nBlocks; bi++) {
            minStep = B4P(terrBlockNeon)(&hots[bi], s0, s1, s2, s3, ord,
                                         v_abstol, v_reltol, v_relInvD, v_chgtol,
                                         v_trtol, v_tfac, v_w0, v_w1, v_w2, v_w3, minStep);
        }
    } else
#endif
    {
        for (int bi = 0; bi < nBlocks; bi++) {
            minStep = B4P(terrBlockNeon)(&hots[bi], s0, s1, s2, s3, ord,
                                         v_abstol, v_reltol, v_relInvD, v_chgtol,
                                         v_trtol, v_tfac, v_w0, v_w1, v_w2, v_w3, minStep);
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

#endif /* NGSPICE_OPT_APPLE */
