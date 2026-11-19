/*
 * Per-version glue (BSIM4v5 / BSIM4v6) for 8-wide AVX-512 SoA evaluation.
 * Included inside b4v5ld.c (with B4_IS_V6=0) and b4v6ld.c (with B4_IS_V6=1).
 *
 * Expects macros:
 *   B4P(name) -> BSIM4v5##name  or  BSIM4v6##name
 *   B4_IS_V6  -> 0 or 1
 */
#if defined(__AVX512F__) && defined(__GNUC__)

#include "ngspice/bsim4_avx512_core.h"

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

    int nBlocks = (nElig + 7) / 8;
    BSIM4Avx512Block *blocks = NULL;
    BSIM4Avx512Dyn *dyn = NULL;
    if (nBlocks > 0) {
        if (posix_memalign((void **)&blocks, 64, (size_t)nBlocks * sizeof(BSIM4Avx512Block)) != 0)
            blocks = NULL;
        if (blocks && posix_memalign((void **)&dyn, 64, (size_t)nBlocks * sizeof(BSIM4Avx512Dyn)) != 0) {
            free(blocks);
            blocks = NULL;
            dyn = NULL;
        }
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
    }
    head->B4P(AvxDynValid) = 1;
}

/*
 * Evaluate one 8-wide block with 8-wide AVX-512 voltage limiting and stamp:
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
    int stamp_mode)
{
    const double * restrict rhsOld = ckt->CKTrhsOld;
    double * restrict s0 = ckt->CKTstate0;
    const double * restrict s1 = ckt->CKTstate1;
    const double * restrict s2 = ckt->CKTstate2;
    int is_pred = (ckt->CKTmode & MODEINITPRED) != 0;
    double xfact = is_pred ? (ckt->CKTdelta / ckt->CKTdeltaOld[1]) : 0.0;
    double one_p_xf = 1.0 + xfact;
    int nact = blk->n_active;
    __mmask8 active_mask = (__mmask8)((1u << nact) - 1u);

    double a_vds[8], a_vgs[8], a_vbs[8];
    double a_s0_vds[8], a_s0_vgs[8], a_s0_vbs[8], a_s0_vbd[8];

#ifndef PREDICTOR
    if (is_pred) {
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

    v8d vds = v8_load(a_vds);
    v8d vgs = v8_load(a_vgs);
    v8d vbs = v8_load(a_vbs);
    if (!is_pred) {
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
        __mmask8 chk_mask = 0;
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

    int is_gear2 = (ckt->CKTintegrateMethod == GEAR && ckt->CKTorder == 2);
    int is_trap2 = (ckt->CKTintegrateMethod == TRAPEZOIDAL && ckt->CKTorder == 2);
    int need_aos = B4P(AvxNeedAosSync);

    BSIM4Avx512Out out;
    bsim4_avx512_eval_block(blk, dyn, vds, vgs, vbs,
                            ckt->CKTgmin, ckt->CKTag[0], ckt->CKTag[1], ckt->CKTag[2],
                            is_gear2, is_trap2, s0, s1, s2, &out, need_aos);

    double mgd_arr[8], mgs_arr[8];
    if (blk->m_has_dint) v8_store(mgd_arr, blk->mgd);
    if (blk->m_has_sint) v8_store(mgs_arr, blk->mgs);

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

    if (stamp_mode == 0) {
        for (int lane = 0; lane < nact; lane++) {
            const int *off = blk->kluOff[lane];
            my_rhs[blk->dNodePrime[lane]] += out.rhsd[lane];
            my_rhs[blk->gNodePrime[lane]] -= out.rhsg[lane];
            my_rhs[blk->bNodePrime[lane]] += out.rhsb[lane];
            my_rhs[blk->sNodePrime[lane]] += out.rhss[lane];

            my_ax[off[0]]  += out.v34[lane];
            my_ax[off[1]]  += out.v35[lane];
            my_ax[off[2]]  += out.v36[lane];
            my_ax[off[3]]  += out.v37[lane];
            my_ax[off[4]]  += out.v44[lane];
            my_ax[off[5]]  += out.v46[lane];
            my_ax[off[6]]  -= out.v47[lane];
            my_ax[off[7]]  -= out.v48[lane];
            my_ax[off[8]]  -= out.v51[lane];
            my_ax[off[9]]  += out.v52[lane];
            my_ax[off[10]] += out.v53[lane];
            my_ax[off[11]] -= out.v55[lane];
            my_ax[off[12]] += out.v58[lane];
            my_ax[off[13]] += out.v59[lane];
            my_ax[off[14]] += out.v60[lane];
            my_ax[off[15]] += out.v61[lane];
        }
        if (blk->m_has_dint) {
            for (int lane = 0; lane < nact; lane++) {
                if (blk->m_has_dint & (1u << lane)) {
                    const int *off = blk->kluOff[lane];
                    double mgd = mgd_arr[lane];
                    my_ax[off[16]] -= mgd;
                    my_ax[off[17]] -= mgd;
                    my_ax[off[18]] += mgd;
                }
            }
        }
        if (blk->m_has_sint) {
            for (int lane = 0; lane < nact; lane++) {
                if (blk->m_has_sint & (1u << lane)) {
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
            my_rhs[blk->dNodePrime[lane]] += out.rhsd[lane];
            my_rhs[blk->gNodePrime[lane]] -= out.rhsg[lane];
            my_rhs[blk->bNodePrime[lane]] += out.rhsb[lane];
            my_rhs[blk->sNodePrime[lane]] += out.rhss[lane];

            *(here->B4P(GPgpPtr)) += out.v34[lane];
            *(here->B4P(GPdpPtr)) += out.v35[lane];
            *(here->B4P(GPspPtr)) += out.v36[lane];
            *(here->B4P(GPbpPtr)) += out.v37[lane];
            *(here->B4P(DPdpPtr)) += out.v44[lane];
            *(here->B4P(DPgpPtr)) += out.v46[lane];
            *(here->B4P(DPspPtr)) -= out.v47[lane];
            *(here->B4P(DPbpPtr)) -= out.v48[lane];
            if (blk->m_has_dint & bit) {
                double mgd = mgd_arr[lane];
                *(here->B4P(DPdPtr)) -= mgd;
                *(here->B4P(DdpPtr)) -= mgd;
                *(here->B4P(DdPtr))  += mgd;
            }
            *(here->B4P(SPdpPtr)) -= out.v51[lane];
            *(here->B4P(SPgpPtr)) += out.v52[lane];
            *(here->B4P(SPspPtr)) += out.v53[lane];
            *(here->B4P(SPbpPtr)) -= out.v55[lane];
            if (blk->m_has_sint & bit) {
                double mgs = mgs_arr[lane];
                *(here->B4P(SPsPtr)) -= mgs;
                *(here->B4P(SspPtr)) -= mgs;
                *(here->B4P(SsPtr))  += mgs;
            }
            *(here->B4P(BPdpPtr)) += out.v58[lane];
            *(here->B4P(BPgpPtr)) += out.v59[lane];
            *(here->B4P(BPspPtr)) += out.v60[lane];
            *(here->B4P(BPbpPtr)) += out.v61[lane];
        }
    } else {
        for (int lane = 0; lane < nact; lane++) {
            B4P(instance) *here = InstArray[blk->instIdx[lane]];
            __mmask8 bit = (__mmask8)(1u << lane);
            here->B4P(rhsdPrime) = out.rhsd[lane];
            here->B4P(rhsgPrime) = out.rhsg[lane];
            here->B4P(rhsbPrime) = out.rhsb[lane];
            here->B4P(rhssPrime) = out.rhss[lane];
            here->B4P(_34) = out.v34[lane];
            here->B4P(_35) = out.v35[lane];
            here->B4P(_36) = out.v36[lane];
            here->B4P(_37) = out.v37[lane];
            here->B4P(_44) = out.v44[lane];
            here->B4P(_46) = out.v46[lane];
            here->B4P(_47) = out.v47[lane];
            here->B4P(_48) = out.v48[lane];
            if (blk->m_has_dint & bit) {
                double mgd = mgd_arr[lane];
                here->B4P(_45) = mgd;
                here->B4P(_49) = mgd;
                here->B4P(_50) = mgd;
            }
            here->B4P(_51) = out.v51[lane];
            here->B4P(_52) = out.v52[lane];
            here->B4P(_53) = out.v53[lane];
            here->B4P(_55) = out.v55[lane];
            if (blk->m_has_sint & bit) {
                double mgs = mgs_arr[lane];
                here->B4P(_54) = mgs;
                here->B4P(_56) = mgs;
                here->B4P(_57) = mgs;
            }
            here->B4P(_58) = out.v58[lane];
            here->B4P(_59) = out.v59[lane];
            here->B4P(_60) = out.v60[lane];
            here->B4P(_61) = out.v61[lane];
        }
    }
}

/*
 * 8-wide AVX-512 convergence test over AvxBlocks + AvxDyn (zero AoS/state chase!).
 * Returns: -1 if AVX-512 convTest not applicable, 0 if all AVX blocks converged, 1 if noncon.
 */
int
B4P(convTestAvx)(GENmodel *inModel, CKTcircuit *ckt)
{
    B4P(model) *head = (B4P(model) *)inModel;
    if (!head->B4P(AvxBuilt) || !head->B4P(AvxDynValid) || !B4P(isAvxEligibleTran)(ckt))
        return -1;
    if (head->B4P(AvxFallbackCount) != 0)
        return -1;

    const BSIM4Avx512Block *blocks = (const BSIM4Avx512Block *)head->B4P(AvxBlocks);
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
            const BSIM4Avx512Block *blk = &blocks[bi];
            const BSIM4Avx512Dyn *d = &dyn[bi];
            int nact = blk->n_active;
            double a_vds[8], a_vgs[8], a_vbs[8];
            for (int lane = 0; lane < nact; lane++) {
                double vs = rhsOld[blk->sNodePrime[lane]];
                a_vds[lane] = rhsOld[blk->dNodePrime[lane]] - vs;
                a_vgs[lane] = rhsOld[blk->gNodePrime[lane]] - vs;
                a_vbs[lane] = rhsOld[blk->bNodePrime[lane]] - vs;
            }
            for (int lane = nact; lane < 8; lane++) {
                a_vds[lane] = a_vds[0]; a_vgs[lane] = a_vgs[0]; a_vbs[lane] = a_vbs[0];
            }
            v8d delvds = v8_load(a_vds) * blk->type - d->last_vds;
            v8d delvgs = v8_load(a_vgs) * blk->type - d->last_vgs;
            v8d delvbs = v8_load(a_vbs) * blk->type - d->last_vbs;

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
    for (int bi = 0; bi < nBlocks; bi++) {
        const BSIM4Avx512Block *blk = &blocks[bi];
        const BSIM4Avx512Dyn *d = &dyn[bi];
        int nact = blk->n_active;
        double a_vds[8], a_vgs[8], a_vbs[8];
        for (int lane = 0; lane < nact; lane++) {
            double vs = rhsOld[blk->sNodePrime[lane]];
            a_vds[lane] = rhsOld[blk->dNodePrime[lane]] - vs;
            a_vgs[lane] = rhsOld[blk->gNodePrime[lane]] - vs;
            a_vbs[lane] = rhsOld[blk->bNodePrime[lane]] - vs;
        }
        for (int lane = nact; lane < 8; lane++) {
            a_vds[lane] = a_vds[0]; a_vgs[lane] = a_vgs[0]; a_vbs[lane] = a_vbs[0];
        }
        v8d delvds = v8_load(a_vds) * blk->type - d->last_vds;
        v8d delvgs = v8_load(a_vgs) * blk->type - d->last_vgs;
        v8d delvbs = v8_load(a_vbs) * blk->type - d->last_vbs;

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

#endif /* __AVX512F__ && __GNUC__ */
