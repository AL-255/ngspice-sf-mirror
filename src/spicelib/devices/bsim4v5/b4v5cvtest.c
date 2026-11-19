/**** BSIM4.5.0 Released by Xuemei (Jane) Xi 07/29/2005 ****/

/**********
 * Copyright 2005 Regents of the University of California. All rights reserved.
 * File: b4cvtest.c of BSIM4.5.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 05/09/2003.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v5def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#ifdef USE_OMP
#include <omp.h>
#endif


static int
BSIM4v5convTestInst(BSIM4v5instance *here, CKTcircuit *ckt)
{
    BSIM4v5model *model = BSIM4v5modPtr(here);
    double delvbd, delvbs, delvds, delvgd, delvgs;
    double delvdbd, delvsbs;
    double delvbd_jct, delvbs_jct;
    double vds, vgs, vgd, vgdo, vbs, vbd;
    double vdbd, vdbs, vsbs;
    double cbhat, cdhat, Idtot, Ibtot;
    double tol0, tol6;
    double mtype = (double) model->BSIM4v5type;
    const double *rhsOld = ckt->CKTrhsOld;
    const double *s0 = ckt->CKTstate0;

    if ((here->BSIM4v5off != 0) && (ckt->CKTmode & MODEINITFIX))
        return 0;

    double vs_p = rhsOld[here->BSIM4v5sNodePrime];
    vds = mtype * (rhsOld[here->BSIM4v5dNodePrime] - vs_p);
    vgs = mtype * (rhsOld[here->BSIM4v5gNodePrime] - vs_p);
    vbs = mtype * (rhsOld[here->BSIM4v5bNodePrime] - vs_p);

    double s0_vgs = s0[here->BSIM4v5vgs];
    double s0_vds = s0[here->BSIM4v5vds];
    vgdo = s0_vgs - s0_vds;
    vbd = vbs - vds;
    vgd = vgs - vds;

    delvbd = vbd - s0[here->BSIM4v5vbd];
    delvgd = vgd - vgdo;
    delvds = vds - s0_vds;
    delvgs = vgs - s0_vgs;
    delvbs = vbs - s0[here->BSIM4v5vbs];

    if (!here->BSIM4v5rbodyMod) {
        delvbd_jct = delvbd;
        delvbs_jct = delvbs;
    } else {
        vdbs = mtype * (rhsOld[here->BSIM4v5dbNode] - vs_p);
        vsbs = mtype * (rhsOld[here->BSIM4v5sbNode] - vs_p);
        vdbd = vdbs - vds;
        delvdbd = vdbd - s0[here->BSIM4v5vdbd];
        delvsbs = vsbs - s0[here->BSIM4v5vsbs];
        delvbd_jct = delvdbd;
        delvbs_jct = delvsbs;
    }

    if (here->BSIM4v5mode >= 0) {
        Idtot = here->BSIM4v5cd + here->BSIM4v5csub - here->BSIM4v5cbd + here->BSIM4v5Igidl;
        cdhat = Idtot - here->BSIM4v5gbd * delvbd_jct
              + (here->BSIM4v5gmbs + here->BSIM4v5gbbs + here->BSIM4v5ggidlb) * delvbs
              + (here->BSIM4v5gm + here->BSIM4v5gbgs + here->BSIM4v5ggidlg) * delvgs
              + (here->BSIM4v5gds + here->BSIM4v5gbds + here->BSIM4v5ggidld) * delvds;
    } else {
        Idtot = here->BSIM4v5cd + here->BSIM4v5cbd - here->BSIM4v5Igidl;
        cdhat = Idtot + here->BSIM4v5gbd * delvbd_jct + here->BSIM4v5gmbs * delvbd
              + here->BSIM4v5gm * delvgd
              - (here->BSIM4v5gds + here->BSIM4v5ggidls) * delvds
              - here->BSIM4v5ggidlg * delvgs - here->BSIM4v5ggidlb * delvbs;
    }
    tol0 = ckt->CKTreltol * MAX(fabs(cdhat), fabs(Idtot)) + ckt->CKTabstol;
    if (fabs(cdhat - Idtot) >= tol0)
        return 1;

    if (model->BSIM4v5rdsMod) {
        double vses = mtype * (rhsOld[here->BSIM4v5sNode] - vs_p);
        double vdes = mtype * (rhsOld[here->BSIM4v5dNode] - vs_p);
        double delvses = vses - s0[here->BSIM4v5vses];
        double vdedo = s0[here->BSIM4v5vdes] - s0_vds;
        double delvded = vdes - vds - vdedo;
        double Isestot = here->BSIM4v5gstot * s0[here->BSIM4v5vses];
        double cseshat = Isestot + here->BSIM4v5gstot * delvses
                       + here->BSIM4v5gstotd * delvds + here->BSIM4v5gstotg * delvgs
                       + here->BSIM4v5gstotb * delvbs;
        double Idedtot = here->BSIM4v5gdtot * vdedo;
        double cdedhat = Idedtot + here->BSIM4v5gdtot * delvded
                       + here->BSIM4v5gdtotd * delvds + here->BSIM4v5gdtotg * delvgs
                       + here->BSIM4v5gdtotb * delvbs;
        double tol1 = ckt->CKTreltol * MAX(fabs(cseshat), fabs(Isestot)) + ckt->CKTabstol;
        double tol2 = ckt->CKTreltol * MAX(fabs(cdedhat), fabs(Idedtot)) + ckt->CKTabstol;
        if ((fabs(cseshat - Isestot) >= tol1) || (fabs(cdedhat - Idedtot) >= tol2))
            return 1;
    }

    if (model->BSIM4v5igcMod) {
        double Igstot, cgshat, Igdtot, cgdhat;
        if (here->BSIM4v5mode >= 0) {
            Igstot = here->BSIM4v5Igs + here->BSIM4v5Igcs;
            cgshat = Igstot + (here->BSIM4v5gIgsg + here->BSIM4v5gIgcsg) * delvgs
                   + here->BSIM4v5gIgcsd * delvds + here->BSIM4v5gIgcsb * delvbs;
            Igdtot = here->BSIM4v5Igd + here->BSIM4v5Igcd;
            cgdhat = Igdtot + here->BSIM4v5gIgdg * delvgd + here->BSIM4v5gIgcdg * delvgs
                   + here->BSIM4v5gIgcdd * delvds + here->BSIM4v5gIgcdb * delvbs;
        } else {
            Igstot = here->BSIM4v5Igs + here->BSIM4v5Igcd;
            cgshat = Igstot + here->BSIM4v5gIgsg * delvgs + here->BSIM4v5gIgcdg * delvgd
                   - here->BSIM4v5gIgcdd * delvds + here->BSIM4v5gIgcdb * delvbd;
            Igdtot = here->BSIM4v5Igd + here->BSIM4v5Igcs;
            cgdhat = Igdtot + (here->BSIM4v5gIgdg + here->BSIM4v5gIgcsg) * delvgd
                   - here->BSIM4v5gIgcsd * delvds + here->BSIM4v5gIgcsb * delvbd;
        }
        double tol3 = ckt->CKTreltol * MAX(fabs(cgshat), fabs(Igstot)) + ckt->CKTabstol;
        double tol4 = ckt->CKTreltol * MAX(fabs(cgdhat), fabs(Igdtot)) + ckt->CKTabstol;
        if ((fabs(cgshat - Igstot) >= tol3) || (fabs(cgdhat - Igdtot) >= tol4))
            return 1;
    }

    if (model->BSIM4v5igbMod) {
        double Igbtot = here->BSIM4v5Igb;
        double cgbhat = (here->BSIM4v5mode >= 0)
            ? (Igbtot + here->BSIM4v5gIgbg * delvgs + here->BSIM4v5gIgbd * delvds + here->BSIM4v5gIgbb * delvbs)
            : (Igbtot + here->BSIM4v5gIgbg * delvgd - here->BSIM4v5gIgbd * delvds + here->BSIM4v5gIgbb * delvbd);
        double tol5 = ckt->CKTreltol * MAX(fabs(cgbhat), fabs(Igbtot)) + ckt->CKTabstol;
        if (fabs(cgbhat - Igbtot) >= tol5)
            return 1;
    }

    Ibtot = here->BSIM4v5cbs + here->BSIM4v5cbd
          - here->BSIM4v5Igidl - here->BSIM4v5Igisl - here->BSIM4v5csub;
    if (here->BSIM4v5mode >= 0) {
        cbhat = Ibtot + here->BSIM4v5gbd * delvbd_jct
              + here->BSIM4v5gbs * delvbs_jct - (here->BSIM4v5gbbs + here->BSIM4v5ggidlb) * delvbs
              - (here->BSIM4v5gbgs + here->BSIM4v5ggidlg) * delvgs
              - (here->BSIM4v5gbds + here->BSIM4v5ggidld) * delvds
              - here->BSIM4v5ggislg * delvgd - here->BSIM4v5ggislb * delvbd + here->BSIM4v5ggisls * delvds;
    } else {
        cbhat = Ibtot + here->BSIM4v5gbs * delvbs_jct + here->BSIM4v5gbd * delvbd_jct
              - (here->BSIM4v5gbbs + here->BSIM4v5ggislb) * delvbd
              - (here->BSIM4v5gbgs + here->BSIM4v5ggislg) * delvgd
              + (here->BSIM4v5gbds + here->BSIM4v5ggisld - here->BSIM4v5ggidls) * delvds
              - here->BSIM4v5ggidlg * delvgs - here->BSIM4v5ggidlb * delvbs;
    }
    tol6 = ckt->CKTreltol * MAX(fabs(cbhat), fabs(Ibtot)) + ckt->CKTabstol;
    if (fabs(cbhat - Ibtot) > tol6)
        return 1;
    return 0;
}

int
BSIM4v5convTest(
GENmodel *inModel,
CKTcircuit *ckt)
{
    BSIM4v5model *model = (BSIM4v5model*)inModel;
#if NGSPICE_OPT_BSIM4_SIMD
    {
        int avx_rc = BSIM4v5convTestAvx(inModel, ckt);
        if (avx_rc >= 0) {
            if (avx_rc > 0)
                ckt->CKTnoncon++;
            return(OK);
        }
    }
#endif
#ifdef USE_OMP
    int InstCount = model->BSIM4v5InstCount;
    BSIM4v5instance **InstArray = model->BSIM4v5InstanceArray;
    if (InstCount > 0 && InstArray != NULL) {
        int noncon = 0;
        int idx;
        if (InstCount > 256 && omp_get_max_threads() > 1) {
#pragma omp parallel for reduction(|:noncon)
            for (idx = 0; idx < InstCount; idx++) {
                if (!noncon && BSIM4v5convTestInst(InstArray[idx], ckt))
                    noncon = 1;
            }
        } else {
            for (idx = 0; idx < InstCount; idx++) {
                if (BSIM4v5convTestInst(InstArray[idx], ckt)) {
                    noncon = 1;
                    break;
                }
            }
        }
        if (noncon)
            ckt->CKTnoncon++;
        return(OK);
    }
#endif
    {
        BSIM4v5instance *here;
        for (; model != NULL; model = BSIM4v5nextModel(model)) {
            for (here = BSIM4v5instances(model); here != NULL;
                 here = BSIM4v5nextInstance(here)) {
                if (BSIM4v5convTestInst(here, ckt)) {
                    ckt->CKTnoncon++;
                    return(OK);
                }
            }
        }
    }
    return(OK);
}
