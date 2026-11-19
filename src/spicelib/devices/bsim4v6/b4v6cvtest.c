/**** BSIM4.6.2 Released by Wenwei Yang 04/05/2008 ****/

/**********
 * Copyright 2006 Regents of the University of California. All rights reserved.
 * File: b4cvtest.c of BSIM4.6.2.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Authors: 2006- Mohan Dunga, Ali Niknejad, Chenming Hu
 * Authors: 2007- Mohan Dunga, Wenwei Yang, Ali Niknejad, Chenming Hu
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 05/09/2003.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v6def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#ifdef USE_OMP
#include <omp.h>
#endif


static int
BSIM4v6convTestInst(BSIM4v6instance *here, CKTcircuit *ckt)
{
    BSIM4v6model *model = BSIM4v6modPtr(here);
    double delvbd, delvbs, delvds, delvgd, delvgs;
    double delvdbd, delvsbs;
    double delvbd_jct, delvbs_jct;
    double vds, vgs, vgd, vgdo, vbs, vbd;
    double vdbd, vdbs, vsbs;
    double cbhat, cdhat, Idtot, Ibtot;
    double tol0, tol6;
    double mtype = (double) model->BSIM4v6type;
    const double *rhsOld = ckt->CKTrhsOld;
    const double *s0 = ckt->CKTstate0;

    if ((here->BSIM4v6off != 0) && (ckt->CKTmode & MODEINITFIX))
        return 0;

    double vs_p = rhsOld[here->BSIM4v6sNodePrime];
    vds = mtype * (rhsOld[here->BSIM4v6dNodePrime] - vs_p);
    vgs = mtype * (rhsOld[here->BSIM4v6gNodePrime] - vs_p);
    vbs = mtype * (rhsOld[here->BSIM4v6bNodePrime] - vs_p);

    double s0_vgs = s0[here->BSIM4v6vgs];
    double s0_vds = s0[here->BSIM4v6vds];
    vgdo = s0_vgs - s0_vds;
    vbd = vbs - vds;
    vgd = vgs - vds;

    delvbd = vbd - s0[here->BSIM4v6vbd];
    delvgd = vgd - vgdo;
    delvds = vds - s0_vds;
    delvgs = vgs - s0_vgs;
    delvbs = vbs - s0[here->BSIM4v6vbs];

    if (!here->BSIM4v6rbodyMod) {
        delvbd_jct = delvbd;
        delvbs_jct = delvbs;
    } else {
        vdbs = mtype * (rhsOld[here->BSIM4v6dbNode] - vs_p);
        vsbs = mtype * (rhsOld[here->BSIM4v6sbNode] - vs_p);
        vdbd = vdbs - vds;
        delvdbd = vdbd - s0[here->BSIM4v6vdbd];
        delvsbs = vsbs - s0[here->BSIM4v6vsbs];
        delvbd_jct = delvdbd;
        delvbs_jct = delvsbs;
    }

    if (here->BSIM4v6mode >= 0) {
        Idtot = here->BSIM4v6cd + here->BSIM4v6csub - here->BSIM4v6cbd + here->BSIM4v6Igidl;
        cdhat = Idtot - here->BSIM4v6gbd * delvbd_jct
              + (here->BSIM4v6gmbs + here->BSIM4v6gbbs + here->BSIM4v6ggidlb) * delvbs
              + (here->BSIM4v6gm + here->BSIM4v6gbgs + here->BSIM4v6ggidlg) * delvgs
              + (here->BSIM4v6gds + here->BSIM4v6gbds + here->BSIM4v6ggidld) * delvds;
    } else {
        Idtot = here->BSIM4v6cd + here->BSIM4v6cbd - here->BSIM4v6Igidl;
        cdhat = Idtot + here->BSIM4v6gbd * delvbd_jct + here->BSIM4v6gmbs * delvbd
              + here->BSIM4v6gm * delvgd
              - (here->BSIM4v6gds + here->BSIM4v6ggidls) * delvds
              - here->BSIM4v6ggidlg * delvgs - here->BSIM4v6ggidlb * delvbs;
    }
    tol0 = ckt->CKTreltol * MAX(fabs(cdhat), fabs(Idtot)) + ckt->CKTabstol;
    if (fabs(cdhat - Idtot) >= tol0)
        return 1;

    if (model->BSIM4v6rdsMod) {
        double vses = mtype * (rhsOld[here->BSIM4v6sNode] - vs_p);
        double vdes = mtype * (rhsOld[here->BSIM4v6dNode] - vs_p);
        double delvses = vses - s0[here->BSIM4v6vses];
        double vdedo = s0[here->BSIM4v6vdes] - s0_vds;
        double delvded = vdes - vds - vdedo;
        double Isestot = here->BSIM4v6gstot * s0[here->BSIM4v6vses];
        double cseshat = Isestot + here->BSIM4v6gstot * delvses
                       + here->BSIM4v6gstotd * delvds + here->BSIM4v6gstotg * delvgs
                       + here->BSIM4v6gstotb * delvbs;
        double Idedtot = here->BSIM4v6gdtot * vdedo;
        double cdedhat = Idedtot + here->BSIM4v6gdtot * delvded
                       + here->BSIM4v6gdtotd * delvds + here->BSIM4v6gdtotg * delvgs
                       + here->BSIM4v6gdtotb * delvbs;
        double tol1 = ckt->CKTreltol * MAX(fabs(cseshat), fabs(Isestot)) + ckt->CKTabstol;
        double tol2 = ckt->CKTreltol * MAX(fabs(cdedhat), fabs(Idedtot)) + ckt->CKTabstol;
        if ((fabs(cseshat - Isestot) >= tol1) || (fabs(cdedhat - Idedtot) >= tol2))
            return 1;
    }

    if (model->BSIM4v6igcMod) {
        double Igstot, cgshat, Igdtot, cgdhat;
        if (here->BSIM4v6mode >= 0) {
            Igstot = here->BSIM4v6Igs + here->BSIM4v6Igcs;
            cgshat = Igstot + (here->BSIM4v6gIgsg + here->BSIM4v6gIgcsg) * delvgs
                   + here->BSIM4v6gIgcsd * delvds + here->BSIM4v6gIgcsb * delvbs;
            Igdtot = here->BSIM4v6Igd + here->BSIM4v6Igcd;
            cgdhat = Igdtot + here->BSIM4v6gIgdg * delvgd + here->BSIM4v6gIgcdg * delvgs
                   + here->BSIM4v6gIgcdd * delvds + here->BSIM4v6gIgcdb * delvbs;
        } else {
            Igstot = here->BSIM4v6Igs + here->BSIM4v6Igcd;
            cgshat = Igstot + here->BSIM4v6gIgsg * delvgs + here->BSIM4v6gIgcdg * delvgd
                   - here->BSIM4v6gIgcdd * delvds + here->BSIM4v6gIgcdb * delvbd;
            Igdtot = here->BSIM4v6Igd + here->BSIM4v6Igcs;
            cgdhat = Igdtot + (here->BSIM4v6gIgdg + here->BSIM4v6gIgcsg) * delvgd
                   - here->BSIM4v6gIgcsd * delvds + here->BSIM4v6gIgcsb * delvbd;
        }
        double tol3 = ckt->CKTreltol * MAX(fabs(cgshat), fabs(Igstot)) + ckt->CKTabstol;
        double tol4 = ckt->CKTreltol * MAX(fabs(cgdhat), fabs(Igdtot)) + ckt->CKTabstol;
        if ((fabs(cgshat - Igstot) >= tol3) || (fabs(cgdhat - Igdtot) >= tol4))
            return 1;
    }

    if (model->BSIM4v6igbMod) {
        double Igbtot = here->BSIM4v6Igb;
        double cgbhat = (here->BSIM4v6mode >= 0)
            ? (Igbtot + here->BSIM4v6gIgbg * delvgs + here->BSIM4v6gIgbd * delvds + here->BSIM4v6gIgbb * delvbs)
            : (Igbtot + here->BSIM4v6gIgbg * delvgd - here->BSIM4v6gIgbd * delvds + here->BSIM4v6gIgbb * delvbd);
        double tol5 = ckt->CKTreltol * MAX(fabs(cgbhat), fabs(Igbtot)) + ckt->CKTabstol;
        if (fabs(cgbhat - Igbtot) >= tol5)
            return 1;
    }

    Ibtot = here->BSIM4v6cbs + here->BSIM4v6cbd
          - here->BSIM4v6Igidl - here->BSIM4v6Igisl - here->BSIM4v6csub;
    if (here->BSIM4v6mode >= 0) {
        cbhat = Ibtot + here->BSIM4v6gbd * delvbd_jct
              + here->BSIM4v6gbs * delvbs_jct - (here->BSIM4v6gbbs + here->BSIM4v6ggidlb) * delvbs
              - (here->BSIM4v6gbgs + here->BSIM4v6ggidlg) * delvgs
              - (here->BSIM4v6gbds + here->BSIM4v6ggidld) * delvds
              - here->BSIM4v6ggislg * delvgd - here->BSIM4v6ggislb * delvbd + here->BSIM4v6ggisls * delvds;
    } else {
        cbhat = Ibtot + here->BSIM4v6gbs * delvbs_jct + here->BSIM4v6gbd * delvbd_jct
              - (here->BSIM4v6gbbs + here->BSIM4v6ggislb) * delvbd
              - (here->BSIM4v6gbgs + here->BSIM4v6ggislg) * delvgd
              + (here->BSIM4v6gbds + here->BSIM4v6ggisld - here->BSIM4v6ggidls) * delvds
              - here->BSIM4v6ggidlg * delvgs - here->BSIM4v6ggidlb * delvbs;
    }
    tol6 = ckt->CKTreltol * MAX(fabs(cbhat), fabs(Ibtot)) + ckt->CKTabstol;
    if (fabs(cbhat - Ibtot) > tol6)
        return 1;
    return 0;
}

int
BSIM4v6convTest(
GENmodel *inModel,
CKTcircuit *ckt)
{
    BSIM4v6model *model = (BSIM4v6model*)inModel;
#if NGSPICE_OPT_ZEN4
    {
        int avx_rc = BSIM4v6convTestAvx(inModel, ckt);
        if (avx_rc >= 0) {
            if (avx_rc > 0)
                ckt->CKTnoncon++;
            return(OK);
        }
    }
#endif
#ifdef USE_OMP
    int InstCount = model->BSIM4v6InstCount;
    BSIM4v6instance **InstArray = model->BSIM4v6InstanceArray;
    if (InstCount > 0 && InstArray != NULL) {
        int noncon = 0;
        int idx;
        if (InstCount > 256 && omp_get_max_threads() > 1) {
#pragma omp parallel for reduction(|:noncon)
            for (idx = 0; idx < InstCount; idx++) {
                if (!noncon && BSIM4v6convTestInst(InstArray[idx], ckt))
                    noncon = 1;
            }
        } else {
            for (idx = 0; idx < InstCount; idx++) {
                if (BSIM4v6convTestInst(InstArray[idx], ckt)) {
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
        BSIM4v6instance *here;
        for (; model != NULL; model = BSIM4v6nextModel(model)) {
            for (here = BSIM4v6instances(model); here != NULL;
                 here = BSIM4v6nextInstance(here)) {
                if (BSIM4v6convTestInst(here, ckt)) {
                    ckt->CKTnoncon++;
                    return(OK);
                }
            }
        }
    }
    return(OK);
}
