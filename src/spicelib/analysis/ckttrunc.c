/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

    /* CKTtrunc(ckt)
     * this is a driver program to iterate through all the various
     * truncation error functions provided for the circuit elements in the
     * given circuit 
     */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/smpdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"

//#define STEPDEBUG
int
CKTtrunc(CKTcircuit *ckt, double *timeStep)
{
    if (ckt->CKTnewtrunc == 0) {
        int i;
        double timetemp;
#ifdef STEPDEBUG
        double debugtemp;
#endif /* STEPDEBUG */
        double startTime;
        int error = OK;

        startTime = SPfrontEnd->IFseconds();

        /* Precompute divided-difference linear weights once per timepoint for CKTterr */
        {
            static const double gearCoeff[] = {
                .5, .2222222222, .1363636364, .096, .07299270073, .05830903790
            };
            static const double trapCoeff[] = { .5, .08333333333 };
            int ord = ckt->CKTorder;
            int k, m, j;
            double deltmp[8];
            double diff[8];
            if (ord >= 1 && ord <= 6 &&
                (ckt->CKTintegrateMethod == GEAR || ckt->CKTintegrateMethod == TRAPEZOIDAL))
            {
                for (k = 0; k <= ord + 1; k++) {
                    for (m = 0; m <= ord + 1; m++)
                        diff[m] = (m == k) ? 1.0 : 0.0;
                    for (m = 0; m <= ord; m++)
                        deltmp[m] = ckt->CKTdeltaOld[m];
                    j = ord;
                    for (;;) {
                        for (m = 0; m <= j; m++)
                            diff[m] = (diff[m] - diff[m + 1]) / deltmp[m];
                        if (--j < 0) break;
                        for (m = 0; m <= j; m++)
                            deltmp[m] = deltmp[m + 1] + ckt->CKTdeltaOld[m];
                    }
                    ckt->CKTterrWeights[k] = diff[0];
                }
                ckt->CKTterrFactor = (ckt->CKTintegrateMethod == GEAR)
                    ? gearCoeff[ord - 1] : trapCoeff[ord - 1];
                ckt->CKTterrInvDelta = 1.0 / ckt->CKTdelta;
                ckt->CKTterrReltolInvDelta = ckt->CKTreltol * ckt->CKTterrInvDelta;
                ckt->CKTterrCacheValid = 1;
            } else {
                ckt->CKTterrCacheValid = 0;
            }
        }

        timetemp = HUGE;
        for (i = 0; i < DEVmaxnum; i++) {
            if (DEVices[i] && DEVices[i]->DEVtrunc && ckt->CKThead[i]) {
#ifdef STEPDEBUG
                debugtemp = timetemp;
#endif /* STEPDEBUG */
                error = DEVices[i]->DEVtrunc(ckt->CKThead[i], ckt, &timetemp);
                if (error) {
                    ckt->CKTterrCacheValid = 0;
                    ckt->CKTstat->STATtranTruncTime += SPfrontEnd->IFseconds()
                        - startTime;
                    return(error);
                }
#ifdef STEPDEBUG
                if (debugtemp != timetemp) {
                    printf("timestep cut by device type %s from %g to %g\n",
                        DEVices[i]->DEVpublic.name, debugtemp, timetemp);
                }
#endif /* STEPDEBUG */
            }
        }
        ckt->CKTterrCacheValid = 0;
        *timeStep = MIN(2 * *timeStep, timetemp);

        ckt->CKTstat->STATtranTruncTime += SPfrontEnd->IFseconds() - startTime;
        return(OK);
    }
    else {
        int i;
        CKTnode* node;
        double timetemp;
        double tmp;
        double diff;
        double tol;
        double startTime;
        int size;

        startTime = SPfrontEnd->IFseconds();

        timetemp = HUGE;
        size = SMPmatSize(ckt->CKTmatrix);
#ifdef STEPDEBUG
        printf("\nNEWTRUNC at time %g, delta %g\n", ckt->CKTtime, ckt->CKTdeltaOld[0]);
#endif
        node = ckt->CKTnodes;
        switch (ckt->CKTintegrateMethod) {

        case TRAPEZOIDAL:
#ifdef STEPDEBUG
            printf("TRAP, Order is %d\n", ckt->CKTorder);
#endif
            switch (ckt->CKTorder) {
            case 1:
                for (i = 1; i < size; i++) {
                    tol = MAX(fabs(ckt->CKTrhs[i]), fabs(ckt->CKTpred[i])) *
                        ckt->CKTlteReltol + ckt->CKTlteAbstol;
                    node = node->next;
                    if (node->type != SP_VOLTAGE) continue;
                    diff = ckt->CKTrhs[i] - ckt->CKTpred[i];
#ifdef STEPDEBUG
                    printf("%s: cor=%g, pred=%g ", node->name,
                        ckt->CKTrhs[i], ckt->CKTpred[i]);
#endif
                    if (diff != 0) {
                        //                if (!AlmostEqualUlps(diff, 0, 10)) {
                        tmp = ckt->CKTlteTrtol * tol * 2 / diff;
                        tmp = ckt->CKTdeltaOld[0] * fabs(tmp);
                        timetemp = MIN(timetemp, tmp);
#ifdef STEPDEBUG
                        printf("tol = %g, diff = %g, h->%g\n", tol, diff, tmp);
#endif
                    }
                    else {
#ifdef STEPDEBUG
                        printf("diff is 0\n");
#endif
                    }
                }
                break;
            case 2:
                for (i = 1; i < size; i++) {
                    tol = MAX(fabs(ckt->CKTrhs[i]), fabs(ckt->CKTpred[i])) *
                        ckt->CKTlteReltol + ckt->CKTlteAbstol;
                    node = node->next;
                    if (node->type != SP_VOLTAGE) continue;
                    diff = ckt->CKTrhs[i] - ckt->CKTpred[i];
#ifdef STEPDEBUG
                    printf("%s: cor=%g, pred=%g ", node->name,
                        ckt->CKTrhs[i], ckt->CKTpred[i]);
#endif
                    if (diff != 0) {
                        //                if (!AlmostEqualUlps(diff, 0, 10)) {
                        tmp = ckt->CKTlteTrtol * tol * 2 / diff;
                        tmp = ckt->CKTdeltaOld[0] * sqrt(fabs(tmp));
                        timetemp = MIN(timetemp, tmp);
#ifdef STEPDEBUG
                        printf("tol = %g, diff = %g, h->%g\n", tol, diff, tmp);
#endif
                    }
                    else {
#ifdef STEPDEBUG
                        printf("diff is 0\n");
#endif
                    }
                }
                break;
                /*        case 2:
                            for(i=1;i<size;i++) {
                                tol = MAX( fabs(ckt->CKTrhs[i]),fabs(ckt->CKTpred[i]))*
                                        ckt->CKTlteReltol+ckt->CKTlteAbstol;
                                node = node->next;
                                if(node->type!= SP_VOLTAGE) continue;
                                diff = ckt->CKTrhs[i]-ckt->CKTpred[i];
                #ifdef STEPDEBUG
                                printf("%s: cor=%g, pred=%g ",node->name,ckt->CKTrhs[i],
                                        ckt->CKTpred[i]);
                #endif
                                if(diff != 0) {
                //                if(!AlmostEqualUlps(diff, 0, 10)) {
                                    tmp = tol * ckt->CKTlteTrtol * (ckt->CKTdeltaOld[0] + ckt->CKTdeltaOld[1])
                                        / (diff * ckt->CKTdelta);
                                    tmp = fabs(tmp);
                                    tmp = exp(log(tmp) / 3);
                                    tmp *= ckt->CKTdelta;
                                    timetemp = MIN(timetemp,tmp);
                #ifdef STEPDEBUG
                                    printf("tol = %g, diff = %g, h->%g\n",tol,diff,tmp);
                #endif
                                } else {
                #ifdef STEPDEBUG
                                    printf("diff is 0\n");
                #endif
                                }
                            }
                            break; */
            default:
                return(E_ORDER);
                break;

            }
            break;

        case GEAR: {
#ifdef STEPDEBUG
            printf("GEAR, Order is %d\n", ckt->CKTorder);
#endif
            double delsum = 0;
            for (i = 0; i <= ckt->CKTorder; i++) {
                delsum += ckt->CKTdeltaOld[i];
            }
            for (i = 1; i < size; i++) {
                node = node->next;
                if (node->type != SP_VOLTAGE) continue;
                tol = MAX(fabs(ckt->CKTrhs[i]), fabs(ckt->CKTpred[i])) *
                    ckt->CKTlteReltol + ckt->CKTlteAbstol;
                diff = (ckt->CKTrhs[i] - ckt->CKTpred[i]);
#ifdef STEPDEBUG
                printf("%s: cor=%g, pred=%g ", node->name, ckt->CKTrhs[i],
                    ckt->CKTpred[i]);
#endif
                if (diff != 0) {
                    tmp = tol * ckt->CKTlteTrtol * delsum / (diff * ckt->CKTdelta);
                    tmp = fabs(tmp);
                    switch (ckt->CKTorder) {
                    case 0:
                        break;
                    case 1:
                        tmp = sqrt(tmp);
                        break;
                    default:
                        tmp = exp(log(tmp) / (ckt->CKTorder + 1));
                        break;
                    }
                    tmp *= ckt->CKTdelta;
                    timetemp = MIN(timetemp, tmp);
#ifdef STEPDEBUG
                    printf("tol = %g, diff = %g, h->%g\n", tol, diff, tmp);
#endif
                }
                else {
#ifdef STEPDEBUG
                    printf("diff is 0\n");
#endif
                }
            }
        }
                 break;

        default:
            return(E_METHOD);

        }
        *timeStep = MIN(2 * *timeStep, timetemp);
        ckt->CKTstat->STATtranTruncTime += SPfrontEnd->IFseconds() - startTime;
        return(OK);
    }
}
