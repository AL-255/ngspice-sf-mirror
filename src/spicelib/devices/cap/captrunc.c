/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: September 2003 Paolo Nenzi
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "capdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#ifdef USE_OMP
#include <omp.h>
#endif


int
CAPtrunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
    CAPmodel *model = (CAPmodel*)inModel;
    CAPensureFastCache(inModel, ckt);
    const CAPfastEntry * restrict arr = model->CAPfastArr;
    int count = model->CAPfastCount;
    int ord = ckt->CKTorder;

    if (ckt->CKTterrCacheValid && (ord == 1 || ord == 2) && count > 0) {
        const double * restrict s0 = ckt->CKTstate0;
        const double * restrict s1 = ckt->CKTstate1;
        const double * restrict s2 = ckt->CKTstates[2];
        const double * restrict s3 = (ord == 2) ? ckt->CKTstates[3] : NULL;
        const double *w = ckt->CKTterrWeights;
        const double w0 = w[0], w1 = w[1], w2 = w[2], w3 = (ord == 2) ? w[3] : 0.0;
        const double abstol = ckt->CKTabstol;
        const double reltol = ckt->CKTreltol;
        const double relInvD = ckt->CKTterrReltolInvDelta;
        const double chgtol = ckt->CKTchgtol;
        const double trtol = ckt->CKTtrtol;
        const double tfac = ckt->CKTterrFactor;
        double minStep = *timeStep;

#ifdef USE_OMP
        if (count >= 128 && omp_get_max_threads() > 1) {
            #pragma omp parallel for schedule(static) reduction(min:minStep)
            for (int k = 0; k < count; k++) {
                int qcap = arr[k].qcap;
                double q0 = s0[qcap], q1 = s1[qcap];
                double c0 = fabs(s0[qcap + 1]), c1 = fabs(s1[qcap + 1]);
                double aq0 = fabs(q0), aq1 = fabs(q1);
                double qmax = (aq0 > aq1) ? aq0 : aq1;
                double volttol = abstol + reltol * ((c0 > c1) ? c0 : c1);
                double chargetol = relInvD * ((qmax > chgtol) ? qmax : chgtol);
                double tol = (volttol > chargetol) ? volttol : chargetol;
                double d0 = w0 * q0 + w1 * q1 + w2 * s2[qcap] + (ord == 2 ? w3 * s3[qcap] : 0.0);
                double tf = tfac * fabs(d0);
                double den = (abstol > tf) ? abstol : tf;
                double num = trtol * tol;
                if (ord == 2) {
                    if (minStep >= 1e150 || num < (minStep * minStep) * den) {
                        double del = sqrt(num / den);
                        if (del < minStep) minStep = del;
                    }
                } else {
                    if (minStep >= 1e150 || num < minStep * den) {
                        double del = num / den;
                        if (del < minStep) minStep = del;
                    }
                }
            }
            *timeStep = minStep;
            return(OK);
        }
#endif
        if (ord == 2) {
            for (int k = 0; k < count; k++) {
                int qcap = arr[k].qcap;
                double q0 = s0[qcap], q1 = s1[qcap];
                double c0 = fabs(s0[qcap + 1]), c1 = fabs(s1[qcap + 1]);
                double aq0 = fabs(q0), aq1 = fabs(q1);
                double qmax = (aq0 > aq1) ? aq0 : aq1;
                double volttol = abstol + reltol * ((c0 > c1) ? c0 : c1);
                double chargetol = relInvD * ((qmax > chgtol) ? qmax : chgtol);
                double tol = (volttol > chargetol) ? volttol : chargetol;
                double d0 = w0 * q0 + w1 * q1 + w2 * s2[qcap] + w3 * s3[qcap];
                double tf = tfac * fabs(d0);
                double den = (abstol > tf) ? abstol : tf;
                double num = trtol * tol;
                if (minStep >= 1e150 || num < (minStep * minStep) * den) {
                    double del = sqrt(num / den);
                    if (del < minStep) minStep = del;
                }
            }
        } else {
            for (int k = 0; k < count; k++) {
                int qcap = arr[k].qcap;
                double q0 = s0[qcap], q1 = s1[qcap];
                double c0 = fabs(s0[qcap + 1]), c1 = fabs(s1[qcap + 1]);
                double aq0 = fabs(q0), aq1 = fabs(q1);
                double qmax = (aq0 > aq1) ? aq0 : aq1;
                double volttol = abstol + reltol * ((c0 > c1) ? c0 : c1);
                double chargetol = relInvD * ((qmax > chgtol) ? qmax : chgtol);
                double tol = (volttol > chargetol) ? volttol : chargetol;
                double d0 = w0 * q0 + w1 * q1 + w2 * s2[qcap];
                double tf = tfac * fabs(d0);
                double den = (abstol > tf) ? abstol : tf;
                double num = trtol * tol;
                if (minStep >= 1e150 || num < minStep * den) {
                    double del = num / den;
                    if (del < minStep) minStep = del;
                }
            }
        }
        *timeStep = minStep;
        return(OK);
    }

#ifdef USE_OMP
    if (count >= 64 && omp_get_max_threads() > 1) {
        double minStep = *timeStep;
        #pragma omp parallel for schedule(static) reduction(min:minStep)
        for (int k = 0; k < count; k++) {
            double localStep = minStep;
            CKTterr(arr[k].qcap, ckt, &localStep);
            if (localStep < minStep)
                minStep = localStep;
        }
        *timeStep = minStep;
        return(OK);
    }
#endif

    for (int k = 0; k < count; k++) {
        CKTterr(arr[k].qcap, ckt, timeStep);
    }
    return(OK);
}
