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

#if NGSPICE_OPT_GENERAL
    if (ckt->CKTterrCacheValid && (ord == 1 || ord == 2 || ord == 3) && count > 0) {
        if ((model->CAPmrStateLazy || CAPmrActiveTruncEnabled()) &&
            model->CAPmrHistValid && !model->CAPmrHasIc && model->CAPmrBuilt)
        {
            int N = model->CAPmrN;
            const double * restrict v0 = ckt->CKTrhsOld;
            const double * restrict v1 = model->CAPmrVhist[0];
            const double * restrict v2 = model->CAPmrVhist[1];
            const double * restrict v3 = model->CAPmrVhist[2];
            const double * restrict v4 = model->CAPmrVhist[3];
            const double * restrict dvdt1 = model->CAPmrDvdtPrev;
            double * restrict Dv = model->CAPmrNodeDd;
            const double *w = ckt->CKTterrWeights;
            const double w0 = w[0], w1 = w[1], w2 = w[2];
            const double w3 = (ord >= 2) ? w[3] : 0.0;
            const double w4 = (ord >= 3) ? w[4] : 0.0;
            const double ag0 = ckt->CKTag[0], ag1 = ckt->CKTag[1], ag2 = ckt->CKTag[2], ag3 = ckt->CKTag[3];
            const int is_trap = (ckt->CKTintegrateMethod == TRAPEZOIDAL);
            const int is_trap2_solved = (is_trap && ag1 >= 0.0);
            const double abstol = ckt->CKTabstol;
            const double reltol = ckt->CKTreltol;
            const double relInvD = ckt->CKTterrReltolInvDelta;
            const double chgtol = ckt->CKTchgtol;
            const double trtol = ckt->CKTtrtol;
            const double tfac = ckt->CKTterrFactor;
            double minStep = *timeStep;

            Dv[0] = 0.0;
            if (ord == 3) {
                for (int i = 1; i < N; i++)
                    Dv[i] = w0 * v0[i] + w1 * v1[i] + w2 * v2[i] + w3 * v3[i] + w4 * v4[i];
            } else if (ord == 2) {
                for (int i = 1; i < N; i++)
                    Dv[i] = w0 * v0[i] + w1 * v1[i] + w2 * v2[i] + w3 * v3[i];
            } else {
                for (int i = 1; i < N; i++)
                    Dv[i] = w0 * v0[i] + w1 * v1[i] + w2 * v2[i];
            }

            const double cap_bound = (minStep < 2.0 * ckt->CKTdelta) ? minStep : (2.0 * ckt->CKTdelta);
            const double tol_min = (abstol > relInvD * chgtol) ? abstol : (relInvD * chgtol);
            const double num_min = trtol * tol_min;
            const double cb_pow = (ord == 3) ? (cap_bound * cap_bound * cap_bound)
                                : ((ord == 2) ? (cap_bound * cap_bound) : cap_bound);
            int use_active_prune = CAPmrActiveTruncEnabled() &&
                                   (cap_bound > 0.0 && cap_bound < 1.0) &&
                                   (abstol * cb_pow < num_min);

            if (use_active_prune) {
                if (minStep > cap_bound) minStep = cap_bound;
                unsigned int gen = ++model->CAPmrStampGen;
                if (gen == 0) {
                    memset(model->CAPmrCapStamp, 0, (size_t)count * sizeof(unsigned int));
                    gen = ++model->CAPmrStampGen;
                }
                unsigned int * restrict stamp = model->CAPmrCapStamp;
                const double * restrict cmax = model->CAPmrNodeCmax;
                const int * restrict ncptr = model->CAPmrNodeCapPtr;
                const int * restrict ncidx = model->CAPmrNodeCapIdx;
                double step_pow = (ord == 3) ? (minStep * minStep * minStep)
                                : ((ord == 2) ? (minStep * minStep) : minStep);
                double B = num_min / (2.0 * tfac * step_pow);

                int sub_only = (ckt->CKTmrSubActive && ckt->CKTmrFastNodes && ckt->CKTmrSubNFast > 0);
                int n_loop = sub_only ? ckt->CKTmrSubNFast : (N - 1);
                for (int li = 0; li < n_loop; li++) {
                    int node = sub_only ? ckt->CKTmrFastNodes[li] : (li + 1);
                    if (node <= 0 || node >= N) continue;
                    if (cmax[node] * fabs(Dv[node]) <= B)
                        continue;
                    int p0 = ncptr[node], p1 = ncptr[node + 1];
                    for (int idx = p0; idx < p1; idx++) {
                        int k = ncidx[idx];
                        if (stamp[k] == gen) continue;
                        stamp[k] = gen;
                        int pn = arr[k].posNode;
                        int nn = arr[k].negNode;
                        double C = arr[k].capac;
                        double d0 = C * (Dv[pn] - Dv[nn]);
                        double tf = tfac * fabs(d0);
                        double den = (abstol > tf) ? abstol : tf;
                        double v0d = v0[pn] - v0[nn];
                        double v1d = v1[pn] - v1[nn];
                        double q0 = C * v0d;
                        double q1 = C * v1d;
                        double dvdt0;
                        if (!is_trap) {
                            dvdt0 = ag0 * v0d + ag1 * v1d
                                  + (ag2 != 0.0 ? ag2 * (v2[pn] - v2[nn]) : 0.0)
                                  + (ag3 != 0.0 ? ag3 * (v3[pn] - v3[nn]) : 0.0);
                        } else if (is_trap2_solved) {
                            dvdt0 = -ag1 * (dvdt1[pn] - dvdt1[nn]) + ag0 * (v0d - v1d);
                        } else {
                            dvdt0 = ag0 * v0d + ag1 * v1d;
                        }
                        double c0 = fabs(C * dvdt0);
                        double c1 = fabs(C * (dvdt1[pn] - dvdt1[nn]));
                        double aq0 = fabs(q0), aq1 = fabs(q1);
                        double qmax = (aq0 > aq1) ? aq0 : aq1;
                        double volttol = abstol + reltol * ((c0 > c1) ? c0 : c1);
                        double chargetol = relInvD * ((qmax > chgtol) ? qmax : chgtol);
                        double tol = (volttol > chargetol) ? volttol : chargetol;
                        double num = trtol * tol;
                        if (ord == 3) {
                            if (num < (minStep * minStep * minStep) * den) {
                                double del = cbrt(num / den);
                                if (del < minStep) {
                                    minStep = del;
                                    B = num_min / (2.0 * tfac * minStep * minStep * minStep);
                                }
                            }
                        } else if (ord == 2) {
                            if (num < (minStep * minStep) * den) {
                                double del = sqrt(num / den);
                                if (del < minStep) {
                                    minStep = del;
                                    B = num_min / (2.0 * tfac * minStep * minStep);
                                }
                            }
                        } else {
                            if (num < minStep * den) {
                                double del = num / den;
                                if (del < minStep) {
                                    minStep = del;
                                    B = num_min / (2.0 * tfac * minStep);
                                }
                            }
                        }
                    }
                }
                *timeStep = minStep;
                return(OK);
            }

            /* Full-sweep nodal CAPtrunc (used when NGSPICE_MR_DEFCAP=1 without NGSPICE_MR_TRUNC) */
            for (int k = 0; k < count; k++) {
                int pn = arr[k].posNode;
                int nn = arr[k].negNode;
                double C = arr[k].capac;
                double d0 = C * (Dv[pn] - Dv[nn]);
                double tf = tfac * fabs(d0);
                double den = (abstol > tf) ? abstol : tf;
                double v0d = v0[pn] - v0[nn];
                double v1d = v1[pn] - v1[nn];
                double q0 = C * v0d;
                double q1 = C * v1d;
                double dvdt0;
                if (!is_trap) {
                    dvdt0 = ag0 * v0d + ag1 * v1d
                          + (ag2 != 0.0 ? ag2 * (v2[pn] - v2[nn]) : 0.0)
                          + (ag3 != 0.0 ? ag3 * (v3[pn] - v3[nn]) : 0.0);
                } else if (is_trap2_solved) {
                    dvdt0 = -ag1 * (dvdt1[pn] - dvdt1[nn]) + ag0 * (v0d - v1d);
                } else {
                    dvdt0 = ag0 * v0d + ag1 * v1d;
                }
                double c0 = fabs(C * dvdt0);
                double c1 = fabs(C * (dvdt1[pn] - dvdt1[nn]));
                double aq0 = fabs(q0), aq1 = fabs(q1);
                double qmax = (aq0 > aq1) ? aq0 : aq1;
                double volttol = abstol + reltol * ((c0 > c1) ? c0 : c1);
                double chargetol = relInvD * ((qmax > chgtol) ? qmax : chgtol);
                double tol = (volttol > chargetol) ? volttol : chargetol;
                double num = trtol * tol;
                if (ord == 3) {
                    if (minStep >= 1e150 || num < (minStep * minStep * minStep) * den) {
                        double del = cbrt(num / den);
                        if (del < minStep) minStep = del;
                    }
                } else if (ord == 2) {
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

        const double * restrict s0 = ckt->CKTstate0;
        const double * restrict s1 = ckt->CKTstate1;
        const double * restrict s2 = ckt->CKTstates[2];
        const double * restrict s3 = (ord >= 2) ? ckt->CKTstates[3] : NULL;
        const double * restrict s4 = (ord >= 3) ? ckt->CKTstates[4] : NULL;
        const double *w = ckt->CKTterrWeights;
        const double w0 = w[0], w1 = w[1], w2 = w[2];
        const double w3 = (ord >= 2) ? w[3] : 0.0;
        const double w4 = (ord >= 3) ? w[4] : 0.0;
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
                double d0 = w0 * q0 + w1 * q1 + w2 * s2[qcap]
                          + (ord >= 2 ? w3 * s3[qcap] : 0.0)
                          + (ord >= 3 ? w4 * s4[qcap] : 0.0);
                double tf = tfac * fabs(d0);
                double den = (abstol > tf) ? abstol : tf;
                double num = trtol * tol;
                if (ord == 3) {
                    if (minStep >= 1e150 || num < (minStep * minStep * minStep) * den) {
                        double del = cbrt(num / den);
                        if (del < minStep) minStep = del;
                    }
                } else if (ord == 2) {
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
        if (ord == 3) {
            for (int k = 0; k < count; k++) {
                int qcap = arr[k].qcap;
                double q0 = s0[qcap], q1 = s1[qcap];
                double c0 = fabs(s0[qcap + 1]), c1 = fabs(s1[qcap + 1]);
                double aq0 = fabs(q0), aq1 = fabs(q1);
                double qmax = (aq0 > aq1) ? aq0 : aq1;
                double volttol = abstol + reltol * ((c0 > c1) ? c0 : c1);
                double chargetol = relInvD * ((qmax > chgtol) ? qmax : chgtol);
                double tol = (volttol > chargetol) ? volttol : chargetol;
                double d0 = w0 * q0 + w1 * q1 + w2 * s2[qcap] + w3 * s3[qcap] + w4 * s4[qcap];
                double tf = tfac * fabs(d0);
                double den = (abstol > tf) ? abstol : tf;
                double num = trtol * tol;
                if (minStep >= 1e150 || num < (minStep * minStep * minStep) * den) {
                    double del = cbrt(num / den);
                    if (del < minStep) minStep = del;
                }
            }
        } else if (ord == 2) {
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
#endif

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
