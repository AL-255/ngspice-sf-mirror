/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: September 2003 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "capdefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include <stdint.h>
#ifdef USE_OMP
#include <omp.h>
#endif

static unsigned long long cap_global_ver = 1;

void
CAPbumpVersion(void)
{
    cap_global_ver++;
}

int
CAPmDelete(GENmodel *inModel)
{
    CAPmodel *model = (CAPmodel *)inModel;
    FREE(model->CAPfastArr);
#if NGSPICE_OPT_GENERAL
    FREE(model->CAPhotArr);
    FREE(model->CAPceqCache);
    model->CAPcacheValid = 0;
#endif
    model->CAPfastCount = 0;
    model->CAPfastVer = 0;
    model->CAPfastKluAxBase = NULL;
    model->CAPfastKluNz = 0;
    cap_global_ver++;
    return OK;
}

static inline int
cap_ptr_to_klu_off(const double *ptr, uintptr_t ax_addr, size_t ax_bytes, int nz)
{
    uintptr_t d = (uintptr_t)ptr - ax_addr;
    if (d < ax_bytes)
        return (int)(d / sizeof(double));
    return nz;
}

void
CAPensureFastCache(GENmodel *inModel, CKTcircuit *ckt)
{
    CAPmodel *head = (CAPmodel *)inModel;
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
    if (head->CAPfastVer == cap_global_ver &&
        head->CAPfastKluAxBase == ax &&
        head->CAPfastKluNz == nz)
        return;

    CAPmodel *m;
    CAPinstance *here;
    int count = 0;
    for (m = head; m != NULL; m = CAPnextModel(m)) {
        for (here = CAPinstances(m); here != NULL; here = CAPnextInstance(here))
            count++;
    }
    FREE(head->CAPfastArr);
#if NGSPICE_OPT_GENERAL
    FREE(head->CAPhotArr);
    FREE(head->CAPceqCache);
    head->CAPcacheValid = 0;
#endif
    head->CAPfastCount = count;
    if (count > 0) {
        head->CAPfastArr = TMALLOC(CAPfastEntry, count);
#if NGSPICE_OPT_GENERAL
        head->CAPhotArr = TMALLOC(CAPhotEntry, count);
        head->CAPceqCache = TMALLOC(double, count);
#endif
        uintptr_t ax_addr = (uintptr_t)ax;
        size_t ax_bytes = (size_t)nz * sizeof(double);
        int k = 0;
        for (m = head; m != NULL; m = CAPnextModel(m)) {
            for (here = CAPinstances(m); here != NULL; here = CAPnextInstance(here)) {
                CAPfastEntry *e = &head->CAPfastArr[k];
                e->capac = here->CAPcapac;
                e->m = here->CAPm;
                e->posNode = here->CAPposNode;
                e->negNode = here->CAPnegNode;
                e->qcap = here->CAPqcap;
                e->ptrPP = here->CAPposPosPtr;
                e->ptrNN = here->CAPnegNegPtr;
                e->ptrPN = here->CAPposNegPtr;
                e->ptrNP = here->CAPnegPosPtr;
                if (ax != NULL && nz > 0) {
                    e->offPP = cap_ptr_to_klu_off(here->CAPposPosPtr, ax_addr, ax_bytes, (int)nz);
                    e->offNN = cap_ptr_to_klu_off(here->CAPnegNegPtr, ax_addr, ax_bytes, (int)nz);
                    e->offPN = cap_ptr_to_klu_off(here->CAPposNegPtr, ax_addr, ax_bytes, (int)nz);
                    e->offNP = cap_ptr_to_klu_off(here->CAPnegPosPtr, ax_addr, ax_bytes, (int)nz);
                } else {
                    e->offPP = e->offNN = e->offPN = e->offNP = 0;
                }
#if NGSPICE_OPT_GENERAL
                CAPhotEntry *he = &head->CAPhotArr[k];
                he->capac = here->CAPcapac;
                he->posNode = here->CAPposNode;
                he->negNode = here->CAPnegNode;
                he->qcap = here->CAPqcap;
#endif
                k++;
            }
        }
    }
    head->CAPfastVer = cap_global_ver;
    head->CAPfastKluAxBase = ax;
    head->CAPfastKluNz = nz;
}

#ifdef KLU
int
CAPkluPrep(GENmodel *inModel, CKTcircuit *ckt)
{
    long mode = ckt->CKTmode;
    if (!(mode & (MODETRAN | MODEAC)))
        return 0;
    if ((mode & MODEINITTRAN) || (mode & MODEDC))
        return 0;
    if ((ckt->CKTintegrateMethod != GEAR && ckt->CKTintegrateMethod != TRAPEZOIDAL) ||
        (ckt->CKTorder != 1 && ckt->CKTorder != 2))
        return 0;
    CAPmodel *head = (CAPmodel *)inModel;
    CAPensureFastCache(inModel, ckt);
    if (!head->CAPfastKluAxBase || head->CAPfastKluNz == 0)
        return 0;
#if NGSPICE_OPT_GENERAL
    {
        KLUmatrix *km = ckt->CKTmatrix->SMPkluMatrix;
        int nz = (int)km->KLUmatrixNZ;
        int n = head->CAPfastCount;
        if (n > 0 && head->CAPhotArr && head->CAPceqCache) {
            /* 1. Pre-bake constant unit-capacitance KLU matrix once per topology/version */
            if (!km->KLUmatrixCapUnitValid || km->KLUmatrixCapVersion != head->CAPfastVer ||
                !km->KLUmatrixAxCapUnit)
            {
                free(km->KLUmatrixAxCapUnit);
                km->KLUmatrixAxCapUnit = NULL;
                if (posix_memalign((void **)&km->KLUmatrixAxCapUnit, 128,
                                   ((size_t)nz + 1) * sizeof(double)) == 0)
                {
                    double * restrict cu = km->KLUmatrixAxCapUnit;
                    memset(cu, 0, ((size_t)nz + 1) * sizeof(double));
                    const CAPfastEntry * restrict arr = head->CAPfastArr;
                    for (int k = 0; k < n; k++) {
                        double mc = arr[k].m * arr[k].capac;
                        cu[arr[k].offPP] += mc;
                        cu[arr[k].offNN] += mc;
                        cu[arr[k].offPN] -= mc;
                        cu[arr[k].offNP] -= mc;
                    }
                    km->KLUmatrixCapUnitValid = 1;
                    km->KLUmatrixCapVersion = head->CAPfastVer;
                    km->KLUmatrixResCapValid = 0;
                    head->CAPcacheValid = 0;
                    km->KLUmatrixRhsCapValid = 0;
                }
            }
            /* 2. Per-timestep cached CAP ceq[] and RHS contribution (independent of Newton vcap!) */
            if (km->KLUmatrixCapUnitValid && km->KLUmatrixAxCapUnit) {
                int rhs_len = (int)km->KLUmatrixNrhs + 1;
                if (rhs_len < ckt->CKTmaxEqNum + 1)
                    rhs_len = ckt->CKTmaxEqNum + 1;
                if (km->KLUmatrixRhsCapLen < rhs_len || !km->KLUmatrixRhsCapCache) {
                    free(km->KLUmatrixRhsCapCache);
                    km->KLUmatrixRhsCapCache = NULL;
                    km->KLUmatrixRhsCapLen = 0;
                    km->KLUmatrixRhsCapValid = 0;
                    head->CAPcacheValid = 0;
                    if (posix_memalign((void **)&km->KLUmatrixRhsCapCache, 128,
                                       (size_t)rhs_len * sizeof(double)) == 0)
                    {
                        km->KLUmatrixRhsCapLen = rhs_len;
                    }
                }
                if (km->KLUmatrixRhsCapCache) {
                    double ag0 = ckt->CKTag[0];
                    double ag1 = ckt->CKTag[1];
                    double ag2 = ckt->CKTag[2];
                    const double * restrict s1 = ckt->CKTstate1;
                    const double * restrict s2 = ckt->CKTstate2;
                    int is_gear2 = (ckt->CKTintegrateMethod == GEAR && ckt->CKTorder == 2);
                    int is_trap2 = (ckt->CKTintegrateMethod == TRAPEZOIDAL && ckt->CKTorder == 2);
                    int int_mode = is_gear2 ? 2 : (is_trap2 ? 1 : 0);

                    if (!head->CAPcacheValid || !km->KLUmatrixRhsCapValid ||
                        head->CAPcacheTime != ckt->CKTtime ||
                        head->CAPcacheAg0 != ag0 || head->CAPcacheAg1 != ag1 ||
                        head->CAPcacheAg2 != ag2 ||
                        head->CAPcacheS1 != s1 || head->CAPcacheS2 != s2 ||
                        head->CAPcacheIntMode != int_mode)
                    {
                        double * restrict rhs_c = km->KLUmatrixRhsCapCache;
                        double * restrict ceq_c = head->CAPceqCache;
                        const CAPfastEntry * restrict arr = head->CAPfastArr;
                        memset(rhs_c, 0, (size_t)rhs_len * sizeof(double));
                        if (is_gear2) {
                            for (int k = 0; k < n; k++) {
                                int qidx = arr[k].qcap;
                                double ceq = ag2 * s2[qidx] + ag1 * s1[qidx];
                                ceq_c[k] = ceq;
                                double mceq = arr[k].m * ceq;
                                rhs_c[arr[k].posNode] -= mceq;
                                rhs_c[arr[k].negNode] += mceq;
                            }
                        } else if (is_trap2) {
                            for (int k = 0; k < n; k++) {
                                int qidx = arr[k].qcap;
                                double ceq = -s1[qidx + 1] * ag1 - ag0 * s1[qidx];
                                ceq_c[k] = ceq;
                                double mceq = arr[k].m * ceq;
                                rhs_c[arr[k].posNode] -= mceq;
                                rhs_c[arr[k].negNode] += mceq;
                            }
                        } else {
                            for (int k = 0; k < n; k++) {
                                int qidx = arr[k].qcap;
                                double ceq = ag1 * s1[qidx];
                                ceq_c[k] = ceq;
                                double mceq = arr[k].m * ceq;
                                rhs_c[arr[k].posNode] -= mceq;
                                rhs_c[arr[k].negNode] += mceq;
                            }
                        }
                        head->CAPcacheTime = ckt->CKTtime;
                        head->CAPcacheAg0 = ag0;
                        head->CAPcacheAg1 = ag1;
                        head->CAPcacheAg2 = ag2;
                        head->CAPcacheS1 = s1;
                        head->CAPcacheS2 = s2;
                        head->CAPcacheIntMode = int_mode;
                        head->CAPcacheValid = 1;
                        km->KLUmatrixRhsCapValid = 1;
                    }
                    return 3;
                }
            }
        }
    }
#endif
    return 1;
}

static inline void
cap_load_klu_one(const CAPfastEntry * restrict e,
                 const double * restrict rhsOld,
                 double * restrict s0, const double * restrict s1, const double * restrict s2,
                 double * restrict my_ax, double * restrict my_rhs,
                 double ag0, double ag1, double ag2,
                 int is_init_pred, int is_gear2, int is_trap2)
{
    int qidx = e->qcap;
    double q0;
    double ccap_val;
#ifndef PREDICTOR
    if (is_init_pred) {
        q0 = s1[qidx];
        s0[qidx] = q0;
    } else {
#endif
        double vcap = rhsOld[e->posNode] - rhsOld[e->negNode];
        q0 = e->capac * vcap;
        s0[qidx] = q0;
#ifndef PREDICTOR
    }
#endif
    if (is_gear2) {
        ccap_val = ag2 * s2[qidx] + ag1 * s1[qidx] + ag0 * q0;
    } else if (is_trap2) {
        ccap_val = -s1[qidx + 1] * ag1 + ag0 * (q0 - s1[qidx]);
    } else {
        ccap_val = ag0 * q0 + ag1 * s1[qidx];
    }
    s0[qidx + 1] = ccap_val;
    double ceq = ccap_val - ag0 * q0;
    double mgeq = e->m * ag0 * e->capac;
    double mceq = e->m * ceq;
    my_ax[e->offPP] += mgeq;
    my_ax[e->offNN] += mgeq;
    my_ax[e->offPN] -= mgeq;
    my_ax[e->offNP] -= mgeq;
    my_rhs[e->posNode] -= mceq;
    my_rhs[e->negNode] += mceq;
}

int
CAPloadKluPar(GENmodel *inModel, CKTcircuit *ckt, double *my_ax, double *my_rhs)
{
    CAPmodel *head = (CAPmodel *)inModel;
    const CAPfastEntry * restrict arr = head->CAPfastArr;
    int n = head->CAPfastCount;
    const double * restrict rhsOld = ckt->CKTrhsOld;
    double * restrict s0 = ckt->CKTstate0;
    const double * restrict s1 = ckt->CKTstate1;
    const double * restrict s2 = ckt->CKTstate2;
    double ag0 = ckt->CKTag[0];
    double ag1 = ckt->CKTag[1];
    double ag2 = ckt->CKTag[2];
    int is_init_pred = (ckt->CKTmode & MODEINITPRED) != 0;
    int is_gear2 = (ckt->CKTintegrateMethod == GEAR && ckt->CKTorder == 2);
    int is_trap2 = (ckt->CKTintegrateMethod == TRAPEZOIDAL && ckt->CKTorder == 2);

#if NGSPICE_OPT_GENERAL
    {
        KLUmatrix *km = ckt->CKTmatrix->SMPkluMatrix;
        if (km && km->KLUmatrixCapUnitValid && km->KLUmatrixRhsCapValid && head->CAPcacheValid) {
            const CAPhotEntry * restrict hot = head->CAPhotArr;
            const double * restrict ceq_c = head->CAPceqCache;
#ifdef USE_OMP
            if (omp_in_parallel()) {
                #pragma omp for schedule(static) nowait
                for (int k = 0; k < n; k++) {
                    int qidx = hot[k].qcap;
                    double q0;
#ifndef PREDICTOR
                    if (is_init_pred) {
                        q0 = s1[qidx];
                        s0[qidx] = q0;
                        s0[qidx + 1] = is_trap2 ? (-s1[qidx + 1] * ag1) : (ceq_c[k] + ag0 * q0);
                    } else
#endif
                    {
                        q0 = hot[k].capac * (rhsOld[hot[k].posNode] - rhsOld[hot[k].negNode]);
                        s0[qidx] = q0;
                        s0[qidx + 1] = ceq_c[k] + ag0 * q0;
                    }
                }
                return OK;
            }
#endif
#ifndef PREDICTOR
            if (is_init_pred) {
                if (is_trap2) {
                    for (int k = 0; k < n; k++) {
                        int qidx = hot[k].qcap;
                        s0[qidx] = s1[qidx];
                        s0[qidx + 1] = -s1[qidx + 1] * ag1;
                    }
                } else {
                    for (int k = 0; k < n; k++) {
                        int qidx = hot[k].qcap;
                        double q0 = s1[qidx];
                        s0[qidx] = q0;
                        s0[qidx + 1] = ceq_c[k] + ag0 * q0;
                    }
                }
                return OK;
            }
#endif
            for (int k = 0; k < n; k++) {
                int qidx = hot[k].qcap;
                double q0 = hot[k].capac * (rhsOld[hot[k].posNode] - rhsOld[hot[k].negNode]);
                s0[qidx] = q0;
                s0[qidx + 1] = ceq_c[k] + ag0 * q0;
            }
            return OK;
        }
    }
#endif

#ifdef USE_OMP
    if (omp_in_parallel()) {
        #pragma omp for schedule(static) nowait
        for (int k = 0; k < n; k++) {
            cap_load_klu_one(&arr[k], rhsOld, s0, s1, s2, my_ax, my_rhs,
                             ag0, ag1, ag2, is_init_pred, is_gear2, is_trap2);
        }
        return OK;
    }
#endif
    if (!is_init_pred) {
        if (is_gear2) {
            for (int k = 0; k < n; k++) {
                const CAPfastEntry *e = &arr[k];
                int qidx = e->qcap;
                double q0 = e->capac * (rhsOld[e->posNode] - rhsOld[e->negNode]);
                s0[qidx] = q0;
                double ceq = ag2 * s2[qidx] + ag1 * s1[qidx];
                s0[qidx + 1] = ceq + ag0 * q0;
                double mgeq = e->m * ag0 * e->capac;
                double mceq = e->m * ceq;
                my_ax[e->offPP] += mgeq;
                my_ax[e->offNN] += mgeq;
                my_ax[e->offPN] -= mgeq;
                my_ax[e->offNP] -= mgeq;
                my_rhs[e->posNode] -= mceq;
                my_rhs[e->negNode] += mceq;
            }
        } else if (is_trap2) {
            for (int k = 0; k < n; k++) {
                const CAPfastEntry *e = &arr[k];
                int qidx = e->qcap;
                double q0 = e->capac * (rhsOld[e->posNode] - rhsOld[e->negNode]);
                s0[qidx] = q0;
                double ceq = -s1[qidx + 1] * ag1 - ag0 * s1[qidx];
                s0[qidx + 1] = ceq + ag0 * q0;
                double mgeq = e->m * ag0 * e->capac;
                double mceq = e->m * ceq;
                my_ax[e->offPP] += mgeq;
                my_ax[e->offNN] += mgeq;
                my_ax[e->offPN] -= mgeq;
                my_ax[e->offNP] -= mgeq;
                my_rhs[e->posNode] -= mceq;
                my_rhs[e->negNode] += mceq;
            }
        } else {
            for (int k = 0; k < n; k++) {
                const CAPfastEntry *e = &arr[k];
                int qidx = e->qcap;
                double q0 = e->capac * (rhsOld[e->posNode] - rhsOld[e->negNode]);
                s0[qidx] = q0;
                double ceq = ag1 * s1[qidx];
                s0[qidx + 1] = ag0 * q0 + ceq;
                double mgeq = e->m * ag0 * e->capac;
                double mceq = e->m * ceq;
                my_ax[e->offPP] += mgeq;
                my_ax[e->offNN] += mgeq;
                my_ax[e->offPN] -= mgeq;
                my_ax[e->offNP] -= mgeq;
                my_rhs[e->posNode] -= mceq;
                my_rhs[e->negNode] += mceq;
            }
        }
        return OK;
    }
    for (int k = 0; k < n; k++) {
        cap_load_klu_one(&arr[k], rhsOld, s0, s1, s2, my_ax, my_rhs,
                         ag0, ag1, ag2, is_init_pred, is_gear2, is_trap2);
    }
    return OK;
}
#endif

int
CAPload(GENmodel *inModel, CKTcircuit *ckt)
/* actually load the current capacitance value into the
 * sparse matrix previously provided
 */
{
    CAPmodel *model = (CAPmodel*)inModel;
    CAPinstance *here;
    int cond1;
    double vcap;
    double geq;
    double ceq;
    int error;
    double m;

    /* check if capacitors are in the circuit or are open circuited */
    if(ckt->CKTmode & (MODETRAN|MODEAC|MODETRANOP) ) {
        long mode = ckt->CKTmode;
        int is_tran_ac = (mode & (MODETRAN | MODEAC)) != 0;
        int is_init_tran = (mode & MODEINITTRAN) != 0;
        int is_init_pred = (mode & MODEINITPRED) != 0;
        /* evaluate device independent analysis conditions */
        cond1 =
            ( ( (mode & MODEDC) && (mode & MODEINITJCT) )
              || ( (mode & MODEUIC) && is_init_tran ) );

#if NGSPICE_OPT_GENERAL
        /* Fast hoisted path for standard transient iterations (Gear 1/2, Trap 1/2) */
        if (is_tran_ac && !cond1 && !is_init_tran &&
            (ckt->CKTintegrateMethod == GEAR || ckt->CKTintegrateMethod == TRAPEZOIDAL) &&
            (ckt->CKTorder == 1 || ckt->CKTorder == 2))
        {
            CAPensureFastCache(inModel, ckt);
            const CAPfastEntry * restrict arr = model->CAPfastArr;
            int n = model->CAPfastCount;
            const double * restrict rhsOld = ckt->CKTrhsOld;
            double * restrict rhs = ckt->CKTrhs;
            double * restrict s0 = ckt->CKTstate0;
            const double * restrict s1 = ckt->CKTstate1;
            const double * restrict s2 = ckt->CKTstate2;
            double ag0 = ckt->CKTag[0];
            double ag1 = ckt->CKTag[1];
            double ag2 = ckt->CKTag[2];
            int is_gear2 = (ckt->CKTintegrateMethod == GEAR && ckt->CKTorder == 2);
            int is_trap2 = (ckt->CKTintegrateMethod == TRAPEZOIDAL && ckt->CKTorder == 2);

            for (int k = 0; k < n; k++) {
                const CAPfastEntry *e = &arr[k];
                int qidx = e->qcap;
                double q0;
                double ccap_val;
#ifndef PREDICTOR
                if (is_init_pred) {
                    q0 = s1[qidx];
                    s0[qidx] = q0;
                } else {
#endif
                    vcap = rhsOld[e->posNode] - rhsOld[e->negNode];
                    q0 = e->capac * vcap;
                    s0[qidx] = q0;
#ifndef PREDICTOR
                }
#endif
                if (is_gear2) {
                    ccap_val = ag2 * s2[qidx] + ag1 * s1[qidx] + ag0 * q0;
                } else if (is_trap2) {
                    ccap_val = -s1[qidx + 1] * ag1 + ag0 * (q0 - s1[qidx]);
                } else {
                    ccap_val = ag0 * q0 + ag1 * s1[qidx];
                }
                s0[qidx + 1] = ccap_val;
                ceq = ccap_val - ag0 * q0;
                double mgeq = e->m * ag0 * e->capac;
                double mceq = e->m * ceq;
                *(e->ptrPP) += mgeq;
                *(e->ptrNN) += mgeq;
                *(e->ptrPN) -= mgeq;
                *(e->ptrNP) -= mgeq;
                rhs[e->posNode] -= mceq;
                rhs[e->negNode] += mceq;
            }
            return(OK);
        }
#endif

        /*  loop through all the capacitor models */
        for( ; model != NULL; model = CAPnextModel(model)) {

            /* loop through all the instances of the model */
            for (here = CAPinstances(model); here != NULL ;
                    here=CAPnextInstance(here)) {

                m = here->CAPm;

                if(cond1) {
                    vcap = here->CAPinitCond;
                } else {
                    vcap = *(ckt->CKTrhsOld+here->CAPposNode) -
                           *(ckt->CKTrhsOld+here->CAPnegNode) ;
                }
                if(ckt->CKTmode & (MODETRAN | MODEAC)) {
#ifndef PREDICTOR
                    if(ckt->CKTmode & MODEINITPRED) {
                        *(ckt->CKTstate0+here->CAPqcap) =
                            *(ckt->CKTstate1+here->CAPqcap);
                    } else { /* only const caps - no poly's */
#endif /* PREDICTOR */
                        *(ckt->CKTstate0+here->CAPqcap) = here->CAPcapac * vcap;
                        if((ckt->CKTmode & MODEINITTRAN)) {
                            *(ckt->CKTstate1+here->CAPqcap) =
                                *(ckt->CKTstate0+here->CAPqcap);
                        }
#ifndef PREDICTOR
                    }
#endif /* PREDICTOR */
                    error = NIintegrate(ckt,&geq,&ceq,here->CAPcapac,
                                        here->CAPqcap);
                    if(error) return(error);
                    if(ckt->CKTmode & MODEINITTRAN) {
                        *(ckt->CKTstate1+here->CAPccap) =
                            *(ckt->CKTstate0+here->CAPccap);
                    }
                    *(here->CAPposPosPtr) += m * geq;
                    *(here->CAPnegNegPtr) += m * geq;
                    *(here->CAPposNegPtr) -= m * geq;
                    *(here->CAPnegPosPtr) -= m * geq;
                    *(ckt->CKTrhs+here->CAPposNode) -= m * ceq;
                    *(ckt->CKTrhs+here->CAPnegNode) += m * ceq;
                } else
                    *(ckt->CKTstate0+here->CAPqcap) = here->CAPcapac * vcap;
            }
        }
    }
    return(OK);
}

