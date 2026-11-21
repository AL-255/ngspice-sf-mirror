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

#if NGSPICE_OPT_GENERAL
static void
cap_mr_free(CAPmodel *head)
{
    FREE(head->CAPmrCscCp);
    FREE(head->CAPmrCscCi);
    FREE(head->CAPmrCscCx);
    FREE(head->CAPmrNodeCmax);
    FREE(head->CAPmrNodeCapPtr);
    FREE(head->CAPmrNodeCapIdx);
    FREE(head->CAPmrCapStamp);
    for (int i = 0; i < 4; i++)
        FREE(head->CAPmrVhist[i]);
    for (int i = 0; i < 3; i++)
        FREE(head->CAPmrQnode[i]);
    FREE(head->CAPmrInodePrev);
    FREE(head->CAPmrDvdtPrev);
    FREE(head->CAPmrDvdtTmp);
    FREE(head->CAPmrNodeDd);
    head->CAPmrBuilt = 0;
    head->CAPmrVer = 0;
    head->CAPmrN = 0;
    head->CAPmrHasIc = 0;
    head->CAPmrCscNz = 0;
    head->CAPmrStampGen = 0;
    head->CAPmrHistValid = 0;
    head->CAPmrStateLazy = 0;
    head->CAPmrIncrAge = 0;
    head->CAPmrAcceptedSteps = 0;
}

int
CAPmrDeferredEnabled(void)
{
    static int init = 0, en = 1;
    if (!init) {
        en = 1;
        if (getenv("NGSPICE_NO_MR_DEFCAP") != NULL) {
            en = 0;
        } else {
            const char *v = getenv("NGSPICE_MR_DEFCAP");
            if (v != NULL && v[0] != '\0')
                en = (v[0] != '0');
        }
        init = 1;
    }
    return en;
}

int
CAPmrActiveTruncEnabled(void)
{
    static int init = 0, en = 1;
    if (!init) {
        en = 1;
        if (getenv("NGSPICE_NO_MR_TRUNC") != NULL) {
            en = 0;
        } else {
            const char *v = getenv("NGSPICE_MR_TRUNC");
            if (v != NULL && v[0] != '\0')
                en = (v[0] != '0');
        }
        init = 1;
    }
    return en;
}

void
CAPmrEnsureBuilt(GENmodel *inModel, CKTcircuit *ckt)
{
    CAPmodel *head = (CAPmodel *)inModel;
    int N = ckt->CKTmaxEqNum;
    if (N <= 1 || head->CAPfastCount <= 0)
        return;
    if (head->CAPmrBuilt && head->CAPmrVer == head->CAPfastVer && head->CAPmrN == N)
        return;

    cap_mr_free(head);

    int count = head->CAPfastCount;
    const CAPfastEntry * restrict arr = head->CAPfastArr;

    /* Check if any capacitor specifies explicit initial condition ic=... */
    int has_ic = 0;
    for (CAPmodel *m = head; m != NULL && !has_ic; m = CAPnextModel(m)) {
        for (CAPinstance *here = CAPinstances(m); here != NULL; here = CAPnextInstance(here)) {
            if (here->CAPicGiven) {
                has_ic = 1;
                break;
            }
        }
    }
    head->CAPmrHasIc = has_ic;

    /* 1. Build node-to-capacitor CSR & per-node Cmax */
    double *cmax = TMALLOC(double, N + 1);
    int *ncptr = TMALLOC(int, N + 2);
    memset(cmax, 0, (size_t)(N + 1) * sizeof(double));
    memset(ncptr, 0, (size_t)(N + 2) * sizeof(int));

    for (int k = 0; k < count; k++) {
        int p = arr[k].posNode;
        int n = arr[k].negNode;
        double ac = fabs(arr[k].capac);
        if (p > 0 && p < N) {
            ncptr[p + 1]++;
            if (ac > cmax[p]) cmax[p] = ac;
        }
        if (n > 0 && n < N && n != p) {
            ncptr[n + 1]++;
            if (ac > cmax[n]) cmax[n] = ac;
        }
    }
    for (int i = 1; i <= N; i++)
        ncptr[i] += ncptr[i - 1];
    int total_inc = ncptr[N];
    int *ncidx = TMALLOC(int, total_inc > 0 ? total_inc : 1);
    int *cur = TMALLOC(int, N + 1);
    memcpy(cur, ncptr, (size_t)(N + 1) * sizeof(int));
    for (int k = 0; k < count; k++) {
        int p = arr[k].posNode;
        int n = arr[k].negNode;
        if (p > 0 && p < N)
            ncidx[cur[p]++] = k;
        if (n > 0 && n < N && n != p)
            ncidx[cur[n]++] = k;
    }
    FREE(cur);

    /* 2. Build compact merged CSC capacitance matrix C_mat (columns 1..N-1) */
    int *cp = TMALLOC(int, N + 2);
    double *acc = TMALLOC(double, N + 1);
    int *mark = TMALLOC(int, N + 1);
    int *touched = TMALLOC(int, N + 1);
    memset(acc, 0, (size_t)(N + 1) * sizeof(double));
    memset(mark, 0, (size_t)(N + 1) * sizeof(int));

    int cap_nz_est = total_inc + N + 16;
    int *ci = TMALLOC(int, cap_nz_est);
    double *cx = TMALLOC(double, cap_nz_est);
    int nz = 0;
    cp[0] = 0;
    cp[1] = 0;
    for (int col = 1; col < N; col++) {
        cp[col] = nz;
        int nt = 0;
        int p0 = ncptr[col], p1 = ncptr[col + 1];
        for (int idx = p0; idx < p1; idx++) {
            int k = ncidx[idx];
            int pn = arr[k].posNode;
            int nn = arr[k].negNode;
            double mc = arr[k].m * arr[k].capac;
            if (mc == 0.0 || pn == nn)
                continue;
            int other = (pn == col) ? nn : pn;
            if (mark[col] != col) {
                mark[col] = col;
                acc[col] = 0.0;
                touched[nt++] = col;
            }
            acc[col] += mc;
            if (other > 0 && other < N) {
                if (mark[other] != col) {
                    mark[other] = col;
                    acc[other] = 0.0;
                    touched[nt++] = other;
                }
                acc[other] -= mc;
            }
        }
        if (nz + nt > cap_nz_est) {
            cap_nz_est = (nz + nt) * 2;
            ci = TREALLOC(int, ci, cap_nz_est);
            cx = TREALLOC(double, cx, cap_nz_est);
        }
        for (int t = 0; t < nt; t++) {
            int r = touched[t];
            double v = acc[r];
            acc[r] = 0.0;
            if (v != 0.0) {
                ci[nz] = r;
                cx[nz] = v;
                nz++;
            }
        }
    }
    cp[N] = nz;
    FREE(acc);
    FREE(mark);
    FREE(touched);

    head->CAPmrN = N;
    head->CAPmrCscNz = nz;
    head->CAPmrCscCp = cp;
    head->CAPmrCscCi = ci;
    head->CAPmrCscCx = cx;
    head->CAPmrNodeCmax = cmax;
    head->CAPmrNodeCapPtr = ncptr;
    head->CAPmrNodeCapIdx = ncidx;
    head->CAPmrCapStamp = TMALLOC(unsigned int, count);
    memset(head->CAPmrCapStamp, 0, (size_t)count * sizeof(unsigned int));
    head->CAPmrStampGen = 0;

    for (int i = 0; i < 4; i++) {
        head->CAPmrVhist[i] = TMALLOC(double, N + 1);
        memset(head->CAPmrVhist[i], 0, (size_t)(N + 1) * sizeof(double));
    }
    for (int i = 0; i < 3; i++) {
        head->CAPmrQnode[i] = TMALLOC(double, N + 1);
        memset(head->CAPmrQnode[i], 0, (size_t)(N + 1) * sizeof(double));
    }
    head->CAPmrInodePrev = TMALLOC(double, N + 1);
    head->CAPmrDvdtPrev  = TMALLOC(double, N + 1);
    head->CAPmrDvdtTmp   = TMALLOC(double, N + 1);
    head->CAPmrNodeDd    = TMALLOC(double, N + 1);
    memset(head->CAPmrInodePrev, 0, (size_t)(N + 1) * sizeof(double));
    memset(head->CAPmrDvdtPrev,  0, (size_t)(N + 1) * sizeof(double));
    memset(head->CAPmrDvdtTmp,   0, (size_t)(N + 1) * sizeof(double));
    memset(head->CAPmrNodeDd,    0, (size_t)(N + 1) * sizeof(double));

    head->CAPmrVer = head->CAPfastVer;
    head->CAPmrBuilt = 1;
}

static inline void
cap_mr_spmv_full(const CAPmodel * restrict head, const double * restrict v, double * restrict q)
{
    int N = head->CAPmrN;
    const int * restrict cp = head->CAPmrCscCp;
    const int * restrict ci = head->CAPmrCscCi;
    const double * restrict cx = head->CAPmrCscCx;
    memset(q, 0, (size_t)(N + 1) * sizeof(double));
    for (int j = 1; j < N; j++) {
        double vj = v[j];
        if (vj == 0.0) continue;
        int p0 = cp[j], p1 = cp[j + 1];
        for (int p = p0; p < p1; p++)
            q[ci[p]] += cx[p] * vj;
    }
}

int
CAPaccept(CKTcircuit *ckt, GENmodel *inModel)
{
    if (!(CAPmrDeferredEnabled() || CAPmrActiveTruncEnabled()))
        return OK;
    if (!(ckt->CKTmode & (MODETRAN | MODETRANOP)) || ckt->CKTsenInfo)
        return OK;

    CAPmodel *head = (CAPmodel *)inModel;
    CAPensureFastCache(inModel, ckt);
    if (head->CAPfastCount < 256) {
        head->CAPmrHistValid = 0;
        head->CAPmrStateLazy = 0;
        return OK;
    }
    CAPmrEnsureBuilt(inModel, ckt);
    if (!head->CAPmrBuilt || head->CAPmrHasIc) {
        head->CAPmrHistValid = 0;
        head->CAPmrStateLazy = 0;
        return OK;
    }

    int N = head->CAPmrN;
    const double * restrict v0 = ckt->CKTrhsOld;

    if (ckt->CKTtime == 0.0) {
        head->CAPmrAcceptedSteps = 0;
        head->CAPmrIncrAge = 0;
        head->CAPmrStateLazy = 0;
        memset(head->CAPmrInodePrev, 0, (size_t)(N + 1) * sizeof(double));
        memset(head->CAPmrDvdtPrev,  0, (size_t)(N + 1) * sizeof(double));
        if (ckt->CKTmode & MODEUIC) {
            for (int i = 0; i < 4; i++)
                memset(head->CAPmrVhist[i], 0, (size_t)(N + 1) * sizeof(double));
            for (int i = 0; i < 3; i++)
                memset(head->CAPmrQnode[i], 0, (size_t)(N + 1) * sizeof(double));
        } else {
            memcpy(head->CAPmrVhist[0], v0, (size_t)N * sizeof(double));
            head->CAPmrVhist[0][0] = 0.0;
            head->CAPmrVhist[0][N] = 0.0;
            memcpy(head->CAPmrVhist[1], head->CAPmrVhist[0], (size_t)(N + 1) * sizeof(double));
            memcpy(head->CAPmrVhist[2], head->CAPmrVhist[0], (size_t)(N + 1) * sizeof(double));
            memcpy(head->CAPmrVhist[3], head->CAPmrVhist[0], (size_t)(N + 1) * sizeof(double));
            cap_mr_spmv_full(head, head->CAPmrVhist[0], head->CAPmrQnode[0]);
            memcpy(head->CAPmrQnode[1], head->CAPmrQnode[0], (size_t)(N + 1) * sizeof(double));
            memcpy(head->CAPmrQnode[2], head->CAPmrQnode[0], (size_t)(N + 1) * sizeof(double));
        }
        head->CAPmrHistValid = 1;
        return OK;
    }

    if (!head->CAPmrHistValid)
        return OK;

    if ((ckt->CKTintegrateMethod != GEAR && ckt->CKTintegrateMethod != TRAPEZOIDAL) ||
        !(ckt->CKTorder == 1 || ckt->CKTorder == 2 ||
          (ckt->CKTintegrateMethod == GEAR && ckt->CKTorder == 3)))
    {
        head->CAPmrHistValid = 0;
        head->CAPmrStateLazy = 0;
        return OK;
    }

    head->CAPmrAcceptedSteps++;

    const double * restrict v1 = head->CAPmrVhist[0];
    const double * restrict v2 = head->CAPmrVhist[1];
    const double * restrict v3 = head->CAPmrVhist[2];
    double * restrict dvdt_new = head->CAPmrDvdtTmp;
    const double * restrict dvdt_old = head->CAPmrDvdtPrev;
    double ag0 = ckt->CKTag[0];
    double ag1 = ckt->CKTag[1];
    double ag2 = ckt->CKTag[2];
    double ag3 = ckt->CKTag[3];
    int is_trap = (ckt->CKTintegrateMethod == TRAPEZOIDAL);
    int is_trap2_solved = (is_trap && ag1 >= 0.0);

    dvdt_new[0] = 0.0;
    if (!is_trap) {
        if (ckt->CKTorder == 3) {
            for (int i = 1; i < N; i++)
                dvdt_new[i] = ag0 * v0[i] + ag1 * v1[i] + ag2 * v2[i] + ag3 * v3[i];
        } else if (ag2 != 0.0) {
            for (int i = 1; i < N; i++)
                dvdt_new[i] = ag0 * v0[i] + ag1 * v1[i] + ag2 * v2[i];
        } else {
            for (int i = 1; i < N; i++)
                dvdt_new[i] = ag0 * v0[i] + ag1 * v1[i];
        }
    } else if (is_trap2_solved) {
        for (int i = 1; i < N; i++)
            dvdt_new[i] = -ag1 * dvdt_old[i] + ag0 * (v0[i] - v1[i]);
    } else {
        for (int i = 1; i < N; i++)
            dvdt_new[i] = ag0 * v0[i] + ag1 * v1[i];
    }

    /* Update nodal charge vector Qnode[0] = C_mat * v0 incrementally (with periodic exact refresh) */
    memcpy(head->CAPmrQnode[2], head->CAPmrQnode[1], (size_t)(N + 1) * sizeof(double));
    memcpy(head->CAPmrQnode[1], head->CAPmrQnode[0], (size_t)(N + 1) * sizeof(double));
    if (++head->CAPmrIncrAge >= 64) {
        cap_mr_spmv_full(head, v0, head->CAPmrQnode[0]);
        head->CAPmrIncrAge = 0;
    } else {
        double * restrict q0 = head->CAPmrQnode[0];
        const int * restrict cp = head->CAPmrCscCp;
        const int * restrict ci = head->CAPmrCscCi;
        const double * restrict cx = head->CAPmrCscCx;
        for (int j = 1; j < N; j++) {
            double dv = v0[j] - v1[j];
            if (dv == 0.0) continue;
            int p0 = cp[j], p1 = cp[j + 1];
            for (int p = p0; p < p1; p++)
                q0[ci[p]] += cx[p] * dv;
        }
    }

    if (is_trap) {
        double * restrict in_p = head->CAPmrInodePrev;
        const double * restrict q0 = head->CAPmrQnode[0];
        const double * restrict q1 = head->CAPmrQnode[1];
        if (is_trap2_solved) {
            for (int i = 1; i < N; i++)
                in_p[i] = ag0 * (q0[i] - q1[i]) - ag1 * in_p[i];
        } else {
            for (int i = 1; i < N; i++)
                in_p[i] = ag0 * q0[i] + ag1 * q1[i];
        }
    }

    /* Rotate accepted voltage history and dv/dt buffer */
    double *v_recycle = head->CAPmrVhist[3];
    head->CAPmrVhist[3] = head->CAPmrVhist[2];
    head->CAPmrVhist[2] = head->CAPmrVhist[1];
    head->CAPmrVhist[1] = head->CAPmrVhist[0];
    head->CAPmrVhist[0] = v_recycle;
    memcpy(v_recycle, v0, (size_t)N * sizeof(double));
    v_recycle[0] = 0.0;
    v_recycle[N] = 0.0;

    head->CAPmrDvdtTmp = head->CAPmrDvdtPrev;
    head->CAPmrDvdtPrev = dvdt_new;

    return OK;
}
#endif

int
CAPmDelete(GENmodel *inModel)
{
    CAPmodel *model = (CAPmodel *)inModel;
    FREE(model->CAPfastArr);
#if NGSPICE_OPT_GENERAL
    FREE(model->CAPhotArr);
    FREE(model->CAPceqCache);
    model->CAPcacheValid = 0;
    cap_mr_free(model);
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
        !(ckt->CKTorder == 1 || ckt->CKTorder == 2 ||
          (ckt->CKTintegrateMethod == GEAR && ckt->CKTorder == 3)))
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
                    double ag3 = ckt->CKTag[3];
                    const double * restrict s1 = ckt->CKTstate1;
                    const double * restrict s2 = ckt->CKTstate2;
                    const double * restrict s3 = ckt->CKTstate3;
                    int is_gear3 = (ckt->CKTintegrateMethod == GEAR && ckt->CKTorder == 3);
                    int is_gear2 = (ckt->CKTintegrateMethod == GEAR && ckt->CKTorder == 2);
                    int is_trap2 = (ckt->CKTintegrateMethod == TRAPEZOIDAL && ckt->CKTorder == 2);
                    int int_mode = is_gear3 ? 3 : (is_gear2 ? 2 : (is_trap2 ? 1 : 0));

                    if (!head->CAPcacheValid || !km->KLUmatrixRhsCapValid ||
                        head->CAPcacheTime != ckt->CKTtime ||
                        head->CAPcacheAg0 != ag0 || head->CAPcacheAg1 != ag1 ||
                        head->CAPcacheAg2 != ag2 || (is_gear3 && head->CAPcacheAg3 != ag3) ||
                        head->CAPcacheS1 != s1 || head->CAPcacheS2 != s2 ||
                        (is_gear3 && head->CAPcacheS3 != s3) ||
                        head->CAPcacheIntMode != int_mode)
                    {
                        double * restrict rhs_c = km->KLUmatrixRhsCapCache;
                        if (CAPmrDeferredEnabled() && head->CAPmrHistValid && !head->CAPmrHasIc && !ckt->CKTsenInfo) {
                            int N = head->CAPmrN;
                            const double * restrict q1 = head->CAPmrQnode[0];
                            const double * restrict q2 = head->CAPmrQnode[1];
                            const double * restrict q3 = head->CAPmrQnode[2];
                            rhs_c[0] = 0.0;
                            if (is_gear3) {
                                for (int i = 1; i < N; i++)
                                    rhs_c[i] = -(ag1 * q1[i] + ag2 * q2[i] + ag3 * q3[i]);
                            } else if (is_gear2) {
                                for (int i = 1; i < N; i++)
                                    rhs_c[i] = -(ag1 * q1[i] + ag2 * q2[i]);
                            } else if (is_trap2) {
                                const double * restrict in1 = head->CAPmrInodePrev;
                                for (int i = 1; i < N; i++)
                                    rhs_c[i] = ag1 * in1[i] + ag0 * q1[i];
                            } else {
                                for (int i = 1; i < N; i++)
                                    rhs_c[i] = -(ag1 * q1[i]);
                            }
                            if (rhs_len > N)
                                memset(rhs_c + N, 0, (size_t)(rhs_len - N) * sizeof(double));
                            head->CAPmrStateLazy = 1;
                        } else {
                            double * restrict ceq_c = head->CAPceqCache;
                            const CAPfastEntry * restrict arr = head->CAPfastArr;
                            memset(rhs_c, 0, (size_t)rhs_len * sizeof(double));
                            if (is_gear3) {
                                for (int k = 0; k < n; k++) {
                                    int qidx = arr[k].qcap;
                                    double ceq = ag3 * s3[qidx] + ag2 * s2[qidx] + ag1 * s1[qidx];
                                    ceq_c[k] = ceq;
                                    double mceq = arr[k].m * ceq;
                                    rhs_c[arr[k].posNode] -= mceq;
                                    rhs_c[arr[k].negNode] += mceq;
                                }
                            } else if (is_gear2) {
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
                            head->CAPmrStateLazy = 0;
                        }
                        head->CAPcacheTime = ckt->CKTtime;
                        head->CAPcacheAg0 = ag0;
                        head->CAPcacheAg1 = ag1;
                        head->CAPcacheAg2 = ag2;
                        head->CAPcacheAg3 = ag3;
                        head->CAPcacheS1 = s1;
                        head->CAPcacheS2 = s2;
                        head->CAPcacheS3 = s3;
                        head->CAPcacheIntMode = int_mode;
                        head->CAPcacheValid = 1;
                        km->KLUmatrixRhsCapValid = 1;
                    }
                    if (head->CAPmrStateLazy)
                        return 4; /* deferred CAP: skip CAPloadKluPar completely during Newton! */
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
                 double * restrict s0, const double * restrict s1,
                 const double * restrict s2, const double * restrict s3,
                 double * restrict my_ax, double * restrict my_rhs,
                 double ag0, double ag1, double ag2, double ag3,
                 int is_init_pred, int is_gear3, int is_gear2, int is_trap2)
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
    if (is_gear3) {
        ccap_val = ag3 * s3[qidx] + ag2 * s2[qidx] + ag1 * s1[qidx] + ag0 * q0;
    } else if (is_gear2) {
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
    const double * restrict s3 = ckt->CKTstate3;
    double ag0 = ckt->CKTag[0];
    double ag1 = ckt->CKTag[1];
    double ag2 = ckt->CKTag[2];
    double ag3 = ckt->CKTag[3];
#if NGSPICE_OPT_GENERAL
    int is_init_pred = ((ckt->CKTmode & MODEINITPRED) != 0) && !ckt->CKTtaPredActive;
#else
    int is_init_pred = (ckt->CKTmode & MODEINITPRED) != 0;
#endif
    int is_gear3 = (ckt->CKTintegrateMethod == GEAR && ckt->CKTorder == 3);
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
            cap_load_klu_one(&arr[k], rhsOld, s0, s1, s2, s3, my_ax, my_rhs,
                             ag0, ag1, ag2, ag3, is_init_pred, is_gear3, is_gear2, is_trap2);
        }
        return OK;
    }
#endif
    if (!is_init_pred) {
        if (is_gear3) {
            for (int k = 0; k < n; k++) {
                const CAPfastEntry *e = &arr[k];
                int qidx = e->qcap;
                double q0 = e->capac * (rhsOld[e->posNode] - rhsOld[e->negNode]);
                s0[qidx] = q0;
                double ceq = ag3 * s3[qidx] + ag2 * s2[qidx] + ag1 * s1[qidx];
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
        } else if (is_gear2) {
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
        cap_load_klu_one(&arr[k], rhsOld, s0, s1, s2, s3, my_ax, my_rhs,
                         ag0, ag1, ag2, ag3, is_init_pred, is_gear3, is_gear2, is_trap2);
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
#if NGSPICE_OPT_GENERAL
        int is_init_pred = ((mode & MODEINITPRED) != 0) && !ckt->CKTtaPredActive;
#else
        int is_init_pred = (mode & MODEINITPRED) != 0;
#endif
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
                    if(is_init_pred) {
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

