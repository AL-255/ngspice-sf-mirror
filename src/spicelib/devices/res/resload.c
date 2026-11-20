/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Apr 2000 - Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "resdefs.h"
#include "ngspice/sperror.h"
#include <stdint.h>

static unsigned long long res_global_ver = 1;

void
RESbumpVersion(void)
{
    res_global_ver++;
}

int
RESmDelete(GENmodel *inModel)
{
    RESmodel *model = (RESmodel *)inModel;
    FREE(model->RESfastArr);
    model->RESfastCount = 0;
    model->RESfastVer = 0;
    res_global_ver++;
    return OK;
}

static void
RESbuildFastCache(RESmodel *head)
{
    RESmodel *m;
    RESinstance *here;
    int count = 0;
    for (m = head; m != NULL; m = RESnextModel(m)) {
        for (here = RESinstances(m); here != NULL; here = RESnextInstance(here))
            count++;
    }
    FREE(head->RESfastArr);
    head->RESfastCount = count;
    if (count > 0) {
        head->RESfastArr = TMALLOC(RESfastEntry, count);
        int k = 0;
        for (m = head; m != NULL; m = RESnextModel(m)) {
            for (here = RESinstances(m); here != NULL; here = RESnextInstance(here)) {
                head->RESfastArr[k].g = here->RESconduct;
                head->RESfastArr[k].posPosPtr = here->RESposPosPtr;
                head->RESfastArr[k].negNegPtr = here->RESnegNegPtr;
                head->RESfastArr[k].posNegPtr = here->RESposNegPtr;
                head->RESfastArr[k].negPosPtr = here->RESnegPosPtr;
                k++;
            }
        }
    }
    head->RESfastVer = res_global_ver;
}

#ifdef KLU
int
RESkluPrep(GENmodel *inModel, CKTcircuit *ckt)
{
    KLUmatrix *km = ckt->CKTmatrix->SMPkluMatrix;
    double *Ax;
    size_t nz;

    if (!km || !km->KLUmatrixAx || km->KLUmatrixNZ == 0 ||
        km->KLUmatrixIsComplex != KLUmatrixReal)
        return 0;

    Ax = km->KLUmatrixAx;
    nz = (size_t)km->KLUmatrixNZ;

    if (!km->KLUmatrixResConstValid ||
        km->KLUmatrixResVersion != res_global_ver ||
        km->KLUmatrixAxResConst == NULL)
    {
        RESmodel *model = (RESmodel *)inModel;
        RESinstance *here;

        free(km->KLUmatrixAxResConst);
        km->KLUmatrixAxResConst = (double *)calloc(nz + 1, sizeof(double));
        if (!km->KLUmatrixAxResConst) {
            km->KLUmatrixResConstValid = 0;
            return 0;
        }
        double * restrict base = km->KLUmatrixAxResConst;
        uintptr_t ax_addr = (uintptr_t)Ax;
        size_t ax_bytes = nz * sizeof(double);
        for (; model != NULL; model = RESnextModel(model)) {
            for (here = RESinstances(model); here != NULL; here = RESnextInstance(here)) {
                double g = here->RESconduct;
                uintptr_t d0 = (uintptr_t)here->RESposPosPtr - ax_addr;
                uintptr_t d1 = (uintptr_t)here->RESnegNegPtr - ax_addr;
                uintptr_t d2 = (uintptr_t)here->RESposNegPtr - ax_addr;
                uintptr_t d3 = (uintptr_t)here->RESnegPosPtr - ax_addr;
                if (d0 < ax_bytes) base[d0 / sizeof(double)] += g;
                if (d1 < ax_bytes) base[d1 / sizeof(double)] += g;
                if (d2 < ax_bytes) base[d2 / sizeof(double)] -= g;
                if (d3 < ax_bytes) base[d3 / sizeof(double)] -= g;
            }
        }
        km->KLUmatrixResVersion = res_global_ver;
        km->KLUmatrixResConstValid = 1;
    }
    return 2; /* pre-baked constant: CKTload merges AxResConst directly */
}
#endif


/* actually load the current resistance value into the sparse matrix
 * previously provided */
int
RESload(GENmodel *inModel, CKTcircuit *ckt)
{
    RESmodel *model = (RESmodel *)inModel;
#if NGSPICE_OPT_GENERAL
    int need_current = (ckt->CKTmode & MODEINITSMSIG) != 0;

    if (!need_current) {
        /* Fast path: REScurrent is only read by noise analysis after MODEINITSMSIG. */
        if (model->RESfastVer != res_global_ver)
            RESbuildFastCache(model);
        const RESfastEntry * restrict arr = model->RESfastArr;
        int n = model->RESfastCount;
        for (int k = 0; k < n; k++) {
            double g = arr[k].g;
            *(arr[k].posPosPtr) += g;
            *(arr[k].negNegPtr) += g;
            *(arr[k].posNegPtr) -= g;
            *(arr[k].negPosPtr) -= g;
        }
        return(OK);
    }
#endif

    /*  loop through all the resistor models */
    for( ; model != NULL; model = RESnextModel(model)) {
        RESinstance *here;

        /* loop through all the instances of the model */
        for (here = RESinstances(model); here != NULL ;
                here = RESnextInstance(here)) {

            here->REScurrent = (*(ckt->CKTrhsOld+here->RESposNode) -
                                *(ckt->CKTrhsOld+here->RESnegNode)) * here->RESconduct;

            *(here->RESposPosPtr) += here->RESconduct;
            *(here->RESnegNegPtr) += here->RESconduct;
            *(here->RESposNegPtr) -= here->RESconduct;
            *(here->RESnegPosPtr) -= here->RESconduct;
        }
    }
    return(OK);
}


/* actually load the current resistance value into the sparse matrix
 * previously provided */
int
RESacload(GENmodel *inModel, CKTcircuit *ckt)
{
    RESmodel *model = (RESmodel *)inModel;
    double g;

    NG_IGNORE(ckt);

    /*  loop through all the resistor models */
    for( ; model != NULL; model = RESnextModel(model)) {
        RESinstance *here;

        /* loop through all the instances of the model */
        for (here = RESinstances(model); here != NULL ;
             here = RESnextInstance(here)) {

            if (here->RESacresGiven)
                g = here->RESacConduct;
            else
                g = here->RESconduct;

            *(here->RESposPosPtr) += g;
            *(here->RESnegNegPtr) += g;
            *(here->RESposNegPtr) -= g;
            *(here->RESnegPosPtr) -= g;
        }
    }
    return(OK);
}
