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
