/**** BSIM4.5.0 Released by Xuemei (Jane) Xi 07/29/2005 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4trunc.c of BSIM4.5.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v5def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#ifdef USE_OMP
#include <omp.h>
#endif


int
BSIM4v5trunc(
GENmodel *inModel,
CKTcircuit *ckt,
double *timeStep)
{
BSIM4v5model *model = (BSIM4v5model*)inModel;
BSIM4v5instance *here;

#if NGSPICE_OPT_ZEN4 && !defined(STEPDEBUG)
    if (BSIM4v5truncAvx(inModel, ckt, timeStep) == 0)
        return(OK);
#endif

#ifdef USE_OMP
    int InstCount = model->BSIM4v5InstCount;
    BSIM4v5instance **InstArray = model->BSIM4v5InstanceArray;
    if (InstCount > 0 && InstArray != NULL) {
        double minStep = *timeStep;
        int idx;
#pragma omp parallel for reduction(min:minStep) if(InstCount > 256 && omp_get_max_threads() > 1)
        for (idx = 0; idx < InstCount; idx++) {
            BSIM4v5instance *inst = InstArray[idx];
            CKTterr(inst->BSIM4v5qb, ckt, &minStep);
            CKTterr(inst->BSIM4v5qg, ckt, &minStep);
            CKTterr(inst->BSIM4v5qd, ckt, &minStep);
            if (inst->BSIM4v5trnqsMod)
                CKTterr(inst->BSIM4v5qcdump, ckt, &minStep);
            if (inst->BSIM4v5rbodyMod) {
                CKTterr(inst->BSIM4v5qbs, ckt, &minStep);
                CKTterr(inst->BSIM4v5qbd, ckt, &minStep);
            }
            if (inst->BSIM4v5rgateMod == 3)
                CKTterr(inst->BSIM4v5qgmid, ckt, &minStep);
        }
        *timeStep = minStep;
        return(OK);
    }
#endif

#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */

    for (; model != NULL; model = BSIM4v5nextModel(model))
    {    for (here = BSIM4v5instances(model); here != NULL;
	      here = BSIM4v5nextInstance(here))
	 {

#ifdef STEPDEBUG
            debugtemp = *timeStep;
#endif /* STEPDEBUG */
            CKTterr(here->BSIM4v5qb,ckt,timeStep);
            CKTterr(here->BSIM4v5qg,ckt,timeStep);
            CKTterr(here->BSIM4v5qd,ckt,timeStep);
            if (here->BSIM4v5trnqsMod)
                CKTterr(here->BSIM4v5qcdump,ckt,timeStep);
            if (here->BSIM4v5rbodyMod)
            {   CKTterr(here->BSIM4v5qbs,ckt,timeStep);
                CKTterr(here->BSIM4v5qbd,ckt,timeStep);
	    }
	    if (here->BSIM4v5rgateMod == 3)
		CKTterr(here->BSIM4v5qgmid,ckt,timeStep);
#ifdef STEPDEBUG
            if(debugtemp != *timeStep)
	    {  printf("device %s reduces step from %g to %g\n",
                       here->BSIM4v5name,debugtemp,*timeStep);
            }
#endif /* STEPDEBUG */
        }
    }
    return(OK);
}
