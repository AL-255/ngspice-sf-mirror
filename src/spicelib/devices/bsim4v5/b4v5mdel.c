/**** BSIM4.5.0 Released by Xuemei (Jane) Xi 07/29/2005 ****/

/**********
 * Copyright 2005 Regents of the University of California. All rights reserved.
 * File: b4mdel.c of BSIM4.5.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v5def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM4v5mDelete(GENmodel *gen_model)
{
    BSIM4v5model *model = (BSIM4v5model*) gen_model;

#ifdef USE_OMP
    FREE(model->BSIM4v5InstanceArray);
#endif
#ifdef KLU
    FREE(model->BSIM4v5KluOffArr);
#endif
#if defined(__AVX512F__) && defined(__GNUC__)
    free(model->BSIM4v5AvxBlocks);
    model->BSIM4v5AvxBlocks = NULL;
    FREE(model->BSIM4v5AvxFallbackIdx);
    model->BSIM4v5AvxBuilt = 0;
#endif

    struct bsim4v5SizeDependParam *p = model->pSizeDependParamKnot;
    while (p) {
        struct bsim4v5SizeDependParam *next_p = p->pNext;
        FREE(p);
        p = next_p;
    }

    FREE(model->BSIM4v5version);

    return OK;
}
