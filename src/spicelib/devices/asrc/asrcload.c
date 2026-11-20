/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "asrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include <math.h>


int    asrc_nvals = 0;
double *asrc_vals = NULL;
double *asrc_derivs = NULL;


/* actually load the current voltage value into the
 * sparse matrix previously provided

 * Evaluate the B-source parse tree (example: exp function):
 * ASRCload asrcload.c
 * IFeval ifeval.c
 * PTeval ifeval.c
 * PTexp ptfuncs.c
*/

int
ASRCload(GENmodel *inModel, CKTcircuit *ckt)
{
    ASRCmodel *model = (ASRCmodel*) inModel;
    ASRCinstance *here;
    int i, j;
    double rhs;
    double difference;
    double factor;

    for (; model; model = ASRCnextModel(model)) {
        for (here = ASRCinstances(model); here; here=ASRCnextInstance(here)) {

            difference = (here->ASRCtemp + here->ASRCdtemp) - 300.15; /* FIXME: tnmom instead of 300.15 */
            factor = 1.0
                + here->ASRCtc1 * difference
                + here->ASRCtc2 * difference * difference;

            if (here->ASRCreciproctc == 1)
                factor = 1 / factor;

            if (here->ASRCreciprocm == 1)
                factor = factor / here->ASRCm;
            else
                factor = factor * here->ASRCm;

#ifdef XSPICE_EXP
            value *= ckt->CKTsrcFact;
            value *= cm_analog_ramp_factor();
#else
            if (ckt->CKTmode & MODETRANOP)
                factor *= ckt->CKTsrcFact;
#endif

            /*
             * Get the function and its derivatives evaluated
             */
            i = here->ASRCtree->numVars;
#if NGSPICE_OPT_GENERAL
            if (here->ASRCisAffine) {
                int nvars = i;
                if (!here->ASRCaffValid ||
                    (here->ASRCaffHasTemp && here->ASRCaffTemp != ckt->CKTtemp) ||
                    here->ASRCaffGmin != ckt->CKTgmin)
                {
                    if (asrc_nvals < nvars) {
                        asrc_nvals = nvars;
                        asrc_vals = TREALLOC(double, asrc_vals, nvars);
                        asrc_derivs = TREALLOC(double, asrc_derivs, nvars);
                    }
                    for (i = 0; i < nvars; i++)
                        asrc_vals[i] = 0.0;
                    if (here->ASRCtree->IFeval(here->ASRCtree, ckt->CKTgmin,
                                               &here->ASRCaffC0, asrc_vals,
                                               nvars > 0 ? here->ASRCaffDerivs : asrc_derivs) != OK)
                    {
                        fprintf(stderr, "    in line %s\n\n", here->gen.GENname);
                        return(E_BADPARM);
                    }
                    int all_fin = isfinite(here->ASRCaffC0);
                    for (i = 0; i < nvars && all_fin; i++)
                        if (!isfinite(here->ASRCaffDerivs[i]))
                            all_fin = 0;
                    if (all_fin) {
                        here->ASRCaffTemp = ckt->CKTtemp;
                        here->ASRCaffGmin = ckt->CKTgmin;
                        here->ASRCaffValid = 1;
                    } else {
                        here->ASRCisAffine = 0;
                    }
                }
                if (here->ASRCisAffine && here->ASRCaffValid) {
                    const double * restrict g = here->ASRCaffDerivs;
                    const int * restrict vars = here->ASRCvars;
                    const double * restrict rhsOld = ckt->CKTrhsOld;
                    double ** restrict ptrs = here->ASRCposPtr;
                    double c0 = here->ASRCaffC0;
                    double val = c0;
                    for (i = 0; i < nvars; i++)
                        val += rhsOld[vars[i]] * g[i];

                    here->ASRCprev_value = val;

                    if (ckt->CKTmode & MODEINITSMSIG) {
                        for (i = 0; i < nvars; i++)
                            here->ASRCacValues[i] = g[i];
                        here->ASRCacValues[nvars] = factor * c0;
                    }

                    j = 0;
                    if (here->ASRCtype == ASRC_VOLTAGE) {
                        *(ptrs[j++]) += 1.0;
                        *(ptrs[j++]) -= 1.0;
                        *(ptrs[j++]) -= 1.0;
                        *(ptrs[j++]) += 1.0;
                        for (i = 0; i < nvars; i++)
                            *(ptrs[j++]) -= g[i] * factor;
                        if (c0 != 0.0)
                            ckt->CKTrhs[here->ASRCbranch] += factor * c0;
                    } else {
                        for (i = 0; i < nvars; i++) {
                            double gf = g[i] * factor;
                            *(ptrs[j++]) += gf;
                            *(ptrs[j++]) -= gf;
                        }
                        if (c0 != 0.0) {
                            double fc0 = factor * c0;
                            ckt->CKTrhs[here->ASRCposNode] -= fc0;
                            ckt->CKTrhs[here->ASRCnegNode] += fc0;
                        }
                    }
                    continue;
                }
            }
#endif
            if (asrc_nvals < i) {
                asrc_nvals = i;
                asrc_vals = TREALLOC(double, asrc_vals, i);
                asrc_derivs = TREALLOC(double, asrc_derivs, i);
            }

            j = 0;

            /*
             * Fill the vector of values from the previous solution
             */
            for (i = 0; i < here->ASRCtree->numVars; i++)
                asrc_vals[i] = ckt->CKTrhsOld[here->ASRCvars[i]];

            if (here->ASRCtree->IFeval(here->ASRCtree, ckt->CKTgmin, &rhs, asrc_vals, asrc_derivs) != OK) {
                fprintf(stderr, "    in line %s\n\n", here->gen.GENname);
                return(E_BADPARM);
            }

            /* The convergence test */
            here->ASRCprev_value = rhs;

            /* The ac load precomputation and storage */
            if (ckt->CKTmode & MODEINITSMSIG)
                for (i = 0; i < here->ASRCtree->numVars; i++)
                    here->ASRCacValues[i] = asrc_derivs[i];

            if (here->ASRCtype == ASRC_VOLTAGE) {

                *(here->ASRCposPtr[j++]) += 1.0;
                *(here->ASRCposPtr[j++]) -= 1.0;
                *(here->ASRCposPtr[j++]) -= 1.0;
                *(here->ASRCposPtr[j++]) += 1.0;

                for (i = 0; i < here->ASRCtree->numVars; i++) {
                    rhs -= (asrc_vals[i] * asrc_derivs[i]);

                    *(here->ASRCposPtr[j++]) -= asrc_derivs[i] * factor;
                }

                ckt->CKTrhs[here->ASRCbranch] += factor * rhs;

            } else {

                for (i = 0; i < here->ASRCtree->numVars; i++) {
                    rhs -= (asrc_vals[i] * asrc_derivs[i]);

                    *(here->ASRCposPtr[j++]) += asrc_derivs[i] * factor;
                    *(here->ASRCposPtr[j++]) -= asrc_derivs[i] * factor;
                }

                ckt->CKTrhs[here->ASRCposNode] -= factor * rhs;
                ckt->CKTrhs[here->ASRCnegNode] += factor * rhs;
            }

            /* Store the rhs for small signal analysis */
            if (ckt->CKTmode & MODEINITSMSIG)
                here->ASRCacValues[here->ASRCtree->numVars] = factor * rhs;
        }
    }

    return(OK);
}
