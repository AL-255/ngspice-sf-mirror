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

        /* Fast hoisted path for standard transient iterations (Gear 1/2, Trap 1/2) */
        if (is_tran_ac && !cond1 && !is_init_tran &&
            (ckt->CKTintegrateMethod == GEAR || ckt->CKTintegrateMethod == TRAPEZOIDAL) &&
            (ckt->CKTorder == 1 || ckt->CKTorder == 2))
        {
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

            for ( ; model != NULL; model = CAPnextModel(model)) {
                for (here = CAPinstances(model); here != NULL;
                        here = CAPnextInstance(here)) {
                    int qidx = here->CAPqcap;
                    double q0;
                    double ccap_val;
                    m = here->CAPm;
#ifndef PREDICTOR
                    if (is_init_pred) {
                        q0 = s1[qidx];
                        s0[qidx] = q0;
                    } else {
#endif
                        vcap = rhsOld[here->CAPposNode] - rhsOld[here->CAPnegNode];
                        q0 = here->CAPcapac * vcap;
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
                    geq = ag0 * here->CAPcapac;
                    double mgeq = m * geq;
                    double mceq = m * ceq;
                    *(here->CAPposPosPtr) += mgeq;
                    *(here->CAPnegNegPtr) += mgeq;
                    *(here->CAPposNegPtr) -= mgeq;
                    *(here->CAPnegPosPtr) -= mgeq;
                    rhs[here->CAPposNode] -= mceq;
                    rhs[here->CAPnegNode] += mceq;
                }
            }
            return(OK);
        }

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

