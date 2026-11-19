/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/*
 * NIconvTest(ckt)
 *  perform the convergence test - returns 1 if any of the 
 *  values in the old and new arrays have changed by more 
 *  than absTol + relTol*(max(old,new)), otherwise returns 0
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/smpdefs.h"
#if NGSPICE_OPT_APPLE
#include <arm_neon.h>
#endif

extern bool ft_ngdebug;


int
NIconvTest(CKTcircuit *ckt)
{
    int i; /* generic loop variable */
    int size;  /* size of the matrix */
    CKTnode *node; /* current matrix entry */
    double old;
    double new;
    double tol;
    static int nancount = 0;

    size = SMPmatSize(ckt->CKTmatrix);
#if NGSPICE_OPT_GENERAL
    if (!ckt->CKTnodeIsVolt || ckt->CKTnodeIsVoltSize != size) {
        FREE(ckt->CKTnodeIsVolt);
        ckt->CKTnodeIsVolt = TMALLOC(unsigned char, size + 1);
        ckt->CKTnodeIsVoltSize = size;
        node = ckt->CKTnodes;
        ckt->CKTnodeIsVolt[0] = 0;
        for (i = 1; i <= size; i++) {
            node = node->next;
            ckt->CKTnodeIsVolt[i] = (node->type == SP_VOLTAGE) ? 1 : 0;
        }
    }
#else
    node = ckt->CKTnodes;
#endif
#ifdef STEPDEBUG
    for (i=1;i<=size;i++) {
        new =  ckt->CKTrhs [i] ;
        old =  ckt->CKTrhsOld [i] ;
        printf("chk for convergence:   %s    new: %g    old: %g\n",CKTnodName(ckt,i),new,old);
    }
#endif /* STEPDEBUG */
#if NGSPICE_OPT_GENERAL
    {
        const double * restrict rhs = ckt->CKTrhs;
        const double * restrict rhsOld = ckt->CKTrhsOld;
        const unsigned char * restrict isVolt = ckt->CKTnodeIsVolt;
        double reltol = ckt->CKTreltol;
        double voltTol = ckt->CKTvoltTol;
        double abstol = ckt->CKTabstol;

        i = 1;
#if NGSPICE_OPT_APPLE
        {
            const float64x2_t vrel = vdupq_n_f64(reltol);
            const float64x2_t vvt  = vdupq_n_f64(voltTol);
            const float64x2_t vat  = vdupq_n_f64(abstol);
            for (; i + 1 <= size; i += 2) {
                float64x2_t vn = vld1q_f64(rhs + i);
                float64x2_t vo = vld1q_f64(rhsOld + i);
                float64x2_t abs_t;
                if ((isVolt[i] & isVolt[i + 1]) != 0) {
                    abs_t = vvt;
                } else {
                    abs_t = (float64x2_t){ isVolt[i] ? voltTol : abstol,
                                           isVolt[i + 1] ? voltTol : abstol };
                }
                float64x2_t mx = vmaxq_f64(vabsq_f64(vo), vabsq_f64(vn));
                float64x2_t vtol = vfmaq_f64(abs_t, vrel, mx);
                float64x2_t diff = vabsq_f64(vsubq_f64(vn, vo));
                uint64x2_t ok_m = vcleq_f64(diff, vtol);
                if (vminvq_u32(vreinterpretq_u32_u64(ok_m)) == 0) {
                    break; /* Handle exact failing/NaN index in scalar loop below */
                }
            }
        }
#endif
        for (; i <= size; i++) {
            new = rhs[i];
            old = rhsOld[i];
            double abs_tol = isVolt[i] ? voltTol : abstol;
            double aold = fabs(old);
            double anew = fabs(new);
            tol = reltol * (aold > anew ? aold : anew) + abs_tol;
            if (!(fabs(new - old) <= tol)) {
                if (isnan(new)) {
                    if (ft_ngdebug && nancount < 10) {
                        fprintf(stderr, "Warning: non-convergence, node %s is nan\n", CKTnodName(ckt, i));
                        nancount++;
                    }
                    else if (ft_ngdebug && nancount == 10) {
                        fprintf(stderr, "    non-convergence warnings (nan) limited to 10 node %s\n", CKTnodName(ckt, i));
                        nancount++;
                    }
                    return 1;
                }
                ckt->CKTtroubleNode = i;
                ckt->CKTtroubleElt = NULL;
                return(1);
            }
        }
    }
#else
    for (i=1;i<=size;i++) {
        node = node->next;
        new =  ckt->CKTrhs [i] ;
        old =  ckt->CKTrhsOld [i] ;
        if (isnan(new)) {
            if (ft_ngdebug && nancount < 10) {
                fprintf(stderr, "Warning: non-convergence, node %s is nan\n", CKTnodName(ckt, i));
                nancount++;
            }
            else if (ft_ngdebug && nancount == 10) {
                fprintf(stderr, "    non-convergence warnings (nan) limited to 10 node %s\n", CKTnodName(ckt, i));
                nancount++;
            }
            return 1;
        }
        if(node->type == SP_VOLTAGE) {
            tol =  ckt->CKTreltol * (MAX(fabs(old),fabs(new))) +
                    ckt->CKTvoltTol;
            if (fabs(new-old) >tol ) {
                ckt->CKTtroubleNode = i;
                ckt->CKTtroubleElt = NULL;
                return(1);
            }
        } else {
            tol =  ckt->CKTreltol * (MAX(fabs(old),fabs(new))) +
                    ckt->CKTabstol;
            if (fabs(new-old) >tol ) {
                ckt->CKTtroubleNode = i;
                ckt->CKTtroubleElt = NULL;
                return(1);
            }
        }
    }
#endif

#ifdef NEWCONV
    i = CKTconvTest(ckt);
    /* The individual testers, called by CKTconvTest, set
     * ckt->CKTnoncon and ckt->CKTtroubleElt appropriately. */
    if (ckt->CKTnoncon != 0) {
        ckt->CKTtroubleNode = 0;
        return(1);
    }
    /* CKTconvTest early-returns nonzero 'i' on the first error
     * in evaluating convergence (such as parameter out of range) so
     * there may be untested devices that have not yet converged */
    if (i)
	ckt->CKTtroubleNode = 0;
    return(i);
#else /* NEWCONV */
    return(0);
#endif /* NEWCONV */
}
