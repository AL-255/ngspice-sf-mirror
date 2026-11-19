/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"

#define ccap (qcap+1)


void
CKTterr(int qcap, CKTcircuit *ckt, double *timeStep)
{
    double volttol;
    double chargetol;
    double tol;
    double del;
    double factor = 0.0;
    double d0;

    if (ckt->CKTterrCacheValid) {
        const double *s0 = ckt->CKTstate0;
        const double *s1 = ckt->CKTstate1;
        const double *w = ckt->CKTterrWeights;
        double reltol = ckt->CKTreltol;
        double abstol = ckt->CKTabstol;
        double q0 = s0[qcap], q1 = s1[qcap];
        double c0 = fabs(s0[ccap]), c1 = fabs(s1[ccap]);
        double aq0 = fabs(q0), aq1 = fabs(q1);
        double qmax = MAX(aq0, aq1);
        int ord = ckt->CKTorder;

        volttol = abstol + reltol * MAX(c0, c1);
        chargetol = ckt->CKTterrReltolInvDelta * MAX(qmax, ckt->CKTchgtol);
        tol = MAX(volttol, chargetol);

        if (ord == 2) {
            d0 = w[0] * q0 + w[1] * q1
               + w[2] * ckt->CKTstates[2][qcap]
               + w[3] * ckt->CKTstates[3][qcap];
            double den = MAX(abstol, ckt->CKTterrFactor * fabs(d0));
            double num = ckt->CKTtrtol * tol;
            double cur = *timeStep;
            if (cur >= 1e150 || num < (cur * cur) * den) {
                del = sqrt(num / den);
                if (del < cur)
                    *timeStep = del;
            }
            return;
        } else if (ord == 1) {
            d0 = w[0] * q0 + w[1] * q1 + w[2] * ckt->CKTstates[2][qcap];
            double den = MAX(abstol, ckt->CKTterrFactor * fabs(d0));
            double num = ckt->CKTtrtol * tol;
            double cur = *timeStep;
            if (cur >= 1e150 || num < cur * den) {
                del = num / den;
                if (del < cur)
                    *timeStep = del;
            }
            return;
        } else {
            int i;
            d0 = w[0] * q0 + w[1] * q1;
            for (i = 2; i <= ord + 1; i++)
                d0 += w[i] * ckt->CKTstates[i][qcap];
            del = ckt->CKTtrtol * tol / MAX(abstol, ckt->CKTterrFactor * fabs(d0));
            if (ord == 3)
                del = cbrt(del);
            else
                del = exp(log(del) / ord);
            if (del < *timeStep)
                *timeStep = del;
            return;
        }
    }

    {
        double diff[8];
        double deltmp[8];
        int i, j;
        static const double gearCoeff[] = {
            .5,
            .2222222222,
            .1363636364,
            .096,
            .07299270073,
            .05830903790
        };
        static const double trapCoeff[] = {
            .5,
            .08333333333
        };

        volttol = ckt->CKTabstol + ckt->CKTreltol *
                MAX( fabs(ckt->CKTstate0[ccap]), fabs(ckt->CKTstate1[ccap]));

        chargetol = MAX(fabs(ckt->CKTstate0[qcap]),fabs(ckt->CKTstate1[qcap]));
        chargetol = ckt->CKTreltol * MAX(chargetol,ckt->CKTchgtol)/ckt->CKTdelta;
        tol = MAX(volttol,chargetol);
        /* now divided differences */
        for(i=ckt->CKTorder+1;i>=0;i--) {
            diff[i] = ckt->CKTstates[i][qcap];
        }
        for(i=0 ; i <= ckt->CKTorder ; i++) {
            deltmp[i] = ckt->CKTdeltaOld[i];
        }
        j = ckt->CKTorder;
        for (;;) {
            for(i=0;i <= j;i++) {
                diff[i] = (diff[i] - diff[i+1])/deltmp[i];
            }
            if (--j < 0) break;
            for(i=0;i <= j;i++) {
                deltmp[i] = deltmp[i+1] + ckt->CKTdeltaOld[i];
            }
        }
        switch(ckt->CKTintegrateMethod) {
            case GEAR:
                factor = gearCoeff[ckt->CKTorder-1];
                break;

            case TRAPEZOIDAL:
                factor = trapCoeff[ckt->CKTorder - 1] ;
                break;
        }
        del = ckt->CKTtrtol * tol/MAX(ckt->CKTabstol,factor * fabs(diff[0]));
        if(ckt->CKTorder == 2) {
            del = sqrt(del);
        } else if (ckt->CKTorder == 3) {
            del = cbrt(del);
        } else if (ckt->CKTorder > 3) {
            del = exp(log(del)/ckt->CKTorder);
        }
        *timeStep = MIN(*timeStep,del);
    }
    return;
}
