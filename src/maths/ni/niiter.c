/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2001 AlansFixes
**********/

/*
 * NIiter(ckt,maxIter)
 *
 *  This subroutine performs the actual numerical iteration.
 *  It uses the sparse matrix stored in the circuit struct
 *  along with the matrix loading program, the load data, the
 *  convergence test function, and the convergence parameters
 */

#include "ngspice/ngspice.h"
#include "ngspice/trandefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/smpdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/fteext.h"

/* Limit the number of 'singular matrix' warnings */
static int msgcount = 0;

/* NIiter() - return value is non-zero for convergence failure */

int
NIiter(CKTcircuit *ckt, int maxIter)
{
    double startTime, *OldCKTstate0 = NULL;
    int error, i, j;

    int iterno = 0;
    int ipass = 0;
#if defined(KLU) && NGSPICE_OPT_GENERAL
    double prev_dx_norm = 1.0e30;
    double step_max_dx = 0.0;
    static int chord_init = 0, chord_disabled = 0, chord_max_cnt = 10;
    if (!chord_init) {
        chord_disabled = (getenv("NGSPICE_NO_CHORD") != NULL);
        const char *sc = getenv("NGSPICE_TA_CHORD");
        if (sc && atoi(sc) > 0)
            chord_max_cnt = (atoi(sc) >= 2) ? atoi(sc) : 32;
        chord_init = 1;
    }
#endif

    /* some convergence issues that get resolved by increasing max iter */
    if (maxIter < 100)
        maxIter = 100;

    if ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)) {
        SWAP(double *, ckt->CKTrhs, ckt->CKTrhsOld);
        error = CKTload(ckt);
        if (error)
            return(error);
        return(OK);
    }

#ifdef WANT_SENSE2
    if (ckt->CKTsenInfo) {
        error = NIsenReinit(ckt);
        if (error)
            return(error);
    }
#endif

    if (ckt->CKTniState & NIUNINITIALIZED) {
        error = NIreinit(ckt); /* always returns 0 */
        if (error) {
#ifdef STEPDEBUG
            printf("re-init returned error \n");
#endif
            return(error);
        }
    }

    /* OldCKTstate0 = TMALLOC(double, ckt->CKTnumStates + 1); */

    for (;;) {

        ckt->CKTnoncon = 0;

#ifdef NEWPRED
        if (!(ckt->CKTmode & MODEINITPRED))
#endif
        {

            error = CKTload(ckt);
            /* printf("loaded, noncon is %d\n", ckt->CKTnoncon); */
            /* fflush(stdout); */
            iterno++;
            if (error) {
                ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
                printf("load returned error \n");
#endif
                FREE(OldCKTstate0);
                return (error);
            }

            /* printf("after loading, before solving\n"); */
            /* CKTdump(ckt); */

            if (!(ckt->CKTniState & NIDIDPREORDER)) {
                error = SMPpreOrder(ckt->CKTmatrix);
                if (error) {
                    ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
                    printf("pre-order returned error \n");
#endif
                    FREE(OldCKTstate0);
                    return(error); /* badly formed matrix */
                }
                ckt->CKTniState |= NIDIDPREORDER;
            }

            if ((ckt->CKTmode & MODEINITJCT) ||
                ((ckt->CKTmode & MODEINITTRAN) && (iterno == 1)))
            {
                ckt->CKTniState |= NISHOULDREORDER;
            }

#if defined(KLU) && NGSPICE_OPT_GENERAL
            int used_chord = 0;
            if (ckt->CKTmrSubActive && ckt->CKTkluMODE && (ckt->CKTmode & MODETRAN)) {
                double dx_norm = 1.0e30, sub_decomp_t = 0.0, sub_solve_t = 0.0;
                int rhs_len = (int)ckt->CKTmatrix->SMPkluMatrix->KLUmatrixNrhs;
                if (rhs_len < ckt->CKTmaxEqNum) rhs_len = ckt->CKTmaxEqNum;
                error = SMPkluSubSolve(ckt->CKTmatrix, ckt->CKTrhs, ckt->CKTrhsOld,
                                       rhs_len, ckt->CKTdiagGmin, ckt->CKTag[0],
                                       iterno, prev_dx_norm, chord_disabled,
                                       SPfrontEnd->IFseconds,
                                       &used_chord, &dx_norm,
                                       &sub_decomp_t, &sub_solve_t,
                                       &ckt->CKTmrSubLuValid, &ckt->CKTmrSubLuAg0,
                                       &ckt->CKTmrSubLuGmin, &ckt->CKTmrSubChordCount);
                ckt->CKTstat->STATdecompTime += sub_decomp_t;
                ckt->CKTstat->STATsolveTime += sub_solve_t;
                if (error) {
                    ckt->CKTstat->STATnumIter += iterno;
                    FREE(OldCKTstate0);
                    return(error);
                }
                prev_dx_norm = dx_norm;
                if (dx_norm > step_max_dx) step_max_dx = dx_norm;
                goto after_lin_solve;
            }
#endif
            if (ckt->CKTniState & NISHOULDREORDER) {
                startTime = SPfrontEnd->IFseconds();

#ifdef KLU
                if (ckt->CKTkluMODE) {
                    ckt->CKTmatrix->SMPkluMatrix->KLUloadDiagGmin = 1 ;
                }
#endif

                error = SMPreorder(ckt->CKTmatrix, ckt->CKTpivotAbsTol,
                                   ckt->CKTpivotRelTol, ckt->CKTdiagGmin);
                ckt->CKTstat->STATreorderTime +=
                    SPfrontEnd->IFseconds() - startTime;
                if (error) {
                    /* new feature - we can now find out something about what is
                     * wrong - so we ask for the troublesome entry
                     * Limit the number of messages to 6, if not 'set ngdebug'.
                     */
                    if (ft_ngdebug || msgcount < 6) {
                        SMPgetError(ckt->CKTmatrix, &i, &j);
                        if(eq(NODENAME(ckt, i), NODENAME(ckt, j)))
                            SPfrontEnd->IFerrorf(ERR_WARNING, "singular matrix:  check node %s\n", NODENAME(ckt, i));
                        else
                            SPfrontEnd->IFerrorf(ERR_WARNING, "singular matrix:  check nodes %s and %s\n", NODENAME(ckt, i), NODENAME(ckt, j));
                        msgcount += 1;
                    }
                    ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
                    printf("reorder returned error \n");
#endif
                    FREE(OldCKTstate0);
                    return(error); /* can't handle these errors - pass up! */
                }
                ckt->CKTniState &= ~NISHOULDREORDER;
#if defined(KLU) && NGSPICE_OPT_GENERAL
                if (ckt->CKTkluMODE && (ckt->CKTmode & MODETRAN)) {
                    ckt->CKTkluLuValid = 1;
                    ckt->CKTkluLuAg0 = ckt->CKTag[0];
                    ckt->CKTkluLuGmin = ckt->CKTdiagGmin;
                    ckt->CKTkluChordCount = 0;
                } else {
                    ckt->CKTkluLuValid = 0;
                }
#endif
            } else {
#if defined(KLU) && NGSPICE_OPT_GENERAL
                if (!chord_disabled && ckt->CKTkluMODE &&
                    (ckt->CKTmode & MODETRAN) &&
                    !(ckt->CKTmode & (MODEINITTRAN | MODEINITJCT | MODEINITFIX)) &&
                    ckt->CKTnoncon == 0 &&
                    ckt->CKTkluLuValid &&
                    ckt->CKTdiagGmin == ckt->CKTkluLuGmin)
                {
                    int vlsi_lu = (ckt->CKTmatrix && ckt->CKTmatrix->SMPkluMatrix &&
                                   ckt->CKTmatrix->SMPkluMatrix->KLUmatrixN >= 10000);
                    int eff_max_cnt = (vlsi_lu && chord_max_cnt < 16) ? 16 : chord_max_cnt;
                    double ag0 = ckt->CKTag[0];
                    double lu_ag0 = ckt->CKTkluLuAg0;
                    int ok_iter = 0;
                    if (ckt->CKTkluChordCount < eff_max_cnt && lu_ag0 > 0.0) {
                        if (vlsi_lu) {
                            /* At VLSI scale (N >= 10k), KLU factor is ~20x slower than backsolve:
                             * aggressively reuse LU whenever exact residual contracts! */
                            int ag0_near = (ag0 >= 0.75 * lu_ag0 && ag0 <= 1.33 * lu_ag0);
                            ok_iter = (iterno >= 2) ? (prev_dx_norm <= 0.50 && ag0_near) : ag0_near;
                        } else {
                            /* Same-timestep polishing (iterno >= 2) or smooth constant-step reuse */
                            ok_iter = (iterno >= 2)
                                ? (prev_dx_norm <= 0.05 && ag0 == lu_ag0)
                                : ((ckt->CKTkluLastStepSmooth && ag0 >= 0.90 * lu_ag0 && ag0 <= 1.11 * lu_ag0) ||
                                   (ckt->CKTtaPredActive && ag0 == lu_ag0));
                        }
                    }
                    if (ok_iter)
                    {
                        double dx_norm = 1.0e30;
                        startTime = SPfrontEnd->IFseconds();
                        if (SMPkluChordSolve(ckt->CKTmatrix, ckt->CKTrhs, ckt->CKTrhsOld,
                                             ckt->CKTdiagGmin, &dx_norm))
                        {
                            int accept;
                            if (vlsi_lu) {
                                double i1_lim = ckt->CKTtaPredActive ? 0.02 : 0.05;
                                accept = (iterno == 1)
                                    ? (dx_norm < i1_lim)
                                    : (dx_norm < 0.30 && (dx_norm <= 0.65 * prev_dx_norm || dx_norm <= 1.0e-4));
                            } else {
                                double i1_lim = ckt->CKTtaPredActive ? 0.004 : 0.015;
                                accept = (iterno == 1)
                                    ? (dx_norm < i1_lim)
                                    : (dx_norm < 0.04 && (dx_norm <= 0.35 * prev_dx_norm || dx_norm <= 5.0e-5));
                            }
                            if (accept) {
                                SMPkluChordCommit(ckt->CKTmatrix, ckt->CKTrhs, ckt->CKTrhsOld);
                                ckt->CKTstat->STATsolveTime += SPfrontEnd->IFseconds() - startTime;
                                ckt->CKTkluChordCount++;
                                prev_dx_norm = dx_norm;
                                if (dx_norm > step_max_dx) step_max_dx = dx_norm;
                                used_chord = 1;
                            }
                        }
                    }
                }
                if (!used_chord) {
#endif
                startTime = SPfrontEnd->IFseconds();

#ifdef KLU
                if (ckt->CKTkluMODE) {
                    ckt->CKTmatrix->SMPkluMatrix->KLUloadDiagGmin = 1 ;
                }
#endif

                error = SMPluFac(ckt->CKTmatrix, ckt->CKTpivotAbsTol,
                                 ckt->CKTdiagGmin);
                ckt->CKTstat->STATdecompTime +=
                    SPfrontEnd->IFseconds() - startTime;

#ifdef KLU
                if ((ckt->CKTkluMODE) && (error == E_SINGULAR)) {

                    /* Francesco Lannutti - 25 Aug 2020
                     * If the matrix is numerically singular during ReFactorization, take the same matrix and factor it from scratch in the same iteration.
                     * This is my mod with KLU. It saves run-time, but also the system at the next iteration may be different.
                     * How do we guarantee that the system is the same at the next iteration? So, the original SPARSE version below sounds like a bug.
                     */
                    if (ft_ngdebug)
                        fprintf (stderr, "Warning: KLU ReFactor failed. Factoring again...\n") ;
                    ckt->CKTniState |= NISHOULDREORDER;
                    ckt->CKTmatrix->SMPkluMatrix->KLUloadDiagGmin = 0 ;
                    error = SMPreorder(ckt->CKTmatrix, ckt->CKTpivotAbsTol, ckt->CKTpivotRelTol, ckt->CKTdiagGmin);
                    ckt->CKTstat->STATreorderTime += SPfrontEnd->IFseconds() - startTime;
                    if (error) {
                        SMPgetError(ckt->CKTmatrix, &i, &j);
                        if (ft_ngdebug || msgcount < 6) {
                            SMPgetError(ckt->CKTmatrix, &i, &j);
                            if (eq(NODENAME(ckt, i), NODENAME(ckt, j)))
                                SPfrontEnd->IFerrorf(ERR_WARNING, "singular matrix:  check node %s\n", NODENAME(ckt, i));
                            else
                                SPfrontEnd->IFerrorf(ERR_WARNING, "singular matrix:  check nodes %s and %s\n", NODENAME(ckt, i), NODENAME(ckt, j));
                            msgcount += 1;
                        }

                        /* CKTload(ckt); */
                        /* SMPprint(ckt->CKTmatrix, stdout); */
                        /* seems to be singular - pass the bad news up */
                        ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
                        printf("lufac returned error \n");
#endif
                        FREE(OldCKTstate0);
                        return(error);
                    }
#if NGSPICE_OPT_GENERAL
                    ckt->CKTniState &= ~NISHOULDREORDER;
#endif
                } else if (error) {
                    if (!(ckt->CKTkluMODE) && (error == E_SINGULAR)) {

                        /* Francesco Lannutti - 25 Aug 2020
                         * If the matrix is numerically singular during ReFactorization, factor it from scratch at the next iteration.
                         * This is the original SPICE3F5 code and uses SPARSE.
                         */

                        ckt->CKTniState |= NISHOULDREORDER;
                        DEBUGMSG(" forced reordering....\n");
                        continue;
                    }
                    /* CKTload(ckt); */
                    /* SMPprint(ckt->CKTmatrix, stdout); */
                    /* seems to be singular - pass the bad news up */
                    ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
                    printf("lufac returned error \n");
#endif
                    FREE(OldCKTstate0);
                    return(error);
                }
#else
                if (error) {
                    if (error == E_SINGULAR) {

                        /* Francesco Lannutti - 25 Aug 2020
                         * If the matrix is numerically singular during ReFactorization, factor it from scratch at the next iteration.
                         * This is the original SPICE3F5 code and uses SPARSE.
                         */

                        ckt->CKTniState |= NISHOULDREORDER;
                        DEBUGMSG(" forced reordering....\n");
                        continue;
                    }
                    /* CKTload(ckt); */
                    /* SMPprint(ckt->CKTmatrix, stdout); */
                    /* seems to be singular - pass the bad news up */
                    ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
                    printf("lufac returned error \n");
#endif
                    FREE(OldCKTstate0);
                    return(error);
                }
#endif
#if defined(KLU) && NGSPICE_OPT_GENERAL
                if (ckt->CKTkluMODE && (ckt->CKTmode & MODETRAN)) {
                    ckt->CKTkluLuValid = 1;
                    ckt->CKTkluLuAg0 = ckt->CKTag[0];
                    ckt->CKTkluLuGmin = ckt->CKTdiagGmin;
                    ckt->CKTkluChordCount = 0;
                } else {
                    ckt->CKTkluLuValid = 0;
                }
                } /* !used_chord */
#endif
            }

            /* moved it to here as if xspice is included then CKTload changes
               CKTnumStates the first time it is run.
               OldCKTstate0 is only used for node damping during DC/TRANOP. */
#if NGSPICE_OPT_GENERAL
            if ((ckt->CKTnodeDamping != 0) &&
                ((ckt->CKTmode & MODETRANOP) || (ckt->CKTmode & MODEDCOP)))
            {
                if (!OldCKTstate0)
                    OldCKTstate0 = TMALLOC(double, ckt->CKTnumStates + 1);
                if (ckt->CKTstate0)
                    memcpy(OldCKTstate0, ckt->CKTstate0,
                           (size_t) ckt->CKTnumStates * sizeof(double));
            }
#else
            if (!OldCKTstate0)
                OldCKTstate0 = TMALLOC(double, ckt->CKTnumStates + 1);
            if (ckt->CKTstate0)
                memcpy(OldCKTstate0, ckt->CKTstate0,
                       (size_t) ckt->CKTnumStates * sizeof(double));
#endif

#if defined(KLU) && NGSPICE_OPT_GENERAL
            if (!used_chord) {
                startTime = SPfrontEnd->IFseconds();
                SMPsolve(ckt->CKTmatrix, ckt->CKTrhs, ckt->CKTrhsSpare);
                ckt->CKTstat->STATsolveTime +=
                    SPfrontEnd->IFseconds() - startTime;
                if (!chord_disabled && ckt->CKTkluMODE && (ckt->CKTmode & MODETRAN)) {
                    int nmax = ckt->CKTmaxEqNum;
                    const double * restrict rn = ckt->CKTrhs;
                    const double * restrict ro = ckt->CKTrhsOld;
                    double mx = 0.0;
                    for (int k = 1; k < nmax; k++) {
                        double d = fabs(rn[k] - ro[k]);
                        if (d > mx) mx = d;
                    }
                    prev_dx_norm = mx;
                    if (mx > step_max_dx) step_max_dx = mx;
                }
            }
        after_lin_solve:
#else
            startTime = SPfrontEnd->IFseconds();
            SMPsolve(ckt->CKTmatrix, ckt->CKTrhs, ckt->CKTrhsSpare);
            ckt->CKTstat->STATsolveTime +=
                SPfrontEnd->IFseconds() - startTime;
#endif
#ifdef STEPDEBUG
            /*XXXX*/
            if (ckt->CKTrhs[0] != 0.0)
                printf("NIiter: CKTrhs[0] = %g\n", ckt->CKTrhs[0]);
            if (ckt->CKTrhsSpare[0] != 0.0)
                printf("NIiter: CKTrhsSpare[0] = %g\n", ckt->CKTrhsSpare[0]);
            if (ckt->CKTrhsOld[0] != 0.0)
                printf("NIiter: CKTrhsOld[0] = %g\n", ckt->CKTrhsOld[0]);
            /*XXXX*/
#endif
            ckt->CKTrhs[0] = 0;
            ckt->CKTrhsSpare[0] = 0;
            ckt->CKTrhsOld[0] = 0;

            if (iterno > maxIter) {
                ckt->CKTstat->STATnumIter += iterno;
                /* we don't use this info during transient analysis */
                if (ckt->CKTcurrentAnalysis != DOING_TRAN) {
                    FREE(errMsg);
                    errMsg = copy("Too many iterations without convergence");
#ifdef STEPDEBUG
                    fprintf(stderr, "too many iterations without convergence: %d iter's (max iter == %d)\n",
                    iterno, maxIter);
#endif
                    FREE(errMsg);
                }
#if NGSPICE_OPT_GENERAL
                ckt->CKTtaLastIter = iterno;
#endif
                FREE(OldCKTstate0);
                return(E_ITERLIM);
            }

            if ((ckt->CKTnoncon == 0) && (iterno != 1))
                ckt->CKTnoncon = NIconvTest(ckt);
            else
                ckt->CKTnoncon = 1;

#ifdef STEPDEBUG
            printf("noncon is %d\n", ckt->CKTnoncon);
#endif
        }

        if ((ckt->CKTnodeDamping != 0) && (ckt->CKTnoncon != 0) &&
            ((ckt->CKTmode & MODETRANOP) || (ckt->CKTmode & MODEDCOP)) &&
            (iterno > 1))
        {
            CKTnode *node;
            double diff, maxdiff = 0;
            for (node = ckt->CKTnodes->next; node; node = node->next)
                if (node->type == SP_VOLTAGE) {
                    diff = fabs(ckt->CKTrhs[node->number] - ckt->CKTrhsOld[node->number]);
                    if (maxdiff < diff)
                        maxdiff = diff;
                }

            if (maxdiff > 10) {
                double damp_factor = 10 / maxdiff;
                if (damp_factor < 0.1)
                    damp_factor = 0.1;
                for (node = ckt->CKTnodes->next; node; node = node->next) {
                    diff = ckt->CKTrhs[node->number] - ckt->CKTrhsOld[node->number];
                    ckt->CKTrhs[node->number] =
                        ckt->CKTrhsOld[node->number] + (damp_factor * diff);
                }
                for (i = 0; i < ckt->CKTnumStates; i++) {
                    diff = ckt->CKTstate0[i] - OldCKTstate0[i];
                    ckt->CKTstate0[i] = OldCKTstate0[i] + (damp_factor * diff);
                }
            }
        }

        if (ckt->CKTmode & MODEINITFLOAT) {
            if ((ckt->CKTmode & MODEDC) && ckt->CKThadNodeset) {
                if (ipass)
                    ckt->CKTnoncon = ipass;
                ipass = 0;
            }
            if (ckt->CKTnoncon == 0) {
#if defined(KLU) && NGSPICE_OPT_GENERAL
                ckt->CKTkluLastStepSmooth = (iterno <= 4 && step_max_dx < 0.05);
#endif
#if NGSPICE_OPT_GENERAL
                ckt->CKTtaLastIter = iterno;
#endif
                ckt->CKTstat->STATnumIter += iterno;
                FREE(OldCKTstate0);
                return(OK);
            }
        } else if (ckt->CKTmode & MODEINITJCT) {
            ckt->CKTmode = (ckt->CKTmode & ~INITF) | MODEINITFIX;
            ckt->CKTniState |= NISHOULDREORDER;
        } else if (ckt->CKTmode & MODEINITFIX) {
            if (ckt->CKTnoncon == 0)
                ckt->CKTmode = (ckt->CKTmode & ~INITF) | MODEINITFLOAT;
            ipass = 1;
        } else if (ckt->CKTmode & MODEINITSMSIG) {
            ckt->CKTmode = (ckt->CKTmode & ~INITF) | MODEINITFLOAT;
        } else if (ckt->CKTmode & MODEINITTRAN) {
            if (iterno <= 1)
                ckt->CKTniState |= NISHOULDREORDER;
            ckt->CKTmode = (ckt->CKTmode & ~INITF) | MODEINITFLOAT;
        } else if (ckt->CKTmode & MODEINITPRED) {
            ckt->CKTmode = (ckt->CKTmode & ~INITF) | MODEINITFLOAT;
        } else {
            ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
            printf("bad initf state \n");
#endif
            FREE(OldCKTstate0);
            return(E_INTERN);
            /* impossible - no such INITF flag! */
        }

        /* build up the lvnim1 array from the lvn array */
        SWAP(double *, ckt->CKTrhs, ckt->CKTrhsOld);
        /* printf("after loading, after solving\n"); */
        /* CKTdump(ckt); */
    }
    /*NOTREACHED*/
}

void NIresetwarnmsg(void) {
    msgcount = 0;
}
