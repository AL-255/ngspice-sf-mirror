#ifndef ngspice_SMPDEFS_H
#define ngspice_SMPDEFS_H

/* Typedef removed by Francesco Lannutti (2012-02) to create the new SMPmatrix structure */
/*
typedef  struct MatrixFrame     SMPmatrix;
*/
typedef  struct MatrixFrame     MatrixFrame;
typedef  struct MatrixElement  *SMPelement;

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000  AlansFixes
**********/

#include <stdio.h>
#include <math.h>
#include "ngspice/complex.h"

#ifdef KLU
#include "ngspice/klu.h"
#include "ngspice/spmatrix.h"
#endif

/* SMPmatrix structure - Francesco Lannutti (2012-02) */
typedef struct sSMPmatrix {
    MatrixFrame *SPmatrix ;                /* pointer to sparse matrix */

#ifdef KLU
    KLUmatrix *SMPkluMatrix ;              /* KLU Pointer to the KLU Matrix Data Structure (only for CIDER, for the moment) */
    unsigned int CKTkluMODE:1 ;            /* KLU MODE parameter to enable KLU or not from the heuristic */
    #define CKTkluON 1                     /* KLU MODE ON definition */
    #define CKTkluOFF 0                    /* KLU MODE OFF definition */
    double CKTkluMemGrowFactor ;           /* KLU Memory Grow Factor - default = 1.2 */
#endif

} SMPmatrix ;


#ifdef KLU
void spDeterminant_KLU (SMPmatrix *, int *, double *, double *) ;
void SMPconvertCOOtoCSC (SMPmatrix *) ;
int  SMPkluEnsureTls (KLUmatrix *, int, int) ;

#ifdef CIDER
void SMPsolveKLUforCIDER (SMPmatrix *, double [], double [], double [], double []) ;
int SMPreorderKLUforCIDER (SMPmatrix *) ;
double *SMPmakeEltKLUforCIDER (SMPmatrix *, int, int) ;
void SMPclearKLUforCIDER (SMPmatrix *) ;
void SMPconvertCOOtoCSCKLUforCIDER (SMPmatrix *) ;
void SMPdestroyKLUforCIDER (SMPmatrix *) ;
int SMPnewMatrixKLUforCIDER (SMPmatrix *, int, unsigned int) ;
int SMPluFacKLUforCIDER (SMPmatrix *) ;
void SMPprintKLUforCIDER (SMPmatrix *, char *) ;
#endif

#else
int SMPaddElt (SMPmatrix *, int, int, double) ;
#endif

double * SMPmakeElt( SMPmatrix * , int , int );
void SMPcClear( SMPmatrix *);
void SMPclear( SMPmatrix *);
int SMPcLUfac( SMPmatrix *, double );
int SMPluFac( SMPmatrix *, double , double );
int SMPcReorder( SMPmatrix * , double , double , int *);
int SMPreorder( SMPmatrix * , double , double , double );
void SMPcaSolve(SMPmatrix *Matrix, double RHS[], double iRHS[],
		double Spare[], double iSpare[]);
void SMPcSolve( SMPmatrix *, double [], double [], double [], double []);
void SMPsolve( SMPmatrix *, double [], double []);
#if defined(KLU) && NGSPICE_OPT_GENERAL
int SMPkluChordSolve(SMPmatrix *Matrix, const double RHS[], const double Xold[],
                     double Gmin, double *dx_norm_out);
void SMPkluChordCommit(SMPmatrix *Matrix, double RHS[], const double Xold[]);
void SMPkluSubExpand1Hop(SMPmatrix *Matrix, unsigned char *is_fast, int n_rhs,
                         const unsigned char *is_rail);
int  SMPkluSubPrepare(SMPmatrix *Matrix, const unsigned char *is_fast, int n_rhs,
                      const unsigned char *is_rail, const int *rail_nodes, int n_rails,
                      int max_nF, int *n_fast_out, const int **fast_rhs_out,
                      int *same_as_prev_out,
                      unsigned long long *cache_hits_out,
                      unsigned long long *cache_misses_out);
void SMPkluSubInitLoad(SMPmatrix *Matrix, double *ax_main, double *rhs_main,
                       const double *res_base, const double *cap_unit,
                       const double *cap_rhs, const double *lat_dc,
                       const double *lat_cap, const double *lat_rhs_dc,
                       const double *lat_rhs_cap, double ag0);
int  SMPkluSubSolve(SMPmatrix *Matrix, double RHS[], const double Xold[],
                    int rhs_len, double Gmin, double ag0, int iterno,
                    double prev_dx_norm, int chord_disabled,
                    double (*sec_fn)(void),
                    int *used_chord_out, double *dx_norm_out,
                    double *decomp_time_out, double *solve_time_out,
                    int *lu_valid_io, double *lu_ag0_io, double *lu_gmin_io,
                    int *chord_cnt_io);
int  SMPkluSubCheckWake(SMPmatrix *Matrix, const double *rhs_now,
                        const double *v_macro, double v_wake);
void SMPkluSubRailUpdate(SMPmatrix *Matrix, const double *ax_main,
                         const double *rhs_main, double *x_sol,
                         const int *rail_nodes, const int *rail_br,
                         const double *rail_sgn, int n_rails);
#endif
int SMPmatSize( SMPmatrix *);
int SMPnewMatrix( SMPmatrix *, int );
void SMPdestroy( SMPmatrix *);
int SMPpreOrder( SMPmatrix *);
void SMPprint( SMPmatrix * , char *);
void SMPprintRHS( SMPmatrix * , char *, double*, double*);
void SMPgetError( SMPmatrix *, int *, int *);
int SMPcProdDiag( SMPmatrix *, SPcomplex *, int *);
int SMPcDProd(SMPmatrix *Matrix, SPcomplex *pMantissa, int *pExponent);
SMPelement * SMPfindElt( SMPmatrix *, int , int , int );
int SMPcZeroCol(SMPmatrix *Matrix, int Col);
int SMPcAddCol(SMPmatrix *Matrix, int Accum_Col, int Addend_Col);
int SMPzeroRow(SMPmatrix *Matrix, int Row);
void SMPconstMult(SMPmatrix *, double);
void SMPmultiply(SMPmatrix *, double *, double *, double *, double *);

#ifdef CIDER
void SMPcSolveForCIDER (SMPmatrix *, double [], double [], double [], double []) ;
int SMPluFacForCIDER (SMPmatrix *) ;
int SMPnewMatrixForCIDER (SMPmatrix *, int, int) ;
void SMPsolveForCIDER (SMPmatrix *, double [], double []) ;
#endif

#endif

