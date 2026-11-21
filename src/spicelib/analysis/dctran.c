/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000  AlansFixes
Modified: 2023 XSPICE breakpoint fix for shared ngspice by Vyacheslav Shevchuk
**********/

/* subroutine to do DC TRANSIENT analysis */


/* line   goto label:
   373    nextTime
   515    resume
   786    past_breakpoint
   985    chkStep */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "cktaccept.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/fteext.h"
#include "ngspice/missing_math.h"
#if defined(KLU) && NGSPICE_OPT_GENERAL
#include "ngspice/devdefs.h"
#include "vsrc/vsrcdefs.h"
#include "isrc/isrcdefs.h"
#include "dio/diodefs.h"
#ifdef USE_OMP
#include <omp.h>
#endif
#endif

/* for setting breakpoints required by dbs data base */
extern struct dbcomm *dbs;
#include "ngspice/ftedebug.h"

#ifdef XSPICE
#include "ngspice/miftypes.h"
#include "ngspice/evt.h"
#include "ngspice/enh.h"
#include "ngspice/mif.h"
#include "ngspice/evtproto.h"
#include "ngspice/ipctiein.h"
#endif

#ifdef SHARED_MODULE
extern int add_bkpt(void);
extern int sharedsync(double*, double*, double, double, double, int, int*, int);
extern int ng_ident;      /* for debugging */
#endif

#define INIT_STATS() \
do { \
    startTime = SPfrontEnd->IFseconds();        \
    startIters = ckt->CKTstat->STATnumIter;     \
    startdTime = ckt->CKTstat->STATdecompTime;  \
    startsTime = ckt->CKTstat->STATsolveTime;   \
    startlTime = ckt->CKTstat->STATloadTime;    \
    startkTime = ckt->CKTstat->STATsyncTime;    \
} while(0)

#define UPDATE_STATS(analysis) \
do { \
    ckt->CKTcurrentAnalysis = analysis; \
    ckt->CKTstat->STATtranTime += SPfrontEnd->IFseconds() - startTime; \
    ckt->CKTstat->STATtranIter += ckt->CKTstat->STATnumIter - startIters; \
    ckt->CKTstat->STATtranDecompTime += ckt->CKTstat->STATdecompTime - startdTime; \
    ckt->CKTstat->STATtranSolveTime += ckt->CKTstat->STATsolveTime - startsTime; \
    ckt->CKTstat->STATtranLoadTime += ckt->CKTstat->STATloadTime - startlTime; \
    ckt->CKTstat->STATtranSyncTime += ckt->CKTstat->STATsyncTime - startkTime; \
} while(0)

#if NGSPICE_OPT_GENERAL
static int
ta_get_env_int(const char *name, int defval)
{
    const char *s = getenv(name);
    if (!s || !*s)
        return defval;
    return atoi(s);
}

static double
ta_get_env_double(const char *name, double defval)
{
    const char *s = getenv(name);
    if (!s || !*s)
        return defval;
    char *end = NULL;
    double v = strtod(s, &end);
    if (end == s)
        return defval;
    return v;
}

static void
ta_ensure_sol_buffers(CKTcircuit *ckt, int size)
{
    int need = size + 1;
    if (ckt->CKTtaSolSize < need) {
        for (int k = 0; k < 4; k++) {
            FREE(ckt->CKTtaSol[k]);
            ckt->CKTtaSol[k] = TMALLOC(double, need);
        }
        FREE(ckt->CKTtaPredVec);
        ckt->CKTtaPredVec = TMALLOC(double, need);
        ckt->CKTtaSolSize = need;
        ckt->CKTtaSolCount = 0;
    }
    if (!ckt->CKTnodeIsVolt || ckt->CKTnodeIsVoltSize != size) {
        FREE(ckt->CKTnodeIsVolt);
        ckt->CKTnodeIsVolt = TMALLOC(unsigned char, need);
        ckt->CKTnodeIsVoltSize = size;
        ckt->CKTnodeIsVolt[0] = 0;
        CKTnode *nd = ckt->CKTnodes;
        for (int k = 1; k <= size && nd && nd->next; k++) {
            nd = nd->next;
            ckt->CKTnodeIsVolt[k] = (nd->type == SP_VOLTAGE) ? 1 : 0;
        }
    }
}

static void
ta_sol_save(CKTcircuit *ckt, const double *sol, double dt_accepted)
{
    int size = SMPmatSize(ckt->CKTmatrix);
    if (size <= 0 || !sol)
        return;
    ta_ensure_sol_buffers(ckt, size);
    double *tmp = ckt->CKTtaSol[3];
    ckt->CKTtaSol[3] = ckt->CKTtaSol[2];
    ckt->CKTtaSol[2] = ckt->CKTtaSol[1];
    ckt->CKTtaSol[1] = ckt->CKTtaSol[0];
    ckt->CKTtaSol[0] = tmp;
    memcpy(ckt->CKTtaSol[0], sol, (size_t)(size + 1) * sizeof(double));
    ckt->CKTtaSol[0][0] = 0.0;
    if (ckt->CKTtaSolCount > 0 && dt_accepted > 0.0) {
        ckt->CKTtaSolDt[2] = ckt->CKTtaSolDt[1];
        ckt->CKTtaSolDt[1] = ckt->CKTtaSolDt[0];
        ckt->CKTtaSolDt[0] = dt_accepted;
    }
    if (ckt->CKTtaSolCount < 4)
        ckt->CKTtaSolCount++;
}

static void
ta_pred_apply(CKTcircuit *ckt, int pred_mode)
{
    int size = SMPmatSize(ckt->CKTmatrix);
    if (size <= 0 || ckt->CKTtaSolCount <= 0 || !ckt->CKTtaSol[0])
        return;
    ta_ensure_sol_buffers(ckt, size);

    double * restrict rhsOld = ckt->CKTrhsOld;
    double * restrict pred   = ckt->CKTtaPredVec;
    const double * restrict x0 = ckt->CKTtaSol[0];
    const unsigned char * restrict isVolt = ckt->CKTnodeIsVolt;
    const int sub_only = (ckt->CKTmrSubActive && ckt->CKTmrIsFast && ckt->CKTmrSubVlen >= size);
    const unsigned char * restrict isFast = sub_only ? ckt->CKTmrIsFast : NULL;
    const double h = ckt->CKTdelta;
    const double h1 = ckt->CKTtaSolDt[0];
    const double h2 = ckt->CKTtaSolDt[1];

    pred[0] = rhsOld[0] = 0.0;

    /* 0th-order hold at t=0 or right after a breakpoint discontinuity */
    if (ckt->CKTtaSolCount == 1 || h1 <= 0.0) {
        memcpy(pred, x0, (size_t)(size + 1) * sizeof(double));
        memcpy(rhsOld, x0, (size_t)(size + 1) * sizeof(double));
        return;
    }

    const double * restrict x1 = ckt->CKTtaSol[1];
    const double r1 = h / h1;

    /* 1st-order slope-limited linear predictor */
    if (pred_mode == 1 || ckt->CKTtaSolCount == 2 || ckt->CKTorder == 1 || h2 <= 0.0) {
        const double r_eff = (pred_mode == 3) ? r1 : ((r1 > 1.5) ? 1.5 : r1);
        for (int k = 1; k <= size; k++) {
            if (isFast && !isFast[k]) {
                pred[k] = rhsOld[k] = x0[k];
                continue;
            }
            double dx = r_eff * (x0[k] - x1[k]);
            if (pred_mode != 3 && isVolt[k]) {
                if (dx > 0.35) dx = 0.35;
                else if (dx < -0.35) dx = -0.35;
            }
            double v = x0[k] + dx;
            pred[k] = v;
            rhsOld[k] = v;
        }
        return;
    }

    /* 2nd-order Lagrange / Newton divided-difference predictor (monotone slope-limited for mode 2) */
    const double * restrict x2 = ckt->CKTtaSol[2];
    const double inv_h1 = 1.0 / h1;
    const double inv_h2 = 1.0 / h2;
    const double c_quad = h * (h + h1) / (h1 + h2);

    if (pred_mode == 3) {
        /* Unclamped exact 2nd-order Lagrange (ablation control) */
        for (int k = 1; k <= size; k++) {
            if (isFast && !isFast[k]) {
                pred[k] = rhsOld[k] = x0[k];
                continue;
            }
            double s0 = (x0[k] - x1[k]) * inv_h1;
            double s1 = (x1[k] - x2[k]) * inv_h2;
            double v = x0[k] + h * s0 + c_quad * (s0 - s1);
            pred[k] = v;
            rhsOld[k] = v;
        }
        return;
    }

    if (r1 > 2.0) {
        /* Large step jump: fall back to clamped linear to prevent quadratic amplification */
        const double r_eff = 1.5;
        for (int k = 1; k <= size; k++) {
            if (isFast && !isFast[k]) {
                pred[k] = rhsOld[k] = x0[k];
                continue;
            }
            double dx = r_eff * (x0[k] - x1[k]);
            if (isVolt[k]) {
                if (dx > 0.35) dx = 0.35;
                else if (dx < -0.35) dx = -0.35;
            }
            double v = x0[k] + dx;
            pred[k] = v;
            rhsOld[k] = v;
        }
        return;
    }

    const double r_entry = (r1 > 1.0) ? 0.5 : (0.5 * r1);
    for (int k = 1; k <= size; k++) {
        if (isFast && !isFast[k]) {
            pred[k] = rhsOld[k] = x0[k];
            continue;
        }
        double d0 = x0[k] - x1[k];
        double d1 = x1[k] - x2[k];
        if (d0 == 0.0 && d1 == 0.0) {
            pred[k] = x0[k];
            rhsOld[k] = x0[k];
            continue;
        }
        double s0 = d0 * inv_h1;
        double s1 = d1 * inv_h2;
        double dx_lin = h * s0;
        double dx;
        if (s0 * s1 > 0.0) {
            double dx_q = dx_lin + c_quad * (s0 - s1);
            double as0 = fabs(s0), as1 = fabs(s1);
            if (as0 >= as1) {
                /* Accelerating: cap quadratic boost to 1.5x linear */
                double lim = 1.5 * fabs(dx_lin);
                if (fabs(dx_q) > lim)
                    dx = 1.5 * dx_lin;
                else
                    dx = dx_q;
            } else if (as0 < 0.25 * as1) {
                /* Rapidly flattening (e.g. CMOS edge reaching VDD/VSS): stay in [0, dx_lin] */
                if (dx_q * dx_lin < 0.0)
                    dx = 0.0;
                else if (fabs(dx_q) > fabs(dx_lin))
                    dx = dx_lin;
                else
                    dx = dx_q;
            } else {
                /* Smooth deceleration (e.g. sine approaching extremum): allow gentle turnaround */
                if (dx_lin > 0.0) {
                    double lo = -0.25 * dx_lin;
                    dx = (dx_q < lo) ? lo : ((dx_q > dx_lin) ? dx_lin : dx_q);
                } else {
                    double hi = -0.25 * dx_lin;
                    dx = (dx_q > hi) ? hi : ((dx_q < dx_lin) ? dx_lin : dx_q);
                }
            }
        } else if (s1 == 0.0) {
            /* Switching edge starting from flat rail: gentle entry */
            dx = r_entry * d0;
        } else {
            /* Local extremum / sign reversal: damped linear */
            dx = 0.25 * dx_lin;
        }
        if (isVolt[k]) {
            if (dx > 0.35) dx = 0.35;
            else if (dx < -0.35) dx = -0.35;
        }
        double v = x0[k] + dx;
        pred[k] = v;
        rhsOld[k] = v;
    }
}
#endif

#if defined(KLU) && NGSPICE_OPT_GENERAL
typedef struct {
    int pos;
    int neg;
    int br;
} MrCfVsrcTriple;

typedef struct {
    int enabled;
    int M;
    double v_mr;
    double v_wake;
    double max_frac;
    int n_rhs;
    unsigned char *is_rail;
    unsigned char *always_fast;
    unsigned char *quiet_streak;
    MrCfVsrcTriple *triples;
    int n_triples;
    int n_rails;
    int rail_node[8];
    int rail_br[8];
    double rail_sgn[8];
    int vprev_init;
    unsigned long long stat_full_acc;
    unsigned long long stat_micro_acc;
    unsigned long long stat_bursts;
    unsigned long long stat_nf_sum;
    unsigned long long stat_cache_hits;
    unsigned long long stat_cache_misses;
    unsigned long long stat_wake_exits;
    unsigned long long stat_fallback_exits;
} MrCfState;

static void
mr_cf_init(MrCfState *cf, CKTcircuit *ckt)
{
    memset(cf, 0, sizeof(*cf));
    ckt->CKTmrSubActive = 0;
    ckt->CKTmrSubBurstLen = 0;
    ckt->CKTmrSubNeedSync = 0;
    ckt->CKTmrSubNFast = 0;
    ckt->CKTmrFastNodes = NULL;

    const char *no_s = getenv("NGSPICE_NO_MR_COMPFAST");
    if (no_s && no_s[0] != '\0' && no_s[0] != '0')
        return;

    int min_n = 2500;
    const char *min_n_s = getenv("NGSPICE_MR_CF_MIN_N");
    if (min_n_s && min_n_s[0] != '\0') {
        int v = atoi(min_n_s);
        if (v >= 64) min_n = v;
    }
    const char *en_s = getenv("NGSPICE_MR_COMPFAST");
    if (en_s && en_s[0] != '\0') {
        if (en_s[0] == '0')
            return;
        min_n = 64; /* explicit opt-in overrides auto N >= 2500 threshold */
    }

    int nthreads = 1;
#ifdef USE_OMP
    nthreads = omp_get_max_threads();
#endif
    if (!ckt->CKTkluMODE || !ckt->CKTmatrix || !ckt->CKTmatrix->SMPkluMatrix ||
        ckt->CKTmatrix->SMPkluMatrix->KLUmatrixIsComplex != KLUmatrixReal ||
        ckt->CKTmatrix->SMPkluMatrix->KLUmatrixN < min_n ||
        nthreads != 1 || ckt->CKTsenInfo != NULL)
    {
        return;
    }
#ifdef XSPICE
    if (ckt->evt && ckt->evt->counts.num_insts > 0)
        return;
#endif

    /* Allow circuits whose stateful devices are Capacitor + BSIM4v5/v6 */
    for (int di = 0; di < DEVmaxnum; di++) {
        if (!DEVices[di] || !ckt->CKThead[di]) continue;
        const char *nm = DEVices[di]->DEVpublic.name;
        if (!nm) return;
        if (strcmp(nm, "Resistor") == 0 ||
            strcmp(nm, "Capacitor") == 0 ||
            strcmp(nm, "BSIM4v5") == 0 ||
            strcmp(nm, "BSIM4v6") == 0 ||
            strcmp(nm, "Diode") == 0 ||
            strcmp(nm, "Vsource") == 0 ||
            strcmp(nm, "Isource") == 0)
            continue;
        /* Reject unknown stateful devices (inductors, LTRA, other MOSFETs, etc.) */
        if (DEVices[di]->DEVtrunc != NULL)
            return;
    }

    int n_rhs = (int)ckt->CKTmatrix->SMPkluMatrix->KLUmatrixNrhs;
    if (n_rhs < ckt->CKTmaxEqNum) n_rhs = ckt->CKTmaxEqNum;
    if (n_rhs <= 2) return;

    int is_vlsi = (ckt->CKTmatrix->SMPkluMatrix->KLUmatrixN >= 10000);
    cf->M = is_vlsi ? 16 : 8;
    if (is_vlsi) {
        double vt = (ckt->CKTvoltTol > 0.0) ? ckt->CKTvoltTol : 1.0e-6;
        cf->v_mr = (5.0 * vt > 2.0e-6) ? (5.0 * vt) : 2.0e-6;
        cf->v_wake = (50.0 * vt > 5.0e-5) ? (50.0 * vt) : 5.0e-5;
    } else {
        cf->v_mr = 2.0e-6;
        cf->v_wake = 5.0e-5;
    }
    cf->max_frac = 0.45;
    const char *s;
    if ((s = getenv("NGSPICE_MR_CF_M")) != NULL && s[0]) {
        int m = atoi(s);
        if (m >= 2 && m <= 64) cf->M = m;
    }
    if ((s = getenv("NGSPICE_MR_CF_VTOL")) != NULL && s[0]) {
        double v = atof(s);
        if (v > 0.0) cf->v_mr = v;
    }
    if ((s = getenv("NGSPICE_MR_CF_WAKE")) != NULL && s[0]) {
        double v = atof(s);
        if (v > 0.0) cf->v_wake = v;
    }
    if ((s = getenv("NGSPICE_MR_CF_MAXFRAC")) != NULL && s[0]) {
        double f = atof(s);
        if (f > 0.05 && f <= 0.95) cf->max_frac = f;
    }

    cf->n_rhs = n_rhs;
    cf->is_rail      = (unsigned char *)calloc((size_t)(n_rhs + 1), sizeof(unsigned char));
    cf->always_fast  = (unsigned char *)calloc((size_t)(n_rhs + 1), sizeof(unsigned char));
    cf->quiet_streak = (unsigned char *)calloc((size_t)(n_rhs + 1), sizeof(unsigned char));
    if (!cf->is_rail || !cf->always_fast || !cf->quiet_streak) {
        free(cf->is_rail); free(cf->always_fast); free(cf->quiet_streak);
        memset(cf, 0, sizeof(*cf));
        return;
    }

    if (ckt->CKTmrSubVlen < n_rhs) {
        FREE(ckt->CKTmrIsFast);
        FREE(ckt->CKTmrVprev1);
        FREE(ckt->CKTmrVprev2);
        FREE(ckt->CKTmrVmacro);
        ckt->CKTmrIsFast = TMALLOC(unsigned char, n_rhs + 1);
        ckt->CKTmrVprev1 = TMALLOC(double, n_rhs + 1);
        ckt->CKTmrVprev2 = TMALLOC(double, n_rhs + 1);
        ckt->CKTmrVmacro = TMALLOC(double, n_rhs + 1);
        ckt->CKTmrSubVlen = n_rhs;
    }
    memset(ckt->CKTmrIsFast, 0, (size_t)(n_rhs + 1));
    memset(ckt->CKTmrVprev1, 0, (size_t)(n_rhs + 1) * sizeof(double));
    memset(ckt->CKTmrVprev2, 0, (size_t)(n_rhs + 1) * sizeof(double));
    memset(ckt->CKTmrVmacro, 0, (size_t)(n_rhs + 1) * sizeof(double));

    /* Scan Vsource instances for grounded DC rails, floating DC triples, and time-varying sources */
    int vcode = CKTtypelook("Vsource");
    int tcap = 0;
    if (vcode >= 0 && ckt->CKThead[vcode]) {
        for (VSRCmodel *m = (VSRCmodel *)ckt->CKThead[vcode]; m; m = VSRCnextModel(m)) {
            for (VSRCinstance *h = VSRCinstances(m); h; h = VSRCnextInstance(h)) {
                int p = h->VSRCposNode;
                int n = h->VSRCnegNode;
                int br = h->VSRCbranch;
                int is_dc = (h->VSRCfunctionType == 0 &&
                             h->VSRCtrnoise_state == NULL &&
                             h->VSRCtrrandom_state == NULL);
                if (is_dc && ((p > 0 && n == 0) || (n > 0 && p == 0))) {
                    int u = (p > 0) ? p : n;
                    double sgn = (p > 0) ? 1.0 : -1.0;
                    if (u < n_rhs) cf->is_rail[u] = 1;
                    if (br > 0 && br < n_rhs) cf->is_rail[br] = 1;
                    if (cf->n_rails < 8 && u < n_rhs && br > 0 && br < n_rhs) {
                        cf->rail_node[cf->n_rails] = u;
                        cf->rail_br[cf->n_rails] = br;
                        cf->rail_sgn[cf->n_rails] = sgn;
                        cf->n_rails++;
                    }
                } else {
                    if (!is_dc) {
                        if (p > 0 && p < n_rhs) cf->always_fast[p] = 1;
                        if (n > 0 && n < n_rhs) cf->always_fast[n] = 1;
                        if (br > 0 && br < n_rhs) cf->always_fast[br] = 1;
                    }
                    if (cf->n_triples >= tcap) {
                        tcap = (tcap == 0) ? 16 : tcap * 2;
                        cf->triples = (MrCfVsrcTriple *)realloc(cf->triples, (size_t)tcap * sizeof(MrCfVsrcTriple));
                    }
                    if (cf->triples) {
                        cf->triples[cf->n_triples].pos = p;
                        cf->triples[cf->n_triples].neg = n;
                        cf->triples[cf->n_triples].br  = br;
                        cf->n_triples++;
                    }
                }
            }
        }
    }

    /* Scan Isource instances for time-varying current sources */
    int icode = CKTtypelook("Isource");
    if (icode >= 0 && ckt->CKThead[icode]) {
        for (ISRCmodel *m = (ISRCmodel *)ckt->CKThead[icode]; m; m = ISRCnextModel(m)) {
            for (ISRCinstance *h = ISRCinstances(m); h; h = ISRCnextInstance(h)) {
                int is_dc = (h->ISRCfunctionType == 0 &&
                             h->ISRCtrnoise_state == NULL &&
                             h->ISRCtrrandom_state == NULL);
                if (!is_dc) {
                    if (h->ISRCposNode > 0 && h->ISRCposNode < n_rhs)
                        cf->always_fast[h->ISRCposNode] = 1;
                    if (h->ISRCnegNode > 0 && h->ISRCnegNode < n_rhs)
                        cf->always_fast[h->ISRCnegNode] = 1;
                }
            }
        }
    }

    /* Mark all Diode terminal nodes always_fast so antenna/ESD diodes stay in F */
    int dcode = CKTtypelook("Diode");
    if (dcode >= 0 && ckt->CKThead[dcode]) {
        for (DIOmodel *m = (DIOmodel *)ckt->CKThead[dcode]; m; m = DIOnextModel(m)) {
            for (DIOinstance *h = DIOinstances(m); h; h = DIOnextInstance(h)) {
                int dnodes[6] = {
                    h->DIOposNode, h->DIOnegNode, h->DIOposPrimeNode,
                    h->DIOposSwPrimeNode, h->DIOqpNode, h->DIOtempNode
                };
                for (int k = 0; k < 6; k++) {
                    int u = dnodes[k];
                    if (u > 0 && u < n_rhs)
                        cf->always_fast[u] = 1;
                }
            }
        }
    }

    /* Ensure grounded DC rails are never marked always_fast */
    for (int u = 1; u < n_rhs; u++) {
        if (cf->is_rail[u])
            cf->always_fast[u] = 0;
    }

    cf->enabled = 1;
}

static void
mr_cf_cleanup(MrCfState *cf, CKTcircuit *ckt)
{
    ckt->CKTmrSubActive = 0;
    ckt->CKTmrSubNeedSync = 0;
    ckt->CKTmrSubBurstLen = 0;
    ckt->CKTmrFastNodes = NULL;
    if (cf->enabled && (getenv("NGSPICE_MR_CF_STATS") != NULL || getenv("NGSPICE_MR_STATS") != NULL)) {
        double avg_nf = cf->stat_bursts ? ((double)cf->stat_nf_sum / (double)cf->stat_bursts) : 0.0;
        double pct = cf->n_rhs > 1 ? (100.0 * avg_nf / (double)(cf->n_rhs - 1)) : 0.0;
        fprintf(stderr,
                "[MR_COMPFAST] N=%d M=%d full=%llu micro=%llu (%.1f%%) bursts=%llu avg_nF=%.1f (%.1f%%) cache(hit=%llu miss=%llu) exits(wake=%llu fb=%llu)\n",
                cf->n_rhs - 1, cf->M, cf->stat_full_acc, cf->stat_micro_acc,
                (cf->stat_full_acc + cf->stat_micro_acc) ?
                    100.0 * (double)cf->stat_micro_acc / (double)(cf->stat_full_acc + cf->stat_micro_acc) : 0.0,
                cf->stat_bursts, avg_nf, pct,
                cf->stat_cache_hits, cf->stat_cache_misses,
                cf->stat_wake_exits, cf->stat_fallback_exits);
    }
    free(cf->is_rail);
    free(cf->always_fast);
    free(cf->quiet_streak);
    free(cf->triples);
    memset(cf, 0, sizeof(*cf));
}

static void
mr_cf_exit_burst(CKTcircuit *ckt)
{
    ckt->CKTmrSubActive = 0;
    ckt->CKTmrSubNeedSync = 1;
    ckt->CKTmrSubBurstLen = 0;
    if (ckt->CKTmatrix && ckt->CKTmatrix->CKTkluMODE && ckt->CKTmatrix->SMPkluMatrix)
        ckt->CKTmatrix->SMPkluMatrix->KLUmatrixRhsCapValid = 0;
}
#endif


int
DCtran(CKTcircuit *ckt,
       int restart)   /* forced restart flag */
{
    TRANan *job = (TRANan *) ckt->CKTcurJob;

    int i;
    double olddelta = 0.0;
    double delta;
    double newdelta;
    double *temp;
#if NGSPICE_OPT_GENERAL
    int ta_pred_mode = ta_get_env_int("NGSPICE_TA_PRED", 0);
    int ta_ctrl_mode = ta_get_env_int("NGSPICE_TA_CTRL", 0);
    int ta_pi_mode   = ta_get_env_int("NGSPICE_TA_PI", ta_ctrl_mode ? 2 : 0);
    double ta_db_hi  = ta_get_env_double("NGSPICE_TA_DB", ta_ctrl_mode ? 1.35 : 0.0);
    if (ta_db_hi > 0.0 && ta_db_hi <= 1.0)
        ta_db_hi = 1.35; /* NGSPICE_TA_DB=1 -> default 1.35 */
    int ta_nl_mode   = ta_get_env_int("NGSPICE_TA_NL", ta_ctrl_mode ? 1 : 0);
    int ta_bkpt_mode = ta_get_env_int("NGSPICE_TA_BKPT", 0);
    int ta_ord3_mode = ta_get_env_int("NGSPICE_TA_ORD3", 0);
    int ta_eff_maxord = ckt->CKTmaxOrder;
    /* PI / deadband / NL / breakpoint / order-3 state */
    int ta_pi_valid = 0;
    double ta_rho_prev = 1.0;
    double ta_h_ratio_prev = 1.0;
    int ta_just_rejected = 0;
    int ta_consec_noncon = 0;
    int ta_steps_since_bkpt = 100;
    int ta_steps_at_ord = 0;
#endif
    double startdTime;
    double startsTime;
    double startlTime;
    double startkTime;
    double startTime;
    int startIters;
    int converged;
    int firsttime;
    int error;
#ifdef WANT_SENSE2
    int save, save2, size;
    long save1;
#endif
    int save_order;
    long save_mode;
    IFuid timeUid;
    IFuid *nameList;
    int numNames;
    double maxstepsize = 0.0;

    bool have_autostop = FALSE, flag_autostop = FALSE;

    int ltra_num;
    CKTnode *node;
#ifdef XSPICE
    /* xspice_breakpoints_processed 0:
       XSPICE models didn't have breakpoints in [last_accepted_time, CKTtime].
       xspice_breakpoints_processed 1:
       convergence criteria are satisfied but XSPICE breakpoint(s) is in the
       time interval [last_accepted_time, CKTtime]. */
    int xspice_breakpoints_processed = 0;

#ifdef SHARED_MODULE
    double olddelta_for_shared_sync = 0.0;
#endif
#endif
#if defined SHARED_MODULE
    int redostep;
#endif
#if defined(KLU) && NGSPICE_OPT_GENERAL
    MrCfState cf;
    memset(&cf, 0, sizeof(cf));
#endif
    if(restart || ckt->CKTtime == 0) {
        /* dctran() is entered here upon starting transient simulation
           with time 0 and restart 1.
           ckt->CKTstep, CKTfinalTime, CKTinitTime, CKTmaxStep have been
           set already in fcn TRANinit() of traninit.c according to
           TSTEP TSTOP TSTART TMAX given on the .tran line. TMAX is set to TSTEP,
           if 'set nostepsizelimit' is not given in .spiceinit.*/

        /* Set the first delta (step) time, typically depending on TSTEP */
        delta=MIN(ckt->CKTfinalTime/100,ckt->CKTstep)/10;

#ifdef STEPDEBUG
        printf("delta = %g    finalTime/100: %g    CKTstep: %g\n",delta,ckt->CKTfinalTime/100,ckt->CKTstep);
#endif
        /* begin LTRA code addition */
        if (ckt->CKTtimePoints != NULL)
            FREE(ckt->CKTtimePoints);

        if (ckt->CKTstep >= ckt->CKTmaxStep)
            maxstepsize = ckt->CKTstep;
        else
            maxstepsize = ckt->CKTmaxStep;

        ckt->CKTsizeIncr = 100;
        ckt->CKTtimeIndex = -1; /* before the DC soln has been stored */
        ckt->CKTtimeListSize = (int) ceil( ckt->CKTfinalTime / maxstepsize );
        ltra_num = CKTtypelook("LTRA");
        if (ltra_num >= 0 && ckt->CKThead[ltra_num] != NULL)
            ckt->CKTtimePoints = TMALLOC(double, ckt->CKTtimeListSize);
        /* end LTRA code addition */

        /* Reset the breakpoint list.
           Add breakpoints at 0 time and TSTOP (final) time */
        if(ckt->CKTbreaks) FREE(ckt->CKTbreaks);
        ckt->CKTbreaks = TMALLOC(double, 2);
        if(ckt->CKTbreaks == NULL) return(E_NOMEM);
        ckt->CKTbreaks[0] = 0;
        ckt->CKTbreaks[1] = ckt->CKTfinalTime;
        ckt->CKTbreakSize = 2;

#ifdef SHARED_MODULE
        add_bkpt();
#endif

#ifdef XSPICE
        /* Modify setting of CKTminBreak
           Set to 10 times delmin (minimum delta step time). */
        if(ckt->CKTminBreak == 0)
            ckt->CKTminBreak = 10.0 * ckt->CKTdelmin;

#else
        if(ckt->CKTminBreak == 0)
            ckt->CKTminBreak = ckt->CKTmaxStep * 5e-5;
#endif

#ifdef XSPICE
        /* set anal_init and anal_type */
        /* Tell the code models what mode we're in */
        g_mif_info.circuit.anal_type = MIF_DC;
        g_mif_info.circuit.anal_init = MIF_TRUE;
#endif
        /* Scan ckt->CKTnodes and create list of node names */
        error = CKTnames(ckt,&numNames,&nameList);
        if(error) return(error);
        SPfrontEnd->IFnewUid (ckt, &timeUid, NULL, "time", UID_OTHER, NULL);
        error = SPfrontEnd->OUTpBeginPlot (ckt, ckt->CKTcurJob,
                                           ckt->CKTcurJob->JOBname,
                                           timeUid, IF_REAL,
                                           numNames, nameList, IF_REAL,
                                           &(job->TRANplot));
        tfree(nameList);
        if(error) return(error);

        /* initialize CKTsoaCheck 'warn' counters */
        if (ckt->CKTsoaCheck)
            error = CKTsoaInit();

        ckt->CKTtime = 0;
        ckt->CKTdelta = 0;
        ckt->CKTbreak = 1;
        firsttime = 1;
#if NGSPICE_OPT_GENERAL
        ckt->CKTtaSolCount = 0;
        ckt->CKTtaPredActive = 0;
        ta_eff_maxord = ckt->CKTmaxOrder;
        if (ta_ord3_mode > 0 && ckt->CKTintegrateMethod == GEAR && ta_eff_maxord < 3) {
            ta_eff_maxord = 3;
            for (int si = 0; si <= ta_eff_maxord + 1; si++) {
                if (!ckt->CKTstates[si] && ckt->CKTnumStates > 0)
                    ckt->CKTstates[si] = TMALLOC(double, ckt->CKTnumStates);
            }
        }
        ta_steps_at_ord = 0;
#endif
        save_mode = (ckt->CKTmode&MODEUIC) | MODETRANOP | MODEINITJCT;
        save_order = ckt->CKTorder;

        have_autostop = cp_getvar("autostop", CP_BOOL, NULL, 0);

/* Add breakpoints here which have been requested by the user setting the
   stop command as 'stop when time = xx'.
   Get data from the global dbs data base.
*/
        if (dbs) {
            struct dbcomm *d;
            for (d = dbs; d; d = d->db_next)
                if ((d->db_type == DB_STOPWHEN) && cieq(d->db_nodename1,"time")
                    && (d->db_value2 > 0)) {
                    CKTsetBreak(ckt, d->db_value2);
                    if (ft_ngdebug)
                        printf("breakpoint set to time = %g\n", d->db_value2);
                }
        }

#ifdef XSPICE
        /* set a breakpoint at end of supply ramping time.
           Must do this after CKTtime is set to 0 above */
        if(ckt->enh->ramp.ramptime > 0.0)
            CKTsetBreak(ckt, ckt->enh->ramp.ramptime);

        /* Call EVTop if event-driven instances exist */
        if(ckt->evt->counts.num_insts != 0) {
            /* use new DCOP algorithm */
            converged = EVTop(ckt,
                        (ckt->CKTmode & MODEUIC) | MODETRANOP | MODEINITJCT,
                        (ckt->CKTmode & MODEUIC) | MODETRANOP | MODEINITFLOAT,
                        ckt->CKTdcMaxIter,
                        MIF_TRUE);
            EVTdump(ckt, IPC_ANAL_DCOP, 0.0);

            EVTop_save(ckt, MIF_FALSE, 0.0);
        } else
#endif
            /* Get operating point for analogue circuit */
            converged = CKTop(ckt,
                (ckt->CKTmode & MODEUIC) | MODETRANOP | MODEINITJCT,
                (ckt->CKTmode & MODEUIC) | MODETRANOP | MODEINITFLOAT,
                ckt->CKTdcMaxIter);

        if(converged != 0) {
            fprintf(stderr,"\nError: Finding the operating point for transient simulation failed \n");
            CKTncDump(ckt);
            fprintf(stderr,"\n");
            fflush(stderr);
        } else if (ckt->CKTmode & MODEUIC && !ft_ngdebug) {
            fprintf(stdout,"Operating point simulation skipped by 'uic',\n");
            fprintf(stdout,"  now using transient initial conditions.\n");
            fflush(stdout);
        } else if (!ft_noacctprint && !ft_noinitprint) {
            fprintf(stdout,"\nInitial Transient Solution\n");
            fprintf(stdout,"--------------------------\n\n");
            fprintf(stdout,"%-30s %15s\n", "Node", "Voltage");
            fprintf(stdout,"%-30s %15s\n", "----", "-------");
            for(node=ckt->CKTnodes->next;node;node=node->next) {
                if (strstr(node->name, "#branch") || !strchr(node->name, '#'))
                    fprintf(stdout,"%-30s %15g\n", node->name,
                                              ckt->CKTrhsOld[node->number]);
            }
            fprintf(stdout,"\n");
            fflush(stdout);
        }

        /* return upon failure to converge during op */
        if (converged != 0) {
            SPfrontEnd->OUTendPlot(job->TRANplot);
            return(converged);
        }
#ifdef XSPICE
        /* set anal_init and anal_type */
        g_mif_info.circuit.anal_init = MIF_TRUE;

        /* Tell the code models what mode we're in */
        g_mif_info.circuit.anal_type = MIF_TRAN;

        /* Breakpoint stuff */
        /* Initialize the temporary breakpoint variables to infinity */
        g_mif_info.breakpoint.current = 1.0e30;
        g_mif_info.breakpoint.last    = 1.0e30;
#endif
        ckt->CKTstat->STATtimePts ++;
        ckt->CKTorder = 1;
        /* Initialze CKTdeltaOld with maximum value */
        for(i=0;i<7;i++) {
            ckt->CKTdeltaOld[i]=ckt->CKTmaxStep;
        }
        ckt->CKTdelta = delta; /* delta set in line 130 */
#ifdef STEPDEBUG
        (void)printf("delta initialized to %g\n",ckt->CKTdelta);
#endif
        ckt->CKTsaveDelta = ckt->CKTfinalTime/50;

#ifdef WANT_SENSE2
        if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN)){
#ifdef SENSDEBUG
            printf("\nTransient Sensitivity Results\n\n");
            CKTsenPrint(ckt);
#endif /* SENSDEBUG */
            save = ckt->CKTsenInfo->SENmode;
            ckt->CKTsenInfo->SENmode = TRANSEN;
            save1 = ckt->CKTmode;
            save2 = ckt->CKTorder;
            ckt->CKTmode = save_mode;
            ckt->CKTorder = save_order;
            error = CKTsenDCtran(ckt);
            if (error)
                return(error);

            ckt->CKTmode = save1;
            ckt->CKTorder = save2;
        }
#endif
        /* modeinittran set here */
        ckt->CKTmode = (ckt->CKTmode&MODEUIC) | MODETRAN | MODEINITTRAN;
        /* Reset the gear variable coefficient matrix. */
        ckt->CKTag[0]=ckt->CKTag[1]=0;
        if (ckt->CKTstate1 && ckt->CKTstate0) {
            memcpy(ckt->CKTstate1, ckt->CKTstate0,
                  (size_t) ckt->CKTnumStates * sizeof(double));
        }

#ifdef WANT_SENSE2
        if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN)){
            size = SMPmatSize(ckt->CKTmatrix);
            for(i = 1; i<=size ; i++)
                ckt->CKTrhsOp[i] = ckt->CKTrhsOld[i];
        }
#endif

        INIT_STATS();
#if defined(KLU) && NGSPICE_OPT_GENERAL
        mr_cf_init(&cf, ckt);
#endif

    /* End of (restart || ckt->CKTtime == 0) */
    } else {
        /* traninit resets CKTmode */
        ckt->CKTmode = (ckt->CKTmode&MODEUIC) | MODETRAN | MODEINITPRED;

        INIT_STATS();
#if defined(KLU) && NGSPICE_OPT_GENERAL
        mr_cf_init(&cf, ckt);
#endif
        if(ckt->CKTminBreak == 0)
            ckt->CKTminBreak = ckt->CKTmaxStep * 5e-5;
        firsttime=0;
        /* To get rawfile working */
        error = SPfrontEnd->OUTpBeginPlot (NULL, NULL,
                                           NULL,
                                           NULL, 0,
                                           666, NULL, 666,
                                           &(job->TRANplot));
        if(error) {
            fprintf(stderr, "Couldn't relink rawfile\n");
            return error;
        }
        goto resume; /* line 515 */
    }

    nextTime:

    /* begin LTRA code addition */
    if (ckt->CKTtimePoints) {
    ckt->CKTtimeIndex++;
        if (ckt->CKTtimeIndex >= ckt->CKTtimeListSize) {
            /* need more space */
            int need;
            need = (int) ceil( (ckt->CKTfinalTime - ckt->CKTtime) / maxstepsize );
            if (need < ckt->CKTsizeIncr)
                need = ckt->CKTsizeIncr;
            ckt->CKTtimeListSize += need;
            ckt->CKTtimePoints = TREALLOC(double, ckt->CKTtimePoints, ckt->CKTtimeListSize);
            ckt->CKTsizeIncr = (int) ceil(1.4 * ckt->CKTsizeIncr);
        }
        ckt->CKTtimePoints[ckt->CKTtimeIndex] = ckt->CKTtime;
    }
    /* end LTRA code addition */

    error = CKTaccept(ckt);
    /* check if current breakpoint is outdated; if so, clear */
    if (ckt->CKTtime > ckt->CKTbreaks[0])
        CKTclrBreak(ckt);

    if (ckt->CKTsoaCheck)
        error = CKTsoaCheck(ckt);

    /*
 * Breakpoint handling scheme:
 * When a timepoint t is accepted (by CKTaccept), clear all previous
 * breakpoints, because they will never be needed again.
 *
 * t may itself be a breakpoint, or indistinguishably close. DON'T
 * clear t itself; recognise it as a breakpoint and act accordingly
 *
 * if t is not a breakpoint, limit the timestep so that the next
 * breakpoint is not crossed
 */

#ifdef STEPDEBUG
    printf("Delta %g accepted at time %g (finaltime: %g)\n",ckt->CKTdelta,ckt->CKTtime,ckt->CKTfinalTime);
    fflush(stdout);
#endif /* STEPDEBUG */
    ckt->CKTstat->STATaccepted ++;
    ckt->CKTbreak = 0;
    /* XXX Error will cause single process to bail. */
    if(error)  {
        UPDATE_STATS(DOING_TRAN);
        return(error);
    }

#ifdef XSPICE
    /*  Send evt data stuff in shared library */
    if (wantevtdata) {
        /* Send event-driven results */
        EVTdump(ckt, IPC_ANAL_TRAN, 0.0);
        /* Then follow with analog results... */
    }
#endif

    /* Output the data of the current accepted time point */
#if defined(KLU) && NGSPICE_OPT_GENERAL
    if (cf.enabled && ckt->CKTmrSubActive && cf.n_rails > 0) {
        SMPkluSubRailUpdate(ckt->CKTmatrix, ckt->CKTmatrix->SMPkluMatrix->KLUmatrixAx,
                            ckt->CKTrhs, ckt->CKTrhsOld,
                            cf.rail_node, cf.rail_br, cf.rail_sgn, cf.n_rails);
    }
#endif
#if NGSPICE_OPT_GENERAL
    if (ta_pred_mode > 0) {
        const double *acc_sol = (ckt->CKTtime == 0.0 || ckt->CKTmrSubActive) ? ckt->CKTrhsOld : ckt->CKTrhs;
        ta_sol_save(ckt, acc_sol, (ckt->CKTtime == 0.0) ? 0.0 : olddelta);
    }
    ta_consec_noncon = 0;
    if (ckt->CKTtime > 0.0) {
        if (ta_steps_since_bkpt < 1000)
            ta_steps_since_bkpt++;
        if (ta_steps_at_ord < 1000)
            ta_steps_at_ord++;
    }
#endif
    if ((ckt->CKTmode & MODEUIC && ckt->CKTtime > 0 && ckt->CKTtime >= ckt->CKTinitTime)
        || (!(ckt->CKTmode & MODEUIC) && ckt->CKTtime >= ckt->CKTinitTime)) {
        CKTdump(ckt, ckt->CKTtime, job->TRANplot);
    }
#ifdef XSPICE
    /* Update event queues/data for accepted timepoint */
    /* Note: this must be done AFTER sending results to SI so it can't
       go next to CKTaccept() above */
    if(ckt->evt->counts.num_insts > 0)
        EVTaccept(ckt, ckt->CKTtime);
#endif
    ckt->CKTstat->STAToldIter = ckt->CKTstat->STATnumIter;
    /* Check for the end of the tran simulation, either by stop time given,
       or final time has been reached. */
    if (have_autostop)
    /* time consuming autostop check only, when variable 'autostop' has been set
       before tran is started.*/
        flag_autostop = check_autostop("tran");
    /* If CKTtime and CKTfinalTime are almost equal, then finish */
    if (flag_autostop || AlmostEqualUlps(ckt->CKTtime, ckt->CKTfinalTime, 100)) {
#ifdef STEPDEBUG
        printf(" done:  time is %g, final time is %g, and tol is %g\n",
        ckt->CKTtime, ckt->CKTfinalTime, ckt->CKTminBreak);
#endif
        SPfrontEnd->OUTendPlot (job->TRANplot);
        job->TRANplot = NULL;
        UPDATE_STATS(0);
#ifdef WANT_SENSE2
        if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN)){
            ckt->CKTsenInfo->SENmode = save;
        }
#endif
        if (flag_autostop)
            fprintf(stdout, "\nNote: Autostop after %e s, all measurement conditions are fulfilled.\n", ckt->CKTtime);

        /* Final return from tran upon success */
#if defined(KLU) && NGSPICE_OPT_GENERAL
        mr_cf_cleanup(&cf, ckt);
#endif
        ckt->CKTag[0] = ckt->CKTag[1] = 0.;
        return(OK);
    }
    if(SPfrontEnd->IFpauseTest()) {
        /* user requested pause... */
#if defined(KLU) && NGSPICE_OPT_GENERAL
        mr_cf_cleanup(&cf, ckt);
#endif
        UPDATE_STATS(DOING_TRAN);
        return(E_PAUSE);
    }
resume:
#ifdef STEPDEBUG
    if( (ckt->CKTdelta <= ckt->CKTfinalTime/50) &&
        (ckt->CKTdelta <= ckt->CKTmaxStep)) {
        ;
    } else {
        if(ckt->CKTfinalTime/50<ckt->CKTmaxStep) {
            (void)printf("limited by Tstop/50\n");
        } else {
            (void)printf("limited by Tmax == %g\n",ckt->CKTmaxStep);
        }
    }
#endif
#ifdef HAS_PROGREP
    if (ckt->CKTtime == 0.)
        SetAnalyse( "tran init", 0);
    else
        SetAnalyse( "tran", (int)((ckt->CKTtime * 1000.) / ckt->CKTfinalTime + 0.5));
#endif

    ckt->CKTdelta = MIN(ckt->CKTdelta,ckt->CKTmaxStep);

#ifdef XSPICE
    /* Cut integration order if first timepoint after breakpoint */
    if ( AlmostEqualUlps( ckt->CKTtime, g_mif_info.breakpoint.last, 100 ) ) {
        ckt->CKTorder = 1;
#if NGSPICE_OPT_GENERAL
        if (ta_pred_mode > 0 && ckt->CKTtaSolCount > 1)
            ckt->CKTtaSolCount = 1;
        ta_pi_valid = 0;
        ta_steps_since_bkpt = 0;
        ta_steps_at_ord = 0;
#endif
    }
#endif

    /* Are we at a breakpoint, or indistinguishably close? */
    if ( AlmostEqualUlps( ckt->CKTtime, ckt->CKTbreaks[0], 100 ) ||
         ckt->CKTbreaks[0] - ckt->CKTtime <= ckt->CKTdelmin) {
        /* First timepoint after a breakpoint: cut integration order
           and limit timestep to .1 times minimum of time to next breakpoint,
           and previous timestep. */
        ckt->CKTorder = 1;
#if NGSPICE_OPT_GENERAL
        if (ta_pred_mode > 0 && ckt->CKTtaSolCount > 1)
            ckt->CKTtaSolCount = 1;
        ta_pi_valid = 0;
        ta_just_rejected = 0;
        ta_steps_since_bkpt = 0;
        ta_steps_at_ord = 0;
#endif
#ifdef STEPDEBUG
        if( (ckt->CKTdelta > .1*ckt->CKTsaveDelta) ||
            (ckt->CKTdelta > .1*(ckt->CKTbreaks[1] - ckt->CKTbreaks[0])) ) {
            if(ckt->CKTsaveDelta < (ckt->CKTbreaks[1] - ckt->CKTbreaks[0]))  {
                (void)printf("limited by pre-breakpoint delta (saveDelta: %g, nxt_breakpt: %g, curr_breakpt: %g\n",
                  ckt->CKTsaveDelta, ckt->CKTbreaks[1], ckt->CKTbreaks[0]);
            } else {
                (void)printf("limited by next breakpoint\n");
            }
        }
#endif

#if NGSPICE_OPT_GENERAL
        if (ta_bkpt_mode > 0 && !firsttime) {
            double next_gap = ckt->CKTbreaks[1] - ckt->CKTbreaks[0];
            double prev_dt = ckt->CKTsaveDelta;
            double bkpt_dt;
            if (next_gap > 2.0 * prev_dt) {
                /* Exiting fast ramp into long flat plateau: avoid 0.1x step collapse */
                double f_flat = (ta_bkpt_mode >= 2) ? 0.50 : 0.25;
                bkpt_dt = MIN(f_flat * prev_dt, 0.25 * next_gap);
            } else {
                /* Entering short ramp: 5-6 points across ramp instead of 10+ */
                double f_ramp = (ta_bkpt_mode >= 2) ? 0.25 : 0.16;
                bkpt_dt = f_ramp * MIN(prev_dt, next_gap);
            }
            ckt->CKTdelta = MIN(ckt->CKTdelta, bkpt_dt);
        } else
#endif
        ckt->CKTdelta = MIN(ckt->CKTdelta, .1 * MIN(ckt->CKTsaveDelta,
            ckt->CKTbreaks[1] - ckt->CKTbreaks[0]));

        if(firsttime) {
            /* set a breakpoint to reduce ringing of current in devices */
            if (ckt->CKTmode & MODEUIC)
                CKTsetBreak(ckt, ckt->CKTstep);

            ckt->CKTdelta /= 10;
#ifdef STEPDEBUG
            (void)printf("delta cut for initial timepoint\n");
#endif
        }

#ifndef XSPICE
        /* don't want to get below delmin for no reason */
        ckt->CKTdelta = MAX(ckt->CKTdelta, ckt->CKTdelmin*2.0);
#endif

    }

#ifndef XSPICE
    /* Force the breakpoint if appropriate */
    else if(ckt->CKTtime + ckt->CKTdelta >= ckt->CKTbreaks[0]) {
        ckt->CKTsaveDelta = ckt->CKTdelta;
        ckt->CKTdelta = ckt->CKTbreaks[0] - ckt->CKTtime;
#ifdef STEPDEBUG
        (void)printf("delta cut to %g to hit breakpoint\n",ckt->CKTdelta);
        fflush(stdout);
#endif
        ckt->CKTbreak = 1; /* why? the current pt. is not a bkpt. */
    }
    /* Try to equalise the last two time steps before the breakpoint,
       if the second step would be smaller than CKTdelta otherwise.*/
    else if (ckt->CKTtime + 1.9 * ckt->CKTdelta > ckt->CKTbreaks[0]) {
        ckt->CKTsaveDelta = ckt->CKTdelta;
        ckt->CKTdelta = (ckt->CKTbreaks[0] - ckt->CKTtime) / 2.;
#ifdef STEPDEBUG
        fprintf(stdout, "Delta equalising step at time %e with delta %e\n", ckt->CKTtime, ckt->CKTdelta);
#endif
    }
#endif /* !XSPICE */


#ifdef XSPICE
    /* More Breakpoint stuff */
    if(ckt->CKTtime + ckt->CKTdelta >= g_mif_info.breakpoint.current) {
        /* If next time > temporary breakpoint, force it to the breakpoint,
           and mark that timestep was set by temporary breakpoint */
        ckt->CKTsaveDelta = ckt->CKTdelta;
        ckt->CKTdelta = g_mif_info.breakpoint.current - ckt->CKTtime;
        g_mif_info.breakpoint.last = ckt->CKTtime + ckt->CKTdelta;
    } else {
        /* Else, mark that timestep was not set by temporary breakpoint */
        g_mif_info.breakpoint.last = 1.0e30;
    }

    /* Throw out any permanent breakpoint with time <= current time or in the
       very near future, unless it is the final stop break. */
#ifdef STEPDEBUG
    printf("    brk_pt: %g    ckt_time: %g    ckt_min_break: %g\n",
           ckt->CKTbreaks[0], ckt->CKTtime, ckt->CKTminBreak);
#endif
    while ((ckt->CKTbreaks[0] <= ckt->CKTtime + ckt->CKTminBreak ||
            AlmostEqualUlps(ckt->CKTbreaks[0], ckt->CKTtime, 100)) &&
           ckt->CKTbreaks[0] < ckt->CKTfinalTime) {
#ifdef STEPDEBUG
        printf("throwing out permanent breakpoint times <= current time "
               "(brk pt: %g)\n",
               ckt->CKTbreaks[0]);
        printf("    ckt_time: %g    ckt_min_break: %g\n",
               ckt->CKTtime, ckt->CKTminBreak);
#endif
        CKTclrBreak(ckt);
    }
    /* Force the breakpoint if appropriate */
    if(ckt->CKTtime + ckt->CKTdelta > ckt->CKTbreaks[0]) {
        ckt->CKTbreak = 1;
        ckt->CKTsaveDelta = ckt->CKTdelta;
        ckt->CKTdelta = ckt->CKTbreaks[0] - ckt->CKTtime;
    }
    /* Try to equalise the last two time steps before the breakpoint,
       if the second step would be smaller than CKTdelta otherwise.*/
    else if (!AlmostEqualUlps(ckt->CKTtime + ckt->CKTdelta, ckt->CKTfinalTime, 100)
#if NGSPICE_OPT_GENERAL
             && ckt->CKTtime + (ta_bkpt_mode > 0 ? 1.5 : 1.9) * ckt->CKTdelta > ckt->CKTbreaks[0]) {
#else
             && ckt->CKTtime + 1.9 * ckt->CKTdelta > ckt->CKTbreaks[0]) {
#endif
        ckt->CKTsaveDelta = ckt->CKTdelta;
        ckt->CKTdelta = (ckt->CKTbreaks[0] - ckt->CKTtime) / 2.0;
#ifdef STEPDEBUG
        fprintf(stdout, "Delta equalising step at time %e with new delta %e,\n remaining time %e and breakpoint at %e\n",
            ckt->CKTtime, ckt->CKTdelta, ckt->CKTbreaks[0] - ckt->CKTtime, ckt->CKTbreaks[0]);
#endif
    }
#ifdef SHARED_MODULE
        /* Either directly go to next time step, or modify ckt->CKTdelta depending on
           synchronization requirements. sharedsync() returns 0. */
    sharedsync(&ckt->CKTtime, &ckt->CKTdelta, 0, ckt->CKTfinalTime,
        ckt->CKTdelmin, 0, &ckt->CKTstat->STATrejected, 0);
#endif

    /* Do event solution */
    if(ckt->evt->counts.num_insts > 0) {

        /* If time = 0 and op_alternate was specified as false during
           dcop analysis, call any changed instances to let them
           post their outputs with their associated delays */
        if((ckt->CKTtime == 0.0) && (! ckt->evt->options.op_alternate))
            EVTiter(ckt);

        /* While there are events on the queue with event time <= next */
        /* projected analog time, process them */
        while((g_mif_info.circuit.evt_step = EVTnext_time(ckt))
               <= (ckt->CKTtime + ckt->CKTdelta)) {

            /* Initialize temp analog bkpt to infinity */
            g_mif_info.breakpoint.current = 1e30;

            /* Pull items off queue and process them */
            EVTdequeue(ckt, g_mif_info.circuit.evt_step);
            EVTiter(ckt);

            /* If any instances have forced an earlier
               next analog time, cut the delta */
            if(ckt->CKTbreaks[0] < g_mif_info.breakpoint.current)
                if(ckt->CKTbreaks[0] > ckt->CKTtime + ckt->CKTminBreak)
                    g_mif_info.breakpoint.current = ckt->CKTbreaks[0];
            if(g_mif_info.breakpoint.current < ckt->CKTtime + ckt->CKTdelta) {
                /* Breakpoint must be > last accepted timepoint
                   and >= current event time */
                if(g_mif_info.breakpoint.current >  ckt->CKTtime + ckt->CKTminBreak  &&
                   g_mif_info.breakpoint.current >= g_mif_info.circuit.evt_step) {
                    ckt->CKTsaveDelta = ckt->CKTdelta;
                    ckt->CKTdelta = g_mif_info.breakpoint.current - ckt->CKTtime;
                    g_mif_info.breakpoint.last = ckt->CKTtime + ckt->CKTdelta;
                }
            }

        } /* end while next event time <= next analog time */
    } /* end if there are event instances */

#else /* no XSPICE */

#ifdef SHARED_MODULE
    /* Either directly go to next time step, or modify ckt->CKTdelta depending on
       synchronization requirements. sharedsync() returns 0.
    */
    sharedsync(&ckt->CKTtime, &ckt->CKTdelta, 0, ckt->CKTfinalTime,
        ckt->CKTdelmin, 0, &ckt->CKTstat->STATrejected, 0);
#endif

#endif  /* no XSPICE */
#if defined(KLU) && NGSPICE_OPT_GENERAL
    if (cf.enabled) {
        int n_rhs = cf.n_rhs;
        if (!cf.vprev_init) {
            memcpy(ckt->CKTmrVprev1, ckt->CKTrhsOld, (size_t)n_rhs * sizeof(double));
            memcpy(ckt->CKTmrVprev2, ckt->CKTrhsOld, (size_t)n_rhs * sizeof(double));
            cf.vprev_init = 1;
        } else {
            memcpy(ckt->CKTmrVprev2, ckt->CKTmrVprev1, (size_t)n_rhs * sizeof(double));
            memcpy(ckt->CKTmrVprev1, ckt->CKTrhsOld,   (size_t)n_rhs * sizeof(double));
            /* Update per-node consecutive quiet streak */
            const double * restrict v0 = ckt->CKTmrVprev1;
            const double * restrict v1 = ckt->CKTmrVprev2;
            unsigned char * restrict qs = cf.quiet_streak;
            double v_mr = cf.v_mr;
            for (int u = 1; u < n_rhs; u++) {
                if (fabs(v0[u] - v1[u]) <= v_mr) {
                    if (qs[u] < 255) qs[u]++;
                } else {
                    qs[u] = 0;
                }
            }
        }

        int at_or_hit_brk = (ckt->CKTbreak != 0 ||
                             AlmostEqualUlps(ckt->CKTtime, ckt->CKTbreaks[0], 100) ||
                             AlmostEqualUlps(ckt->CKTtime + ckt->CKTdelta, ckt->CKTfinalTime, 100));

        if (ckt->CKTmrSubActive) {
            cf.stat_micro_acc++;
            ckt->CKTmrSubBurstLen++;
            if (ckt->CKTmrSubBurstLen >= cf.M - 1 || at_or_hit_brk || ckt->CKTorder < 2) {
                mr_cf_exit_burst(ckt);
            }
        } else {
            cf.stat_full_acc++;
            ckt->CKTmrSubNeedSync = 0;
            if (!firsttime && ckt->CKTstat->STATaccepted >= 5 &&
                ckt->CKTorder >= 2 && !at_or_hit_brk)
            {
                unsigned char * restrict is_f = ckt->CKTmrIsFast;
                const unsigned char * restrict rail = cf.is_rail;
                const unsigned char * restrict af   = cf.always_fast;
                const unsigned char * restrict qs   = cf.quiet_streak;

                is_f[0] = 0;
                for (int u = 1; u < n_rhs; u++) {
                    if (rail[u]) {
                        is_f[u] = 0;
                    } else if (af[u]) {
                        is_f[u] = 1;
                    } else {
                        is_f[u] = (qs[u] < 2) ? 1 : 0;
                    }
                }
                SMPkluSubExpand1Hop(ckt->CKTmatrix, is_f, n_rhs, rail);
                /* Branch closure for floating voltage sources */
                for (int ti = 0; ti < cf.n_triples; ti++) {
                    int p = cf.triples[ti].pos;
                    int n = cf.triples[ti].neg;
                    int b = cf.triples[ti].br;
                    int any = ((p > 0 && p < n_rhs && is_f[p]) ||
                               (n > 0 && n < n_rhs && is_f[n]) ||
                               (b > 0 && b < n_rhs && is_f[b]));
                    if (any) {
                        if (p > 0 && p < n_rhs && !rail[p]) is_f[p] = 1;
                        if (n > 0 && n < n_rhs && !rail[n]) is_f[n] = 1;
                        if (b > 0 && b < n_rhs && !rail[b]) is_f[b] = 1;
                    }
                }
                int nF_est = 0;
                for (int u = 1; u < n_rhs; u++) {
                    if (rail[u]) is_f[u] = 0;
                    else if (is_f[u]) nF_est++;
                }
                int max_nF = (int)(cf.max_frac * (double)(n_rhs - 1));
                if (nF_est > 0 && nF_est <= max_nF) {
                    int nF_actual = 0;
                    int same_as_prev = 0;
                    const int *fast_rhs = NULL;
                    if (SMPkluSubPrepare(ckt->CKTmatrix, is_f, n_rhs, rail,
                                         cf.rail_node, cf.n_rails, max_nF,
                                         &nF_actual, &fast_rhs, &same_as_prev,
                                         &cf.stat_cache_hits, &cf.stat_cache_misses))
                    {
                        for (int fk = 0; fk < nF_actual; fk++) {
                            int fn = fast_rhs[fk];
                            if (fn > 0 && fn < n_rhs)
                                is_f[fn] = 1;
                        }
                        memcpy(ckt->CKTmrVmacro, ckt->CKTrhsOld, (size_t)n_rhs * sizeof(double));
                        ckt->CKTmrSubActive = 1;
                        ckt->CKTmrSubBurstLen = 0;
                        ckt->CKTmrSubNFast = nF_actual;
                        ckt->CKTmrFastNodes = fast_rhs;
                        ckt->CKTmrSubEpoch++;
                        (void)same_as_prev;
                        ckt->CKTmrSubLuValid = 0;
                        ckt->CKTmrSubChordCount = 0;
                        cf.stat_bursts++;
                        cf.stat_nf_sum += (unsigned long long)nF_actual;
                    }
                }
            }
        }
    }
#endif
    for(i=5; i>=0; i--)
        ckt->CKTdeltaOld[i+1] = ckt->CKTdeltaOld[i];
    ckt->CKTdeltaOld[0] = ckt->CKTdelta;

#if NGSPICE_OPT_GENERAL
    {
        int rot_ord = (ta_eff_maxord > ckt->CKTmaxOrder) ? ta_eff_maxord : ckt->CKTmaxOrder;
        temp = ckt->CKTstates[rot_ord + 1];
        for (i = rot_ord; i >= 0; i--) {
            ckt->CKTstates[i + 1] = ckt->CKTstates[i];
        }
        ckt->CKTstates[0] = temp;
    }
#else
    temp = ckt->CKTstates[ckt->CKTmaxOrder+1];
    for(i=ckt->CKTmaxOrder;i>=0;i--) {
        ckt->CKTstates[i+1] = ckt->CKTstates[i];
    }
    ckt->CKTstates[0] = temp;
#endif

/* 600 */
    for (;;) {
#if defined SHARED_MODULE
        redostep = 1;
#endif
#ifdef XSPICE
/* This is needed here to allow CAPask to output currents
   during Transient analysis. */
        ckt->CKTcurrentAnalysis = DOING_TRAN;

        xspice_breakpoints_processed = 0;
#endif
        olddelta=ckt->CKTdelta;
        /* time abort? */
        ckt->CKTtime += ckt->CKTdelta;
        ckt->CKTdeltaOld[0]=ckt->CKTdelta;
        NIcomCof(ckt);
#ifdef PREDICTOR
        error = NIpred(ckt);
#endif /* PREDICTOR */
#if NGSPICE_OPT_GENERAL
        if (ta_pred_mode > 0 && (ckt->CKTmode & MODEINITPRED)) {
            ta_pred_apply(ckt, ta_pred_mode);
            ckt->CKTtaPredActive = 1;
        } else {
            ckt->CKTtaPredActive = 0;
        }
#endif
        save_mode = ckt->CKTmode;
        save_order = ckt->CKTorder;
#ifdef XSPICE

        /* Initialize temporary breakpoint to infinity */
        g_mif_info.breakpoint.current = 1.0e30;

        /* Add convergence problem reporting flags */
        /* delta is forced to equal delmin on last attempt near line 650 */
        if(ckt->CKTdelta <= ckt->CKTdelmin)
            ckt->enh->conv_debug.last_NIiter_call = MIF_TRUE;
        else
            ckt->enh->conv_debug.last_NIiter_call = MIF_FALSE;

/* Call all hybrids */

        /* Set evt_step */
        if(ckt->evt->counts.num_insts > 0) {
            g_mif_info.circuit.evt_step = ckt->CKTtime;
        }
#endif
        /* Central solver step */
        converged = NIiter(ckt,ckt->CKTtranMaxIter);
#if NGSPICE_OPT_GENERAL
        ckt->CKTtaPredActive = 0;
#endif

        ckt->CKTstat->STATtimePts ++;
        ckt->CKTmode = (ckt->CKTmode&MODEUIC)|MODETRAN | MODEINITPRED;
        if(firsttime && ckt->CKTstate1 && ckt->CKTstate2 && ckt->CKTstate3) {
            memcpy(ckt->CKTstate2, ckt->CKTstate1,
                   (size_t) ckt->CKTnumStates * sizeof(double));
            memcpy(ckt->CKTstate3, ckt->CKTstate1,
                   (size_t) ckt->CKTnumStates * sizeof(double));
#if NGSPICE_OPT_GENERAL
            if (ta_eff_maxord >= 3 && ckt->CKTstate4) {
                memcpy(ckt->CKTstate4, ckt->CKTstate1,
                       (size_t) ckt->CKTnumStates * sizeof(double));
            }
#endif
        }
        /* txl, cpl addition */
        if (converged == 1111) {
                return(converged);
        }

        /* If no convergence in Central solver step */
        if(converged != 0) {

#if defined(KLU) && NGSPICE_OPT_GENERAL
            if (cf.enabled && ckt->CKTmrSubActive) {
                /* Micro-step failed convergence: immediately abort burst and retry
                 * as a full-circuit macro-sync step at the SAME delta & order! */
                mr_cf_exit_burst(ckt);
                cf.stat_fallback_exits++;
                ckt->CKTtime = ckt->CKTtime - ckt->CKTdelta;
                ckt->CKTstat->STATrejected ++;
                memcpy(ckt->CKTrhsOld, ckt->CKTmrVprev1, (size_t)cf.n_rhs * sizeof(double));
                continue;
            }
#endif
#ifndef SHARED_MODULE
            ckt->CKTtime = ckt->CKTtime -ckt->CKTdelta;
            ckt->CKTstat->STATrejected ++;
#else
            redostep = 1;
#endif
#if NGSPICE_OPT_GENERAL
            ta_pi_valid = 0;
            ta_just_rejected = 1;
            ta_steps_at_ord = 0;
            if (ta_nl_mode > 0 && ta_consec_noncon == 0 && !firsttime) {
                ckt->CKTdelta *= 0.25;
                if (ckt->CKTorder > 2)
                    ckt->CKTorder = 2;
                ta_consec_noncon++;
            } else {
                ckt->CKTdelta = ckt->CKTdelta / 8;
                ckt->CKTorder = 1;
                ta_consec_noncon++;
            }
            if (firsttime) {
                ckt->CKTmode = (ckt->CKTmode&MODEUIC) | MODETRAN | MODEINITTRAN;
                ckt->CKTorder = 1;
            }
#else
            ckt->CKTdelta = ckt->CKTdelta/8;
#ifdef STEPDEBUG
            (void)printf("delta cut to %g for non-convergence\n",ckt->CKTdelta);
            fflush(stdout);
#endif
            if(firsttime) {
                ckt->CKTmode = (ckt->CKTmode&MODEUIC) | MODETRAN | MODEINITTRAN;
            }
            ckt->CKTorder = 1;
#endif

#ifdef XSPICE
        /* Add Breakpoint stuff */
        } else if(g_mif_info.breakpoint.current < ckt->CKTtime) {
            /* Force backup if temporary breakpoint is < current time:
               - save old delta
               - retract time by old delta
               - new delta by difference between by retracted time and breakpoint */

        past_breakpoint:
            ckt->CKTsaveDelta = ckt->CKTdelta;
            ckt->CKTtime -= ckt->CKTdelta;
            ckt->CKTdelta = g_mif_info.breakpoint.current - ckt->CKTtime;
            g_mif_info.breakpoint.last = ckt->CKTtime + ckt->CKTdelta;
            xspice_breakpoints_processed = 1;

            if(firsttime) {
                ckt->CKTmode = (ckt->CKTmode&MODEUIC)|MODETRAN | MODEINITTRAN;
            }
            ckt->CKTorder = 1;
#endif

        } else {
#if defined(KLU) && NGSPICE_OPT_GENERAL
            if (cf.enabled && ckt->CKTmrSubActive &&
                SMPkluSubCheckWake(ckt->CKTmatrix, ckt->CKTrhsOld,
                                   ckt->CKTmrVmacro, cf.v_wake))
            {
                /* Pre-accept boundary guard: a fast boundary node moved > v_wake.
                 * Reject this micro-step BEFORE acceptance and re-solve the exact
                 * same timestep on the full circuit for zero phase/propagation lag. */
                mr_cf_exit_burst(ckt);
                cf.stat_wake_exits++;
                ckt->CKTtime = ckt->CKTtime - ckt->CKTdelta;
                ckt->CKTstat->STATrejected ++;
                memcpy(ckt->CKTrhsOld, ckt->CKTmrVprev1, (size_t)cf.n_rhs * sizeof(double));
                continue;
            }
#endif
            /* If converged:
               - go to next time step if this was the first time.
               - If not the first time step, don not accept, but check the truncation errors,
                 and reduce delta accordingly, thenm redo the step, to bound the error. */
            if (firsttime) {
#ifdef WANT_SENSE2
                if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN)){
                    save1 = ckt->CKTmode;
                    save2 = ckt->CKTorder;
                    ckt->CKTmode = save_mode;
                    ckt->CKTorder = save_order;
                    error = CKTsenDCtran (ckt);
                    if (error)
                        return(error);

                    ckt->CKTmode = save1;
                    ckt->CKTorder = save2;
                }
#endif
                firsttime = 0;
#if !defined SHARED_MODULE
                /* no check on first time point */
                goto nextTime; /* line 373 */
#else
                redostep = 0;
                goto chkStep; /* line 987 */
#endif
            }
            newdelta = ckt->CKTdelta;
            /* Scan through all devices, estimate the truncation error,
               and reduce the time step, if necessary.*/
            error = CKTtrunc(ckt,&newdelta);
            if(error) {
                UPDATE_STATS(DOING_TRAN);
                return(error);
            }
            if (newdelta > .9 * ckt->CKTdelta) {
#if defined(XSPICE)
                /* The timestep has succeeded.  XSPICE instances with
                 * both analog and event ports ("hybrids") and others
                 * that have called cm_irreversible() receive an EVENT
                 * call here that allows them to capture their final
                 * port values and advance co-simulations.  As this is an EVENT
                 * call, they are not expected to do any integrations,
                 * so there is no need for a further convergence test.
                 */

                if (ckt->evt->counts.num_hybrids > 0) {
                    g_mif_info.circuit.evt_step = ckt->CKTtime;
                    EVTcall_hybrids(ckt);
                    if (g_mif_info.breakpoint.current < ckt->CKTtime) {
                        /* A hybrid requested a breakpoint in the past. */
                        goto past_breakpoint; /* line 786 */
                    }
                }
#endif
                /* don't raise the order for backward Euler */
                if ((ckt->CKTorder == 1) && (ckt->CKTmaxOrder > 1)) {
#if NGSPICE_OPT_GENERAL
                    if (ta_bkpt_mode > 0 && ta_steps_since_bkpt == 0) {
                        /* First accepted step after a breakpoint: CKTstates[3] is pre-breakpoint,
                         * so skip the invalid/redundant 2nd CKTtrunc sweep and promote directly
                         * when order-1 LTE is clean (Gear-2 on the next step uses only post-bkpt states). */
                        if (newdelta >= ckt->CKTdelta) {
                            ckt->CKTorder = 2;
                            ta_pi_valid = 0;
                            ta_steps_at_ord = 0;
                        }
                    } else
#endif
                    {
                        newdelta = ckt->CKTdelta;
                        ckt->CKTorder = 2;
                        error = CKTtrunc(ckt, &newdelta);
                        if (error) {
                            UPDATE_STATS(DOING_TRAN);
                            return(error);
                        }
                        if (newdelta <= 1.05 * ckt->CKTdelta) {
                            ckt->CKTorder = 1;
                        }
#if NGSPICE_OPT_GENERAL
                        else {
                            ta_pi_valid = 0;
                            ta_steps_at_ord = 0;
                        }
#endif
                    }
                }
#if NGSPICE_OPT_GENERAL
                else if (ta_ord3_mode > 0 && ckt->CKTintegrateMethod == GEAR &&
                         ta_eff_maxord >= 3 && !ckt->CKTmrSubActive)
                {
                    if (ckt->CKTorder == 2 && ta_steps_at_ord >= 3 && (ta_steps_at_ord % 3 == 0) &&
                        ta_steps_since_bkpt >= 4 && newdelta < 0.95 * ckt->CKTmaxStep)
                    {
                        double h3 = ckt->CKTdelta;
                        ckt->CKTorder = 3;
                        error = CKTtrunc(ckt, &h3);
                        if (error) {
                            UPDATE_STATS(DOING_TRAN);
                            return(error);
                        }
                        if (h3 > 1.15 * newdelta) {
                            newdelta = h3;
                            ta_steps_at_ord = 0;
                            ta_pi_valid = 0;
                        } else {
                            ckt->CKTorder = 2;
                        }
                    } else if (ckt->CKTorder == 3 &&
                               ((ta_steps_at_ord >= 5 && (ta_steps_at_ord % 5 == 0)) ||
                                newdelta < 1.05 * ckt->CKTdelta))
                    {
                        double h2 = ckt->CKTdelta;
                        ckt->CKTorder = 2;
                        error = CKTtrunc(ckt, &h2);
                        if (error) {
                            UPDATE_STATS(DOING_TRAN);
                            return(error);
                        }
                        if (h2 > 1.05 * newdelta) {
                            newdelta = h2;
                            ta_steps_at_ord = 0;
                            ta_pi_valid = 0;
                        } else {
                            ckt->CKTorder = 3;
                        }
                    }
                }
#endif
                /* time point OK  - 630 */
#if NGSPICE_OPT_GENERAL
                if (ta_pi_mode > 0 || ta_db_hi > 1.0 || ta_nl_mode > 0) {
                    double h_cur = ckt->CKTdelta;
                    double rho_raw = (h_cur > 0.0) ? (newdelta / h_cur) : 1.0;
                    double prop_delta = newdelta;
                    if (ta_pi_mode > 0) {
                        if (ta_just_rejected) {
                            if (prop_delta > h_cur)
                                prop_delta = h_cur;
                            ta_just_rejected = 0;
                            ta_pi_valid = 0;
                        } else if (ta_pi_valid) {
                            double r_filt = rho_raw;
                            if (ta_pi_mode == 1) {
                                /* Gustafsson PI (DC gain 0.60, predictive brake on dropping rho) */
                                r_filt = pow(rho_raw, 0.85) * pow(ta_rho_prev, -0.25);
                            } else if (ta_pi_mode == 2) {
                                /* Soderlind H211PI (DC gain 0.80, smooth damping + brake) */
                                r_filt = pow(rho_raw, 0.50) * pow(ta_rho_prev, 0.30) * pow(ta_h_ratio_prev, -0.20);
                            } else {
                                /* Predictive brake-only PI: never slows smooth ramp-ups, brakes into active edges */
                                if (rho_raw < 1.25 && rho_raw < ta_rho_prev)
                                    r_filt = pow(rho_raw, 0.85) * pow(ta_rho_prev, -0.25);
                            }
                            if (r_filt < 0.5) r_filt = 0.5;
                            if (r_filt > 2.0) r_filt = 2.0;
                            double h_pi = h_cur * r_filt;
                            if (h_pi < prop_delta)
                                prop_delta = h_pi;
                        }
                        ta_rho_prev = (rho_raw < 0.2) ? 0.2 : ((rho_raw > 2.5) ? 2.5 : rho_raw);
                        ta_pi_valid = 1;
                    } else {
                        ta_just_rejected = 0;
                    }
                    if (ta_nl_mode > 0) {
                        int n_it = ckt->CKTtaLastIter;
                        if (n_it >= 15) {
                            double cap_d = 0.75 * h_cur;
                            if (prop_delta > cap_d) prop_delta = cap_d;
                        } else if (n_it >= 11) {
                            if (prop_delta > h_cur) prop_delta = h_cur;
                        } else if (n_it >= 7) {
                            double cap_d = 1.30 * h_cur;
                            if (prop_delta > cap_d) prop_delta = cap_d;
                        }
                    }
                    if (ta_db_hi > 1.0) {
                        /* SUNDIALS IDA step-size deadband: hold exact CKTdelta -> ag0 constant,
                         * gated on rho_raw <= ta_db_hi so PI/NL throttled growth is never trapped! */
                        if (rho_raw >= 0.95 && rho_raw <= ta_db_hi &&
                            prop_delta >= 0.95 * h_cur && prop_delta <= ta_db_hi * h_cur)
                            prop_delta = h_cur;
                    }
                    ta_h_ratio_prev = (h_cur > 0.0) ? (prop_delta / h_cur) : 1.0;
                    if (ta_h_ratio_prev < 0.2) ta_h_ratio_prev = 0.2;
                    if (ta_h_ratio_prev > 2.5) ta_h_ratio_prev = 2.5;
                    ckt->CKTdelta = prop_delta;
                } else {
                    ckt->CKTdelta = newdelta;
                }
#else
                ckt->CKTdelta = newdelta;
#endif

#ifdef NDEV
                if (!ft_norefprint) {
                    /* show a time process indicator, by Gong Ding, gdiso@ustc.edu */
                    if (ckt->CKTtime / ckt->CKTfinalTime * 100 < 10.0)
                        printf("%%%3.2lf\b\b\b\b\b", ckt->CKTtime / ckt->CKTfinalTime * 100);
                    else  if (ckt->CKTtime / ckt->CKTfinalTime * 100 < 100.0)
                        printf("%%%4.2lf\b\b\b\b\b\b", ckt->CKTtime / ckt->CKTfinalTime * 100);
                    else
                        printf("%%%5.2lf\b\b\b\b\b\b\b", ckt->CKTtime / ckt->CKTfinalTime * 100);
                    fflush(stdout);
                }
#endif

#ifdef STEPDEBUG
                (void)printf(
                  "delta set to truncation error result: %g. Point accepted at CKTtime: %g\n",
                  ckt->CKTdelta,ckt->CKTtime);
                fflush(stdout);
#endif

#ifdef WANT_SENSE2
                if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN)){
                    save1 = ckt->CKTmode;
                    save2 = ckt->CKTorder;
                    ckt->CKTmode = save_mode;
                    ckt->CKTorder = save_order;
                    error = CKTsenDCtran(ckt);
                    if (error)
                        return (error);

                    ckt->CKTmode = save1;
                    ckt->CKTorder = save2;
                }
#endif

#if !defined SHARED_MODULE
                /* trapezoidal */
                goto nextTime;  /* line 373 */
#else
                redostep = 0;
                goto chkStep;  /* line 987 */
#endif
            } else {
                /* not (newdelta > .9 * ckt->CKTdelta): reject the step
                - redo the time
                - apply the new (reduced) delta */
#if defined(KLU) && NGSPICE_OPT_GENERAL
                if (cf.enabled && ckt->CKTmrSubActive && newdelta <= 0.35 * ckt->CKTdelta) {
                    /* Sharp local truncation cut during micro-burst: fall back to full step */
                    mr_cf_exit_burst(ckt);
                    cf.stat_fallback_exits++;
                    memcpy(ckt->CKTrhsOld, ckt->CKTmrVprev1, (size_t)cf.n_rhs * sizeof(double));
                }
#endif
#ifndef SHARED_MODULE
                ckt->CKTtime = ckt->CKTtime -ckt->CKTdelta;
                ckt->CKTstat->STATrejected ++;
#else
                redostep = 1;
#endif
#if NGSPICE_OPT_GENERAL
                ta_pi_valid = 0;
                ta_just_rejected = 1;
                ta_steps_at_ord = 0;
                if (ckt->CKTorder > 2)
                    ckt->CKTorder = 2;
                ta_h_ratio_prev = (ckt->CKTdelta > 0.0) ? (newdelta / ckt->CKTdelta) : 0.5;
                if (ta_h_ratio_prev < 0.2) ta_h_ratio_prev = 0.2;
#endif
                ckt->CKTdelta = newdelta;
#ifdef STEPDEBUG
                (void)printf(
                    "delta set to truncation error result: point rejected\n");
#endif
            }
        }
        /* Set the new delta to delmin (minimum delta allowed). However:
           If the new delta has been less than the minimum delta
           for the second time, bail out with 'Timestep too small'. */
        if (ckt->CKTdelta <= ckt->CKTdelmin) {
            if (olddelta > ckt->CKTdelmin) {
                ckt->CKTdelta = ckt->CKTdelmin;
#ifdef STEPDEBUG
                (void)printf("delta at delmin\n");
#endif
            } else {
                UPDATE_STATS(DOING_TRAN);
#if defined(KLU) && NGSPICE_OPT_GENERAL
                mr_cf_cleanup(&cf, ckt);
#endif
                errMsg = CKTtrouble(ckt, "Timestep too small");
                SPfrontEnd->OUTendPlot(job->TRANplot);
                job->TRANplot = NULL;
                UPDATE_STATS(0);
                ckt->CKTag[0] = ckt->CKTag[1] = 0.;
                /* return upon convergence failure */
                return(E_TIMESTEP);
            }
        }
#ifdef XSPICE
        /* Do event backup */
        if(ckt->evt->counts.num_insts > 0) {
#ifdef SHARED_MODULE
            double discard_start_time = ckt->CKTtime + ckt->CKTdelta;
            /* ngspice in executable mode subtracts olddelta from the time
               before new delta calculation, but it keeps delta in CKTtime and
               postpones subtraction in library mode. Delayed subtraction leads
               to incorrect points dropping because ckt->CKTdelta is almost always
               less than olddelta if there are convergence issues, and EVTbackup
               may drop valid events that need to be processed within
               [last_accepted_time, last_accepted_time + ckt->CKTdelta] range
               after delta adjustment. */
            if (redostep && xspice_breakpoints_processed == 0)
                discard_start_time -= olddelta;
            EVTbackup(ckt, discard_start_time);
#else
            EVTbackup(ckt, ckt->CKTtime + ckt->CKTdelta);
#endif
        }

#endif

#ifdef SHARED_MODULE
        /* redostep == 0:
           Either directly go to next time step, or modify ckt->CKTdelta depending on
           synchronization requirements. sharedsync() returns 0.
           redostep == 1:
           No convergence, or too large truncation error.
           Redo the last time step by subtracting olddelta, and modify ckt->CKTdelta
           depending on synchronization requirements. sharedsync() returns 1.
           User-supplied redo request:
           sharedsync() may return 1 if the user has decided to do so in the callback
           function.
        */
chkStep:
#ifdef XSPICE
       // There is no need to subtract olddelta from ckt->CKTtime one more time
       // if it has been subtracted during XSPICE breakpoint processing.
       // olddelta will be reinitialized on
       // the new iteration, so it reassigning here should be safe. It can't be
       // zeroed during breakpoint processing because it takes part in the
       // "timestep too small" check.
        olddelta_for_shared_sync = olddelta;
        if (xspice_breakpoints_processed)
                 olddelta_for_shared_sync = 0.0;
        if(sharedsync(&ckt->CKTtime, &ckt->CKTdelta, olddelta_for_shared_sync, ckt->CKTfinalTime,
                 ckt->CKTdelmin, redostep, &ckt->CKTstat->STATrejected, 1) == 0)
            goto nextTime; /* line 373 */
#else
        if(sharedsync(&ckt->CKTtime, &ckt->CKTdelta, olddelta, ckt->CKTfinalTime,
                 ckt->CKTdelmin, redostep, &ckt->CKTstat->STATrejected, 1) == 0)
            goto nextTime; /* line 373 */
#endif // XSPICE
#endif // SHARED_MODULE

    }
    /* NOTREACHED */
}


/* If we have a 'step' command, and then 'quit' before reaching the final time,
   remove the run plot memory. */
int
DCtran_step_quit(CKTcircuit* ckt) {
    if (!ckt || !ckt->CKTcurJob) /* nothing to delete */
        return 0;
    if (ckt->CKTcurJob->JOBtype != 4) /* only tran sim */
        return 0;
    if (!ft_stepcheck()) /* only after 'step' */
        return 0;
    TRANan* job = (TRANan*)ckt->CKTcurJob;
    if (!job->TRANplot) /* already done */
        return 0;
    SPfrontEnd->OUTendPlot(job->TRANplot);
    job->TRANplot = NULL;
    return(OK);
}
