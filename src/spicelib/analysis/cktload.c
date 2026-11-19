/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/
/*
 */

/* CKTload(ckt)
 * this is a driver program to iterate through all the various
 * load functions provided for the circuit elements in the
 * given circuit
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include <string.h>
#ifdef USE_OMP
#include <omp.h>
#endif

#ifdef XSPICE
#include "ngspice/enh.h"
/* gtri - add - wbk - 11/26/90 - add include for MIF global data */
#include "ngspice/mif.h"
/* gtri - end - wbk - 11/26/90 */
#endif

static int ZeroNoncurRow(SMPmatrix *matrix, CKTnode *nodes, int rownum);

// #define PER_DEVICE_STATS
int
CKTload(CKTcircuit *ckt)
{
    int i;
    int size;
    double startTime, td0;
    CKTnode *node;
    int error;
#ifdef STEPDEBUG
    int noncon;
#endif /* STEPDEBUG */

#ifdef XSPICE
    /* gtri - begin - Put resistors to ground at all nodes */
    /*   SMPmatrix  *matrix; maschmann : deleted , because unused */

    double gshunt;
    int num_nodes;

    /* gtri - begin - Put resistors to ground at all nodes */
#endif

    td0 = SPfrontEnd->IFseconds();

#ifdef PER_DEVICE_STATS
    ckt->CKTstat->devTimes[DEVmaxnum] += SPfrontEnd->IFseconds() - td0;
    ckt->CKTstat->devCounts[DEVmaxnum]++;
#endif

    startTime = SPfrontEnd->IFseconds();
    size = SMPmatSize(ckt->CKTmatrix);
#ifdef STEPDEBUG
    noncon = ckt->CKTnoncon;
#endif /* STEPDEBUG */

#if NGSPICE_OPT_GENERAL && defined(KLU) && defined(USE_OMP)
    if (ckt->CKTmatrix->CKTkluMODE &&
        ckt->CKTmatrix->SMPkluMatrix &&
        ckt->CKTmatrix->SMPkluMatrix->KLUmatrixIsComplex == KLUmatrixReal &&
        ckt->CKTmatrix->SMPkluMatrix->KLUmatrixAx != NULL &&
        ckt->CKTmatrix->SMPkluMatrix->KLUmatrixNZ > 0 &&
        !(ckt->CKTmode & MODEDC))
    {
        KLUmatrix *km = ckt->CKTmatrix->SMPkluMatrix;
        int nz = (int)km->KLUmatrixNZ;
        int rhs_len = (int)km->KLUmatrixNrhs + 1;
        if (rhs_len < size + 1)
            rhs_len = size + 1;
        double * restrict ax_main = km->KLUmatrixAx;
        double * restrict rhs_main = ckt->CKTrhs;

        int par_devs[16];
        int par_count = 0;
        int skip_mask_res = -1;
        int has_res_base = 0;
#if NGSPICE_OPT_APPLE
        int has_cap_base = 0;
#endif
        unsigned char handled[256];
        int use_handled = (DEVmaxnum <= 256);
        if (use_handled)
            memset(handled, 0, (size_t)DEVmaxnum);

        for (i = 0; i < DEVmaxnum; i++) {
            if (DEVices[i] && DEVices[i]->DEVkluPrep && ckt->CKThead[i]) {
                int prep = DEVices[i]->DEVkluPrep(ckt->CKThead[i], ckt);
                if (prep == 2) {
                    has_res_base = 1;
                    skip_mask_res = i;
                    if (use_handled) handled[i] = 1;
                }
#if NGSPICE_OPT_APPLE
                else if (prep == 3 && DEVices[i]->DEVkluLoadPar && par_count < 16) {
                    has_cap_base = 1;
                    par_devs[par_count++] = i;
                    if (use_handled) handled[i] = 1;
                }
#endif
                else if (prep == 1 && DEVices[i]->DEVkluLoadPar && par_count < 16) {
                    par_devs[par_count++] = i;
                    if (use_handled) handled[i] = 1;
                }
            }
        }

        const double * restrict res_base = has_res_base ? km->KLUmatrixAxResConst : NULL;
#if NGSPICE_OPT_APPLE
        const double * restrict cap_unit = has_cap_base ? km->KLUmatrixAxCapUnit : NULL;
        const double * restrict cap_rhs  = has_cap_base ? km->KLUmatrixRhsCapCache : NULL;
        const double cap_ag0 = ckt->CKTag[0];
#endif
        int nthreads = omp_get_max_threads();
        int par_error = 0;

        if (nthreads > 1 && par_count > 0 && SMPkluEnsureTls(km, nthreads, rhs_len)) {
            double * restrict ax_tls_base = km->KLUmatrixAxTls;
            double * restrict rhs_tls_base = km->KLUmatrixRhsTls;
            size_t ax_stride = km->KLUmatrixAxTlsStride;
            size_t rhs_stride = km->KLUmatrixRhsTlsStride;
            size_t ax_zero_bytes = ((size_t)nz + 1) * sizeof(double);
            size_t rhs_zero_bytes = (size_t)rhs_len * sizeof(double);

            #pragma omp parallel num_threads(nthreads) reduction(|:par_error)
            {
                int tid = omp_get_thread_num();
                double * restrict my_ax  = ax_tls_base  + (size_t)tid * ax_stride;
                double * restrict my_rhs = rhs_tls_base + (size_t)tid * rhs_stride;
                memset(my_ax,  0, ax_zero_bytes);
                memset(my_rhs, 0, rhs_zero_bytes);

                for (int k = 0; k < par_count; k++) {
                    int di = par_devs[k];
                    int e = DEVices[di]->DEVkluLoadPar(ckt->CKThead[di], ckt, my_ax, my_rhs);
                    if (e) par_error |= e;
                }

                #pragma omp barrier

                /* Cache-tiled deterministic parallel reduction (t = 0 .. nthreads-1) */
                #pragma omp for schedule(static) nowait
                for (int jb = 0; jb < nz; jb += 256) {
                    int jend = jb + 256;
                    if (jend > nz) jend = nz;
#if NGSPICE_OPT_APPLE
                    if (res_base != NULL && cap_unit != NULL) {
                        for (int j = jb; j < jend; j++)
                            ax_main[j] = (res_base[j] + cap_ag0 * cap_unit[j]) + ax_tls_base[j];
                    } else if (cap_unit != NULL) {
                        for (int j = jb; j < jend; j++)
                            ax_main[j] = cap_ag0 * cap_unit[j] + ax_tls_base[j];
                    } else
#endif
                    if (res_base != NULL) {
                        for (int j = jb; j < jend; j++)
                            ax_main[j] = res_base[j] + ax_tls_base[j];
                    } else {
                        for (int j = jb; j < jend; j++)
                            ax_main[j] = ax_tls_base[j];
                    }
                    for (int t = 1; t < nthreads; t++) {
                        const double * restrict src = ax_tls_base + (size_t)t * ax_stride;
                        for (int j = jb; j < jend; j++)
                            ax_main[j] += src[j];
                    }
                }
                #pragma omp for schedule(static)
                for (int jb = 0; jb < rhs_len; jb += 256) {
                    int jend = jb + 256;
                    if (jend > rhs_len) jend = rhs_len;
#if NGSPICE_OPT_APPLE
                    if (cap_rhs != NULL) {
                        for (int j = jb; j < jend; j++)
                            rhs_main[j] = cap_rhs[j] + rhs_tls_base[j];
                    } else
#endif
                    {
                        for (int j = jb; j < jend; j++)
                            rhs_main[j] = rhs_tls_base[j];
                    }
                    for (int t = 1; t < nthreads; t++) {
                        const double * restrict src = rhs_tls_base + (size_t)t * rhs_stride;
                        for (int j = jb; j < jend; j++)
                            rhs_main[j] += src[j];
                    }
                }
            }
            if (ckt->CKTnoncon)
                ckt->CKTtroubleNode = 0;
            if (par_error) return(par_error);
        } else {
            /* Single-thread (or no parallel devices): stamp directly into main KLU Ax & CKTrhs */
#if NGSPICE_OPT_APPLE
            if (cap_unit != NULL) {
                int want_res = (res_base != NULL) ? 1 : 0;
                if (!km->KLUmatrixAxResCapBase) {
                    if (posix_memalign((void **)&km->KLUmatrixAxResCapBase, 128,
                                       ((size_t)nz + 1) * sizeof(double)) != 0)
                        km->KLUmatrixAxResCapBase = NULL;
                }
                if (km->KLUmatrixAxResCapBase) {
                    if (!km->KLUmatrixResCapValid ||
                        km->KLUmatrixResCapAg0 != cap_ag0 ||
                        km->KLUmatrixResCapHasRes != want_res)
                    {
                        double * restrict rcb = km->KLUmatrixAxResCapBase;
                        if (res_base != NULL) {
                            for (int j = 0; j < nz; j++)
                                rcb[j] = res_base[j] + cap_ag0 * cap_unit[j];
                        } else {
                            for (int j = 0; j < nz; j++)
                                rcb[j] = cap_ag0 * cap_unit[j];
                        }
                        rcb[nz] = 0.0;
                        km->KLUmatrixResCapAg0 = cap_ag0;
                        km->KLUmatrixResCapHasRes = want_res;
                        km->KLUmatrixResCapValid = 1;
                    }
                    memcpy(ax_main, km->KLUmatrixAxResCapBase, (size_t)nz * sizeof(double));
                } else if (res_base != NULL) {
                    for (int j = 0; j < nz; j++)
                        ax_main[j] = res_base[j] + cap_ag0 * cap_unit[j];
                } else {
                    for (int j = 0; j < nz; j++)
                        ax_main[j] = cap_ag0 * cap_unit[j];
                }
            } else
#endif
            if (res_base != NULL)
                memcpy(ax_main, res_base, (size_t)nz * sizeof(double));
            else
                memset(ax_main, 0, (size_t)nz * sizeof(double));
            ax_main[nz] = 0.0;
#if NGSPICE_OPT_APPLE
            if (cap_rhs != NULL)
                memcpy(rhs_main, cap_rhs, (size_t)rhs_len * sizeof(double));
            else
#endif
            memset(rhs_main, 0, (size_t)rhs_len * sizeof(double));

            for (int k = 0; k < par_count; k++) {
                int di = par_devs[k];
                error = DEVices[di]->DEVkluLoadPar(ckt->CKThead[di], ckt, ax_main, rhs_main);
                if (ckt->CKTnoncon)
                    ckt->CKTtroubleNode = 0;
                if (error) return(error);
            }
        }

        /* Load all remaining devices (VSRC, ISRC, ASRC, DIO, etc.) */
        for (i = 0; i < DEVmaxnum; i++) {
            if (use_handled) {
                if (handled[i]) continue;
            } else {
                int skip = (i == skip_mask_res);
                for (int k = 0; k < par_count && !skip; k++)
                    if (par_devs[k] == i) skip = 1;
                if (skip) continue;
            }
            if (DEVices[i] && DEVices[i]->DEVload && ckt->CKThead[i]) {
                error = DEVices[i]->DEVload(ckt->CKThead[i], ckt);
                if (ckt->CKTnoncon)
                    ckt->CKTtroubleNode = 0;
                if (error) return(error);
            }
        }
        goto after_device_loads;
    }
#endif

    memset(ckt->CKTrhs, 0, (size_t)(size + 1) * sizeof(double));
    SMPclear(ckt->CKTmatrix);

    for (i = 0; i < DEVmaxnum; i++) {
        if (DEVices[i] && DEVices[i]->DEVload && ckt->CKThead[i]) {
#ifdef PER_DEVICE_STATS
            td0 = SPfrontEnd->IFseconds();
#endif
            error = DEVices[i]->DEVload (ckt->CKThead[i], ckt);
#ifdef PER_DEVICE_STATS
            ckt->CKTstat->devTimes[i] += SPfrontEnd->IFseconds() - td0;
            ckt->CKTstat->devCounts[i]++;
#endif
            if (ckt->CKTnoncon)
                ckt->CKTtroubleNode = 0;
#ifdef STEPDEBUG
            if (noncon != ckt->CKTnoncon) {
                printf("device type %s nonconvergence\n",
                       DEVices[i]->DEVpublic.name);
                noncon = ckt->CKTnoncon;
            }
#endif /* STEPDEBUG */
            if (error) return(error);
        }
    }

#if NGSPICE_OPT_GENERAL && defined(KLU) && defined(USE_OMP)
after_device_loads:
#endif


#ifdef XSPICE
    /* gtri - add - wbk - 11/26/90 - reset the MIF init flags */

    /* init is set by CKTinit and should be true only for first load call */
    g_mif_info.circuit.init = MIF_FALSE;

    /* anal_init is set by CKTdoJob and is true for first call */
    /* of a particular analysis type */
    g_mif_info.circuit.anal_init = MIF_FALSE;

    /* gtri - end - wbk - 11/26/90 */

    /* gtri - begin - Put resistors to ground at all nodes. */
    /* Value of resistor is set by new "rshunt" option.     */

    if (ckt->enh->rshunt_data.enabled) {
        gshunt = ckt->enh->rshunt_data.gshunt;
        num_nodes = ckt->enh->rshunt_data.num_nodes;
        for (i = 0; i < num_nodes; i++) {
            *(ckt->enh->rshunt_data.diag[i]) += gshunt;
        }
    }

    /* gtri - end - Put resistors to ground at all nodes */
#endif

    if (ckt->CKTmode & MODEDC) {
        /* consider doing nodeset & ic assignments */
        if (ckt->CKTmode & (MODEINITJCT | MODEINITFIX)) {
            /* do nodesets */
            for (node = ckt->CKTnodes; node; node = node->next) {
                if (node->nsGiven) {
                    if (ZeroNoncurRow(ckt->CKTmatrix, ckt->CKTnodes,
                                      node->number)) {
                        ckt->CKTrhs[node->number] = 1.0e10 * node->nodeset *
                                                      ckt->CKTsrcFact;
                        *(node->ptr) = 1e10;
                    } else {
                        ckt->CKTrhs[node->number] = node->nodeset *
                                                      ckt->CKTsrcFact;
                        *(node->ptr) = 1;
                    }
                    /* DAG: Original CIDER fix. If above fix doesn't work,
                     * revert to this.
                     */
                    /*
                     *  ckt->CKTrhs[node->number] += 1.0e10 * node->nodeset;
                     *  *(node->ptr) += 1.0e10;
                     */
                }
            }
        }
        if ((ckt->CKTmode & MODETRANOP) && (!(ckt->CKTmode & MODEUIC))) {
            for (node = ckt->CKTnodes; node; node = node->next) {
                if (node->icGiven) {
                    if (ZeroNoncurRow(ckt->CKTmatrix, ckt->CKTnodes,
                                      node->number)) {
                        /* Original code:
                         ckt->CKTrhs[node->number] += 1.0e10 * node->ic;
                        */
                        ckt->CKTrhs[node->number] = 1.0e10 * node->ic *
                                                      ckt->CKTsrcFact;
                        *(node->ptr) += 1.0e10;
                    } else {
                        /* Original code:
                          ckt->CKTrhs[node->number] = node->ic;
                        */
                        ckt->CKTrhs[node->number] = node->ic*ckt->CKTsrcFact; /* AlansFixes */
                        *(node->ptr) = 1;
                    }
                    /* DAG: Original CIDER fix. If above fix doesn't work,
                     * revert to this.
                     */
                    /*
                     *  ckt->CKTrhs[node->number] += 1.0e10 * node->ic;
                     *  *(node->ptr) += 1.0e10;
                     */
                }
            }
        }
    }
    /* SMPprint(ckt->CKTmatrix, stdout); if you want to debug, this is a
    good place to start ... */

    ckt->CKTstat->STATloadTime += SPfrontEnd->IFseconds()-startTime;
    return(OK);
}

static int
ZeroNoncurRow(SMPmatrix *matrix, CKTnode *nodes, int rownum)
{
    CKTnode     *n;
    double      *x;
    int         currents;

    currents = 0;
    for (n = nodes; n; n = n->next) {
        x = (double *) SMPfindElt(matrix, rownum, n->number, 0);
        if (x) {
            if (n->type == SP_CURRENT)
                currents = 1;
            else
                *x = 0.0;
        }
    }

    return currents;
}
