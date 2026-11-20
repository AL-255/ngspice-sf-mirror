/* ========================================================================== */
/* === KLU_refactor ========================================================= */
/* ========================================================================== */

/* Factor the matrix, after ordering and analyzing it with KLU_analyze, and
 * factoring it once with KLU_factor.  This routine cannot do any numerical
 * pivoting.  The pattern of the input matrix (Ap, Ai) must be identical to
 * the pattern given to KLU_factor.
 */

#include <string.h>
#include "klu_internal.h"


/* ========================================================================== */
/* === KLU_refactor ========================================================= */
/* ========================================================================== */

Int KLU_refactor        /* returns TRUE if successful, FALSE otherwise */
(
    /* inputs, not modified */
    Int Ap [ ],         /* size n+1, column pointers */
    Int Ai [ ],         /* size nz, row indices */
    double Ax [ ],
    KLU_symbolic *Symbolic,

    /* input/output */
    KLU_numeric *Numeric,
    KLU_common  *Common
)
{
    Entry ukk, ujk, s ;
    Entry *Offx, *Lx, *Ux, *X, *Az, *Udiag ;
    double *Rs ;
    Int *P, *Q, *R, *Pnum, *Offp, *Offi, *Ui, *Li, *Pinv, *Lip, *Uip, *Llen,
        *Ulen ;
    Unit **LUbx ;
    Unit *LU ;
    Int k1, k2, nk, k, block, oldcol, pend, oldrow, n, p, newrow, scale,
        nblocks, poff, i, j, up, ulen, llen, maxblock, nzoff ;

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    if (Common == NULL)
    {
        return (FALSE) ;
    }
    if (Common->status == KLU_EMPTY_MATRIX)
    {
        return (FALSE) ;
    }
    Common->status = KLU_OK ;

    if (Numeric == NULL)
    {
        /* invalid Numeric object */
        Common->status = KLU_INVALID ;
        return (FALSE) ;
    }

    Common->numerical_rank = EMPTY ;
    Common->singular_col = EMPTY ;

    Az = (Entry *) Ax ;

    /* ---------------------------------------------------------------------- */
    /* get the contents of the Symbolic object */
    /* ---------------------------------------------------------------------- */

    n = Symbolic->n ;
    P = Symbolic->P ;
    Q = Symbolic->Q ;
    R = Symbolic->R ;
    nblocks = Symbolic->nblocks ;
    maxblock = Symbolic->maxblock ;

    /* ---------------------------------------------------------------------- */
    /* get the contents of the Numeric object */
    /* ---------------------------------------------------------------------- */

    Pnum = Numeric->Pnum ;
    Offp = Numeric->Offp ;
    Offi = Numeric->Offi ;
    Offx = (Entry *) Numeric->Offx ;

    LUbx = (Unit **) Numeric->LUbx ;

    scale = Common->scale ;
    if (scale > 0)
    {
        /* factorization was not scaled, but refactorization is scaled */
        if (Numeric->Rs == NULL)
        {
            Numeric->Rs = KLU_malloc (n, sizeof (double), Common) ;
            if (Common->status < KLU_OK)
            {
                Common->status = KLU_OUT_OF_MEMORY ;
                return (FALSE) ;
            }
        }
    }
    else
    {
        /* no scaling for refactorization; ensure Numeric->Rs is freed.  This
         * does nothing if Numeric->Rs is already NULL. */
        Numeric->Rs = KLU_free (Numeric->Rs, n, sizeof (double), Common) ;
    }
    Rs = Numeric->Rs ;

    Pinv = Numeric->Pinv ;
    X = (Entry *) Numeric->Xwork ;
    Common->nrealloc = 0 ;
    Udiag = Numeric->Udiag ;
    nzoff = Symbolic->nzoff ;

    /* ---------------------------------------------------------------------- */
    /* check the input matrix compute the row scale factors, Rs */
    /* ---------------------------------------------------------------------- */

    double *invRs = NULL ;
#if !defined(COMPLEX) && NGSPICE_OPT_GENERAL
    static int partial_init = 0, partial_disabled = 0 ;
    if (!partial_init)
    {
        partial_disabled = (getenv ("NGSPICE_NO_PARTIAL_KLU") != NULL) ;
        partial_init = 1 ;
    }
    int use_partial = (!partial_disabled && scale > 0 &&
        Numeric->ref_prev_valid && Numeric->ref_blk_col != NULL &&
        n > 0 && Ap != NULL && Ai != NULL && Az != NULL && Rs != NULL &&
        Ap [0] == 0 && Ap [n] == Numeric->ref_nz &&
        Numeric->ref_az_prev != NULL && Numeric->ref_inv_rs != NULL &&
        Numeric->ref_dirty != NULL) ;
#else
    int use_partial = 0 ;
#endif

    /* do no scale, or check the input matrix, if scale < 0 */
    if (scale > 0)
    {
        if (n <= 0 || Ap == NULL || Ai == NULL || Az == NULL || Rs == NULL ||
            Ap [0] != 0 || Ap [n] < 0)
        {
            Common->status = KLU_INVALID ;
            return (FALSE) ;
        }
#if !defined(COMPLEX) && NGSPICE_OPT_GENERAL
        if (use_partial)
        {
            invRs = Numeric->ref_inv_rs ;
        }
        else
#endif
        {
        Int nz = Ap [n] ;
        for (k = 0 ; k < n ; k++)
        {
            Rs [k] = 0.0 ;
        }
        if (scale == 1)
        {
            const Int * restrict Air = Ai ;
            double * restrict Rsr = Rs ;
            for (p = 0 ; p < nz ; p++)
            {
                double a ;
#ifndef COMPLEX
                a = fabs (Az [p]) ;
#else
                ABS (a, Az [p]) ;
#endif
                Rsr [Air [p]] += a ;
            }
        }
        else
        {
            const Int * restrict Air = Ai ;
            double * restrict Rsr = Rs ;
            for (p = 0 ; p < nz ; p++)
            {
                double a ;
#ifndef COMPLEX
                a = fabs (Az [p]) ;
#else
                ABS (a, Az [p]) ;
#endif
                Int r = Air [p] ;
                double cur = Rsr [r] ;
                Rsr [r] = (a > cur) ? a : cur ;
            }
        }
        /* Numeric->Xwork has size 4*n Entry's; X[0..n-1] is used for scatter,
         * and Iwork is unused during refactor, so (double*)(X + n) has >= 3*n doubles. */
        invRs = (double *) (X + n) ;
        for (k = 0 ; k < n ; k++)
        {
            double rk = Rs [k] ;
            if (rk == 0.0)
            {
                Rs [k] = 1.0 ;
                invRs [k] = 1.0 ;
            }
            else
            {
                invRs [k] = 1.0 / rk ;
            }
        }
        }
    }
    else if (scale == 0)
    {
        if (n <= 0 || Ap == NULL || Ai == NULL || Az == NULL ||
            Ap [0] != 0 || Ap [n] < 0)
        {
            Common->status = KLU_INVALID ;
            return (FALSE) ;
        }
    }

    /* ---------------------------------------------------------------------- */
    /* build lazy precomputed scatter map for KLU_refactor (reused every call) */
    /* ---------------------------------------------------------------------- */

#if NGSPICE_OPT_GENERAL
    if (Numeric->ref_blk_col == NULL && n > 0 && Ap != NULL && Ai != NULL &&
        Ap [0] == 0 && Ap [n] >= 0)
    {
        Int nz_map = Ap [n] ;
        Int *r_off_p   = KLU_malloc (nzoff + 1, sizeof (Int), Common) ;
        Int *r_blk_col = KLU_malloc (n + 1, sizeof (Int), Common) ;
        Int *r_blk_p   = KLU_malloc (nz_map + 1, sizeof (Int), Common) ;
        Int *r_blk_row = KLU_malloc (nz_map + 1, sizeof (Int), Common) ;
        if (r_off_p != NULL && r_blk_col != NULL && r_blk_p != NULL &&
            r_blk_row != NULL && Common->status >= KLU_OK)
        {
            Int poff_b = 0, bidx = 0 ;
            for (block = 0 ; block < nblocks ; block++)
            {
                k1 = R [block] ;
                k2 = R [block+1] ;
                for (k = k1 ; k < k2 ; k++)
                {
                    r_blk_col [k] = bidx ;
                    oldcol = Q [k] ;
                    pend = Ap [oldcol+1] ;
                    for (p = Ap [oldcol] ; p < pend ; p++)
                    {
                        newrow = Pinv [Ai [p]] - k1 ;
                        if (newrow < 0 && poff_b < nzoff)
                        {
                            r_off_p [poff_b++] = p ;
                        }
                        else
                        {
                            r_blk_p [bidx] = p ;
                            r_blk_row [bidx] = newrow ;
                            bidx++ ;
                        }
                    }
                }
            }
            r_blk_col [n] = bidx ;
            Numeric->ref_nz = nz_map ;
            Numeric->ref_poff = poff_b ;
            Numeric->ref_off_p = r_off_p ;
            Numeric->ref_blk_col = r_blk_col ;
            Numeric->ref_blk_p = r_blk_p ;
            Numeric->ref_blk_row = r_blk_row ;
#ifndef COMPLEX
            Numeric->ref_inv_udiag = KLU_malloc (n + 1, sizeof (double), Common) ;
            if (Numeric->ref_inv_udiag == NULL || Common->status < KLU_OK)
            {
                KLU_free (Numeric->ref_inv_udiag, n + 1, sizeof (double), Common) ;
                Numeric->ref_inv_udiag = NULL ;
                Common->status = KLU_OK ;
            }
            if (!partial_disabled)
            {
                Numeric->ref_az_prev     = KLU_malloc (nz_map + 1, sizeof (double), Common) ;
                Numeric->ref_inv_rs      = KLU_malloc (n + 1, sizeof (double), Common) ;
                Numeric->ref_inv_rs_perm = KLU_malloc (n + 1, sizeof (double), Common) ;
                Numeric->ref_dirty       = KLU_malloc (n + 1, sizeof (unsigned char), Common) ;
                if (Numeric->ref_az_prev == NULL || Numeric->ref_inv_rs == NULL ||
                    Numeric->ref_inv_rs_perm == NULL || Numeric->ref_dirty == NULL ||
                    Common->status < KLU_OK)
                {
                    KLU_free (Numeric->ref_az_prev,     nz_map + 1, sizeof (double), Common) ;
                    KLU_free (Numeric->ref_inv_rs,      n + 1, sizeof (double), Common) ;
                    KLU_free (Numeric->ref_inv_rs_perm, n + 1, sizeof (double), Common) ;
                    KLU_free (Numeric->ref_dirty,       n + 1, sizeof (unsigned char), Common) ;
                    Numeric->ref_az_prev     = NULL ;
                    Numeric->ref_inv_rs      = NULL ;
                    Numeric->ref_inv_rs_perm = NULL ;
                    Numeric->ref_dirty       = NULL ;
                    Common->status = KLU_OK ;
                }
            }
#endif
        }
        else
        {
            KLU_free (r_off_p,   nzoff + 1, sizeof (Int), Common) ;
            KLU_free (r_blk_col, n + 1, sizeof (Int), Common) ;
            KLU_free (r_blk_p,   nz_map + 1, sizeof (Int), Common) ;
            KLU_free (r_blk_row, nz_map + 1, sizeof (Int), Common) ;
            Common->status = KLU_OK ;
        }
    }

    const Int * restrict ref_off_p   = Numeric->ref_off_p ;
    const Int * restrict ref_blk_col = Numeric->ref_blk_col ;
    const Int * restrict ref_blk_p   = Numeric->ref_blk_p ;
    const Int * restrict ref_blk_row = Numeric->ref_blk_row ;
    int have_ref_map = (ref_blk_col != NULL && Numeric->ref_nz == Ap [n]) ;
#else
    const Int * restrict ref_off_p   = NULL ;
    const Int * restrict ref_blk_col = NULL ;
    const Int * restrict ref_blk_p   = NULL ;
    const Int * restrict ref_blk_row = NULL ;
    int have_ref_map = 0 ;
#endif

    /* ---------------------------------------------------------------------- */
    /* clear workspace X */
    /* ---------------------------------------------------------------------- */

    for (k = 0 ; k < maxblock ; k++)
    {
        /* X [k] = 0 */
        CLEAR (X [k]) ;
    }

    poff = 0 ;

    /* ---------------------------------------------------------------------- */
    /* factor each block */
    /* ---------------------------------------------------------------------- */

    if (scale <= 0)
    {

        /* ------------------------------------------------------------------ */
        /* no scaling */
        /* ------------------------------------------------------------------ */

        if (have_ref_map)
        {
            Int npoff = Numeric->ref_poff ;
            for (p = 0 ; p < npoff ; p++)
            {
                Offx [p] = Az [ref_off_p [p]] ;
            }
            poff = npoff ;
        }

        for (block = 0 ; block < nblocks ; block++)
        {

            /* -------------------------------------------------------------- */
            /* the block is from rows/columns k1 to k2-1 */
            /* -------------------------------------------------------------- */

            k1 = R [block] ;
            k2 = R [block+1] ;
            nk = k2 - k1 ;

            if (nk == 1)
            {

                /* ---------------------------------------------------------- */
                /* singleton case */
                /* ---------------------------------------------------------- */

                CLEAR (s) ;
                if (have_ref_map)
                {
                    Int t0 = ref_blk_col [k1] ;
                    Int t1 = ref_blk_col [k1 + 1] ;
                    for (p = t0 ; p < t1 ; p++)
                    {
                        s = Az [ref_blk_p [p]] ;
                    }
                }
                else
                {
                    oldcol = Q [k1] ;
                    pend = Ap [oldcol+1] ;
                    for (p = Ap [oldcol] ; p < pend ; p++)
                    {
                        newrow = Pinv [Ai [p]] - k1 ;
                        if (newrow < 0 && poff < nzoff)
                        {
                            /* entry in off-diagonal block */
                            Offx [poff] = Az [p] ;
                            poff++ ;
                        }
                        else
                        {
                            /* singleton */
                            s = Az [p] ;
                        }
                    }
                }
                Udiag [k1] = s ;
#if !defined(COMPLEX) && NGSPICE_OPT_GENERAL
                if (Numeric->ref_inv_udiag != NULL)
                    Numeric->ref_inv_udiag [k1] = (s != 0.0) ? (1.0 / s) : 0.0 ;
#endif

            }
            else
            {

                /* ---------------------------------------------------------- */
                /* construct and factor the kth block */
                /* ---------------------------------------------------------- */

                Lip  = Numeric->Lip  + k1 ;
                Llen = Numeric->Llen + k1 ;
                Uip  = Numeric->Uip  + k1 ;
                Ulen = Numeric->Ulen + k1 ;
                LU = LUbx [block] ;

                for (k = 0 ; k < nk ; k++)
                {

                    /* ------------------------------------------------------ */
                    /* scatter kth column of the block into workspace X */
                    /* ------------------------------------------------------ */

                    if (have_ref_map)
                    {
                        Int t0 = ref_blk_col [k + k1] ;
                        Int t1 = ref_blk_col [k + k1 + 1] ;
                        for (p = t0 ; p < t1 ; p++)
                        {
                            X [ref_blk_row [p]] = Az [ref_blk_p [p]] ;
                        }
                    }
                    else
                    {
                        oldcol = Q [k+k1] ;
                        pend = Ap [oldcol+1] ;
                        for (p = Ap [oldcol] ; p < pend ; p++)
                        {
                            newrow = Pinv [Ai [p]] - k1 ;
                            if (newrow < 0 && poff < nzoff)
                            {
                                /* entry in off-diagonal block */
                                Offx [poff] = Az [p] ;
                                poff++ ;
                            }
                            else
                            {
                                /* (newrow,k) is an entry in the block */
                                X [newrow] = Az [p] ;
                            }
                        }
                    }

                    /* ------------------------------------------------------ */
                    /* compute kth column of U, and update kth column of A */
                    /* ------------------------------------------------------ */

                    GET_POINTER (LU, Uip, Ulen, Ui, Ux, k, ulen) ;
                    for (up = 0 ; up < ulen ; up++)
                    {
                        j = Ui [up] ;
                        ujk = X [j] ;
                        /* X [j] = 0 */
                        CLEAR (X [j]) ;
                        Ux [up] = ujk ;
                        if (IS_NONZERO (ujk))
                        {
                            GET_POINTER (LU, Lip, Llen, Li, Lx, j, llen) ;
#if !defined(COMPLEX) && NGSPICE_OPT_GENERAL
                            double * restrict Xr = X ;
                            const Int * restrict Lir = Li ;
                            const double * restrict Lxr = Lx ;
                            for (p = 0 ; p + 3 < llen ; p += 4)
                            {
                                Int i0 = Lir [p] ;
                                Int i1 = Lir [p + 1] ;
                                Int i2 = Lir [p + 2] ;
                                Int i3 = Lir [p + 3] ;
                                double x0 = Xr [i0] ;
                                double x1 = Xr [i1] ;
                                double x2 = Xr [i2] ;
                                double x3 = Xr [i3] ;
                                x0 -= Lxr [p]     * ujk ;
                                x1 -= Lxr [p + 1] * ujk ;
                                x2 -= Lxr [p + 2] * ujk ;
                                x3 -= Lxr [p + 3] * ujk ;
                                Xr [i0] = x0 ;
                                Xr [i1] = x1 ;
                                Xr [i2] = x2 ;
                                Xr [i3] = x3 ;
                            }
                            for ( ; p < llen ; p++)
                            {
                                Xr [Lir [p]] -= Lxr [p] * ujk ;
                            }
#else
                            for (p = 0 ; p < llen ; p++)
                            {
                                /* X [Li [p]] -= Lx [p] * ujk */
                                MULT_SUB (X [Li [p]], Lx [p], ujk) ;
                            }
#endif
                        }
                    }
                    /* get the diagonal entry of U */
                    ukk = X [k] ;
                    /* X [k] = 0 */
                    CLEAR (X [k]) ;
                    /* singular case */
                    if (IS_ZERO (ukk))
                    {
                        /* matrix is numerically singular */
                        Common->status = KLU_SINGULAR ;
                        if (Common->numerical_rank == EMPTY)
                        {
                            Common->numerical_rank = k+k1 ;
                            Common->singular_col = Q [k+k1] ;
                        }
                        if (Common->halt_if_singular)
                        {
                            /* do not continue the factorization */
                            Numeric->ref_prev_valid = 0 ;
                            Numeric->ref_inv_udiag_valid = 0 ;
                            return (FALSE) ;
                        }
                    }
                    Udiag [k+k1] = ukk ;
                    /* gather and divide by pivot to get kth column of L */
                    GET_POINTER (LU, Lip, Llen, Li, Lx, k, llen) ;
#if !defined(COMPLEX) && NGSPICE_OPT_GENERAL
                    if (!IS_ZERO (ukk))
                    {
                        double inv_ukk = 1.0 / ukk ;
                        if (Numeric->ref_inv_udiag != NULL)
                            Numeric->ref_inv_udiag [k + k1] = inv_ukk ;
                        if (llen > 0)
                        {
                            double * restrict Xr = X ;
                            const Int * restrict Lir = Li ;
                            double * restrict Lxr = Lx ;
                            for (p = 0 ; p + 3 < llen ; p += 4)
                            {
                                Int i0 = Lir [p] ;
                                Int i1 = Lir [p + 1] ;
                                Int i2 = Lir [p + 2] ;
                                Int i3 = Lir [p + 3] ;
                                double x0 = Xr [i0] ;
                                double x1 = Xr [i1] ;
                                double x2 = Xr [i2] ;
                                double x3 = Xr [i3] ;
                                Xr [i0] = 0.0 ;
                                Xr [i1] = 0.0 ;
                                Xr [i2] = 0.0 ;
                                Xr [i3] = 0.0 ;
                                Lxr [p]     = x0 * inv_ukk ;
                                Lxr [p + 1] = x1 * inv_ukk ;
                                Lxr [p + 2] = x2 * inv_ukk ;
                                Lxr [p + 3] = x3 * inv_ukk ;
                            }
                            for ( ; p < llen ; p++)
                            {
                                i = Lir [p] ;
                                Lxr [p] = Xr [i] * inv_ukk ;
                                Xr [i] = 0.0 ;
                            }
                        }
                    }
                    else
#endif
                    for (p = 0 ; p < llen ; p++)
                    {
                        i = Li [p] ;
                        DIV (Lx [p], X [i], ukk) ;
                        CLEAR (X [i]) ;
                    }

                }
            }
        }

    }
    else
    {

        /* ------------------------------------------------------------------ */
        /* scaling */
        /* ------------------------------------------------------------------ */

        if (have_ref_map)
        {
            Int npoff = Numeric->ref_poff ;
            const Int * restrict Air = Ai ;
            const double * restrict iRs = invRs ;
#if !defined(COMPLEX) && NGSPICE_OPT_GENERAL
            if (use_partial)
            {
                double * restrict azp = Numeric->ref_az_prev ;
                for (p = 0 ; p < npoff ; p++)
                {
                    Int idx = ref_off_p [p] ;
                    double v = Az [idx] ;
                    if (v != azp [idx])
                    {
                        azp [idx] = v ;
                        Offx [p] = v * iRs [Air [idx]] ;
                    }
                }
            }
            else
#endif
            for (p = 0 ; p < npoff ; p++)
            {
                Int idx = ref_off_p [p] ;
                oldrow = Air [idx] ;
#ifndef COMPLEX
                Offx [p] = Az [idx] * iRs [oldrow] ;
#else
                Offx [p].Real = Az [idx].Real * iRs [oldrow] ;
                Offx [p].Imag = Az [idx].Imag * iRs [oldrow] ;
#endif
            }
            poff = npoff ;
        }

        for (block = 0 ; block < nblocks ; block++)
        {

            /* -------------------------------------------------------------- */
            /* the block is from rows/columns k1 to k2-1 */
            /* -------------------------------------------------------------- */

            k1 = R [block] ;
            k2 = R [block+1] ;
            nk = k2 - k1 ;

            if (nk == 1)
            {

                /* ---------------------------------------------------------- */
                /* singleton case */
                /* ---------------------------------------------------------- */

                CLEAR (s) ;
                if (have_ref_map)
                {
                    Int t0 = ref_blk_col [k1] ;
                    Int t1 = ref_blk_col [k1 + 1] ;
#if !defined(COMPLEX) && NGSPICE_OPT_GENERAL
                    if (use_partial && t0 + 1 == t1)
                    {
                        Int idx = ref_blk_p [t0] ;
                        double v = Az [idx] ;
                        if (v != Numeric->ref_az_prev [idx])
                        {
                            double u = v * invRs [Ai [idx]] ;
                            Numeric->ref_az_prev [idx] = v ;
                            Udiag [k1] = u ;
                            if (Numeric->ref_inv_udiag != NULL && u != 0.0)
                                Numeric->ref_inv_udiag [k1] = 1.0 / u ;
                        }
                        continue ;
                    }
#endif
                    for (p = t0 ; p < t1 ; p++)
                    {
                        Int idx = ref_blk_p [p] ;
                        oldrow = Ai [idx] ;
#ifndef COMPLEX
                        s = Az [idx] * invRs [oldrow] ;
#if NGSPICE_OPT_GENERAL
                        if (use_partial) Numeric->ref_az_prev [idx] = Az [idx] ;
#endif
#else
                        s.Real = Az [idx].Real * invRs [oldrow] ;
                        s.Imag = Az [idx].Imag * invRs [oldrow] ;
#endif
                    }
                }
                else
                {
                    oldcol = Q [k1] ;
                    pend = Ap [oldcol+1] ;
                    for (p = Ap [oldcol] ; p < pend ; p++)
                    {
                        oldrow = Ai [p] ;
                        newrow = Pinv [oldrow] - k1 ;
                        if (newrow < 0 && poff < nzoff)
                        {
                            /* entry in off-diagonal block */
                            /* Offx [poff] = Az [p] / Rs [oldrow] */
#ifndef COMPLEX
                            Offx [poff] = Az [p] * invRs [oldrow] ;
#else
                            Offx [poff].Real = Az [p].Real * invRs [oldrow] ;
                            Offx [poff].Imag = Az [p].Imag * invRs [oldrow] ;
#endif
                            poff++ ;
                        }
                        else
                        {
                            /* singleton */
                            /* s = Az [p] / Rs [oldrow] */
#ifndef COMPLEX
                            s = Az [p] * invRs [oldrow] ;
#else
                            s.Real = Az [p].Real * invRs [oldrow] ;
                            s.Imag = Az [p].Imag * invRs [oldrow] ;
#endif
                        }
                    }
                }
                Udiag [k1] = s ;
#if !defined(COMPLEX) && NGSPICE_OPT_GENERAL
                if (Numeric->ref_inv_udiag != NULL)
                    Numeric->ref_inv_udiag [k1] = (s != 0.0) ? (1.0 / s) : 0.0 ;
#endif

            }
            else
            {

                /* ---------------------------------------------------------- */
                /* construct and factor the kth block */
                /* ---------------------------------------------------------- */

                Lip  = Numeric->Lip  + k1 ;
                Llen = Numeric->Llen + k1 ;
                Uip  = Numeric->Uip  + k1 ;
                Ulen = Numeric->Ulen + k1 ;
                LU = LUbx [block] ;
#if !defined(COMPLEX) && NGSPICE_OPT_GENERAL
                unsigned char * restrict col_dirty = Numeric->ref_dirty ;
                double * restrict azp = Numeric->ref_az_prev ;
#endif

                for (k = 0 ; k < nk ; k++)
                {

                    /* ------------------------------------------------------ */
                    /* Incremental column-dirty check: skip unchanged columns */
                    /* ------------------------------------------------------ */
#if !defined(COMPLEX) && NGSPICE_OPT_GENERAL
                    if (use_partial)
                    {
                        Int t0 = ref_blk_col [k + k1] ;
                        Int t1 = ref_blk_col [k + k1 + 1] ;
                        int dirty = 0 ;
                        for (p = t0 ; p < t1 ; p++)
                        {
                            Int idx = ref_blk_p [p] ;
                            if (Az [idx] != azp [idx])
                            {
                                dirty = 1 ;
                                break ;
                            }
                        }
                        GET_POINTER (LU, Uip, Ulen, Ui, Ux, k, ulen) ;
                        if (!dirty)
                        {
                            const Int * restrict Uir = Ui ;
                            for (up = 0 ; up < ulen ; up++)
                            {
                                if (col_dirty [Uir [up]])
                                {
                                    dirty = 1 ;
                                    break ;
                                }
                            }
                        }
                        if (!dirty)
                        {
                            col_dirty [k] = 0 ;
                            continue ;
                        }
                        col_dirty [k] = 1 ;
                        const Int * restrict Air = Ai ;
                        const double * restrict iRs = invRs ;
                        for (p = t0 ; p < t1 ; p++)
                        {
                            Int idx = ref_blk_p [p] ;
                            double v = Az [idx] ;
                            azp [idx] = v ;
                            X [ref_blk_row [p]] = v * iRs [Air [idx]] ;
                        }
                    }
                    else
#endif
                    /* ------------------------------------------------------ */
                    /* scatter kth column of the block into workspace X */
                    /* ------------------------------------------------------ */

                    if (have_ref_map)
                    {
                        Int t0 = ref_blk_col [k + k1] ;
                        Int t1 = ref_blk_col [k + k1 + 1] ;
                        const Int * restrict Air = Ai ;
                        const double * restrict iRs = invRs ;
                        for (p = t0 ; p < t1 ; p++)
                        {
                            Int idx = ref_blk_p [p] ;
                            oldrow = Air [idx] ;
#ifndef COMPLEX
                            X [ref_blk_row [p]] = Az [idx] * iRs [oldrow] ;
#else
                            X [ref_blk_row [p]].Real = Az [idx].Real * iRs [oldrow] ;
                            X [ref_blk_row [p]].Imag = Az [idx].Imag * iRs [oldrow] ;
#endif
                        }
                    }
                    else
                    {
                        oldcol = Q [k+k1] ;
                        pend = Ap [oldcol+1] ;
                        for (p = Ap [oldcol] ; p < pend ; p++)
                        {
                            oldrow = Ai [p] ;
                            newrow = Pinv [oldrow] - k1 ;
                            if (newrow < 0 && poff < nzoff)
                            {
                                /* entry in off-diagonal part */
                                /* Offx [poff] = Az [p] / Rs [oldrow] */
#ifndef COMPLEX
                                Offx [poff] = Az [p] * invRs [oldrow] ;
#else
                                Offx [poff].Real = Az [p].Real * invRs [oldrow] ;
                                Offx [poff].Imag = Az [p].Imag * invRs [oldrow] ;
#endif
                                poff++ ;
                            }
                            else
                            {
                                /* (newrow,k) is an entry in the block */
                                /* X [newrow] = Az [p] / Rs [oldrow] */
#ifndef COMPLEX
                                X [newrow] = Az [p] * invRs [oldrow] ;
#else
                                X [newrow].Real = Az [p].Real * invRs [oldrow] ;
                                X [newrow].Imag = Az [p].Imag * invRs [oldrow] ;
#endif
                            }
                        }
                    }

                    /* ------------------------------------------------------ */
                    /* compute kth column of U, and update kth column of A */
                    /* ------------------------------------------------------ */

                    if (!use_partial)
                    {
                        GET_POINTER (LU, Uip, Ulen, Ui, Ux, k, ulen) ;
                    }
                    for (up = 0 ; up < ulen ; up++)
                    {
                        j = Ui [up] ;
                        ujk = X [j] ;
                        /* X [j] = 0 */
                        CLEAR (X [j]) ;
                        Ux [up] = ujk ;
                        if (IS_NONZERO (ujk))
                        {
                            GET_POINTER (LU, Lip, Llen, Li, Lx, j, llen) ;
#if !defined(COMPLEX) && NGSPICE_OPT_GENERAL
                            double * restrict Xr = X ;
                            const Int * restrict Lir = Li ;
                            const double * restrict Lxr = Lx ;
                            for (p = 0 ; p + 3 < llen ; p += 4)
                            {
                                Int i0 = Lir [p] ;
                                Int i1 = Lir [p + 1] ;
                                Int i2 = Lir [p + 2] ;
                                Int i3 = Lir [p + 3] ;
                                double x0 = Xr [i0] ;
                                double x1 = Xr [i1] ;
                                double x2 = Xr [i2] ;
                                double x3 = Xr [i3] ;
                                x0 -= Lxr [p]     * ujk ;
                                x1 -= Lxr [p + 1] * ujk ;
                                x2 -= Lxr [p + 2] * ujk ;
                                x3 -= Lxr [p + 3] * ujk ;
                                Xr [i0] = x0 ;
                                Xr [i1] = x1 ;
                                Xr [i2] = x2 ;
                                Xr [i3] = x3 ;
                            }
                            for ( ; p < llen ; p++)
                            {
                                Xr [Lir [p]] -= Lxr [p] * ujk ;
                            }
#else
                            for (p = 0 ; p < llen ; p++)
                            {
                                /* X [Li [p]] -= Lx [p] * ujk */
                                MULT_SUB (X [Li [p]], Lx [p], ujk) ;
                            }
#endif
                        }
                    }
                    /* get the diagonal entry of U */
                    ukk = X [k] ;
                    /* X [k] = 0 */
                    CLEAR (X [k]) ;
                    /* singular case */
                    if (IS_ZERO (ukk))
                    {
                        /* matrix is numerically singular */
                        Common->status = KLU_SINGULAR ;
                        if (Common->numerical_rank == EMPTY)
                        {
                            Common->numerical_rank = k+k1 ;
                            Common->singular_col = Q [k+k1] ;
                        }
                        if (Common->halt_if_singular)
                        {
                            /* do not continue the factorization */
                            Numeric->ref_prev_valid = 0 ;
                            Numeric->ref_inv_udiag_valid = 0 ;
                            return (FALSE) ;
                        }
                    }
                    Udiag [k+k1] = ukk ;
                    /* gather and divide by pivot to get kth column of L */
                    GET_POINTER (LU, Lip, Llen, Li, Lx, k, llen) ;
#if !defined(COMPLEX) && NGSPICE_OPT_GENERAL
                    if (!IS_ZERO (ukk))
                    {
                        double inv_ukk = 1.0 / ukk ;
                        if (Numeric->ref_inv_udiag != NULL)
                            Numeric->ref_inv_udiag [k + k1] = inv_ukk ;
                        if (llen > 0)
                        {
                            double * restrict Xr = X ;
                            const Int * restrict Lir = Li ;
                            double * restrict Lxr = Lx ;
                            for (p = 0 ; p + 3 < llen ; p += 4)
                            {
                                Int i0 = Lir [p] ;
                                Int i1 = Lir [p + 1] ;
                                Int i2 = Lir [p + 2] ;
                                Int i3 = Lir [p + 3] ;
                                double x0 = Xr [i0] ;
                                double x1 = Xr [i1] ;
                                double x2 = Xr [i2] ;
                                double x3 = Xr [i3] ;
                                Xr [i0] = 0.0 ;
                                Xr [i1] = 0.0 ;
                                Xr [i2] = 0.0 ;
                                Xr [i3] = 0.0 ;
                                Lxr [p]     = x0 * inv_ukk ;
                                Lxr [p + 1] = x1 * inv_ukk ;
                                Lxr [p + 2] = x2 * inv_ukk ;
                                Lxr [p + 3] = x3 * inv_ukk ;
                            }
                            for ( ; p < llen ; p++)
                            {
                                i = Lir [p] ;
                                Lxr [p] = Xr [i] * inv_ukk ;
                                Xr [i] = 0.0 ;
                            }
                        }
                    }
                    else
#endif
                    for (p = 0 ; p < llen ; p++)
                    {
                        i = Li [p] ;
                        DIV (Lx [p], X [i], ukk) ;
                        CLEAR (X [i]) ;
                    }
                }
            }
        }
    }

    /* ---------------------------------------------------------------------- */
    /* permute scale factors Rs according to pivotal row order */
    /* ---------------------------------------------------------------------- */

    if (scale > 0 && !use_partial)
    {
#if !defined(COMPLEX) && NGSPICE_OPT_GENERAL
        if (!partial_disabled && Common->status == KLU_OK &&
            have_ref_map && Numeric->ref_az_prev != NULL &&
            Numeric->ref_inv_rs != NULL && Numeric->ref_inv_rs_perm != NULL &&
            Numeric->ref_dirty != NULL)
        {
            memcpy (Numeric->ref_inv_rs, invRs, (size_t) n * sizeof (double)) ;
            for (k = 0 ; k < n ; k++)
                Numeric->ref_inv_rs_perm [k] = invRs [Pnum [k]] ;
            memcpy (Numeric->ref_az_prev, Az, (size_t) Ap [n] * sizeof (double)) ;
            Numeric->ref_prev_valid = 1 ;
        }
#endif
        for (k = 0 ; k < n ; k++)
        {
            REAL (X [k]) = Rs [Pnum [k]] ;
        }
        for (k = 0 ; k < n ; k++)
        {
            Rs [k] = REAL (X [k]) ;
        }
    }
#if !defined(COMPLEX) && NGSPICE_OPT_GENERAL
    if (Common->status == KLU_OK && scale > 0 && Numeric->ref_inv_udiag != NULL)
        Numeric->ref_inv_udiag_valid = 1 ;
    else
        Numeric->ref_inv_udiag_valid = 0 ;
#endif

#ifndef NDEBUG
    ASSERT (Offp [n] == poff) ;
    ASSERT (Symbolic->nzoff == poff) ;
    PRINTF (("\n------------------- Off diagonal entries, new:\n")) ;
    ASSERT (KLU_valid (n, Offp, Offi, Offx)) ;
    if (Common->status == KLU_OK)
    {
        PRINTF (("\n ########### KLU_BTF_REFACTOR done, nblocks %d\n",nblocks));
        for (block = 0 ; block < nblocks ; block++)
        {
            k1 = R [block] ;
            k2 = R [block+1] ;
            nk = k2 - k1 ;
            PRINTF ((
                "\n================KLU_refactor output: k1 %d k2 %d nk %d\n",
                k1, k2, nk)) ;
            if (nk == 1)
            {
                PRINTF (("singleton  ")) ;
                PRINT_ENTRY (Udiag [k1]) ;
            }
            else
            {
                Lip = Numeric->Lip + k1 ;
                Llen = Numeric->Llen + k1 ;
                LU = (Unit *) Numeric->LUbx [block] ;
                PRINTF (("\n---- L block %d\n", block)) ;
                ASSERT (KLU_valid_LU (nk, TRUE, Lip, Llen, LU)) ;
                Uip = Numeric->Uip + k1 ;
                Ulen = Numeric->Ulen + k1 ;
                PRINTF (("\n---- U block %d\n", block)) ;
                ASSERT (KLU_valid_LU (nk, FALSE, Uip, Ulen, LU)) ;
            }
        }
    }
#endif

    return (TRUE) ;
}
