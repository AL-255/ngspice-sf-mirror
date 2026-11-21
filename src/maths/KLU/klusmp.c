/* KLU to SMP Interface
 * Francesco Lannutti
 * July 2020
*/

#include "ngspice/config.h"
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "ngspice/spmatrix.h"
#include "../sparse/spdefs.h"
#include "ngspice/smpdefs.h"
#include "ngspice/fteext.h"

#if defined (_MSC_VER)
extern double scalbn(double, int);
#define logb _logb
extern double logb(double);
#endif

#ifdef HAS_WINGUI
#include "ngspice/wstdio.h"
#endif

#include "ngspice/mif.h"
#include "ngspice/evt.h"

static int
CircuitIsDigital (void)
{
#ifdef XSPICE
    return g_mif_info.ckt && g_mif_info.ckt->evt && g_mif_info.ckt->evt->counts.num_nodes != 0 ;
#else
    return 0 ;
#endif
}

static void LoadGmin_CSC (double **diag, unsigned int n, double Gmin) ;
static void LoadGmin (SMPmatrix *eMatrix, double Gmin) ;

typedef struct sElement {
    unsigned int row ;
    unsigned int col ;
    double *pointer ;
    unsigned int group ;
} Element ;

static int
CompareRow (const void *a, const void *b)
{
    Element *A = (Element *) a ;
    Element *B = (Element *) b ;

    return
        (A->row > B->row) ?  1 :
        (A->row < B->row) ? -1 :
        0 ;
}

static int
CompareColumn (const void *a, const void *b)
{
    Element *A = (Element *) a ;
    Element *B = (Element *) b ;

    return
        (A->col > B->col) ?  1 :
        (A->col < B->col) ? -1 :
        0 ;
}

static void
Compress (unsigned int *Ai, unsigned int *Bp, unsigned int n, unsigned int nz)
{
    unsigned int i, j ;

    for (i = 0 ; i <= Ai [0] ; i++)
        Bp [i] = 0 ;

    j = Ai [0] + 1 ;
    for (i = 1 ; i < nz ; i++)
    {
        if (Ai [i] == Ai [i - 1] + 1)
        {
            Bp [j] = i ;
            j++ ;
        }
        else if (Ai [i] > Ai [i - 1] + 1)
        {
            for ( ; j <= Ai [i] ; j++)
                Bp [j] = i ;
        }
    }

    for ( ; j <= n ; j++)
        Bp [j] = i ;
}

int
BindCompare (const void *a, const void *b)
{
    BindElement *A = (BindElement *) a ;
    BindElement *B = (BindElement *) b ;

    return
        (A->COO > B->COO) ?  1 :
        (A->COO < B->COO) ? -1 :
        0 ;
}

#ifdef CIDER
int
BindCompareKLUforCIDER (const void *a, const void *b)
{
    BindElementKLUforCIDER *A = (BindElementKLUforCIDER *) a ;
    BindElementKLUforCIDER *B = (BindElementKLUforCIDER *) b ;

    return
        (A->COO > B->COO) ?  1 :
        (A->COO < B->COO) ? -1 :
        0 ;
}

int
BindKluCompareCSCKLUforCIDER (const void *a, const void *b)
{
    BindElementKLUforCIDER *A = (BindElementKLUforCIDER *) a ;
    BindElementKLUforCIDER *B = (BindElementKLUforCIDER *) b ;

    return
        (A->CSC_Complex > B->CSC_Complex) ?  1 :
        (A->CSC_Complex < B->CSC_Complex) ? -1 :
        0 ;
}
#endif

void SMPconvertCOOtoCSC (SMPmatrix *Matrix)
{
    Element *MatrixCOO ;
    KluLinkedListCOO *current, *temp ;
    unsigned int *Ap_COO, current_group, i, j ;

    if (Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ == 0) {
        /* Assign N and NZ */
        Matrix->SMPkluMatrix->KLUmatrixN = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixNrhs = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixNZ = 0 ;

        /* Allocate Diag Gmin CSC Vector */
        Matrix->SMPkluMatrix->KLUmatrixDiag = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAxTls = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixRhsTls = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAxTlsStride = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixRhsTlsStride = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixTlsThreads = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixAxResConst = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixResConstValid = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixResVersion = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixAxCapUnit = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixCapUnitValid = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixCapVersion = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixAxResCapBase = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixResCapAg0 = 0.0 ;
        Matrix->SMPkluMatrix->KLUmatrixResCapValid = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixResCapHasRes = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixRhsCapCache = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixRhsCapLen = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixRhsCapValid = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixIdentityMap = -1 ;
        Matrix->SMPkluMatrix->KLUmatrixAxLatentDC = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAxLatentCap = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixRhsLatentDC = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixRhsLatentCap = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixRhsLatentLen = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixLatentCount = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixLatentOps = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixResCapHasLatent = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixSubCache = NULL ;

        /* Allocate the temporary COO Column Index */
        Ap_COO = NULL ;

        /* Allocate the needed KLU data structures */
        Matrix->SMPkluMatrix->KLUmatrixAp = (int *) malloc (sizeof (int)) ;
        Matrix->SMPkluMatrix->KLUmatrixAi = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixBindStructCOO = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAx = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAxComplex = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixIntermediate = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex = NULL ;

        /* Set the Matrix as Real */
        Matrix->SMPkluMatrix->KLUmatrixIsComplex = KLUmatrixReal ;

        return ;
    }

    /* Allocate the compressed COO elements */
    MatrixCOO = (Element *) malloc (Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ * sizeof (Element)) ;

    /* Populate the compressed COO elements and COO value of Binding Table */
    /* Delete the Linked List in the meantime */
    i = 0 ;
    temp = Matrix->SMPkluMatrix->KLUmatrixLinkedListCOO ;
    while (temp != NULL) {
        MatrixCOO [i].row = temp->row ;
        MatrixCOO [i].col = temp->col ;
        MatrixCOO [i].pointer = temp->pointer ;
        MatrixCOO [i].group = 0 ;
        current = temp ;
        temp = temp->next ;
        free (current->pointer) ; // We need only the memory address, we don't need to access it
        free (current) ;
        current = NULL ;
        i++ ;
    }

    /* Order the MatrixCOO along the columns */
    qsort (MatrixCOO, Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ, sizeof (Element), CompareColumn) ;

    /* Order the MatrixCOO along the rows */
    i = 0 ;
    while (i < Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ)
    {
        /* Look for the next column */
        for (j = i + 1 ; j < Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ ; j++)
        {
            if (MatrixCOO [j].col != MatrixCOO [i].col)
            {
                break ;
            }
        }

        qsort (MatrixCOO + i, j - i, sizeof (Element), CompareRow) ;

        i = j ;
    }

    /* Look for columns with all structural zeroes, not numerical zeroes - Circuits Matrix are NxN, so checking for columns is sufficient */
    unsigned int n = MatrixCOO [Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ - 1].col + 1 ;
//    Matrix->SMPkluMatrix->KLUmatrixNodeCollapsingOldToNew = (unsigned int *) malloc ((n + 1) * sizeof (unsigned int)) ;
    Matrix->SMPkluMatrix->KLUmatrixNodeCollapsingNewToOld = (unsigned int *) malloc ((n + 1) * sizeof (unsigned int)) ;
    for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ ; i++) {
        // Initialization
//        Matrix->SMPkluMatrix->KLUmatrixNodeCollapsingOldToNew [MatrixCOO [i].col] = MatrixCOO [i].col ;
        Matrix->SMPkluMatrix->KLUmatrixNodeCollapsingNewToOld [MatrixCOO [i].col] = MatrixCOO [i].col ;
    }
//    Matrix->SMPkluMatrix->KLUmatrixNodeCollapsingOldToNew [n] = n ;
    Matrix->SMPkluMatrix->KLUmatrixNodeCollapsingNewToOld [n] = n ;

    unsigned int search_index = 0 ;
    while (search_index < Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ - 1)
    {
        int col_index = -1 ;
        unsigned int col_diff = 0 ;
        for (i = search_index, j = search_index + 1 ; i < Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ - 1 ; i++, j++)
        {
            col_diff = MatrixCOO [j].col - MatrixCOO [i].col ;
            if (col_diff > 1)
            {
                col_index = (int)(MatrixCOO [i].col) ;
                break ;
            }
        }
        search_index = i ;

        /* If col_index != -1 --> Row/Col elimination of all nodes greater than this one - compact by col_diff */
        if (col_index != -1)
        {
            for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ ; i++)
            {
                if (MatrixCOO [i].col > (unsigned int)col_index) {
//                    Matrix->SMPkluMatrix->KLUmatrixNodeCollapsingOldToNew [MatrixCOO [i].col] = MatrixCOO [i].col - col_diff + 1 ;
                    Matrix->SMPkluMatrix->KLUmatrixNodeCollapsingNewToOld [MatrixCOO [i].col - col_diff + 1] = MatrixCOO [i].col ;
                    MatrixCOO [i].col = MatrixCOO [i].col - col_diff + 1 ;
                }
                if (MatrixCOO [i].row > (unsigned int)col_index) {
                    MatrixCOO [i].row = MatrixCOO [i].row - col_diff + 1 ;
                }
            }
        }
    }

    /* Assign labels to avoid duplicates */
    for (i = 0, j = 1 ; i < Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ - 1 ; i++, j++) {
        if ((MatrixCOO [i].col == MatrixCOO [j].col) && (MatrixCOO [i].row == MatrixCOO [j].row)) {
            // If col and row are the same
            MatrixCOO [j].group = MatrixCOO [i].group ;
        } else if ((MatrixCOO [i].col != MatrixCOO [j].col) || (MatrixCOO [i].row != MatrixCOO [j].row)) {
            // If or col either row are different, it isn't a duplicate, so assign the next label and store it in 'nz'
            MatrixCOO [j].group = MatrixCOO [i].group + 1 ;
        } else {
            printf ("Error: Strange behavior during label assignment\n") ;
        }
    }

    /* Assign N and NZ */
    Matrix->SMPkluMatrix->KLUmatrixN = (unsigned int)MatrixCOO [Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ - 1].col + 1 ;
    Matrix->SMPkluMatrix->KLUmatrixNrhs = Matrix->SMPkluMatrix->KLUmatrixN + 1 ;
    Matrix->SMPkluMatrix->KLUmatrixNZ = MatrixCOO [Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ - 1].group + 1 ;

    /* Allocate Diag Gmin CSC Vector */
    Matrix->SMPkluMatrix->KLUmatrixDiag = (double **) malloc (Matrix->SMPkluMatrix->KLUmatrixN * sizeof (double *)) ;
    for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN ; i++) {
        Matrix->SMPkluMatrix->KLUmatrixDiag [i] = NULL ;
    }

    /* Allocate the temporary COO Column Index */
    Ap_COO = (unsigned int *) malloc (Matrix->SMPkluMatrix->KLUmatrixNZ * sizeof (unsigned int)) ;

    /* Allocate the needed KLU data structures */
    Matrix->SMPkluMatrix->KLUmatrixAp = (int *) malloc ((Matrix->SMPkluMatrix->KLUmatrixN + 1) * sizeof (int)) ;
    Matrix->SMPkluMatrix->KLUmatrixAi = (int *) malloc (Matrix->SMPkluMatrix->KLUmatrixNZ * sizeof (int)) ;
    Matrix->SMPkluMatrix->KLUmatrixBindStructCOO = (BindElement *) malloc (Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ * sizeof (BindElement)) ;
    /* +1 extra trailing element serves as a branchless trash sink for ground-node stamps */
    Matrix->SMPkluMatrix->KLUmatrixAx = (double *) malloc ((Matrix->SMPkluMatrix->KLUmatrixNZ + 1) * sizeof (double)) ;
    if (Matrix->SMPkluMatrix->KLUmatrixAx)
        Matrix->SMPkluMatrix->KLUmatrixAx[Matrix->SMPkluMatrix->KLUmatrixNZ] = 0.0 ;
    Matrix->SMPkluMatrix->KLUmatrixAxComplex = (double *) malloc (2 * Matrix->SMPkluMatrix->KLUmatrixNZ * sizeof (double)) ;
    Matrix->SMPkluMatrix->KLUmatrixIntermediate = (double *) malloc (Matrix->SMPkluMatrix->KLUmatrixN * sizeof (double)) ;
    Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex = (double *) malloc (2 * Matrix->SMPkluMatrix->KLUmatrixN * sizeof (double)) ;

    /* Copy back the Matrix in partial CSC */
    for (i = 0, current_group = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ ; i++)
    {
        if (MatrixCOO [i].group > current_group) {
            current_group = MatrixCOO [i].group ;
        }

        Ap_COO [current_group] = MatrixCOO [i].col ;
        Matrix->SMPkluMatrix->KLUmatrixAi [current_group] = (int)MatrixCOO [i].row ;
        Matrix->SMPkluMatrix->KLUmatrixBindStructCOO [i].COO = MatrixCOO [i].pointer ;
        Matrix->SMPkluMatrix->KLUmatrixBindStructCOO [i].CSC = &(Matrix->SMPkluMatrix->KLUmatrixAx [current_group]) ;
        Matrix->SMPkluMatrix->KLUmatrixBindStructCOO [i].CSC_Complex = &(Matrix->SMPkluMatrix->KLUmatrixAxComplex [2 * current_group]) ;
        if (MatrixCOO [i].col == MatrixCOO [i].row) {
            Matrix->SMPkluMatrix->KLUmatrixDiag [MatrixCOO [i].col] = Matrix->SMPkluMatrix->KLUmatrixBindStructCOO [i].CSC ;
        }
    }

    /* Compress the COO Column Index to CSC Column Index */
    Compress (Ap_COO, (unsigned int *)Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixN, Matrix->SMPkluMatrix->KLUmatrixNZ) ;

    /* Free the temporary stuff */
    free (Ap_COO) ;
    free (MatrixCOO) ;

    /* Sort the Binding Table */
    qsort (Matrix->SMPkluMatrix->KLUmatrixBindStructCOO, Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ, sizeof (BindElement), BindCompare) ;

    /* Set the Matrix as Real */
    Matrix->SMPkluMatrix->KLUmatrixIsComplex = KLUmatrixReal ;

    return ;
}

#ifdef CIDER
typedef struct sElementKLUforCIDER {
    unsigned int row ;
    unsigned int col ;
    double *pointer ;
} ElementKLUforCIDER ;

void SMPconvertCOOtoCSCKLUforCIDER (SMPmatrix *Matrix)
{
    ElementKLUforCIDER *MatrixCOO ;
    unsigned int *Ap_COO, i, j, nz ;

    /* Count the non-zero elements and store it */
    nz = 0 ;
    for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN * Matrix->SMPkluMatrix->KLUmatrixN ; i++) {
        if ((Matrix->SMPkluMatrix->KLUmatrixRowCOOforCIDER [i] != -1) && (Matrix->SMPkluMatrix->KLUmatrixColCOOforCIDER [i] != -1)) {
            nz++ ;
        }
    }
    Matrix->SMPkluMatrix->KLUmatrixNZ = nz ;

    /* Allocate the compressed COO elements */
    MatrixCOO = (ElementKLUforCIDER *) malloc (nz * sizeof (ElementKLUforCIDER)) ;

    /* Allocate the temporary COO Column Index */
    Ap_COO = (unsigned int *) malloc (nz * sizeof (unsigned int)) ;

    /* Allocate the needed KLU data structures */
    Matrix->SMPkluMatrix->KLUmatrixAp = (int *) malloc ((Matrix->SMPkluMatrix->KLUmatrixN + 1) * sizeof (int)) ;
    Matrix->SMPkluMatrix->KLUmatrixAi = (int *) malloc (nz * sizeof (int)) ;
    Matrix->SMPkluMatrix->KLUmatrixBindStructForCIDER = (BindElementKLUforCIDER *) malloc (nz * sizeof (BindElementKLUforCIDER)) ;
    Matrix->SMPkluMatrix->KLUmatrixAxComplex = (double *) malloc (2 * nz * sizeof (double)) ;
    Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex = (double *) malloc (2 * Matrix->SMPkluMatrix->KLUmatrixN * sizeof (double)) ;

    /* Populate the compressed COO elements and COO value of Binding Table */
    j = 0 ;
    for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN * Matrix->SMPkluMatrix->KLUmatrixN ; i++) {
        if ((Matrix->SMPkluMatrix->KLUmatrixRowCOOforCIDER [i] != -1) && (Matrix->SMPkluMatrix->KLUmatrixColCOOforCIDER [i] != -1)) {
            MatrixCOO [j].row = (unsigned int)Matrix->SMPkluMatrix->KLUmatrixRowCOOforCIDER [i] ;
            MatrixCOO [j].col = (unsigned int)Matrix->SMPkluMatrix->KLUmatrixColCOOforCIDER [i] ;
            MatrixCOO [j].pointer = &(Matrix->SMPkluMatrix->KLUmatrixValueComplexCOOforCIDER [2 * i]) ;
            j++ ;
        }
    }

    /* Order the MatrixCOO along the columns */
    qsort (MatrixCOO, nz, sizeof (ElementKLUforCIDER), CompareColumn) ;

    /* Order the MatrixCOO along the rows */
    i = 0 ;
    while (i < nz)
    {
        /* Look for the next column */
        for (j = i + 1 ; j < nz ; j++)
        {
            if (MatrixCOO [j].col != MatrixCOO [i].col)
            {
                break ;
            }
        }

        qsort (MatrixCOO + i, j - i, sizeof (ElementKLUforCIDER), CompareRow) ;

        i = j ;
    }

    /* Copy back the Matrix in partial CSC */
    for (i = 0 ; i < nz ; i++)
    {
        Ap_COO [i] = MatrixCOO [i].col ;
        Matrix->SMPkluMatrix->KLUmatrixAi [i] = (int)MatrixCOO [i].row ;
        Matrix->SMPkluMatrix->KLUmatrixBindStructForCIDER [i].COO = MatrixCOO [i].pointer ;
        Matrix->SMPkluMatrix->KLUmatrixBindStructForCIDER [i].CSC_Complex = &(Matrix->SMPkluMatrix->KLUmatrixAxComplex [2 * i]) ;
    }

    /* Compress the COO Column Index to CSC Column Index */
    Compress (Ap_COO, (unsigned int *)Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixN, nz) ;

    /* Free the temporary stuff */
    free (Ap_COO) ;
    free (MatrixCOO) ;

    /* Sort the Binding Table */
    qsort (Matrix->SMPkluMatrix->KLUmatrixBindStructForCIDER, nz, sizeof (BindElementKLUforCIDER), BindCompareKLUforCIDER) ;

    return ;
}
#endif

/*
 * SMPmakeElt()
 */
double *
SMPmakeElt (SMPmatrix *Matrix, int Row, int Col)
{
    KluLinkedListCOO *temp ;

    if (Matrix->CKTkluMODE) {
        if ((Row > 0) && (Col > 0)) {
            Row = Row - 1 ;
            Col = Col - 1 ;
            temp = (KluLinkedListCOO *) malloc (sizeof (KluLinkedListCOO)) ;
            temp->row = (unsigned int)Row ;
            temp->col = (unsigned int)Col ;
            temp->pointer = (double *) malloc (sizeof (double)) ;
            temp->next = Matrix->SMPkluMatrix->KLUmatrixLinkedListCOO ;
            Matrix->SMPkluMatrix->KLUmatrixLinkedListCOO = temp ;
            Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ++ ;
            return temp->pointer ;
        } else {
            return Matrix->SMPkluMatrix->KLUmatrixTrashCOO ;
        }
    } else {
        return spGetElement (Matrix->SPmatrix, Row, Col) ;
    }
}

#ifdef CIDER
double *
SMPmakeEltKLUforCIDER (SMPmatrix *Matrix, int Row, int Col)
{
    if (Matrix->CKTkluMODE) {
        if ((Row > 0) && (Col > 0)) {
            Row = Row - 1 ;
            Col = Col - 1 ;
            Matrix->SMPkluMatrix->KLUmatrixRowCOOforCIDER [Row * (int)Matrix->SMPkluMatrix->KLUmatrixN + Col] = Row ;
            Matrix->SMPkluMatrix->KLUmatrixColCOOforCIDER [Row * (int)Matrix->SMPkluMatrix->KLUmatrixN + Col] = Col ;
            return &(Matrix->SMPkluMatrix->KLUmatrixValueComplexCOOforCIDER [2 * (Row * (int)Matrix->SMPkluMatrix->KLUmatrixN + Col)]) ;
        } else {
            return Matrix->SMPkluMatrix->KLUmatrixTrashCOO ;
        }
    } else {
        return spGetElement (Matrix->SPmatrix, Row, Col) ;
    }
}
#endif

/*
 * SMPcClear()
 */

void
SMPcClear (SMPmatrix *Matrix)
{
    if (Matrix->CKTkluMODE)
    {
        if (Matrix->SMPkluMatrix->KLUmatrixNZ > 0)
            memset (Matrix->SMPkluMatrix->KLUmatrixAxComplex, 0,
                    2 * (size_t) Matrix->SMPkluMatrix->KLUmatrixNZ * sizeof (double)) ;
    } else {
        spClear (Matrix->SPmatrix) ;
    }
}

/*
 * SMPclear()
 */

void
SMPclear (SMPmatrix *Matrix)
{
    if (Matrix->CKTkluMODE)
    {
        if (Matrix->SMPkluMatrix->KLUmatrixNZ > 0)
            memset (Matrix->SMPkluMatrix->KLUmatrixAx, 0,
                    (size_t) Matrix->SMPkluMatrix->KLUmatrixNZ * sizeof (double)) ;
    } else {
        spClear (Matrix->SPmatrix) ;
    }
}

#ifdef CIDER
void
SMPclearKLUforCIDER (SMPmatrix *Matrix)
{
    unsigned int i ;

    for (i = 0 ; i < 2 * Matrix->SMPkluMatrix->KLUmatrixNZ ; i++) {
        Matrix->SMPkluMatrix->KLUmatrixAxComplex [i] = 0 ;
    }
}
#endif

#define NG_IGNORE(x)  (void)x

/*
 * SMPcLUfac()
 */
/*ARGSUSED*/

int
SMPcLUfac (SMPmatrix *Matrix, double PivTol)
{
    int ret ;

    NG_IGNORE (PivTol) ;

    if (Matrix->CKTkluMODE)
    {
        if (CircuitIsDigital() && Matrix->SMPkluMatrix->KLUmatrixN == 0) {
          // XSPICE pure digital circuits produce empty KLU matrix
          return 0 ;
        }

        ret = klu_z_refactor (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi, Matrix->SMPkluMatrix->KLUmatrixAxComplex,
                              Matrix->SMPkluMatrix->KLUmatrixSymbolic, Matrix->SMPkluMatrix->KLUmatrixNumeric, Matrix->SMPkluMatrix->KLUmatrixCommon) ;

        if (ret == 0)
        {
            if (Matrix->SMPkluMatrix->KLUmatrixCommon == NULL) {
                fprintf(stderr, "Error (ReFactor Complex): KLUcommon object is NULL. A problem occurred\n");
                return 0 ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_SINGULAR) {
                if (ft_ngdebug) {
                    fprintf(stderr, "Warning (ReFactor Complex): KLU Matrix is SINGULAR\n");
                    fprintf(stderr, "  Numerical Rank: %d\n", Matrix->SMPkluMatrix->KLUmatrixCommon->numerical_rank);
                    fprintf(stderr, "  Singular Node: %d\n", Matrix->SMPkluMatrix->KLUmatrixCommon->singular_col + 1);
                }
                return E_SINGULAR ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_EMPTY_MATRIX)
            {
                fprintf (stderr, "Error (ReFactor Complex): KLU Matrix is empty\n") ;
                return 0 ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixNumeric == NULL) {
                fprintf (stderr, "Error (ReFactor Complex): KLUnumeric object is NULL. A problem occurred\n") ;
            }
            return 1 ;
        } else {
            return 0 ;
        }
    } else {
        spSetComplex (Matrix->SPmatrix) ;
        return spFactor (Matrix->SPmatrix) ;
    }
}

/*
 * SMPluFac()
 */
/*ARGSUSED*/

int
SMPluFac (SMPmatrix *Matrix, double PivTol, double Gmin)
{
    int ret ;

    NG_IGNORE (PivTol) ;

    if (Matrix->CKTkluMODE)
    {
        if (CircuitIsDigital() && Matrix->SMPkluMatrix->KLUmatrixN == 0) {
          // XSPICE pure digital circuits produce empty KLU matrix
          return 0 ;
        }

        if (Matrix->SMPkluMatrix->KLUloadDiagGmin) {
            LoadGmin_CSC (Matrix->SMPkluMatrix->KLUmatrixDiag, Matrix->SMPkluMatrix->KLUmatrixN, Gmin) ;
        }

        ret = klu_refactor (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi, Matrix->SMPkluMatrix->KLUmatrixAx,
                            Matrix->SMPkluMatrix->KLUmatrixSymbolic, Matrix->SMPkluMatrix->KLUmatrixNumeric, Matrix->SMPkluMatrix->KLUmatrixCommon) ;

        if (ret == 0)
        {
            if (Matrix->SMPkluMatrix->KLUmatrixCommon == NULL) {
                fprintf (stderr, "Error (ReFactor): KLUcommon object is NULL. A problem occurred\n") ;
                return 1;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_SINGULAR) {
                if (ft_ngdebug) {
                    fprintf(stderr, "Warning (ReFactor): KLU Matrix is SINGULAR\n");
                    fprintf(stderr, "  Numerical Rank: %d\n", Matrix->SMPkluMatrix->KLUmatrixCommon->numerical_rank);
                    fprintf(stderr, "  Singular Node: %d\n", Matrix->SMPkluMatrix->KLUmatrixCommon->singular_col + 1);
                }
                return E_SINGULAR ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_EMPTY_MATRIX)
            {
                fprintf (stderr, "Error (ReFactor): KLU Matrix is empty\n") ;
                return 0 ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixNumeric == NULL) {
                fprintf (stderr, "Error (ReFactor): KLUnumeric object is NULL. A problem occurred\n") ;
                return 1 ;
            }
            return 1;
        } else {
            return 0 ;
        }
    } else {
        spSetReal (Matrix->SPmatrix) ;
        LoadGmin (Matrix, Gmin) ;
        return spFactor (Matrix->SPmatrix) ;
    }
}

#ifdef CIDER
int
SMPluFacKLUforCIDER (SMPmatrix *Matrix)
{
    unsigned int i ;
    double *KLUmatrixAx ;
    int ret ;

    if (Matrix->CKTkluMODE)
    {
        if (CircuitIsDigital() && Matrix->SMPkluMatrix->KLUmatrixN == 0) {
          // XSPICE pure digital circuits produce empty KLU matrix
          return 0 ;
        }

        if (Matrix->SMPkluMatrix->KLUmatrixIsComplex) {
            ret = klu_z_refactor (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi, Matrix->SMPkluMatrix->KLUmatrixAxComplex,
                                  Matrix->SMPkluMatrix->KLUmatrixSymbolic, Matrix->SMPkluMatrix->KLUmatrixNumeric, Matrix->SMPkluMatrix->KLUmatrixCommon) ;
        } else {
            /* Allocate the Real Matrix */
            KLUmatrixAx = (double *) malloc (Matrix->SMPkluMatrix->KLUmatrixNZ * sizeof(double)) ;

            /* Copy the Complex Matrix into the Real Matrix */
            for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixNZ ; i++) {
                KLUmatrixAx [i] = Matrix->SMPkluMatrix->KLUmatrixAxComplex [2 * i] ;
            }

            /* Re-Factor the Real Matrix */
            ret = klu_refactor (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi, KLUmatrixAx,
                                Matrix->SMPkluMatrix->KLUmatrixSymbolic, Matrix->SMPkluMatrix->KLUmatrixNumeric, Matrix->SMPkluMatrix->KLUmatrixCommon) ;

            /* Free the Real Matrix Storage */
            free (KLUmatrixAx) ;
        }

        if (ret == 0)
        {
            if (Matrix->SMPkluMatrix->KLUmatrixCommon == NULL) {
                fprintf (stderr, "Error (ReFactor for CIDER): KLUcommon object is NULL. A problem occurred\n") ;
                return 1;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_SINGULAR) {
                if (ft_ngdebug) {
                    fprintf(stderr, "Warning (ReFactor for CIDER): KLU Matrix is SINGULAR\n");
                    fprintf(stderr, "  Numerical Rank: %d\n", Matrix->SMPkluMatrix->KLUmatrixCommon->numerical_rank);
                    fprintf(stderr, "  Singular Node: %d\n", Matrix->SMPkluMatrix->KLUmatrixCommon->singular_col + 1);
                }
                return E_SINGULAR ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_EMPTY_MATRIX)
            {
                fprintf (stderr, "Error (ReFactor for CIDER): KLU Matrix is empty\n") ;
                return 0 ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixNumeric == NULL) {
                fprintf (stderr, "Error (ReFactor for CIDER): KLUnumeric object is NULL. A problem occurred\n") ;
                return 1 ;
            }
            return 1;
        } else {
            return 0 ;
        }
    } else {
        return spFactor (Matrix->SPmatrix) ;
    }
}
#endif

/*
 * SMPcReorder()
 */

int
SMPcReorder (SMPmatrix *Matrix, double PivTol, double PivRel, int *NumSwaps)
{
    if (Matrix->CKTkluMODE)
    {
        if (CircuitIsDigital() && Matrix->SMPkluMatrix->KLUmatrixN == 0) {
          // XSPICE pure digital circuits produce empty KLU matrix
          return 0 ;
        }

        Matrix->SMPkluMatrix->KLUmatrixCommon->tol = PivRel ;

        if (Matrix->SMPkluMatrix->KLUmatrixNumeric != NULL) {
            klu_free_numeric (&(Matrix->SMPkluMatrix->KLUmatrixNumeric), Matrix->SMPkluMatrix->KLUmatrixCommon) ;
        }
        Matrix->SMPkluMatrix->KLUmatrixNumeric = klu_z_factor (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi,
                                                               Matrix->SMPkluMatrix->KLUmatrixAxComplex, Matrix->SMPkluMatrix->KLUmatrixSymbolic,
                                                               Matrix->SMPkluMatrix->KLUmatrixCommon) ;

        if (Matrix->SMPkluMatrix->KLUmatrixNumeric == NULL)
        {
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_SINGULAR) {
                if (ft_ngdebug) {
                    fprintf(stderr, "Warning (Factor Complex): KLU Matrix is SINGULAR\n");
                    fprintf(stderr, "  Numerical Rank: %d\n", Matrix->SMPkluMatrix->KLUmatrixCommon->numerical_rank);
                    fprintf(stderr, "  Singular Node: %d\n", Matrix->SMPkluMatrix->KLUmatrixCommon->singular_col + 1);
                }
                return E_SINGULAR ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon == NULL) {
                fprintf (stderr, "Error (Factor Complex): KLUnumeric object is NULL. A problem occurred\n") ;
                fprintf (stderr, "Error (Factor Complex): KLUcommon object is NULL. A problem occurred\n") ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_EMPTY_MATRIX) {
                fprintf (stderr, "Error (Factor Complex): KLU Matrix is empty\n") ;
                return 0 ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixSymbolic == NULL) {
                fprintf (stderr, "Error (Factor Complex): KLUnumeric object is NULL. A problem occurred\n") ;
                fprintf (stderr, "Error (Factor Complex): KLUsymbolic object is NULL. A problem occurred\n") ;
            }
            return 1 ;
        } else {
            return 0 ;
        }
    } else {
        *NumSwaps = 1 ;
        spSetComplex (Matrix->SPmatrix) ;
        return spOrderAndFactor (Matrix->SPmatrix, NULL, (spREAL)PivRel, (spREAL)PivTol, YES) ;
    }
}

/*
 * SMPreorder()
 */

int
SMPreorder (SMPmatrix *Matrix, double PivTol, double PivRel, double Gmin)
{
    if (Matrix->CKTkluMODE)
    {
        if (CircuitIsDigital() && Matrix->SMPkluMatrix->KLUmatrixN == 0) {
          // XSPICE pure digital circuits produce empty KLU matrix
          return 0 ;
        }

        if (Matrix->SMPkluMatrix->KLUloadDiagGmin) {
            LoadGmin_CSC (Matrix->SMPkluMatrix->KLUmatrixDiag, Matrix->SMPkluMatrix->KLUmatrixN, Gmin) ;
        }
        Matrix->SMPkluMatrix->KLUmatrixCommon->tol = PivRel ;

        if (Matrix->SMPkluMatrix->KLUmatrixNumeric != NULL) {
            klu_free_numeric (&(Matrix->SMPkluMatrix->KLUmatrixNumeric), Matrix->SMPkluMatrix->KLUmatrixCommon) ;
        }

        Matrix->SMPkluMatrix->KLUmatrixNumeric = klu_factor (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi,
                                                             Matrix->SMPkluMatrix->KLUmatrixAx, Matrix->SMPkluMatrix->KLUmatrixSymbolic,
                                                             Matrix->SMPkluMatrix->KLUmatrixCommon) ;

        if (Matrix->SMPkluMatrix->KLUmatrixNumeric == NULL)
        {
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_SINGULAR) {
                if (ft_ngdebug) {
                    fprintf(stderr, "Warning (Factor): KLU Matrix is SINGULAR\n");
                    fprintf(stderr, "  Numerical Rank: %d\n", Matrix->SMPkluMatrix->KLUmatrixCommon->numerical_rank);
                    fprintf(stderr, "  Singular Node: %d\n", Matrix->SMPkluMatrix->KLUmatrixCommon->singular_col + 1);
                }
                return E_SINGULAR ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon == NULL) {
                fprintf (stderr, "Error (Factor): KLUnumeric object is NULL. A problem occurred\n") ;
                fprintf (stderr, "Error (Factor): KLUcommon object is NULL. A problem occurred\n") ;
                return 1;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_EMPTY_MATRIX) {
                fprintf (stderr, "Error (Factor): KLU Matrix is empty\n") ;
                return 0 ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixSymbolic == NULL) {
                fprintf (stderr, "Error (Factor): KLUnumeric object is NULL. A problem occurred\n") ;
                fprintf (stderr, "Error (Factor): KLUsymbolic object is NULL. A problem occurred\n") ;
            }
            return 1 ;
        } else {
            return 0 ;
        }
    } else {
        spSetReal (Matrix->SPmatrix) ;
        LoadGmin (Matrix, Gmin) ;
        return spOrderAndFactor (Matrix->SPmatrix, NULL, (spREAL)PivRel, (spREAL)PivTol, YES) ;
    }
}

#ifdef CIDER
int
SMPreorderKLUforCIDER (SMPmatrix *Matrix)
{
    unsigned int i ;
    double *KLUmatrixAx ;

    if (Matrix->CKTkluMODE)
    {
        if (CircuitIsDigital() && Matrix->SMPkluMatrix->KLUmatrixN == 0) {
          // XSPICE pure digital circuits produce empty KLU matrix
          return 0 ;
        }

        if (Matrix->SMPkluMatrix->KLUmatrixNumeric != NULL) {
            klu_free_numeric (&(Matrix->SMPkluMatrix->KLUmatrixNumeric), Matrix->SMPkluMatrix->KLUmatrixCommon) ;
        }
        if (Matrix->SMPkluMatrix->KLUmatrixIsComplex) {
            Matrix->SMPkluMatrix->KLUmatrixNumeric = klu_z_factor (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi,
                                                                   Matrix->SMPkluMatrix->KLUmatrixAxComplex, Matrix->SMPkluMatrix->KLUmatrixSymbolic,
                                                                   Matrix->SMPkluMatrix->KLUmatrixCommon) ;
        } else {
            /* Allocate the Real Matrix */
            KLUmatrixAx = (double *) malloc (Matrix->SMPkluMatrix->KLUmatrixNZ * sizeof(double)) ;

            /* Copy the Complex Matrix into the Real Matrix */
            for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixNZ ; i++) {
                KLUmatrixAx [i] = Matrix->SMPkluMatrix->KLUmatrixAxComplex [2 * i] ;
            }

            /* Factor the Real Matrix */
            Matrix->SMPkluMatrix->KLUmatrixNumeric = klu_factor (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi,
                                                                 KLUmatrixAx, Matrix->SMPkluMatrix->KLUmatrixSymbolic,
                                                                 Matrix->SMPkluMatrix->KLUmatrixCommon) ;

            /* Free the Real Matrix Storage */
            free (KLUmatrixAx) ;
        }

        if (Matrix->SMPkluMatrix->KLUmatrixNumeric == NULL)
        {
            if (Matrix->SMPkluMatrix->KLUmatrixCommon == NULL) {
                fprintf (stderr, "Error (Factor for CIDER): KLUnumeric object is NULL. A problem occurred\n") ;
                fprintf (stderr, "Error (Factor for CIDER): KLUcommon object is NULL. A problem occurred\n") ;
                return 1;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_SINGULAR) {
                if (ft_ngdebug) {
                    fprintf(stderr, "Warning (Factor for CIDER): KLU Matrix is SINGULAR\n");
                    fprintf(stderr, "  Numerical Rank: %d\n", Matrix->SMPkluMatrix->KLUmatrixCommon->numerical_rank);
                    fprintf(stderr, "  Singular Node: %d\n", Matrix->SMPkluMatrix->KLUmatrixCommon->singular_col + 1);
                }
                return E_SINGULAR ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_EMPTY_MATRIX) {
                fprintf (stderr, "Error (Factor for CIDER): KLU Matrix is empty\n") ;
                return 0 ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixSymbolic == NULL) {
                fprintf (stderr, "Error (Factor for CIDER): KLUnumeric object is NULL. A problem occurred\n") ;
                fprintf (stderr, "Error (Factor for CIDER): KLUsymbolic object is NULL. A problem occurred\n") ;
                return 1 ;
            }
            return 1;
        } else {
            return 0 ;
        }
    } else {
        return spFactor (Matrix->SPmatrix) ;
    }
}
#endif

/*
 * SMPcaSolve()
 */
void
SMPcaSolve (SMPmatrix *Matrix, double RHS[], double iRHS[], double Spare[], double iSpare[])
{
    int ret ;
    unsigned int i ;

    NG_IGNORE (iSpare) ;
    NG_IGNORE (Spare) ;

    if (Matrix->CKTkluMODE)
    {
        for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN ; i++)
        {
            Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex [2 * i] = RHS [i + 1] ;
            Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex [2 * i + 1] = iRHS [i + 1] ;
        }

        ret = klu_z_solve (Matrix->SMPkluMatrix->KLUmatrixSymbolic, Matrix->SMPkluMatrix->KLUmatrixNumeric, (int)Matrix->SMPkluMatrix->KLUmatrixN, 1,
                           Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex, Matrix->SMPkluMatrix->KLUmatrixCommon) ;

        for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN ; i++)
        {
            RHS [i + 1] = Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex [2 * i] ;
            iRHS [i + 1] = Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex [2 * i + 1] ;
        }
    } else {
        spSolveTransposed (Matrix->SPmatrix, RHS, RHS, iRHS, iRHS) ;
    }
}

/*
 * SMPcSolve()
 */

void
SMPcSolve (SMPmatrix *Matrix, double RHS[], double iRHS[], double Spare[], double iSpare[])
{
    int ret ;
    unsigned int i ;

    NG_IGNORE (iSpare) ;
    NG_IGNORE (Spare) ;

    if (Matrix->CKTkluMODE)
    {
        for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN ; i++)
        {
            Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex [2 * i] = RHS [i + 1] ;
            Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex [2 * i + 1] = iRHS [i + 1] ;
        }

        ret = klu_z_solve (Matrix->SMPkluMatrix->KLUmatrixSymbolic, Matrix->SMPkluMatrix->KLUmatrixNumeric, (int)Matrix->SMPkluMatrix->KLUmatrixN, 1,
                           Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex, Matrix->SMPkluMatrix->KLUmatrixCommon) ;

        for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN ; i++)
        {
            RHS [i + 1] = Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex [2 * i] ;
            iRHS [i + 1] = Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex [2 * i + 1] ;
        }
    } else {
        spSolve (Matrix->SPmatrix, RHS, RHS, iRHS, iRHS) ;
    }
}

/*
 * SMPsolve()
 */

void
SMPsolve (SMPmatrix *Matrix, double RHS[], double Spare[])
{
    int ret ;
    unsigned int i ;

    NG_IGNORE (Spare) ;

    if (Matrix->CKTkluMODE)
    {
        if (CircuitIsDigital() && Matrix->SMPkluMatrix->KLUmatrixN == 0) {
          // XSPICE pure digital circuits produce empty KLU matrix
          return ;
        }

#if NGSPICE_OPT_GENERAL
        {
            KLUmatrix *km = Matrix->SMPkluMatrix ;
            unsigned int n = km->KLUmatrixN ;
            if (km->KLUmatrixIdentityMap < 0) {
                int is_id = 1 ;
                const unsigned int *map = km->KLUmatrixNodeCollapsingNewToOld ;
                for (i = 0 ; i < n ; i++) {
                    if (map [i + 1] != i + 1) {
                        is_id = 0 ;
                        break ;
                    }
                }
                km->KLUmatrixIdentityMap = is_id ;
            }
            if (km->KLUmatrixIdentityMap == 1) {
                ret = klu_solve (km->KLUmatrixSymbolic, km->KLUmatrixNumeric, (int)n, 1,
                                 RHS + 1, km->KLUmatrixCommon) ;
                if (ret == 0) {
                    if (km->KLUmatrixCommon == NULL)
                        fprintf (stderr, "Error (Solve): KLUcommon object is NULL. A problem occurred\n") ;
                    else if (km->KLUmatrixCommon->status == KLU_SINGULAR && ft_ngdebug)
                        fprintf(stderr, "Warning (Solve): KLU Matrix is SINGULAR\n") ;
                }
                RHS [0] = 0.0 ;
                if (km->KLUmatrixNrhs > n + 1)
                    memset (RHS + n + 1, 0, (size_t)(km->KLUmatrixNrhs - (n + 1)) * sizeof (double)) ;
                return ;
            }
        }
#endif
        for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN ; i++) {
            if (Matrix->SMPkluMatrix->KLUmatrixNodeCollapsingNewToOld [i + 1] != 0) {
                Matrix->SMPkluMatrix->KLUmatrixIntermediate [i] = RHS [Matrix->SMPkluMatrix->KLUmatrixNodeCollapsingNewToOld [i + 1]] ;
            }
        }

        ret = klu_solve (Matrix->SMPkluMatrix->KLUmatrixSymbolic, Matrix->SMPkluMatrix->KLUmatrixNumeric, (int)Matrix->SMPkluMatrix->KLUmatrixN, 1,
                         Matrix->SMPkluMatrix->KLUmatrixIntermediate, Matrix->SMPkluMatrix->KLUmatrixCommon) ;

        if (ret == 0)
        {
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_SINGULAR) {
                if (ft_ngdebug) {
                    fprintf(stderr, "Warning (Solve): KLU Matrix is SINGULAR\n");
                    fprintf(stderr, "  Numerical Rank: %d\n", Matrix->SMPkluMatrix->KLUmatrixCommon->numerical_rank);
                    fprintf(stderr, "  Singular Node: %d\n", Matrix->SMPkluMatrix->KLUmatrixCommon->singular_col + 1);
                }
                /* FIXME: Do we need a 'return E_SINGULAR' here? */
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon == NULL) {
                fprintf (stderr, "Error (Solve): KLUcommon object is NULL. A problem occurred\n") ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_EMPTY_MATRIX)
            {
                fprintf (stderr, "Error (Solve): KLU Matrix is empty\n") ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixNumeric == NULL) {
                fprintf (stderr, "Error (Solve): KLUnumeric object is NULL. A problem occurred\n") ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixSymbolic == NULL) {
                fprintf (stderr, "Error (Solve): KLUsymbolic object is NULL. A problem occurred\n") ;
            }
        }

        if (Matrix->SMPkluMatrix->KLUmatrixNrhs > 0) {
            memset (RHS, 0, (size_t)Matrix->SMPkluMatrix->KLUmatrixNrhs * sizeof (double)) ;
        }

        for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN ; i++) {
            if (Matrix->SMPkluMatrix->KLUmatrixNodeCollapsingNewToOld [i + 1] != 0) {
                RHS [Matrix->SMPkluMatrix->KLUmatrixNodeCollapsingNewToOld [i + 1]] = Matrix->SMPkluMatrix->KLUmatrixIntermediate [i] ;
            }
        }
    } else {
        spSolve (Matrix->SPmatrix, RHS, RHS, NULL, NULL) ;
    }
}

#if NGSPICE_OPT_GENERAL
/*
 * Shamanskii / Modified Chord Newton step for KLU:
 * Computes exact nonlinear residual r = RHS - (A + Gmin*I) * Xold into
 * KLUmatrixIntermediate, solves L_old U_old dx = r reusing the existing
 * KLUmatrixNumeric factors, and returns max_j |dx[j]| in *dx_norm_out.
 * Leaves RHS and KLUmatrixAx untouched so the caller can seamlessly fall
 * back to SMPluFac + SMPsolve if contraction is insufficient.
 */
int
SMPkluChordSolve (SMPmatrix *Matrix, const double RHS[], const double Xold[],
                  double Gmin, double *dx_norm_out)
{
    KLUmatrix *km ;
    unsigned int n, j ;
    const unsigned int * restrict map ;
    const int * restrict Ap ;
    const int * restrict Ai ;
    const double * restrict Ax ;
    double * restrict r ;
    int ret ;

    if (!Matrix->CKTkluMODE)
        return 0 ;
    km = Matrix->SMPkluMatrix ;
    if (!km || km->KLUmatrixN == 0 || !km->KLUmatrixSymbolic || !km->KLUmatrixNumeric)
        return 0 ;

    n   = km->KLUmatrixN ;
    map = km->KLUmatrixNodeCollapsingNewToOld ;
    Ap  = km->KLUmatrixAp ;
    Ai  = km->KLUmatrixAi ;
    Ax  = km->KLUmatrixAx ;
    r   = km->KLUmatrixIntermediate ;

    for (j = 0 ; j < n ; j++) {
        unsigned int idx = map [j + 1] ;
        r [j] = idx ? RHS [idx] : 0.0 ;
    }

    for (j = 0 ; j < n ; j++) {
        unsigned int idx = map [j + 1] ;
        double xj = idx ? Xold [idx] : 0.0 ;
        if (xj == 0.0)
            continue ;
        if (Gmin != 0.0)
            r [j] -= Gmin * xj ;
        int p = Ap [j] ;
        int pend = Ap [j + 1] ;
        for ( ; p + 3 < pend ; p += 4) {
            int i0 = Ai [p], i1 = Ai [p + 1], i2 = Ai [p + 2], i3 = Ai [p + 3] ;
            double r0 = r [i0], r1 = r [i1], r2 = r [i2], r3 = r [i3] ;
            r0 -= Ax [p]     * xj ;
            r1 -= Ax [p + 1] * xj ;
            r2 -= Ax [p + 2] * xj ;
            r3 -= Ax [p + 3] * xj ;
            r [i0] = r0 ; r [i1] = r1 ; r [i2] = r2 ; r [i3] = r3 ;
        }
        for ( ; p < pend ; p++)
            r [Ai [p]] -= Ax [p] * xj ;
    }

    ret = klu_solve (km->KLUmatrixSymbolic, km->KLUmatrixNumeric, (int) n, 1,
                     r, km->KLUmatrixCommon) ;
    if (ret == 0)
        return 0 ;

    double max_dx = 0.0 ;
    for (j = 0 ; j < n ; j++) {
        double a = fabs (r [j]) ;
        if (!(a < 1.0e100)) /* catches NaN / Inf / blowup */
            return 0 ;
        if (a > max_dx)
            max_dx = a ;
    }
    *dx_norm_out = max_dx ;
    return 1 ;
}

void
SMPkluChordCommit (SMPmatrix *Matrix, double RHS[], const double Xold[])
{
    KLUmatrix *km = Matrix->SMPkluMatrix ;
    unsigned int n = km->KLUmatrixN ;
    const unsigned int * restrict map = km->KLUmatrixNodeCollapsingNewToOld ;
    const double * restrict dx = km->KLUmatrixIntermediate ;
    unsigned int j ;

    if (km->KLUmatrixNrhs > 0)
        memset (RHS, 0, (size_t) km->KLUmatrixNrhs * sizeof (double)) ;
    for (j = 0 ; j < n ; j++) {
        unsigned int idx = map [j + 1] ;
        if (idx != 0)
            RHS [idx] = Xold [idx] + dx [j] ;
    }
}

/* ==========================================================================
 * Prototype 3: Compound-Fast Active-Submatrix KLU Engine (NGSPICE_MR_COMPFAST)
 *
 * Solves A_FF * x_F = b_F - A_FS * x_S on the small active partition F during
 * multirate micro-steps while caching up to 16 active-set symbolic/numeric
 * factorizations (LRU) and supporting submatrix Shamanskii Chord Newton.
 * ========================================================================== */

#define KLU_SUB_CACHE_SIZE 16
#define KLU_SUB_MAX_RAILS  8

typedef struct sKLUsubEntry {
    unsigned long long hash ;
    unsigned long long last_tick ;
    int valid ;
    int numeric_valid ;
    int nF ;
    int sub_nz ;
    int bnd_nz ;
    int n_wake ;
    int *SubToRhs ;      /* [nF] 1-based RHS equation indices */
    unsigned char *InSub ; /* [in_sub_len] 1 iff 1-based RHS node is in this submatrix */
    int in_sub_len ;
    int *SubAp ;         /* [nF + 1] CSC column pointers */
    int *SubAi ;         /* [sub_nz] CSC row indices */
    int *SubAxMap ;      /* [sub_nz] index into full KLUmatrixAx */
    int *SubDiag ;       /* [nF] index into SubAx for diagonal, or -1 */
    double *SubAx ;      /* [sub_nz + 1] gathered numerical values */
    double *SubRhs ;     /* [nF] RHS / solution scratch */
    double *SubBaseAx ;  /* [sub_nz + 1] cached linear+latent base for current (ag0, latOps) */
    double *BndBaseAx ;  /* [bnd_nz + 1] cached linear+latent base for current (ag0, latOps) */
    double *RailBaseAx ; /* [rail_total + 1] cached linear+latent base for current (ag0, latOps) */
    double  base_ag0 ;
    int     base_lat_ops ;
    int     base_lat_cnt ;
    int     base_flags ;
    int     base_valid ;
    int *BndRow ;        /* [bnd_nz] submatrix row index 0..nF-1 */
    int *BndColRhs ;     /* [bnd_nz] slow 1-based RHS index */
    int *BndAxMap ;      /* [bnd_nz] index into full KLUmatrixAx */
    int *WakeRhs ;       /* [n_wake] fast 1-based RHS nodes adjacent to non-rail slow nodes */
    double *WakeScale ;  /* [n_wake] optional sensitivity weight max|A_{S,F}/A_{S,S}| (NULL => 1.0) */
    /* Grounded DC rail full-row maps (excluding rail branch col) for exact i(Vvdd) */
    int n_rails ;
    int rail_node [KLU_SUB_MAX_RAILS] ;
    int rail_br_col [KLU_SUB_MAX_RAILS] ;
    int rail_f_ptr [KLU_SUB_MAX_RAILS + 1] ;
    int *RailFAxMap ;
    int *RailFColRhs ;
    double rail_saved_rhs [KLU_SUB_MAX_RAILS] ;
    klu_common Common ;
    klu_symbolic *Symbolic ;
    klu_numeric *Numeric ;
} KLUsubEntry ;

typedef struct sKLUsubCache {
    unsigned long long tick ;
    KLUsubEntry entries [KLU_SUB_CACHE_SIZE] ;
    KLUsubEntry *active ;
    /* Lazy symmetric adjacency CSR in 1-based RHS node space for O(deg) 1-hop expansion */
    int adj_built ;
    int adj_n_rhs ;
    int *adj_p ;
    int *adj_i ;
    /* Scratch buffers for pattern extraction */
    int scratch_n ;
    int *sub_idx ;       /* [N] full KLU col/row -> sub index 0..nF-1, or -1 */
    int *tmp_fast_rhs ;  /* [N] */
    unsigned char *tmp_wake_mark ; /* [N] */
    double *tmp_wake_scale ;       /* [N] */
} KLUsubCache ;

static void
klu_sub_free_entry (KLUsubEntry *e)
{
    if (!e || !e->valid)
        return ;
    if (e->Numeric)
        klu_free_numeric (&e->Numeric, &e->Common) ;
    if (e->Symbolic)
        klu_free_symbolic (&e->Symbolic, &e->Common) ;
    free (e->SubToRhs) ;
    free (e->InSub) ;
    free (e->SubAp) ;
    free (e->SubAi) ;
    free (e->SubAxMap) ;
    free (e->SubDiag) ;
    free (e->SubAx) ;
    free (e->SubRhs) ;
    free (e->SubBaseAx) ;
    free (e->BndBaseAx) ;
    free (e->RailBaseAx) ;
    free (e->BndRow) ;
    free (e->BndColRhs) ;
    free (e->BndAxMap) ;
    free (e->WakeRhs) ;
    free (e->WakeScale) ;
    free (e->RailFAxMap) ;
    free (e->RailFColRhs) ;
    memset (e, 0, sizeof (*e)) ;
}

static void
SMPkluSubDestroy (KLUmatrix *km)
{
    if (!km || !km->KLUmatrixSubCache)
        return ;
    KLUsubCache *sc = km->KLUmatrixSubCache ;
    for (int i = 0 ; i < KLU_SUB_CACHE_SIZE ; i++)
        klu_sub_free_entry (&sc->entries [i]) ;
    free (sc->adj_p) ;
    free (sc->adj_i) ;
    free (sc->sub_idx) ;
    free (sc->tmp_fast_rhs) ;
    free (sc->tmp_wake_mark) ;
    free (sc->tmp_wake_scale) ;
    free (sc) ;
    km->KLUmatrixSubCache = NULL ;
}

static KLUsubCache *
klu_sub_ensure_cache (KLUmatrix *km)
{
    if (km->KLUmatrixSubCache)
        return km->KLUmatrixSubCache ;
    KLUsubCache *sc = (KLUsubCache *) calloc (1, sizeof (KLUsubCache)) ;
    km->KLUmatrixSubCache = sc ;
    return sc ;
}

static void
klu_sub_ensure_adj (KLUmatrix *km, KLUsubCache *sc, int n_rhs)
{
    if (sc->adj_built && sc->adj_n_rhs == n_rhs)
        return ;
    free (sc->adj_p) ;
    free (sc->adj_i) ;
    sc->adj_p = NULL ;
    sc->adj_i = NULL ;
    sc->adj_built = 0 ;

    int n = (int) km->KLUmatrixN ;
    const int * restrict Ap = km->KLUmatrixAp ;
    const int * restrict Ai = km->KLUmatrixAi ;
    const unsigned int * restrict map = km->KLUmatrixNodeCollapsingNewToOld ;
    if (n <= 0 || !Ap || !Ai || !map)
        return ;

    int *deg = (int *) calloc ((size_t) (n_rhs + 1), sizeof (int)) ;
    if (!deg)
        return ;

    /* Undirected edges: for each off-diagonal (r, c), add r->c and c->r */
    for (int j = 0 ; j < n ; j++) {
        int cj = (int) map [j + 1] ;
        if (cj <= 0 || cj >= n_rhs) continue ;
        int p0 = Ap [j], p1 = Ap [j + 1] ;
        for (int p = p0 ; p < p1 ; p++) {
            int i = Ai [p] ;
            if (i == j) continue ;
            int ri = (int) map [i + 1] ;
            if (ri <= 0 || ri >= n_rhs) continue ;
            deg [cj]++ ;
            deg [ri]++ ;
        }
    }

    int *adj_p = (int *) malloc ((size_t) (n_rhs + 1) * sizeof (int)) ;
    if (!adj_p) { free (deg) ; return ; }
    adj_p [0] = 0 ;
    for (int u = 0 ; u < n_rhs ; u++)
        adj_p [u + 1] = adj_p [u] + deg [u] ;
    int total = adj_p [n_rhs] ;
    int *adj_i = (int *) malloc ((size_t) (total + 1) * sizeof (int)) ;
    if (!adj_i) { free (deg) ; free (adj_p) ; return ; }

    memcpy (deg, adj_p, (size_t) n_rhs * sizeof (int)) ;
    for (int j = 0 ; j < n ; j++) {
        int cj = (int) map [j + 1] ;
        if (cj <= 0 || cj >= n_rhs) continue ;
        int p0 = Ap [j], p1 = Ap [j + 1] ;
        for (int p = p0 ; p < p1 ; p++) {
            int i = Ai [p] ;
            if (i == j) continue ;
            int ri = (int) map [i + 1] ;
            if (ri <= 0 || ri >= n_rhs) continue ;
            adj_i [deg [cj]++] = ri ;
            adj_i [deg [ri]++] = cj ;
        }
    }
    free (deg) ;
    sc->adj_p = adj_p ;
    sc->adj_i = adj_i ;
    sc->adj_n_rhs = n_rhs ;
    sc->adj_built = 1 ;
}

void
SMPkluSubExpand1Hop (SMPmatrix *Matrix, unsigned char *is_fast, int n_rhs,
                     const unsigned char *is_rail)
{
    if (!Matrix || !Matrix->CKTkluMODE || !Matrix->SMPkluMatrix)
        return ;
    KLUmatrix *km = Matrix->SMPkluMatrix ;
    int n = (int) km->KLUmatrixN ;
    if (n <= 0) return ;

    static int eps_init = 0, has_eps_ov = 0 ;
    static double eps_ov = 0.0 ;
    if (!eps_init) {
        const char *s = getenv ("NGSPICE_MR_CF_EDGE_EPS") ;
        if (s && s[0]) {
            eps_ov = atof (s) ;
            has_eps_ov = 1 ;
        }
        eps_init = 1 ;
    }
    double edge_eps = has_eps_ov ? eps_ov : ((n >= 10000) ? 0.02 : 0.0) ;

    /* Strong-coupling directional 1-hop expansion for VLSI-scale matrices:
     * Pull slow node `row` into F only if fast node `j` exerts significant
     * sensitivity |A_{row,j}| >= edge_eps * |A_{row,row}| (breaks weak fF
     * OpenRCX coupling caps, OFF MOSFETs, and backward drain->gate unioning). */
    if (edge_eps > 0.0 && km->KLUmatrixAp && km->KLUmatrixAi &&
        km->KLUmatrixAx && km->KLUmatrixNodeCollapsingNewToOld)
    {
        const int * restrict Ap = km->KLUmatrixAp ;
        const int * restrict Ai = km->KLUmatrixAi ;
        const double * restrict Ax = km->KLUmatrixAx ;
        double ** restrict Diag = km->KLUmatrixDiag ;
        const unsigned int * restrict map = km->KLUmatrixNodeCollapsingNewToOld ;
        for (int j = 0 ; j < n ; j++) {
            int u = (int) map [j + 1] ;
            if (u <= 0 || u >= n_rhs || is_fast [u] != 1)
                continue ;
            if (is_rail && is_rail [u])
                continue ;
            int p0 = Ap [j], p1 = Ap [j + 1] ;
            for (int p = p0 ; p < p1 ; p++) {
                int row = Ai [p] ;
                int v = (int) map [row + 1] ;
                if (v <= 0 || v >= n_rhs || is_fast [v] != 0 || (is_rail && is_rail [v]))
                    continue ;
                double aij = fabs (Ax [p]) ;
                if (aij == 0.0)
                    continue ;
                double aii = (Diag && Diag [row]) ? fabs (*Diag [row]) : 0.0 ;
                if (aii == 0.0 || aij >= edge_eps * aii)
                    is_fast [v] = 2 ;
            }
        }
        for (int u = 1 ; u < n_rhs ; u++) {
            if (is_fast [u] == 2)
                is_fast [u] = 1 ;
        }
        return ;
    }

    KLUsubCache *sc = klu_sub_ensure_cache (km) ;
    if (!sc) return ;
    klu_sub_ensure_adj (km, sc, n_rhs) ;
    if (!sc->adj_built) return ;

    const int * restrict ap = sc->adj_p ;
    const int * restrict ai = sc->adj_i ;

    /* Mark 1-hop non-rail neighbors with value 2, then promote to 1 */
    for (int u = 1 ; u < n_rhs ; u++) {
        if (is_fast [u] != 1)
            continue ;
        if (is_rail && is_rail [u])
            continue ;
        int p0 = ap [u], p1 = ap [u + 1] ;
        for (int p = p0 ; p < p1 ; p++) {
            int v = ai [p] ;
            if (v > 0 && v < n_rhs && is_fast [v] == 0 && (!is_rail || !is_rail [v]))
                is_fast [v] = 2 ;
        }
    }
    for (int u = 1 ; u < n_rhs ; u++) {
        if (is_fast [u] == 2)
            is_fast [u] = 1 ;
    }
}

int
SMPkluSubPrepare (SMPmatrix *Matrix, const unsigned char *is_fast, int n_rhs,
                  const unsigned char *is_rail, const int *rail_nodes, int n_rails,
                  int max_nF, int *n_fast_out, const int **fast_rhs_out,
                  int *same_as_prev_out,
                  unsigned long long *cache_hits_out,
                  unsigned long long *cache_misses_out)
{
    if (!Matrix || !Matrix->CKTkluMODE || !Matrix->SMPkluMatrix)
        return 0 ;
    KLUmatrix *km = Matrix->SMPkluMatrix ;
    int n = (int) km->KLUmatrixN ;
    if (n <= 0 || !km->KLUmatrixAp || !km->KLUmatrixAi || !km->KLUmatrixNodeCollapsingNewToOld)
        return 0 ;

    KLUsubCache *sc = klu_sub_ensure_cache (km) ;
    if (!sc) return 0 ;

    if (sc->scratch_n < n) {
        free (sc->sub_idx) ;
        free (sc->tmp_fast_rhs) ;
        free (sc->tmp_wake_mark) ;
        free (sc->tmp_wake_scale) ;
        sc->sub_idx        = (int *) malloc ((size_t) n * sizeof (int)) ;
        sc->tmp_fast_rhs   = (int *) malloc ((size_t) n * sizeof (int)) ;
        sc->tmp_wake_mark  = (unsigned char *) malloc ((size_t) n * sizeof (unsigned char)) ;
        sc->tmp_wake_scale = (double *) malloc ((size_t) n * sizeof (double)) ;
        if (!sc->sub_idx || !sc->tmp_fast_rhs || !sc->tmp_wake_mark || !sc->tmp_wake_scale) {
            sc->scratch_n = 0 ;
            return 0 ;
        }
        sc->scratch_n = n ;
    }

    const unsigned int * restrict map = km->KLUmatrixNodeCollapsingNewToOld ;
    int * restrict sub_idx = sc->sub_idx ;
    int * restrict tmp_fast = sc->tmp_fast_rhs ;
    KLUsubEntry *prev_active = sc->active ;

    int nF = 0 ;
    unsigned long long h = 14695981039346656037ULL ;
    for (int j = 0 ; j < n ; j++) {
        int rn = (int) map [j + 1] ;
        if (rn > 0 && rn < n_rhs && is_fast [rn]) {
            sub_idx [j] = nF ;
            tmp_fast [nF] = rn ;
            h ^= (unsigned long long) rn ;
            h *= 1099511628211ULL ;
            nF++ ;
        } else {
            sub_idx [j] = -1 ;
        }
    }
    if (nF <= 0)
        return 0 ;

    sc->tick++ ;

    /* 1. Check 16-entry LRU cache for exact match OR tight superset match */
    int lru_slot = 0 ;
    unsigned long long lru_best = ~0ULL ;
    int sup_limit = (int) (1.25 * (double) nF) + 32 ;
    if (sup_limit > max_nF) sup_limit = max_nF ;
    KLUsubEntry *best_sup = NULL ;

    for (int s = 0 ; s < KLU_SUB_CACHE_SIZE ; s++) {
        KLUsubEntry *e = &sc->entries [s] ;
        if (e->valid) {
            if (e->hash == h && e->nF == nF &&
                memcmp (e->SubToRhs, tmp_fast, (size_t) nF * sizeof (int)) == 0)
            {
                best_sup = e ;
                break ;
            }
            if (e->nF >= nF && e->nF <= sup_limit && e->InSub && e->in_sub_len >= n_rhs &&
                (!best_sup || e->nF < best_sup->nF))
            {
                int is_sup = 1 ;
                const unsigned char * restrict ins = e->InSub ;
                for (int k = 0 ; k < nF ; k++) {
                    if (!ins [tmp_fast [k]]) { is_sup = 0 ; break ; }
                }
                if (is_sup)
                    best_sup = e ;
            }
            if (e->last_tick < lru_best) {
                lru_best = e->last_tick ;
                lru_slot = s ;
            }
        } else {
            lru_slot = s ;
            lru_best = 0 ;
        }
    }

    if (best_sup) {
        best_sup->last_tick = sc->tick ;
        sc->active = best_sup ;
        if (n_fast_out) *n_fast_out = best_sup->nF ;
        if (fast_rhs_out) *fast_rhs_out = best_sup->SubToRhs ;
        if (same_as_prev_out) *same_as_prev_out = (best_sup == prev_active) ? 1 : 0 ;
        if (cache_hits_out) (*cache_hits_out)++ ;
        return 1 ;
    }

    if (cache_misses_out) (*cache_misses_out)++ ;

    /* 2. Union hysteresis: if active entry + requested set stays compact, build union
     *    so subsequent bursts hit the superset cache instead of thrashing klu_analyze */
    if (prev_active && prev_active->valid && prev_active->InSub && prev_active->in_sub_len >= n_rhs) {
        const unsigned char * restrict pins = prev_active->InSub ;
        int union_cnt = 0 ;
        for (int j = 0 ; j < n ; j++) {
            int rn = (int) map [j + 1] ;
            if (rn > 0 && rn < n_rhs && (is_fast [rn] || pins [rn]) &&
                (!is_rail || !is_rail [rn]))
                union_cnt++ ;
        }
        int union_lim = (int) (1.25 * (double) nF) + 48 ;
        if (union_lim > max_nF) union_lim = max_nF ;
        if (union_cnt > nF && union_cnt <= union_lim) {
            nF = 0 ;
            h = 14695981039346656037ULL ;
            for (int j = 0 ; j < n ; j++) {
                int rn = (int) map [j + 1] ;
                if (rn > 0 && rn < n_rhs && (is_fast [rn] || pins [rn]) &&
                    (!is_rail || !is_rail [rn]))
                {
                    sub_idx [j] = nF ;
                    tmp_fast [nF] = rn ;
                    h ^= (unsigned long long) rn ;
                    h *= 1099511628211ULL ;
                    nF++ ;
                } else {
                    sub_idx [j] = -1 ;
                }
            }
        }
    }

    /* Cache miss: evict lru_slot and build new submatrix entry */
    KLUsubEntry *e = &sc->entries [lru_slot] ;
    if (e == prev_active)
        prev_active = NULL ;
    klu_sub_free_entry (e) ;

    const int * restrict Ap = km->KLUmatrixAp ;
    const int * restrict Ai = km->KLUmatrixAi ;

    /* Count sub_nz, bnd_nz, and full rail row entries (excluding rail branch cols) */
    int nr = (n_rails > KLU_SUB_MAX_RAILS) ? KLU_SUB_MAX_RAILS : n_rails ;
    if (nr < 0) nr = 0 ;
    int rail_klu_row [KLU_SUB_MAX_RAILS] ;
    int rail_cnt [KLU_SUB_MAX_RAILS] ;
    for (int r = 0 ; r < nr ; r++) {
        rail_klu_row [r] = -1 ;
        rail_cnt [r] = 0 ;
        e->rail_node [r] = rail_nodes [r] ;
        e->rail_br_col [r] = -1 ;
    }
    for (int j = 0 ; j < n && nr > 0 ; j++) {
        int rn = (int) map [j + 1] ;
        for (int r = 0 ; r < nr ; r++) {
            if (rn == rail_nodes [r])
                rail_klu_row [r] = j ;
        }
    }

    int sub_nz = 0, bnd_nz = 0 ;
    for (int j = 0 ; j < n ; j++) {
        int p0 = Ap [j], p1 = Ap [j + 1] ;
        int col_rhs = (int) map [j + 1] ;
        /* Skip branch-current columns (is_rail && col != rail_node) when counting rail row */
        int col_is_rail_br = (is_rail && col_rhs > 0 && col_rhs < n_rhs && is_rail [col_rhs]) ? 1 : 0 ;
        if (col_is_rail_br) {
            for (int r = 0 ; r < nr ; r++) {
                if (col_rhs == rail_nodes [r]) { col_is_rail_br = 0 ; break ; }
            }
        }
        if (sub_idx [j] >= 0) {
            for (int p = p0 ; p < p1 ; p++) {
                int row = Ai [p] ;
                if (sub_idx [row] >= 0)
                    sub_nz++ ;
                if (!col_is_rail_br) {
                    for (int r = 0 ; r < nr ; r++) {
                        if (row == rail_klu_row [r]) { rail_cnt [r]++ ; break ; }
                    }
                }
            }
        } else {
            for (int p = p0 ; p < p1 ; p++) {
                int row = Ai [p] ;
                if (sub_idx [row] >= 0)
                    bnd_nz++ ;
                if (!col_is_rail_br) {
                    for (int r = 0 ; r < nr ; r++) {
                        if (row == rail_klu_row [r]) { rail_cnt [r]++ ; break ; }
                    }
                }
            }
        }
    }

    e->nF = nF ;
    e->sub_nz = sub_nz ;
    e->bnd_nz = bnd_nz ;
    e->n_rails = nr ;
    e->SubToRhs = (int *) malloc ((size_t) nF * sizeof (int)) ;
    e->InSub    = (unsigned char *) calloc ((size_t) (n_rhs + 1), sizeof (unsigned char)) ;
    e->in_sub_len = n_rhs ;
    e->SubAp    = (int *) malloc ((size_t) (nF + 1) * sizeof (int)) ;
    e->SubAi    = (int *) malloc ((size_t) (sub_nz + 1) * sizeof (int)) ;
    e->SubAxMap = (int *) malloc ((size_t) (sub_nz + 1) * sizeof (int)) ;
    e->SubDiag  = (int *) malloc ((size_t) nF * sizeof (int)) ;
    e->SubAx    = (double *) malloc ((size_t) (sub_nz + 1) * sizeof (double)) ;
    e->SubRhs   = (double *) malloc ((size_t) nF * sizeof (double)) ;
    e->SubBaseAx = (double *) malloc ((size_t) (sub_nz + 1) * sizeof (double)) ;
    e->BndBaseAx = (double *) malloc ((size_t) (bnd_nz + 1) * sizeof (double)) ;
    e->BndRow   = (int *) malloc ((size_t) (bnd_nz + 1) * sizeof (int)) ;
    e->BndColRhs= (int *) malloc ((size_t) (bnd_nz + 1) * sizeof (int)) ;
    e->BndAxMap = (int *) malloc ((size_t) (bnd_nz + 1) * sizeof (int)) ;

    int rail_total = 0 ;
    e->rail_f_ptr [0] = 0 ;
    for (int r = 0 ; r < nr ; r++) {
        rail_total += rail_cnt [r] ;
        e->rail_f_ptr [r + 1] = rail_total ;
    }
    e->RailFAxMap  = (int *) malloc ((size_t) (rail_total + 1) * sizeof (int)) ;
    e->RailFColRhs = (int *) malloc ((size_t) (rail_total + 1) * sizeof (int)) ;
    e->RailBaseAx  = (double *) malloc ((size_t) (rail_total + 1) * sizeof (double)) ;

    if (!e->SubToRhs || !e->InSub || !e->SubAp || !e->SubAi || !e->SubAxMap || !e->SubDiag ||
        !e->SubAx || !e->SubRhs || !e->SubBaseAx || !e->BndBaseAx || !e->RailBaseAx ||
        !e->BndRow || !e->BndColRhs || !e->BndAxMap ||
        !e->RailFAxMap || !e->RailFColRhs)
    {
        e->valid = 1 ;
        klu_sub_free_entry (e) ;
        return 0 ;
    }

    memcpy (e->SubToRhs, tmp_fast, (size_t) nF * sizeof (int)) ;
    for (int sf = 0 ; sf < nF ; sf++) {
        int rn = tmp_fast [sf] ;
        if (rn > 0 && rn < n_rhs)
            e->InSub [rn] = 1 ;
    }
    memset (sc->tmp_wake_mark, 0, (size_t) nF) ;
    int use_wake_scale = (n >= 10000 && km->KLUmatrixAx != NULL) ;
    if (use_wake_scale)
        memset (sc->tmp_wake_scale, 0, (size_t) nF * sizeof (double)) ;
    for (int sf = 0 ; sf < nF ; sf++)
        e->SubDiag [sf] = -1 ;

    int rail_pos [KLU_SUB_MAX_RAILS] ;
    for (int r = 0 ; r < nr ; r++)
        rail_pos [r] = e->rail_f_ptr [r] ;

    const double * restrict Ax_full = km->KLUmatrixAx ;
    double ** restrict Diag_full = km->KLUmatrixDiag ;
    int sp = 0, bp = 0 ;
    for (int j = 0 ; j < n ; j++) {
        int p0 = Ap [j], p1 = Ap [j + 1] ;
        int sc_col = sub_idx [j] ;
        int col_rhs = (int) map [j + 1] ;
        int col_is_rail_br = (is_rail && col_rhs > 0 && col_rhs < n_rhs && is_rail [col_rhs]) ? 1 : 0 ;
        if (col_is_rail_br) {
            for (int r = 0 ; r < nr ; r++) {
                if (col_rhs == rail_nodes [r]) { col_is_rail_br = 0 ; break ; }
            }
        }
        if (sc_col >= 0) {
            e->SubAp [sc_col] = sp ;
            for (int p = p0 ; p < p1 ; p++) {
                int row = Ai [p] ;
                int sr = sub_idx [row] ;
                if (sr >= 0) {
                    e->SubAi [sp] = sr ;
                    e->SubAxMap [sp] = p ;
                    if (sr == sc_col)
                        e->SubDiag [sc_col] = sp ;
                    sp++ ;
                } else if (use_wake_scale) {
                    int row_rhs = (int) map [row + 1] ;
                    if (row_rhs > 0 && row_rhs < n_rhs && (!is_rail || !is_rail [row_rhs])) {
                        double aij = fabs (Ax_full [p]) ;
                        if (aij > 0.0) {
                            double aii = (Diag_full && Diag_full [row]) ? fabs (*Diag_full [row]) : 0.0 ;
                            double ratio = (aii > 0.0) ? (aij / aii) : 1.0 ;
                            if (ratio > 1.0) ratio = 1.0 ;
                            if (ratio > sc->tmp_wake_scale [sc_col]) {
                                sc->tmp_wake_scale [sc_col] = ratio ;
                                sc->tmp_wake_mark [sc_col] = 1 ;
                            }
                        }
                    }
                }
                if (!col_is_rail_br) {
                    for (int r = 0 ; r < nr ; r++) {
                        if (row == rail_klu_row [r]) {
                            int rp = rail_pos [r]++ ;
                            e->RailFAxMap [rp] = p ;
                            e->RailFColRhs [rp] = col_rhs ;
                            break ;
                        }
                    }
                }
            }
        } else {
            int col_is_rail = (is_rail && col_rhs > 0 && col_rhs < n_rhs && is_rail [col_rhs]) ? 1 : 0 ;
            for (int p = p0 ; p < p1 ; p++) {
                int row = Ai [p] ;
                int sr = sub_idx [row] ;
                if (sr >= 0) {
                    e->BndRow [bp] = sr ;
                    e->BndColRhs [bp] = col_rhs ;
                    e->BndAxMap [bp] = p ;
                    bp++ ;
                    if (!use_wake_scale && !col_is_rail)
                        sc->tmp_wake_mark [sr] = 1 ;
                }
                if (!col_is_rail_br) {
                    for (int r = 0 ; r < nr ; r++) {
                        if (row == rail_klu_row [r]) {
                            int rp = rail_pos [r]++ ;
                            e->RailFAxMap [rp] = p ;
                            e->RailFColRhs [rp] = col_rhs ;
                            break ;
                        }
                    }
                }
            }
        }
    }
    e->SubAp [nF] = sp ;

    int n_wake = 0 ;
    for (int sf = 0 ; sf < nF ; sf++)
        if (sc->tmp_wake_mark [sf]) n_wake++ ;
    e->n_wake = n_wake ;
    e->WakeRhs = (int *) malloc ((size_t) (n_wake + 1) * sizeof (int)) ;
    e->WakeScale = use_wake_scale ? (double *) malloc ((size_t) (n_wake + 1) * sizeof (double)) : NULL ;
    if (!e->WakeRhs || (use_wake_scale && !e->WakeScale)) {
        e->valid = 1 ;
        klu_sub_free_entry (e) ;
        return 0 ;
    }
    int wi = 0 ;
    for (int sf = 0 ; sf < nF ; sf++) {
        if (sc->tmp_wake_mark [sf]) {
            e->WakeRhs [wi] = e->SubToRhs [sf] ;
            if (use_wake_scale)
                e->WakeScale [wi] = sc->tmp_wake_scale [sf] ;
            wi++ ;
        }
    }

    klu_defaults (&e->Common) ;
    e->Common.memgrow = km->KLUmatrixCommon->memgrow ;
    e->Symbolic = klu_analyze (nF, e->SubAp, e->SubAi, &e->Common) ;
    if (!e->Symbolic) {
        e->valid = 1 ;
        klu_sub_free_entry (e) ;
        return 0 ;
    }

    e->hash = h ;
    e->last_tick = sc->tick ;
    e->numeric_valid = 0 ;
    e->valid = 1 ;
    sc->active = e ;

    if (n_fast_out) *n_fast_out = e->nF ;
    if (fast_rhs_out) *fast_rhs_out = e->SubToRhs ;
    if (same_as_prev_out) *same_as_prev_out = 0 ;
    return 1 ;
}

void
SMPkluSubInitLoad (SMPmatrix *Matrix, double * restrict ax_main, double * restrict rhs_main,
                   const double * restrict res_base, const double * restrict cap_unit,
                   const double * restrict cap_rhs, const double * restrict lat_dc,
                   const double * restrict lat_cap, const double * restrict lat_rhs_dc,
                   const double * restrict lat_rhs_cap, double ag0)
{
    KLUmatrix *km = Matrix->SMPkluMatrix ;
    KLUsubCache *sc = km ? km->KLUmatrixSubCache : NULL ;
    KLUsubEntry *e = sc ? sc->active : NULL ;
    if (!e) return ;

    const int * restrict smap = e->SubAxMap ;
    const int * restrict bmap = e->BndAxMap ;
    const int * restrict frhs = e->SubToRhs ;
    int snz = e->sub_nz ;
    int bnz = e->bnd_nz ;
    int nF  = e->nF ;
    int rtot = (e->n_rails > 0) ? e->rail_f_ptr [e->n_rails] : 0 ;
    int has_lat = (lat_dc != NULL && lat_cap != NULL) ? 1 : 0 ;
    int flags = (res_base != NULL ? 1 : 0) | (cap_unit != NULL ? 2 : 0) | (has_lat ? 4 : 0) ;

    /* Cache linear+latent base values per (ag0, latent_ops, flags) so Newton iters 2..K
     * scatter from ONE contiguous L1/L2 buffer instead of 4 scattered full-matrix arrays! */
    double * restrict sbase = e->SubBaseAx ;
    double * restrict bbase = e->BndBaseAx ;
    double * restrict rbase = e->RailBaseAx ;
    if (!e->base_valid || e->base_ag0 != ag0 ||
        e->base_lat_ops != km->KLUmatrixLatentOps ||
        e->base_lat_cnt != km->KLUmatrixLatentCount ||
        e->base_flags != flags)
    {
        if (has_lat) {
            if (res_base && cap_unit) {
                for (int k = 0 ; k < snz ; k++) {
                    int p = smap [k] ;
                    sbase [k] = (res_base [p] + lat_dc [p]) + ag0 * (cap_unit [p] + lat_cap [p]) ;
                }
                for (int k = 0 ; k < bnz ; k++) {
                    int p = bmap [k] ;
                    bbase [k] = (res_base [p] + lat_dc [p]) + ag0 * (cap_unit [p] + lat_cap [p]) ;
                }
                for (int k = 0 ; k < rtot ; k++) {
                    int p = e->RailFAxMap [k] ;
                    rbase [k] = (res_base [p] + lat_dc [p]) + ag0 * (cap_unit [p] + lat_cap [p]) ;
                }
            } else if (res_base) {
                for (int k = 0 ; k < snz ; k++) {
                    int p = smap [k] ;
                    sbase [k] = (res_base [p] + lat_dc [p]) + ag0 * lat_cap [p] ;
                }
                for (int k = 0 ; k < bnz ; k++) {
                    int p = bmap [k] ;
                    bbase [k] = (res_base [p] + lat_dc [p]) + ag0 * lat_cap [p] ;
                }
                for (int k = 0 ; k < rtot ; k++) {
                    int p = e->RailFAxMap [k] ;
                    rbase [k] = (res_base [p] + lat_dc [p]) + ag0 * lat_cap [p] ;
                }
            } else if (cap_unit) {
                for (int k = 0 ; k < snz ; k++) {
                    int p = smap [k] ;
                    sbase [k] = lat_dc [p] + ag0 * (cap_unit [p] + lat_cap [p]) ;
                }
                for (int k = 0 ; k < bnz ; k++) {
                    int p = bmap [k] ;
                    bbase [k] = lat_dc [p] + ag0 * (cap_unit [p] + lat_cap [p]) ;
                }
                for (int k = 0 ; k < rtot ; k++) {
                    int p = e->RailFAxMap [k] ;
                    rbase [k] = lat_dc [p] + ag0 * (cap_unit [p] + lat_cap [p]) ;
                }
            } else {
                for (int k = 0 ; k < snz ; k++) {
                    int p = smap [k] ;
                    sbase [k] = lat_dc [p] + ag0 * lat_cap [p] ;
                }
                for (int k = 0 ; k < bnz ; k++) {
                    int p = bmap [k] ;
                    bbase [k] = lat_dc [p] + ag0 * lat_cap [p] ;
                }
                for (int k = 0 ; k < rtot ; k++) {
                    int p = e->RailFAxMap [k] ;
                    rbase [k] = lat_dc [p] + ag0 * lat_cap [p] ;
                }
            }
        } else if (res_base && cap_unit) {
            for (int k = 0 ; k < snz ; k++) { int p = smap [k] ; sbase [k] = res_base [p] + ag0 * cap_unit [p] ; }
            for (int k = 0 ; k < bnz ; k++) { int p = bmap [k] ; bbase [k] = res_base [p] + ag0 * cap_unit [p] ; }
            for (int k = 0 ; k < rtot ; k++) { int p = e->RailFAxMap [k] ; rbase [k] = res_base [p] + ag0 * cap_unit [p] ; }
        } else if (res_base) {
            for (int k = 0 ; k < snz ; k++) sbase [k] = res_base [smap [k]] ;
            for (int k = 0 ; k < bnz ; k++) bbase [k] = res_base [bmap [k]] ;
            for (int k = 0 ; k < rtot ; k++) rbase [k] = res_base [e->RailFAxMap [k]] ;
        } else if (cap_unit) {
            for (int k = 0 ; k < snz ; k++) sbase [k] = ag0 * cap_unit [smap [k]] ;
            for (int k = 0 ; k < bnz ; k++) bbase [k] = ag0 * cap_unit [bmap [k]] ;
            for (int k = 0 ; k < rtot ; k++) rbase [k] = ag0 * cap_unit [e->RailFAxMap [k]] ;
        } else {
            memset (sbase, 0, (size_t) snz * sizeof (double)) ;
            memset (bbase, 0, (size_t) bnz * sizeof (double)) ;
            memset (rbase, 0, (size_t) rtot * sizeof (double)) ;
        }
        e->base_ag0 = ag0 ;
        e->base_lat_ops = km->KLUmatrixLatentOps ;
        e->base_lat_cnt = km->KLUmatrixLatentCount ;
        e->base_flags = flags ;
        e->base_valid = 1 ;
    }

    for (int k = 0 ; k < snz ; k++)
        ax_main [smap [k]] = sbase [k] ;
    for (int k = 0 ; k < bnz ; k++)
        ax_main [bmap [k]] = bbase [k] ;
    for (int k = 0 ; k < rtot ; k++)
        ax_main [e->RailFAxMap [k]] = rbase [k] ;
    ax_main [km->KLUmatrixNZ] = 0.0 ;

    /* Initialize ONLY fast RHS entries */
    if (lat_rhs_dc && lat_rhs_cap) {
        if (cap_rhs) {
            for (int sf = 0 ; sf < nF ; sf++) {
                int f = frhs [sf] ;
                rhs_main [f] = cap_rhs [f] + lat_rhs_dc [f] + ag0 * lat_rhs_cap [f] ;
            }
        } else {
            for (int sf = 0 ; sf < nF ; sf++) {
                int f = frhs [sf] ;
                rhs_main [f] = lat_rhs_dc [f] + ag0 * lat_rhs_cap [f] ;
            }
        }
    } else if (cap_rhs) {
        for (int sf = 0 ; sf < nF ; sf++) {
            int f = frhs [sf] ;
            rhs_main [f] = cap_rhs [f] ;
        }
    } else {
        for (int sf = 0 ; sf < nF ; sf++)
            rhs_main [frhs [sf]] = 0.0 ;
    }

    for (int r = 0 ; r < e->n_rails ; r++) {
        int u = e->rail_node [r] ;
        double rval = (cap_rhs ? cap_rhs [u] : 0.0) ;
        if (lat_rhs_dc && lat_rhs_cap)
            rval += lat_rhs_dc [u] + ag0 * lat_rhs_cap [u] ;
        rhs_main [u] = rval ;
    }
}

int
SMPkluSubSolve (SMPmatrix *Matrix, double RHS[], const double Xold[],
                int rhs_len, double Gmin, double ag0, int iterno,
                double prev_dx_norm, int chord_disabled,
                double (*sec_fn)(void),
                int *used_chord_out, double *dx_norm_out,
                double *decomp_time_out, double *solve_time_out,
                int *lu_valid_io, double *lu_ag0_io, double *lu_gmin_io,
                int *chord_cnt_io)
{
    KLUmatrix *km = Matrix->SMPkluMatrix ;
    KLUsubCache *sc = km ? km->KLUmatrixSubCache : NULL ;
    KLUsubEntry *e = sc ? sc->active : NULL ;
    if (!e || !e->Symbolic)
        return E_SINGULAR ;

    int nF  = e->nF ;
    int snz = e->sub_nz ;
    int bnz = e->bnd_nz ;
    const double * restrict ax_main = km->KLUmatrixAx ;
    const int * restrict smap = e->SubAxMap ;
    const int * restrict frhs = e->SubToRhs ;
    double * restrict sax  = e->SubAx ;
    double * restrict srhs = e->SubRhs ;

    if (used_chord_out) *used_chord_out = 0 ;
    if (decomp_time_out) *decomp_time_out = 0.0 ;
    if (solve_time_out) *solve_time_out = 0.0 ;

    /* Save rail RHS before overwriting RHS with solution */
    for (int r = 0 ; r < e->n_rails ; r++)
        e->rail_saved_rhs [r] = RHS [e->rail_node [r]] ;

    /* 1. Gather A_FF into SubAx (+ Gmin on diagonal if active) */
    for (int k = 0 ; k < snz ; k++)
        sax [k] = ax_main [smap [k]] ;
    if (Gmin != 0.0) {
        const int * restrict sdiag = e->SubDiag ;
        for (int sf = 0 ; sf < nF ; sf++) {
            int d = sdiag [sf] ;
            if (d >= 0) sax [d] += Gmin ;
        }
    }

    /* 2. Build effective RHS: b_eff = b_F - A_FS * x_S */
    for (int sf = 0 ; sf < nF ; sf++)
        srhs [sf] = RHS [frhs [sf]] ;
    {
        const int * restrict brow = e->BndRow ;
        const int * restrict bcol = e->BndColRhs ;
        const int * restrict bmap = e->BndAxMap ;
        for (int k = 0 ; k < bnz ; k++)
            srhs [brow [k]] -= ax_main [bmap [k]] * Xold [bcol [k]] ;
    }

    /* 3. Optional Submatrix Shamanskii Chord Newton */
    {
        int vlsi_sub = (km->KLUmatrixN >= 10000) ;
        int max_c = vlsi_sub ? 16 : 8 ;
        double lu_ag0 = lu_ag0_io ? *lu_ag0_io : 0.0 ;
        int ag0_ok = vlsi_sub
            ? (lu_ag0 > 0.0 && ag0 >= 0.75 * lu_ag0 && ag0 <= 1.33 * lu_ag0)
            : (lu_ag0 == ag0) ;
        int iter_ok = vlsi_sub
            ? (iterno == 1 || prev_dx_norm <= 0.50)
            : (iterno >= 2 && prev_dx_norm <= 0.05) ;
        if (!chord_disabled && iter_ok && e->numeric_valid && e->Numeric &&
            lu_valid_io && *lu_valid_io && ag0_ok &&
            lu_gmin_io && *lu_gmin_io == Gmin &&
            chord_cnt_io && *chord_cnt_io < max_c)
        {
            double t0 = sec_fn ? sec_fn () : 0.0 ;
            /* Residual r = b_eff - A_FF * x_F_old */
            const int * restrict sap = e->SubAp ;
            const int * restrict sai = e->SubAi ;
            for (int j = 0 ; j < nF ; j++) {
                double xj = Xold [frhs [j]] ;
                if (xj == 0.0) continue ;
                int p0 = sap [j], p1 = sap [j + 1] ;
                for (int p = p0 ; p < p1 ; p++)
                    srhs [sai [p]] -= sax [p] * xj ;
            }
            int ret = klu_solve (e->Symbolic, e->Numeric, nF, 1, srhs, &e->Common) ;
            if (ret != 0) {
                double max_dx = 0.0 ;
                int ok = 1 ;
                for (int j = 0 ; j < nF ; j++) {
                    double a = fabs (srhs [j]) ;
                    if (!(a < 1.0e100)) { ok = 0 ; break ; }
                    if (a > max_dx) max_dx = a ;
                }
                int accept = 0 ;
                if (ok) {
                    if (vlsi_sub) {
                        accept = (iterno == 1)
                            ? (max_dx < 0.05)
                            : (max_dx < 0.30 && (max_dx <= 0.65 * prev_dx_norm || max_dx <= 1.0e-4)) ;
                    } else {
                        accept = (max_dx < 0.04 && (max_dx <= 0.35 * prev_dx_norm || max_dx <= 5.0e-5)) ;
                    }
                }
                if (accept) {
                    memcpy (RHS, Xold, (size_t) rhs_len * sizeof (double)) ;
                    for (int sf = 0 ; sf < nF ; sf++)
                        RHS [frhs [sf]] = Xold [frhs [sf]] + srhs [sf] ;
                    if (solve_time_out && sec_fn) *solve_time_out = sec_fn () - t0 ;
                    if (used_chord_out) *used_chord_out = 1 ;
                    if (dx_norm_out) *dx_norm_out = max_dx ;
                    (*chord_cnt_io)++ ;
                    return spOKAY ;
                }
            }
            /* Chord did not contract enough: rebuild b_eff and fall through to refactor */
            for (int sf = 0 ; sf < nF ; sf++)
                srhs [sf] = RHS [frhs [sf]] ;
            const int * restrict brow = e->BndRow ;
            const int * restrict bcol = e->BndColRhs ;
            const int * restrict bmap = e->BndAxMap ;
            for (int k = 0 ; k < bnz ; k++)
                srhs [brow [k]] -= ax_main [bmap [k]] * Xold [bcol [k]] ;
        }
    }

    /* 4. Refactor or Factor the small nF x nF submatrix */
    double t0 = sec_fn ? sec_fn () : 0.0 ;
    int fac_ok = 0 ;
    if (e->numeric_valid && e->Numeric) {
        fac_ok = klu_refactor (e->SubAp, e->SubAi, sax, e->Symbolic, e->Numeric, &e->Common) ;
        if (!fac_ok || e->Common.status == KLU_SINGULAR) {
            klu_free_numeric (&e->Numeric, &e->Common) ;
            e->numeric_valid = 0 ;
            fac_ok = 0 ;
        }
    }
    if (!fac_ok) {
        e->Numeric = klu_factor (e->SubAp, e->SubAi, sax, e->Symbolic, &e->Common) ;
        if (!e->Numeric || e->Common.status == KLU_SINGULAR || e->Common.status < KLU_OK) {
            if (e->Numeric)
                klu_free_numeric (&e->Numeric, &e->Common) ;
            e->numeric_valid = 0 ;
            if (lu_valid_io) *lu_valid_io = 0 ;
            if (decomp_time_out && sec_fn) *decomp_time_out = sec_fn () - t0 ;
            return E_SINGULAR ;
        }
        e->numeric_valid = 1 ;
    }
    double t1 = sec_fn ? sec_fn () : 0.0 ;
    if (decomp_time_out) *decomp_time_out = t1 - t0 ;

    if (lu_valid_io) *lu_valid_io = 1 ;
    if (lu_ag0_io) *lu_ag0_io = ag0 ;
    if (lu_gmin_io) *lu_gmin_io = Gmin ;
    if (chord_cnt_io) *chord_cnt_io = 0 ;

    /* 5. Backsolve nF x nF system */
    int sret = klu_solve (e->Symbolic, e->Numeric, nF, 1, srhs, &e->Common) ;
    if (sret == 0) {
        if (solve_time_out && sec_fn) *solve_time_out = sec_fn () - t1 ;
        return E_SINGULAR ;
    }

    /* 6. Scatter fast solution while keeping slow nodes bit-identical to Xold */
    memcpy (RHS, Xold, (size_t) rhs_len * sizeof (double)) ;
    double max_dx = 0.0 ;
    for (int sf = 0 ; sf < nF ; sf++) {
        int idx = frhs [sf] ;
        double v = srhs [sf] ;
        if (!(fabs (v) < 1.0e100)) {
            if (solve_time_out && sec_fn) *solve_time_out = sec_fn () - t1 ;
            return E_SINGULAR ;
        }
        double d = fabs (v - Xold [idx]) ;
        if (d > max_dx) max_dx = d ;
        RHS [idx] = v ;
    }
    if (solve_time_out && sec_fn) *solve_time_out = sec_fn () - t1 ;
    if (dx_norm_out) *dx_norm_out = max_dx ;
    return spOKAY ;
}

int
SMPkluSubCheckWake (SMPmatrix *Matrix, const double * restrict rhs_now,
                    const double * restrict v_macro, double v_wake)
{
    KLUmatrix *km = Matrix ? Matrix->SMPkluMatrix : NULL ;
    KLUsubCache *sc = km ? km->KLUmatrixSubCache : NULL ;
    KLUsubEntry *e = sc ? sc->active : NULL ;
    if (!e) return 1 ;

    const int * restrict wr = e->WakeRhs ;
    const double * restrict ws = e->WakeScale ;
    int nw = e->n_wake ;
    if (ws) {
        for (int i = 0 ; i < nw ; i++) {
            int u = wr [i] ;
            if (ws [i] * fabs (rhs_now [u] - v_macro [u]) > v_wake)
                return 1 ;
        }
    } else {
        for (int i = 0 ; i < nw ; i++) {
            int u = wr [i] ;
            if (fabs (rhs_now [u] - v_macro [u]) > v_wake)
                return 1 ;
        }
    }
    return 0 ;
}

void
SMPkluSubRailUpdate (SMPmatrix *Matrix, const double * restrict ax_main,
                     const double * restrict rhs_main, double * restrict x_sol,
                     const int *rail_nodes, const int *rail_br,
                     const double *rail_sgn, int n_rails)
{
    (void) rhs_main ;
    KLUmatrix *km = Matrix ? Matrix->SMPkluMatrix : NULL ;
    KLUsubCache *sc = km ? km->KLUmatrixSubCache : NULL ;
    KLUsubEntry *e = sc ? sc->active : NULL ;
    if (!e || e->n_rails <= 0 || n_rails <= 0)
        return ;

    int nr = (e->n_rails < n_rails) ? e->n_rails : n_rails ;
    for (int r = 0 ; r < nr ; r++) {
        int u = e->rail_node [r] ;
        if (u != rail_nodes [r] || rail_br [r] <= 0)
            continue ;
        double sum = e->rail_saved_rhs [r] ;
        int p0 = e->rail_f_ptr [r], p1 = e->rail_f_ptr [r + 1] ;
        for (int k = p0 ; k < p1 ; k++)
            sum -= ax_main [e->RailFAxMap [k]] * x_sol [e->RailFColRhs [k]] ;
        x_sol [rail_br [r]] = rail_sgn [r] * sum ;
    }
}
#endif

#ifdef CIDER
void
SMPsolveKLUforCIDER (SMPmatrix *Matrix, double RHS[], double RHSsolution[], double iRHS[], double iRHSsolution[])
{
    int ret ;
    unsigned int i ;
    double *KLUmatrixIntermediate ;

    if (Matrix->CKTkluMODE)
    {
        if (Matrix->SMPkluMatrix->KLUmatrixIsComplex) {
            for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN ; i++)
            {
                Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex [2 * i] = RHS [i + 1] ;
                Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex [2 * i + 1] = iRHS [i + 1] ;
            }

            ret = klu_z_solve (Matrix->SMPkluMatrix->KLUmatrixSymbolic, Matrix->SMPkluMatrix->KLUmatrixNumeric, (int)Matrix->SMPkluMatrix->KLUmatrixN, 1,
                               Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex, Matrix->SMPkluMatrix->KLUmatrixCommon) ;

            for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN ; i++)
            {
                RHSsolution [i + 1] = Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex [2 * i] ;
                iRHSsolution [i + 1] = Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex [2 * i + 1] ;
            }
        } else {
            /* Allocate the Intermediate Vector */
            KLUmatrixIntermediate = (double *) malloc (Matrix->SMPkluMatrix->KLUmatrixN * sizeof(double)) ;

            for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN ; i++) {
                KLUmatrixIntermediate [i] = RHS [i + 1] ;
            }

            ret = klu_solve (Matrix->SMPkluMatrix->KLUmatrixSymbolic, Matrix->SMPkluMatrix->KLUmatrixNumeric, (int)Matrix->SMPkluMatrix->KLUmatrixN, 1,
                             KLUmatrixIntermediate, Matrix->SMPkluMatrix->KLUmatrixCommon) ;

            for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN ; i++) {
                RHSsolution [i + 1] = KLUmatrixIntermediate [i] ;
            }

            /* Free the Intermediate Vector */
            free (KLUmatrixIntermediate) ;
        }

    } else {

        spSolve (Matrix->SPmatrix, RHS, RHSsolution, iRHS, iRHSsolution) ;
    }
}
#endif

int
SMPkluEnsureTls (KLUmatrix *km, int nthreads, int rhs_len)
{
    size_t ax_need, rhs_need ;

    if (!km || nthreads <= 0)
        return 0 ;

    ax_need = ((size_t)km->KLUmatrixNZ + 1 + 7) & ~(size_t)7 ;
    rhs_need = ((size_t)rhs_len + 1 + 7) & ~(size_t)7 ;

    if (km->KLUmatrixTlsThreads >= nthreads &&
        km->KLUmatrixAxTlsStride >= ax_need &&
        km->KLUmatrixRhsTlsStride >= rhs_need &&
        km->KLUmatrixAxTls != NULL &&
        km->KLUmatrixRhsTls != NULL)
    {
        return 1 ;
    }

    free (km->KLUmatrixAxTls) ;
    free (km->KLUmatrixRhsTls) ;
    km->KLUmatrixAxTls = NULL ;
    km->KLUmatrixRhsTls = NULL ;
    km->KLUmatrixTlsThreads = 0 ;

    if (posix_memalign ((void **)&km->KLUmatrixAxTls, 64,
                        (size_t)nthreads * ax_need * sizeof (double)) != 0)
        km->KLUmatrixAxTls = NULL ;
    if (posix_memalign ((void **)&km->KLUmatrixRhsTls, 64,
                        (size_t)nthreads * rhs_need * sizeof (double)) != 0)
        km->KLUmatrixRhsTls = NULL ;

    if (!km->KLUmatrixAxTls || !km->KLUmatrixRhsTls) {
        free (km->KLUmatrixAxTls) ;
        free (km->KLUmatrixRhsTls) ;
        km->KLUmatrixAxTls = NULL ;
        km->KLUmatrixRhsTls = NULL ;
        return 0 ;
    }

    km->KLUmatrixAxTlsStride = ax_need ;
    km->KLUmatrixRhsTlsStride = rhs_need ;
    km->KLUmatrixTlsThreads = nthreads ;
    return 1 ;
}

/*
 * SMPmatSize()
 */
int
SMPmatSize (SMPmatrix *Matrix)
{
    if (Matrix->CKTkluMODE) {
        return (int)Matrix->SMPkluMatrix->KLUmatrixN ;
    } else {
        return spGetSize (Matrix->SPmatrix, 1) ;
    }
}

/*
 * SMPnewMatrix()
 */
int
SMPnewMatrix (SMPmatrix *Matrix, int size)
{
    int Error ;

    if (Matrix->CKTkluMODE) {
        /* Allocate the KLU Matrix Data Structure */
        Matrix->SMPkluMatrix = (KLUmatrix *) malloc (sizeof (KLUmatrix)) ;
        Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixLinkedListCOO = NULL ;

        /* Initialize the KLU Matrix Internal Pointers */
        Matrix->SMPkluMatrix->KLUmatrixCommon = (klu_common *) malloc (sizeof (klu_common)) ; ;
        Matrix->SMPkluMatrix->KLUmatrixSymbolic = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixNumeric = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAp = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAi = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAx = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAxComplex = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixIsComplex = KLUmatrixReal ;
        Matrix->SMPkluMatrix->KLUmatrixIntermediate = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixNZ = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixBindStructCOO = NULL ;
//        Matrix->SMPkluMatrix->KLUmatrixNodeCollapsingOldToNew = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixNodeCollapsingNewToOld = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixDiag = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAxTls = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixRhsTls = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAxTlsStride = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixRhsTlsStride = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixTlsThreads = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixAxResConst = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixResConstValid = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixResVersion = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixAxCapUnit = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixCapUnitValid = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixCapVersion = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixAxResCapBase = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixResCapAg0 = 0.0 ;
        Matrix->SMPkluMatrix->KLUmatrixResCapValid = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixResCapHasRes = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixRhsCapCache = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixRhsCapLen = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixRhsCapValid = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixIdentityMap = -1 ;
        Matrix->SMPkluMatrix->KLUmatrixAxLatentDC = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAxLatentCap = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixRhsLatentDC = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixRhsLatentCap = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixRhsLatentLen = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixLatentCount = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixLatentOps = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixResCapHasLatent = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixSubCache = NULL ;

        /* Initialize the KLU Common Data Structure */
        klu_defaults (Matrix->SMPkluMatrix->KLUmatrixCommon) ;
        Matrix->SMPkluMatrix->KLUmatrixCommon->memgrow = Matrix->CKTkluMemGrowFactor ;

        /* Allocate KLU data structures */
        Matrix->SMPkluMatrix->KLUmatrixN = (unsigned int)size ;
        Matrix->SMPkluMatrix->KLUmatrixTrashCOO = (double *) malloc (2 * sizeof (double)) ;

        return spOKAY ;
    } else {
        Matrix->SPmatrix = spCreate (size, 1, &Error) ;
        return Error ;
    }
}

#ifdef CIDER
int
SMPnewMatrixKLUforCIDER (SMPmatrix *Matrix, int size, unsigned int KLUmatrixIsComplex)
{
    int Error ;
    unsigned int i ;

    if (Matrix->CKTkluMODE) {
        /* Allocate the KLU Matrix Data Structure */
        Matrix->SMPkluMatrix = (KLUmatrix *) malloc (sizeof (KLUmatrix)) ;

        /* Initialize the KLU Matrix Internal Pointers */
        Matrix->SMPkluMatrix->KLUmatrixCommon = (klu_common *) malloc (sizeof (klu_common)) ; ;
        Matrix->SMPkluMatrix->KLUmatrixSymbolic = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixNumeric = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAp = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAi = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAxComplex = NULL ;
        if (KLUmatrixIsComplex) {
            Matrix->SMPkluMatrix->KLUmatrixIsComplex = KLUMatrixComplex ;
        } else {
            Matrix->SMPkluMatrix->KLUmatrixIsComplex = KLUmatrixReal ;
        }
        Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixNZ = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixBindStructForCIDER = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixValueComplexCOOforCIDER = NULL ;

        Matrix->SMPkluMatrix->KLUmatrixDiag = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAxTls = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixRhsTls = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAxTlsStride = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixRhsTlsStride = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixTlsThreads = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixAxResConst = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixResConstValid = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixResVersion = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixAxCapUnit = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixCapUnitValid = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixCapVersion = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixAxResCapBase = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixResCapAg0 = 0.0 ;
        Matrix->SMPkluMatrix->KLUmatrixResCapValid = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixResCapHasRes = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixRhsCapCache = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixRhsCapLen = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixRhsCapValid = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixIdentityMap = -1 ;
        Matrix->SMPkluMatrix->KLUmatrixAxLatentDC = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAxLatentCap = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixRhsLatentDC = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixRhsLatentCap = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixRhsLatentLen = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixLatentCount = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixLatentOps = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixResCapHasLatent = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixSubCache = NULL ;

        /* Initialize the KLU Common Data Structure */
        klu_defaults (Matrix->SMPkluMatrix->KLUmatrixCommon) ;

        /* Allocate KLU data structures */
        Matrix->SMPkluMatrix->KLUmatrixN = (unsigned int)size ;
        Matrix->SMPkluMatrix->KLUmatrixColCOOforCIDER = (int *) malloc (Matrix->SMPkluMatrix->KLUmatrixN * Matrix->SMPkluMatrix->KLUmatrixN * sizeof(int)) ;
        Matrix->SMPkluMatrix->KLUmatrixRowCOOforCIDER = (int *) malloc (Matrix->SMPkluMatrix->KLUmatrixN * Matrix->SMPkluMatrix->KLUmatrixN * sizeof(int)) ;
        Matrix->SMPkluMatrix->KLUmatrixTrashCOO = (double *) malloc (2 * sizeof(double)) ;
        Matrix->SMPkluMatrix->KLUmatrixValueComplexCOOforCIDER = (double *) malloc (2 * Matrix->SMPkluMatrix->KLUmatrixN * Matrix->SMPkluMatrix->KLUmatrixN * sizeof(double)) ;

        /* Pre-set the values of Row and Col */
        for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN * Matrix->SMPkluMatrix->KLUmatrixN ; i++) {
            Matrix->SMPkluMatrix->KLUmatrixRowCOOforCIDER [i] = -1 ;
            Matrix->SMPkluMatrix->KLUmatrixColCOOforCIDER [i] = -1 ;
        }

        return spOKAY ;
    } else {
        Matrix->SPmatrix = spCreate (size, (int)KLUmatrixIsComplex, &Error) ;
        return Error ;
    }
}
#endif

/*
 * SMPdestroy()
 */

void
SMPdestroy (SMPmatrix *Matrix)
{
    if (Matrix->CKTkluMODE)
    {
        klu_free_numeric (&(Matrix->SMPkluMatrix->KLUmatrixNumeric), Matrix->SMPkluMatrix->KLUmatrixCommon) ;
        klu_free_symbolic (&(Matrix->SMPkluMatrix->KLUmatrixSymbolic), Matrix->SMPkluMatrix->KLUmatrixCommon) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAp) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAi) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAx) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAxComplex) ;
        free (Matrix->SMPkluMatrix->KLUmatrixIntermediate) ;
        free (Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex) ;
        free (Matrix->SMPkluMatrix->KLUmatrixBindStructCOO) ;
//        free (Matrix->SMPkluMatrix->KLUmatrixNodeCollapsingOldToNew) ;
        free (Matrix->SMPkluMatrix->KLUmatrixNodeCollapsingNewToOld) ;
        free (Matrix->SMPkluMatrix->KLUmatrixTrashCOO) ;
        Matrix->SMPkluMatrix->KLUmatrixAp = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAi = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAx = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAxComplex = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixIntermediate = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixBindStructCOO = NULL ;
//        Matrix->SMPkluMatrix->KLUmatrixNodeCollapsingOldToNew = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixNodeCollapsingNewToOld = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixTrashCOO = NULL ;
        free (Matrix->SMPkluMatrix->KLUmatrixDiag) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAxTls) ;
        free (Matrix->SMPkluMatrix->KLUmatrixRhsTls) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAxResConst) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAxCapUnit) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAxResCapBase) ;
        free (Matrix->SMPkluMatrix->KLUmatrixRhsCapCache) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAxLatentDC) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAxLatentCap) ;
        free (Matrix->SMPkluMatrix->KLUmatrixRhsLatentDC) ;
        free (Matrix->SMPkluMatrix->KLUmatrixRhsLatentCap) ;
#if NGSPICE_OPT_GENERAL
        SMPkluSubDestroy (Matrix->SMPkluMatrix) ;
#endif
        free (Matrix->SMPkluMatrix->KLUmatrixCommon) ;
        free (Matrix->SMPkluMatrix) ;
    } else {
        spDestroy (Matrix->SPmatrix) ;
    }
}

#ifdef CIDER
void
SMPdestroyKLUforCIDER (SMPmatrix *Matrix)
{
    if (Matrix->CKTkluMODE)
    {
        klu_free_numeric (&(Matrix->SMPkluMatrix->KLUmatrixNumeric), Matrix->SMPkluMatrix->KLUmatrixCommon) ;
        klu_free_symbolic (&(Matrix->SMPkluMatrix->KLUmatrixSymbolic), Matrix->SMPkluMatrix->KLUmatrixCommon) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAp) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAi) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAxComplex) ;
        free (Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex) ;
        free (Matrix->SMPkluMatrix->KLUmatrixBindStructForCIDER) ;
        free (Matrix->SMPkluMatrix->KLUmatrixColCOOforCIDER) ;
        free (Matrix->SMPkluMatrix->KLUmatrixRowCOOforCIDER) ;
        free (Matrix->SMPkluMatrix->KLUmatrixValueComplexCOOforCIDER) ;
        free (Matrix->SMPkluMatrix->KLUmatrixTrashCOO) ;
        Matrix->SMPkluMatrix->KLUmatrixAp = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAi = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAxComplex = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixBindStructForCIDER = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixColCOOforCIDER = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixRowCOOforCIDER = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixValueComplexCOOforCIDER = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixTrashCOO = NULL ;
        free (Matrix->SMPkluMatrix->KLUmatrixDiag) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAxTls) ;
        free (Matrix->SMPkluMatrix->KLUmatrixRhsTls) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAxResConst) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAxCapUnit) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAxResCapBase) ;
        free (Matrix->SMPkluMatrix->KLUmatrixRhsCapCache) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAxLatentDC) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAxLatentCap) ;
        free (Matrix->SMPkluMatrix->KLUmatrixRhsLatentDC) ;
        free (Matrix->SMPkluMatrix->KLUmatrixRhsLatentCap) ;
#if NGSPICE_OPT_GENERAL
        SMPkluSubDestroy (Matrix->SMPkluMatrix) ;
#endif
        free (Matrix->SMPkluMatrix->KLUmatrixCommon) ;
        free (Matrix->SMPkluMatrix) ;
    } else {
        spDestroy (Matrix->SPmatrix) ;
    }
}
#endif

/*
 * SMPpreOrder()
 */

int
SMPpreOrder (SMPmatrix *Matrix)
{
    if (Matrix->CKTkluMODE)
    {
        if (CircuitIsDigital() && Matrix->SMPkluMatrix->KLUmatrixN == 0) {
          // XSPICE pure digital circuits produce empty KLU matrix
          return 0 ;
        }

        Matrix->SMPkluMatrix->KLUmatrixSymbolic = klu_analyze ((int)Matrix->SMPkluMatrix->KLUmatrixN, Matrix->SMPkluMatrix->KLUmatrixAp,
                                                               Matrix->SMPkluMatrix->KLUmatrixAi, Matrix->SMPkluMatrix->KLUmatrixCommon) ;

        if (Matrix->SMPkluMatrix->KLUmatrixSymbolic == NULL)
        {
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_EMPTY_MATRIX)
            {
                fprintf (stderr, "Error (PreOrder): KLU Matrix is empty\n") ;
                return 0 ;
            } else {
                fprintf (stderr, "Error (PreOrder): KLUsymbolic object is NULL. A problem occurred\n") ;
                return 1 ;
            }
        } else {
            return 0 ;
        }
    } else {
        spMNA_Preorder (Matrix->SPmatrix) ;
        return spError (Matrix->SPmatrix) ;
    }
}

/*
 * SMPprintRHS()
 */

void
SMPprintRHS (SMPmatrix *Matrix, char *Filename, RealVector RHS, RealVector iRHS)
{
    if (!Matrix->CKTkluMODE)
        spFileVector (Matrix->SPmatrix, Filename, RHS, iRHS) ;
}

/*
 * SMPprint()
 */

void
SMPprint (SMPmatrix *Matrix, char *Filename)
{
    if (Matrix->CKTkluMODE)
    {
        if (Matrix->SMPkluMatrix->KLUmatrixIsComplex)
        {
            klu_z_print (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi, Matrix->SMPkluMatrix->KLUmatrixAxComplex,
                         (int)Matrix->SMPkluMatrix->KLUmatrixN, NULL, NULL) ;
        } else {
            klu_print (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi, Matrix->SMPkluMatrix->KLUmatrixAx,
                       (int)Matrix->SMPkluMatrix->KLUmatrixN, NULL, NULL) ;
        }
    } else {
        if (Filename)
            spFileMatrix (Matrix->SPmatrix, Filename, "Circuit Matrix", 0, 1, 1) ;
        else
            spPrint (Matrix->SPmatrix, 0, 1, 1) ;
    }
}

#ifdef CIDER
void
SMPprintKLUforCIDER (SMPmatrix *Matrix, char *Filename)
{
    unsigned int i ;
    double *KLUmatrixAx ;

    if (Matrix->CKTkluMODE)
    {
        if (Matrix->SMPkluMatrix->KLUmatrixIsComplex)
        {
            klu_z_print (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi, Matrix->SMPkluMatrix->KLUmatrixAxComplex,
                         (int)Matrix->SMPkluMatrix->KLUmatrixN, NULL, NULL) ;
        } else {
            /* Allocate the Real Matrix */
            KLUmatrixAx = (double *) malloc (Matrix->SMPkluMatrix->KLUmatrixNZ * sizeof(double)) ;

            /* Copy the Complex Matrix into the Real Matrix */
            for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixNZ ; i++) {
                KLUmatrixAx [i] = Matrix->SMPkluMatrix->KLUmatrixAxComplex [2 * i] ;
            }

            /* Print the Real Matrix */
            klu_print (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi, KLUmatrixAx, (int)Matrix->SMPkluMatrix->KLUmatrixN, NULL, NULL) ;

            /* Free the Real Matrix Storage */
            free (KLUmatrixAx) ;
        }
    } else {
        if (Filename)
            spFileMatrix (Matrix->SPmatrix, Filename, "Circuit Matrix", 0, 1, 1) ;
        else
            spPrint (Matrix->SPmatrix, 0, 1, 1) ;
    }
}
#endif

/*
 * SMPgetError()
 */
void
SMPgetError (SMPmatrix *Matrix, int *Col, int *Row)
{
    if (Matrix->CKTkluMODE)
    {
        if (Matrix->SMPkluMatrix->KLUmatrixNZ == 0) {
            *Row = 0 ;
            *Col = 0 ;
        } else {
            *Row = Matrix->SMPkluMatrix->KLUmatrixCommon->singular_col + 1 ;
            *Col = Matrix->SMPkluMatrix->KLUmatrixCommon->singular_col + 1 ;
        }
    } else {
        spWhereSingular (Matrix->SPmatrix, Row, Col) ;
    }
}

void
spDeterminant_KLU (SMPmatrix *Matrix, int *pExponent, RealNumber *pDeterminant, RealNumber *piDeterminant)
{
    int I, Size ;
    RealNumber Norm, nr, ni ;
    ComplexNumber Pivot, cDeterminant, Udiag ;

    int *P, *Q ;
    double *Rs, *Ux, *Uz ;
    unsigned int nSwap, nSwapP, nSwapQ ;

#define  NORM(a)     (nr = ABS((a).Real), ni = ABS((a).Imag), MAX (nr,ni))

    *pExponent = 0 ;

    if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_SINGULAR)
    {
	*pDeterminant = 0.0 ;
        if (Matrix->SMPkluMatrix->KLUmatrixIsComplex == KLUMatrixComplex)
        {
            *piDeterminant = 0.0 ;
        }
        return ;
    }

    Size = (int)Matrix->SMPkluMatrix->KLUmatrixN ;
    I = 0 ;

    P = (int *) malloc ((size_t)Matrix->SMPkluMatrix->KLUmatrixN * sizeof (int)) ;
    Q = (int *) malloc ((size_t)Matrix->SMPkluMatrix->KLUmatrixN * sizeof (int)) ;

    Ux = (double *) malloc ((size_t)Matrix->SMPkluMatrix->KLUmatrixN * sizeof (double)) ;

    Rs = (double *) malloc ((size_t)Matrix->SMPkluMatrix->KLUmatrixN * sizeof (double)) ;

    if (Matrix->SMPkluMatrix->KLUmatrixIsComplex == KLUMatrixComplex)        /* Complex Case. */
    {
	cDeterminant.Real = 1.0 ;
        cDeterminant.Imag = 0.0 ;

        Uz = (double *) malloc ((size_t)Matrix->SMPkluMatrix->KLUmatrixN * sizeof (double)) ;
/*
        int *Lp, *Li, *Up, *Ui, *Fp, *Fi, *P, *Q ;
        double *Lx, *Lz, *Ux, *Uz, *Fx, *Fz, *Rs ;
        Lp = (int *) malloc (((size_t)Matrix->CKTkluN + 1) * sizeof (int)) ;
        Li = (int *) malloc ((size_t)Matrix->CKTkluNumeric->lnz * sizeof (int)) ;
        Lx = (double *) malloc ((size_t)Matrix->CKTkluNumeric->lnz * sizeof (double)) ;
        Lz = (double *) malloc ((size_t)Matrix->CKTkluNumeric->lnz * sizeof (double)) ;
        Up = (int *) malloc (((size_t)Matrix->CKTkluN + 1) * sizeof (int)) ;
        Ui = (int *) malloc ((size_t)Matrix->CKTkluNumeric->unz * sizeof (int)) ;
        Ux = (double *) malloc ((size_t)Matrix->CKTkluNumeric->unz * sizeof (double)) ;
        Uz = (double *) malloc ((size_t)Matrix->CKTkluNumeric->unz * sizeof (double)) ;
        Fp = (int *) malloc (((size_t)Matrix->CKTkluN + 1) * sizeof (int)) ;
        Fi = (int *) malloc ((size_t)Matrix->CKTkluNumeric->Offp [Matrix->CKTkluN] * sizeof (int)) ;
        Fx = (double *) malloc ((size_t)Matrix->CKTkluNumeric->Offp [Matrix->CKTkluN] * sizeof (double)) ;
        Fz = (double *) malloc ((size_t)Matrix->CKTkluNumeric->Offp [Matrix->CKTkluN] * sizeof (double)) ;
        klu_z_extract (Matrix->CKTkluNumeric, Matrix->CKTkluSymbolic,
                       Lp, Li, Lx, Lz,
                       Up, Ui, Ux, Uz,
                       Fp, Fi, Fx, Fz,
                       P, Q, Rs, NULL,
                       Matrix->CKTkluCommon) ;
*/
        klu_z_extract_Udiag (Matrix->SMPkluMatrix->KLUmatrixNumeric, Matrix->SMPkluMatrix->KLUmatrixSymbolic, Ux, Uz, P, Q, Rs, Matrix->SMPkluMatrix->KLUmatrixCommon) ;
/*
        for (I = 0 ; I < Matrix->CKTkluNumeric->lnz ; I++)
        {
            printf ("L - Value: %-.9g\t%-.9g\n", Lx [I], Lz [I]) ;
        }
        for (I = 0 ; I < Matrix->CKTkluNumeric->unz ; I++)
        {
            printf ("U - Value: %-.9g\t%-.9g\n", Ux [I], Uz [I]) ;
        }
        for (I = 0 ; I < Matrix->CKTkluNumeric->Offp [Matrix->CKTkluN] ; I++)
        {
            printf ("F - Value: %-.9g\t%-.9g\n", Fx [I], Fz [I]) ;
        }

        for (I = 0 ; I < Matrix->CKTkluN ; I++)
        {
            printf ("U - Value: %-.9g\t%-.9g\n", Ux [I], Uz [I]) ;
        }
*/
        nSwapP = 0 ;
        for (I = 0 ; I < (int)Matrix->SMPkluMatrix->KLUmatrixN ; I++)
        {
            if (P [I] != I)
            {
                nSwapP++ ;
            }
        }
        nSwapP /= 2 ;

        nSwapQ = 0 ;
        for (I = 0 ; I < (int)Matrix->SMPkluMatrix->KLUmatrixN ; I++)
        {
            if (Q [I] != I)
            {
                nSwapQ++ ;
            }
        }
        nSwapQ /= 2 ;

        nSwap = nSwapP + nSwapQ ;
/*
        free (Lp) ;
        free (Li) ;
        free (Lx) ;
        free (Lz) ;
        free (Up) ;
        free (Ui) ;
        free (Fp) ;
        free (Fi) ;
        free (Fx) ;
        free (Fz) ;
*/
        I = 0 ;
        while (I < Size)
        {
            Udiag.Real = 1 / (Ux [I] * Rs [I]) ;
            Udiag.Imag = Uz [I] * Rs [I] ;

//            printf ("Udiag.Real: %-.9g\tUdiag.Imag %-.9g\n", Udiag.Real, Udiag.Imag) ;

            CMPLX_RECIPROCAL (Pivot, Udiag) ;
            CMPLX_MULT_ASSIGN (cDeterminant, Pivot) ;

//            printf ("cDeterminant.Real: %-.9g\tcDeterminant.Imag %-.9g\n", cDeterminant.Real, cDeterminant.Imag) ;

	    /* Scale Determinant. */
            Norm = NORM (cDeterminant) ;
            if (Norm != 0.0)
            {
		while (Norm >= 1.0e12)
                {
		    cDeterminant.Real *= 1.0e-12 ;
                    cDeterminant.Imag *= 1.0e-12 ;
                    *pExponent += 12 ;
                    Norm = NORM (cDeterminant) ;
                }
                while (Norm < 1.0e-12)
                {
		    cDeterminant.Real *= 1.0e12 ;
                    cDeterminant.Imag *= 1.0e12 ;
                    *pExponent -= 12 ;
                    Norm = NORM (cDeterminant) ;
                }
            }
            I++ ;
        }

	/* Scale Determinant again, this time to be between 1.0 <= x < 10.0. */
        Norm = NORM (cDeterminant) ;
        if (Norm != 0.0)
        {
	    while (Norm >= 10.0)
            {
		cDeterminant.Real *= 0.1 ;
                cDeterminant.Imag *= 0.1 ;
                (*pExponent)++ ;
                Norm = NORM (cDeterminant) ;
            }
            while (Norm < 1.0)
            {
		cDeterminant.Real *= 10.0 ;
                cDeterminant.Imag *= 10.0 ;
                (*pExponent)-- ;
                Norm = NORM (cDeterminant) ;
            }
        }
        if (nSwap % 2 != 0)
        {
            CMPLX_NEGATE (cDeterminant) ;
        }

        *pDeterminant = cDeterminant.Real ;
        *piDeterminant = cDeterminant.Imag ;

        free (Uz) ;
    }
    else
    {
	/* Real Case. */
        *pDeterminant = 1.0 ;

        klu_extract_Udiag (Matrix->SMPkluMatrix->KLUmatrixNumeric, Matrix->SMPkluMatrix->KLUmatrixSymbolic, Ux, P, Q, Rs, Matrix->SMPkluMatrix->KLUmatrixCommon) ;

        nSwapP = 0 ;
        for (I = 0 ; I < (int)Matrix->SMPkluMatrix->KLUmatrixN ; I++)
        {
            if (P [I] != I)
            {
                nSwapP++ ;
            }
        }
        nSwapP /= 2 ;

        nSwapQ = 0 ;
        for (I = 0 ; I < (int)Matrix->SMPkluMatrix->KLUmatrixN ; I++)
        {
            if (Q [I] != I)
            {
                nSwapQ++ ;
            }
        }
        nSwapQ /= 2 ;

        nSwap = nSwapP + nSwapQ ;

        while (I < Size)
        {
            *pDeterminant /= (Ux [I] * Rs [I]) ;

	    /* Scale Determinant. */
            if (*pDeterminant != 0.0)
            {
		while (ABS(*pDeterminant) >= 1.0e12)
                {
		    *pDeterminant *= 1.0e-12 ;
                    *pExponent += 12 ;
                }
                while (ABS(*pDeterminant) < 1.0e-12)
                {
		    *pDeterminant *= 1.0e12 ;
                    *pExponent -= 12 ;
                }
            }
            I++ ;
        }

	/* Scale Determinant again, this time to be between 1.0 <= x <
           10.0. */
        if (*pDeterminant != 0.0)
        {
	    while (ABS(*pDeterminant) >= 10.0)
            {
		*pDeterminant *= 0.1 ;
                (*pExponent)++ ;
            }
            while (ABS(*pDeterminant) < 1.0)
            {
		*pDeterminant *= 10.0 ;
                (*pExponent)-- ;
            }
        }
        if (nSwap % 2 != 0)
        {
            *pDeterminant = -*pDeterminant ;
        }
    }

    free (P) ;
    free (Q) ;
    free (Ux) ;
    free (Rs) ;
}

/*
 * SMPcProdDiag()
 *    note: obsolete for Spice3d2 and later
 */
int
SMPcProdDiag (SMPmatrix *Matrix, SPcomplex *pMantissa, int *pExponent)
{
    if (Matrix->CKTkluMODE)
    {
        spDeterminant_KLU (Matrix, pExponent, &(pMantissa->real), &(pMantissa->imag)) ;
    } else {
        spDeterminant (Matrix->SPmatrix, pExponent, &(pMantissa->real), &(pMantissa->imag)) ;
    }
    return spError (Matrix->SPmatrix) ;
}

/*
 * SMPcDProd()
 */
int
SMPcDProd (SMPmatrix *Matrix, SPcomplex *pMantissa, int *pExponent)
{
    double	re, im, x, y, z;
    int		p;

    if (Matrix->CKTkluMODE)
    {
        spDeterminant_KLU (Matrix, &p, &re, &im) ;
    } else {
        spDeterminant (Matrix->SPmatrix, &p, &re, &im) ;
    }

#ifndef M_LN2
#define M_LN2   0.69314718055994530942
#endif
#ifndef M_LN10
#define M_LN10  2.30258509299404568402
#endif

#ifdef debug_print
    printf ("Determinant 10: (%20g,%20g)^%d\n", re, im, p) ;
#endif

    /* Convert base 10 numbers to base 2 numbers, for comparison */
    y = p * M_LN10 / M_LN2;
    x = (int) y;
    y -= x;

    /* ASSERT
     *	x = integral part of exponent, y = fraction part of exponent
     */

    /* Fold in the fractional part */
#ifdef debug_print
    printf (" ** base10 -> base2 int =  %g, frac = %20g\n", x, y) ;
#endif
    z = pow (2.0, y) ;
    re *= z ;
    im *= z ;
#ifdef debug_print
    printf (" ** multiplier = %20g\n", z) ;
#endif

    /* Re-normalize (re or im may be > 2.0 or both < 1.0 */
    if (re != 0.0)
    {
	y = logb (re) ;
	if (im != 0.0)
	    z = logb (im) ;
	else
	    z = 0 ;
    } else if (im != 0.0) {
	z = logb (im) ;
	y = 0 ;
    } else {
	/* Singular */
	/*printf("10 -> singular\n");*/
	y = 0 ;
	z = 0 ;
    }

#ifdef debug_print
    printf (" ** renormalize changes = %g,%g\n", y, z) ;
#endif
    if (y < z)
	y = z ;

    *pExponent = (int)(x + y) ;
    x = scalbn (re, (int) -y) ;
    z = scalbn (im, (int) -y) ;
#ifdef debug_print
    printf (" ** values are: re %g, im %g, y %g, re' %g, im' %g\n", re, im, y, x, z) ;
#endif
    pMantissa->real = scalbn (re, (int) -y) ;
    pMantissa->imag = scalbn (im, (int) -y) ;

#ifdef debug_print
    printf ("Determinant 10->2: (%20g,%20g)^%d\n", pMantissa->real, pMantissa->imag, *pExponent) ;
#endif

    if (Matrix->CKTkluMODE)
    {
        return 0 ;
    } else {
        return spError (Matrix->SPmatrix) ;
    }
}



/*
 *  The following routines need internal knowledge of the Sparse data
 *  structures.
 */

/*
 *  LOAD GMIN
 *
 *  This routine adds Gmin to each diagonal element.  Because Gmin is
 *  added to the current diagonal, which may bear little relation to
 *  what the outside world thinks is a diagonal, and because the
 *  elements that are diagonals may change after calling spOrderAndFactor,
 *  use of this routine is not recommended.  It is included here simply
 *  for compatibility with Spice3.
 */


static void
LoadGmin_CSC (double **diag, unsigned int n, double Gmin)
{
    unsigned int i ;

    if (Gmin != 0.0) {
        for (i = 0 ; i < n ; i++) {
            if (diag [i] != NULL) {
                // Not all the elements on the diagonal are present, when the circuit is parsed
                *(diag [i]) += Gmin ;
            }
        }
    }
}

static void
LoadGmin (SMPmatrix *eMatrix, double Gmin)
{
    MatrixPtr Matrix = eMatrix->SPmatrix ;
    int I ;
    ArrayOfElementPtrs Diag ;
    ElementPtr diag ;

    /* Begin `LoadGmin'. */
    assert (IS_SPARSE (Matrix)) ;

    if (Gmin != 0.0) {
	Diag = Matrix->Diag ;
	for (I = Matrix->Size ; I > 0 ; I--)
        {
	    if ((diag = Diag [I]) != NULL)
		diag->Real += Gmin ;
	}
    }
    return ;
}




/*
 *  FIND ELEMENT
 *
 *  This routine finds an element in the matrix by row and column number.
 *  If the element exists, a pointer to it is returned.  If not, then NULL
 *  is returned unless the CreateIfMissing flag is TRUE, in which case a
 *  pointer to the new element is returned.
 */

SMPelement *
SMPfindElt (SMPmatrix *eMatrix, int Row, int Col, int CreateIfMissing)
{
    MatrixPtr Matrix = eMatrix->SPmatrix ;

    if (eMatrix->CKTkluMODE)
    {
        int i ;

        Row = Row - 1 ;
        Col = Col - 1 ;
        if (Col < 0) {
//            printf ("Information: Cannot find an element with row '%d' and column '%d' in the KLU matrix\n", Row, Col) ;
            return NULL ;
        }
        for (i = eMatrix->SMPkluMatrix->KLUmatrixAp [Col] ; i < eMatrix->SMPkluMatrix->KLUmatrixAp [Col + 1] ; i++) {
            if (eMatrix->SMPkluMatrix->KLUmatrixAi [i] == Row) {
                if (eMatrix->SMPkluMatrix->KLUmatrixIsComplex == KLUmatrixReal) {
                    return (SMPelement *) &(eMatrix->SMPkluMatrix->KLUmatrixAx [i]) ;
                } else if (eMatrix->SMPkluMatrix->KLUmatrixIsComplex == KLUMatrixComplex) {
                    return (SMPelement *) &(eMatrix->SMPkluMatrix->KLUmatrixAxComplex [2 * i]) ;
                } else {
                    printf ("Information: Cannot find an element with row '%d' and column '%d' in the KLU matrix\n", Row, Col) ;
                    return NULL ;
                }
            }
        }
        return NULL ;
    } else {
        ElementPtr Element ;

        /* Begin `SMPfindElt'. */
        assert (IS_SPARSE (Matrix)) ;
        Row = Matrix->ExtToIntRowMap [Row] ;
        Col = Matrix->ExtToIntColMap [Col] ;
        Element = Matrix->FirstInCol [Col] ;
        Element = spcFindElementInCol (Matrix, &Element, Row, Col, CreateIfMissing) ;
        return (SMPelement *)Element ;
    }
}

/* XXX The following should probably be implemented in spUtils */

/*
 * SMPcZeroCol()
 */
int
SMPcZeroCol (SMPmatrix *eMatrix, int Col)
{
    MatrixPtr Matrix = eMatrix->SPmatrix ;
    ElementPtr	Element ;

    if (eMatrix->CKTkluMODE)
    {
        int i ;
        for (i = eMatrix->SMPkluMatrix->KLUmatrixAp [Col - 1] ; i < eMatrix->SMPkluMatrix->KLUmatrixAp [Col] ; i++)
        {
            eMatrix->SMPkluMatrix->KLUmatrixAxComplex [2 * i] = 0 ;
            eMatrix->SMPkluMatrix->KLUmatrixAxComplex [2 * i + 1] = 0 ;
        }
        return 0 ;
    } else {
        Col = Matrix->ExtToIntColMap [Col] ;
        for (Element = Matrix->FirstInCol [Col] ; Element != NULL ; Element = Element->NextInCol)
        {
            Element->Real = 0.0 ;
            Element->Imag = 0.0 ;
        }
        return spError (Matrix) ;
    }
}

/*
 * SMPcAddCol()
 */
int
SMPcAddCol (SMPmatrix *eMatrix, int Accum_Col, int Addend_Col)
{
    MatrixPtr Matrix = eMatrix->SPmatrix ;
    ElementPtr	Accum, Addend, *Prev ;

    Accum_Col = Matrix->ExtToIntColMap [Accum_Col] ;
    Addend_Col = Matrix->ExtToIntColMap [Addend_Col] ;

    Addend = Matrix->FirstInCol [Addend_Col] ;
    Prev = &Matrix->FirstInCol [Accum_Col] ;
    Accum = *Prev;

    while (Addend != NULL)
    {
	while (Accum && Accum->Row < Addend->Row)
        {
	    Prev = &Accum->NextInCol ;
	    Accum = *Prev ;
	}
	if (!Accum || Accum->Row > Addend->Row)
        {
	    Accum = spcCreateElement (Matrix, Addend->Row, Accum_Col, Prev, 0) ;
	}
	Accum->Real += Addend->Real ;
	Accum->Imag += Addend->Imag ;
	Addend = Addend->NextInCol ;
    }

    return spError (Matrix) ;
}

/*
 * SMPzeroRow()
 */
int
SMPzeroRow (SMPmatrix *eMatrix, int Row)
{
    MatrixPtr Matrix = eMatrix->SPmatrix ;
    ElementPtr	Element ;

    Row = Matrix->ExtToIntColMap [Row] ;

    if (Matrix->RowsLinked == NO)
	spcLinkRows (Matrix) ;

    if (Matrix->PreviousMatrixWasComplex || Matrix->Complex)
    {
	for (Element = Matrix->FirstInRow[Row] ; Element != NULL; Element = Element->NextInRow)
	{
	    Element->Real = 0.0 ;
	    Element->Imag = 0.0 ;
	}
    } else {
	for (Element = Matrix->FirstInRow [Row] ; Element != NULL ; Element = Element->NextInRow)
	{
	    Element->Real = 0.0 ;
	}
    }

    return spError (Matrix) ;
}

/*
 * SMPconstMult()
 */
void
SMPconstMult (SMPmatrix *Matrix, double constant)
{
    if (Matrix->CKTkluMODE)
    {
        if (Matrix->SMPkluMatrix->KLUmatrixIsComplex)
        {
            klu_z_constant_multiply (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAxComplex,
                                     (int)Matrix->SMPkluMatrix->KLUmatrixN, Matrix->SMPkluMatrix->KLUmatrixCommon, constant) ;
        } else {
            klu_constant_multiply (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAx,
                                   (int)Matrix->SMPkluMatrix->KLUmatrixN, Matrix->SMPkluMatrix->KLUmatrixCommon, constant) ;
        }
    } else {
        spConstMult (Matrix->SPmatrix, constant) ;
    }
}

/*
 * SMPmultiply()
 */
void
SMPmultiply (SMPmatrix *Matrix, double *RHS, double *Solution, double *iRHS, double *iSolution)
{
    if (Matrix->CKTkluMODE)
    {
        int *Ap_CSR, *Ai_CSR ;
        double *Ax_CSR ;

        Ap_CSR = (int *) malloc ((size_t)(Matrix->SMPkluMatrix->KLUmatrixN + 1) * sizeof (int)) ;
        Ai_CSR = (int *) malloc ((size_t)Matrix->SMPkluMatrix->KLUmatrixNZ * sizeof (int)) ;

        if (Matrix->SMPkluMatrix->KLUmatrixIsComplex)
        {
            Ax_CSR = (double *) malloc ((size_t)(2 * Matrix->SMPkluMatrix->KLUmatrixNZ) * sizeof (double)) ;
            klu_z_convert_matrix_in_CSR (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi, Matrix->SMPkluMatrix->KLUmatrixAxComplex, Ap_CSR,
                                         Ai_CSR, Ax_CSR, (int)Matrix->SMPkluMatrix->KLUmatrixN, (int)Matrix->SMPkluMatrix->KLUmatrixNZ, Matrix->SMPkluMatrix->KLUmatrixCommon) ;
            klu_z_matrix_vector_multiply (Ap_CSR, Ai_CSR, Ax_CSR, RHS, Solution, iRHS, iSolution, NULL, NULL,
                                          (int)Matrix->SMPkluMatrix->KLUmatrixN, Matrix->SMPkluMatrix->KLUmatrixCommon) ;
        } else {
            Ax_CSR = (double *) malloc ((size_t)Matrix->SMPkluMatrix->KLUmatrixNZ * sizeof (double)) ;
            klu_convert_matrix_in_CSR (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi, Matrix->SMPkluMatrix->KLUmatrixAx, Ap_CSR, Ai_CSR,
                                       Ax_CSR, (int)Matrix->SMPkluMatrix->KLUmatrixN, (int)Matrix->SMPkluMatrix->KLUmatrixNZ, Matrix->SMPkluMatrix->KLUmatrixCommon) ;
            klu_matrix_vector_multiply (Ap_CSR, Ai_CSR, Ax_CSR, RHS, Solution, NULL, NULL,
                                        (int)Matrix->SMPkluMatrix->KLUmatrixN, Matrix->SMPkluMatrix->KLUmatrixCommon) ;
            iSolution = iRHS ;
        }

        free (Ap_CSR) ;
        free (Ai_CSR) ;
        free (Ax_CSR) ;
    } else {
        spMultiply (Matrix->SPmatrix, RHS, Solution, iRHS, iSolution) ;
    }
}

