#include "oocdag.h"
#define PRINT_H2D       0
#define PRINT_D2H       0
#define PRINT_dpotrf    0
#define PRINT_dtrsm     0
#define PRINT_dsyrk     0
#define PRINT_dgemm     0
#define printif(x,...) if(x)printf(__VA_ARGS__)

  static char COMM_MASK[] = {0xC0, 0x00, 0x00};
//static char *COMM_MASK = NULL;
  static char COMP_MASK[] = {0x3F, 0xFF, 0xFF};
//static char *COMP_MASK = NULL;


void CORE_H2D(Quark *quark)
{
    int M; int N; double *H; int ldh; double *D; int NB;
    quark_unpack_args_6(quark, M, N, H, ldh, D, NB);
    int r = QUARK_Thread_Rank(quark);
    printif(PRINT_H2D, "H2D start %d %lf\n", r, MPI_Wtime());
//  printf("thread %d transfer %lf %p -> %p\n", r, H[0], H, D);
    #ifdef USE_MIC
        offload_dSetMatrix(M, N, H, ldh, D, NB, r);
    #else
    {
        cublasStatus cu_status = cublasSetMatrix(M, N, sizeof(double), H, ldh, D, NB);
        CHKERR(cu_status);
    }
    #endif
    printif(PRINT_H2D, "H2D end %d %lf\n", r, MPI_Wtime());
}

void QUARK_H2D(Quark *quark, Quark_Task_Flags *task_flags, 
        int M, int N, double *H, int ldh, double *D, int NB)
{
    QUARK_Insert_Task(quark, CORE_H2D, task_flags,
        sizeof(int),                &M,         VALUE,
        sizeof(int),                &N,         VALUE,
        sizeof(double),             H,          INPUT,
        sizeof(int),                &ldh,       VALUE,
        sizeof(double),             D,          OUTPUT,// | LOCALITY,
        sizeof(int),                &NB,        VALUE,
        0);
}

void CORE_D2H(Quark *quark)
{
    int M; int N; double *D; int NB; double *H; int ldh;
    quark_unpack_args_6(quark, M, N, D, NB, H, ldh);
    int r = QUARK_Thread_Rank(quark);
    printif(PRINT_D2H, "D2H start %d %lf\n", r, MPI_Wtime());
    #ifdef USE_MIC
        offload_dGetMatrix(M, N, D, NB, H, ldh, r);
    #else
    {
        cublasStatus cu_status = cublasGetMatrix(M, N, sizeof(double), D, NB, H, ldh);
        CHKERR(cu_status);
    }
    #endif
//  printf("thread %d transfer %lf %p <- %p\n", r, H[0], H, D);
    printif(PRINT_D2H, "D2H end %d %lf\n", r, MPI_Wtime());
}

void QUARK_D2H(Quark *quark, Quark_Task_Flags *task_flags,
        int M, int N, double *D, int NB, double *H, int ldh)
{
    QUARK_Insert_Task(quark, CORE_D2H, task_flags,
        sizeof(int),                &M,         VALUE,
        sizeof(int),                &N,         VALUE,
        sizeof(double),             D,          INPUT,// | LOCALITY,
        sizeof(int),                &NB,        VALUE,
        sizeof(double),             H,          OUTPUT,
        sizeof(int),                &ldh,       VALUE,
        0);
}

/* dpotrf */
void CORE_incore_dpotrf(Quark *quark)   
{
    char uplo; int N; double *A; int NB;
    quark_unpack_args_4(quark, uplo, N, A, NB);
    int info;
    int r = QUARK_Thread_Rank(quark);
    printif(PRINT_dpotrf, "dpotrf start %d %lf\n", r, MPI_Wtime());
    #ifdef USE_HOST_DPOTRF
        dpotrf_(&uplo, &N, A, &NB, &info); //computation on host
    #else
        #ifdef USE_MIC
            offload_dpotrf(&uplo, &N, A, &NB, &info, r);
        #else
//          ierr = magma_dpotrf_gpu(uplo, &N, A, &NB, &info);
        #endif
    #endif
    printif(info != 0, "dpotrf return info=%d\n", info);
    printif(PRINT_dpotrf, "dpotrf end %d %lf\n", r, MPI_Wtime());
}

void QUARK_incore_dpotrf(Quark *quark, Quark_Task_Flags *task_flags,
        const char *uplo, int N, double *A, int NB)
{
    QUARK_Insert_Task(quark, CORE_incore_dpotrf, task_flags,
        sizeof(char),           uplo,   VALUE,
        sizeof(int),            &N,     VALUE,
        sizeof(double),         A,      INOUT,   
        sizeof(int),            &NB,    VALUE,
        0);
}

/* dtrsm */
void CORE_incore_dtrsm(Quark *quark)
{
    char side; char uplo; char transA; char diag;
    int M; int N; double alpha; double *A; double *B; int NB;
    quark_unpack_args_10(quark, side, uplo, transA, diag, M, N, alpha, A, B, NB);
    int r = QUARK_Thread_Rank(quark);
    printif(PRINT_dtrsm, "dtrsm start %d %lf\n", r, MPI_Wtime());
    #ifdef USE_MIC
        offload_dtrsm(&side, &uplo, &transA, &diag,
            &M, &N, &alpha, A, &NB, B, &NB, r);
    #else
        #ifdef USE_CUBLASV2
        {
            cublasSideMode_t cside = (side=='L')||(side=='l') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
            cublasFillMode_t cuplo = (uplo=='L')||(side=='l') ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
            cublasOperation_t ctrans = (transA == 'N')||(transA == 'n') ? CUBLAS_OP_N :
                                       (transA == 'T')||(transA == 't') ? CUBLAS_OP_T : CUBLAS_OP_C;
            cublasDiagType_t cdiag = (diag=='N')||(diag=='n') ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT;
            
            cublasDtrsm_v2(worker_handle[r], cside, cuplo, ctrans, cdiag, 
                M, N, &alpha, A, NB, B, NB);
        }
        #else
            cublasDtrsm(side, uplo, trans, diag, 
                M, N, alpha, A, NB, B, NB);
        #endif
    #endif
    printif(PRINT_dtrsm, "dtrsm end %d %lf\n", r, MPI_Wtime());
}

void QUARK_incore_dtrsm(Quark *quark, Quark_Task_Flags *task_flags,
                       const char *side, const char *uplo, const char *transA, const char *diag,
                       int M, int N, double alpha, const double *A, double *B, int NB)
{
    QUARK_Insert_Task(quark, CORE_incore_dtrsm, task_flags,
        sizeof(char),           side,       VALUE,
        sizeof(char),           uplo,       VALUE,
        sizeof(char),           transA,     VALUE,
        sizeof(char),           diag,       VALUE,
        sizeof(int),            &M,         VALUE,
        sizeof(int),            &N,         VALUE,
        sizeof(double),         &alpha,     VALUE,
        sizeof(double),         A,          INPUT,
        sizeof(double),         B,          INOUT,// | LOCALITY, //LOCALITY = try to let same quark thread to use this data
        sizeof(int),            &NB,        VALUE,
        0);
}

/*dsyrk*/
void CORE_incore_dsyrk(Quark *quark)
{
    char uplo; char trans; int N; int K;
    double alpha; double *A; double beta; double *C; int NB;
    quark_unpack_args_9(quark, uplo, trans, N, K, alpha, A, beta, C, NB);
    int r = QUARK_Thread_Rank(quark);
    printif(PRINT_dsyrk, "dsyrk start %d %lf\n", r, MPI_Wtime());
    #ifdef USE_MIC
        offload_dsyrk(&uplo, &trans, &N, &K,
                    &alpha, A, &NB, &beta, C, &NB, r);
    #else
        #ifdef USE_CUBLASV2
        {
            cublasFillMode_t cuplo = (uplo == 'L')||(uplo == 'l') ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
            cublasOperation_t ctrans = (trans == 'N')||(trans == 'n') ? CUBLAS_OP_N :
                                       (trans == 'T')||(trans == 't') ? CUBLAS_OP_T : CUBLAS_OP_C;
            cublasDsyrk_v2(worker_handle[r], cuplo, ctrans, N, K,
                    &alpha, A, NB, &beta, C, NB);
        }
        #else
            cublasDsyrk(uplo, trans, N, K, 
                    alpha, A, NB, beta, C, NB);
        #endif
    #endif
    printif(PRINT_dsyrk, "dsyrk end %d %lf\n", r, MPI_Wtime());
}

void QUARK_incore_dsyrk(Quark *quark, Quark_Task_Flags *task_flags,
                      const char *uplo, const char *trans, int N, int K,
                      double alpha, const double *A, double beta, double *C, int NB)
{
    QUARK_Insert_Task(quark, CORE_incore_dsyrk, task_flags,
        sizeof(char),                   uplo,       VALUE,
        sizeof(char),                   trans,      VALUE,
        sizeof(int),                    &N,         VALUE,
        sizeof(int),                    &K,         VALUE,
        sizeof(double),                 &alpha,     VALUE,
        sizeof(double),                 A,          INPUT,
        sizeof(double),                 &beta,      VALUE,
        sizeof(double),                 C,          INOUT,
        sizeof(int),                    &NB,        VALUE,
        0);
}

/*dgemm*/
void CORE_incore_dgemm(Quark *quark)
{
    char transA; char transB; int M; int N; int K;
    double alpha; double *A; double *B; double beta; double *C; int NB;
    quark_unpack_args_11(quark, transA, transB, M, N, K, alpha, A, B, beta, C, NB);
    int r = QUARK_Thread_Rank(quark);
    printif(PRINT_dgemm, "dgemm start %d %lf\n", r, MPI_Wtime());
    #ifdef USE_MIC
        offload_dgemm(&transA, &transB, &M, &N, &K,
                   &alpha, A, &NB, B, &NB, &beta, C, &NB, r);
    #else
        #ifdef USE_CUBLASV2
        {
            cublasOperation_t ctransA = (transA == 'N')||(transA == 'n') ? CUBLAS_OP_N :
                                        (transA == 'T')||(transA == 't') ? CUBLAS_OP_T : CUBLAS_OP_C;
            cublasOperation_t ctransB = (transB == 'N')||(transB == 'n') ? CUBLAS_OP_N :
                                        (transB == 'T')||(transB == 't') ? CUBLAS_OP_T : CUBLAS_OP_C;
            cublasDgemm_v2(worker_handle[r], ctransA, ctransB, M, N, K,
                  &alpha, A, NB, B, NB, &beta, C, NB);
        }
        #else
            cublasDgemm(transA, transB, M, N, K,
                  alpha, A, NB, B, NB, beta, C, NB);
        #endif
    #endif
    printif(PRINT_dgemm, "dgemm end %d %lf\n", r, MPI_Wtime());
}

void QUARK_incore_dgemm(Quark *quark, Quark_Task_Flags *task_flags,
                      const char *transA, const char *transB, int M, int N, int K,
                      double alpha, const double *A, const double *B, double beta, double *C, int NB)

{
     QUARK_Insert_Task(quark, CORE_incore_dgemm, task_flags,
         sizeof(char),          transA,     VALUE,
         sizeof(char),          transB,     VALUE,
         sizeof(int),           &M,         VALUE,
         sizeof(int),           &N,         VALUE,
         sizeof(int),           &K,         VALUE,
         sizeof(double),        &alpha,     VALUE,
         sizeof(double),        A,          INPUT,
         sizeof(double),        B,          INPUT,
         sizeof(double),        &beta,      VALUE,
         sizeof(double),        C,          INOUT,
         sizeof(int),           &NB,        VALUE,
         0);
}

void A2Y(Quark *quark, double *A, double *Y, int LDA, int NB, int M, int N)
{
    #define A(ib,jb) A[(size_t)(jb)*NB*LDA+(ib)*NB]
    #define Y(ib,jb) Y[(size_t)((ib)+(jb)*Ym-((jb)*((jb)+1))/2)*NB*NB]
    Quark_Task_Flags tflags = Quark_Task_Flags_Initializer;
    QUARK_Task_Flag_Set(&tflags, TASK_COLOR, (intptr_t) "gray");
    QUARK_Task_Flag_Set(&tflags, TASK_LABEL, (intptr_t) "A2Y");
    QUARK_Task_Flag_Set(&tflags, TASK_PRIORITY, 1);
    QUARK_Task_Flag_Set(&tflags, TASK_LOCK_TO_THREAD_MASK, (intptr_t) COMM_MASK);
    int IB, JB, LM, LN;
    int ib, jb;
    int Ym = (M + NB - 1) / NB;
    for(JB = 0, jb = 0; JB < N; JB+=NB, jb++){
        LN = MIN(JB+NB, N) - JB;
        for(IB = JB, ib = jb; IB < M; IB+=NB, ib++){
            LM = MIN(IB+NB, M) - IB;
            /*copy one block*/
            QUARK_H2D(quark, &tflags, LM, LN, &A(ib,jb), LDA, &Y(ib,jb), NB);
        }
    }
    #undef A
    #undef Y
}

void Y2A(Quark *quark, double *Y, double *A, int LDA, int NB, int M, int N)
{
    #define A(ib,jb) A[(size_t)(jb)*NB*LDA+(ib)*NB]
    #define Y(ib,jb) Y[(size_t)((ib)+(jb)*Ym-((jb)*((jb)+1))/2)*NB*NB]
    Quark_Task_Flags tflags = Quark_Task_Flags_Initializer;
    QUARK_Task_Flag_Set(&tflags, TASK_COLOR, (intptr_t) "gray");
    QUARK_Task_Flag_Set(&tflags, TASK_LABEL, (intptr_t) "Y2A");
    QUARK_Task_Flag_Set(&tflags, TASK_PRIORITY, 1);
    QUARK_Task_Flag_Set(&tflags, TASK_LOCK_TO_THREAD_MASK, (intptr_t) COMM_MASK);
    int IB, JB, LM, LN;
    int ib, jb;
    int Ym = (M + NB - 1) / NB;
    for(JB = 0, jb = 0; JB < N; JB+=NB, jb++){
        LN = MIN(JB+NB, N) - JB;
        for(IB = JB, ib = jb; IB < M; IB+=NB, ib++){
            LM = MIN(IB+NB, M) - IB;
            /*copy one block*/
            QUARK_D2H(quark, &tflags, LM, LN, &Y(ib,jb), NB, &A(ib,jb), LDA);
        }
    } 
    #undef A
    #undef Y
}

void A2X(Quark *quark, double *A, int LDA, double *X, int NB, int K)
{
    #define A(ib) A[(size_t)(ib)*NB]
    #define X(ib) X[(size_t)(ib)*NB*NB]
    Quark_Task_Flags tflags = Quark_Task_Flags_Initializer;
    QUARK_Task_Flag_Set(&tflags, TASK_COLOR, (intptr_t) "gray");
    QUARK_Task_Flag_Set(&tflags, TASK_LABEL, (intptr_t) "A2X");
    QUARK_Task_Flag_Set(&tflags, TASK_PRIORITY, 1);
    QUARK_Task_Flag_Set(&tflags, TASK_LOCK_TO_THREAD_MASK, (intptr_t) COMM_MASK);
    int IB, LM, LN = NB;
    int ib;
    for(IB = 0, ib = 0; IB < K; IB+=NB, ib++){
        LM = MIN(IB+NB, K) - IB;
        /*copy one block*/
        QUARK_H2D(quark, &tflags, LM, LN, &A(ib), LDA, &X(ib), NB);
    }
    #undef A
    #undef X
}

// update Y(0:H,0:K) by X(0:H) on device
// NB and H determine the storage pattern of Y
void ooc_syrk(Quark *quark, double *X, double *Y, int H, int K, int NB)
{
    #define X(ib) X[(size_t)(ib)*NB*NB]
    #define Y(ib,jb) Y[(size_t)((ib)+(jb)*Ym-(jb)*((jb)+1)/2)*NB*NB]
    int IB, JB, LM, LN, LK = NB;
    int ib, jb;
    int Ym = (H + NB - 1) / NB;
    
    Quark_Task_Flags tflags3 = Quark_Task_Flags_Initializer;              
    QUARK_Task_Flag_Set(&tflags3, TASK_COLOR, (intptr_t) "yellow");
    QUARK_Task_Flag_Set(&tflags3, TASK_LABEL, (intptr_t) "SYRK");
    QUARK_Task_Flag_Set(&tflags3, TASK_LOCK_TO_THREAD_MASK, (intptr_t) COMP_MASK);

    Quark_Task_Flags tflags4 = Quark_Task_Flags_Initializer;
    QUARK_Task_Flag_Set(&tflags4, TASK_COLOR, (intptr_t) "forestgreen");
    QUARK_Task_Flag_Set(&tflags4, TASK_LABEL, (intptr_t) "GEMM");
    QUARK_Task_Flag_Set(&tflags4, TASK_LOCK_TO_THREAD_MASK, (intptr_t) COMP_MASK);

    for(JB = 0, jb = 0; JB < K; JB+=NB, jb++){
        IB = JB;
        ib = jb;
        LN = MIN(JB+NB, K) - JB;
        /* dsyrk update on diag */
        QUARK_incore_dsyrk(quark, &tflags3, "L", "N", LN, LK, -1.0, &X(ib), 1.0, &Y(ib,jb), NB);

        /* dgemm update on off-diag */
        for(IB = JB+NB, ib = jb+1; IB < H; IB+=NB, ib++){
            LM = MIN(IB+NB, H) - IB;
            QUARK_incore_dgemm(quark, &tflags4, "N", "T", LM, LN, LK, -1.0, &X(ib), &X(jb), 1.0, &Y(ib,jb), NB);
        }
    }
    #undef X
    #undef Y
}

void ooc_incore(Quark *quark, double *A, double *Y, int LDA, int NB, int M, int N)
{
    #define A(ib,jb) A[(size_t)(jb)*NB*LDA+(ib)*NB]
    #define Y(ib,jb) Y[(size_t)((ib)+(jb)*Ym-(jb)*((jb)+1)/2)*NB*NB]
    int IB, JB, LN, LM;
    int ib, jb;
    int H, K;
    int Ym = (M + NB - 1) / NB;

    Quark_Task_Flags tflags1 = Quark_Task_Flags_Initializer;
    QUARK_Task_Flag_Set(&tflags1, TASK_COLOR, (intptr_t) "gray");
    QUARK_Task_Flag_Set(&tflags1, TASK_LABEL, (intptr_t) "Y2A");
    QUARK_Task_Flag_Set(&tflags1, TASK_PRIORITY, 1);
    QUARK_Task_Flag_Set(&tflags1, TASK_LOCK_TO_THREAD_MASK, (intptr_t) COMM_MASK);

    Quark_Task_Flags tflags2 = Quark_Task_Flags_Initializer;
    QUARK_Task_Flag_Set(&tflags2, TASK_COLOR, (intptr_t) "brown");
    QUARK_Task_Flag_Set(&tflags2, TASK_LABEL, (intptr_t) "POTRF");
    QUARK_Task_Flag_Set(&tflags2, TASK_LOCK_TO_THREAD_MASK, (intptr_t) COMP_MASK);

    Quark_Task_Flags tflags3 = Quark_Task_Flags_Initializer;
    QUARK_Task_Flag_Set(&tflags3, TASK_COLOR, (intptr_t) "gray");
    QUARK_Task_Flag_Set(&tflags3, TASK_LABEL, (intptr_t) "A2Y");
    QUARK_Task_Flag_Set(&tflags3, TASK_PRIORITY, 1);
    QUARK_Task_Flag_Set(&tflags3, TASK_LOCK_TO_THREAD_MASK, (intptr_t) COMM_MASK);

    Quark_Task_Flags tflags4 = Quark_Task_Flags_Initializer;
    QUARK_Task_Flag_Set(&tflags4, TASK_COLOR, (intptr_t) "lightblue");
    QUARK_Task_Flag_Set(&tflags4, TASK_LABEL, (intptr_t) "TRSM");
    QUARK_Task_Flag_Set(&tflags4, TASK_LOCK_TO_THREAD_MASK, (intptr_t) COMP_MASK);

    for(JB = 0, jb = 0; JB < N; JB+=NB, jb++){
        /* dpotrf diag */
        LN = MIN(JB+NB, N) - JB;
        {
            #ifdef USE_HOST_DPOTRF
                QUARK_D2H(quark, &tflags1, LN, LN, &Y(jb,jb), NB, &A(jb,jb), LDA);
                QUARK_incore_dpotrf(quark, &tflags2, "L", LN, &A(jb,jb), LDA);
                QUARK_H2D(quark, &tflags3, LN, LN, &A(jb,jb), LDA, &Y(jb,jb), NB);
            #else
                QUARK_incore_dpotrf(quark, &tflags2, "L", LN, &Y(jb,jb), NB);
                QUARK_D2H(quark, &tflags1, LN, LN, &Y(jb,jb), NB, &A(jb,jb), LDA);
            #endif
        }

        /* dtrsm column */
        {
            for(IB = JB+NB, ib = jb+1; IB < M; IB+=NB, ib++){
                LM = MIN(IB+NB, M) - IB;
                QUARK_incore_dtrsm(quark, &tflags4, "R", "L", "T", "N", // Right side Lower triangular Transpose Non unit triangular
                        LM, LN, 1.0, &Y(jb,jb), &Y(ib,jb), NB);
                QUARK_D2H(quark, &tflags1, LM, LN, &Y(ib,jb), NB, &A(ib,jb), LDA);
            }
        }

        /* Right-looking update */
        if(JB + NB < N){
            H = M-JB-NB;
            K = N-JB-NB;
            ooc_syrk(quark, &Y(jb+1,jb), &Y(jb+1,jb+1), H, K, NB);
        }
    }
    #undef Y
}


