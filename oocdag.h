//for io
#define NSIZE 32
// quark threads
#define OOC_NTHREADS 4

// verify
//#define VERIFY

#ifdef USE_MIC
    #include "ooc_offload.h"
#else 
    #ifdef USE_CUBLASV2
        #include <cuda_runtime.h>
        #include <cuda_runtime_api.h>
        #include <cublas_v2.h>
        #define cublasStatus cublasStatus_t
        extern cublasHandle_t worker_handle[OOC_NTHREADS];
    #else
        #include <cuda_runtime_api.h>
        #include <cublas.h>
        #include "cublasOps.h"
    #endif
#endif
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <assert.h>
#include <sched.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <mkl.h>
#include <mkl_blacs.h>
#include <quark.h>
#include <string.h>
//#include <lapacke.h>
//#include <common.h>
//#include <cblas.h>


#ifndef MAX
#define MAX(x,y)  ( ((x) > (y)) ? (x) : (y) )
#endif

#ifndef MIN
#define MIN(x,y)  ( ((x) < (y)) ? (x) : (y) )
#endif

#ifndef ABS
#define ABS(x)  ( ((x) > 0) ? (x) : (-(x)) )
#endif

#ifndef USE_MIC
    #ifndef PRINT_ERR
        #define PRINT_ERR(cu_status) { \
            switch(cu_status){ \
            case CUBLAS_STATUS_NOT_INITIALIZED: \
              { printf("CUBLAS_STATUS_NOT_INITIALIZED\n"); break; } \
            case CUBLAS_STATUS_ALLOC_FAILED: \
              { printf("CUBLAS_STATUS_ALLOC_FAILED\n"); break; } \
            case CUBLAS_STATUS_INVALID_VALUE: \
              { printf("CUBLAS_STATUS_INVALID_VALUE\n"); break; } \
            case CUBLAS_STATUS_MAPPING_ERROR: \
              { printf("CUBLAS_STATUS_MAPPING_ERROR\n"); break; } \
            case CUBLAS_STATUS_EXECUTION_FAILED: \
              { printf("CUBLAS_STATUS_EXECUTION_FAILED\n"); break; } \
            case CUBLAS_STATUS_INTERNAL_ERROR: \
              { printf("CUBLAS_STATUS_INTERNAL_ERROR\n"); break; } \
            default: \
              { printf("unknown error\n"); } \
            } \
        }
    #endif
    #ifndef CHKERR
        #define CHKERR(cu_status) { \
            if(cu_status != CUBLAS_STATUS_SUCCESS){ \
                PRINT_ERR(cu_status); \
            } \
            assert(cu_status == CUBLAS_STATUS_SUCCESS); \
        }
    #endif
#endif

extern "C"
void pdmatgen(int *ICTXT, char *AFORM, char *DIAG,
              int *M, int *N, int *MB, int *NB, double *A,
              int *LDA, int *IAROW, int *IACOL, int *ISEED,
              int *IROFF, int *IRNUM, int *ICOFF, int *ICNUM,
              int *MYROW, int *MYCOL, int *NPROW, int *NPCOL);

void CORE_H2D(Quark *quark);
void QUARK_H2D(Quark *quark, Quark_Task_Flags *task_flags, 
        int M, int N, double *H, int ldh, double *D, int NB);

void CORE_D2H(Quark *quark);
void QUARK_D2H(Quark *quark, Quark_Task_Flags *task_flags,
        int M, int N, double *D, int NB, double *H, int ldh);

void CORE_incore_dpotrf(Quark *quark);
void QUARK_incore_dpotrf( Quark *quark, Quark_Task_Flags *task_flags,
        const char *uplo, int N, double *A, int NB);

void CORE_incore_dtrsm(Quark *quark);
void QUARK_incore_dtrsm(Quark *quark, Quark_Task_Flags *task_flags,
                       const char *side, const char *uplo, const char *transA, const char *diag,
                       int M, int N, double alpha, const double *A, double *B, int NB);

void CORE_incore_dsyrk(Quark *quark);
void QUARK_incore_dsyrk(Quark *quark, Quark_Task_Flags *task_flags,
                      const char *uplo, const char *trans, int N, int K,
                      double alpha, const double *A, double beta, double *C, int NB);

void CORE_incore_dgemm(Quark *quark);
void QUARK_incore_dgemm(Quark *quark, Quark_Task_Flags *task_flags,
                      const char *transA, const char *transB, int M, int N, int K,
                      double alpha, const double *A, const double *B, double beta, double *C, int NB);

void A2Y(Quark *quark, double *A, double *Y, int LDA, int NB, int M, int N);
void Y2A(Quark *quark, double *Y, double *A, int LDA, int NB, int M, int N);
void A2X(Quark *quark, double *A, int LDA, double *X, int NB, int K);
void ooc_syrk(Quark *quark, double *X, double *Y, int H, int K, int NB);
void ooc_incore(Quark *quark, double *A, double *Y, int LDA, int NB, int M, int N);

#ifdef USE_MIC
void CORE_bind(Quark *quark);
void QUARK_bind(Quark *quark);
#endif







