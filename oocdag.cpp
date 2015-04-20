#include "oocdag.h"

#ifndef USE_MIC
    #ifdef USE_CUBLASV2
        cublasHandle_t worker_handle[OOC_NTHREADS];
    #endif
#endif


void matprint(double *A, int N, char AL);
void Test_dpotrf(double *A, int N);
int find_Yn(int bb, int memBlock, int jb);

/*--------------------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------------------*/
             
double Cholesky(Quark *quark, double *A, int N, int NB, int LDA, size_t memsize) 
{
    #define A(ib,jb) A[(size_t)(jb)*NB*LDA+(ib)*NB]

    #ifndef USE_MIC
    cublasStatus cu_status;
    #endif
         
    int bb = (N + NB - 1) / NB;
    int YM, YN;
    int Ym, Yn;
    int JB;
    int jb, jjb;
    int memBlock = memsize/sizeof(double)/NB/NB;
    double *X, *Y;

    #ifdef USE_MIC
        Y = (double*) offload_Alloc((size_t)memBlock*NB*NB*sizeof(double), 0);
        assert(Y != NULL);
    #else
        #ifdef USE_CUBLASV2
        {
            cudaError_t ierr;
            ierr = cudaMalloc((void **) &Y, (size_t) memBlock*NB*NB*sizeof(double));
            assert(ierr == cudaSuccess);
        }
        #else
            cu_status = cublasAlloc((size_t) memBlock*NB*NB, sizeof(double), (void **) &Y);
            CHKERR(cu_status);
        #endif
    #endif
    
    double t1;
    double llttime = MPI_Wtime();

    
    /*--------------------------------------*/   

    /*     The main Ypanel loop     */

//  QUARK_Barrier(quark);
    for (JB = 0, jb = 0; JB < N; JB+=YN, jb+=Yn)
    {
        //determine size of Ypanel
        Ym = bb - jb;
        Yn = find_Yn(bb, memBlock, jb);
        YM = N - JB;
        YN = MIN((jb+Yn)*NB, N) - jb*NB;
        X = Y + (size_t)(memBlock-Ym)*NB*NB;
        printf("bb %d jb %d YM %d YN %d Ym %d Yn %d Y %p X %p\n", bb, jb, YM, YN, Ym, Yn, Y, X);

        /* Copy in data */
        A2Y(quark, &A(jb,jb), Y, LDA, NB, YM, YN);

        /* Left-looking */
        for(jjb = 0; jjb < jb; jjb++){
            /* copy from A to X */
            A2X(quark, &A(jb,jjb), LDA, X, NB, YM);
            ooc_syrk(quark, X, Y, YM, YN, NB);
        }

        /* incore factorization */
        ooc_incore(quark, &A(jb,jb), Y, LDA, NB, YM, YN);
    
        /* Copy out data */
//      Y2A(quark, Y, &A(jb,jb), LDA, NB, YM, YN);
//      QUARK_Barrier(quark); // reduce parallelism
//      goto oasdfh; // early stop

    }
oasdfh:
    QUARK_Barrier(quark);
    llttime = MPI_Wtime() - llttime;
    printf("llt time %lf %lf\n", llttime, MPI_Wtime());
    printf("%lf %lf\n", A[(N-1)*LDA+N-1], MPI_Wtime());
    /*--------------------------------------*/   

    #ifdef USE_MIC
        offload_Free(Y,0);
    #else
        #ifdef USE_CUBLASV2
        {
            cudaError_t ierr;
            ierr = cudaFree((void *) Y);
            assert(ierr == cudaSuccess);
            Y = 0;
        }
        #else
            cu_status = cublasFree(Y);
            CHKERR(cu_status);
        #endif
    #endif
    return llttime;
    #undef A
} 

#ifdef USE_MIC
void mklmem(int NB){
    double *H = (double*) malloc(NB*NB*sizeof(double));
    double *D = (double*) offload_Alloc(3*NB*NB*sizeof(double), 0);
    double *E = D + NB*NB;
    double *F = D + 2*NB*NB;
    int info;
    double alpha = 1.0, beta = 1.0;
    size_t memsize;
    {
        #pragma offload target(mic:0)
        {
            memsize = mkl_peak_mem_usage(MKL_PEAK_MEM);
        }
        printf("mkl_peak_mem_usage %zd\n", memsize);
        offload_dSetMatrix(NB, NB, H, NB, D, NB, 0);
        offload_dGetMatrix(NB, NB, D, NB, H, NB, 0);
        offload_dpotrf("L", &NB, D, &NB, &info, 0);
        offload_dtrsm("R", "L", "T", "N",
                    &NB, &NB, &alpha, D, &NB, E, &NB, 0);
        offload_dsyrk("L", "N", &NB, &NB,
                    &alpha, D, &NB, &beta, E, &NB, 0);
        offload_dgemm("N", "T", &NB, &NB, &NB,
                    &alpha, D, &NB, E, &NB, &beta, F, &NB, 0);
    }
    offload_Free(D, 0);
    offload_Free(E, 0);
    offload_Free(F, 0);
    free(H);
}
#endif
#ifdef USE_MIC
void warmup(Quark *q){
    int NB = 200;
    double *H = (double*) malloc(NB*NB*OOC_NTHREADS*sizeof(double));
    double *D = (double*) offload_Alloc(NB*NB*OOC_NTHREADS*sizeof(double), 0);
    
    {
        Quark_Task_Flags tflags = Quark_Task_Flags_Initializer;
//      for(int r = 0; r < OOC_NTHREADS; r++){
        for(int r = 0; r < 2; r++){
            QUARK_Task_Flag_Set(&tflags, TASK_LOCK_TO_THREAD, r);
//          QUARK_Task_Flag_Set(&tflags, THREAD_SET_TO_MANUAL_SCHEDULING, (r==0)||(r==1));
            QUARK_Insert_Task(q, CORE_H2D, &tflags,
                sizeof(int),                &NB,        VALUE,
                sizeof(int),                &NB,        VALUE,
                sizeof(double),             H+r*NB*NB,  INPUT,
                sizeof(int),                &NB,        VALUE,
                sizeof(double),             D+r*NB*NB,  OUTPUT,
                sizeof(int),                &NB,        VALUE,
                0);
            QUARK_Insert_Task(q, CORE_D2H, &tflags,
                sizeof(int),                &NB,        VALUE,
                sizeof(int),                &NB,        VALUE,
                sizeof(double),             D+r*NB*NB,  INPUT,
                sizeof(int),                &NB,        VALUE,
                sizeof(double),             H+r*NB*NB,  OUTPUT,
                sizeof(int),                &NB,        VALUE,
                0);
        }
    }
    QUARK_Barrier(q);
    offload_Free(D, 0);
    free(H);
}
#endif
/*---------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------*/

/*A function printing the lower triangular part or the whole of an N*N square matrix stored as a 1D array*/
void matprint(double *A, int N, int LDA, char AL)
{
    #define A(i,j) A[(j)*LDA+(i)]
    int i,j;
    if (AL=='L'){
        for(i=0;i<N;i++){
            for(j=0;j<=i;j++){
                printf("%lf ", A(i,j));               
            }
            for(j=i+1;j<N;j++){
                printf("%lf ", 0.0);               
            }
            printf("\n");
        }
    }else if(AL=='A'){
        for(i=0;i<N;i++){
            for(j=0;j<N;j++){
                printf("%lf ",A(i,j));
            }
            printf("\n");
        }
    }else{
        printf("Invalid A or L!\n");
    }
    #undef A
}

void Test_dpotrf(double *A,int N)
{
/*   Quark *quark=QUARK_New(OOC_NTHREADS);
     Quark_Task_Flags tflags=Quark_Task_Flags_Initializer;
     QUARK_incore_dpotrf(quark,&tflags,(int)'L',A,N);
     QUARK_Delete(quark); */
     int info;
     dpotrf_("L", &N, A, &N, &info); //computation on host
     assert(info == 0);
} 

int main(int argc, char **argv)
{
    #define test_A(i,j) test_A[(size_t)(j)*N+(i)]
    #define test_A2(i,j) test_A2[(size_t)(j)*N+(i)]
    int N,NB,w,LDA,BB;
    size_t memsize; //bytes
    int iam, nprocs, mydevice;
    int ICTXT, nprow, npcol, myprow, mypcol;
    int i_one = 1, i_zero = 0, i_negone = -1;
    double d_one = 1.0, d_zero = 0.0, d_negone = -1.0;
    int IASEED = 100;
/*  printf("N=?\n");
    scanf("%ld",&N);
    printf("NB=?\n");
    scanf("%d", &NB);
    printf("width of Y panel=?\n");
    scanf("%ld",&w);
*/
    if(argc < 4){
        printf("invalid arguments N NB memsize(M)\n");
        exit(1);
    }
    N = atoi(argv[1]);
    NB = atoi(argv[2]);
    memsize = (size_t)atoi(argv[3])*1024*1024;
    BB = (N + NB - 1) / NB;
    w = memsize/sizeof(double)/BB/NB/NB - 1;
    assert(w > 0);
    LDA = N + 0; //padding

    int do_io = (N <= NSIZE);
    double llttime;
    double gflops;
    
    nprow = npcol = 1;
    blacs_pinfo_(&iam, &nprocs);
    blacs_get_(&i_negone, &i_zero, &ICTXT);
    blacs_gridinit_(&ICTXT, "R", &nprow, &npcol);
    blacs_gridinfo_(&ICTXT, &nprow, &npcol, &myprow, &mypcol);
    #ifdef USE_MIC
        #ifdef __INTEL_OFFLOAD
            printf("offload compilation enabled\ninitialize each MIC\n");
            offload_init(&iam, &mydevice);
            #pragma offload target(mic:0)
            {
                mkl_peak_mem_usage(MKL_PEAK_MEM_ENABLE);
            }
        #else
            if(isroot)
                printf("offload compilation not enabled\n");
            exit(0);
        #endif
    #else
        #ifdef USE_CUBLASV2
        {
            cublasStatus_t cuStatus;
            for(int r = 0; r < OOC_NTHREADS; r++){
                cuStatus = cublasCreate(&worker_handle[r]);
                assert(cuStatus == CUBLAS_STATUS_SUCCESS);
            }
        }
        #else
            cublasInit();
        #endif
    #endif

    double *test_A = (double*)memalign(64,(size_t)LDA*N*sizeof(double)); // for chol
#ifdef VERIFY
    double *test_A2 = (double*)memalign(64,(size_t)LDA*N*sizeof(double)); // for verify
#endif
    
    /*Initialize A */
    int i,j;
    printf("Initialize A ... "); fflush(stdout);
    llttime = MPI_Wtime();
    pdmatgen(&ICTXT, "Symm", "Diag", &N,
         &N, &NB, &NB,
         test_A, &LDA, &i_zero, &i_zero,
         &IASEED, &i_zero, &N, &i_zero, &N,
         &myprow, &mypcol, &nprow, &npcol); 
    llttime = MPI_Wtime() - llttime;
    printf("time %lf\n", llttime);
              
    /*print test_A*/
    if(do_io){
        printf("Original A=\n\n");
        matprint(test_A, N, LDA, 'A');
    }

    /*Use directed unblocked Cholesky factorization*/    
    /*
    t1 = clock();
    Test_dpotrf(test_A2,N);
    t2 = clock();
    printf ("time for unblocked Cholesky factorization on host %f \n",
        ((float) (t2 - t1)) / CLOCKS_PER_SEC);
    */
    
    /*print test_A*/
    /*
    if(do_io){
        printf("Unblocked result:\n\n");
        matprint(test_A2,N,'L');   
    }
    */ 

    /*Use tile algorithm*/
    Quark *quark = QUARK_New(OOC_NTHREADS);
    QUARK_DOT_DAG_Enable(quark, 0);
    #ifdef USE_MIC
//      mklmem(NB);
        printf("QUARK MIC affinity binding\n");
        QUARK_bind(quark);
        printf("offload warm up\n");
        warmup(quark);
    #endif
    QUARK_DOT_DAG_Enable(quark, quark_getenv_int("QUARK_DOT_DAG_ENABLE", 0));
    printf("LLT start %lf\n", MPI_Wtime());
    llttime = Cholesky(quark,test_A,N,NB,LDA,memsize);
    printf("LLT end %lf\n", MPI_Wtime());
    QUARK_Delete(quark);
    #ifdef USE_MIC
        offload_destroy();
    #else
        #ifdef USE_CUBLASV2
        {
            cublasStatus_t cuStatus;
            for(int r = 0; r < OOC_NTHREADS; r++){ 
                cuStatus = cublasDestroy(worker_handle[r]);
                assert(cuStatus == CUBLAS_STATUS_SUCCESS);
            }
        }
        #else
            cublasShutdown();
        #endif
    #endif

    gflops = (double) N;
    gflops = gflops/3.0 + 0.5;
    gflops = gflops*(double)(N)*(double)(N);
    gflops = gflops/llttime/1024.0/1024.0/1024.0;
    printf ("N NB memsize(MB) quark_pthreads time Gflops\n%d %d %lf %d %lf %lf\n",
        N, NB, (double)memsize/1024/1024, OOC_NTHREADS, llttime, gflops);
    #ifdef USE_MIC
        #pragma offload target(mic:0)
        {
            memsize = mkl_peak_mem_usage(MKL_PEAK_MEM_RESET);
        }
        printf("mkl_peak_mem_usage %lf MB\n", (double)memsize/1024.0/1024.0);
    #endif

    /*Update and print L*/             
    if(do_io){
        printf("L:\n\n");
        matprint(test_A,N,LDA,'L');
    }
#ifdef VERIFY
    printf("Verify... ");
    llttime = MPI_Wtime();
  /*
   * ------------------------
   * check difference betwen 
   * test_A and test_A2
   * ------------------------
   */
    /*
    {
    double maxerr = 0;
    double maxerr2 = 0;

    for (j = 0; j < N; j++)
      {
        for (i = j; i < N; i++)
          {
            double err = (test_A (i, j) - test_A2 (i, j));
            err = ABS (err);
            maxerr = MAX (err, maxerr);
            maxerr2 = maxerr2 + err * err;
          };
      };
    maxerr2 = sqrt (ABS (maxerr2));
    printf ("max difference between test_A and test_A2 %lf \n", maxerr);
    printf ("L2 difference between test_A and test_A2 %lf \n", maxerr2);
    };
    */

  /*
   * ------------------
   * over-write test_A2
   * ------------------
   */
   
    pdmatgen(&ICTXT, "Symm", "Diag", &N,
         &N, &NB, &NB,
         test_A2, &LDA, &i_zero,
         &i_zero, &IASEED, &i_zero, &N, &i_zero, &N,
         &myprow, &mypcol, &nprow, &npcol);

  /*
   * ---------------------------------------
   * after solve, test_A2 should be identity
   * ---------------------------------------
   */
  // test_A = chol(B) = L;
  // test_A2 = B
  // solve L*L'*X = B
  // if L is correct, X is identity */
     
    {
    int uplo = 'L';
    const char *uplo_char = ((uplo == (int) 'U')
                    || (uplo == (int) 'u')) ? "U" : "L";
    int info = 0;
    int nrhs = N;
    int LDA = N;
    int ldb = N;
    dpotrs(uplo_char, &N, &nrhs, test_A, &LDA, test_A2, &ldb, &info);
    assert (info == 0);
    }

    {
    double maxerr = 0;
    double maxerr2 = 0;

    for (j = 0; j < N; j++)
      {
        for (i = 0; i < N; i++)
          {
            double eyeij = (i == j) ? 1.0 : 0.0;
            double err = (test_A2 (i, j) - eyeij);
            err = ABS (err);
            maxerr = MAX (maxerr, err);
            maxerr2 = maxerr2 + err * err;
          };
      };

    maxerr2 = sqrt (ABS (maxerr2));
    printf("time %lf\n", MPI_Wtime() - llttime);
    printf ("max error %lf \n", maxerr);
    printf ("max L2 error %lf \n", maxerr2);
    }
#endif

    free(test_A);test_A=NULL;
#ifdef VERIFY
    free(test_A2);test_A2=NULL;
#endif
    blacs_gridexit_(&ICTXT);
    blacs_exit_(&i_zero);
    return 0;
    #undef test_A
    #undef test_A2
}

/* decide the # of block-columns in A to be sent into Y-panel */
int find_Yn(int bb, int memBlock, int jb)
{
    static int Yn = 0; //Yn can only grow, except for the last panel
    int Ym = bb - jb;
//  printf("bb %d memBlock %d jb %d Yn %d Ym %d\n", bb, memBlock, jb, Yn, Ym);
    if(Ym*(Ym+1) <= (memBlock-Ym)*2){
        Yn = Ym;
    }else{
        while((memBlock-Ym)*2 >= (2*Ym-Yn)*(Yn+1)){
            if(Yn == Ym) break;
            Yn++;
        }
    }
    return Yn;
}

