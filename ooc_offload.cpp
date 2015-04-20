#ifdef USE_MIC
#include "ooc_offload.h"
#define NUMTHREAD 2                         // threads that call these functions
#define PADDINGSIZE 0                         // try to prevent false sharing of DBUFFER
#define BUFFERSIZE (4*1024*1024-PADDINGSIZE)    // per thread 4M*sizeof(double) = 32MByte
// beaware when DBUFFERin_ and DBUFFERout_ are too big to fit in 2GB BSS

static int MYRANK;
static int MYDEVICE;
__attribute__((target(mic),aligned(4096))) static double DBUFFERin_[(BUFFERSIZE+PADDINGSIZE)*NUMTHREAD];
__attribute__((target(mic),aligned(4096))) static double DBUFFERout_[(BUFFERSIZE+PADDINGSIZE)*NUMTHREAD];

void offload_init(int *myrank, int *mydevice){
    MYRANK = *myrank;
    MYDEVICE = _Offload_number_of_devices();
    if(MYDEVICE){
        MYDEVICE = MYRANK % _Offload_number_of_devices();
    }
    *mydevice = MYDEVICE;

    #pragma offload_transfer target(mic:MYDEVICE) nocopy(DBUFFERin_:alloc_if(1) free_if(0) align(2*1024*1024))
    #pragma offload_transfer target(mic:MYDEVICE) nocopy(DBUFFERout_:alloc_if(1) free_if(0) align(2*1024*1024))
}

extern "C" void offload_init_(int *myrank, int *mydevice){
    offload_init(myrank, mydevice);
}

void offload_destroy(){
    #pragma offload_transfer target(mic:MYDEVICE) nocopy(DBUFFERin_:alloc_if(0) free_if(1))
    #pragma offload_transfer target(mic:MYDEVICE) nocopy(DBUFFERout_:alloc_if(0) free_if(1))
}

extern "C" void offload_destroy_(){
    offload_destroy();    
}

intptr_t offload_Alloc(size_t size, int r){
    intptr_t ptr = 0;
    #pragma offload target(mic:MYDEVICE)
    {
        ptr = (intptr_t) memalign(2*1024*1024, size);
    }
//  printf("%zd %zd\n", ptr, ptr+size-1);
//  offload_touch((void*)ptr, size);
    return ptr;
}

void offload_touch(void* p, size_t size, int r){
    intptr_t ptr = (intptr_t) p;
    #pragma offload target(mic:MYDEVICE)
    {
        char* C = (char*) ptr;
        double* D;
        size_t i, iend;
        double B[8];
        iend = size % 8;
        for(i = 0; i < iend; i++){
            B[i] = C[i];
        }
        D = (double*) (C+i);
        iend = (size / 8) % 8;
        for(i = 0 ; i < iend; i++){
            B[i] = D[i];
        }
        iend = size / 8;
        for( ; i < iend; i = i + 8){
            B[0] = D[i];
            B[1] = D[i+1];
            B[2] = D[i+2];
            B[3] = D[i+3];
            B[4] = D[i+4];
            B[5] = D[i+5];
            B[6] = D[i+6];
            B[7] = D[i+7];
        }
    }
}

void offload_Free(void* p, int r){
    intptr_t ptr = (intptr_t)p;
    #pragma offload target(mic:MYDEVICE)
    {
        free((void*)ptr);
    }
}

void offload_dSetVector(int n, double *x, int incx, double *y, int incy, int r){
/*
 *  copy x at host to y at device
 *  incx is the index increment of x, incy is the index increment of y
 *  n elements are copied
 *  algorithm works for negative values of incx and incy, but gives undefined behavior
 */

//  assert(n >= 0);
    // copy x to DBUFFERR, offload transfer in to DBUFFERR, copy to y
#define DBUFFERR (DBUFFERin_+SLICE)
    size_t SLICE = r*(BUFFERSIZE+PADDINGSIZE);
    int incB = 1;
    size_t start = 0;
    size_t end = start + BUFFERSIZE - 1;
           end = MIN(end, n - 1);
    int length = MIN(end - start + 1, BUFFERSIZE);
    double *xstart;
    intptr_t yptr = (intptr_t)y;

//  printf("offload_dSetVector start\n");
    for(start = 0; start < n; start = end + 1){ 
        end = start + BUFFERSIZE - 1;
        end = MIN(end, n - 1);
    
        length = MIN(end - start + 1, BUFFERSIZE);
        xstart = x + start*incx;

        dcopy(&length, xstart, &incx, DBUFFERR, &incB);
        #pragma offload target(mic:MYDEVICE) in(DBUFFERin_[SLICE:length]:alloc_if(0) free_if(0)) \
                                             in(yptr,incy,length,incB)
        {
            double *ystart = ((double*)yptr) + start*incy;
            dcopy(&length, DBUFFERR, &incB, ystart, &incy);
        }
    }
#undef DBUFFERR
}

void offload_dGetVector(int n, double *x, int incx, double *y, int incy, int r){
/*
 *  copy x at device to y at host
 *  incx is the index increment of x, incy is the index increment of y
 *  n elements are copied
 *  algorithm works for negative values of incx and incy, but gives undefined behavior
 */
//  assert(n >= 0);
  
    // copy x to DBUFFERR, offload transfer out to DBUFFERR, copy to y
#define DBUFFERR (DBUFFERout_+SLICE)
    size_t SLICE = r*(BUFFERSIZE+PADDINGSIZE);
    int incB = 1;
    size_t start = 0;
    size_t end = start + BUFFERSIZE - 1;
           end = MIN(end, n - 1);
    int length = MIN(end - start + 1, BUFFERSIZE);
    double *ystart;
    intptr_t xptr = (intptr_t)x;
    
//  printf("offload_dGetVector start %d %p\n", r, DBUFFERR);
    for(start = 0; start < n; start = end + 1){ 
        end = start + BUFFERSIZE - 1;
        end = MIN(end, n - 1);
    
        length = MIN(end - start + 1, BUFFERSIZE);
        ystart = y + start*incy;
        /*
        #pragma offload target(mic:MYDEVICE) nocopy(DBUFFERout_[SLICE:length]:alloc_if(0) free_if(0))
                                         //  in(xptr,incx,length,incb) // no problem with this (optional) line
        {
            double *xstart = ((double*)xptr) + start*incx;
            dcopy(&length, xstart, &incx, DBUFFERR, &incB);
        }
        #pragma offload_transfer target(mic:MYDEVICE) out(DBUFFERR:length(length)) signal(r)
        #pragma offload_wait target(mic:MYDEVICE) wait(r)
        */
        // original
        #pragma offload target(mic:MYDEVICE) out(DBUFFERout_[SLICE:length]: alloc_if(0) free_if(0))  
                                         //  in(xptr,incx,length,incb) // no problem with this (optional) line
        {
            double *xstart = ((double*)xptr) + start*incx;
            dcopy(&length, xstart, &incx, DBUFFERR, &incB);
        }
        //
        dcopy(&length, DBUFFERR, &incB, ystart, &incy);
    }
//  printf("offload_dGetVector end %d\n", r);
#undef DBUFFERR
}

void offload_dSetMatrix(int rows, int cols, double *a, int lda, double *b, int ldb, int r){
/*
 * a is at host, b is at device
 */
#define da(i,j)  (((double*)aptr) + IDX2F(i,j,lda))
#define db(i,j)  (((double*)bptr) + IDX2F(i,j,ldb))
//  assert(rows >= 0); assert(cols >= 0); assert(ia >= 1); assert(ja >= 1); assert(lda >= 1);
//  assert(ib >= 1); assert(jb >= 1); assert(ldb >= 1);

    // copy a to DBUFFERR, offload transfer in to DBUFFERR, copy to y
#define DBUFFERR (DBUFFERin_+SLICE)
    size_t SLICE = r*(BUFFERSIZE+PADDINGSIZE);
    int inca = 1;
    int incB = 1;
    size_t start = 0;
    size_t end = rows;
           end = end*cols - 1;
           end = MIN(end, start + BUFFERSIZE - 1);
    int length = MIN(end - start + 1, BUFFERSIZE);
    int clength;
    int ia = 1; 
    int ja = 1;
    int ib = 1;
    int jb = 1;
    int ra, rb;
    int filled = 0;
    intptr_t aptr = (intptr_t)a;
    intptr_t bptr = (intptr_t)b;

//  printf("offload_dSetMatrix start\n");
//  printf("rows %d cols %d a %zd lda %d b %zd ldb %d\n", rows, cols, aptr, lda, bptr, ldb);
//  printf("offload_dSetMatrix skip\n"); return;
    for(start = 0; start < rows*cols; start = end + 1){ 
        end = start + BUFFERSIZE - 1;
        end = MIN(end, rows*cols - 1);
    
        length = MIN(end - start + 1, BUFFERSIZE);
        if(lda == rows){
            dcopy(&length, da(ia,ja), &inca, DBUFFERR, &incB);
            ra = length % rows;
            ia = ia + ra;
            if(ia > rows){
                ia = ia - rows;
                ja = ja + (length - ra)/rows + 1;
            }else{
                ja = ja + (length - ra)/rows;
            }
        }else{
            filled = 0;
            while(filled < length){
                if(length - filled < rows - ia + 1){
                    clength = length - filled;
                    dcopy(&clength, da(ia,ja), &inca, DBUFFERR+filled, &incB);
                    ia = ia + clength;
                }else{
                    clength = rows - ia + 1;
                    dcopy(&clength, da(ia,ja), &inca, DBUFFERR+filled, &incB);
                    ia = 1;
                    ja = ja + 1;
                }
                filled = filled + clength;
            }
        }
        #pragma offload target(mic:MYDEVICE) in(DBUFFERin_[SLICE:length]:alloc_if(0) free_if(0))
        {
            int incb = 1;
            if(ldb == rows){
                dcopy(&length, DBUFFERR, &incB, db(ib,jb), &incb);
                rb = length % rows;
                ib = ib + rb;
                if(ib > rows){
                    ib = ib - rows;
                    jb = jb + (length - rb)/rows + 1;
                }else{
                    jb = jb + (length - rb)/rows;
                }
            }else{
                filled = 0;
                while(filled < length){
                    if(length - filled < rows - ib + 1){
                        clength = length - filled;
                        dcopy(&clength, DBUFFERR+filled, &incB, db(ib,jb), &incb);
                        ib = ib + clength;
                    }else{
                        clength = rows - ib + 1;
                        dcopy(&clength, DBUFFERR+filled, &incB, db(ib,jb), &incb);
                        ib = 1;
                        jb = jb + 1;
                    }
                    filled = filled + clength;
                }
            }
        }
    }
//  printf("offload_dSetMatrix end\n");
#undef DBUFFERR
#undef da
#undef db
}

void offload_dGetMatrix(int rows, int cols, double *a, int lda, double *b, int ldb, int r){
/*
 * a is at device, b is at host
 */
#define da(i,j)  (((double*)aptr) + IDX2F(i,j,lda))
#define db(i,j)  (((double*)bptr) + IDX2F(i,j,ldb))
//  assert(rows >= 0); assert(cols >= 0); assert(ia >= 1); assert(ja >= 1); assert(lda >= 1);
//  assert(ib >= 1); assert(jb >= 1); assert(ldb >= 1);

    // copy a to DBUFFERR, offload transfer out to DBUFFERR, copy to y
#define DBUFFERR (DBUFFERout_+SLICE)
    size_t SLICE = r*(BUFFERSIZE+PADDINGSIZE);
    int incB = 1;
    int incb = 1;
    size_t start = 0;
    size_t end = rows;
           end = end*cols - 1;
           end = MIN(end, start + BUFFERSIZE - 1);
    int length = MIN(end - start + 1, BUFFERSIZE);
    int clength;
    int ia = 1; 
    int ja = 1;
    int ib = 1;
    int jb = 1;
    int ra, rb;
    int filled = 0;
    intptr_t aptr = (intptr_t)a;
    intptr_t bptr = (intptr_t)b;

//  printf("offload_dGetMatrix start\n");
    for(start = 0; start < rows*cols; start = end + 1){ 
        end = start + BUFFERSIZE - 1;
        end = MIN(end, rows*cols - 1);
    
        length = MIN(end - start + 1, BUFFERSIZE);
        #pragma offload target(mic:MYDEVICE) out(DBUFFERout_[SLICE:length]:alloc_if(0) free_if(0))
        {
            int inca = 1;
            if(lda == rows){
                dcopy(&length, da(ia,ja), &inca, DBUFFERR, &incB);
                ra = length % rows;
                ia = ia + ra;
                if(ia > rows){
                    ia = ia - rows;
                    ja = ja + (length - ra)/rows + 1;
                }else{
                    ja = ja + (length - ra)/rows;
                }
            }else{
                filled = 0;
                while(filled < length){
                    if(length - filled < rows - ia + 1){
                        clength = length - filled;
                        dcopy(&clength, da(ia,ja), &inca, DBUFFERR+filled, &incB);
                        ia = ia + clength;
                    }else{
                        clength = rows - ia + 1;
                        dcopy(&clength, da(ia,ja), &inca, DBUFFERR+filled, &incB);
                        ia = 1;
                        ja = ja + 1;
                    }
                    filled = filled + clength;
                }
            }
        }

        if(ldb == rows){
            dcopy(&length, DBUFFERR, &incB, db(ib,jb), &incb);
            rb = length % rows;
            ib = ib + rb;
            if(ib > rows){
                ib = ib - rows;
                jb = jb + (length - rb)/rows + 1;
            }else{
                jb = jb + (length - rb)/rows;
            }
        }else{
            filled = 0;
            while(filled < length){
                if(length - filled < rows - ib + 1){
                    clength = length - filled;
                    dcopy(&clength, DBUFFERR+filled, &incB, db(ib,jb), &incb);
                    ib = ib + clength;
                }else{
                    clength = rows - ib + 1;
                    dcopy(&clength, DBUFFERR+filled, &incB, db(ib,jb), &incb);
                    ib = 1;
                    jb = jb + 1;
                }
                filled = filled + clength;
            }
        }
    }
//  printf("offload_dGetMatrix end\n");
#undef DBUFFERR
#undef da
#undef db
}

void offload_dtrSetMatrix(char uplo, int rows, int cols, double *a, int lda, double *b, int ldb, int r){
/*
 * a is at host, b is at device, for trapezoidal matrices
 */
#define da(i,j)  (((double*)aptr) + IDX2F(i,j,lda))
#define db(i,j)  (((double*)bptr) + IDX2F(i,j,ldb))
//  assert(rows >= 0); assert(cols >= 0); assert(ia >= 1); assert(ja >= 1); assert(lda >= 1);
//  assert(ib >= 1); assert(jb >= 1); assert(ldb >= 1);

    // copy a to DBUFFERR, offload transfer in to DBUFFERR, copy to y
#define DBUFFERR (DBUFFERin_+SLICE)
    size_t SLICE = r*(BUFFERSIZE+PADDINGSIZE);
    int is_lower = (uplo == 'L')||(uplo == 'l');
    int inca, incb;
    int incB = 1;
    size_t start = 0;
    size_t end = rows;
           end = end*cols - 1;
           end = MIN(end, start + BUFFERSIZE - 1);
    int length = MIN(end - start + 1, BUFFERSIZE);
    int clength;
    size_t total;
    int ia = 1; 
    int ja = 1;
    int ib = 1;
    int jb = 1;
    int ra, rb;
    int filled = 0;
    intptr_t aptr = (intptr_t)a;
    intptr_t bptr = (intptr_t)b;
        
    if(is_lower){
        cols = MIN(rows, cols);
        inca = 1; incb = 1;
        total = 2*rows - cols;
        total = total*cols + cols;
        total = total/2; 
    }else{
        rows = MIN(rows, cols);
        inca = lda; incb = ldb;
        total = 2*cols - rows;
        total = total*rows + rows;
        total = total/2;
    }
    for(start = 0; start < total; start = end + 1){ 
        end = start + BUFFERSIZE - 1;
        end = MIN(end, total - 1);
    
        length = MIN(end - start + 1, BUFFERSIZE);
        filled = 0;
        if(is_lower){
            while(filled < length){
                if(length - filled < rows - ia + 1){
                    clength = length - filled;
                    dcopy(&clength, da(ia,ja), &inca, DBUFFERR+filled, &incB);
                    ia = ia + clength;
                }else{
                    clength = rows - ia + 1;
                    dcopy(&clength, da(ia,ja), &inca, DBUFFERR+filled, &incB);
                    ja = ja + 1;
                    ia = ja;
                }
                filled = filled + clength;
            }
        }else{
            while(filled < length){
                if(length - filled < cols - ja + 1){
                    clength = length - filled;
                    dcopy(&clength, da(ia,ja), &inca, DBUFFERR+filled, &incB);
                    ja = ja + clength;
                }else{
                    clength = cols - ja + 1;
                    dcopy(&clength, da(ia,ja), &inca, DBUFFERR+filled, &incB);
                    ia = ia + 1;
                    ja = ia;
                }
                filled = filled + clength;
            }
        }
        #pragma offload target(mic:MYDEVICE) in(DBUFFERin_[SLICE:length]:alloc_if(0) free_if(0))
        {
            filled = 0;
            if(is_lower){
                while(filled < length){
                    if(length - filled < rows - ib + 1){
                        clength = length - filled;
                        dcopy(&clength, DBUFFERR+filled, &incB, db(ib,jb), &incb);
                        ib = ib + clength;
                    }else{
                        clength = rows - ib + 1;
                        dcopy(&clength, DBUFFERR+filled, &incB, db(ib,jb), &incb);
                        jb = jb + 1;
                        ib = jb;
                    }
                    filled = filled + clength;
                }
                while(filled < length){
                    if(length - filled < cols - ib + 1){
                        clength = length - filled;
                        dcopy(&clength, DBUFFERR+filled, &incB, db(ib,jb), &incb);
                        jb = jb + clength;
                    }else{
                        clength = rows - ib + 1;
                        dcopy(&clength, DBUFFERR+filled, &incB, db(ib,jb), &incb);
                        ib = ib + 1;
                        jb = ib;
                    }
                    filled = filled + clength;
                }
            }
        }
    }
#undef DBUFFERR
#undef da
#undef db
}

void offload_dtrGetMatrix(char uplo, int rows, int cols, double *a, int lda, double *b, int ldb, int r){
#define da(i,j)  (((double*)aptr) + IDX2F(i,j,lda))
#define db(i,j)  (((double*)bptr) + IDX2F(i,j,ldb))
    // for trapezoidol matrices
    
    // copy a to DBUFFERR, offload transfer out to DBUFFERR, copy to y
#define DBUFFERR (DBUFFERout_+SLICE)
    size_t SLICE = r*(BUFFERSIZE+PADDINGSIZE);
    int is_lower = (uplo == 'L')||(uplo == 'l');
    int incB = 1;
    int inca, incb;
    size_t start = 0;
    size_t end = rows;
           end = end*cols - 1;
           end = MIN(end, start + BUFFERSIZE - 1);
    int length = MIN(end - start + 1, BUFFERSIZE);
    int clength;
    size_t total;
    int ia = 1; 
    int ja = 1;
    int ib = 1;
    int jb = 1;
    int filled = 0;
    intptr_t aptr = (intptr_t)a;
    intptr_t bptr = (intptr_t)b;

    if(is_lower){
        cols = MIN(rows, cols);
        inca = 1; incb = 1;
        total = 2*rows - cols;
        total = total*cols + cols;
        total = total/2; 
    }else{
        rows = MIN(rows, cols);
        inca = lda; incb = ldb;
        total = 2*cols - rows;
        total = total*rows + rows;
        total = total/2;
    }
    for(start = 0; start < total; start = end + 1){ 
        end = start + BUFFERSIZE - 1;
        end = MIN(end, total - 1);
    
        length = MIN(end - start + 1, BUFFERSIZE);
        #pragma offload target(mic:MYDEVICE) out(DBUFFERout_[SLICE:length]:alloc_if(0) free_if(0))
        {
            filled = 0;
            if(is_lower){
                while(filled < length){
                    if(length - filled < rows - ia + 1){
                        clength = length - filled;
                        dcopy(&clength, da(ia,ja), &inca, DBUFFERR+filled, &incB);
                        ia = ia + clength;
                    }else{
                        clength = rows - ia + 1;
                        dcopy(&clength, da(ia,ja), &inca, DBUFFERR+filled, &incB);
                        ja = ja + 1;
                        ia = ja;
                    }
                    filled = filled + clength;
                }
            }else{
                while(filled < length){
                    if(length - filled < cols - ja + 1){
                        clength = length - filled;
                        dcopy(&clength, da(ia,ja), &inca, DBUFFERR+filled, &incB);
                        ja = ja + clength;
                    }else{
                        clength = cols - ja + 1;
                        dcopy(&clength, da(ia,ja), &inca, DBUFFERR+filled, &incB);
                        ia = ia + 1;
                        ja = ia;
                    }
                    filled = filled + clength;
                }
            }
        }

        filled = 0;
        if(is_lower){
            while(filled < length){
                if(length - filled < rows - ib + 1){
                    clength = length - filled;
                    dcopy(&clength, DBUFFERR+filled, &incB, db(ib,jb), &incb);
                    ib = ib + clength;
                }else{
                    clength = rows - ib + 1;
                    dcopy(&clength, DBUFFERR+filled, &incB, db(ib,jb), &incb);
                    jb = jb + 1;
                    ib = jb;
                }
                filled = filled + clength;
            }
            while(filled < length){
                if(length - filled < cols - ib + 1){
                    clength = length - filled;
                    dcopy(&clength, DBUFFERR+filled, &incB, db(ib,jb), &incb);
                    jb = jb + clength;
                }else{
                    clength = rows - ib + 1;
                    dcopy(&clength, DBUFFERR+filled, &incB, db(ib,jb), &incb);
                    ib = ib + 1;
                    jb = ib;
                }
                filled = filled + clength;
            }
        }
    }
#undef DBUFFERR
#undef da
#undef db
}

void offload_dcopy(int n, const double *x, int incx, double *y, int incy, int r){
/*
 *  perform dcopy on the device. x,y pre-exist on the device
 */
    intptr_t xptr = (intptr_t)x;
    intptr_t yptr = (intptr_t)y;
//  printf("offload_dcopy start\n");
    #pragma offload target(mic:MYDEVICE)
    {
        dcopy(&n, (double*)xptr, &incx, (double*)yptr, &incy);
    }
//  printf("offload_dcopy end\n");
}

void offload_dgemm(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                   const double *alpha, const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
                   const double *beta, double *c, const MKL_INT *ldc, int r){
/*
 * perform dgemm on the device. a,b,c pre-exist on the device
 */
    intptr_t aptr = (intptr_t)a;
    intptr_t bptr = (intptr_t)b;
    intptr_t cptr = (intptr_t)c;
//  printf("offload_dgemm start\n");
    #pragma offload target(mic:MYDEVICE) in(transa,transb,m,n,k:length(1)) \
                                         in(alpha,lda,ldb,beta,ldc:length(1)) in(aptr,bptr,cptr)
    {
        dgemm(transa,transb,m,n,k,alpha,(double*)aptr,lda,(double*)bptr,ldb,beta,(double*)cptr,ldc); 
    }
//  printf("offload_dgemm end\n");
}

void offload_dsyrk(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
                   const double *alpha, const double *a, const MKL_INT *lda, const double *beta,
                   double *c, const MKL_INT *ldc, int r){
/*
 * perform dsyrk on the device. a,c pre-exist on the device
 */
    intptr_t aptr = (intptr_t)a;
    intptr_t cptr = (intptr_t)c;
//  printf("offload_dsyrk start\n");
    #pragma offload target(mic:MYDEVICE) in(uplo,trans,n,k:length(1)) \
                                         in(alpha,lda,beta,ldc:length(1)) in(aptr,cptr)
    {
        dsyrk(uplo,trans,n,k,alpha,(double*)aptr,lda,beta,(double*)cptr,ldc);
    }
//  printf("offload_dsyrk end\n");
}

void offload_dtrsm(const char *side, const char *uplo, const char *transa, const char *diag,
                   const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
                   double *b, const MKL_INT *ldb, int r){
/*
 * perform dtrsm on the device. a,b pre-exist on the device
 */
    intptr_t aptr = (intptr_t)a;
    intptr_t bptr = (intptr_t)b;
    #pragma offload target(mic:MYDEVICE) in(side,uplo,transa,diag,m,n,alpha,lda,ldb:length(1)) 
    {
//      printf("dtrsmbefore %lf %p\n", ((double*)bptr)[0], (double*)bptr);
        dtrsm(side,uplo,transa,diag,m,n,alpha,(double*)aptr,lda,(double*)bptr,ldb);
//      printf("dtrsmafter  %lf %p\n", ((double*)bptr)[0], (double*)bptr);
    }
}

void offload_dpotrf( const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, 
                     MKL_INT* info, int r){
/*
 * perform dpotrf on the device. a pre-exists on the device
 */
    intptr_t aptr = (intptr_t)a;
//  printf("potrf start\n");
    #pragma offload target(mic:MYDEVICE) in(uplo,n,lda:length(1)) out(info:length(1))
    {
        dpotrf(uplo,n,(double*)aptr,lda,info);
    }
//  printf("potrf end\n");
}
#endif
