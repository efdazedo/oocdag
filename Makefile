LIBS = -lquark -llapacke -lm -lpthread -lcblas -llapack -lgfortran -lrefblas 
LPATH = -L/nics/c/home/tianchong/oocchol/lib
SRCS =
OBJS =
CC=mpicc
CFLAGS = -O2 -us -g \
	-DUSE_CUBLASV2 \
	-DUSE_HOST_DPOTRF
#-DUSE_MIC

FC = mpiifort
FFLAGS = -O2 -us -g \
	-DUSE_CUBLASV2 \
	-DUSE_HOST_DPOTRF
#-DUSE_MIC

MKLROOT=/global/opt/intel/composer_xe_2015.0.090/mkl

IPATH = -I/nics/c/home/tianchong/oocchol/include \
	-I$(MKLROOT)/include \
	-I$(MAGMA_DIR)/include \
	-I$(CUDA_HOME)/include

All: oocdag

include Make.lindep
liblin.a: $(LINSRC)
	$(FC) $(FFLAGS) -c $(LINSRC)
	ar r liblin.a $(LINOBJECTS); ranlib liblin.a

oocdag: oocdag.o oocquark.o oocaffinity.o ooc_offload.o liblin.a
	mpiicc $(CFLAGS) -o oocdag \
	oocdag.o \
	oocquark.o \
	oocaffinity.o \
	ooc_offload.o \
	liblin.a \
	-lifcore \
	-openmp \
	-mkl \
	-offload-option,mic,compiler,"-z defs" \
	-L$(CUDA_HOME)/lib64 -lcudart -lcublas \
	-L$(MPICH_HOME)/lib -lmpi_mt \
	-L$(MKLROOT)/lib/intel64 \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core \
	-lmkl_blacs_intelmpi_lp64 \
	-lmkl_scalapack_lp64 \
	-lmkl_intel_lp64 -lmkl_blas95_lp64 \
	-lpthread -lm \
	$(LPATH) $(LIBS)

oocdag.o: oocdag.cpp
	mpiicc $(CFLAGS) $(IPATH) \
	-lstdc++ \
	-openmp \
	-Wno-deprecated -c oocdag.cpp

oocquark.o: oocquark.cpp
	mpiicc $(CFLAGS) $(IPATH) \
	-lstdc++ \
	-openmp \
	-Wno-deprecated -c oocquark.cpp

oocaffinity.o: oocaffinity.cpp
	mpiicc $(CFLAGS) $(IPATH) \
	-lstdc++ \
	-openmp \
	-Wno-deprecated -c oocaffinity.cpp

ooc_offload.o: ooc_offload.cpp
	mpiicc $(CFLAGS) $(IPATH) \
	-mkl -offload-option,mic,compiler,"-z defs" \
	-lstdc++ -Wno-deprecated \
	-openmp \
	-c ooc_offload.cpp

clean:
	rm -f oocdag
	rm -f oocdag.o
	rm -f oocquark.o
	rm -f oocaffinity.o
	rm -f ooc_offload.o
	rm -f liblin.a
 
