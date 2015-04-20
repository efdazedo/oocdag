#!/bin/bash
#PBS -N LLT
#PBS -A UT-BEACON-KWONG
#PBS -j oe
#PBS -l walltime=01:00:00,nodes=1:nvidia:gpus=4
#PBS -m abe
#PBS -M ss.sing@hotmail.com

cd /nics/b/home/csure09/oocchol/oocdaggpu/

pwd
#free
#module swap intel-compilers intel-compilers/2015.0.090
#module load cuda
# module list

export KMP_AFFINITY=granularity=fine,compact,1,0
export MKL_NUM_THREADS=32


#record these setting to output
echo 
echo PBS_NUM_NODES=$PBS_NUM_NODES
echo PBS_NUM_PPN=$PBS_NUM_PPN

 
export CUDA_PROFILE=1



export QUARK_UNROLL_TASKS=200
#export QUARK_DOT_DAG_ENABLE=1

echo N $N NB $NB memsize $memsize
micmpiexec -n 1 ./oocdag $N $NB $memsize
echo N $1 NB $2 memsize $3
micmpiexec -n 1 ./oocdag $1 $2 $3


