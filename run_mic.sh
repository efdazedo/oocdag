#!/bin/bash
#PBS -N LLT
#PBS -A UT-BEACON-KWONG
#PBS -j oe
#PBS -l walltime=02:00:00,nodes=1
#PBS -m abe
#PBS -M ss.sing@hotmail.com

cd /nics/b/home/csure09/oocchol/oocdaggpu/

pwd
#free
#module swap intel-compilers intel-compilers/2015.0.090
# module list

#performance tuning on MIC
export MIC_ENV_PREFIX=MIC
export MIC_USE_2MB_BUFFERS=2M
#export MIC_KMP_AFFINITY="explicit,verbose,granularity=fine,proclist=[{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20},{21,22,23,24},{25,26,27,28},{29,30,31,32},{33,34,35,36},{37,38,39,40},{41,42,43,44},{45,46,47,48},{49,50,51,52},{53,54,55,56},{57,58,59,60},{61,62,63,64},{65,66,67,68},{69,70,71,72},{73,74,75,76},{77,78,79,80},{81,82,83,84},{85,86,87,88},{89,90,91,92},{93,94,95,96},{97,98,99,100},{101,102,103,104},{105,106,107,108},{109,110,111,112},{113,114,115,116},{117,118,119,120},{121,122,123,124},{125,126,127,128},{129,130,131,132},{133,134,135,136},{137,138,139,140},{141,142,143,144},{145,146,147,148},{149,150,151,152},{153,154,155,156},{157,158,159,160},{161,162,163,164},{165,166,167,168},{169,170,171,172},{173,174,175,176},{177,178,179,180},{181,182,183,184},{185,186,187,188},{189,190,191,192},{193,194,195,196},{197,198,199,200},{201,202,203,204},{205,206,207,208},{209,210,211,212},{213,214,215,216},{217,218,219,220},{221,222,223,224},{225,226,227,228},{229,230,231,232},{233,234,235,236}]"
#export MIC_KMP_AFFINITY=explicit,verbose,granularity=fine,proclist=[1-236:1] 
#export MIC_MKL_NUM_THREADS=16
export KMP_AFFINITY=granularity=fine,compact,1,0
export MKL_NUM_THREADS=32

#initialize only the MIC to be used
export OFFLOAD_INIT=on_offload
#MIC list
export OFFLOAD_DEVICES=0
export OFFLOAD_REPORT=

#record these setting to output
echo 
echo PBS_NUM_NODES=$PBS_NUM_NODES
echo PBS_NUM_PPN=$PBS_NUM_PPN

echo MIC_ENV_PREFIX=$MIC_ENV_PREFIX
echo MIC_USE_2MB_BUFFERS=$MIC_USE_2MB_BUFFERS
echo MIC_MKL_NUM_THREADS=$MIC_MKL_NUM_THREADS
echo KMP_AFFINITY=$KMP_AFFINITY
echo MKL_NUM_THREADS=$MKL_NUM_THREADS


echo OFFLOAD_INIT=$OFFLOAD_INIT
echo OFFLOAD_DEVICES=$OFFLOAD_DEVICES


#export OMP_NUM_THREADS=4
#export MIC_OMP_NUM_THREADS=240
#export KMP_OMP_NUM_THREADS=240

export QUARK_UNROLL_TASKS=200
#export QUARK_DOT_DAG_ENABLE=1

echo N $N NB $NB memsize $memsize
micmpiexec -n 1 ./oocdag $N $NB $memsize
echo N $1 NB $2 memsize $3
micmpiexec -n 1 ./oocdag $1 $2 $3


