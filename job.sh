#!/bin/bash

#qsub -v N=20000,NB=3000,memsize=5120 ./run_mic.sh
./run_gpu.sh 120000 2500 5120
