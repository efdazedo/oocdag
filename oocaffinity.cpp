#ifdef USE_MIC
#include "oocdag.h"
__attribute__((target(mic))) static intptr_t cpuMaskPtr;
__attribute__((target(mic))) static intptr_t statusPtr;


void CORE_bind(Quark *q){
    int r = QUARK_Thread_Rank(q);
    #pragma offload target(mic:0) 
    {
//      int map1[] = {0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208};
//      int map1[] = {0, 4, 8, 12, 28, 44, 60, 76, 92, 108, 124, 140, 156, 172, 188, 204, 220};
//      int map1[] = {0, 4, 8, 12, 44, 76, 108, 140, 172, 204};
        int map1[] = {0, 8, 16, 48, 80, 112, 144, 176, 208};
//      int map1[] = {0, 8, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224};
//      int map2[] = {16, 16, 16, 16, 16, 16, 16, 16,  16,  16,  16,  16,  16,  16};
//      int map2[] = {4, 4, 4, 16, 16, 16, 16, 16, 16, 16, 16,  16,  16,  16,  16,  16,  16};
//      int map2[] = {4, 4, 4, 32, 32, 32, 32, 32, 32, 32};
        int map2[] = {8, 8, 32, 32, 32, 32, 32, 32, 32};
//      int map2[] = {8, 8, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16};
        sched_setaffinity(0, sizeof(cpu_set_t), ((cpu_set_t*)cpuMaskPtr) + map1[r]);
        omp_set_num_threads(map2[r]);
        mkl_set_num_threads_local(map2[r]);
        #pragma omp parallel default (shared)
        {
            int i = omp_get_thread_num();
            if (sched_setaffinity(0, sizeof(cpu_set_t), ((cpu_set_t*)cpuMaskPtr) + map1[r] + i) == -1) {
                ((int*)statusPtr)[map1[r]+i] = -1;
//              printf("Error: sched_setaffinity %d %d\n", r, i);
            }else{
                ((int*)statusPtr)[map1[r]+i] = 0;
//              printf("Okay:  sched_setaffinity %d %d\n", r, i);
            }
        }
    }
}

void QUARK_bind(Quark *q){
    int statuscheck = 0;
    #pragma offload target(mic:0)
    {
        cpu_set_t* cpuMask = (cpu_set_t*) malloc(240*sizeof(cpu_set_t));
        int* status = (int*) malloc(240*sizeof(int));
        for(int i = 0; i < 240; i++){
                CPU_ZERO(cpuMask + i);
                CPU_SET(i + 1, cpuMask + i);
                status[i] = 1;
        }
        cpuMaskPtr = (intptr_t) cpuMask;
        statusPtr = (intptr_t) status;
    }
    {
        Quark_Task_Flags tflags = Quark_Task_Flags_Initializer;
        for(int r = 0; r < OOC_NTHREADS; r++){
            QUARK_Task_Flag_Set(&tflags, TASK_LOCK_TO_THREAD, r);
            QUARK_Insert_Task(q, CORE_bind, &tflags,
                0);
            QUARK_Barrier(q);
        }
    }
    #pragma offload target(mic:0)
    {
        int* status = (int*) statusPtr;
        for(int i = 0; i < 240; i++){
            if(status[i] != 0){
                printf("Error: sched_setaffinity %d %d\n", i, status[i]);
                statuscheck++;
            }
        }
        free((cpu_set_t*)cpuMaskPtr);
        free(status);
    }
//  printf("%d\n", statuscheck);
    assert(statuscheck == 0);
}
#endif
