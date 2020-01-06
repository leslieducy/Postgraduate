#include <stdio.h>
#include <omp.h>

int main(){
    int coresNum = omp_get_num_procs();
    printf("core num %d \n",coresNum);

    #pragma omp parallel for
    for(int j = 0; j < coresNum; j++){
        printf("j=[%d], ThreadId =[%d]\n", j, omp_get_thread_num());
    }
}
