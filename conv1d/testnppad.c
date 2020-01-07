#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
void nppad1(float ret_z[], float z[], int N, int C, int H, int W, int padding[], float constant_values){
    int pad_H = (H+padding[0]*2);
    int pad_W = (W+padding[1]*2);
    // 四个循环写成一个
    int x_top = (int)N*C*pad_H*pad_W;
    // # pragma omp parallel for
    for(int i=0; i < N; i++){
        for(int j=0; j < C; j++){
            for(int k=0; k < pad_H; k++){
                for(int t=0; t < pad_W; t++){
                    if(k < padding[0] || k > (H+padding[0]-1) || t < padding[1] || t > (H+padding[1]-1)){
                        ret_z[i*C*pad_H*pad_W + j*pad_H*pad_W + k*pad_W + t] = constant_values;
                    }else{
                        ret_z[i*C*pad_H*pad_W + j*pad_H*pad_W + k*pad_W + t] = z[i*C*H*W + j*H*W + (k-padding[0])*W + (t-padding[1])];
                    }
                }
            }
        }
    }
}
void nppad2(float ret_z[], float z[], int N, int C, int H, int W, int padding[], float constant_values){
    int pad_H = (H+padding[0]*2);
    int pad_W = (W+padding[1]*2);
    // 四个循环写成一个
    int x_top = (int)N*C*pad_H*pad_W;
    // # pragma omp parallel for
    for(int x=0; x < x_top; x++){
        int i = x / (C*pad_H*pad_W);
        int j = (x % (C*pad_H*pad_W)) / (pad_H*pad_W);
        int k = ((x % (C*pad_H*pad_W)) % (pad_H*pad_W)) / pad_W;
        int t = (((x % (C*pad_H*pad_W)) % (pad_H*pad_W)) % pad_W) / 1;

        if(k < padding[0] || k > (H+padding[0]-1) || t < padding[1] || t > (H+padding[1]-1)){
            ret_z[x] = constant_values;
        }else{
            ret_z[x] = z[i*C*H*W + j*H*W + (k-padding[0])*W + (t-padding[1])];
        }     
    }
}
void nppad3(float ret_z[], float z[], int N, int C, int H, int W, int padding[], float constant_values){
    int pad_H = (H+padding[0]*2);
    int pad_W = (W+padding[1]*2);
    // 四个循环写成一个
    int x_top = (int)N*C*pad_H*pad_W;
    # pragma omp parallel for
    for(int x=0; x < x_top; x++){
        int i = x / (C*pad_H*pad_W);
        int j = (x % (C*pad_H*pad_W)) / (pad_H*pad_W);
        int k = ((x % (C*pad_H*pad_W)) % (pad_H*pad_W)) / pad_W;
        int t = (((x % (C*pad_H*pad_W)) % (pad_H*pad_W)) % pad_W) / 1;

        if(k < padding[0] || k > (H+padding[0]-1) || t < padding[1] || t > (H+padding[1]-1)){
            ret_z[x] = constant_values;
        }else{
            ret_z[x] = z[i*C*H*W + j*H*W + (k-padding[0])*W + (t-padding[1])];
        }     
    }
}

int main(){
    int N=2,C=3,H=64,W=64;
    int D=4,K1=3,K2=3;
    float z[N*C*H*W];//需要运算矩阵
    // 赋值
    for(int i=0; i <N*C*H*W; i++){
        z[i] = (float)i;
    }
    float k[C*D*K1*K2];
    // 赋值
    for(int i=0; i <C*D*K1*K2; i++){
        k[i] = (float)i;
    }
    float b[] = {1, 1, 1, 1};
    int padding[2] = {1,1};
    int strides[2] = {1,1};
    int pad_H = (H+padding[0]*2);
    int pad_W = (W+padding[1]*2);
    float *conv_z;
    conv_z = (float *)malloc(sizeof(float)*(N* D* (1 + (pad_H-K1) / strides[0])* (1 + (pad_W-K2) / strides[1])));

    float *padding_z;
    padding_z = (float *)malloc(sizeof(float)*N*C*pad_H*pad_W);

    // test 1
    struct timeval start_t, end_t;       //声明时间结构体变量
	// 时间
	gettimeofday(&start_t, NULL);        //记录开始的时间
    for(int i = 0; i < 1000; i++){
        nppad1(padding_z, z, N, C, H, W, padding, 0);
    }  
    
    gettimeofday(&end_t, NULL);          //记录结束的时间            
	double total_t_sec, total_t_usec, total_t;                    //变量声明   
	total_t_sec = (double)(end_t.tv_sec - start_t.tv_sec);        //计算秒数
	total_t_usec = (double)(end_t.tv_usec - start_t.tv_usec);     //计算微秒数
	total_t = total_t_sec + total_t_usec / 1000000.0;             //计算总时间
	printf("cost1 time:%lf \n",total_t);
    // test 2
	// 时间
	gettimeofday(&start_t, NULL);        //记录开始的时间  
    for(int i = 0; i < 1000; i++){
        nppad2(padding_z, z, N, C, H, W, padding, 0);
    }  
    gettimeofday(&end_t, NULL);          //记录结束的时间              
	total_t_sec = (double)(end_t.tv_sec - start_t.tv_sec);        //计算秒数
	total_t_usec = (double)(end_t.tv_usec - start_t.tv_usec);     //计算微秒数
	total_t = total_t_sec + total_t_usec / 1000000.0;             //计算总时间
	printf("cost2 time:%lf \n",total_t);
    // test paraller 3
	// 时间
	gettimeofday(&start_t, NULL);        //记录开始的时间  
    for(int i = 0; i < 1000; i++){
        nppad3(padding_z, z, N, C, H, W, padding, 0);
    }  
    gettimeofday(&end_t, NULL);          //记录结束的时间            
	total_t_sec = (double)(end_t.tv_sec - start_t.tv_sec);        //计算秒数
	total_t_usec = (double)(end_t.tv_usec - start_t.tv_usec);     //计算微秒数
	total_t = total_t_sec + total_t_usec / 1000000.0;             //计算总时间
	printf("cost3 time:%lf \n",total_t);

    free(padding_z);
    free(conv_z);

    // int coresNum = omp_get_num_procs();
    // printf("core num %d \n",coresNum);

    // #pragma omp parallel for
    // for(int j = 0; j < coresNum; j++){
    //     printf("j=[%d], ThreadId =[%d]\n", j, omp_get_thread_num());
    // }
}
