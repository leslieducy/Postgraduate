#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
void nppad(float ret_z[], float z[], int N, int C, int H, int W, int padding[], float constant_values){
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
// 数组求和
float npsum(float a[], float b[], int length){
    float sum = 0;
    for(int i=0;  i < length; i++){
        sum += a[i]*b[i];
    }
    // sum /=length;
    // printf("outsum:%f", sum);
    return sum;
}
// 提取卷积z元素
void getsumz(float sum_z[], float padding_z[], int N, int C, int pad_H, int pad_W,int K1, int K2, int n, int h, int w){
    // padding_z[n, :, h:h + k1, w:w + k2]
    
    int h_K1 = h+K1;
    int w_K2 = w+K2;
    for(int j=0; j < C; j++){
        for(int k=h; k < h_K1; k++){
            for(int t=w; t < w_K2; t++){
                sum_z[j*K1*K2 + (k-h)*K1 + (t-w)] = padding_z[n*C*pad_H*pad_W + j*pad_H*pad_W + k*pad_W + t];
            }
        }
    }
 
}
// 提取卷积k元素
void getsumk(float sum_k[], float padding_k[], int C, int D, int K1, int K2, int d){
    // K[:, d]
    for(int i=0; i < C; i++){
        for(int k=0; k < K1; k++){
            for(int t=0; t < K2; t++){
                sum_k[i*K1*K2 + k*K2 + t] = padding_k[i*D*K1*K2 + d*K1*K2 + k*K2 + t];
            }
        }
    }
}


// 卷积层前向
void conv_forward1(float conv_z[], float z[], float k[], float b[], int padding[], int strides[], int N, int C, int H, int W, int D, int K1, int K2){
    int pad_H = (H+padding[0]*2);
    int pad_W = (W+padding[1]*2);
    float *padding_z;
    padding_z = (float *)malloc(sizeof(float)*N*C*pad_H*pad_W);
    nppad(padding_z, z, N, C, H, W, padding, 0);
    // if ((pad_H - K1) % strides[0] == 0 || (pad_W - K2) % strides[1] == 0){
    //     return "步长不为1时，步长必须刚好能够被整除"
    // }
    int F_H = pad_H-K1+1;
    int F_W = pad_W-K2+1;
    // 提出循环的部分
    // 总的计算长度
    int sum_len = C*K1*K2;
    // 卷积核
    float sum_k[sum_len];
    float sum_z[sum_len];
    for(int n=0; n<N; n++){
        for(int d=0; d<D; d++){
            getsumk(sum_k, k, C, D, K1, K2, d);
            for (int h = 0; h < F_H; h+=strides[0]){
                for (int w = 0; w < F_W; w+=strides[1]){
                    getsumz(sum_z, padding_z, N, C, pad_H, pad_W, K1, K2, n, h, w);
                    // conv_z[n, d, h // strides[0], w // strides[1]]
                    conv_z[n*D*(F_H)/strides[0]*(F_W)/strides[1] + d*(F_H)/strides[0]*(F_W)/strides[1] + h/strides[0]*(F_W)/strides[1] + w / strides[1]] = npsum(sum_z,sum_k,sum_len) + b[d];
                }
            }
        }
    }
    free(padding_z);
}
// 卷积层前向
void conv_forward2(float conv_z[], float z[], float k[], float b[], int padding[], int strides[], int N, int C, int H, int W, int D, int K1, int K2){
    int pad_H = (H+padding[0]*2);
    int pad_W = (W+padding[1]*2);
    float *padding_z;
    padding_z = (float *)malloc(sizeof(float)*N*C*pad_H*pad_W);
    nppad(padding_z, z, N, C, H, W, padding, 0);
    // if ((pad_H - K1) % strides[0] == 0 || (pad_W - K2) % strides[1] == 0){
    //     return "步长不为1时，步长必须刚好能够被整除"
    // }
    int F_H = pad_H-K1+1;
    int F_W = pad_W-K2+1;
    // 提出循环的部分
    // 总的计算长度
    int sum_len = C*K1*K2;
    // 卷积核
    float sum_k[sum_len];
    float sum_z[sum_len];
    // 四个循环写成一个
    int x_top = N*D*(F_H/strides[0])*(F_W/strides[1]);
    int w_divi = 1;
    int h_divi = w_divi*(F_W/strides[1]);
    int d_divi = h_divi*(F_H/strides[0]);
    int n_divi = d_divi*D;
    // # pragma omp parallel for
    for(int x=0; x < x_top; x++){
        int n = x / n_divi;
        int d = (x % n_divi) / d_divi;
        int h = ((x % n_divi) % d_divi) / h_divi;
        int w = (((x % n_divi) % d_divi) % h_divi) / 1;
        if((x % n_divi) % d_divi == 0){
            getsumk(sum_k, k, C, D, K1, K2, d);
        }
        getsumz(sum_z, padding_z, N, C, pad_H, pad_W, K1, K2, n, h, w);
        conv_z[x] = npsum(sum_z,sum_k,sum_len) + b[d];
    }

    free(padding_z);
}
// 卷积层前向
void conv_forward3(float conv_z[], float z[], float k[], float b[], int padding[], int strides[], int N, int C, int H, int W, int D, int K1, int K2){
    int pad_H = (H+padding[0]*2);
    int pad_W = (W+padding[1]*2);
    float *padding_z;
    padding_z = (float *)malloc(sizeof(float)*N*C*pad_H*pad_W);
    nppad(padding_z, z, N, C, H, W, padding, 0);
    // if ((pad_H - K1) % strides[0] == 0 || (pad_W - K2) % strides[1] == 0){
    //     return "步长不为1时，步长必须刚好能够被整除"
    // }
    int F_H = pad_H-K1+1;
    int F_W = pad_W-K2+1;
    // 提出循环的部分
    // 总的计算长度
    int sum_len = C*K1*K2;
    // 卷积核
    float sum_k[sum_len];
    float sum_z[sum_len];
    // 四个循环写成一个
    int x_top = N*D*(F_H)/strides[0]*(F_W)/strides[1];
    # pragma omp parallel for
    for(int x=0; x < x_top; x++){
        int n = x / (D*(F_H)/strides[0]*(F_W)/strides[1]);
        int d = (x % (D*(F_H)/strides[0]*(F_W)/strides[1])) / ((F_H)/strides[0]*(F_W)/strides[1]);
        int h = ((x % (D*(F_H)/strides[0]*(F_W)/strides[1])) % ((F_H)/strides[0]*(F_W)/strides[1])) / ((F_W)/strides[1]);
        int w = (((x % (D*(F_H)/strides[0]*(F_W)/strides[1])) % ((F_H)/strides[0]*(F_W)/strides[1])) % ((F_W)/strides[1])) / 1;
        if((x % (D*(F_H)/strides[0]*(F_W)/strides[1])) % ((F_H)/strides[0]*(F_W)/strides[1]) == 0){
            getsumk(sum_k, k, C, D, K1, K2, d);
        }
        getsumz(sum_z, padding_z, N, C, pad_H, pad_W, K1, K2, n, h, w);
        conv_z[x] = npsum(sum_z,sum_k,sum_len) + b[d];
    }

    free(padding_z);
}
int main(){
    int N=2,C=3,H=28,W=28;
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
    // test 1
    struct timeval start_t, end_t;       //声明时间结构体变量
	// 时间
	gettimeofday(&start_t, NULL);        //记录开始的时间
    for(int i = 0; i < 100; i++){
        conv_forward1(conv_z, z, k, b, padding, strides, N, C, H, W, D, K1, K2);
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
    for(int i = 0; i < 100; i++){
        conv_forward2(conv_z, z, k, b, padding, strides, N, C, H, W, D, K1, K2);
    }  
    gettimeofday(&end_t, NULL);          //记录结束的时间              
	total_t_sec = (double)(end_t.tv_sec - start_t.tv_sec);        //计算秒数
	total_t_usec = (double)(end_t.tv_usec - start_t.tv_usec);     //计算微秒数
	total_t = total_t_sec + total_t_usec / 1000000.0;             //计算总时间
	printf("cost2 time:%lf \n",total_t);
    // test paraller 3
	// 时间
	gettimeofday(&start_t, NULL);        //记录开始的时间  
    for(int i = 0; i < 100; i++){
        conv_forward3(conv_z, z, k, b, padding, strides, N, C, H, W, D, K1, K2);
    }  
    gettimeofday(&end_t, NULL);          //记录结束的时间            
	total_t_sec = (double)(end_t.tv_sec - start_t.tv_sec);        //计算秒数
	total_t_usec = (double)(end_t.tv_usec - start_t.tv_usec);     //计算微秒数
	total_t = total_t_sec + total_t_usec / 1000000.0;             //计算总时间
	printf("cost3 time:%lf \n",total_t);

    free(conv_z);

    // int coresNum = omp_get_num_procs();
    // printf("core num %d \n",coresNum);

    // #pragma omp parallel for
    // for(int j = 0; j < coresNum; j++){
    //     printf("j=[%d], ThreadId =[%d]\n", j, omp_get_thread_num());
    // }
}
