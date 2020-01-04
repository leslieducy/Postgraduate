#include <stdlib.h>
#include <stdio.h>
// #include <omp.h>

// 填充数组
void nppad(float ret_z[], float z[], int N, int C, int H, int W, int padding[], float constant_values){
    int pad_H = (H+padding[0]*2);
    int pad_W = (W+padding[1]*2);
    
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
    // printf("pad model out%d: \n",N*C*pad_H*pad_W);
	// for(int i=0;i<N*C*pad_H;i++)
    // {
    //     for(int j = 0; j < pad_W; j++){
    //         printf("%f ",ret_z[i*pad_W + j]);
    //     }
	//     printf("\n");
	// }
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
void conv_forward(float conv_z[], float z[], float k[], float b[], int padding[], int strides[], int N, int C, int H, int W, int D, int K1, int K2){
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


int main()
{
	int N=2,C=3,H=5,W=5;
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
    // nppad(ret_z, z, N, C, H, W, padding, 0);

	// conv(xxx,yyy,zzz,ah, aw, fh, fw);//卷积运算
    conv_forward(conv_z, z, k, b, padding, strides, N, C, H, W, D, K1, K2);
	// 输出结果
    printf("pad model out%d: \n",N* D* (1 + (pad_H-K1) / strides[0])* (1 + (pad_W-K2) / strides[1]));
	for(int i=0;i<N* D* (1 + (pad_H-K1) / strides[0]);i++)
    {
        for(int j = 0; j < (1 + (pad_W-K2) / strides[1]); j++){
            printf("%f ",conv_z[i*(1 + (pad_W-K2) / strides[1]) + j]);
        }
	    printf("\n");
	}
    free(conv_z);
	return 0;
 }