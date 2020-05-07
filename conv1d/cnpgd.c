#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
// 输入数组，数组长度，输出的数组各维度数组(向前累积记录，第一个数最大)，维度数组长度，返回的数组，模式
// void Grad(float matrix[], int dim[], float theta[], int matrix_len, int dim_len, int mode)
// {
// 	// 中间值
// 	float mid = 0;
//     // 对每个梯度方向都进行梯度计算
// 	for(int d=1; d < dim_len; d++){
//         // 数组的每个元素都需要计算该方向上的梯度
//         for(int m=0; m < matrix_len; m++){
//             // 该维度的第一个或最后一个是b-a,中间的元素都是(c-a)/2
//             if ((m % dim[d-1]) / dim[d] < 1){
// 				// printf("matrix[m]%f ", matrix[m]);
// 				mid = matrix[m + dim[d]] - matrix[m];
//             }else if((m % dim[d-1]) / dim[d] > ((dim[d-1] / dim[d]) - 2) ){
// 				mid = matrix[m] - matrix[m - dim[d]];
// 			}else{
// 				mid = (matrix[m + dim[d]] - matrix[m - dim[d]])/2;
// 			}
// 			// printf("%f ", mid);
// 			theta[(d-1)*matrix_len + m] = mid;
//         }
//     }
// }
void Grad(float matrix[], int dim[], float theta[], int matrix_len, int dim_len, int mode)
{
	// 中间值
	float mid = 0;

	int mdfor = matrix_len*(dim_len-1);
	# pragma omp parallel for
	for(int md=0; md < mdfor; md++){
		int m = md % matrix_len;
		int d = md / matrix_len + 1;
		if ((m % dim[d-1]) / dim[d] < 1){
			mid = matrix[m + dim[d]] - matrix[m];
		}else if((m % dim[d-1]) / dim[d] > ((dim[d-1] / dim[d]) - 2)){
			mid = matrix[m] - matrix[m - dim[d]];
		}else{
			mid = (matrix[m + dim[d]] - matrix[m - dim[d]])/2;
		}
		// printf("%f ", mid);
		theta[(d-1)*matrix_len + m] = mid;
	}
    // 数组的每个元素都需要计算每个方向上的梯度
    // for(int m=0; m < matrix_len; m++){
    // 	// 对每个梯度方向都进行梯度计算
	// 	for(int d=1; d < dim_len; d++){
    //         // 该维度的第一个或最后一个是b-a,中间的元素都是(c-a)/2
    //         if ((m % dim[d-1]) / dim[d] < 1){
	// 			// printf("matrix[m]%f ", matrix[m]);
	// 			mid = matrix[m + dim[d]] - matrix[m];
    //         }else if((m % dim[d-1]) / dim[d] > ((dim[d-1] / dim[d]) - 2)){
	// 			mid = matrix[m] - matrix[m - dim[d]];
	// 		}else{
	// 			mid = (matrix[m + dim[d]] - matrix[m - dim[d]])/2;
	// 		}
	// 		// printf("%f ", mid);
	// 		theta[(d-1)*matrix_len + m] = mid;
    //     }
    // }
}


int main()
{
	//double matrix[4][3] = {{4,3,1},{6,8,1},{18,26,1},{55,77,1}};
	//double result[4] = {7,10,55,120};
	// float matrix[4][3] = {{1, 4, 1}, {2, 5, 1}, {5, 1, 1}, {4, 2, 1}};
	float matrix[12] = {1, 4, 1, 2, 5, 1, 5, 1, 1, 4, 2, 1};
	int matrix_len = 12;
	int dim[] ={12,6,3,1};
	int dim_len = 4;
	float theta[matrix_len*(dim_len-1)];
	int mode = 0;
	Grad(matrix, dim, theta, matrix_len, dim_len, mode);
	printf("\n");
	for(int m=0; m < matrix_len*(dim_len-1); m++){
		printf("%f ", theta[m]);
	}
	
}