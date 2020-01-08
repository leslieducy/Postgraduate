# encoding: utf-8
import ctypes
import numpy as np
import time 

# 引用编译好的C文件
ll = ctypes.cdll.LoadLibrary # 我这是在linux下 cdll ，windows调用 windll 之类的
# lib = ll("./pyconv.so")
lib = ll("./mppyconv.so")
# 将python类型转换为C类型并调用函数,传入 numpy数据，输出numpy数据
def p2cso(z, K, b, padding=(0, 0), strides=(1, 1)):
    # 初始化
    N, _, H, W = z.shape
    C, D, k1, k2 = K.shape
    
    pad_H = (H+padding[0]*2)
    pad_W = (W+padding[1]*2)
    float_z = ctypes.c_float*(N*C*H*W)
    c_z = float_z(*(z.flatten().tolist()))
    float_k = ctypes.c_float*(C*D*k1*k2)
    c_k = float_k(*(K.flatten().tolist()))
    float_b = ctypes.c_float*D
    c_b = float_b(*(b.flatten().tolist()))

    int_padding = ctypes.c_int*2
    c_padding = int_padding(*list(padding))
    int_strides = ctypes.c_int*2
    c_strides = int_strides(*list(strides))

    float_conv_z = ctypes.c_float*(N* D* (1+(pad_H-k1)//strides[0])* (1+(pad_W-k2)//strides[1]) )
    c_conv_z = float_conv_z()

    lib.conv_forward(c_conv_z, c_z, c_k, c_b, c_padding, c_strides, N, C, H, W, D, k1, k2)

    conv_z = np.array(c_conv_z).reshape(N, D, 1 + (pad_H - k1) // strides[0], 1 + (pad_W - k2) // strides[1])
    return conv_z

if __name__ == "__main__":
    z = np.array(range(2*3*5*5))
    z = z.reshape(2,3,5,5)
    K = np.array(range(3*4*3*3))
    K = K.reshape(3,4,3,3)
    b = np.array([1,1,1,1])

    start = time.time()

    conv_z = p2cso(z, K, b, padding=(0, 0), strides=(1, 1))

    end = time.time()
    print(end-start)
    # print(conv_z)
    
