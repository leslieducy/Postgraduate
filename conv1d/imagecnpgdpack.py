# encoding: utf-8
import ctypes
import time
import numpy as np

def npgdpack(matrix, dim, theta, matrix_len, dim_len, mode=0):
    lib.Grad(matrix, dim, theta, matrix_len, dim_len, mode=0)

if __name__ == "__main__":

    import numpy as np
    import nn.load_mnist as nnload
    # from nn.utils import to_categorical
    import os
    import random
    url = os.path.join(os.path.abspath('.'),'nn/monkey_dataset/')
    train_set, val_set, test_set = nnload.load_monkey_datasets(url)
    train_x=np.reshape(train_set[0],(-1,3,300,300))
    # train_set, val_set, test_set = nnload.load_mnist_datasets('nn/mnist.pkl.gz')
    # train_x = np.reshape(train_set[0],(-1,28,28)).astype(np.float64)
    # y_train,val_y,y_test=to_categorical(train_set[1]),to_categorical(val_set[1]),to_categorical(test_set[1])
    train_num = train_x.shape[0]
    def next_batch(batch_size):
        idx=np.random.choice(train_num,batch_size)
        return train_x[idx]
    nparr = next_batch(128)
    nparr.reshape(-1,3,400,300)
    # nparr.reshape(-1,28,28)
    # print(nparr,nparr.shape)
    fornum = 100
    # print(nparr,nparr.shape)
    # nparr = np.random.rand(30,30,30)
    # matrix = np.array([[[1, 4, 1], [2, 5, 1]], [[5, 1, 1], [4, 2, 1]]])
    # np
    start = time.time()
    for i in range(fornum):
        np.gradient(nparr)

    end = time.time()
    print("p",end-start)
    # c
    matrix_len = nparr.size
    dim_len = nparr.ndim + 1
    ll = ctypes.cdll.LoadLibrary # 我这是在linux下 cdll ，windows调用 windll 之类的
    # 编译命令 gcc -o npgdmid.so -shared -fPIC cnpgd.c -fopenmp
    lib = ll("./npgdmid.so")

    mid_dim = matrix_len
    dim_list = [matrix_len]
    for shape in nparr.shape:
        mid_dim = mid_dim//shape
        dim_list.append(mid_dim)
    
    dim_ct = ctypes.c_int*dim_len
    dim = dim_ct(*dim_list)
    matrix_ct = ctypes.c_float*matrix_len
    matrix = matrix_ct(*(nparr.ravel()).tolist())
    theta_ct = ctypes.c_float*(matrix_len*(dim_len-1))
    theta = theta_ct()
    start = time.time()
    for i in range(fornum):
        npgdpack(matrix, dim, theta, matrix_len, dim_len, mode=0)

    end = time.time()
    print("c",end-start)
