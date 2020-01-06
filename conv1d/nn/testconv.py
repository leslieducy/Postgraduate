import numpy as np
import time 
from nn.load_mnist import load_mnist_datasets
from nn.utils import to_categorical

from conv1dlayers import  conv_forward_bak
import pyximport
pyximport.install()
from clayers import conv_forward
from tensorflow.nn import conv2d
from p2cclayers import p2cso

# 随机产生的测试数据
def RamdomInput():
    z = np.array(range(2*3*64*128))
    z = z.reshape(2,3,64,128)
    K = np.array(range(3*4*3*3))
    K = K.reshape(3,4,3,3)
    b = np.array([1,1,1,1])
    return z, K, b


def MNISTImageInput():
    # load datasets
    path = 'mnist.pkl.gz'
    train_set, val_set, test_set = load_mnist_datasets(path)
    z= np.reshape(train_set[0],(-1,1,28,28))[0:1].astype(np.float64)
    # print(z)

    weights_scale = 1e-2
    filters = 1
    K = weights_scale * np.random.randn(1, filters, 3, 3).astype(np.float64)
    b = np.zeros(filters).astype(np.float64)
    # z = np.array(range(2*3*5*5))
    # z = z.reshape(2,3,5,5)
    # K = np.array(range(3*4*3*3))
    # K = K.reshape(3,4,3,3)
    # b = np.array([1,1,1,1])
    return z, K, b

if __name__ == "__main__":

    z, K, b = RamdomInput()
    # z, K, b = MNISTImageInput()

    # numpy的python版本测试
    start = time.time()

    conv_z = conv_forward_bak(z, K, b, padding=(0, 0), strides=(1, 1))

    end = time.time()
    print(end-start)

    # numpy的Cython版本测试
    
    z = np.array(range(2*3*5*5)).astype(np.float64)
    z = z.reshape(2,3,5,5)
    K = np.array(range(3*4*3*3)).astype(np.float64)
    K = K.reshape(3,4,3,3)
    b = np.array([1,1,1,1]).astype(np.float64)
    start = time.time()

    conv_z = conv_forward(z, K, b, padding=(0, 0), strides=(1, 1))

    end = time.time()
    print(end-start)
    
    # tensorflow版本测试
    start = time.time()

    conv_z = conv2d(
        input=z,
        filter=K,
        strides=(1, 1),
        padding='SAME',
        data_format='NCHW',
        dilations=[1, 1, 1, 1],
        name=None
    )
    # print(conv_z)

    end = time.time()
    print(end-start)
    # print(conv_z)

    # 自己的版本测试
    start = time.time()

    conv_z = p2cso(z, K, b, padding=(0, 0), strides=(1, 1))

    end = time.time()
    print(end-start)
