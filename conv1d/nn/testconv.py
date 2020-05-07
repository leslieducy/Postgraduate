import numpy as np
import time 
import os
import csv
from load_mnist import load_mnist_datasets,load_monkey_datasets
from nn.utils import to_categorical

from conv1dlayers import  conv_forward_bak
import pyximport
pyximport.install()
from clayers import conv_forward
# import tensorflow as tf
# from tensorflow.nn import conv2d
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
    return z, K, b

def MonkeyImageInput():
    # load datasets
    url = os.path.join(os.path.abspath('.'),'monkey_dataset/')
    train_set, val_set, test_set = load_monkey_datasets(url)
    z= np.reshape(train_set[0],(-1,3,300,300))[0:1].astype(np.float64)
    # print(z)

    weights_scale = 1e-2
    filters = 3
    K = weights_scale * np.random.randn(3, filters, 3, 3).astype(np.float64)
    b = np.zeros(filters).astype(np.float64)
    return z, K, b

def saveCSV(title='result', data=[[0,1]]):
    header = ["No","CostTime"]
    with open(title+'.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

if __name__ == "__main__":
    for_num = 100
    exp_name = 'monkey'
    # z, K, b = RamdomInput()
    # z, K, b = MNISTImageInput()
    z, K, b = MonkeyImageInput()

    # numpy的python版本测试
    save_data = []
    for i in range(for_num):
        start = time.time()
        conv_z = conv_forward_bak(z, K, b, padding=(0, 0), strides=(1, 1))
        end = time.time()
        save_data.append([i,end-start])
        # print(end-start)
    saveCSV(title=exp_name+'_numpy', data=save_data)


    # numpy的Cython版本测试
    save_data = []
    for i in range(for_num):
        start = time.time()
        conv_z = conv_forward(z, K, b, padding=(0, 0), strides=(1, 1)) 
        end = time.time()
        save_data.append([i,end-start])
        # print(end-start)
    saveCSV(title=exp_name+'_cython', data=save_data)

    # 自己的版本测试
    save_data = []
    for i in range(for_num):
        start = time.time()
        conv_z = p2cso(z, K, b, padding=(0, 0), strides=(1, 1))
        end = time.time()
        save_data.append([i,end-start])
        # print(end-start)
    saveCSV(title=exp_name+'_openmp', data=save_data)


    # tensorflow版本测试
    # start = time.time()

    # conv_z = conv2d(
    #     input=z,
    #     filter=K,
    #     strides=(1, 1),
    #     padding='SAME',
    #     data_format='NCHW',
    #     dilations=[1, 1, 1, 1],
    #     name=None
    # )
    # with tf.Session() as sess:
    #     print(conv_z.eval())

    # end = time.time()
    # print(end-start)
    # print(conv_z)