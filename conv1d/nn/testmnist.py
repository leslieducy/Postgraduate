# 定义权重、神经元、梯度
import numpy as np
weights = {}
weights_scale = 1e-2
filters = 1
fc_units=64
weights["K1"] = weights_scale * np.random.randn(1, filters, 3, 3).astype(np.float64)
weights["b1"] = np.zeros(filters).astype(np.float64)
weights["W2"] = weights_scale * np.random.randn(filters * 13 * 13, fc_units).astype(np.float64)
weights["b2"] = np.zeros(fc_units).astype(np.float64)
weights["W3"] = weights_scale * np.random.randn(fc_units, 10).astype(np.float64)
weights["b3"] = np.zeros(10).astype(np.float64)

# 初始化神经元和梯度
nuerons={}
gradients={}


# 定义前向传播和反向传播
from nn.layers import conv_backward,fc_forward,fc_backward
from nn.layers import flatten_forward,flatten_backward
from nn.activations import relu_forward,relu_backward
from nn.losses import cross_entropy_loss


import pyximport
pyximport.install()
# from nn.clayers import conv_forward,max_pooling_forward,max_pooling_backward
from nn.clayers import max_pooling_forward,max_pooling_backward

from p2cclayers import p2cso as conv_forward
# from tensorflow.nn import conv2d as conv_forward
# from conv1dlayers import  conv_forward_bak as conv_forward
# from nn.clayers import conv_forward


# 定义前向传播
def forward(X):
    nuerons["conv1"]=conv_forward(X.astype(np.float64),weights["K1"],weights["b1"])
    # nuerons["conv1"] = conv_forward(
    #     input=X.astype(np.float64),
    #     filter=weights["K1"],
    #     strides=(1, 1),
    #     padding='SAME',
    #     data_format='NCHW',
    #     dilations=weights["b1"],
    #     name=None
    # )
    nuerons["conv1_relu"]=relu_forward(nuerons["conv1"])
    nuerons["maxp1"]=max_pooling_forward(nuerons["conv1_relu"].astype(np.float64),pooling=(2,2))

    nuerons["flatten"]=flatten_forward(nuerons["maxp1"])
    
    nuerons["fc2"]=fc_forward(nuerons["flatten"],weights["W2"],weights["b2"])
    nuerons["fc2_relu"]=relu_forward(nuerons["fc2"])
    
    nuerons["y"]=fc_forward(nuerons["fc2_relu"],weights["W3"],weights["b3"])

    return nuerons["y"]

# 定义反向传播
def backward(X,y_true):
    loss,dy=cross_entropy_loss(nuerons["y"],y_true)
    gradients["W3"],gradients["b3"],gradients["fc2_relu"]=fc_backward(dy,weights["W3"],nuerons["fc2_relu"])
    gradients["fc2"]=relu_backward(gradients["fc2_relu"],nuerons["fc2"])
    
    gradients["W2"],gradients["b2"],gradients["flatten"]=fc_backward(gradients["fc2"],weights["W2"],nuerons["flatten"])
    
    gradients["maxp1"]=flatten_backward(gradients["flatten"],nuerons["maxp1"])
       
    gradients["conv1_relu"]=max_pooling_backward(gradients["maxp1"].astype(np.float64),nuerons["conv1_relu"].astype(np.float64),pooling=(2,2))
    gradients["conv1"]=relu_backward(gradients["conv1_relu"],nuerons["conv1"])
    gradients["K1"],gradients["b1"],_=conv_backward(gradients["conv1"],weights["K1"],X)
    return loss

# 获取精度
def get_accuracy(X,y_true):
    y_predict=forward(X)
    return np.mean(np.equal(np.argmax(y_predict,axis=-1),
                            np.argmax(y_true,axis=-1)))

from nn.load_mnist import load_mnist_datasets
from nn.utils import to_categorical
train_set, val_set, test_set = load_mnist_datasets('mnist.pkl.gz')
train_x,val_x,test_x=np.reshape(train_set[0],(-1,1,28,28)),np.reshape(val_set[0],(-1,1,28,28)),np.reshape(test_set[0],(-1,1,28,28))
train_y,val_y,test_y=to_categorical(train_set[1]),to_categorical(val_set[1]),to_categorical(test_set[1])

# 随机选择训练样本
train_num = train_x.shape[0]
def next_batch(batch_size):
    idx=np.random.choice(train_num,batch_size)
    return train_x[idx],train_y[idx]

x,y= next_batch(16)
# print("x.shape:{},y.shape:{}".format(x.shape,y.shape))

from nn.optimizers import SGD
# 初始化变量
batch_size=2
steps = 1000

# 更新梯度
sgd=SGD(weights,lr=0.01,decay=1e-6)
# 计时
import time
start = time.time()

for s in range(steps):
    X,y=next_batch(batch_size)

    # 前向过程
    forward(X)
    # 反向过程
    loss=backward(X,y)

    sgd.iterate(weights,gradients)

    if s % 100 ==0:
        idx=np.random.choice(len(val_x),200)
        # 每步的输出
        # print("\n step:{} ; loss:{}".format(s,loss))
        # print(" train_acc:{};  val_acc:{}".format(get_accuracy(X,y),get_accuracy(val_x[idx],val_y[idx])))

end = time.time()
print(end-start)
# 总的输出
# print("\n final result test_acc:{};  val_acc:{}".
#       format(get_accuracy(test_x,test_y),get_accuracy(val_x,val_y)))