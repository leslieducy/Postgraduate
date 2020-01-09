import numpy as np

from nn.load_mnist import load_monkey_datasets
from nn.utils import to_categorical
import os
url = os.path.join(os.path.abspath('.'),'nn/monkey_dataset/')
train_set, val_set, test_set = load_monkey_datasets(url)
x_train,val_x,x_test=np.reshape(train_set[0],(-1,3,128,128)),np.reshape(val_set[0],(-1,3,128,128)),np.reshape(test_set[0],(-1,3,128,128))
y_train,val_y,y_test=to_categorical(train_set[1]),to_categorical(val_set[1]),to_categorical(test_set[1])

print('训练数据集样本数： %d ,标签个数 %d ' % (len(x_train), len(y_train)))
print('测试数据集样本数： %d ,标签个数  %d ' % (len(x_test), len(y_test)))

print(x_train.shape)
print(x_test.shape)


#定义模型架构：
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = (3,3),data_format='channels_first', padding = 'same', activation = 'relu',input_shape = (28, 28, 1)))
model.add(MaxPool2D(pool_size = 2))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()

#编译模型
model.compile(loss = 'categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

#训练模型
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# print(x_train.shape)
# print(x_test.shape)
model.fit(x_train, y_train, batch_size=128, epochs = 10, verbose=1, validation_data=(x_test, y_test))

#评估模型
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])