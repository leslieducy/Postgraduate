import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 50               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # learning rate

def load_data(dataset_path='./train_data/'):
    features = []
    labels = []
    # 遍历训练文件夹中1到400的图片
    for item in range(0,400):
        img = Image.open(dataset_path+str(item+1)+'.jpg')
        # 将PIL.Image图片类型转成numpy.array
        img_ndarray = np.asarray(img, dtype='float32') / 256
        features.append(img_ndarray)
        labels.append(int(item/40))
    # print(np.array(features),np.array(labels))
    return np.array(features),np.array(labels, dtype='int64')

def load_test(dataset_path='./test_data/'):
    features = []
    # 遍历训练文件夹中1到400的图片
    for item in range(0,50):
        img = Image.open(dataset_path+str(item+1)+'.jpg')
        # 将PIL.Image图片类型转成numpy.array
        img_ndarray = np.asarray(img, dtype='float32') / 256
        features.append(img_ndarray)
    return np.array(features)

features,labels = load_data()
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=0.2,random_state=0)
x_train = torch.unsqueeze(torch.from_numpy(x_train), dim=1)
y_train = torch.from_numpy(y_train)
x_test = torch.unsqueeze(torch.from_numpy(x_test), dim=1)
y_test = torch.from_numpy(y_test)
train_data = Data.TensorDataset(x_train,y_train)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 56, 56)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 56, 56)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 28, 28)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 28, 28)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 14, 14)
        )
        self.out = nn.Linear(32 * 14 * 14, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization


cnn = CNN()
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
loss_list = []
# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader

        output = cnn(b_x)[0]            # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        loss_list.append(loss.data.numpy())
        if step % 50 == 0:
            test_output, last_layer = cnn(x_test)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == y_test.data.numpy()).astype(int).sum()) / float(y_test.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)



plt.plot(loss_list)
plt.show()
test_data = load_test()
test_data = torch.unsqueeze(torch.from_numpy(test_data), dim=1)
test_output, _ = cnn(test_data)
pred_y = torch.max(test_output, 1)[1].data.numpy()

print(pred_y)


