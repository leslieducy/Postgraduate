import os

# third-party library
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 1
TIME_STEP = 10          # rnn time step / image height（处理数据时就需要设置好）
INPUT_SIZE = 1         # rnn input size / image width
LR = 0.01              # learning rate
DOWNLOAD_MNIST = False

# show data
# steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)  # float32 for converting torch FloatTensor
# x_np = np.sin(steps)
# y_np = np.cos(steps)
# plt.plot(steps, y_np, 'r-', label='target (cos)')
# plt.plot(steps, x_np, 'b-', label='input (sin)')
# plt.legend(loc='best')
# plt.show()

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,  # 每次的输入大小
            hidden_size=32, # rnn hidden unit
            num_layers=1,   # 有几层 RNN layers
            batch_first=True,# input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )
       
        self.out = nn.Linear(32, 1)   # fully connected layer, output 10 classes

    def forward(self, x, h_state):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        # 每个时间点的 r_out 都要输出
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state   # return x for visualization


rnn = RNN()
print(rnn)  # net architecture

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all rnn parameters
loss_func = nn.MSELoss()                       # 回归的损失函数

# 构造dataset
x_list = []
y_list = []
for step in range(100):
    start, end = step * np.pi, (step+1)*np.pi   # time range
    # use sin predicts cos
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32, endpoint=False)  # float32 for converting torch FloatTensor
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    x_list.append(x_np)
    y_list.append(y_np)

x = torch.from_numpy(np.array(x_list, dtype=np.float32)[:, :, np.newaxis])    # shape (batch, time_step, input_size)
y = torch.from_numpy(np.array(y_list, dtype=np.float32)[:, :, np.newaxis])    # （100， 10， 1）
torch_dataset = Data.TensorDataset(x, y)
train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=False)

h_state = None      # for initial hidden state
plt.figure(1, figsize=(12, 5))
plt.ion()           # continuously plot
# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        # print(b_y.shape)
        prediction, h_state = rnn(b_x, h_state)   # rnn output
        # print(prediction.shape)
        # h_state = h_state.data        # repack the hidden state, break the connection from last iteration
        loss = loss_func(prediction, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward(retain_graph=True)                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        # plotting
        # if step%10 == 0:
        start, end = step * np.pi, (step+1)*np.pi   # time range
        # use sin predicts cos
        steps = np.linspace(start, end, TIME_STEP, dtype=np.float32, endpoint=False)  # float32 for converting torch FloatTensor
        
        plt.plot(steps, b_y.numpy().flatten(), 'r-')
        plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
        plt.draw(); 
        plt.pause(0.05)
plt.ioff()
plt.show()