#coding:utf8
from torch import nn
from torch.functional import F
from .BasicModule import BasicModule

class EasyNet(BasicModule):
    '''
    code from torchvision/models/easynet.py
    结构参考 <https://arxiv.org/abs/1404.5997>
    '''
    def __init__(self, n_feature = 4, n_output = 8):
        super(EasyNet, self).__init__()
        self.model_name = 'easynet'
        self.rnn = nn.LSTM(
            input_size=4,
            hidden_size=16,
            num_layers=1,
            batch_first=True,
        )
        # self.features = nn.Sequential(
        #     nn.Linear(n_feature,16),
        #     nn.ReLU(inplace=True),
        #     # nn.Linear(16, 32),
        #     # nn.ReLU(inplace=True),
        #     # nn.Linear(32, 64),
        #     # nn.ReLU(inplace=True),
        # )
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(16, 32),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(32, 16),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(16, n_output),
        # )
        self.out = nn.Linear(16, n_output) 
        # self.hidden = nn.Linear(n_feature, 10)   # hidden layer
        # nn.ReLU() 
        # nn.Linear(n_feature, 10)
        # self.out = nn.Linear(10, 4)   # output layer
 
    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None 表示 hidden state 会用全0的 state

        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        out = self.out(r_out[:, -1, :])
        return out
        # x = self.features(x)
        # x = x.view(1, 30000)
        # x = self.classifier(x)
        # x = self.out(x) 
        return x


        

