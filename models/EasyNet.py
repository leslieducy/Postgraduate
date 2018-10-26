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
        self.features = nn.Sequential(
            nn.Linear(n_feature,16),
            nn.ReLU(inplace=True),
            # nn.Linear(16, 32),
            # nn.ReLU(inplace=True),
            # nn.Linear(32, 64),
            # nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(16, n_output),
        )
        # self.out = nn.Linear(10, n_output) 
        # self.hidden = nn.Linear(n_feature, 10)   # hidden layer
        # nn.ReLU() 
        # nn.Linear(n_feature, 10)
        # self.out = nn.Linear(10, 4)   # output layer
 
    def forward(self, x):
        x = self.features(x)
        # x = x.view(1, 30000)
        x = self.classifier(x)
        # x = self.out(x) 
        return x


        

