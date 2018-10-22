import torch 
from torch.autograd import Variable 
import torch.nn.functional as F 
  
# 生成数据 
# 分别生成2组各100个数据点，增加正态噪声，后标记以y0=0 y1=1两类标签，最后cat连接到一起 
n_data = torch.ones(100,2) 
# torch.normal(means, std=1.0, out=None) 
x0 = torch.normal(2*n_data, 1) # 以tensor的形式给出输出tensor各元素的均值，共享标准差 
y0 = torch.zeros(100) 
x1 = torch.normal(-2*n_data, 1) 
y1 = torch.ones(100) 
  
x = torch.cat((x0, x1), 0).type(torch.FloatTensor) # 组装（连接） 
y = torch.cat((y0, y1), 0).type(torch.LongTensor) 
  
# 置入Variable中 
x, y = Variable(x), Variable(y) 
  
class Net(torch.nn.Module): 
    def __init__(self, n_feature, n_hidden, n_output): 
        super(Net, self).__init__() 
        self.hidden = torch.nn.Linear(n_feature, n_hidden) 
        self.out = torch.nn.Linear(n_hidden, n_output) 
    
    def forward(self, x): 
        x = F.relu(self.hidden(x)) 
        x = self.out(x) 
        return x 
  
net = Net(n_feature=2, n_hidden=10, n_output=2) 
  
optimizer = torch.optim.SGD(net.parameters(), lr=0.012) 
loss_func = torch.nn.CrossEntropyLoss() 
  
print(x[0]) 
for t in range(100): 
    out = net(x)
    print("比较：",out)
    loss = loss_func(out, y) # loss是定义为神经网络的输出与样本标签y的差别，故取softmax前的值 
    # print(loss)
    
    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step() 