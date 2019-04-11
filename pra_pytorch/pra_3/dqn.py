import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import gym
import cx_Oracle as cx      #导入模块
import datetime
import matplotlib.pyplot as plt
# 超参数
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # 最优选择动作百分比
GAMMA = 0.9                 # 奖励递减参数
TARGET_REPLACE_ITER = 100   # Q 现实网络的更新频率
MEMORY_CAPACITY = 10000      # 记忆库大小
N_ACTIONS = 1049  # 动作总数
N_STATES = 1049   # 状态总数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA = [2013,5,4]
con = cx.connect('test', 'herron', '127.0.0.1:1521/TestDatabase')  #创建连接
def query_has_action(x, time):
    road_no = x.index(1)
    now_time = datetime.datetime(*(DATA + time))
    later_time = now_time + datetime.timedelta(minutes=5)
    if now_time.strftime("%Y-%m-%d") != later_time.strftime("%Y-%m-%d"):
        print(now_time.strftime("%Y-%m-%d"), later_time.strftime("%Y-%m-%d"))
        print("一天结束")
        return None
    cursor = con.cursor()       #创建游标
    cursor.execute("select * from REQUEST t where on_time > '"+
        now_time.strftime("%H:%M:%S") + "' and on_time < '"+
        later_time.strftime("%H:%M:%S")+"' and day_no = '" + 
        now_time.strftime("%Y-%m-%d") +"'")  #执行sql语句
    # data = cursor.fetchone()        #获取一条数据
    data = cursor.fetchall()       #获取全部数据
    end_road_list = []
    for item in data:
        end_road_list.append((item[0],item[10]))
    # print(end_road_list)     #打印数据
    cursor.close()  #关闭游标
    return end_road_list


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10000)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(10000, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value

class DQN(object):
    def __init__(self):
        self.eval_net,self.target_net = Net(),Net()
        self.eval_net.to(device)
        self.target_net.to(device)
        self.learn_step_counter = 0     # 用于 target 更新计时
        self.memory_counter = 0         # 记忆库记数
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # 初始化记忆库
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # torch 的优化器
        self.loss_func = nn.MSELoss()   # 误差公式
    def choose_action(self, x, time):
        end_road_list = query_has_action(x, time)
        if end_road_list == None:
            return None
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(device)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x).cpu().data.numpy()[0]
            action, max_prob = None, -1
            for item in end_road_list:
                if actions_value[item[1]-1] > max_prob:
                    action = item
                    max_prob = actions_value[item[1]-1]
        else:   # random
            rand = np.random.randint(0, len(end_road_list))
            action = end_road_list[rand]
        return action
    def store_transition(self, s, a, r , s_):
        transition = np.hstack((s, [a, r], s_))
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % MEMORY_CAPACITY
        # print(transition.shape)
        self.memory[index, :] = transition
        self.memory_counter += 1
    def learn(self):
        # target net 参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES]).to(device)
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]).to(device)
        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a).to(device)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach().to(device)     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1).to(device)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def get_init_state():
    
    state_list = [0] * N_STATES
    state_list[409] = 1
    time = [0,9,0]
    return state_list,time
def get_step_state(a):
    if a == None:
        return None, None, ''
    cursor = con.cursor()       #创建游标
    # print(a)
    cursor.execute("select * from REQUEST t where REQUEST_ID = '" + str(a)+"'")  #执行sql语句
    data = cursor.fetchone()       #获取一条数据
    cursor.close()  #关闭游标
    if data:
        s_num, exp_time, r = data[10], data[4], data[2]
        s_ = [0] * N_STATES
        s_[s_num - 1] = 1
        return s_, r, exp_time
    else:
        return None, None, ''
dqn = DQN()
mon_plt = []
# 训练次数
train_n = 30
for i_episode in range(train_n):
    s, time = get_init_state()
    money = 0
    while True:
        # 选动作
        q_a = dqn.choose_action(s, time)
        if q_a is None:
            # 没有找到动作请求, 进入下回合
            break
        # 得到环境反馈
        s_, r, exp_time = get_step_state(q_a[0])
        
        exp_time = exp_time.split(':')
        time = [int(x) for x in exp_time]
        # 修改 reward, 使 DQN 快速学习
        # x, x_dot, theta, theta_dot = s_
        # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        # r = r1 + r2
        # 存记忆
        dqn.store_transition(s, q_a[1], r, s_)
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn() # 记忆库满了就进行学习
        s = s_
        money += r
    print("第",i_episode,"次：",money)
    mon_plt.append(money)
x = range(0,train_n,1)
plt.figure(figsize=(8,6), dpi=80)
plt.title("DQN")
plt.xlabel("train(reward)") 
plt.ylabel("n") 
plt.plot(x, mon_plt, label="precision")
plt.legend(loc='upper right')
# plt.ylim(0,100)
plt.show()
print("过程全部结束")
con.close()     #关闭数据库连接