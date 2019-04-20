import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import gym
import cx_Oracle as cx      #导入模块
import datetime
import matplotlib.pyplot as plt
# 超参数
BATCH_SIZE = 512
LR = 0.05                   # learning rate
EPSILON = 0.9               # 最优选择动作百分比
GAMMA = 0.95                # 奖励递减参数
TARGET_REPLACE_ITER = 100   # Q 现实网络的更新频率
MEMORY_CAPACITY = 1000      # 记忆库大小
N_ACTIONS = 1049  # 动作总数
N_STATES = 1049   # 状态总数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA = [2013,5,4]
con = cx.connect('test', 'herron', '127.0.0.1:1521/TestDatabase')  #创建连接
FINISHED_LIST = []

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(N_STATES, 5000),
            nn.ReLU(),
            nn.Linear(5000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 5000),
        )
        self.out = nn.Linear(5000, N_ACTIONS)
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
        if end_road_list is None or len(end_road_list) == 0:
            return None
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(device)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x).cpu().data.numpy()[0]
            action, max_prob = None, actions_value[end_road_list[0][1]-1]-1
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
        # print("损失函数值：",loss)
        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
def query_has_action(x, time):
    road_no = x.index(1)
    cursor = con.cursor()       #创建游标
    cursor.execute("select NEI_ROAD from NEIGHBOR t where PRI_ROAD = "+str(road_no))  #执行sql语句
    neigh_road_list = cursor.fetchall()       #获取一条数据
    cursor.close()  #关闭游标
    neigh_road_list = [str(x[0]) for x in neigh_road_list]
    neigh_road_list.append(str(road_no))
    now_time = datetime.datetime(*(DATA + time))
    before_time = now_time - datetime.timedelta(minutes=5)

    if now_time.strftime("%Y-%m-%d") != before_time.strftime("%Y-%m-%d"):
        before_time = datetime.datetime(*(DATA))
    
    wait = 0
    while True:
        wait += 1
        later_time = now_time + datetime.timedelta(minutes=(5*wait))
        if now_time.strftime("%Y-%m-%d") != later_time.strftime("%Y-%m-%d"):
            print(now_time.strftime("%Y-%m-%d"), later_time.strftime("%Y-%m-%d"))
            print("一天结束")
            return None
        cursor = con.cursor()       #创建游标
        placeholders = ','.join(":%d" % i for i,_ in enumerate(neigh_road_list))
        sql = ("select * from TEM_REQ t where on_time > '"
            + before_time.strftime("%H:%M:%S") + "' and on_time < '"
            + later_time.strftime("%H:%M:%S")+"' and day_no = '" 
            + now_time.strftime("%Y-%m-%d") +"' and start_road in (%s)" % placeholders)
        # print(sql)
        # print(neigh_road_list)
        cursor.execute(sql,neigh_road_list)  #执行sql语句
        # data = cursor.fetchone()        #获取一条数据
        data = cursor.fetchall()       #获取全部数据
        cursor.close()  #关闭游标
        end_road_list = []
        for item in data:
            if item[0] not in FINISHED_LIST:
                end_road_list.append((item[0],item[10]))
                
        # print(len(end_road_list))
        if len(end_road_list)>1:
            break
    return end_road_list
def get_init_state():
    start_road = 409
    state_list = [-1] * N_STATES
    state_list[start_road] = 1
    time = [0,9,0]
    end_road_list = query_has_action(state_list,time)
    for item in end_road_list:
        if item == start_road:
            continue
        state_list[item[1]-1] = 0
    return state_list,time
def get_step_state(a):
    if a == None:
        return None, None, ''
    cursor = con.cursor()       #创建游标
    cursor.execute("select * from TEM_REQ t where REQUEST_ID = '" + str(a)+"'")  #执行sql语句
    data = cursor.fetchone()       #获取一条数据
    # print(a)
    cursor.close()  #关闭游标
    if data:
        s_no, exp_time, r = data[10], data[4], data[2]
        s_ = [-1] * N_STATES
        s_[s_no - 1] = 1
        end_road_list = query_has_action(s_,time)
        if end_road_list is None:
            return None, r, exp_time
        for item in end_road_list:
            if item[1] == s_no:
                continue
            s_[item[1]-1] = 0
        return s_, r, exp_time
    else:
        print("没有数据")
        return None, None, ''
dqn = DQN()
mon_plt = []
# 训练次数
train_n = 500
for i_episode in range(train_n):
    FINISHED_LIST = []
    s, time = get_init_state()
    money = 0
    while True:
        # 选动作
        q_a = dqn.choose_action(s, time)
        if q_a is None:
            # 没有找到动作请求, 进入下回合
            break
        FINISHED_LIST.append(q_a[0])
        # 得到环境反馈
        # print("开始获取反馈")
        s_, r, exp_time = get_step_state(q_a[0])
        # print(time, exp_time)
        money += r
        if s_ is None:
            # 没有找到下一个动作, 进入下回合
            break
        # print(s_,r)
        exp_time = exp_time.split(':')
        time = [int(x) for x in exp_time]
        # 修改 reward, 使 DQN 快速学习
        # r = r1 + r2
        # 存记忆
        dqn.store_transition(s, q_a[1], r, s_)
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn() # 记忆库满了就进行学习
        s = s_
    print("第",i_episode,"次：",money)
    mon_plt.append(money)
print(np.mean(mon_plt))
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