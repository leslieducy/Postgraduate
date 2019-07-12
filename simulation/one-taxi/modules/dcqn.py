import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
# 超参数
BATCH_SIZE = 32
LR = 0.001                   # learning rate
EPSILON = 0.9               # 最优选择动作百分比
GAMMA = 0.9                # 奖励递减参数
TARGET_REPLACE_ITER = 100   # Q 现实网络的更新频率
MEMORY_CAPACITY = 1000      # 记忆库大小
N_ACTIONS = 1049  # 动作总数
N_STATES = 1049   # 状态总数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNet(nn.Module):
    def __init__(self):
        super(CNet,self).__init__()
        self.conv1 = nn.Sequential(  # input shape (2, 24, 1049)
            nn.Conv2d(
                in_channels=2,      # input height
                out_channels=16,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),      # output shape (16, 24, 1049)
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (16, 12, 524)
        )
        # self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
        #     nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
        #     nn.ReLU(),  # activation
        #     nn.MaxPool2d(2),  # output shape (32, 7, 7)
        # )
        self.out = nn.Linear(16*12*524, N_STATES)   # fully connected layer, output 10 classes
        self.out.weight.data.normal_(0, 0.1)
        # for layer in [self.fc1, self.out]:
        #     nn.init.normal_(layer.weight, mean=0., std=0.1)
        #     nn.init.constant_(layer.bias, 0.)
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 16, 24, 1049)
        out_put = self.out(x)
        return out_put

class DCQN(object):
    def __init__(self):
        self.eval_net,self.target_net = CNet(),CNet()
        self.eval_net.to(device)
        self.target_net.to(device)
        self.learn_step_counter = 0     # 用于 target 更新计时
        self.memory_counter = 0         # 记忆库记数
        self.memory = [0 for i in range(MEMORY_CAPACITY)]       # 初始化记忆库
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # torch 的优化器
        self.loss_func = nn.MSELoss()   # 误差公式
        self.loss_list = []

    def store_transition(self, s, a, r , s_):
        # 状态s和下一状态s_列表化,列表索引 = 路段号 - 1
        s1_list = self.deal_raw_status(s)
        s2_list = self.deal_raw_status(s_)
        transition = (s1_list, a, r, s2_list)
        # transition = np.hstack((s1_list, np.array([[a, r]]), s2_list))
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % MEMORY_CAPACITY
        # print(transition.shape)
        self.memory[index] = transition
        self.memory_counter += 1

    def learn(self):
        # 记忆库满了才进行学习，不满直接返回
        if self.memory_counter <= MEMORY_CAPACITY:
            return
        # target net 参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = [self.memory[index] for index in sample_index]
        b_s,b_a,b_r,b_s_ = [], [], [], []
        for memory in b_memory:
            b_s.append(memory[0])
            b_a.append([memory[1]])
            b_r.append(memory[2])
            b_s_.append(memory[3])
        b_s = torch.FloatTensor(b_s).to(device)
        b_a = torch.LongTensor(np.array(b_a).astype(int)).to(device)
        b_r = torch.FloatTensor(b_r).to(device)
        b_s_ = torch.FloatTensor(b_s_).to(device)
        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        # print(self.eval_net(b_s).shape,b_a.shape)
        q_eval = self.eval_net(b_s).gather(1, b_a).to(device)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach().to(device)     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1).to(device)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        # print("损失函数值：",loss.cpu().detach().numpy())
        self.loss_list.append(loss.cpu().detach().numpy())
        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    # 当前状态(路段号，日期时间)、待选订单、路段-区域列表
    def choose_action(self, status, req_list):
        # 可选择的订单中包含的路段列表
        road_useful = []
        for req in req_list:
            if req[10] - 1 not in road_useful:
                road_useful.append(req[10] - 1)
        # 状态处理成三维矩阵(是否工作日，时刻，路段)
        state_list = self.deal_raw_status(status)
        
        x = torch.unsqueeze(torch.FloatTensor(state_list), 0).to(device)
        # EPSILON的可能是强化选择，其他可能是随机
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x).cpu().data.numpy()[0]
            # 求网络输出最大概率的那项
            sel_area_pos, max_prob = None, actions_value[road_useful[0]]-1
            for item in road_useful:
                if actions_value[item] > max_prob:
                    sel_area_pos = item
                    max_prob = actions_value[item]
            # 从所有订单中选出属于网络输出的路段号的订单，从中选择金额最大的订单
            max_action = req_list[0]
            for item in req_list:
                if item[10] == sel_area_pos and item[2] > max_action[2]:
                    max_action = item
            action = max_action
        else:
            rand = np.random.randint(0, len(req_list))
            action = req_list[rand]
        return action

    def deal_raw_status(self, raw_status):
        state_np = np.zeros((2,24,1049))
        raw_date = raw_status[1].strftime("%w")
        deal_date = 0
        # 判断是否是休息日, 是则记为1
        if int(raw_date) > 5:
            deal_date = 1
        raw_time = raw_status[1].strftime("%H")
        deal_time = int(raw_time) - 1 
        deal_road = raw_status[0] - 1
        state_np[deal_date][deal_time][deal_road] = 1
        return state_np