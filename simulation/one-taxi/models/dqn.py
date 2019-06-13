import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
# 超参数
BATCH_SIZE = 32
LR = 0.001                   # learning rate
EPSILON = 0.99               # 最优选择动作百分比
GAMMA = 0.9                # 奖励递减参数
TARGET_REPLACE_ITER = 100   # Q 现实网络的更新频率
MEMORY_CAPACITY = 1000      # 记忆库大小
N_ACTIONS = 1049  # 动作总数
N_STATES = 1049   # 状态总数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(N_STATES, 500),
            nn.Sigmoid(),
            nn.Linear(500, 100),
            nn.Sigmoid(),
        )
        # self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(100, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)
    def forward(self, x):
        x = self.fc1(x)
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
        self.loss_list = []

    # # 选择没有订单的区域，则收益为0
    # def choose_action(self, road_id, req_list, road_area):
    #     area_useful = []
    #     for req in req_list:
    #         if road_area[req[10] - 1] not in area_useful:
    #             area_useful.append(road_area[req[10] - 1])
    #     state_list = [-1 for i in range(N_STATES)]
    #     # 列表索引 = 路段号 - 1（road_area与state_list设计相同）
    #     state_list[road_id-1] = 1
    #     x = torch.unsqueeze(torch.FloatTensor(state_list), 0).to(device)
    #     if np.random.uniform() < EPSILON:
    #         action_value = self.eval_net.forward(x).cpu()
    #         sel_area_pos = torch.max(action_value, 1)[1].data.numpy()[0]
    #         print("sel_area_pos：",sel_area_pos)
    #         if sel_area_pos not in area_useful:
    #             return int(sel_area_pos)
    #         # 记录选中区域的所有路段号
    #         road_all = [i+1 for i,v in enumerate(road_area) if v==sel_area_pos]
    #         # 选出个直接收益最大的请求
    #         max_action = req_list[0]
    #         for road_item in road_all:
    #             for item in req_list:
    #                 if item[10] == road_item and item[2] > max_action[2]:
    #                     max_action = item
    #         action = max_action
    #     else: # random
    #         rand = np.random.randint(0, len(req_list))
    #         action = req_list[rand]
    #     return action

    def store_transition(self, s, a, r , s_):
        # 状态s和下一状态s_列表化,列表索引 = 路段号 - 1
        s1_list = [-1 for i in range(N_STATES)]
        s1_list[s-1] = 1
        s2_list = [-1 for i in range(N_STATES)]
        s2_list[s_-1] = 1

        transition = np.hstack((s1_list, [a, r], s2_list))
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % MEMORY_CAPACITY
        # print(transition.shape)
        self.memory[index, :] = transition
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
        # print("损失函数值：",loss.cpu().detach().numpy())
        self.loss_list.append(loss.cpu().detach().numpy())
        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # 当前状态、待选订单、路段-区域列表
    def choose_action(self, status, req_list, road_area):
        area_useful = []
        for req in req_list:
            if road_area[req[10] - 1] not in area_useful:
                area_useful.append(road_area[req[10] - 1])
        state_list = [-1 for i in range(N_STATES)]
        # 列表索引 = 路段号 - 1（road_area与state_list设计相同）
        state_list[status-1] = 1
        x = torch.unsqueeze(torch.FloatTensor(state_list), 0).to(device)
        # EPSILON的可能是强化选择，其他可能是随机
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x).cpu().data.numpy()[0]
            # 求网络输出最大概率的那项
            sel_area_pos, max_prob = None, actions_value[area_useful[0]]-1
            for item in area_useful:
                if actions_value[item] > max_prob:
                    sel_area_pos = item
                    max_prob = actions_value[item]
            # 记录路段号的数组
            road_all = [i+1 for i,v in enumerate(road_area) if v==sel_area_pos]
            max_action = req_list[0]
            for road_item in road_all:
                for item in req_list:
                    if item[10] == road_item and item[2] > max_action[2]:
                        max_action = item
            action = max_action
        else:
            rand = np.random.randint(0, len(req_list))
            action = req_list[rand]
        return action