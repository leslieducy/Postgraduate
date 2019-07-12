import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

import numpy as np
import datetime

UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
MAX_EP = 300

N_STATES = 1049
N_ACTIONS = 8
MEMORY_CAPACITY = 1000      # 记忆库大小
GAMMA = 0.995                # 奖励递减参数
BATCH_SIZE = 512
LR = 0.005                   # learning rate
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)
    
class ACNet(nn.Module):
    # 包含一个Actor（输出（连续：动作空间的概率密度函数；离散问题：所有动作的可能性））和一个Critic（状态值）
    def __init__(self, s_dim=N_STATES, a_dim=N_ACTIONS):
        super(ACNet, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 200).to(device)
        self.pi2 = nn.Linear(200, a_dim).to(device)
        self.v1 = nn.Linear(s_dim, 100).to(device)
        self.v2 = nn.Linear(100, 1).to(device)
        for layer in [self.pi1, self.pi2, self.v1, self.v2]:
            nn.init.normal_(layer.weight, mean=0., std=0.1)
            nn.init.constant_(layer.bias, 0.)
        self.distribution = torch.distributions.Categorical

        self.loss_list = []
        self.learn_step_counter = 0     # 用于 target 更新计时
        self.memory_counter = 0         # 记忆库记数
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # 初始化记忆库

    def forward(self, x):
        pi1 = F.relu6(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = F.relu6(self.v1(x))
        values = self.v2(v1)
        return logits, values

    # Actor基于概率选行为
    def choose_action(self, road_id, req_list, road_area):
        state_list = [-1 for i in range(N_STATES)]
        # 列表索引 = 路段号 - 1（road_area与state_list设计相同）
        state_list[road_id-1] = 1
        s = torch.unsqueeze(torch.FloatTensor(state_list), 0).to(device)

        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        # m是按prob概率分布的空间
        m = self.distribution(prob)
        # 取按m实例的第一个数
        sel_area_pos = m.sample().cpu().numpy()[0]
        # print("sel_area_pos",sel_area_pos)
        area_useful = []
        for req in req_list:
            if road_area[req[10] - 1] not in area_useful:
                area_useful.append(road_area[req[10] - 1])
        if sel_area_pos not in area_useful:
            return int(sel_area_pos)
        # 记录选中区域的所有路段号
        road_all = [i+1 for i,v in enumerate(road_area) if v==sel_area_pos]
        # 选出个直接收益最大的请求
        max_action = req_list[0]
        for road_item in road_all:
            for item in req_list:
                if item[10] == road_item and item[2] > max_action[2]:
                    max_action = item
        action = max_action
        # print("action",action)
        return action

    # Actor根据Critic的评分修改选行为的概率
    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


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
        opt = torch.optim.Adam(self.parameters(), lr=LR)    # torch 的优化器      # global optimizer
        # # target net 参数更新
        # if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
        #     self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES]).to(device)
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]).to(device)

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        # q_eval = self.eval_net(b_s).gather(1, b_a).to(device)  # shape (batch, 1)
        # q_next = self.target_net(b_s_).detach().to(device)     # detach from graph, don't backpropagate
        # q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1).to(device)   # shape (batch, 1)
        v_s_ = self.forward(b_s)[-1].gather(1, b_a).to(device)  # shape (batch, 1)
        # buffer_v_target = []
        # for r in b_r[::-1]:    # reverse buffer r
        #     v_s_ = r + GAMMA * v_s_
        #     buffer_v_target.append(v_s_)
        # buffer_v_target.reverse()
        q_next = self.forward(b_s_)[-1].detach().to(device)     # detach from graph, don't backpropagate
        buffer_v_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1).to(device)   # shape (batch, 1)
        loss = self.loss_func(b_s,b_a,buffer_v_target)
            
        # print("损失函数值：",loss.cpu().detach().numpy())
        self.loss_list.append(loss.cpu().detach().numpy())
        # calculate local gradients 
        opt.zero_grad()
        loss.backward()
        opt.step()