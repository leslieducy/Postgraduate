"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""

import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
# import gym
import os
os.environ["OMP_NUM_THREADS"] = "1"
import cx_Oracle as cx      #导入模块
import numpy as np
import datetime

UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
MAX_EP = 300

# env = gym.make('CartPole-v0')
# N_S = env.observation_space.shape[0]
# N_A = env.action_space.n

# env = gym.make('CartPole-v0')
N_S = 1049
N_A = 8
DATA = [2013,5,4]
FINISHED_LIST = []
LOSS_LIST = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    # 包含一个Actor（输出（连续：动作空间的概率密度函数；离散问题：所有动作的可能性））和一个Critic（状态值）
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 200)
        self.pi2 = nn.Linear(200, a_dim)
        self.v1 = nn.Linear(s_dim, 100)
        self.v2 = nn.Linear(100, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = F.relu6(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = F.relu6(self.v1(x))
        values = self.v2(v1)
        return logits, values
    # Actor基于概率选行为
    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        # m = self.distribution(prob)
        # return m.sample().numpy()[0]
        return prob[0]
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


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)           # local network
        self.ROAD_AREA = self.init_area()
    def choose_action(self, x, time, con):
        end_road_list = self.query_has_action(x, time,con)
        if end_road_list is None or len(end_road_list) == 0:
            return None
        area_useful = []
        for item in end_road_list:
            if self.ROAD_AREA[item[1]] not in area_useful:
                area_useful.append(self.ROAD_AREA[item[1]])
        # input only one sample
        if np.random.uniform() < 0.9:   # greedy
            actions_value = self.lnet.choose_action(v_wrap(np.array(x)[None, :]))
            # actions_value = self.lnet.forward(x).data.numpy()[0]  
            sel_area_pos, max_prob = None, actions_value[area_useful[0]]-1
            for item in area_useful:
                if actions_value[item] > max_prob:
                    sel_area_pos = item
                    max_prob = actions_value[item]
            
            road_all = [i for i,v in enumerate(self.ROAD_AREA) if v==(sel_area_pos+1)]
            max_action = end_road_list[0]
            for road_item in road_all:
                for item in end_road_list:
                    if item[1] == road_item and item[2] > max_action[2]:
                        max_action = item
            action = max_action
        else:   # random
            rand = np.random.randint(0, len(end_road_list))
            action = end_road_list[rand]
        return action
    def query_has_action(self, x, time, con):
        road_no = list(x).index(1)
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
                    # id，end_road，money
                    end_road_list.append((item[0],item[10],item[2]))
                    
            # print(len(end_road_list))
            if len(end_road_list)>1:
                break
        return end_road_list
    def get_step_state(self, a, time, con):
        if a == None:
            return None, None, ''
        cursor = con.cursor()       #创建游标
        cursor.execute("select * from TEM_REQ t where REQUEST_ID = '" + str(a)+"'")  #执行sql语句
        data = cursor.fetchone()       #获取一条数据
        cursor.close()  #关闭游标
        if data:
            s_no, exp_time, rm = data[10], data[4], data[2]
            s_ = [-1] * N_S
            s_[s_no - 1] = 1
            r = 0
            end_road_list = self.query_has_action(s_,time,con)
            if end_road_list is None:
                return None, rm, exp_time,r
            # 当前收益越大，reward越大
            r = 1 if rm > 10000 else rm/10000
            return s_, rm, exp_time, r
        else:
            print("没有数据")
            return None, None, ''
    def get_init_state(self, con):
        start_road = 409
        state_list = [-1] * N_S
        state_list[start_road] = 1
        time = [0,9,0]
        end_road_list = self.query_has_action(state_list,time,con)
        for item in end_road_list:
            if item[1] == start_road+1:
                continue
            state_list[item[1]-1] = 0
        return state_list,time
    def init_area(self):
        con = cx.connect('test', 'herron', '127.0.0.1:1521/TestDatabase')  #创建连接
        cursor = con.cursor()       #创建游标
        cursor.execute("select distinct ROAD_ID,CLUSTER_TYPE from ROAD_PROP t ")  #执行sql语句
        data_list = cursor.fetchall()
        cursor.close()  #关闭游标
        con.close()
        ret_list = [0]*1050
        for (road_id,cluster_type) in data_list:
            ret_list[road_id] = cluster_type
        return ret_list
    def run(self):
        total_step = 1
        con = cx.connect('test', 'herron', '127.0.0.1:1521/TestDatabase')  #创建连接
        while self.g_ep.value < MAX_EP:
            FINISHED_LIST = []
            s, time = self.get_init_state(con)
            money = 0.
            buffer_s, buffer_a, buffer_r = [], [], []
            # ep_r = 0.
            while True:
                # 选动作
                q_a = self.choose_action(s, time, con)
                # 没有找到动作请求, 进入下回合
                if q_a is None:
                    break
                FINISHED_LIST.append(q_a[0])
                # 得到环境反馈
                area = self.ROAD_AREA[q_a[1]]
                s_, rm, exp_time,r = self.get_step_state(q_a[0], time, con)
                money += rm
                
                done = False
                # 没有找到下一个动作, 进入下回合
                if s_ is None:
                    done = True
                    # break
                # print(s_,r)
                exp_time = exp_time.split(':')
                time = [int(x) for x in exp_time]
                # ep_r += r
                buffer_a.append(area)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    s_ = np.array(s_)
                    buffer_a = np.array(buffer_a)
                    buffer_r = np.array(buffer_r)
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, money, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)
        con.close()


if __name__ == "__main__":
    gnet = Net(N_S, N_A)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=0.0001)      # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    
    # con = cx.connect('test', 'herron', '127.0.0.1:1521/TestDatabase')  #创建连接
    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count()-1)]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]
    # con.close()     #关闭数据库连接
    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()

