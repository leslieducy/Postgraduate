import cx_Oracle as cx      #导入模块
import datetime as dati
from . import road
import random
class Car(object):
    def __init__(self, car_data, con):
        self.id = id
        self.dbcon = con
        self.day_no = str(car_data[3])
        self.finished = []
        self.start_road_id = car_data[7]
        self.start_time = dati.datetime.strptime((self.day_no + car_data[1]), '%Y-%m-%d%H:%M:%S')
        self.next_road_id = self.start_road_id
        self.next_time = self.start_time
        self.wandering_num = 0
        self.income = 0
    
    # 每分钟都遍历一下状态，决定下一步动作
    def getStaus(self, datatime, reqday, sel_type=0, nn=None):
        # 正在完成订单中，跳过状态检测
        if datatime < self.next_time:
            return
        # 判断是否在休息中，工作8小时后开始休息，跳过状态检测
        if datatime > self.start_time + dati.timedelta(hours=8):
            return
        now_road = road.Road(self.next_road_id, self.dbcon)
        req_all = reqday.getReq(now_road, datatime)
        # 司机根据不同的策略进行操作
        if sel_type == 0:
            self.randomSelect(req_all, datatime, reqday, now_road)
        elif sel_type == 1:
            self.greedySelect(req_all, datatime, reqday, now_road)
        elif sel_type == 2:
            self.evaluateSelect(req_all, datatime, reqday, now_road)
        elif sel_type == 3:
            self.dqnSelect(req_all, datatime, reqday, now_road, nn)
        elif sel_type == 4:
            self.acSelect(req_all, datatime, reqday, now_road, nn)

    # 随机选择:从当前路段周围的订单中随机选择
    def randomSelect(self, req_all, datatime, reqday, now_road):
        # 判断是否有订单可选择,没有则游荡
        if len(req_all) > 0:
            req = req_all[random.randrange(0,len(req_all))]
            self.accept_req(req, datatime)
            self.wandering_num = 0
            reqday.overReq(req[0])
        else:
            self.wandering_num += 1
            # 闲逛超过五分钟随机进入下一路段
            if self.wandering_num > 5:
                self.next_road_id = now_road.getRandomNeighbor()

    # 贪心选择：从当前最高的前两个订单中选择,并朝收益最大区域的路游荡
    def greedySelect(self, req_all, datatime, reqday, now_road):
        # 判断是否有订单可选择,没有则游荡
        if len(req_all) > 0:
            pre_sel_list = []
            if len(req_all) < 3:
                pre_sel_list = req_all
            else:
                req_all.sort(key=lambda x:x[2], reverse=True)
                pre_sel_list = req_all[0:3]
            req = pre_sel_list[random.randrange(0,len(pre_sel_list))]
            self.accept_req(req, datatime)
            self.wandering_num = 0
            reqday.overReq(req[0])
        else:
            self.wandering_num += 1
            # 闲逛超过五分钟选择潜在收益最大的下一路段
            if self.wandering_num > 5:
                self.next_road_id = now_road.getGreedyNeighbor()

    # 函数评估：评估订单(y=a*x1+(1-a)*x2)，选择评价值高的订单,并朝收益最大区域的路游荡
    def evaluateSelect(self, req_all, datatime, reqday, now_road):
        # 判断是否有订单可选择,没有则游荡
        if len(req_all) > 0:
            pre_sel_list = []
            alpha = 0.618
            hide_dict = {
                6:3261.303620,
                2:2644.065908,
                0:1981.736600,
                4:1557.090526,
                3:885.348000,
                5:599.893800,
                7:424.014600,
                1:365.509620,
            }
            for req in req_all:
                cursor = self.dbcon.cursor()       #创建游标
                cursor.execute("select CLUSTER_TYPE from ROAD_PROP t where ROAD_ID = "+str(req[10])) 
                cluster_type = cursor.fetchone()       #获取全部数据
                cursor.close()
                if cluster_type is None:
                    continue
                fun_y = alpha*req[2] + (1-alpha)*hide_dict[cluster_type[0]]
                pre_sel_list.append((req,fun_y))
            pre_sel_list.sort(key=lambda x:x[1], reverse=True)
            # 未分区域没有则随便选择,小于3个取最大，多于三个在前三个中选
            if len(pre_sel_list) < 1:
                req = req_all[random.randrange(0,len(req_all))]
            elif len(pre_sel_list) < 3:
                req = pre_sel_list[0][0]
            else:
                three_temp = pre_sel_list[0:3]
                req = three_temp[random.randrange(0,len(three_temp))][0]

            self.accept_req(req, datatime)
            self.wandering_num = 0
            reqday.overReq(req[0])
        else:
            self.wandering_num += 1
            # 闲逛超过五分钟随机进入下一路段
            if self.wandering_num > 5:
                self.next_road_id = now_road.getGreedyNeighbor()

    # dqn强化学习
    def dqnSelect(self, req_all, datatime, reqday, now_road, dqn):
        # 判断是否有订单可选择,没有则游荡
        if len(req_all) > 0:
            # 根据聚类区域学习,选择订单
            road_area = now_road.init_area()
            req = dqn.choose_action(self.next_road_id, req_all, road_area)
            # 选择的区域不存在订单
            if isinstance(req, int):
                # 存储当前路段号、订单所属区域、回报、下一个路段号
                dqn.store_transition(self.next_road_id, req, 0, self.next_road_id)
                dqn.learn()
                # 没有选择则重新选择
                req = dqn.choose_action(self.next_road_id, req_all, road_area)
                return

            # 当前收益越大，reward越大
            r = req[2]
            # 存储当前路段号、订单所属区域、回报、下一个路段号
            dqn.store_transition(self.next_road_id, road_area[req[10]-1], r, req[10])
            # rm = req[2]
            # r = 1 if rm > 10000 else rm/10000
            dqn.learn()

            self.accept_req(req, datatime)
            self.wandering_num = 0
            reqday.overReq(req[0])
        else:
            self.wandering_num += 1
            # 闲逛超过五分钟随机进入下一路段
            if self.wandering_num > 5:
                self.next_road_id = now_road.getGreedyNeighbor()

    # ac强化学习
    def acSelect(self, req_all, datatime, reqday, now_road, acn):
        # 判断是否有订单可选择,没有则游荡
        if len(req_all) > 0:
            # 根据聚类区域学习,选择订单
            road_area = now_road.init_area()
            req = acn.choose_action(self.next_road_id, req_all, road_area)
            # 选择的区域不存在订单
            while isinstance(req, int):
                # 存储当前路段号、订单所属区域、回报、下一个路段号
                acn.store_transition(self.next_road_id, req, 0, self.next_road_id)
                acn.learn()
                # 没有选择则重新选择
                req = acn.choose_action(self.next_road_id, req_all, road_area)
                return

            # 当前收益越大，reward越大
            r = req[2]
            # 存储当前路段号、订单所属区域、回报、下一个路段号
            acn.store_transition(self.next_road_id, road_area[req[10]-1], r, req[10])
            # rm = req[2]
            # r = 1 if rm > 10000 else rm/10000
            acn.learn()

            self.accept_req(req, datatime)
            self.wandering_num = 0
            reqday.overReq(req[0])
        else:
            self.wandering_num += 1
            # 闲逛超过五分钟随机进入下一路段
            if self.wandering_num > 5:
                self.next_road_id = now_road.getGreedyNeighbor()
        

    def accept_req(self, req, datatime):
        # self.status = 1
        self.income += req[2]
        self.next_road_id = req[10]
        # 根据订单开始-完成时间进行模拟的时间处理
        now_time = datatime
        on_time = dati.datetime.strptime(self.day_no + req[3], '%Y-%m-%d%H:%M:%S')
        off_time = dati.datetime.strptime(self.day_no + req[4], '%Y-%m-%d%H:%M:%S')
        self.next_time = now_time + (off_time - on_time)
        # print("self.next_time",self.next_time)
    