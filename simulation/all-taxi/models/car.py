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
    def getStaus(self, datatime, reqday, sel_type=0):
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
            self.a3cSelect(req_all, datatime, reqday, now_road)

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

    # 强化学习
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
    