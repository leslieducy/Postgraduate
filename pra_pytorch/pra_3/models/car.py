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
        self.start_time = dati.datetime.strptime((self.day_no +" "+ car_data[1]), '%Y-%m-%d %H:%M:%S')
        self.next_road_id = self.start_road_id
        self.next_time = self.start_time
        self.wandering_num = 0
        self.income = 0
    # def __init__(self, id, day_no):
    #     self.id = id
    #     self.day_no = day_no
    #     self.finished = []
    #     con = cx.connect('test', 'herron', '127.0.0.1:1521/TestDatabase')  #创建连接
    #     cursor = con.cursor()       #创建游标
    #     cursor.execute("select * from taximan t where TAXIMAN_ID = "+str(self.id)+" and DAY_NO = "+str(self.day_no)) 
    #     car_data = cursor.fetchone()       #获取一条数据
    #     cursor.close()  
    #     con.close()     
    #     self.start_road_id = car_data[7]
    #     self.start_time = dati.datetime.strptime((self.day_no +" "+ car_data[1]), '%Y-%m-%d %H:%M:%S')
    #     self.next_road_id = self.start_road_id
    #     self.next_time = self.start_time
    #     self.wandering_num = 0
    #     self.income = 0
    #     # # 该车的状态（0,1）（空闲，工作）
    #     # self.status = 0
    # 每分钟都遍历一下状态，决定下一步动作
    def getStaus(self, datatime, reqday):
        if datatime < self.next_time:
            return
        print(self.next_road_id)
        now_road = road.Road(self.next_road_id, self.dbcon)
        req_all = reqday.getReq(now_road, datatime)
        # 判断是否有订单可选择,没有则游荡
        if len(req_all) > 0:
            req = self.randomSelect(req_all)
            self.accept_req(req, datatime)
            self.wandering_num = 0
            reqday.overReq(req[0])
        else:
            self.wandering_num += 1
            self.wandering(now_road)


    def randomSelect(self,req_all):
        return req_all[random.randrange(0,len(req_all))]

    def wandering(self,now_road):
        # 闲逛超过五分钟随机进入下一路段
        if self.wandering_num > 5:
            self.next_road_id = now_road.getRandomNeighbor()
    def accept_req(self, req, datatime):
        # self.status = 1
        self.income += req[2]
        self.next_road_id = req[10]
        # 根据订单开始-完成时间进行模拟的时间处理
        now_time = datatime
        on_time = dati.datetime.strptime(self.day_no +" "+ req[3], '%Y-%m-%d %H:%M:%S')
        off_time = dati.datetime.strptime(self.day_no +" "+ req[4], '%Y-%m-%d %H:%M:%S')
        self.next_time = now_time + (off_time - on_time)
        print("self.next_time",self.next_time)
    