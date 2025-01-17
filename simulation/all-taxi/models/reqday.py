import cx_Oracle as cx      #导入模块
import datetime as dati
from . import road
import numpy as np
import random
# 失效时间间隔
TIME_INTERVAL = 5
class Reqday(object):
    def __init__(self, start_datatime, con):
        self.dbcon = con
        self.day_no = start_datatime.strftime("%Y-%m-%d") 
        # cursor = self.dbcon.cursor()       #创建游标
        # cursor.execute("select * from TEM_REQ t where DAY_NO = "+str(self.day_no)) 
        # req_list = cursor.fetchall()       #获取全部数据
        # cursor.close()
        # # 当天所有请求的id
        # self.all_req = [str(x[0]) for x in req_list]
        self.over_req = []

    # 已完成订单记录
    def overReq(self,req_id):
        self.over_req.append(str(req_id))

    # 假设订单5分钟后立即失效。获取前5分钟内周边全部未被匹配的订单。
    # road是Road对象
    def getReq(self, road, datatime):
        now_time = datatime
        before_time = now_time - dati.timedelta(minutes=TIME_INTERVAL)
        neigh_road_list = road.getAllRoad()
        cursor = self.dbcon.cursor()       #创建游标
        # range(0,len(self.over_req+))
        nei_road_param = ','.join(":%d" % i for i,_ in enumerate(neigh_road_list))
        sql = ("select * from TEM_REQ t where on_time > '"+ before_time.strftime("%H:%M:%S") 
            + "' and on_time < '"+ now_time.strftime("%H:%M:%S")
            +"' and day_no = '" + now_time.strftime("%Y-%m-%d") 
            +"' and start_road in ('',%s)" % nei_road_param)
        cursor.execute(sql, neigh_road_list)  #执行sql语句
        req_all = cursor.fetchall()
        cursor.close()
        ret_req_all = []
        for req in req_all:
            # 该概率表示寻找的机会
            if str(req[0]) not in self.over_req:
                if random.randint(1,10)>5:
                    ret_req_all.append(req)
                else:
                    self.overReq(req[0])
                    
        # print("req_all",req_all)
        # print(ret_req_all)
        return ret_req_all
