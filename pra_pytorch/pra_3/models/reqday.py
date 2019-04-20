import cx_Oracle as cx      #导入模块
import datetime as dati
from . import road
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
    # 假设订单5分钟后立即失效。获取前5分钟内全部未被匹配的订单。
    # road是Road对象
    def getReq(self, road, datatime):
        now_time = datatime
        before_time = now_time - dati.timedelta(minutes=TIME_INTERVAL)
        neigh_road_list = road.getAllRoad()
        cursor = self.dbcon.cursor()       #创建游标
        # range(0,len(self.over_req+))
        nei_road_param = ','.join(":%d" % i for i,_ in enumerate(neigh_road_list))
        # req_id_param = ','.join(":%d" % (len(nei_road_param)+i) for i,_ in enumerate(self.over_req))
        # print(req_id_param,nei_road_param)
        sql = ("select * from TEM_REQ t where on_time > '"+ before_time.strftime("%H:%M:%S") 
            + "' and on_time < '"+ now_time.strftime("%H:%M:%S")
            +"' and day_no = '" + now_time.strftime("%Y-%m-%d") 
            +"' and start_road in ('',%s)" % nei_road_param)
        # if len(nei_road_param) < 1:
        #     return
        # elif len(req_id_param) > 0:
        #     sql += (" and request_id not in (%s)" % req_id_param)
        # print(sql)
        # param_list = neigh_road_list + self.over_req
        # print(param_list)
        cursor.execute(sql, neigh_road_list)  #执行sql语句
        req_all = cursor.fetchall()
        cursor.close()
        ret_req_all = []
        for req in req_all:
            if req[0] not in self.over_req:
                ret_req_all.append(req)
        # print("req_all",req_all)
        return ret_req_all
