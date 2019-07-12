import cx_Oracle as cx      #导入模块
import random
class Road(object):
    def __init__(self, id, con):
        self.id = id
        self.dbcon = con
        cursor = con.cursor()       #创建游标
        cursor.execute("select NEI_ROAD from NEIGHBOR t where PRI_ROAD = "+str(self.id)) 
        neigh_road_list = cursor.fetchall()       #获取全部数据
        cursor.close()
        self.neighbor = [x[0] for x in neigh_road_list]

    # 随机获取一个相邻路段
    def getRandomNeighbor(self):
        return self.neighbor[random.randrange(0,len(self.neighbor))]
    # 朝有订单的路段移动
    def getHasOrderNeighbor(self, req_list):
        for road_id in self.neighbor:
            if road_id in req_list:
                return road_id

        return self.neighbor[random.randrange(0,len(self.neighbor))]
    # 贪心选取一个潜在收益大的相邻路段，区域排名（6>2>0>4>3>5>7>1）
    def getGreedyNeighbor(self):
        data_list = []
        for road_id in self.neighbor:
            cursor = self.dbcon.cursor()       #创建游标
            cursor.execute("select ROAD_ID,CLUSTER_TYPE from ROAD_PROP t where ROAD_ID = "+str(road_id)) 
            data = cursor.fetchone()       #获取全部数据
            # 单路口路段潜力最差，不选择
            if data is not None:
                data_list.append(data)
            cursor.close()
        area_max_min = [6,2,0,4,3,5,7,1]
        data_list.sort(key=lambda x : area_max_min.index(x[1]), reverse=False)
        ret_list = []
        ret_type = data_list[0][1]
        for data in data_list:
            if data[1] == ret_type:
                ret_list.append(data[0])
        return ret_list[random.randrange(0,len(ret_list))]
    # 返回所有路段的聚类区域列表
    def init_area(self):
        cursor = self.dbcon.cursor()       #创建游标
        cursor.execute("select distinct ROAD_ID,CLUSTER_TYPE from ROAD_PROP t ")  #执行sql语句
        data_list = cursor.fetchall()
        cursor.close()  #关闭游标
        ret_list = [0 for i in range(1049)]
        for (road_id,cluster_type) in data_list:
            ret_list[road_id-1] = cluster_type
        return ret_list
    # 获取相邻路段和本身路段
    def getAllRoad(self):
        neigh_road_list = self.neighbor
        neigh_road_list.append(self.id)
        return neigh_road_list
    def getID(self):
        return self.id