import cx_Oracle as cx      #导入模块
import random
class Road(object):
    def __init__(self, id, con):
        self.id = id
        self.dbcon = con
        cursor = con.cursor()       #创建游标
        print("Road",self.id)
        cursor.execute("select NEI_ROAD from NEIGHBOR t where PRI_ROAD = "+str(self.id)) 
        neigh_road_list = cursor.fetchall()       #获取全部数据
        cursor.close()
        self.neighbor = [str(x[0]) for x in neigh_road_list]
    # 随机获取一个相邻路段
    def getRandomNeighbor(self):
        return self.neighbor[random.randrange(0,len(self.neighbor))]
    # 获取相邻路段和本身路段
    def getAllRoad(self):
        neigh_road_list = self.neighbor
        neigh_road_list.append(str(self.id))
        return neigh_road_list
    def getID(self):
        return self.id