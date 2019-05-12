import numpy as np
import matplotlib.pyplot as plt
from . import sensor
import math
import random
# import pysnooper

class Area():
    # 初始化区域
    def __init__(self,RADIUS,ANGLE):
        self.width = 1000
        self.height = 500
        self.status_list = np.zeros((self.height, self.width))
        self.backup_ss_list = []
        self.invalid_ss_list = []
        self.ss_list = []
        self.die_ss = False
        self.guard_line = 200
        self.RADIUS = RADIUS
        self.ANGLE = ANGLE

    # 标准传感器分布
    def standardSensor(self):
        i,j = self.RADIUS * math.sin(math.pi * (abs(self.ANGLE)/2)/180),200
        # 标准传感器分布，两个传感器间的距离d=2*r*sin(a/2)，离标准线的高h=r*cos(a/2)，构造标准分布的偏转角b=90-a/2
        d = int(2 * self.RADIUS * math.sin(math.pi * (abs(self.ANGLE)/2)/180))
        h = int(self.RADIUS * math.cos(math.pi * (abs(self.ANGLE)/2)/180))
        # 随机从20-100产生能量值,从中选出一个能量设为10（即产生的空洞）
        energy_num = int(self.width/d)
        x = random.randint(0, energy_num)
        for n in range(energy_num):
            energy = random.randint(20,100)
            if n == x:
                energy = 10
            self.ss_list.append(sensor.Sensor(i, j+h, -(90-abs(self.ANGLE)/2), energy, self.RADIUS, self.ANGLE))
            i += d

    # 随机生成备用传感器
    def backupRandomSensor(self, num=3):
        # x_list = [200, 350, 620]
        # y_list = [320, 150, 160]
        for i in range(num):
            x = random.randint(self.RADIUS, self.width-self.RADIUS)
            y = random.randint(self.RADIUS, self.height-self.RADIUS)
            self.backup_ss_list.append(sensor.Sensor(x, y, np.random.randint(-180,180), self.RADIUS, self.ANGLE))

    # 生成备用传感器
    def backupSensor(self, num=3):
        x_list = [200, 350, 620]
        y_list = [320, 150, 160]
        for i in range(num):
            self.backup_ss_list.append(sensor.Sensor(x_list[i], y_list[i], np.random.randint(-180,180), self.RADIUS, self.ANGLE))

    # 构建栅栏
    def buildBarrier(self):
        build_list = []
        for ss in self.ss_list:
            ret_list = ss.coverSize()
            build_list.extend(ret_list)
        # print(build_list)
        for (x, y) in build_list:
            if x > 0 and x < self.width and y < self.height and y > 0:
                self.status_list[y][x] += 1
        # print(self.status_list)

    # 每天所有传感器的消耗
    def dayByDay(self):
        for ss in self.ss_list:
            ss.dissipation()
            if not ss.isEnoughEnergy():
                print(ss.x,ss.y,ss.energy)
                for (x, y) in ss.coverSize():
                    if x > 0 and x < self.width and y < self.height and y > 0:
                        self.status_list[y][x] -= 1
                self.die_ss = ss
                self.ss_list.remove(ss)
                return False
        return True
            
        # print(self.ss_list[0].energy)

    # 画出当前状态 
    def drawState(self,plt):
        # x1,y1表示检测范围内，x2,y3代表未检测范围，x3,y3表示传感器
        x1,y1,x2,y2,x3,y3,x4,y4 = [],[],[],[],[],[],[],[]
        for x in range(0, self.width):
            for y in range(0, self.height):
                if self.status_list[y][x] > 0:
                    x1.append(x)
                    y1.append(y)
                else:
                    x2.append(x)
                    y2.append(y)
        for ss in self.ss_list:
            x3.append(ss.x)
            y3.append(ss.y)
        for bss in self.backup_ss_list:
            x4.append(bss.x)
            y4.append(bss.y)
        plt.ylim(0, 1000)
        plt.scatter(x1,y1,marker='x',color = 'red', s = 0.2 ,label = 'cover')
        plt.scatter(x2,y2,marker='o', color = 'white', s = 0.2, label = 'blank')
        plt.scatter(x3,y3,marker='o', color = 'blue', s = 1.8, label = 'sensor')
        plt.scatter(x4,y4,marker='o', color = 'green', s = 1.8, label = 'backup-sensor')
        plt.pause(0.1)
        

    # 捕获漏洞
    # def catchLeak(self):
    #     status_max = np.max(self.status_list, axis = 1)

    # 修补栅栏
    def repairBarrier(self):
        # 定位漏洞位置
        left, right = self.width, 0
        for (x, y) in self.die_ss.coverSize():
            if y == self.guard_line:
                if x < left:
                    left = x
                elif x > right:
                    right = x
        # 找到能移动旋转的传感器才跳出循环，success_tag是判断是否成功修复
        success_tag = False
        while(len(self.backup_ss_list) != len(self.invalid_ss_list)):
            # 距离左边最近的备用传感器
            min_dis_left, left_ss = math.pow(self.backup_ss_list[0].x - left, 2) + math.pow(self.backup_ss_list[0].y - self.guard_line, 2), self.backup_ss_list[0]
            for bss in self.backup_ss_list:
                # 如果当前传感器属于无效传感器则跳过
                if bss in self.invalid_ss_list:
                    continue
                dis =  math.pow(bss.x - left, 2) + math.pow(bss.y - self.guard_line, 2)
                if dis < min_dis_left:
                    min_dis_left, left_ss = dis, bss
            # 距离右边最近的备用传感器
            min_dis_right, right_ss = math.pow(self.backup_ss_list[0].x - right, 2) + math.pow(self.backup_ss_list[0].y - self.guard_line, 2), self.backup_ss_list[0]
            for bss in self.backup_ss_list:
                # 如果当前传感器属于无效传感器则跳过
                if bss in self.invalid_ss_list:
                    continue
                dis =  math.pow(bss.x - right, 2) + math.pow(bss.y - self.guard_line, 2)
                if dis < min_dis_right:
                    min_dis_right, right_ss = dis, bss
            # 寻找合适的备用传感器
            bss_sel, dis_max, point_max_belong = None, 0, -1
            if left_ss.isEqual(right_ss):
                bss_sel = left_ss
                if min_dis_left <= min_dis_right:
                    dis_max = min_dis_right
                    point_max_belong = 1
                else:
                    dis_max = min_dis_left
                    point_max_belong = 0

            elif min_dis_left <= min_dis_right:
                bss_sel = left_ss
                dis_max = math.pow(bss_sel.x - right, 2) + math.pow(bss_sel.y - self.guard_line, 2)
                point_max_belong = 1
            elif min_dis_right < min_dis_left:
                bss_sel = right_ss
                dis_max = math.pow(bss_sel.x - left, 2) + math.pow(bss_sel.y - self.guard_line, 2)
                point_max_belong = 0
            # 旋转或迁移
            if dis_max < left_ss.radius:
                rotate_a = bss_sel.rotate((left, self.guard_line))
                print("旋转%s度，修补漏洞" % rotate_a)
            else:
                rotate_a = bss_sel.move(((left,right)[point_max_belong], self.guard_line),point_max_belong)
                if rotate_a is False:
                    self.invalid_ss_list.append(bss_sel)
                    continue
                print("移动并旋转%s度，修补漏洞" % rotate_a)
            # 已寻找到合适的传感器，则可直接跳出循环
            success_tag = True
            break
        
        self.backup_ss_list.remove(bss_sel)
        self.ss_list.append(bss_sel)
        for (x, y) in bss_sel.coverSize():
            if x > 0 and x < self.width and y < self.height and y > 0:
                self.status_list[y][x] += 1
        return success_tag

        
