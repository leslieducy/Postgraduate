import math
import numpy as np

class Sensor():
    # 初始化 固有属性（角度大小、半径大小、能量剩余），空间属性（x坐标、y坐标、偏转角度带大小）
    def __init__(self, x, y, deflection, energy=80, radius=100, angle=-60):
        self.angle = angle
        self.radius = radius
        self.energy = energy
        self.init_energy = self.energy
        self.x = x
        self.y = y
        self.deflection = deflection
    # 返回覆盖的所有点
    def coverSize(self):
        ret_list = []
        for i in range(int(self.x-self.radius), int(self.x+self.radius)):
            for j in range(self.y-self.radius, self.y+self.radius):
                # 大于半径的都去除
                if math.pow(i-self.x,2)+math.pow(j-self.y,2) > math.pow(self.radius,2):
                    continue
                # 包含竖线
                if  i == self.x:
                    if (self.deflection > 0 and (j-self.y) > 0 and self.deflection+self.angle<90 and self.deflection>90):
                        ret_list.append((i,j))
                    elif (self.deflection < 0 and (j-self.y) < 0 and self.deflection+self.angle<-90 and self.deflection>-90):
                        ret_list.append((i,j))
                    continue
                kAngle = self.calGrad(self.x, i, self.y, j)
                # +2是用来误差处理的
                if kAngle+2 > self.deflection+self.angle and kAngle-2 < self.deflection:
                    ret_list.append((i,j))
        return ret_list
    
    # 正常耗能
    def dissipation(self, energy_size=1):
        self.energy -= energy_size

    # 检查剩余能量
    def isEnoughEnergy(self):
        if self.energy > 5:
            return True
        else:
            return False

    # 判断两个传感器相同
    def isEqual(self, ss):
        tag = True
        if self.x == ss.x:
            tag = False
        if self.y == ss.y:
            tag = False
        if self.energy == ss.energy:
            tag = False
        return tag

    # 旋转角度达到覆盖目的point(x,y)
    def rotate(self, point):
        now_a = self.calGrad(point[0], self.x, point[1], self.y)
        old_deflection = self.deflection
        if now_a > 0:
            self.deflection = now_a - 180
        elif now_a < 0:
            self.deflection = 180 + now_a
        else:
            self.deflection = 0
        rotate_a = self.deflection - old_deflection
        return rotate_a

    # 移动并旋转角度达到覆盖目的point(x,y),max_belong(0左1右),diss_unit(单位距离的能量消耗)
    def move(self, point, max_belong, diss_unit=0.5):
        now_a = self.calGrad(point[0], self.x, point[1], self.y)
        old_deflection = self.deflection
        if now_a > 0:
            self.deflection = now_a - 180
        elif now_a < 0:
            self.deflection = 180 + now_a
        else:
            self.deflection = 0
        if max_belong == 1:
            self.deflection -= self.angle
        x = self.radius * math.cos(math.pi * now_a/180)
        y = self.radius * math.sin(math.pi * now_a/180)
        # 能量消耗（距离*单位距离能量消耗）
        dissipation = math.sqrt(math.pow(point[0]+round(x) - self.x,2) + math.pow(point[1]+round(y) - self.y,2)) * diss_unit
        print("移动耗能大小为：",dissipation)
        if self.energy <  dissipation:
            print("该传感器能量不足以支撑转移，正在重新选择...")
            return False
        self.x = point[0] + round(x)
        self.y = point[1] + round(y)
        
        rotate_a = self.deflection - old_deflection
        print(rotate_a)
        return rotate_a

    # 已知两点计算斜率=>弧度=>角度
    def calGrad(self,x1,x2,y1,y2):
        k = (y2-y1)/(x2-x1)
        h = math.atan(k)
        a = math.degrees(h)
        if x2-x1 < 0:
            if a > 0:
                a = a - 180
            elif a < 0:
                a = 180 + a
        return a