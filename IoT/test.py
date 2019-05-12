from models import area, sensor
import matplotlib.pyplot as plt

area_obj = area.Area()
area_obj.standardSensor()
# area_obj.randomSensor()
area_obj.backupSensor()
area_obj.buildBarrier()
day_num = 5

plt.close()  #clf() # 清图  cla() # 清坐标轴 close() # 关窗口
# fig=plt.figure()
plt.xlim((0, 1000))
plt.ylim((0, 500))
plt.ion()  #interactive mode on

area_obj.drawState(plt)
for dn in range(day_num):
    # 存在传感器能量耗尽
    if area_obj.dayByDay() is False:
        area_obj.drawState(plt)
        print("检测到漏洞！")
        area_obj.repairBarrier()
    area_obj.drawState(plt)
    print("第%s天状态绘制完毕" % dn)
# plt.pause(3) 
plt.ioff()
plt.show()