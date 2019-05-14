from models import area, sensor
import matplotlib.pyplot as plt

radius_list = list(range(10, 250, 1))
# angle_list = range(0, -180, -5)
# RADIUS =  100
ANGLE = -30

result_list = []
# 实验次数 ，用来计算修复的成功率 p = 成功修复次数/总次数。 
test_num = 100

for RADIUS in radius_list:
    success_num = 0
    for number in range(test_num):
        area_obj = area.Area(RADIUS, ANGLE)
        area_obj.standardSensor()
        area_obj.backupRandomSensor()
        area_obj.buildBarrier()

        day_num = 5
        success_tag = True
        for dn in range(day_num):
            # 存在传感器能量耗尽
            if area_obj.dayByDay() is False:
                print("检测到漏洞！")
                success_tag = area_obj.repairBarrier()
                break
        if success_tag:
            success_num += 1
    # 计算成功率
    print("半径:",RADIUS)
    print("修补成功率:",success_num/test_num)
    # print("半径",RADIUS,"的", success_num,"修补成功率！" )
    result_list.append(success_num/test_num)

plt.plot(radius_list, result_list)
plt.title('Radius-P')
plt.xlabel('Radius')
plt.ylabel('P')
plt.show()





