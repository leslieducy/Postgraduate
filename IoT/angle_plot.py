from models import area, sensor
import matplotlib.pyplot as plt

# radius_list = range(50, 500, 10)
# ANGLE = -60
angle_list = list(range(-5, -60, -1))
RADIUS =  100

result_list = []
# 实验次数 ，用来计算修复的成功率 p = 成功修复次数/总次数。 
test_num = 20

for ANGLE in angle_list:
    success_num = 0
    for number in range(test_num):
        area_obj = area.Area(RADIUS, ANGLE)
        area_obj.standardSensor()
        area_obj.backupRandomSensor()
        area_obj.buildBarrier()

        day_num = 20
        success_tag = True
        for dn in range(day_num):
            # 存在传感器能量耗尽
            if area_obj.dayByDay() is False:
                print("检测到漏洞！")
                success_tag = area_obj.repairBarrier()
                # 只要有一个不能修复则直接，修复失败
                if success_tag is False:
                    break
                # break
        if success_tag:
            success_num += 1
    # 计算成功率
    print("角度:",RADIUS)
    print("修补成功率:",success_num/test_num)
    result_list.append(success_num/test_num)
angle_list = [-x for x in angle_list]
plt.plot(angle_list, result_list)
plt.title('Angle-P')
plt.xlabel('Angle')
plt.ylabel('P')
plt.show()





