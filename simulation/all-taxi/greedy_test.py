from models import reqday, car
import datetime as dati
import cx_Oracle as cx      #导入模块

import matplotlib.pyplot as plt
import numpy as np
# 司机选择订单的属性（0-4）（随机，贪心，评估，强化）
SELECT_TYPE = 1

con = cx.connect('test', 'herron', '127.0.0.1:1521/TestDatabase')  #创建连接
# 初始化时间状态及订单发布平台
start_datatime = dati.datetime.strptime('2013-05-04 00:05:00', '%Y-%m-%d %H:%M:%S')
now_datatime = start_datatime
reqday = reqday.Reqday(start_datatime,con)
# print(start_datatime.strftime("%Y-%m-%d"))

# 初始化所有出租车
car_all = []
cursor = con.cursor()       #创建游标
cursor.execute("select * from taximan t where DAY_NO = '%s'" % start_datatime.strftime("%Y-%m-%d"))
car_data_list = cursor.fetchall()       #获取所有数据
cursor.close()
car_all = [car.Car(car_data,con) for car_data in car_data_list]

# 时间开始循环，直至下一天
while now_datatime.strftime("%Y-%m-%d") == start_datatime.strftime("%Y-%m-%d"):
    for car in car_all:
        car.getStaus(now_datatime, reqday, SELECT_TYPE)
        
    print(str(now_datatime)+"贪心匹配结束")
    now_datatime += dati.timedelta(minutes=1)

con.close()
# 所有司机收入图
mon_plt = []
for car in car_all:
    mon_plt.append(car.income)
print("Greedy:",np.mean(mon_plt))
x = range(0,len(mon_plt),1)
plt.figure(figsize=(8,6), dpi=80)
plt.title("Greedy")
plt.xlabel("train(reward)")
plt.ylabel("n") 
plt.plot(x, mon_plt, label="precision")
plt.legend(loc='upper right')
# plt.ylim(0,100)
plt.show()

