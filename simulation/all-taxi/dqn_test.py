from models import reqday, car, dqn
import datetime as dati
import cx_Oracle as cx      #导入模块

import matplotlib.pyplot as plt
import numpy as np
# 司机选择订单的属性（0-3）（随机，贪心，评估，dqn强化，A3C强化）
SELECT_TYPE = 3

# 计时
START_T = dati.datetime.now()

con = cx.connect('test', 'herron', '127.0.0.1:1521/TestDatabase')  #创建连接

# 初始化时间状态及订单发布平台
start_datatime = dati.datetime.strptime('2013-05-04 00:05:00', '%Y-%m-%d %H:%M:%S')
now_datatime = start_datatime
reqday = reqday.Reqday(start_datatime,con)

# 初始化所有出租车
car_all = []
cursor = con.cursor()       #创建游标
cursor.execute("select * from taximan t where DAY_NO = '%s'" % start_datatime.strftime("%Y-%m-%d"))
car_data_list = cursor.fetchall()       #获取所有数据
cursor.close()
car_all = [car.Car(car_data,con) for car_data in car_data_list]

# 初始化DQN神经网络
dqn = dqn.DQN()
# 时间开始循环，直至下一天
while now_datatime.strftime("%Y-%m-%d") == start_datatime.strftime("%Y-%m-%d"):
    for car in car_all:
        car.getStaus(now_datatime, reqday, SELECT_TYPE, dqn)
        
    print(str(now_datatime)+" DQN匹配结束")
    now_datatime += dati.timedelta(minutes=1)
con.close()

# 计时
END_T = dati.datetime.now()
print("共花费%s秒" % str((END_T-START_T).seconds))

# 所有司机收入图
mon_plt = []
for car in car_all:
    mon_plt.append(car.income)
print("DQN:",np.mean(mon_plt))
print("已完成订单数:", len(reqday.over_req))

import pandas as pd
columns = ["income"]
test=pd.DataFrame(columns=columns, data=mon_plt)
test.to_csv('DQN.csv', encoding='utf-8')

x = range(0,len(mon_plt),1)
plt.figure(figsize=(8,6), dpi=80)
plt.title("DQN")
plt.subplot(1,2,1)
plt.plot(x, mon_plt, label="precision")
plt.xlabel("train(reward)")
plt.ylabel("money")
plt.subplot(1,2,2)
plt.plot(range(0,len(dqn.loss_list),1), dqn.loss_list, label="loss")
plt.xlabel("n")
plt.ylabel("value") 

plt.legend(loc='upper right')
# plt.ylim(0,100)
plt.show()
