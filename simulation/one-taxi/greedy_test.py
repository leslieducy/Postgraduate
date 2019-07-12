from models import reqday, car
import datetime as dati
from utils import result
import cx_Oracle as cx      #导入模块

import matplotlib.pyplot as plt
import numpy as np

def simulate():
    # 司机选择订单的属性（0-3）（随机，贪心，评估，dqn强化，A3C强化）
    SELECT_TYPE = 1
    con = cx.connect('test', 'herron', '127.0.0.1:1521/TestDatabase')  #创建连接
    # 选择一辆出租车获取其一星期的原始数据
    cursor = con.cursor()       #创建游标
    cursor.execute("select * from taximan t where TAXIMAN_ID = '101' and DAY_NO in ('2013-03-01','2013-03-02','2013-03-03','2013-03-04','2013-03-05','2013-03-06','2013-03-07') order by DAY_NO")
    car_data_list = cursor.fetchall()       #获取所有数据
    cursor.close()
    car_all_status = [car.Car(car_data,con) for car_data in car_data_list]

    aim_car_list = []
    # 循环开始每一天(每辆车代表一天)
    for aim_car in car_all_status:
        start_datatime = aim_car.start_datatime
        end_datatime = aim_car.end_datatime
        # 初始化时间状态及订单发布平台
        now_datatime = start_datatime
        reqday_obj = reqday.Reqday(start_datatime,con)
        # 时间开始循环，直至到了最后时间
        while now_datatime < end_datatime:
            aim_car.getStaus(now_datatime, reqday_obj, SELECT_TYPE)
            now_datatime += dati.timedelta(minutes=1)
        aim_car_list.append(aim_car)
    con.close()
    return aim_car_list,reqday_obj
if __name__ == "__main__":
    # 计时
    START_T = dati.datetime.now()
    car_income_plot = []
    car_wandering_plot = []
    reqday_plot = []
    for i_episode in range(100):
        aim_car_list,reqday_obj = simulate()
        print(i_episode,"次实验结束。")
        car_income_plot.append([aim_car.income for aim_car in aim_car_list])
        car_wandering_plot.append([aim_car.wandering_all+aim_car.wandering_num for aim_car in aim_car_list])
        reqday_plot.append(len(reqday_obj.over_req))
    # 计时
    END_T = dati.datetime.now()
    print("共花费%s秒" % str((END_T-START_T).seconds))

    # 所有司机收入图
    mon_plt = [np.mean(car_income) for car_income in car_income_plot]
    wandering_plt = [np.mean(car_wandering) for car_wandering in car_wandering_plot]
    print("Greedy:",np.mean(mon_plt))
    print("司机空车时间平均数:", np.mean(wandering_plt))
    print("完成订单平均数:", np.mean(reqday_plot))

    resd = result.ResultDeal("Greedy")
    resd.plotIncome(mon_plt)
    resd.plotWandering(wandering_plt)
    resd.saveCSV({"money":mon_plt, "wandering_time":wandering_plt})