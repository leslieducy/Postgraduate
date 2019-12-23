from models import reqday, car
from modules import dqn
from utils import result
import datetime as dati
import cx_Oracle as cx      #导入模块
import copy
import matplotlib.pyplot as plt

import numpy as np

def simulate(dqn_obj, taxi_sql):
   # 司机选择订单的属性（0-3）（随机，贪心，评估，dqn强化，A3C强化）
    SELECT_TYPE = 3
    con = cx.connect('test', 'herron', '127.0.0.1:1521/TestDatabase')  #创建连接
    # 选择一辆出租车获取其一星期的原始数据
    cursor = con.cursor()       #创建游标
    cursor.execute(taxi_sql)
    car_data_list = cursor.fetchall()       #获取所有数据
    cursor.close()
    car_all_status = [car.Car(car_data,con) for car_data in car_data_list]

    aim_car_list = []
    reqday_list = []
    # 循环开始每一天（司机每个值都视为一天）
    for aim_car in car_all_status:
        start_datatime = aim_car.start_datatime
        end_datatime = aim_car.start_datatime+dati.timedelta(hours=8)
        # 初始化时间状态及订单发布平台
        now_datatime = start_datatime
        reqday_obj = reqday.Reqday(start_datatime,con)

        # 时间开始循环，直至到了最后时间
        while now_datatime < end_datatime:
            aim_car.getStaus(now_datatime, reqday_obj, SELECT_TYPE, dqn_obj)
            # print(str(now_datatime)+" DQN匹配结束")
            now_datatime += dati.timedelta(minutes=1)
        aim_car_list.append(aim_car)
        reqday_list.append(len(reqday_obj.over_req))
    con.close()
    return aim_car_list,reqday_list
if __name__ == "__main__":
    # 计时
    START_T = dati.datetime.now()
    car_income_plot = []
    reqday_plot = []
    car_wandering_plot = []
    # 初始化DQN神经网络
    dqn_obj = dqn.DQN()
    # plt.ion()
    plt.figure(1)
    # 训练
    train_num = 30
    ion_plot = []
    for i_episode in range(train_num):
        taxi_sql = "select * from taximan t where TAXIMAN_ID = '807' and DAY_NO in ('2013-03-01','2013-03-02','2013-03-03','2013-03-04','2013-03-05','2013-03-06','2013-03-07') order by DAY_NO"
        aim_car_list,reqday_list = simulate(dqn_obj, taxi_sql)
        # ion_plot.append(np.mean(dqn_obj.loss_list))
        # plt.plot(ion_plot,c='r',ls='-', marker='o', mec='b',mfc='w')  ## 保存历史数据
        print(i_episode,"次训练结束。")
        # plt.pause(0.1)
    # plt.ioff()
    # 测试
    for i_episode in range(100):
        dqn_test_obj = copy.deepcopy(dqn_obj)
        taxi_sql = "select * from taximan t where TAXIMAN_ID = '807' and DAY_NO in ('2013-03-08','2013-03-09','2013-03-10','2013-03-11','2013-03-12','2013-03-13','2013-03-14') order by DAY_NO"
        aim_car_list,reqday_list = simulate(dqn_test_obj, taxi_sql)
        car_income_plot.append([aim_car.income/100 for aim_car in aim_car_list])
        car_wandering_plot.append([aim_car.wandering_all+aim_car.wandering_num for aim_car in aim_car_list])
        reqday_plot=reqday_list
        print(i_episode,"次测试结束。")
    
    # 计时
    END_T = dati.datetime.now()
    print("共花费%s秒" % str((END_T-START_T).seconds))

    # 所有司机收入图（按星期计算平均0，分司机平均1）
    # [np.mean(car_income) for car_income in car_income_plot]

    # car_income = np.array(car_income_plot)
    # car_wandering = np.array(car_wandering_plot)
    # mon_plt = car_income.mean(axis=0)
    # wandering_plt = car_wandering.mean(axis=0)
    # print('car_income_plot',car_income_plot,'car_wandering_plot',car_wandering_plot)

    mon_plt = np.array(car_income_plot).mean(axis=1)
    wandering_plt = np.array(car_wandering_plot).mean(axis=1)
    result_title = "DQNTrain"+str(train_num)
    print(result_title,np.mean(mon_plt))
    print("司机空车时间平均数:", np.mean(wandering_plt))
    print("完成订单平均数:", np.mean(reqday_plot))
    resd = result.ResultDeal(result_title)
    resd.plotIncome(mon_plt)
    resd.plotWandering(wandering_plt)
    resd.saveCSV({"money":mon_plt, "wandering_time":wandering_plt})
    # resd.saveCSV({"money":mon_plt, "wandering_time":wandering_plt, "reqday_plot":reqday_plot})
