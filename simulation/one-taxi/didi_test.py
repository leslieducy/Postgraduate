from models import reqday, car, road
import datetime as dati
import cx_Oracle as cx      #导入模块

from utils import result
import matplotlib.pyplot as plt
import numpy as np

# 模拟didi的组合优化，将订单与司机相互匹配。
def simulate():
    # 司机选择订单的属性（0-4）（随机，贪心，评估，强化）
    SELECT_TYPE = 0
    con = cx.connect('test', 'herron', '127.0.0.1:1521/TestDatabase')  #创建连接
    # 初始化一星期的时间
    start_datatime_list = ['2013-03-08','2013-03-09','2013-03-10','2013-03-11','2013-03-12','2013-03-13','2013-03-14']
    # 根据每天时间初始化发布平台，车辆
    car_all_list = []
    reqday_all_list = []
    for start_datatime in start_datatime_list:
        start_datatime = dati.datetime.strptime(start_datatime+' 00:05:00', '%Y-%m-%d %H:%M:%S')
        now_datatime = start_datatime
        reqday_obj = reqday.Reqday(start_datatime,con)

        # 初始化所有出租车
        car_all = []
        cursor = con.cursor()       #创建游标
        cursor.execute("select * from taximan t where DAY_NO = '%s'" % start_datatime.strftime("%Y-%m-%d"))
        car_data_list = cursor.fetchall()       #获取所有数据
        cursor.close()
        car_all = [car.Car(car_data,con) for car_data in car_data_list]
        # print('car_all',len(car_all))
        car_all = np.random.choice(car_all, 750)
        # 保存3分钟内未完成的订单

        # 时间开始循环，直至下一天
        while now_datatime.strftime("%Y-%m-%d") == start_datatime.strftime("%Y-%m-%d"):
            # 获取该时刻的订单
            req_data_list = reqday_obj.getTimeReq(now_datatime)
            # 并为每个订单匹配司机（计算该时刻每个订单与司机匹配度，确保整体的匹配度最大）
            for req in req_data_list:
                req_finish_tag = False
                now_road = road.Road(req[7], con)
                for car_obj in car_all:
                    # 选择已经开工且并未休息的空闲车，并且位置在对应的路段上
                    if car_obj.wandering_num > 0 and now_datatime > car_obj.start_datatime and now_datatime < car_obj.start_datatime+dati.timedelta(hours=8) and car_obj.next_road_id == now_road.id:
                        car_obj.accept_req(req, now_datatime)
                        req_finish_tag = True
                        reqday_obj.overReq(req)
                        car_obj.wandering_num = 0
                        break
                    # elif car_obj.wandering_num == 0:
                    #     print("正在运行中")
                # if len(car_all) == 0:
                if req_finish_tag:
                    continue
                for car_obj in car_all:
                    # 选择已经开工且并未休息的空闲车，并且位置在对应的路段周围
                    if car_obj.wandering_num > 0 and car_obj.next_road_id in now_road.neighbor:
                        car_obj.wandering_num += 1
                        # car_obj.accept_req(req, now_datatime)
                        car_obj.accept_req(req, now_datatime+dati.timedelta(minutes=1))
                        reqday_obj.overReq(req)
                        car_obj.wandering_all += car_obj.wandering_num
                        car_obj.wandering_num = 0
                        break
            # 1、对所有没接到订单的车进行随机巡游操作，2、对所有刚完成订单的司机进行巡游赋值操作。
            for car_obj in car_all:
                if now_datatime >= car_obj.start_datatime+dati.timedelta(hours=8):
                    continue
                now_road = road.Road(car_obj.next_road_id, con)
                # 对所有没接到订单的车进行随机巡游操作。
                if car_obj.wandering_num > 0:
                    car_obj.wandering_num += 1
                    # 闲逛超过五分钟随机进入下一路段
                    if car_obj.wandering_num > 5:
                        car_obj.next_road_id = now_road.getRandomNeighbor()
                # 对于刚完成订单的车
                if car_obj.next_time == now_datatime:
                    car_obj.wandering_num += 1
            now_datatime += dati.timedelta(minutes=1)

        print(str(now_datatime)+"集中派单结束")
        car_all_list.append(car_all)
        reqday_all_list.append(reqday_obj)
    con.close()
    return car_all_list, reqday_all_list

if __name__ == "__main__":
    # 计时
    START_T = dati.datetime.now()
    car_income_plot = []
    car_wandering_plot = []
    reqday_plot = []
    for i_episode in range(1):
        car_all_list,reqday_all_list = simulate()
        print(i_episode,"次实验结束。")
        for aim_car_list in car_all_list:
            car_income_plot.append([aim_car.income for aim_car in aim_car_list])
            car_wandering_plot.append([aim_car.wandering_all+aim_car.wandering_num for aim_car in aim_car_list])
        for reqday_obj in reqday_all_list:
            reqday_plot.append(len(reqday_obj.over_req))
    # 计时
    END_T = dati.datetime.now()
    print("共花费%s秒" % str((END_T-START_T).seconds))

    # 所有司机收入图
    car_income = np.array(car_income_plot)
    car_wandering = np.array(car_wandering_plot)
    mon_plt = car_income.mean(axis=0)
    wandering_plt = car_wandering.mean(axis=0)
    # mon_plt = [np.mean(car_income) for car_income in car_income_plot]
    # wandering_plt = [np.mean(car_wandering) for car_wandering in car_wandering_plot]
    # print("AllCentralized:",np.mean(mon_plt))
    print("SatCentralized:",np.mean(mon_plt))
    print("司机空车时间平均数:", np.mean(wandering_plt))
    print("完成订单数:", np.mean(reqday_plot)/len(car_income_plot[0]))

    resd = result.ResultDeal("AllCentralized")
    # resd = result.ResultDeal("SatCentralized")
    resd.plotIncome(mon_plt)
    resd.plotWandering(wandering_plt)
    # resd.saveCSV({"money":mon_plt, "wandering_time":wandering_plt, "reqday_plot":reqday_plot})
    resd.saveCSV({"money":mon_plt, "wandering_time":wandering_plt})
