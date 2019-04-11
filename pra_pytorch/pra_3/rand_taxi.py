import cx_Oracle as cx      #导入模块
import datetime
import matplotlib.pyplot as plt
import numpy as np

DATA = [2013,5,4]
con = cx.connect('test', 'herron', '127.0.0.1:1521/TestDatabase')  #创建连接
mon_plt = []

def get_init_state():
    road_id = 409
    time = [0,9,0]
    return road_id,time
def get_step_state(a):
    if a == None:
        return None, None, ''
    cursor = con.cursor()       #创建游标
    # print(a)
    cursor.execute("select * from REQUEST t where REQUEST_ID = '" + str(a)+"'")  #执行sql语句
    data = cursor.fetchone()       #获取一条数据
    cursor.close()  #关闭游标
    if data:
        s_num, exp_time, r = data[10], data[4], data[2]
        return s_num, r, exp_time
    else:
        return None, None, ''
def choose_action(road_id, time):
    now_time = datetime.datetime(*(DATA + time))
    later_time = now_time + datetime.timedelta(minutes=5)
    if now_time.strftime("%Y-%m-%d") != later_time.strftime("%Y-%m-%d"):
        print(now_time.strftime("%Y-%m-%d"), later_time.strftime("%Y-%m-%d"))
        print("一天结束")
        return None
    cursor = con.cursor()       #创建游标
    cursor.execute("select * from REQUEST t where on_time > '"+
        now_time.strftime("%H:%M:%S") + "' and on_time < '"+
        later_time.strftime("%H:%M:%S")+"' and day_no = '" + 
        now_time.strftime("%Y-%m-%d") +"'")  #执行sql语句
    # data = cursor.fetchone()        #获取一条数据
    data = cursor.fetchall()       #获取全部数据
    cursor.close()  #关闭游标

    end_road_list = []
    for item in data:
        end_road_list.append((item[0],item[10]))
    rand_act = np.random.randint(0,len(end_road_list))
    
    return end_road_list[rand_act]
# 训练次数
train_n = 30
for i_episode in range(train_n):
    road_id, time = get_init_state()
    money = 0
    while True:
        # 选动作
        q_a = choose_action(road_id, time)
        if q_a is None:
            # 没有找到动作请求, 进入下回合
            break
        # 得到环境反馈
        s_, r, exp_time = get_step_state(q_a[0])
        exp_time = exp_time.split(':')
        time = [int(x) for x in exp_time]
        money += r
    print("第",i_episode,"次：",money)
    mon_plt.append(money)
x = range(0,train_n,1)
plt.figure(figsize=(8,6), dpi=80)
plt.title("DQN")
plt.xlabel("train(reward)") 
plt.ylabel("n") 
plt.plot(x, mon_plt, label="precision")
plt.legend(loc='upper right')
# plt.ylim(0,100)
plt.show()
print("过程全部结束")
con.close()     #关闭数据库连接
