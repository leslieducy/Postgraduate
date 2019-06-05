import cx_Oracle as cx      #导入模块
import datetime
import matplotlib.pyplot as plt
import numpy as np

DATA = [2013,5,4]
con = cx.connect('test', 'herron', '127.0.0.1:1521/TestDatabase')  #创建连接
mon_plt = []
FINISHED_LIST = []

def get_init_state():
    road_id = 409
    time = [0,9,0]
    return road_id,time
def get_step_state(a):
    if a == None:
        return None, None, ''
    cursor = con.cursor()       #创建游标
    # print(a)
    cursor.execute("select * from TEM_REQ t where REQUEST_ID = '" + str(a)+"'")  #执行sql语句
    data = cursor.fetchone()       #获取一条数据
    cursor.close()  #关闭游标
    if data:
        s_num, exp_time, r = data[10], data[4], data[2]
        return s_num, r, exp_time
    else:
        return None, None, ''
def choose_action(road_id, time):
    road_no = road_id
    cursor = con.cursor()       #创建游标
    cursor.execute("select NEI_ROAD from NEIGHBOR t where PRI_ROAD = "+str(road_no))  #执行sql语句
    neigh_road_list = cursor.fetchall()       #获取一条数据
    cursor.close()  #关闭游标
    neigh_road_list = [str(x[0]) for x in neigh_road_list]
    neigh_road_list.append(str(road_no))
    now_time = datetime.datetime(*(DATA + time))
    before_time = now_time - datetime.timedelta(minutes=5)

    if now_time.strftime("%Y-%m-%d") != before_time.strftime("%Y-%m-%d"):
        before_time = datetime.datetime(*(DATA))
    
    wait = 0
    while True:
        wait += 1
        later_time = now_time + datetime.timedelta(minutes=(5*wait))
        if now_time.strftime("%Y-%m-%d") != later_time.strftime("%Y-%m-%d"):
            print(now_time.strftime("%Y-%m-%d"), later_time.strftime("%Y-%m-%d"))
            print("一天结束")
            return None
        cursor = con.cursor()       #创建游标
        placeholders = ','.join(":%d" % i for i,_ in enumerate(neigh_road_list))
        sql = ("select * from TEM_REQ t where on_time > '"
            + before_time.strftime("%H:%M:%S") + "' and on_time < '"
            + later_time.strftime("%H:%M:%S")+"' and day_no = '" 
            + now_time.strftime("%Y-%m-%d") +"' and start_road in (%s)" % placeholders)
        # print(sql)
        # print(neigh_road_list)
        cursor.execute(sql,neigh_road_list)  #执行sql语句
        # data = cursor.fetchone()        #获取一条数据
        data = cursor.fetchall()       #获取全部数据
        cursor.close()  #关闭游标
        end_road_list = []
        for item in data:
            if item[0] not in FINISHED_LIST:
                end_road_list.append((item[0],item[10]))
                
        # print(len(end_road_list))
        if len(end_road_list)>1:
            break
    rand_act = np.random.randint(0,len(end_road_list))
    return end_road_list[rand_act]
# 训练次数
train_n = 100
for i_episode in range(train_n):
    FINISHED_LIST = []
    road_id, time = get_init_state()
    print("司机初始化成功,开始模拟...")
    money = 0
    while True:
        # 选动作
        q_a = choose_action(road_id, time)
        # print(q_a)
        # print("动作选择成功！")
        if q_a is None:
            # 没有找到动作请求, 进入下回合
            break
        FINISHED_LIST.append(q_a[0])
        # 得到环境反馈
        s_, r, exp_time = get_step_state(q_a[0])
        exp_time = exp_time.split(':')
        time = [int(x) for x in exp_time]
        money += r
    # print(FINISHED_LIST)
    print("第",i_episode,"次：",money)
    mon_plt.append(money)
print("模拟完成！")
print(np.mean(mon_plt))
x = range(0,train_n,1)
plt.figure(figsize=(8,6), dpi=80)
plt.title("Random")
plt.xlabel("train(reward)") 
plt.ylabel("n") 
plt.plot(x, mon_plt, label="precision")
plt.legend(loc='upper right')
# plt.ylim(0,100)
plt.show()
print("过程全部结束")
con.close()     #关闭数据库连接
