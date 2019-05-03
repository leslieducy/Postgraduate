import matplotlib.pyplot as plt  
import numpy as np  
from sklearn.cluster import AgglomerativeClustering
from sklearn import datasets 

import cx_Oracle as cx      #导入模块

def getData():
    con = cx.connect('test', 'herron', '127.0.0.1:1521/TestDatabase')  #创建连接
    cursor = con.cursor()       #创建游标
    cursor.execute("select ROAD_ID,PRI_VERTEX,NEI_VERTEX from ROAD_PROP t")
    car_data_list = cursor.fetchall()
    cursor.close()
    con.close()
    return car_data_list
def insertData(data_list):
    print("开始链接数据库，执行更新数据...")
    con = cx.connect('test', 'herron', '127.0.0.1:1521/TestDatabase')  #创建连接
    cursor = con.cursor()       #创建游标
    sql = '''
    UPDATE ROAD_PROP SET CLUSTER_TYPE= :1
    WHERE ROAD_ID= :2 and PRI_VERTEX= :3 and NEI_VERTEX= :4
    '''
    # cursor.prepare(sql)
    rown = cursor.executemany(sql, data_list)
    con.commit()
    cursor.close()
    con.close()
    print("执行完成！")

X = np.array(getData(), dtype=int)

estimator = AgglomerativeClustering(linkage='ward', n_clusters=8)
# estimator = KMeans(n_clusters=10)  # 构造聚类器
estimator.fit(X)  # 聚类
label_pred = estimator.labels_  # 获取聚类标签
# 绘制k-means结果
markers = ['o','*','+']
for i in range(len(set(label_pred))):
    x_plot = X[label_pred == i]
    color = '#'+str(np.random.randint(i*111111,(i+1)*111111)).zfill(6)
    marker = markers[np.random.randint(0,len(markers))]
    plt.scatter(x_plot[:, 1], x_plot[:, 2], c=color, marker=marker, label='label'+str(i))  
    
insert_data = []
y = np.array([label_pred], dtype=int)
result = np.insert(X, 0, values=y, axis=1)
# result = np.c_(X,y.T)
for item in result:
    it_list = []
    for it in item:
        it_list.append(int(it))
    insert_data.append(tuple(it_list))
insertData(insert_data)
plt.xlabel('sepal length')  
plt.ylabel('sepal width')  
plt.legend(loc=2)
plt.show() 