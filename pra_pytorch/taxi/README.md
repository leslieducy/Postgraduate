# 这里是一个出租车请求的强化学习模型
## 数据集：
### 1. 司机表设计（taximan）
|字段名|类型|值介绍|
|:-|:-|:-|
|taximan_ID|number|一共1252个司机|
|start_time|time|0:01:00 - 23:59:00，每天大概五万个订单,跨天订单以起始时间为主，时间具体到分钟|
|end-time|time|0:01:00 - 23:59:00|
|day_no|string|2013-03-01 - 2013-05-20，共计81天（5月21日到30日数据有问题，订单一共才19个，所以暂不考虑）|
|income|number|司机当天的收入，价格单位（分）|
|start_LNG|float|司机当天开始工作的起始位置经度|
|start_LAT|float|司机当天开始工作的起始位置纬度|
|start_road|number|司机当天开始工作的起始位置(由经纬度得到)|
|end_LNG|float|司机当天结束工作的最终位置经度|
|end_LAT|float|司机当天结束工作的最终位置纬度|
|end_road|number|司机当天结束工作的最终位置(由经纬度得到)|
### 2. 订单表设计（request）
|字段名|类型|值介绍|
|:-|:-|:-|
|request_ID|number|每天大概5万个订单|
|day_no|string|2013-03-01 - 2013-05-20，共计81天|
|cost|number|订单花费，价格单位（分），起步6元（600分），每天最高有100-500元|
|on_time|time|0:01:00 - 23:59:00，订单发出时间|
|on_LNG|float|订单的起始位置经度|
|on_LAT|float|订单的起始位置纬度|
|on_road|number|订单的起始位置(由经纬度得到)|
|off_time|time|0:01:00 - 23:59:00，订单预计结束时间|
|off_LNG|float|订单的目的位置经度|
|off_LAT|float|订单的目的位置纬度|
|off_road|number|订单的目的位置(由经纬度得到)|
### 3. 邻路表设计（neighbor）
|字段名|类型|值介绍|
|:-|:-|:-|
|pri_road|number|路段号|
|nei_road|number|一个相邻的路段号|
### Oracle数据导入的SQL语句
```
1. 创建司机表并导入数据
create table taximan as
select distinct taximan_ID, start_time, end_time, day_no, income, 
  ton1.ONLON as start_LNG, ton1.ONLAT as start_LAT, ton1.ROADPOINTBELONG as start_road,
  ton2.OFFLON as end_LNG, ton2.OFFLAT as end_LAT, ton2.ROADPOINTBELONG as end_road
from (select t1.UNIT_ID as taximan_ID,to_char(min(t1.ONTIME),'HH24:mi:ss') as start_time,to_char(max(t2.OFFTIME),'HH24:mi:ss') as end_time,
    to_char(t1.ONTIME,'yyyy-mm-dd') as day_no,sum(t1.RUNMONEY) as income
  from ALL_ROADNETWORK345_ON t1,ALL_ROADNETWORK345_OFF_NEXT t2
  where t1.businesshis_id=t2.businesshis_id
  group by to_char(ONTIME,'yyyy-mm-dd'),t1.UNIT_ID) tem1, ALL_ROADNETWORK345_ON ton1,ALL_ROADNETWORK345_OFF_NEXT ton2
where ton1.unit_id=tem1.taximan_ID and to_char(ton1.ONTIME,'yyyy-mm-dd')=tem1.day_no and to_char(ton1.ONTIME,'HH24:mi:ss')=tem1.start_time
  and ton2.unit_id=tem1.taximan_ID and to_char(ton2.OFFTIME,'yyyy-mm-dd')=tem1.day_no and to_char(ton2.OFFTIME,'HH24:mi:ss')=tem1.end_time
  
2. 创建订单表并导入数据
create table request as
select t1.BUSINESSHIS_ID as request_ID, to_char(t1.ONTIME,'yyyy-mm-dd') as day_no, t1.RUNMONEY as cost, 
  to_char(t1.ONTIME,'HH24:mi:ss') as on_time, to_char(t2.OFFTIME,'HH24:mi:ss') as off_time,
  t1.ONLON as start_LNG, t1.ONLAT as start_LAT, t1.ROADPOINTBELONG as start_road,
  t2.OFFLON as end_LNG, t2.OFFLAT as end_LAT, t2.ROADPOINTBELONG as end_road
from ALL_ROADNETWORK345_ON t1,ALL_ROADNETWORK345_OFF_NEXT t2
where t1.businesshis_id=t2.businesshis_id

3. 创建相邻路段表（neighbor）
create table neighbor1 as 
select c1.VERTEXNEARROAD as pri_road,c2.VERTEXNEARROAD as nei_road from ROADVERTEX c1,ROADVERTEX c2 where c1.VERTEXID=c2.VERTEXID and c1.VERTEXNEARROAD != c2.VERTEXNEARROAD

单路口的路段没有包含进来
4. 创建道路属性表（road_prop）
create table road_prop as
select c1.VERTEXNEARROAD as road_id, c1.VERTEXID as pri_vertex,c2.VERTEXID as nei_vertex from ROADVERTEX c1,ROADVERTEX c2 where c1.VERTEXNEARROAD=c2.VERTEXNEARROAD and c1.VERTEXID!=c2.VERTEXID
```
## 模拟
1. 原始收入情况

直接读取数据

2. 强化学习后的收入情况

根据某一司机的情况进行选择，从出发时刻开始，查看全局的订单，根据qdn的学习作出选择，一直到下一天。dqn设计：选的是动作，网络学习的是区域

3. 其他方法的收入情况
   
## 问题记录：
1. 司机可选择订单的操作从数据库读取太慢。
> 解决：建一个临时请求表，只包含一天的数据。
2. 强化学习与随机选择的区别：
强化100次平均值：135302.0,108856.5,133758.0,131681.5
随机所有司机的平均值：152703.513174404
聚类后：
强化100次：65099.0
强化200次：60980.5,57224.0,58841.0
DQN 300次：49057.8,51771.0,55614.666666666664,64042.666666666664,62978.666666666664,62129.333333333336
A3C 300次：62917.187443884，58686.313559519054，64048.5954354126
随机100次：46317.0,46894.0,48651.0,48162.0, 53480.333333333336,51044.0
贪心100次：53332.0,49926.0,49919.0,56409.333333333336
3. 回报值的问题

## 类抽象
1. 司机类
接取订单
2. 订单类
3. 道路类
返回邻路
4. DQN类

## 聚类
新建道路属性表，并手动新建CLUSTER_TYPE字段，并通过聚类（层次）算法得到不同道路所属的类别（即区域），更新回CLUSTER_TYPE字段。