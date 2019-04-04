# 这里是一个出租车请求的模拟预测模型
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
|end_Road|number|司机当天结束工作的最终位置(由经纬度得到)|
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
### Oracle数据导入的SQL语句
```
1. 创建司机表并导入数据
create table taximan as
select taximan_ID, start_time, end_time, day_no, income, 
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
```
