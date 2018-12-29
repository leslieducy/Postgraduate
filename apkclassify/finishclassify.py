import csv
import numpy as np
list_train = []
list_res_train = []
with open("prunedata/finish_train.csv","r") as csvfile:
    reader = csv.reader(csvfile)
    #这里不需要readlines
    for line in reader:
        line_int = list(map(lambda x:int(float(x)),line))
        list_train.append(line_int)
csvfile.close()
with open("prunedata/finish_trainresult.csv","r") as csvfile:
    reader = csv.reader(csvfile)
    #这里不需要readlines
    for line in reader:
        line_int = list(map(lambda x:int(float(x)),line))
        list_res_train.append(line_int)
csvfile.close()
list_bad = []
list_well = []
for i,res in enumerate(list_res_train):
    if res[0] == 1:
        list_well.append(list_train[i][1:])
    else:
        list_bad.append(list_train[i][1:])
arr_bad = np.array(list_bad)
arr_well = np.array(list_well)
# 列相加求和
sum_bad = arr_bad.sum(axis=0)
sum_well = arr_well.sum(axis=0)
# 拉普拉斯校准：x+1是处理特征数出现为0次的情况，会影响概率计算。
# 得到所有特征值为1的概率，当特征为0时，取1-plist_bad[x]或1-plist_well[x]即可
plist_bad = list(map(lambda x:(x+0.0000000000001)/len(list_bad),sum_bad))
plist_well = list(map(lambda x:(x+0.0000000000001)/len(list_well),sum_well))
# 总体信息
pre_bad = len(plist_bad) / (len(plist_bad) + len(plist_well))
pre_well = len(plist_well) / (len(plist_bad) + len(plist_well))
# 测试开始
list_test = []
list_res_test = []
with open("prunedata/finish_test.csv","r") as csvfile:
    reader = csv.reader(csvfile)
    #这里不需要readlines
    for line in reader:
        line_int = list(map(lambda x:int(float(x)),line))
        list_test.append(line_int)
csvfile.close()
with open("prunedata/finish_testresult.csv","r") as csvfile:
    reader = csv.reader(csvfile)
    #这里不需要readlines
    for line in reader:
        line_int = list(map(lambda x:int(float(x)),line))
        list_res_test.append(line_int)
csvfile.close()
num_true = 0 
# 根据朴素贝叶斯计算测试集每一项的P（好）和P（坏），P(好) = P(好|总数)*P(a1=x|好)*P(a2=x|好)...，同理求出P（坏），判断大小做出结论
p_bad_list = []
p_well_list = []
for i,test in enumerate(list_test):
    arr_test = np.array(test[1:])
    p_bad = 0
    for ii,bad in enumerate(arr_test):
        # 因为概率的值都较小，为了防止下溢出，用log函数进行处理：log(ab) = log(a) + log(b)
        if bad == 0:
            p_bad += np.log( 1 - plist_bad[ii])
            # 处理误差，当log计算出现负数，归零
            if(np.isnan(p_bad)):
                p_well = 0
        else:
            p_bad += np.log(plist_bad[ii])
    p_bad += np.log(pre_bad)
    if  np.isnan(p_bad):
        p_bad = 0
    p_bad_list.append(p_bad)
    p_well = 0
    for ii,well in enumerate(arr_test):
        if well == 0:
            p_well += np.log( 1 - plist_well[ii])
            # 处理误差，当log计算出现负数，归零
            if(np.isnan(p_well)):
                p_well = 0
        else:
            p_well += np.log(plist_well[ii])
    p_well += np.log(pre_well)
    if  np.isnan(p_well):
        p_well = 0
    p_well_list.append(p_well)
    # print(list_res_test[i][0],p_bad>p_well)
    if (p_bad > p_well and list_res_test[i][0] == 2) or (p_bad < p_well and list_res_test[i][0] == 1):
        num_true += 1
print(num_true/len(list_test))

# 记录判别数组
p_bad_per_arr = np.array(p_bad_list)
p_well_per_arr = np.array(p_well_list)

list_train = []
list_res_train = []
with open("prunedata/finish_trainapi.csv","r") as csvfile:
    reader = csv.reader(csvfile)
    #这里不需要readlines
    for line in reader:
        line_int = list(map(lambda x:int(float(x)),line))
        list_train.append(line_int)
csvfile.close()
with open("prunedata/finish_trainresult.csv","r") as csvfile:
    reader = csv.reader(csvfile)
    #这里不需要readlines
    for line in reader:
        line_int = list(map(lambda x:int(float(x)),line))
        list_res_train.append(line_int)
csvfile.close()
list_bad = []
list_well = []
for i,res in enumerate(list_res_train):
    if res[0] == 1:
        list_well.append(list_train[i][1:])
    else:
        list_bad.append(list_train[i][1:])
arr_bad = np.array(list_bad)
arr_well = np.array(list_well)
# 列相加求和
sum_bad = arr_bad.sum(axis=0)
sum_well = arr_well.sum(axis=0)
# 拉普拉斯校准：x+1是处理特征数出现为0次的情况，会影响概率计算。
# 得到所有特征值为1的概率，当特征为0时，取1-plist_bad[x]或1-plist_well[x]即可
plist_bad = list(map(lambda x:(x+0.0000000000001)/len(list_bad),sum_bad))
plist_well = list(map(lambda x:(x+0.0000000000001)/len(list_well),sum_well))
# 总体信息
pre_bad = len(plist_bad) / (len(plist_bad) + len(plist_well))
pre_well = len(plist_well) / (len(plist_bad) + len(plist_well))
# 测试开始
list_test = []
list_res_test = []
with open("prunedata/finish_testapi.csv","r") as csvfile:
    reader = csv.reader(csvfile)
    #这里不需要readlines
    for line in reader:
        line_int = list(map(lambda x:int(float(x)),line))
        list_test.append(line_int)
csvfile.close()
with open("prunedata/finish_testresult.csv","r") as csvfile:
    reader = csv.reader(csvfile)
    #这里不需要readlines
    for line in reader:
        line_int = list(map(lambda x:int(float(x)),line))
        list_res_test.append(line_int)
csvfile.close()
num_true = 0 
# 根据朴素贝叶斯计算测试集每一项的P（好）和P（坏），P(好) = P(好|总数)*P(a1=x|好)*P(a2=x|好)...，同理求出P（坏），判断大小做出结论
p_bad_list = []
p_well_list = []
for i,test in enumerate(list_test):
    arr_test = np.array(test[1:])
    p_bad = 0
    for ii,bad in enumerate(arr_test):
        # 因为概率的值都较小，为了防止下溢出，用log函数进行处理：log(ab) = log(a) + log(b)
        if bad == 0:
            p_bad += np.log( 1 - plist_bad[ii])
            # 处理误差，当log计算出现负数，归零
            if(np.isnan(p_bad)):
                p_well = 0
        else:
            p_bad += np.log(plist_bad[ii])
    # 重复注释
    # p_bad += np.log(pre_bad)
    if  np.isnan(p_bad):
        p_bad = 0
    p_bad_list.append(p_bad)
    p_well = 0
    for ii,well in enumerate(arr_test):
        if well == 0:
            p_well += np.log( 1 - plist_well[ii])
            # 处理误差，当log计算出现负数，归零
            if(np.isnan(p_well)):
                p_well = 0
        else:
            p_well += np.log(plist_well[ii])
    # 重复注释
    # p_well += np.log(pre_well)
    if  np.isnan(p_well):
        p_well = 0
    p_well_list.append(p_well)
    
    # print(list_res_test[i][0],p_bad>p_well)
    if (p_bad > p_well and list_res_test[i][0] == 2) or (p_bad < p_well and list_res_test[i][0] == 1):
        num_true += 1
print(num_true/len(list_test))

# 记录判别数组
p_bad_api_arr = np.array(p_bad_list)
p_well_api_arr = np.array(p_well_list)
p_well_arr = p_well_per_arr + p_well_api_arr
p_bad_arr = p_bad_per_arr + p_bad_api_arr
num_true = 0
for i,_ in enumerate(list_res_test):
    # print(list_res_test[i][0],p_bad_arr[i]>p_well_arr[i])
    if (p_bad_arr[i] >= p_well_arr[i] and list_res_test[i][0] == 2) or (p_bad_arr[i] < p_well_arr[i] and list_res_test[i][0] == 1):
        num_true += 1
    elif p_bad_arr[i] >= p_well_arr[i] and list_res_test[i][0] == 1:
        print("这个把好的误判称坏的。")
    elif p_bad_arr[i] < p_well_arr[i] and list_res_test[i][0] == 2:
        print("这个把坏的误判称好的。")

print(num_true/len(list_test))


