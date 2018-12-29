import os
import numpy as np
import csv
# 将两种数据不同步的部分删除掉，是数据长度一致
# 恶意软件数据处理部分
finish_bad = []
finish_bad_api = []

fm = open("bad.txt", 'r')
fls = fm.readlines()
fm_api = open("badapi.txt", 'r')
fls_api = fm_api.readlines()
for fl in fls:
    name_list = fl.strip().split(":",1)[0]
    for fl_a in fls_api:
        if fl_a.strip().split(":",1)[0] in name_list:
            finish_bad.append(fl)
            finish_bad_api.append(fl_a)
fm.close()
fm_api.close()
# 正常软件数据处理部分
finish_well = []
finish_well_api = []

fm = open("well.txt", 'r')
fls = fm.readlines()
fm_api = open("wellapi.txt", 'r')
fls_api = fm_api.readlines()
for fl in fls:
    name_list = fl.strip().split(":",1)[0]
    for fl_a in fls_api:
        if fl_a.strip().split(":",1)[0] in name_list:
            finish_well.append(fl)
            finish_well_api.append(fl_a)
fm.close()
fm_api.close()

# 构建数据部分
# 权限部分
random_list = np.random.randint(0,10,(len(finish_bad) if len(finish_bad)>len(finish_well) else len(finish_well)))
perm_list = []
fm = open("all_permission.txt", 'r')
fls = fm.readlines()
for fl in fls:
    line = fl.strip()
    # 没有的属性默认为0
    perm_list.append(line)
fm.close()

out_train_name = "./prunedata/finish_train.csv"
out_train_res_name = "./prunedata/finish_trainresult.csv"
out_test_name = "./prunedata/finish_test.csv"
out_test_res_name = "./prunedata/finish_testresult.csv"

csv_train_file = open(out_train_name, "w", newline='')
csv_train_res_file = open(out_train_res_name, "w", newline='')
csv_test_file = open(out_test_name, "w", newline='')
csv_test_res_file = open(out_test_res_name, "w", newline='')

fls = finish_bad
arr = np.zeros(len(perm_list) + 1)
for i, fl in enumerate(fls):
    arr = np.zeros(len(perm_list) + 1)
    # 恶意apk值为2，保存在数组的第一位
    arr[0] = 2
    line = fl.strip()
    prop_list = line.split(":",1)[-1].split(",")
    for prop in prop_list:
        # 存在的属性设置为1(此处可能存在误差)
        arr[perm_list.index(prop) + 1] = 1
    arr_list = arr.tolist()
    # 根据随机概率分别存入训练数据集或测试数据集
    if random_list[i] >= 3:
        csv_train_res_writer = csv.writer(csv_train_res_file)
        csv_train_res_writer.writerow(arr_list[0:1])
        csv_train_writer = csv.writer(csv_train_file)
        csv_train_writer.writerow(arr_list[1:])
    else:
        csv_test_res_writer = csv.writer(csv_test_res_file)
        csv_test_res_writer.writerow(arr_list[0:1])
        csv_test_writer = csv.writer(csv_test_file)
        csv_test_writer.writerow(arr_list[1:])

fls = finish_well
arr = np.zeros(len(perm_list) + 1)
for i, fl in enumerate(fls):
    arr = np.zeros(len(perm_list) + 1)
    # 正常apk值为1，保存在数组的第一位
    arr[0] = 1
    line = fl.strip()
    prop_list = line.split(":",1)[-1].split(",")
    for prop in prop_list:
        # 存在的属性设置为1
        arr[perm_list.index(prop) + 1] = 1
    arr_list = arr.tolist()
    # 根据随机概率分别存入训练数据集或测试数据集
    if random_list[i] >= 3:
        csv_train_res_writer = csv.writer(csv_train_res_file)
        csv_train_res_writer.writerow(arr_list[0:1])
        csv_train_writer = csv.writer(csv_train_file)
        csv_train_writer.writerow(arr_list[1:])
    else:
        csv_test_res_writer = csv.writer(csv_test_res_file)
        csv_test_res_writer.writerow(arr_list[0:1])
        csv_test_writer = csv.writer(csv_test_file)
        csv_test_writer.writerow(arr_list[1:])

csv_train_file.close()
csv_test_file.close()

# api部分
perm_list = []
fm = open("all_service.txt", 'r')
fls = fm.readlines()
for fl in fls:
    line = fl.strip()
    # 没有的属性默认为0
    perm_list.append(line)
fm.close()

out_train_name = "./prunedata/finish_trainapi.csv"
out_test_name = "./prunedata/finish_testapi.csv"
csv_train_file = open(out_train_name, "w", newline='')
csv_test_file = open(out_test_name, "w", newline='')

fls = finish_bad_api
arr = np.zeros(len(perm_list) + 1)
for i, fl in enumerate(fls):
    arr = np.zeros(len(perm_list) + 1)
    # 恶意apk值为2，保存在数组的第一位
    arr[0] = 2
    line = fl.strip()
    prop_list = line.split(":",1)[-1].split(",")
    for prop in prop_list:
        if prop == '':
            continue
        # 存在的属性设置为1
        arr[perm_list.index(prop) + 1] = 1
    arr_list = arr.tolist()
    # 根据随机概率分别存入训练数据集或测试数据集
    if random_list[i] >= 3:
        csv_train_writer = csv.writer(csv_train_file)
        csv_train_writer.writerow(arr_list[1:])
    else:
        csv_test_writer = csv.writer(csv_test_file)
        csv_test_writer.writerow(arr_list[1:])

fls = finish_well_api
arr = np.zeros(len(perm_list) + 1)
for i, fl in enumerate(fls):
    arr = np.zeros(len(perm_list) + 1)
    # 正常apk值为1，保存在数组的第一位
    arr[0] = 1
    line = fl.strip()
    prop_list = line.split(":",1)[-1].split(",")
    for prop in prop_list:
        if prop == '':
            continue
        # 存在的属性设置为1
        arr[perm_list.index(prop) + 1] = 1
    arr_list = arr.tolist()
    # 根据随机概率分别存入训练数据集或测试数据集
    if random_list[i] >= 3:
        csv_train_writer = csv.writer(csv_train_file)
        csv_train_writer.writerow(arr_list[1:])
    else:
        csv_test_writer = csv.writer(csv_test_file)
        csv_test_writer.writerow(arr_list[1:])

csv_train_file.close()
csv_test_file.close()
