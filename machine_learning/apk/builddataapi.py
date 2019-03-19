
import os
import numpy as np
import csv
import random
# from collections import OrderedDict

perm_list = []

fm = open("all_service.txt", 'r')
fls = fm.readlines()
for fl in fls:
    line = fl.strip()
    # 没有的属性默认为0
    perm_list.append(line)
fm.close()

out_file_name = "./prunedata/pointapi.csv"
out_train_name = "./prunedata/trainapi.csv"
out_train_res_name = "./prunedata/trainresultapi.csv"
out_test_name = "./prunedata/testapi.csv"
out_test_res_name = "./prunedata/testresultapi.csv"
if os.path.exists(out_file_name):
    os.remove(out_file_name)
if os.path.exists(out_train_name):
    os.remove(out_train_name)
if os.path.exists(out_test_name):
    os.remove(out_test_name)
csv_file = open(out_file_name, "w", newline='')
csv_train_file = open(out_train_name, "w", newline='')
csv_train_res_file = open(out_train_res_name, "w", newline='')
csv_test_file = open(out_test_name, "w", newline='')
csv_test_res_file = open(out_test_res_name, "w", newline='')

fm = open("badapi.txt", 'r')
fls = fm.readlines()
arr = np.zeros(len(perm_list) + 1)
for fl in fls:
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
    csv_writer = csv.writer(csv_file)
    arr_list = arr.tolist()
    csv_writer.writerow(arr_list)
    # 根据随机概率分别存入训练数据集或测试数据集
    if random.randint(1,10) > 3:
        csv_train_res_writer = csv.writer(csv_train_res_file)
        csv_train_res_writer.writerow(arr_list[0:1])
        csv_train_writer = csv.writer(csv_train_file)
        csv_train_writer.writerow(arr_list[1:])
    else:
        csv_test_res_writer = csv.writer(csv_test_res_file)
        csv_test_res_writer.writerow(arr_list[0:1])
        csv_test_writer = csv.writer(csv_test_file)
        csv_test_writer.writerow(arr_list[1:])

fm.close()

fm = open("wellapi.txt", 'r')
fls = fm.readlines()
arr = np.zeros(len(perm_list) + 1)
for fl in fls:
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
    csv_writer = csv.writer(csv_file)
    arr_list = arr.tolist()
    csv_writer.writerow(arr_list)
    # 根据随机概率分别存入训练数据集或测试数据集
    if random.randint(1,10) > 3:
        csv_train_res_writer = csv.writer(csv_train_res_file)
        csv_train_res_writer.writerow(arr_list[0:1])
        csv_train_writer = csv.writer(csv_train_file)
        csv_train_writer.writerow(arr_list[1:])
    else:
        csv_test_res_writer = csv.writer(csv_test_res_file)
        csv_test_res_writer.writerow(arr_list[0:1])
        csv_test_writer = csv.writer(csv_test_file)
        csv_test_writer.writerow(arr_list[1:])
fm.close()

csv_file.close()
csv_train_file.close()
csv_test_file.close()
