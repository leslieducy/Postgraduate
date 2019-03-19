#coding:utf-8

from math import log
import operator
import re
import pandas as pd
import matplotlib.pyplot as plt
# import treePlotter

def pretreatment_dataset(filename,ratio):
    """
    Pclass（客场等级）;
    Survived（生还状况），统一规定用1代表生还，用0代表死亡;
    Name（名字）;
    Age（年龄）;
    Embarked（登船港口）;
    Sex（性别）;
    """
    dataset=[]
    labels=['客场等级', '生还状况', '名字', '年龄', '登船港口', '性别']
    with open(filename,"r",encoding='utf-8') as fr:
        all_lines=fr.readlines()   #list形式,每行为1个str
        for line in all_lines[1:]:
            line = re.sub(r'([^0-9]),\s([^0-9])', '/', line, count=0)
            line = re.sub(r'"', '', line, count=0)
            line=line.strip().split(',')   #以逗号为分割符拆分列表
            # 去除第一列序号及序号为50，264，265，310，337行的不规范数据
            if len(line) == 7:
                dataset.append(line[1:])
    # 数据数字化
    df = pd.DataFrame(dataset)
    df.columns = labels
    df = df[(~df['年龄'].isin(['NA']))]
    # df['年龄'].replace('NA',30, inplace = True)
    level_mapping = {
           '3rd': 3,
           '2nd': 2,
           '1st': 1}
    df['客场等级'] = df['客场等级'].map(level_mapping)
    class_mapping = {label:idx for idx,label in enumerate(set(df['登船港口']))}
    df['登船港口'] = df['登船港口'].map(class_mapping)
    class_mapping = {label:idx for idx,label in enumerate(set(df['性别']))}
    df['性别'] = df['性别'].map(class_mapping)
    # # 更改列的顺序
    df_level = df['生还状况']
    df.drop('生还状况',axis=1, inplace=True)
    df.insert(5,'生还状况',df_level)
    df.drop('名字',axis=1, inplace=True)
    # 全部打乱
    df = df.sample(frac=1.0)  
    cut_idx = int(round(ratio * df.shape[0]))
    df_train, df_test = df.iloc[:cut_idx], df.iloc[cut_idx:]
    df_train_filename,df_test_filename = 'df_train.csv','df_test.csv'
    df_train.to_csv(df_train_filename,index=0)
    df_test.to_csv(df_test_filename,index=0)
    return df_train_filename,df_test_filename
    

def read_dataset(filename, istrain=True):
    dataset = []
    result = []
    labels=['客场等级', '年龄', '登船港口', '性别']
    with open(filename,"r",encoding='utf-8') as fr:
        all_lines=fr.readlines()   #list形式,每行为1个str
        for line in all_lines[1:]:
            line=line.strip().split(',')   #以逗号为分割符拆分列表
            if istrain:
                dataset.append(line)
            else:
                # dataset.append(line[0:-1])
                # result.append(int(line[-1]))
                dataset.append(line)
    if istrain:
        return dataset[1:],labels
    else:
        return dataset[1:]
        # return dataset[1:],result
"""
决策树的生成，data_set为训练集，attribute_label为属性名列表
决策树用字典结构表示，递归的生成
"""
def C45_createTree(data_set ,attribute_label):
    label_list = [entry[-1] for entry in data_set]
    if label_list.count(label_list[0]) == len(label_list): #如果所有的数据都属于同一个类别，则返回该类别
        return label_list[0]
    if len(data_set[0]) == 1: #如果数据没有属性值数据，则返回该其中出现最多的类别作为分类
        return most_voted_attribute(label_list)
    best_attribute_index, best_split_point = attribute_selection_method(data_set)
    best_attribute = attribute_label[best_attribute_index]
    decision_tree = { best_attribute:{}}
    del(attribute_label[best_attribute_index]) #找到最佳划分属性后需要将其从属性名列表中删除
    """
    如果best_split_point为空，说明此时最佳划分属性的类型为离散值，否则为连续值
    """
    if best_split_point == None:
        attribute_list = [entry[best_attribute_index] for entry in data_set]
        attribute_set = set(attribute_list)
        for attribute in attribute_set: 
            sub_labels = attribute_label[:]
            decision_tree[best_attribute][attribute] = C45_createTree(
                split_data_set(data_set,best_attribute_index,attribute,continuous=False),sub_labels)
    else:
        """
        最佳划分属性类型为连续值，此时计算出的最佳划分点将数据集一分为二，划分字段取名为<=和>
        """
        sub_labels = attribute_label[:]
        decision_tree[best_attribute]["<="+str(best_split_point)] = C45_createTree(
            split_data_set(data_set, best_attribute_index, best_split_point, True, 0), sub_labels)
        sub_labels = attribute_label[:]
        decision_tree[best_attribute][">" + str(best_split_point)] = C45_createTree(
            split_data_set(data_set, best_attribute_index, best_split_point, True, 1), sub_labels)
    return decision_tree
 
"""
通过信息增益比来计算最佳划分属性
属性分为离散值和连续值两种情况，分别对两种情况进行相应计算
"""
def attribute_selection_method(data_set):
    num_attributes = len(data_set[0])-1 #属性的个数，减1是因为去掉了标签
    info_D = calc_info_D(data_set)  #香农熵
    max_grian_rate = 0.0  #最大信息增益比
    best_attribute_index = -1
    best_split_point = None
    continuous = False
    for i in range(num_attributes):
        attribute_list = [entry[i] for entry in data_set]
        info_A_D = 0.0  #信息增益
        split_info_D = 0.0  #熵
        if attribute_list[0] not in set(['0','1','2','3']):
            continuous = True
        """
        属性为连续值，先对该属性下的所有离散值进行排序
        然后每相邻的两个值之间的中点作为划分点计算信息增益比，对应最大增益比的划分点为最佳划分点
        由于可能多个连续值可能相同，所以通过set只保留其中一个值
        """
        if continuous == True:
            attribute_list = sorted(attribute_list)
            temp_set = set(attribute_list) #通过set来剔除相同的值
            attribute_list = [attr for attr in temp_set]
            split_points = []
            for index in range(len(attribute_list) - 1):
                #求出各个划分点
                split_points.append((float(attribute_list[index]) + float(attribute_list[index + 1])) / 2)
            for split_point in split_points:#对划分点进行遍历
                info_A_D = 0.0
                split_info_D = 0.0
                for part in range(2): #最佳划分点将数据一分为二，因此循环2次即可得到两段数据
                    sub_data_set = split_data_set(data_set, i, split_point, True, part)
                    prob = len(sub_data_set) / float(len(data_set))
                    info_A_D += prob * calc_info_D(sub_data_set)
                    split_info_D -= prob * log(prob, 2)
                if split_info_D==0:
                    split_info_D+=1
                """
                由于关于属性A的熵split_info_D可能为0，因此需要特殊处理
                常用的做法是把求所有属性熵的平均，为了方便，此处直接加1
                """
                grian_rate = (info_D - info_A_D) / split_info_D #计算信息增益比
                if grian_rate > max_grian_rate:
                    max_grian_rate = grian_rate
                    best_split_point = split_point
                    best_attribute_index = i
        else: #划分属性为离散值
            attribute_list = [entry[i] for entry in data_set]  # 求属性列表
            attribute_set = set(attribute_list)
            for attribute in attribute_set: #对每个属性进行遍历
                sub_data_set = split_data_set(data_set, i, attribute, False)
                prob = len(sub_data_set) / float(len(data_set))
                info_A_D += prob * calc_info_D(sub_data_set)
                split_info_D -= prob * log(prob, 2)
            if split_info_D == 0:
                split_info_D += 1
            grian_rate = (info_D - info_A_D) / split_info_D #计算信息增益比
            if grian_rate > max_grian_rate:
                max_grian_rate = grian_rate
                # print(max_grian_rate)
                best_attribute_index = i
                best_split_point = None  #如果最佳属性是离散值，此处将分割点置为空留作判定
 
    return best_attribute_index,best_split_point
 
"""
多数表决：返回标签列表中数量最大的类
"""
def most_voted_attribute(label_list):
    label_nums = {}
    for label in label_list:
        if label in label_nums.keys():
            label_nums[label] += 1
        else:
            label_nums[label] = 1
    sorted_label_nums = sorted(label_nums.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_label_nums[0][0]
 
 
"""
计算数据集的信息熵
"""
def calc_info_D(data_set):
    num_entries = len(data_set)
    label_nums = {} #为每个类别建立字典，value为对应该类别的数目
    for entry in data_set:
        label = entry[-1]
        if label in label_nums.keys():
            label_nums[label]+=1
        else:
            label_nums[label]=1
    info_D = 0.0
    for label in label_nums.keys():
        prob = float(label_nums[label])/num_entries
        info_D -= prob * log(prob,2)
    return info_D
 
"""
按属性划分子数据集，分为离散属性的划分与连续属性的划分
index为划分属性的下标，value在离散属性划分的情况下为划分属性的值，continuous决定了是离散还是连续属性划分
part在连续属性划分时使用，为0时表示得到划分点左边的数据集，1时表示得到划分点右边的数据集
"""
def split_data_set(data_set, index, value, continuous, part=0):
    res_data_set = []
    # 连续值属性
    if continuous == True:
        for entry in data_set:
            # 求划分点左侧的数据集
            if part == 0 and float(entry[index])<= value: 
                reduced_entry = entry[:index]
                reduced_entry.extend(entry[index + 1:]) #划分后去除数据中第index列的值
                res_data_set.append(reduced_entry)
            # 求划分点右侧的数据集
            if part ==1 and float(entry[index])> value:
                reduced_entry = entry[:index]
                reduced_entry.extend(entry[index + 1:])
                res_data_set.append(reduced_entry)
    # 离散值属性
    else: 
        for entry in data_set:
            # 按数据集中第index列的值等于value的分数据集
            if entry[index] == value: 
                reduced_entry = entry[:index]
                reduced_entry.extend(entry[index+1:]) #划分后去除数据中第index列的值
                res_data_set.append(reduced_entry)
    return res_data_set
 
"""
对一项测试数据进行预测，通过递归来预测该项数据的标签
decision_tree:字典结构的决策树
attribute_labels:数据的属性名列表
one_test_data：预测的一项测试数据
"""
def decision_tree_predict(decision_tree, attribute_labels, one_test_data):
    first_key = list(decision_tree.keys())[0]
    second_dic = decision_tree[first_key]
    attribute_index = attribute_labels.index(first_key)
    res_label = 0
    for key in second_dic.keys(): #属性分连续值和离散值，连续值对应<=和>两种情况
        if key[0] == '<':
            value = float(key[2:])
            if float(one_test_data[attribute_index])<= value:
                if type(second_dic[key]).__name__ =='dict':
                    res_label = decision_tree_predict(second_dic[key], attribute_labels, one_test_data)
                else:
                    res_label = second_dic[key]
        elif key[0] == '>':
            #print(key[1:])
            value = float(key[1:])
            if float(one_test_data[attribute_index]) > value:
                if type(second_dic[key]).__name__ == 'dict':
                    res_label = decision_tree_predict(second_dic[key], attribute_labels, one_test_data)
                else:
                    res_label = second_dic[key]
 
        else:
            if one_test_data[attribute_index] == key:
                if type(second_dic[key]).__name__ =='dict':
                    res_label = decision_tree_predict(second_dic[key], attribute_labels, one_test_data)
                else:
                    res_label = second_dic[key]
    return res_label

if __name__ == '__main__':
    precision_list = []
    recall_list = []
    for i in range(1,20):
        # 数据预处理（产生训练数据和测试数据）
        filename='dataset.txt'
        df_train_filename,df_test_filename = pretreatment_dataset(filename, 0.05*i)
        train_data, labels = read_dataset(df_train_filename)
        print(u"训练数据集长度",len(train_data))
        print ("Ent(D):",calc_info_D(train_data))

        print(u"下面开始创建相应的决策树-------")
        # 拷贝，createTree会改变labels
        labels_tmp = labels[:]
        decision_tree= C45_createTree(train_data,labels_tmp)
        test_data = read_dataset(df_test_filename,istrain=False)
        all_datas_length = len(test_data)
        TP,FN,FP,TN = 0,0,0,0
        #计算准确率
        for ite in range(all_datas_length):
            classify_lables = decision_tree_predict(decision_tree,labels,test_data[ite])
            origin_lables = test_data[ite][-1]
            if origin_lables == '1' and classify_lables == '1':
                TP += 1
            elif origin_lables == '1' and classify_lables == '0':
                FN += 1
            elif origin_lables == '0' and classify_lables == '1':
                FP += 1
            elif origin_lables == '0' and classify_lables == '0':
                TN += 1
        # print(TP,FN,FP,TN)
        # 查准率、查全率
        precision_p = TP / (TP + FP)
        recall_p = TP / (TP + FN)
        precision_list.append(100*precision_p)
        recall_list.append(100*recall_p)
        # print('测试集大小%d，准确率为:%.1f%%%.1f%%'%(len(test_data), 100*precision_p, 100*recall_p))
    x = range(5,100,5)
    plt.figure(figsize=(8,6), dpi=80)
    plt.title("C4.5 decision tree")
    plt.xlabel("train(%)") 
    plt.ylabel("%") 
    plt.plot(x,precision_list, label="precision")
    plt.plot(x,recall_list, label="recall")
    plt.legend(loc='upper right')
    plt.ylim(0,100)
    plt.show()

        