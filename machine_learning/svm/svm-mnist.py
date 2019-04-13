from scipy.io import loadmat

import numpy as np
from numpy import linalg
import cvxopt

import matplotlib.pyplot as plt
import pylab as pl

def generate_data(cut_idx, end_idx):
    m1 = loadmat("./data/mnist_train.mat")
    m2 = loadmat("./data/mnist_train_labels.mat")
    # print(m1.keys(),m2.keys())
    usps_all = np.array(m1['mnist_train'],dtype=float)/255
    usps_all_labels = np.array(m2['mnist_train_labels'],dtype=float).flatten()
    
    # 切分训练与测试数据集
    # cut_idx = 2000
    # end_idx = 3000
    # df_train, df_test = usps_all[:cut_idx], usps_all[cut_idx:]
    df_train, df_test = usps_all[:cut_idx], usps_all[cut_idx:end_idx]
    # df_train_labels, df_test_labels = usps_all_labels[:cut_idx], usps_all_labels[cut_idx:]
    df_train_labels, df_test_labels = usps_all_labels[:cut_idx], usps_all_labels[cut_idx:end_idx]
    print(set(usps_all_labels))
    return df_train,df_train_labels,df_test,df_test_labels

def split_train_dict(df_train,df_train_labels):
    train_dict=dict()
    for feature,label in zip(df_train,df_train_labels):
        if label not in train_dict:
            train_dict[label] = []
        train_dict[label].append(feature)
    # print(train_dict.keys())
    return train_dict

def recombine(key_a, feature_list_a, key_b, feature_list_b):
    feature_a = np.array(feature_list_a)
    # labels_a = np.ones(len(feature_list_a))*(1)
    labels_a = np.ones(len(feature_list_a))*(key_a)
    feature_b = np.array(feature_list_b)
    # labels_b = np.ones(len(feature_list_b))*(-1)
    labels_b = np.ones(len(feature_list_b))*(key_b)

    key_train = np.vstack((feature_a, feature_b))
    key_train_labels = np.hstack((labels_a, labels_b))
    return key_train, key_train_labels


# 实现SVM算法的类
class SVM(object):
    def __init__(self,kernel_type = 'liner',p=3,C = None):
        self.kernel_type = kernel_type
        self.p = p
        self.C = C

    def kernelGen(self,x,y):
        if self.kernel_type is 'liner':
            return np.dot(x,y)
        elif self.kernel_type is 'polynomial':
            return (1 + np.dot(x, y)) ** self.p
        else :
            return np.exp(-linalg.norm(x-y)**2 / (2 * (self.p ** 2)))

    def fit(self,X,y):
        row,col = np.shape(X)
        K = np.zeros([row,row])
        for i in range (row):
            for j in range(row):
                K[i,j] = self.kernelGen(X[i],X[j])

        P = cvxopt.matrix(np.outer(y,y)*K)
        q = cvxopt.matrix(np.ones(row)*-1)
        A = cvxopt.matrix(y,(1,row))
        b = cvxopt.matrix(0.0)
        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(row) * -1))
            h = cvxopt.matrix(np.zeros(row))
        else :
            C = float(self.C)
            tmp1 = np.diag(np.ones(row) * -1)
            tmp2 = np.identity(row)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(row)
            tmp2 = np.ones(row) *C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        alafa = np.ravel(solution['x'])
        # print(alafa)
        sv = alafa>1e-4
        # print(sv)
        index = np.arange(row)[sv]
        self.alafa = alafa[sv]
        # print(self.alafa)
        self.sv_x = X[sv]  # sv's data
        self.sv_y = y[sv]  # sv's labels
        print("%d support vectors out of %d points" % (len(self.alafa), row))

        self.b = 0
        for n in range(len(self.alafa)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.alafa * self.sv_y * K[index[n],sv])
        if self.alafa.size > 0:
            self.b /= len(self.alafa)

    def project(self,X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for alafa, sv_y, sv_x in zip(self.alafa, self.sv_y, self.sv_x):
                s += alafa * sv_y * self.kernelGen(X[i],sv_x)
            y_predict[i] = s
        return y_predict + self.b

    def predict(self,X):
        return np.sign(self.project(X))

class TestAndTrain(object):
    def __init__(self,kernel_type = 'liner',p=3,C=None,a=1,b=-1):
        self.a = a
        self.b = b
        self.clf = SVM(kernel_type, p, C)

    def trainSVM (self,X_train,y_train):
        y_train = np.where(y_train == self.a,1.0,-1.0)
        print(X_train)
        self.clf.fit(X_train, y_train)

    def predictSVM(self,X_test):
        y_predict = self.clf.predict(X_test)
        y_predict = np.where(y_predict == 1.0,self.a,self.b)
        return y_predict
        # self.plot_contour(X_train[y_train==1], X_train[y_train==-1], clf,kernel_type,p)

def start_train(cut_idx, end_idx):

    # tt.trainSVM(X_train,y_train,X_test,y_test,'liner')
    # tt.trainSVM(X_train,y_train,X_test,y_test,'liner',3,0.5)
    # tt.trainSVM(X_train,y_train,X_test,y_test,'gaussion',5)

    X_train,y_train,X_test,y_test = generate_data(cut_idx, end_idx)
    train_dict = split_train_dict(X_train,y_train)
    # print(train_dict.keys())
    SVM_dict = {}
    train_keys = train_dict.keys()
    print("开始训练...")
    # 抽取数据1对1训练：
    for vs_a in train_keys:
        for vs_b in train_keys:
            if vs_a == vs_b:
                continue
            key_train, key_train_labels = recombine(vs_a, train_dict[vs_a], vs_b, train_dict[vs_b])
            # tt.trainSVM(X_train,y_train,X_test,y_test,'liner',3,0.5, a=vs_a, b=vs_b)
            # tt = TestAndTrain('liner',3,0.5, a=vs_a, b=vs_b)
            tt = TestAndTrain('gaussion',5, a=vs_a, b=vs_b)
            tt.trainSVM(key_train,key_train_labels)
            SVM_dict[frozenset((vs_a,vs_b))] = tt

    # 用测试数据对其测试结果
    # 构建DAG
    # 使用frozenset保存当前划分的类别，当前为[0,2,...,9]
    print("开始测试")
    result = np.array([frozenset(range(0,10)) for _ in range(len(X_test))])
    predict_array = np.column_stack((np.array(X_test),result))
    for vs_a in range(0,10):
        for vs_b in range(9,-1,-1):
            # print(vs_a, vs_b)
            # print(predict_array[1:10,-1])
            if vs_a == vs_b:
                continue
            # 提取出含有当前1对1分类的类别元素，并记录位置
            index_array = []
            for itera,item in enumerate(predict_array[:,-1]):
                if item.issuperset(frozenset((vs_a,vs_b))):
                    index_array.append(itera)
            test_data = predict_array[index_array,:-1]
            tt = SVM_dict[frozenset((vs_a,vs_b))]
            y_predict = tt.predictSVM(test_data)
            # 根据结果减少划分列别的可能性
            for ind, pred in zip(index_array, y_predict):
                old = predict_array[ind][-1]
                new = set(old)
                new.remove(vs_a if pred == vs_b else vs_b)
                predict_array[ind][-1] = frozenset(new)
    # print("结果",predict_array[:,-1])
    y_pred = [list(x)[0] for x in predict_array[:,-1]]
    correct = np.sum(y_pred == y_test)
    print("%d out of %d predictions correct" % (correct, len(y_pred)))
    print("正确率为：" , correct/len(y_pred))
    return correct/len(y_pred)


if __name__ == "__main__":
    correct_ratio_list = []
    for i in range(1000,4000,300):
        correct_ratio = start_train(i, i+1200)
        correct_ratio_list.append(correct_ratio)
    x = range(1000,4000,300)
    plt.figure(figsize=(8,6), dpi=80)
    plt.title("SVM MNIST")
    plt.xlabel("train(%)") 
    plt.ylabel("train size") 
    plt.plot(x,correct_ratio_list, label="precision")
    plt.legend(loc='upper right')
    plt.xlim(1000,4000)
    plt.show()

