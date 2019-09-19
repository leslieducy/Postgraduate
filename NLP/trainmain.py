import pandas as pd
import numpy as np

import random

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 读取汉字编码表
def get_chinese_codes():
    words_list = pd.read_csv('wordsno.csv',index_col=0,low_memory=False,encoding='utf-8')
    return words_list.iloc[:,0].values.tolist()

def load_data(dataset_path='./couplet/train/'):
    # 读取输入和输出,汉字编码表
    in_txt = pd.read_csv(dataset_path+'in.csv',index_col=0,low_memory=False,encoding='utf-8')
    out_txt = pd.read_csv(dataset_path+'out.csv',index_col=0,low_memory=False,encoding='utf-8')
    chinese_codes = get_chinese_codes()
    in_num_txt = in_txt.applymap(lambda x:chinese_codes.index(x) if not pd.isna(x) else x)
    out_num_txt = out_txt.applymap(lambda x:chinese_codes.index(x) if not pd.isna(x) else x)
    features = in_num_txt.values
    labels = out_num_txt.values
    print("vector_len:", features.shape[0])
    return features, labels
    # for item in range(0,400):
    #     img = Image.open(dataset_path+str(item+1)+'.jpg')
    #     # 将PIL.Image图片类型转成numpy.array
    #     img_ndarray = np.asarray(img, dtype='float32') / 256
    #     features.append(img_ndarray)
    #     labels.append(int(item/40))
    # # print(np.array(features),np.array(labels))
    # return np.array(features),np.array(labels, dtype='int64')

def load_test(dataset_path='./test_data/'):
    features = []
    # 遍历训练文件夹中1到400的图片
    for item in range(0,50):
        img = Image.open(dataset_path+str(item+1)+'.jpg')
        # 将PIL.Image图片类型转成numpy.array
        img_ndarray = np.asarray(img, dtype='float32') / 256
        features.append(img_ndarray)
    return np.array(features)



class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 64),
            # (-1,1)
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 12),
            nn.Tanh(),
            nn.Linear(12, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 28*28),
            # (0,1)
            nn.Sigmoid(),
        )
       

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded



if __name__ == "__main__":
    # 加载数据
    features, labels = load_data(dataset_path='./couplet/train/')
    train_data = Data.TensorDataset(features, labels)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    autoencoder = AutoEncoder()
    autoencoder = autoencoder.cuda()      # Moves all model parameters and buffers to the GPU.
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    
    # training and testing
    for epoch in range(EPOCH):
        for step, (x, _) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
            b_x = np.eye(in_num_txt.values.shape[0])[b_x].astype(np.float32).cuda().view(-1, 28*28)
            b_y = np.eye(out_num_txt.values.shape[0])[b_y].astype(np.float32).cuda().view(-1, 28*28)
            # b_x = x.cuda().view(-1, 28*28)
            # b_y = x.cuda().view(-1, 28*28)
            encoded, decoded = autoencoder(b_x)               # rnn output
            loss = loss_func(decoded, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients


# txt转存csv(额外工作)
    # trans_csv(dataset_path='./couplet/train/',file_name='in')
    # trans_csv(dataset_path='./couplet/train/',file_name='out')
    # trans_csv(dataset_path='./couplet/test/',file_name='in')
    # trans_csv(dataset_path='./couplet/test/',file_name='out')
def trans_csv(dataset_path='./couplet/train/',file_name='in'):
    in_file = open(dataset_path + file_name + '.txt', "r",encoding='utf-8') 
    row = in_file.readlines() 
    in_txt_list = [] 
    for line in row: 
        line = list(line.strip().split(' ')) 
        in_txt_list.append(line) 
    df = pd.DataFrame(in_txt_list)
    df.to_csv(dataset_path + file_name + '.csv',encoding='utf_8_sig')
    
# 对所有的编号,包括逗号也编号
def code_all_words():
    words_list=[]
    
    dataset_path='./couplet/train/'
    in_txt = pd.read_csv(dataset_path+'in.csv',index_col=0,low_memory=False,encoding='utf-8')
    for item in in_txt.values.flat:
        if item not in words_list and not pd.isna(item):
            words_list.append(item)
    out_txt = pd.read_csv(dataset_path+'out.csv',index_col=0,low_memory=False,encoding='utf-8')
    for item in out_txt.values.flat:
        if item not in words_list and not pd.isna(item):
            words_list.append(item)
    dataset_path='./couplet/test/'
    in_txt = pd.read_csv(dataset_path+'in.csv',index_col=0,low_memory=False,encoding='utf-8')
    for item in in_txt.values.flat:
        if item not in words_list and not pd.isna(item):
            words_list.append(item)
    out_txt = pd.read_csv(dataset_path+'out.csv',index_col=0,low_memory=False,encoding='utf-8')
    for item in out_txt.values.flat:
        if item not in words_list and not pd.isna(item):
            words_list.append(item)

    df = pd.DataFrame(words_list)
    df.to_csv('wordsno.csv',encoding='utf_8_sig')