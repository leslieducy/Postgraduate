import os
from torch.utils import data
import numpy as np
from torchvision import  transforms as T
import csv
import torch
from config import opt
import math


class Seg3D(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):
        # 主要目标： 获取所有三维点阵，并根据训练，验证，测试划分数据
        self.test = test
        self.frame = [pt for pt in os.listdir(os.path.join(root, opt.train_data_pts))][opt.test_data_root_start:]
        self.cut_num = 20
        self.frame_num = len(self.frame) * self.cut_num
        # 洗牌
        np.random.seed(100)
        self.frame = np.random.permutation(self.frame)

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        # 一次返回1/20帧
        frame_path = self.frame[int(index/self.cut_num)]
        frame_batch = index % self.cut_num
        # frame_path2 = self.frame[index + 1]
        
        if self.test:
            csv_file_pts = open(os.path.join(opt.test_data_root, opt.train_data_pts, frame_path))
            csv_file_intensity = open(os.path.join(opt.test_data_root, opt.train_data_intensity, frame_path))
            # csv_file_category = open(os.path.join(opt.test_data_root, opt.train_data_category, frame_path))
            csv_file_length = open(os.path.join(opt.test_data_root, opt.train_data_pts, frame_path))
        else:
            csv_file_pts = open(os.path.join(opt.train_data_root, opt.train_data_pts, frame_path))
            csv_file_intensity = open(os.path.join(opt.train_data_root, opt.train_data_intensity, frame_path))
            csv_file_category = open(os.path.join(opt.train_data_root, opt.train_data_category, frame_path))
            csv_file_length = open(os.path.join(opt.train_data_root, opt.train_data_pts, frame_path))
            csv_file_category_line = csv.reader(csv_file_category)

        csv_file_pts_line = csv.reader(csv_file_pts)
        csv_file_intensity_line = csv.reader(csv_file_intensity)

        # 1/cut_num帧的大小计算，设定分帧上下位置
        count = 0
        for pts_line in csv_file_length:
            count += 1
        
        frame_batch_size = math.ceil(count / self.cut_num)
        minCol = frame_batch * frame_batch_size
        maxCol = frame_batch * frame_batch_size + frame_batch_size

        data = []
        # 计算保存的位置
        pos = 0
        for pts_line in csv_file_pts_line:
            pos += 1
            if pos > minCol and pos <= maxCol:
                pts_num = []
                for pts_double in pts_line:
                    pts_num.append(float(pts_double))
                data.append(pts_num)
        pos = 0
        i = 0
        for intensity_line in csv_file_intensity_line:
            pos += 1
            if pos > minCol and pos <= maxCol:
                data[i].append(float(intensity_line[0]))
                i += 1

        pos = 0
        label = []
        if self.test:
            for _ in range(0, i):
                label.append(0)
        else: 
            for category_line in csv_file_category_line:
                pos += 1
                if pos > minCol and pos <= maxCol:
                    label.append(int(category_line[0]))
        # 57920
        if i < 3000:
            for _ in range(i, 3000):
                data.append([0, 0, 0, 0])
                label.append(0)
        if self.test:
            # 额外返回的是调用的组号和帧的真实大小，方便后期写入
            extra = [index, frame_batch_size]
            return torch.Tensor(data), extra
        else:
            print(torch.Tensor(label))
            return torch.Tensor(data), torch.Tensor(label)

    def __len__(self):
        # print("frame_num",self.frame_num)
        return self.frame_num
