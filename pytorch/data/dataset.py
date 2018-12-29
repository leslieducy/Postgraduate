import os
from torch.utils import data
import numpy as np
from torchvision import  transforms as T
import csv
import torch
from config import opt
import math


class Seg3D(data.Dataset):

    def __init__(self, frame_path, transforms=None, train=True, test=False):
        # 主要目标： 获取所有三维点阵，并根据训练，验证，测试划分数据
        self.test = test
        if self.test:
            csv_file_pts = open(os.path.join(opt.test_data_root, opt.data_pts, frame_path))
            csv_file_intensity = open(os.path.join(opt.test_data_root, opt.data_intensity, frame_path))
            # csv_file_category = open(os.path.join(opt.test_data_root, opt.data_category, frame_path))
        else:
            csv_file_pts = open(os.path.join(opt.train_data_root, opt.data_pts, frame_path))
            csv_file_intensity = open(os.path.join(opt.train_data_root, opt.data_intensity, frame_path))
            csv_file_category = open(os.path.join(opt.train_data_root, opt.data_category, frame_path))
        
        csv_file_pts_line = csv.reader(csv_file_pts)
        csv_file_intensity_line = csv.reader(csv_file_intensity)

        data = []
        for pts_line in csv_file_pts_line:
            pts_num = []
            for pts_double in pts_line:
                pts_num.append(float(pts_double))
            data.append(pts_num)
        i = 0
        for intensity_line in csv_file_intensity_line:
            data[i].append(float(intensity_line[0]))
            i += 1

        label = []
        if self.test:
            for _ in range(0, i):
                label.append(0)
        else: 
            csv_file_category_line = csv.reader(csv_file_category)
            for category_line in csv_file_category_line:
                label.append(int(category_line[0]))

        self.frame = data
        self.frame_label = label
        self.frame_num = i
        # 洗牌
        # np.random.seed(100)
        # self.frame = np.random.permutation(self.frame)

        # if transforms is None:
        #     normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
        #                             std=[0.229, 0.224, 0.225])

        #     if self.test or not train:
        #         self.transforms = T.Compose([
        #             T.Resize(224),
        #             T.CenterCrop(224),
        #             T.ToTensor(),
        #             normalize
        #         ])
        #     else:
        #         self.transforms = T.Compose([
        #             T.Resize(256),
        #             T.RandomResizedCrop(224),
        #             T.RandomHorizontalFlip(),
        #             T.ToTensor(),
        #             normalize
        #         ])

    def __getitem__(self, index):
        # print(self.frame_label[index])
        if self.test:
            # extra = [index, frame_batch_size]
            return torch.Tensor(self.frame[index]), []
        else:
            return torch.Tensor(self.frame[index]), self.frame_label[index]
        
        

    def __len__(self):
        # print("frame_num",self.frame_num)
        return self.frame_num
