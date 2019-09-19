import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch


# gpu/cpu
cuda = True if torch.cuda.is_available() else False

# 1、输入处理 2、定义模型 3、开始训练

# 创建解析对象
parser = argparse.ArgumentParser()
# 向解析对象中添加命令行参数和选项
# epoch = 200，批大小 = 64，学习率 = 0.0002，衰减率 = 0.5/0.999，线程数 = 8，设置图片 潜在维数 = 100，样本尺寸 = 28 * 28，通道数 = 1，样本显示的间隔 = 400
parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
# 解析参数
opt = parser.parse_args()
print(opt)
# 输入训练样本 Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
# dataset:加载数据的数据集；batch_size:每批次加载的数据量；shuffle：默认false，若为True，表示在每个epoch打乱数据；sampler：定义从数据集中绘制示例的策略,如果指定，shuffle必须为False  
os.makedirs("./mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# 图片转数字相关的参数（通道数，长，宽）
img_shape = (opt.channels, opt.img_size, opt.img_size)
# 生成器（接收图片，生成图片）
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 定义每层网络相同的网络结构（线性层，BN层，激活层）
        def block(in_feat, out_feat, normalize=True):
            # 这里简单的只对输入数据做线性转换，使用BN，添加LeakyReLU非线性激活层
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # 创建生成器网络模型
        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    # 前向
    def forward(self, z):
        # 生成假样本
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        # 返回生成图像
        return img

# 判别器（接收图片，返回类别概率）
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            # 因需判别真假，这里使用Sigmoid函数给出标量的判别结果
            nn.Sigmoid(),
        )
    # 判别
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


# Initialize generator and discriminator 
generator = Generator()
discriminator = Discriminator()
# Loss function 损失函数：二分类交叉熵函数
adversarial_loss = torch.nn.BCELoss()
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()



# Optimizers 优化器，G和D都使用Adam
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 训练开始并保存结果
# 如果根目录下不存在images文件夹，则创建images存放生成图像结果
os.makedirs("images", exist_ok=True)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
for epoch in range(opt.n_epochs):
    # dataloader=[step,(x,y)]
    for step, (imgs, _) in enumerate(dataloader):
        # Adversarial ground truths 真与假数组
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # 输入真实图片的数据形式转成Tensor打包成Variable
        real_imgs = Variable(imgs.type(Tensor))
        # -----------------
        #  Train Generator
        # 梯度清除
        optimizer_G.zero_grad()
        # 随机生成一张图片的数据形式
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        # 通过G得到一批生成的图片
        gen_imgs = generator(z)
        # 用D判断生成的图片并计算损失函数，再反向传递更新G
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        optimizer_D.zero_grad()
        # 损失函数包括两个：真实与判断为真，G生成与判断为假两个方面，再求平均，反向更新D
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # 每批次的输出
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, step, len(dataloader), d_loss.item(), g_loss.item())
        )
        # 保存结果
        batches_done = epoch * len(dataloader) + step
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:36], "images/%d.png" % batches_done, nrow=6, normalize=True)