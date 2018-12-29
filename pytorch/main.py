#coding:utf8
from config import opt
import os
import torch as t
import models
from data.dataset import Seg3D
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
from utils.visualize import Visualizer
from tqdm import tqdm
import fire

def test(**kwargs):
    from config import opt
    cuda = t.device("cuda")
    opt.parse(kwargs)
    # import ipdb
    # ipdb.set_trace()
    # configure model
    model = getattr(models, opt.model)().eval()
    if opt.test_model_path:
        model.load(opt.test_model_path)
    if opt.use_gpu: 
        try:model.cuda();print("使用成功")
        except:print("使用失败")
    # data
    frame_file = [pt for pt in os.listdir(os.path.join(opt.test_data_root, opt.data_pts))]
    for file_path in frame_file:
        # file_path = "000d4e61-1203-4819-98ab-afbb619010b0_channelVELO_TOP.csv"
        test_data = Seg3D(file_path,test=True)
        test_dataloader = DataLoader(test_data,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers)
        print("正在处理文件：",file_path,"，处理中...")
        for ii,(data,_) in enumerate(test_dataloader):
            input = Variable(data)
            if opt.use_gpu:
                input = input.cuda()
            input = input.view(-1, 1, 4)
            score = model(input)
            results = []
            probability = t.nn.functional.softmax(score, dim=1).data.tolist()
            for probability_ in probability:
                label = probability_.index(max(probability_))
                results.append([label])
            write_csv(results,os.path.join(opt.test_data_root, opt.data_category, file_path))
        
        print("文件：",file_path,"，处理完毕！")

    # return results

def write_csv(results,file_name):
    import csv
    with open(file_name,'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results)
    
def train(**kwargs):
    from config import opt
    # 根据命令行参数更新配置
    opt.parse(kwargs)
    # vis = Visualizer(opt.env)

    # step1: configure model
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()
    # step2: data
    frame_file = [pt for pt in os.listdir(os.path.join(opt.train_data_root, opt.data_pts))]
    for file_path in frame_file:
        print("正在训练文件：",file_path,",请稍等...")
        # file_path = "000d4e61-1203-4819-98ab-afbb619010b0_channelVELO_TOP.csv"
        train_data = Seg3D(file_path,train=True)
        train_dataloader = DataLoader(train_data,opt.batch_size,
                            shuffle=True,num_workers=opt.num_workers)
        print("数据加载成功")
        # step3: criterion and optimizer
        criterion = t.nn.CrossEntropyLoss()
        lr = opt.lr
        # step4: meters
        loss_meter = meter.AverageValueMeter()
        confusion_matrix = meter.ConfusionMeter(2)
        previous_loss = 1e100
        print("训练开始")
        # train
        for epoch in range(opt.max_epoch):
            loss_meter.reset()
            confusion_matrix.reset()
            for ii,(data,label) in enumerate(train_dataloader):
                input = Variable(data)
                if opt.use_gpu:
                    input = input.cuda()
                    target = target.cuda()
                # train model 
                # 数据格式暂存问题（待修改）
                input = input.view(-1, 1, 4)
                target = Variable(label).type(t.LongTensor) 

                optimizer = t.optim.Adam(model.parameters(),lr = lr,weight_decay = opt.weight_decay)
                optimizer.zero_grad()
                print("模型数据载入完毕")
                score = model(input)
                loss = criterion(score, target)
                loss.backward()
                optimizer.step()
                
                # meters update and visualize
                loss_meter.add(loss.item())
                print("损失：",loss_meter.value()[0])
                
                # update learning rate
                if loss_meter.value()[0] > previous_loss:          
                    lr = lr * opt.lr_decay
                    # 第二种降低学习率的方法:不会有moment等信息的丢失
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                
            model.save()
            previous_loss = loss_meter.value()[0]

def help():
    from config import opt
    '''
    打印帮助的信息： python file.py help
    '''
    
    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__=='__main__':
    fire.Fire()
