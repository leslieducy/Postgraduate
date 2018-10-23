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
    test_data = Seg3D(opt.test_data_root,test=True)
    test_dataloader = DataLoader(test_data,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers)
    results = []
    frame = [pt for pt in os.listdir(os.path.join(opt.test_data_root, opt.train_data_pts))][opt.test_data_root_start:]
    print(len(frame))
    for ii,(data, extra) in enumerate(test_dataloader):
        input = t.Tensor(data)
        if opt.use_gpu: input = input.cuda()
        score = model(input)
        for si in range(len(score)):
            results = []
            probability = t.nn.functional.softmax(score[si], dim=1).data.tolist()
            for probability_ in probability:
                label = probability_.index(max(probability_))
                results.append([label])
                # results.append(probability_)
            index = extra[0]
            frame_batch_size = extra[1]
            results = results[:frame_batch_size[si].data]
            write_csv(results,os.path.join(opt.test_data_root, opt.train_data_category, frame[int(index[si].data/20)]))
            print("第",opt.test_data_root_start + int(index[si].data/20) + 1,"个文件，第",int(index[si].data%20) + 1,"批")


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
    train_data = Seg3D(opt.train_data_root,train=True)
    # val_data = Seg3D(opt.train_data_root,train=False)
    train_dataloader = DataLoader(train_data,opt.batch_size,
                        shuffle=True,num_workers=opt.num_workers)
    # val_dataloader = DataLoader(val_data,opt.batch_size,
    #                     shuffle=False,num_workers=opt.num_workers)
    
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
            # train model 
            print(ii)
            input = Variable(data)
            target = Variable(label).type(t.LongTensor) 
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()

            optimizer = t.optim.Adam(model.parameters(),lr = lr,weight_decay = opt.weight_decay)
            optimizer.zero_grad()
            print("模型数据载入完毕")
            score = model(input)
            # loss_func = t.nn.CrossEntropyLoss()
            # output = Variable(t.randn(10, 120).float())
            # target = Variable(t.FloatTensor(10).uniform_(0, 120).long())
            # loss = loss_func(score, target)
            criterion = t.nn.CrossEntropyLoss()
            print(len(score[0]))
            # for si in range(len(score)):
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()
            
            # meters update and visualize
            loss_meter.add(loss.item())
            print("损失：",loss_meter.value()[0])
            # confusion_matrix.add(score.data, target.long().data)

            # if ii%opt.print_freq==opt.print_freq-1:
            #     # vis.plot('loss', loss_meter.value()[0])
                
            #     # 如果需要的话，进入debug模式
            #     if os.path.exists(opt.debug_file):
            #         import ipdb
            #         ipdb.set_trace()
            
            # update learning rate
            if loss_meter.value()[0] > previous_loss:          
                lr = lr * opt.lr_decay
                # 第二种降低学习率的方法:不会有moment等信息的丢失
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
        model.save()

        # validate and visualize

        # val_cm,val_accuracy = val(model,val_dataloader)

        # vis.plot('val_accuracy',val_accuracy)
        # vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
                    # epoch = epoch,loss = loss_meter.value()[0],val_cm = str(val_cm.value()),train_cm=str(confusion_matrix.value()),lr=lr))
        

        

        previous_loss = loss_meter.value()[0]

def val(model,dataloader):
    from config import opt
    ''' 
    计算模型在验证集上的准确率等信息
    '''
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input)
        val_label = Variable(label.type(t.LongTensor))
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input)
        confusion_matrix.add(score.data.squeeze(), label.type(t.LongTensor))

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy

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
