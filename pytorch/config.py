#coding:utf8
import warnings


class DefaultConfig(object):
    env = 'default' # visdom 环境
    model = 'EasyNet' # 使用的模型，名字必须与models/__init__.py中的名字一致
    
    train_data_root = './data/train/' # 训练集存放路径
    # test_data_root = "./data/test/" # 测试集存放路径
    test_data_root = "./data/test/" # 测试集存放路径
    test_data_root_start = 0 # 测试集存放路径

    data_pts = 'pts'
    data_intensity = 'intensity'
    data_category = 'category'
    # load_model_path = 'checkpoints/model.pth' # 加载预训练的模型的路径，为None代表不加载
    load_model_path = None # 加载预训练的模型的路径，为None代表不加载
    test_model_path = 'checkpoints/easynet_1027_.pt' # 加载测试时的模型的路径

    batch_size = 512 # batch size
    use_gpu = False # user GPU or not
    num_workers = 4 # how many workers for loading data
    print_freq = 10 # print info every N batch

    debug_file = '/tmp/debug' # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'
    
    max_epoch = 1
    lr = 0.2 # initial learning rate
    lr_decay = 0.95 # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4 # 损失函数


def parse(self,kwargs):
        '''
        根据字典kwargs 更新 config参数
        '''
        for k,v in kwargs.items():
            if not hasattr(self,k):
                warnings.warn("Warning: opt has not attribut %s" %k)
            setattr(self,k,v)

        print('user config:')
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k,getattr(self,k))


DefaultConfig.parse = parse
opt =DefaultConfig()
# opt.parse = parse
