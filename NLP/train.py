import pandas as pd
import numpy as np


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
    features = in_num_txt.values.astype(np.float32)
    labels = out_num_txt.values.astype(np.float32)
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
if __name__ == "__main__":
    # 加载数据
    features, labels = load_data(dataset_path='./couplet/train/')


# 定义神经网络
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)











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