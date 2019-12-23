import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.layers import Embedding
from keras.layers import Conv1D,GlobalAveragePooling1D
from keras.datasets import imdb
from sklearn.metrics import accuracy_score,classification_report
    
# 参数 最大特征数6000 单个句子最大长度400
max_features = 6000
max_length = 400
(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_features)
# print(x_train) # 一堆句子，每个句子有有一堆单词编码
# print(y_train) # 一堆0或1
# print(len(x_train),'train observations')
# print(len(x_test),'test observations')
    
wind = imdb.get_word_index() # 给单词编号，用数字代替单词
revind = dict((k, v) for k, v in enumerate(wind))
# 单词编号:情感词性编号 字典 => 情感词性编号:一堆该词性的单词编号列表
# print(x_train[0])
# print(y_train[0])
    
def decode(sent_list): # 逆映射字典解码 数字=>单词
    new_words = []
    for i in sent_list:
        new_words.append(revind[i])
    comb_words = " ".join(new_words)
    return comb_words
# print(decode(x_train[0]))
    
# 将句子填充到最大长度400 使数据长度保持一致
x_train = sequence.pad_sequences(x_train,maxlen=max_length)
x_test = sequence.pad_sequences(x_test,maxlen=max_length)
print('x_train.shape:',x_train.shape)
print('x_test.shape:',x_test.shape)
    
## Keras框架 深度学习 一维CNN模型
# 参数
batch_size = 32
embedding_dims = 60
num_kernels = 260
kernel_size = 3
hidden_dims = 300
epochs = 3
# 建立模型
model = Sequential()
model.add(Embedding(max_features,embedding_dims,input_length=max_length))
model.add(Dropout(0.2))
model.add(Conv1D(num_kernels,kernel_size,padding='valid',activation='relu',strides=1))
model.add(GlobalAveragePooling1D())
model.add(Dense(hidden_dims))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
    
model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_split=0.2)
    
# 模型预测
y_train_predclass = model.predict_classes(x_train,batch_size=batch_size)
y_test_preclass = model.predict_classes(x_test,batch_size=batch_size)
y_train_predclass.shape = y_train.shape
y_test_preclass.shape = y_test.shape
    
print('\n\nCNN 1D - Train accuracy:',round(accuracy_score(y_train,y_train_predclass),3))
print('\nCNN 1D of Training data\n',classification_report(y_train,y_train_predclass))
print('\nCNN 1D - Train Confusion Matrix\n\n',pd.crosstab(y_train,y_train_predclass,
                    rownames=['Actuall'],colnames=['Predicted']))
print('\nCNN 1D - Test accuracy:',round(accuracy_score(y_test,y_test_preclass),3))
print('\nCNN 1D of Test data\n',classification_report(y_test,y_test_preclass))
print('\nCNN 1D - Test Confusion Matrix\n\n',pd.crosstab(y_test,y_test_preclass,
                    rownames=['Actuall'],colnames=['Predicted']))