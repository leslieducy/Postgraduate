from PIL import Image
import os
import pickle
import numpy as np
import time
from random import shuffle
# from sklearn.externals import joblib
 
 
def create_dataset(setname, write_to_file_root, channel, image_suffix, num_test, step_every_print=500):
    """ Create znyp Dataset
    input:
        setname: the file should be followed the directory tree (e.g., './classes')
        write_to_file_root: the root location of output file (e.g., './')
        channel:
            1: there are only 1 channel
            3: the image have three channels (R, G, B)
        image_suffix: the image suffix of the dataset (e.g., '.jpg')
        step_every_print: number of steps required to print the imformation
    """
    Filelist = []
    for _, dirnames, _ in os.walk(setname):
        for dirname in dirnames:
            for root, _, filenames in os.walk(os.path.join(setname,dirname)):
                for filename in filenames:
                    if filename.endswith(image_suffix):
                        Filelist.append(os.path.join(root, filename))
 
    shuffle(Filelist)
    test_Filelist = Filelist[:10]
    val_Filelist = Filelist[10:num_test]
    train_Filelist = Filelist[num_test:]
 
    _pickel_dataset(test_Filelist, write_to_file_root, 'testsh', channel, step_every_print)
    _pickel_dataset(val_Filelist, write_to_file_root, 'valsh', channel, step_every_print)
    _pickel_dataset(train_Filelist, write_to_file_root, 'trainsh', channel, step_every_print)
 
 
def _pickel_dataset(filenames, write_to_file_root, category, channel, step_every_print=500):
    """ Pickel Dataset
    input:
        filenames: a list of image location
        write_to_file_root: the root location of output file (e.g., './')
        category: dataset category
        channel:
            1: there are only 1 channel
            3: the image have three channels (R, G, B)
        step_every_print: number of steps required to print the imformation
    return
        a Python "pickled" object
    """
    start_time = time.time()
    data = []
    labels = []
    num_images = 0
 
    for idx, filename in enumerate(filenames):
        if idx % step_every_print == 0:
            print('Current append image : %s' % filename)
        im = Image.open(filename)
        im = im.resize((300, 300),Image.ANTIALIAS)
        im = (np.array(im))
        H, W = im.shape[0], im.shape[1]

        if channel == 1:
            r = im.flatten()
            g = []
            b = []
        elif channel == 3:
            # get pixel from red channel, then green then blue
            r = im[:, :, 0].flatten()
            g = im[:, :, 1].flatten()
            b = im[:, :, 2].flatten()
        else:
            raise Exception('The channel for the image should be 1 or 3')
 
        num_images += 1
        # append the label
        label = int(filename.split(os.sep)[-1][1])
        labels.append(label)
        # append the pixel
        data += (list(r) + list(g) + list(b))
    
    # convert the list to numpy
    data = np.array(data, np.uint8)
    datadict = {'data': data, 'labels': labels, 'height': 300, 'width': 300, 'channel': channel, 'num_images': num_images}
 
    # write to pickle
    outname = os.path.join(write_to_file_root, category + '_data' + '.pkl')
    f = open(outname, 'wb')
    pickle.dump(datadict, f, True)
    f.close()
    end_time = time.time()
    print('%s set took %.2f seconds' % (category, end_time - start_time))
 
    # write to bin
    # output_file = open('data_batch_1.bin', 'wb')
    # data.tofile(output_file)
    # output_file.close()
 
if __name__ == '__main__':
    # create dataset
    create_dataset(setname='./training_image', write_to_file_root='./monkey_dataset', channel=3, image_suffix='.jpg', num_test=50)
 