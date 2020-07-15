# -*- coding: UTF-8 -*-

"""
Train our RNN on extracted features or images.
"""
#import tensorflow.compat.v1 as tf 
#tf.disable_v2_behavior() 
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger,ReduceLROnPlateau
from models import ResearchModels
from data import DataSet
import time
import os.path
import math
import numpy as np
from keras.utils import to_categorical , Sequence
import tensorflow as tf
from keras.applications.imagenet_utils import preprocess_input
import json
import random
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image

#tf.executing_eagerly()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'



def parse_function(example_proto):
    features = {'image_raw': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)}
    features = tf.io.parse_single_example(example_proto, features)
    #img = tf.image.decode_jpeg(features['image_raw'])
    #img = tf.io.decode_raw(features['image_raw'], tf.uint8)
    featureList = tf.io.decode_raw(features['image_raw'], tf.uint8)
    #print('featureList  type(featureList):', featureList, type(featureList))
    imgList = tf.reshape(featureList, shape=(10, 224, 224, 3))
    label = tf.cast(features['label'], tf.int64)
    return imgList, label
    


#新数据集读取方式
res_size = 224 
batch_size = 16


def getJasonDataInfo(Jason):
    dataDict={}
    with open(Jason, encoding='utf-8') as f:
        #lines = f.readline()    
        #dataDict = json.loads(lines)
        dataDict = json.load(f)
        f.close()
    return dataDict
    
def getDataAndLabel(Jason):
    dataDict = getJasonDataInfo(Jason)
    dataFiles=[]
    labels = []
    for realFile in dataDict.keys():
        fakeDataListDict = dataDict[realFile]
        dataDictKeys = list(fakeDataListDict.keys()) 
        fakeFileKey = random.choice(dataDictKeys)
        dirIndex=random.choice(fakeDataListDict[fakeFileKey])
        
        realFileFullDir = os.path.join(realFile, dirIndex)
        fakeFileFullDir = os.path.join(fakeFileKey, dirIndex)
        dataFiles.append(realFileFullDir)
        dataFiles.append(fakeFileFullDir)
        labels.append(0)
        labels.append(1)
    #-------这段是为了打乱一下------------
    shuffle_list = []
    for i in range(len(dataFiles)):
        shuffle_list.append([dataFiles[i],labels[i]])
    random.shuffle(shuffle_list)
    dataFiles=[]
    labels = []
    for i in range(len(shuffle_list)):
        dataFiles.append(shuffle_list[i][0])
        labels.append(shuffle_list[i][-1])
    #-----------------------------------
    return dataFiles, labels

# def get_data_loader(trainDataJson, valDataJson):
#     train_list, train_label = getDataAndLabel(trainDataJson)
#     test_list, test_label = getDataAndLabel(valDataJson)

#     train_set = Dataset_CRNN(train_list, train_label, transform=transform)
#     valid_set = Dataset_CRNN(test_list, test_label, transform=transform)

#     # train_loader = data.DataLoader(train_set, **params)
#     # valid_loader = data.DataLoader(valid_set, **params)
#     return train_set, valid_set
def read_images(imgsDir):
    X = []
    #beginFramNum = maxFrameIndex - 5
    imgFiles = sorted(os.listdir(imgsDir))
    if len(imgFiles)!=5:
        print(imgFiles)
    try:
        for img in imgFiles:
            imgPath = os.path.join(imgsDir, img)
            if not os.path.exists(imgPath):
                print(imgPath)
                continue
            else:
                #print('imgPath：', imgPath)
                image = Image.open(imgPath)
            imagearray = np.array(image)
            # print(type(imagearray))
            # print(imagearray.shape)
            image =preprocess_input(imagearray,mode="tf")
            X.append(image)
        
        X = np.array(X)
        
    except Exception as e:
        print("Prediction error on imgPath %s: %s" % (imgPath, str(e)))
        
    return X

def get_data_loader(DataJson,steps_per_epoch):
    train_list, train_label = getDataAndLabel(DataJson)
    shuffle_list = []
    for i in range(math.floor(len(train_list)/batch_size)):
        little_list = []
        for j in range(batch_size):
            little_list.append([train_list[i*batch_size+j],train_label[i*batch_size+j]])
        shuffle_list.append(little_list)
    random.shuffle(shuffle_list)
    Iterator = iter(shuffle_list)
    count = 0
    while True:
        count += 1
        if count >= steps_per_epoch:
            random.shuffle(shuffle_list)
            Iterator = iter(shuffle_list)
            print('again')
            count = 0
        new_little_list = next(Iterator)
        final_X= np.zeros((batch_size,5,res_size,res_size,3), dtype=float)
        final_y = []
        for i in range(len(new_little_list)):
            new_train_path = new_little_list[i][0]
            new_train_label = new_little_list[i][-1]
            X = read_images(new_train_path)
            # print(type(X))
            # print(len(X))
            final_X[i] = np.array(X)
            final_y.append(new_train_label)
        final_y = np.array(final_y)
        yield final_X, final_y 

#-----------到这里都是军治哥torch下的结构-----------

def myIter(batch_size, tfRec):  
    dataset = tf.data.TFRecordDataset(tfRec)  #  
    dataset = dataset.map(parse_function)
    dataset = dataset.shuffle(buffer_size=2000)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    Iterator = iter(dataset)
    while True:
        img, label = next(Iterator)
        x = np.array(img)
        x = preprocess_input(x,mode="tf")
        y = np.array((label))
        yield x, y 
        
        

def train(data_type, seq_length, model, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=32, nb_epoch=100):
    # Helper: Save the model.

    checkpointer = ModelCheckpoint(
        filepath=os.path.join('data', 'checkpoints', model + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('data', 'logs', model))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=10)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'logs', model + '-' + 'training-' + \
       str(timestamp) + '.log'))

    rp = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                               verbose=1)

    # trainTxtFile = '../../seq10Train.txt'
    # testTxtFileList = '../../seq10Test.txt'
    
    '''
    # Get the data and process it.
    if image_shape is None:
        
        data = DataSet(     
            trainTxtFile,
            testTxtFileList,        
            seq_length=seq_length,
            class_limit=class_limit            
        )
    else:
        
        data = DataSet(
            trainTxtFile,
            testTxtFileList,
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )   
    '''
#--------------------------------------tfrecord读取方式-----------------------------------------------
    # testTfRec = '../../../zhaijunzhi/code_dev/lrcn_img_cls/data/kailu_data_multi_scales/newsplit/seq10Val_10frameOffset.tfrecord'
    # testTfRec =   '/mnt/media2/data/deepfake/02_tfrecord/224x224facenet_seq10Val.tfrecord'    
    # testIter  = myIter(batch_size, testTfRec)
    
    # trainTfRec = '../../../zhaijunzhi/code_dev/lrcn_img_cls/data/kailu_data_multi_scales/newsplit/seq10Train_10frameOffset.tfrecord'
    # trainTfRec = '/mnt/media2/data/deepfake/02_tfrecord/224x224facenet_seq10Train.tfrecord'
    # trainIter  = myIter(batch_size, trainTfRec)
#----------------------------------------------------------------------------------------------------  
    trainDataJson = '/mnt/mnt/users/zhaijunzhi/code_dev/video-classification-master/ResNetCRNN/kailu_data/split0/newdata_facenet5seq/dstTrainMetaData.json'
    valDataJson = '/mnt/mnt/users/zhaijunzhi/code_dev/video-classification-master/ResNetCRNN/kailu_data/split0/newdata_facenet5seq/dstValMetaData.json'  

    steps_per_epoch = math.floor( 25876 // batch_size)#samples_num/batch_size，一个epochSS要的步数
    validation_steps = math.floor( 3054 // batch_size)

    # train_loader, valid_loader = get_data_loader(trainDataJson, valDataJson)
    train_loader = get_data_loader(trainDataJson,steps_per_epoch)
    valid_loader = get_data_loader(valDataJson,validation_steps)
    # Get samples per epoch.

    # steps_per_epoch = math.floor( 25876 // batch_size)#samples_num/batch_size，一个epochSS要的步数
    # validation_steps = math.floor( 3054 // batch_size)

    # Get the model.
    rm = ResearchModels(2, model, seq_length, saved_model)  #第一个参数原来是len(Dataset.classed)
    for layer in rm.model.layers:
        layer.trainable =True
    # Fit!

    if load_to_memory:
        # Use standard fit.
        rm.model.fit(
            X,
            y,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger],
            epochs=nb_epoch)
    else:
        # Use fit generator.
        rm.model.fit_generator(
            # generator=trainIter,
            generator=train_loader,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[early_stopper, csv_logger, checkpointer,rp],
            validation_data=valid_loader,
            validation_steps=validation_steps
            # ,shuffle=True
            #workers=4
            )


def main():
    """These are the main training settings. Set each before running
    this file."""
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d
    model = 'conv_3d'
    #saved_model = "/mnt/mnt/users/peibowen/code/3d_conv/data/checkpoints/conv_3d-images.003-0.079.hdf5"  # None or weights file
    saved_model = None
    class_limit = None  # int, can be 1-101 or None
    seq_length = 5
    load_to_memory = False  # pre-load the sequences into memory
    batch_size = 16
    nb_epoch = 30

    # Chose images or features and image shape based on network.
    if model in ['conv_3d', 'c3d', 'lrcn']:
        data_type = 'images'
        image_shape = (224, 224, 3)
    elif model in ['lstm', 'mlp']:
        data_type = 'features'
        image_shape = None
    else:
        raise ValueError("Invalid model. See train.py for options.")

    train(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape,
          load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()
    # trainDataJson = '/mnt/mnt/users/zhaijunzhi/code_dev/video-classification-master/ResNetCRNN/kailu_data/split0/newdata_facenet5seq/dstTrainMetaData.json'
    # valDataJson = '/mnt/mnt/users/zhaijunzhi/code_dev/video-classification-master/ResNetCRNN/kailu_data/split0/newdata_facenet5seq/dstValMetaData.json'  
    # train_loader = get_data_loader(trainDataJson)
    # valid_loader = get_data_loader(valDataJson)
    # a = next(train_loader)
    # print(len(a[0]))
    # print(a[0].shape)
    # print(type(a[0]))
