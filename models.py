"""
A collection of models we'll use to attempt to classify videos.
"""
# import tensorflow.compat.v1 as tf 
# tf.disable_v2_behavior()
from keras.regularizers import l2 as L2_reg
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, BatchNormalization, Activation
from keras.layers import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop, SGD, Nadam
from keras.layers import TimeDistributed
from keras.layers import (Conv2D, MaxPooling3D, Conv3D, MaxPooling2D)
from keras.applications.vgg19 import VGG19
from keras.applications.resnet import ResNet101
from collections import deque
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.applications.xception import Xception
import tensorflow as tf

from efficientnet.keras import EfficientNetB4  #jzk加
import sys

def spatial_pyramid_pooling(input, levels):
    input_shape = input.get_shape().as_list()
    pyramid = []
    for n in levels:
        stride_1 = np.floor(float(input_shape[1] / n)).astype(np.int32)
        stride_2 = np.floor(float(input_shape[2] / n)).astype(np.int32)
        ksize_1 = stride_1 + (input_shape[1] % n)
        ksize_2 = stride_2 + (input_shape[2] % n)
        pool = tf.nn.max_pool(input,
                              ksize=[1, ksize_1, ksize_2, 1],
                              strides=[1, stride_1, stride_2, 1],
                              padding='VALID')
        pyramid.append(tf.reshape(pool, [input_shape[0], -1]))
    spp_pool = tf.concat(pyramid, axis=1)
    return spp_pool

#base_model = VGG19(weights='imagenet')
#model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)


class ResearchModels():
    def __init__(self, nb_classes, model, seq_length,
                 saved_model=None, features_length=2048):
        """
        `model` = one of:
            lstm
            lrcn
            mlp
            conv_3d
            c3d
        `nb_classes` = the number of classes to predict
        `seq_length` = the length of our video sequences
        `saved_model` = the path to a saved Keras model to load
        """

        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        # Get the appropriate model.



        optimizer = Adam(lr=1e-5, decay=1e-6)
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        elif model == 'lstm':
            print("Loading LSTM model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm()
        elif model == 'lrcn':
            print("Loading CNN-LSTM model.")
            self.input_shape = (seq_length, 224, 224, 3)
            self.model = self.lrcn()
        elif model == 'mlp':
            print("Loading simple MLP.")
            self.input_shape = (seq_length, features_length)
            self.model = self.mlp()
        elif model == 'conv_3d':
            print("Loading Conv3D")
            self.input_shape = (seq_length, 224, 224, 3)
            self.model = self.conv_3d()
        elif model == 'c3d':
            print("Loading C3D")
            self.input_shape = (seq_length, 224, 224, 3)
            self.model = self.c3d()
        else:
            print("Unknown network.")
            sys.exit()
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4, decay=1e-6), metrics=metrics)
        #return model#, model_parallel
        

        # Now compile the network.
        # optimizer = Adam(lr=1e-5, decay=1e-6)    
        # #self.model = multi_gpu_model(self.model, gpus=2)
        # self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)

        print(self.model.summary())

    def lstm(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()
        model.add(LSTM(2048, return_sequences=False,
                       input_shape=self.input_shape,
                       dropout=0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def lrcn(self):
        """Build a CNN into RNN.
        Starting version from:
            https://github.com/udacity/self-driving-car/blob/master/
                steering-models/community-models/chauffeur/models.py

        Heavily influenced by VGG-16:
            https://arxiv.org/abs/1409.1556

        Also known as an LRCN:
            https://arxiv.org/pdf/1411.4389.pdf
        """
        def add_default_block(model, kernel_filters, init, reg_lambda):

            # conv
            model.add(TimeDistributed(Conv2D(kernel_filters, (3, 3), padding='same',
                                             kernel_initializer=init, kernel_regularizer=L2_reg(l=reg_lambda))))
            model.add(TimeDistributed(BatchNormalization()))
            model.add(TimeDistributed(Activation('relu')))
            # conv
            model.add(TimeDistributed(Conv2D(kernel_filters, (3, 3), padding='same',
                                             kernel_initializer=init, kernel_regularizer=L2_reg(l=reg_lambda))))
            model.add(TimeDistributed(BatchNormalization()))
            model.add(TimeDistributed(Activation('relu')))
            # max pool
            model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

            return model

        #initialiser = 'glorot_uniform'
        reg_lambda  = 0.001

        model = Sequential()

        # first (non-default) block
        model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), padding='same',
                                  kernel_initializer=initialiser, kernel_regularizer=L2_reg(l=reg_lambda)),
                                  input_shape=self.input_shape
                                  ))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        
        model.add(TimeDistributed(Conv2D(32, (3,3), kernel_initializer=initialiser, kernel_regularizer=L2_reg(l=reg_lambda))))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))        
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        # 2nd-5th (default) blocks
        model = add_default_block(model, 64,  init=initialiser, reg_lambda=reg_lambda)
        model = add_default_block(model, 128, init=initialiser, reg_lambda=reg_lambda)
        model = add_default_block(model, 256, init=initialiser, reg_lambda=reg_lambda)
        model = add_default_block(model, 512, init=initialiser, reg_lambda=reg_lambda)
        
        
        #spatial_pyramid_pooling(model, levels=[4, 2, 1])
        
        # LSTM output head
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(256, return_sequences=False, dropout=0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model
        
    def resnet_spp():
        input_shape=self.input_shape
        model = inference(input_shape, FLAGS.num_residual_blocks, reuse=False)
        
    def mlp(self):
        """Build a simple MLP. It uses extracted features as the input
        because of the otherwise too-high dimensionality."""
        # Model.
        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def conv_3d(self):
        """
        Build a 3D convolutional network, based loosely on C3D.
            https://arxiv.org/pdf/1412.0767.pdf
        """
        # Model.
        # base_model = VGG19(weights='imagenet',include_top=False)
        base_model = EfficientNetB4(weights='imagenet',include_top=False)    #jzk
        #pretrained_cnn = base_model
        # pretrained_cnn=Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
        pretrained_cnn=Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

        print(pretrained_cnn.summary())
        height=224
        width=224
        input_shape = (5, height, width, 3) # (seq_len, height, width, channel)
        model = Sequential()
        model.add(TimeDistributed(pretrained_cnn, input_shape=input_shape))

        #model.add(LSTM(256, return_sequences=True, dropout=0.5))
        # model.add(Conv3D(256, (5,2,2), activation='relu', kernel_regularizer=L2_reg(0.001)))
        # model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        # model.add(Conv3D(128, (3,2,2), activation='relu', kernel_regularizer=L2_reg(0.001)))
        # model.add(MaxPooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1)))
        # model.add(Conv3D(128, (3,3,3), activation='relu', kernel_regularizer=L2_reg(0.001)))
        # model.add(Conv3D(128, (3,3,3), activation='relu', kernel_regularizer=L2_reg(0.001)))
        # model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        # model.add(Conv3D(192, (2,2,2), activation='relu', kernel_regularizer=L2_reg(0.001)))
        # model.add(Conv3D(192, (2,2,2), activation='relu', kernel_regularizer=L2_reg(0.001)))
        # model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        # model.add(Dropout(0.5))
        model.add(Flatten())
        # model.add(Dense(32, kernel_regularizer=L2_reg(0.001)))
        # model.add(Dropout(0.5))
        # model.add(Dense(32, kernel_regularizer=L2_reg(0.001))) 
        model.add(Dense(32, kernel_regularizer=L2_reg(0.001))) 
        model.add(Dropout(0.7))
        model.add(Dense(1, activation='sigmoid'))
        print(model.inputs)
        print(model.outputs)
        for layer in model.layers:
            print(layer.name)
            print('input: ',layer.input_shape)
            print('output: ',layer.output_shape)

        #for layer in model.layers:
            #print(layer.name)
            #print('input: ',layer.input_shape)
            #print('output: ',layer.output_shape)
            #print(model.inputs)
            #print(model.outputs)
            #print(model.get_config())
            #print(model.to_json())

        
        return model

    def c3d(self):
        """
        Build a 3D convolutional network, aka C3D.
            https://arxiv.org/pdf/1412.0767.pdf

        With thanks:
            https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2
        """
        model = Sequential()
        # 1st layer group
        model.add(Conv3D(64, (1,9,10), activation="relu",
                         padding='valid', name='space-conv1',
                         input_shape=self.input_shape, kernel_initializer='RandomNormal'))
        #model.add(PReLU())
        model.add(MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2),
                               padding='valid', name='pool1'))
        # 2nd layer group
        model.add(Conv3D(128, (1,7,8), activation="relu",
                         padding='valid', name='space-conv2', kernel_initializer='RandomNormal'))
        #model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                               padding='valid', name='pool2'))
        #3rd layer group
        model.add(Conv3D(128, (1,4,4), activation="relu",
                         padding='valid', name='space-conv3a', kernel_initializer='RandomNormal'))
        #model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
        #model.add(Conv3D(128, (1,1,1), activation='relu',
        #                 padding='valid', name='conv3b', kernel_initializer='RandomNormal'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                               padding='valid', name='pool3'))
        # # 4th layer group
        model.add(Conv3D(32, (4,1,1), activation="relu",
            padding='valid', name='time-conv4a'))
        #model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))

        # model.add(Conv3D(128, 3, activation='relu',
        #                  padding='valid', name='conv4b'))
        model.add(MaxPooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1), padding='valid', name='pool4'))

        # 5th layer group
        model.add(Conv3D(64, (2,1,1), activation="relu",
                         padding='valid', name='time-conv5a', kernel_initializer='RandomNormal'))
        #model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
        #model.add(Conv3D(128, (1,1,1), activation='relu',
        #                 padding='valid', name='conv5b', kernel_initializer='RandomNormal'))
        #model.add(ZeroPadding3D(padding=(0, 1, 1)))
        #model.add(MaxPooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1),
        #                       padding='valid', name='pool5'))
        model.add(Flatten())

        # FC layers group
        model.add(Dense(512,activation="relu", name='fc6', kernel_initializer='RandomNormal'))
        
        model.add(Dropout(0.5))
        model.add(Dense(512,activation="relu", name='fc7', kernel_initializer='RandomNormal'))
        #model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='sigmoid'))
        
        # grads = gradients(model.output,[model.input])
        # print('input_shape:',model.output_shape,'output_shape:',model.input_shape)
        # print('grads:',grads)
        # print(len(grads))
        # print('grads[0]_shape:',grads[0].shape)
        
        return model
