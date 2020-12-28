import keras as keras
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Dense, Input, Add, Activation, GlobalAveragePooling2D, BatchNormalization
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model, load_model
from keras import optimizers, regularizers
from train_plot import PlotLearning


class ResNet:
    def __init__(self, epochs=600, batch_size=128, load_weights=True,level="32",transfer=False,x_train=None,y_train=None,x_val=None,y_val=None,loss_fn=None,istransfer=False):
        self.name               = 'resnet_'+level+'_baseline_600' 
        self.model_filename     = 'models/'+self.name+'.h5'
        self.stack_n            = 5
        self.num_classes        = 10
        self.img_rows           = x_train.shape[1] 
        self.img_cols           = x_train.shape[2] 
        self.img_channels       = x_train.shape[3] 
        self.batch_size         = batch_size
        self.epochs             = epochs
        self.iterations         = x_train.shape[0] // self.batch_size   # MNIST数据集应该是60000
        self.weight_decay       = 0.0001
        self.log_filepath       = r'logs/single/'
        self._model             = None  # 防止后续出现未定义的情况
        self.param_count        = None
        self.x_train,self.y_train    = x_train,y_train
        self.x_val,self.y_val      = x_val,y_val
        self.loss_fn            = loss_fn  if loss_fn is not None else 'categorical_crossentropy'  
        self.transfer            = istransfer
        
        if load_weights:
            try:
                self._model = load_model(self.model_filename)  # _model表示私有变量，保存了.h5加载过来的参数，load_model库函数详见models.py
                self.param_count = self._model.count_params()                
                print('Successfully loaded', self.name)
            except (ImportError, ValueError, OSError):
                print('Failed to load', self.name)


    def construct(self):
        # build network 
        img_input = Input(shape=(self.img_rows,self.img_cols,self.img_channels))
        output    = self.residual_network(img_input,self.num_classes,self.stack_n)
        self._model = Model(img_input, output)
        

    def scheduler(self, epoch):
        lr = 1e-3
#         if epoch > 100:
#             lr *= 1e-3
#         elif epoch > 70:
#             lr *= 1e-2
#         elif epoch > 40:
#             lr *= 1e-1
        if epoch > 600:
            lr *= 1e-3
        elif epoch > 450:
            lr *= 1e-2
        elif epoch > 300:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr

    def residual_network(self, img_input,classes_num=100,stack_n=5):  # ResNet参数，但不包括weights
         # 一个残差块！！！ 从模型图片来看就是从一个Add到另一个Add之间的部分
        def residual_block(intput, out_channel,increase=False, stage=0, block=0):
            if increase:
                stride = (2,2)
            else:
                stride = (1,1)

            pre_bn   = BatchNormalization()(intput)  # （input）为这一层的输入
            pre_relu = Activation('relu')(pre_bn)

            conv_1 = Conv2D(out_channel,kernel_size=(3,3),strides=stride,padding='same',
                            kernel_initializer="he_normal", # he_normal——he正态分布初始化
                            kernel_regularizer=regularizers.l2(self.weight_decay))(pre_relu)
            
            # 记得改回去 —— 修改残差块
            bn_1   = BatchNormalization()(conv_1)
            relu1  = Activation('relu')(bn_1)
            conv_2 = Conv2D(out_channel,kernel_size=(3,3),strides=(1,1),padding='same',
                            kernel_initializer="he_normal",
                            kernel_regularizer=regularizers.l2(self.weight_decay))(relu1)
            if increase:  # 印证了结构图中有的会在右侧线路中多一个卷积层
                projection = Conv2D(out_channel,
                                    kernel_size=(1,1),
                                    strides=(2,2),
                                    padding='same',
                                    kernel_initializer="he_normal",
                                    kernel_regularizer=regularizers.l2(self.weight_decay))(intput)
                block = Add(name='block_'+str(stage)+'_'+str(block))([conv_2, projection])
            else:
                block = Add(name='block_'+str(stage)+'_'+str(block))([intput,conv_2])       # 残差的概念：块中的最后一个输出和块中的第一个输入要相加
            return block

        # build model
        # total layers = stack_n * 3 * 2 + 2
        # stack_n = 5 by default, total layers = 32
        # input: 32x32x3 output: 32x32x16
        x = Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(self.weight_decay))(img_input)

        # input: 32x32x16 output: 32x32x16
        for i in range(stack_n):
            x = residual_block(x,16,False, stage=1, block=i)   # 16就是channel！与输入维度无关，最终结果直接由输出维度决定

        x = residual_block(x,32,True, stage=2, block=0)
        for i in range(1,stack_n):
            x = residual_block(x,32,False, stage=2, block=i)
        
        # input: 16x16x32 output: 8x8x64
        x = residual_block(x,64,True, stage=3, block=0)
        for i in range(1,stack_n):
            x = residual_block(x,64,False, stage=3, block=i)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)

        # input: 64 output: 100
        x = Dense(classes_num,
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        # 必须这么写
        x = Activation('softmax')(x)
        return x

    def train(self):          
        if not self.transfer:
            self.construct()
        
        if self.x_val is not None and self.y_val is not None:
            validation_data=(self.x_val, self.y_val)
        else:
            validation_data=None

        # set optimizer
        sgd = optimizers.SGD(lr=.01, momentum=0.9, nesterov=True) # momentum-动量法,nesterov-NAG
        # adam = optimizers.Adam()
        self._model.compile(loss=self.loss_fn, optimizer=sgd, metrics=['accuracy'])
        self._model.summary()

        # set callback
        tb_cb = TensorBoard(log_dir=self.log_filepath, histogram_freq=0)
        change_lr = LearningRateScheduler(self.scheduler)
        checkpoint = ModelCheckpoint('models/'+self.name+'_best_adam3'+'.h5', # 这里保存了model
                monitor='val_acc', verbose=0, save_best_only= True, mode='auto')
        plot_callback = PlotLearning()
        cbks = [change_lr, tb_cb, plot_callback]

        # set data augmentation
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(horizontal_flip=False,
                                    width_shift_range=0.125,
                                    height_shift_range=0.125,
                                    fill_mode='constant',cval=0.)

        datagen.fit(self.x_train)

        # start training
        self._model.fit_generator(datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size),
                            steps_per_epoch=self.iterations,
                            epochs=self.epochs,
                            callbacks=cbks,
                            validation_data=validation_data)
        self._model.save('models/'+self.name+'_weight_final_sgd'+'.h5')
        self.param_count = self._model.count_params()
        


    def predict(self, img): # img可以是多张图片
        return self._model.predict(img, batch_size=self.batch_size)
    
    def predict_one(self, img):# 只返回第一张图片的预测结果
        return self.predict(img)[0]

    def accuracy(self):
        return self._model.evaluate(self.x_val, self.y_val, verbose=0)[1]
